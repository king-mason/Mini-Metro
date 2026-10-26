"""Pygame viewer for designs and play-outs, with playback controls.

Two presets share every control and differ only in defaults:
  showcase  design steps advance on their own, games run at 2x, and the next map starts
            a few seconds after game over.
  dev       design steps wait for a key, every candidate is drawn with its scores,
            stations are tinted by load factor, and game over waits for a key.

Keys (H shows them in the window):
  SPACE         pause / resume; during design, auto-advance on / off
  RIGHT, LEFT   faster / slower; during design, RIGHT is the next step
  .             one tick while paused
  ENTER         skip ahead: rest of the design, rest of the search, or rest of the game
  R             replay this design with new passengers
  N             next map
  D             dev overlay on / off
  H             key help
  S             screenshot (to paths.screenshots in config.yaml)
  ESC, Q        quit

Drawing functions (draw_network and friends) are module-level so play.py can reuse them.
"""

import math
import os
import time

import pygame

from heuristic import station_load_factors
from objects import StationType
from rules import ActionType, action_label, action_segments
from simulator import (MiniMetroSim, config, current_spawn_chance, current_spawn_interval,
                       max_overcrowd_ticks, max_waiting)

LINE_COLORS = [
    (220,  50,  50),  # red
    ( 50, 130, 220),  # blue
    ( 50, 180,  80),  # green
    (230, 160,  30),  # orange
    (160,  60, 200),  # purple
    ( 40, 190, 200),  # teal
    (220, 100, 160),  # pink
]
BG_COLOR = (245, 245, 240)
HUD_COLOR = (50, 50, 50)
MUTED = (140, 140, 140)
ALERT = (200, 40, 40)
STATION_OUTLINE = (30, 30, 30)
GHOST = (150, 150, 150)

STATION_RADIUS = 14
PASSENGER_R = 5
LINE_WIDTH = 5
LINE_GAP = 7             # spacing between parallel lines sharing a segment
TRAIN_LEN, TRAIN_WID = 30, 16
WAITING_SLOTS = 12       # passenger slots drawn around a station; matches stations.capacity
FONT_SIZE = 16
SMALL_FONT_SIZE = 13

SPEEDS = (1, 2, 4, 8, 16, 32, 64)   # ticks per frame at 60 fps
FPS = 60

KEYS = {
    pygame.K_SPACE: "pause",
    pygame.K_RIGHT: "faster",
    pygame.K_LEFT: "slower",
    pygame.K_PERIOD: "tick",
    pygame.K_RETURN: "skip",
    pygame.K_KP_ENTER: "skip",
    pygame.K_r: "replay",
    pygame.K_n: "next",
}
HELP = [
    "SPACE        pause / resume (design: auto-advance)",
    "RIGHT LEFT   faster / slower (design: RIGHT = next step)",
    ".            one tick while paused",
    "ENTER        skip ahead (design, search, or game)",
    "R            replay design with new passengers",
    "N            next map",
    "D            dev overlay",
    "S            screenshot",
    "H            close this help",
    "ESC Q        quit",
]


class NextMap(Exception):
    """Raised out of a design or search when the viewer asks for the next map."""


# ---- drawing ----

def draw_shape(surface, x, y, stype, fill=(255, 255, 255), outline=(0, 0, 0), r=STATION_RADIUS, width=2):
    """A station shape (or a passenger's destination shape) centered at (x, y)."""
    x, y = int(x), int(y)

    def poly(points):
        pygame.draw.polygon(surface, fill, points)
        pygame.draw.polygon(surface, outline, points, width)

    if stype == StationType.SQUARE:
        rect = pygame.Rect(x - r, y - r, r * 2, r * 2)
        pygame.draw.rect(surface, fill, rect)
        pygame.draw.rect(surface, outline, rect, width)
    elif stype == StationType.TRIANGLE:
        poly([(x, y - r), (x - int(r * 0.866), y + r // 2), (x + int(r * 0.866), y + r // 2)])
    elif stype == StationType.DIAMOND:
        poly([(x, y - r), (x + r, y), (x, y + r), (x - r, y)])
    elif stype == StationType.RHOMBUS:
        s = r // 2
        poly([(x - r + s, y - r + 2), (x + r, y - r + 2), (x + r - s, y + r - 2), (x - r, y + r - 2)])
    elif stype == StationType.CROSS:
        t = r // 3
        poly([(x - t, y - r), (x + t, y - r), (x + t, y - t), (x + r, y - t), (x + r, y + t), (x + t, y + t),
              (x + t, y + r), (x - t, y + r), (x - t, y + t), (x - r, y + t), (x - r, y - t), (x - t, y - t)])
    elif stype == StationType.PENTAGON:
        poly([(x + int(r * math.sin(math.radians(i * 72))), y - int(r * math.cos(math.radians(i * 72))))
              for i in range(5)])
    elif stype == StationType.STAR:
        poly([(x + int((r if i % 2 == 0 else r // 2) * math.cos(math.radians(i * 36 - 90))),
               y + int((r if i % 2 == 0 else r // 2) * math.sin(math.radians(i * 36 - 90))))
              for i in range(10)])
    elif stype == StationType.DROPLET:
        poly([(x + int(r * 0.8 * math.cos(a)), y + r // 4 + int(r * 0.8 * math.sin(a)))
              for a in [math.radians(d) for d in range(-30, 211, 20)]] + [(x, y - r)])
    elif stype == StationType.LEMON:
        rect = pygame.Rect(x - r, y - int(r * 0.7), r * 2, int(r * 1.4))
        pygame.draw.ellipse(surface, fill, rect)
        pygame.draw.ellipse(surface, outline, rect, width)
    else:  # CIRCLE
        pygame.draw.circle(surface, fill, (x, y), r)
        pygame.draw.circle(surface, outline, (x, y), r, width)


def draw_dashed(surface, p1, p2, color, width=3, dash=10):
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    steps = max(1, int(length / dash))
    for k in range(0, steps, 2):
        t1, t2 = k / steps, min(1.0, (k + 1) / steps)
        pygame.draw.line(surface, color, (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1),
                         (x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2), width)


def line_color(line_idx: int):
    return LINE_COLORS[line_idx % len(LINE_COLORS)]


def segment_offsets(sim: MiniMetroSim) -> dict[tuple[int, int], tuple[float, float]]:
    """Sideways shift of each (line, segment) so lines sharing a segment run side by side."""
    index = {id(s): i for i, s in enumerate(sim.stations)}
    users: dict[tuple[int, int], list[int]] = {}
    for li, line in enumerate(sim.lines):
        for k in range(len(line.stations) - 1):
            a, b = index[id(line.stations[k])], index[id(line.stations[k + 1])]
            users.setdefault((min(a, b), max(a, b)), []).append(li)
    offsets = {}
    for li, line in enumerate(sim.lines):
        for k in range(len(line.stations) - 1):
            a, b = index[id(line.stations[k])], index[id(line.stations[k + 1])]
            lo, hi = sim.stations[min(a, b)], sim.stations[max(a, b)]
            sharing = users[(min(a, b), max(a, b))]
            amount = (sharing.index(li) - (len(sharing) - 1) / 2) * LINE_GAP
            dx, dy = hi.x - lo.x, hi.y - lo.y
            length = math.hypot(dx, dy) or 1.0
            offsets[(li, k)] = (-dy / length * amount, dx / length * amount)
    return offsets


def draw_lines(surface, sim: MiniMetroSim, offsets, highlight: int | None = None):
    for li, line in enumerate(sim.lines):
        color = line_color(li)
        width = LINE_WIDTH + 3 if li == highlight else LINE_WIDTH
        for k in range(len(line.stations) - 1):
            a, b = line.stations[k], line.stations[k + 1]
            ox, oy = offsets[(li, k)]
            pygame.draw.line(surface, color, (a.x + ox, a.y + oy), (b.x + ox, b.y + oy), width)
        # terminal bars, as in Mini Metro, so it is clear where each line ends
        for end, inner, k in ((0, 1, 0), (-1, -2, len(line.stations) - 2)):
            if len(line.stations) < 2:
                break
            e, i = line.stations[end], line.stations[inner]
            ox, oy = offsets[(li, k)]
            dx, dy = e.x - i.x, e.y - i.y
            length = math.hypot(dx, dy) or 1.0
            ux, uy = dx / length, dy / length
            tip = (e.x + ox + ux * (STATION_RADIUS + 10), e.y + oy + uy * (STATION_RADIUS + 10))
            pygame.draw.line(surface, color, (e.x + ox, e.y + oy), tip, width)
            pygame.draw.line(surface, color, (tip[0] - uy * 8, tip[1] + ux * 8), (tip[0] + uy * 8, tip[1] - ux * 8), width)


def _heat(rho: float):
    """Station fill for the dev overlay: white when lightly loaded, red at load factor >= 1.5."""
    t = max(0.0, min(1.0, (rho - 0.5) / 1.0))
    return (255, int(255 - 155 * t), int(255 - 215 * t))


def draw_stations(surface, sim: MiniMetroSim, font=None, outlines: dict[int, tuple] | None = None,
                  loads: dict[int, float] | None = None, labels: bool = False):
    """Stations with their waiting passengers in WAITING_SLOTS slots around them, filled
    clockwise from the top, so a full ring means a full station. Overcrowded stations turn
    red and, given a `font`, show the overflow. `loads` tints stations by load factor and
    `labels` adds station indices (dev overlay)."""
    outlines = outlines or {}
    for i, station in enumerate(sim.stations):
        n_waiting = len(station.waiting)
        crowded = n_waiting > max_waiting
        outline = ALERT if crowded else outlines.get(i, STATION_OUTLINE)
        fill = _heat(loads[i]) if loads is not None and loads.get(i, 0) != float("inf") else (255, 255, 255)
        draw_shape(surface, station.x, station.y, station.type, fill=fill, outline=outline, width=3)
        for slot, passenger in enumerate(list(station.waiting)[:WAITING_SLOTS]):
            angle = 2 * math.pi * slot / WAITING_SLOTS - math.pi / 2
            px = station.x + (STATION_RADIUS + 11) * math.cos(angle)
            py = station.y + (STATION_RADIUS + 11) * math.sin(angle)
            draw_shape(surface, px, py, passenger.target_type, fill=(200, 200, 200),
                       outline=(100, 100, 100), r=PASSENGER_R, width=1)
        if labels and font is not None:
            text = f"{i}" if loads is None or loads.get(i) in (None, float("inf")) else f"{i}:{loads[i]:.1f}"
            surface.blit(font.render(text, True, MUTED), (station.x + STATION_RADIUS + 4, station.y - STATION_RADIUS - 14))
        if crowded and font is not None:
            surf = font.render(f"+{n_waiting - WAITING_SLOTS}", True, ALERT)
            surface.blit(surf, (station.x - surf.get_width() // 2, station.y + STATION_RADIUS + 18))


def draw_trains(surface, sim: MiniMetroSim, offsets):
    """Each train as a car pointing along its track, with its passengers' destinations."""
    for li, line in enumerate(sim.lines):
        stations = line.stations
        if len(stations) < 2:
            continue
        for train in line.trains:
            i = train.position
            j = min(max(i + train.direction, 0), len(stations) - 1)
            a, b = stations[i], stations[j]
            dx, dy = b.x - a.x, b.y - a.y
            length = math.hypot(dx, dy) or 1.0
            ux, uy = dx / length, dy / length
            vx, vy = -uy, ux
            ox, oy = offsets.get((li, min(i, j)), (0.0, 0.0))
            cx = a.x + dx * train.progress + ox
            cy = a.y + dy * train.progress + oy
            hl, hw = TRAIN_LEN / 2, TRAIN_WID / 2
            corners = [(cx + ux * su * hl + vx * sv * hw, cy + uy * su * hl + vy * sv * hw)
                       for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            pygame.draw.polygon(surface, line_color(li), corners)
            pygame.draw.polygon(surface, (20, 20, 20), corners, 2)
            for slot, passenger in enumerate(train.passengers[:6]):
                col, row = slot % 3 - 1, slot // 3 - 0.5
                draw_shape(surface, cx + ux * col * 8 + vx * row * 7, cy + uy * col * 8 + vy * row * 7,
                           passenger.target_type, fill=(255, 255, 255), outline=(40, 40, 40), r=3, width=1)


def draw_network(surface, sim: MiniMetroSim, font=None, highlight: int | None = None, outlines=None,
                 loads=None, labels: bool = False):
    offsets = segment_offsets(sim)
    draw_lines(surface, sim, offsets, highlight)
    draw_stations(surface, sim, font, outlines, loads, labels)
    draw_trains(surface, sim, offsets)


def draw_panel(surface, font, rows, pos, anchor="topleft", alpha=225, pad=8):
    """A translucent box of text rows at `pos`. A row is a string or (text, color[, swatch
    color]); a swatch draws a small color key before the text."""
    rendered = []
    for row in rows:
        text, color, swatch = (row, HUD_COLOR, None) if isinstance(row, str) else (tuple(row) + (None,))[:3]
        rendered.append((font.render(text, True, color), swatch))
    sw = font.get_height()
    width = max((s.get_width() + (sw + 6 if swatch else 0) for s, swatch in rendered), default=0) + 2 * pad
    height = sum(s.get_height() + 2 for s, _ in rendered) + 2 * pad
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill((255, 255, 255, alpha))
    y = pad
    for surf, swatch in rendered:
        x = pad
        if swatch:
            pygame.draw.rect(panel, swatch, pygame.Rect(x, y + 3, sw, sw - 6))
            x += sw + 6
        panel.blit(surf, (x, y))
        y += surf.get_height() + 2
    rect = panel.get_rect(**{anchor: pos})
    surface.blit(panel, rect)
    return rect


# ---- viewer ----

class Visualizer:
    def __init__(self, width: int, height: int, mode: str = "showcase", caption: str = "Mini Metro Sim"):
        pygame.init()
        self.surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption(caption)
        self.font = pygame.font.SysFont("menlo,dejavusansmono,monospace", FONT_SIZE)
        self.small_font = pygame.font.SysFont("menlo,dejavusansmono,monospace", SMALL_FONT_SIZE)
        self.big_font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 26, bold=True)
        self.clock = pygame.time.Clock()
        self.width, self.height = width, height

        self.dev = mode == "dev"
        self.auto_design = not self.dev       # showcase: design steps advance by themselves
        self.design_delay = 0.35              # seconds per design step when auto-advancing
        self.speed = SPEEDS.index(4 if self.dev else 2)
        self.auto_next = None if self.dev else 6.0   # seconds from game over to the next map
        self.next_label = "next map"                  # what N does, for the game-over hint
        self.show_help = False
        self.paused = False

        self.info: dict = {}      # shown in the title row: map, passengers, policy, ...
        self.stage = ""           # what is happening now
        self.episode = 0
        self.design_step = 0
        self._note, self._note_until = "", 0.0

    # ---- public hooks ----

    def new_map(self, **info):
        """Start a new episode; `info` (e.g. map=..., policy=...) is shown in the title row."""
        self.episode += 1
        self.info = {"episode": self.episode, **info}
        self.design_step = 0
        self.paused = False

    def set_info(self, **info):
        self.info.update(info)

    def show_design_step(self, sim: MiniMetroSim, action: tuple, label: str, scored: list | None = None) -> bool:
        """on_step hook for Agent.design: show the design so far with the next action dashed.
        In dev mode every candidate is dashed too, numbered as in the score table. Returns
        False once the viewer skips the rest of the design."""
        self.design_step += 1
        self.stage = f"Designing - step {self.design_step}: {label}"
        deadline = time.monotonic() + self.design_delay
        while True:
            self._frame_design(sim, action, scored)
            for cmd in self._commands():
                if cmd == "skip":
                    return False
                if cmd == "next":
                    raise NextMap
                if cmd == "pause":
                    self.auto_design = not self.auto_design
                    deadline = time.monotonic() + self.design_delay
                if cmd in ("faster", "tick"):
                    return True
            if self.auto_design and time.monotonic() >= deadline:
                return True
            self.clock.tick(FPS)

    def show_search(self, round_idx: int, rounds: int, sim: MiniMetroSim, stats: dict | None) -> bool:
        """on_round hook for search.improve: show the best design so far. Returns False to
        stop the search (ENTER)."""
        if stats is None:
            self.stage = f"Polishing the design by play-out search ({rounds} rounds)..."
        else:
            kept = ", ".join(f"{k} x{v}" for k, v in stats["accepted"].items()) or "nothing yet"
            self.stage = (f"Search round {round_idx}/{rounds}: {stats['start']:.0f} -> ~{stats['final']:.0f}"
                          f" ticks (kept {kept})")
        for cmd in self._commands():
            if cmd == "skip":
                return False
            if cmd == "next":
                raise NextMap
        self._frame(sim)
        return True

    def play(self, sim: MiniMetroSim, max_ticks: int = 50_000) -> str:
        """Run the game on screen until it ends, then hold on the result.
        Returns "replay" or "next"."""
        self.paused = False
        while sim.time < max_ticks and not sim.failed:
            ticks = 0 if self.paused else SPEEDS[self.speed]
            for cmd in self._commands():
                if cmd == "pause":
                    self.paused = not self.paused
                elif cmd == "faster":
                    self.speed = min(self.speed + 1, len(SPEEDS) - 1)
                elif cmd == "slower":
                    self.speed = max(self.speed - 1, 0)
                elif cmd == "tick" and self.paused:
                    ticks = 1
                elif cmd == "skip":
                    self._fast_forward(sim, max_ticks)
                elif cmd in ("replay", "next"):
                    return cmd
            for _ in range(ticks):
                sim.step()
                if sim.failed or sim.time >= max_ticks:
                    break
            self.stage = "Playing" + ("  [paused - SPACE resumes, . steps]" if self.paused else "")
            self._frame(sim)
            self.clock.tick(FPS)
        return self._game_over(sim, max_ticks)

    def close(self):
        pygame.quit()

    # ---- loops ----

    def _fast_forward(self, sim, max_ticks):
        while sim.time < max_ticks and not sim.failed:
            for _ in range(500):
                sim.step()
                if sim.failed or sim.time >= max_ticks:
                    break
            self.stage = f"Skipping to the end... tick {sim.time}"
            self._commands()  # keep the window responsive; ESC still quits
            self._frame(sim)

    def _game_over(self, sim, max_ticks) -> str:
        start = time.monotonic()
        while True:
            for cmd in self._commands():
                if cmd in ("replay", "next"):
                    return cmd
                if cmd in ("skip", "pause"):
                    return "next"
            waited = time.monotonic() - start
            if self.auto_next is not None and waited >= self.auto_next:
                return "next"
            if sim.failed:
                crowded = [f"{s.type.name.lower()}{f' #{i}' if self.dev else ''}"
                           for i, s in enumerate(sim.stations) if len(s.waiting) > max_waiting]
                headline = f"Game over at tick {sim.time}: {' and '.join(crowded)} station overflowed"
            else:
                headline = f"Reached the {max_ticks}-tick limit"
            after = (f"  (next map in {self.auto_next - waited:.0f}s)" if self.auto_next is not None else "")
            self.stage = "Game over"
            self._frame(sim, banner=[headline, f"{sim.delivered} passengers delivered",
                                     f"R replay with new passengers   N {self.next_label}" + after])
            self.clock.tick(FPS)

    # ---- input ----

    def _commands(self) -> list[str]:
        """Handle global keys here; return the rest as command names (see KEYS)."""
        commands = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                raise SystemExit
            if event.key == pygame.K_d:
                self.dev = not self.dev
            elif event.key in (pygame.K_h, pygame.K_SLASH):
                self.show_help = not self.show_help
            elif event.key == pygame.K_s:
                self.screenshot()
            elif event.key in KEYS:
                commands.append(KEYS[event.key])
        return commands

    def screenshot(self) -> str:
        folder = config.get("paths", {}).get("screenshots", "screenshots/")
        os.makedirs(folder, exist_ok=True)
        name = f"map{self.info.get('map', 'x')}_ep{self.episode}_{time.strftime('%H%M%S')}.png"
        path = os.path.join(folder, name)
        pygame.image.save(self.surface, path)
        self.note(f"saved {path}")
        return path

    def note(self, text: str, seconds: float = 2.5):
        """A short message under the status rows (e.g. after a screenshot)."""
        self._note, self._note_until = text, time.monotonic() + seconds

    # ---- frames ----

    def _frame(self, sim: MiniMetroSim, banner: list[str] | None = None, under=None, over=None):
        """Draw one frame. `under` / `over` are drawn below / above the network."""
        self.surface.fill(BG_COLOR)
        if under is not None:
            under()
        draw_network(self.surface, sim, self.small_font, loads=station_load_factors(sim) if self.dev else None,
                     labels=self.dev)
        if over is not None:
            over()
        self._hud(sim)
        if banner:  # in the empty band below the map, so the failed station stays visible
            draw_panel(self.surface, self.big_font, [(banner[0], ALERT if sim.failed else HUD_COLOR)],
                       (self.width // 2, self.height - 64), anchor="midbottom", alpha=235, pad=12)
            draw_panel(self.surface, self.font, banner[1:], (self.width // 2, self.height - 60),
                       anchor="midtop", alpha=235, pad=8)
        if self.show_help:
            draw_panel(self.surface, self.font, [("Keys", HUD_COLOR)] + HELP,
                       (self.width // 2, self.height // 2), anchor="center", alpha=245, pad=16)
        pygame.display.flip()

    def _frame_design(self, sim: MiniMetroSim, action: tuple, scored: list | None):
        def dashed(cand, color, width):
            for a, b in action_segments(sim, cand):
                p, q = sim.stations[a], sim.stations[b]
                draw_dashed(self.surface, (p.x, p.y), (q.x, q.y), color, width)

        def candidates():
            for rank, (cand, _, _) in enumerate(scored[:8], start=1):
                if cand != action:
                    dashed(cand, GHOST, 2)
                    for a, b in action_segments(sim, cand)[:1]:
                        p, q = sim.stations[a], sim.stations[b]
                        self.surface.blit(self.small_font.render(str(rank), True, MUTED),
                                          ((p.x + q.x) // 2 + 4, (p.y + q.y) // 2 + 4))

        def chosen():
            if action[0] == ActionType.ADD_TRAIN:
                line = sim.lines[action[1]]
                for p, q in zip(line.stations, line.stations[1:]):
                    draw_dashed(self.surface, (p.x, p.y), (q.x, q.y), (255, 255, 255), width=2, dash=6)
            elif action[0] != ActionType.NOOP:
                dashed(action, line_color(len(sim.lines) if action[0] == ActionType.NEW_LINE else action[1]), 5)
            if self.dev and scored:
                self._score_table(sim, scored, action)

        self._frame(sim, under=candidates if self.dev and scored else None, over=chosen)

    def _score_table(self, sim: MiniMetroSim, scored: list, chosen: tuple):
        """Bottom-left table of the best candidates; the chosen one is marked with '>'."""
        rows = [(f"   {'candidate':36s} {'prior':>7s} {'Q':>7s} {'total':>7s}", MUTED)]
        ranked = list(enumerate(scored, start=1))
        for rank, (cand, prior, q) in ranked[:6] + [r for r in ranked[6:] if r[1][0] == chosen]:
            mark = ">" if cand == chosen else " "
            rows.append((f"{mark}{rank:<2d}{action_label(cand, sim):36.36s} {prior:+7.3f} {q:+7.3f} {prior + q:+7.3f}",
                         HUD_COLOR if cand == chosen else MUTED))
        draw_panel(self.surface, self.small_font, rows, (10, self.height - 10), anchor="bottomleft")

    def _hud(self, sim: MiniMetroSim):
        info = self.info
        title = "   ".join(f"{k} {v}" for k, v in info.items() if v is not None)
        rows = [title, (self.stage, HUD_COLOR)]
        if sim.time > 0:
            danger = sim.overcrowded_ticks / max_overcrowd_ticks
            rows.append((f"tick {sim.time:6d}   delivered {sim.delivered:6d}   speed {SPEEDS[self.speed]}x"
                         f"   overcrowding {sim.overcrowded_ticks}/{max_overcrowd_ticks}",
                         ALERT if danger > 0.5 else HUD_COLOR))
        if self.dev:
            waiting = sum(len(s.waiting) for s in sim.stations)
            rows.append((f"spawn every {current_spawn_interval(sim.time):.1f} ticks at p={current_spawn_chance(sim.time):.2f}"
                         f"   waiting {waiting}", MUTED))
        if self._note and time.monotonic() < self._note_until:
            rows.append((self._note, MUTED))
        draw_panel(self.surface, self.font, rows, (10, 10))

        legend = [(f"L{li}  {len(line.stations):2d} stations  {len(line.trains)} train"
                   f"{'s' if len(line.trains) != 1 else ' '}", HUD_COLOR, line_color(li))
                  for li, line in enumerate(sim.lines)]
        if legend:
            draw_panel(self.surface, self.font, legend, (self.width - 10, 10), anchor="topright")
        hint = "H keys   D dev overlay" if not self.dev else "H keys   D dev overlay (on)"
        draw_panel(self.surface, self.small_font, [(hint, MUTED)], (self.width - 10, self.height - 10),
                   anchor="bottomright", alpha=160)
