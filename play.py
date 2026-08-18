"""Interactive build-your-own-network mode.

Run: python play.py

Build phase (default):
  Left-click a station    : start a NEW line, or EXTEND the active line
  Right-click             : deselect active line / cancel
  Click an existing line  : select it (its endpoints become extend targets)
  T + left-click on line  : add a train to that line
  N                       : force NEW_LINE mode on next click
  SPACE / ENTER           : start the simulation
  ESC                     : quit

Play phase:
  SPACE                   : pause / resume
  ESC                     : quit
"""

import pygame
import yaml
import math
import random

from simulator import MiniMetroSim
from agent import apply_action, ActionType, MAX_TRAINS, MAX_TRAINS_PER_LINE
from visualizer import (
    LINE_COLORS, STATION_RADIUS, BG_COLOR, HUD_COLOR, FONT_SIZE,
    STATION_OUTLINE, _draw_station, _draw_shape, _offset_line_segment,
    _train_screen_pos, PASSENGER_R, TRAIN_W, TRAIN_H,
)


with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)
WIDTH, HEIGHT = _cfg["resolution"]
NUM_STATIONS = 20
MAX_LINES = _cfg["lines"]["max"]


def _station_at(sim: MiniMetroSim, pos: tuple[int, int], radius: int = STATION_RADIUS + 4) -> int | None:
    px, py = pos
    for i, s in enumerate(sim.stations):
        if (s.x - px) ** 2 + (s.y - py) ** 2 <= radius * radius:
            return i
    return None


def _line_at(sim: MiniMetroSim, pos: tuple[int, int], tol: int = 8) -> int | None:
    """Return the index of a line whose segment passes within `tol` of pos."""
    px, py = pos
    for li, line in enumerate(sim.lines):
        for k in range(len(line.stations) - 1):
            s1, s2 = line.stations[k], line.stations[k + 1]
            x1, y1, x2, y2 = s1.x, s1.y, s2.x, s2.y
            dx, dy = x2 - x1, y2 - y1
            length_sq = dx * dx + dy * dy
            if length_sq == 0:
                continue
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
            ex, ey = x1 + t * dx, y1 + t * dy
            if (ex - px) ** 2 + (ey - py) ** 2 <= tol * tol:
                return li
    return None


def _draw_lines(surface, sim: MiniMetroSim, highlight_line: int | None = None):
    for i, line in enumerate(sim.lines):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        stations = line.stations
        thickness = 6 if i == highlight_line else 4
        for seg_idx in range(len(stations) - 1):
            s1, s2 = stations[seg_idx], stations[seg_idx + 1]
            if id(s1) < id(s2):
                p1, p2 = (s1.x, s1.y), (s2.x, s2.y)
            else:
                p1, p2 = (s2.x, s2.y), (s1.x, s1.y)
            edge_set = {(s1, s2), (s2, s1)}
            lines_on_edge = [
                l for l in sim.lines
                if any(
                    (l.stations[j], l.stations[j + 1]) in edge_set
                    for j in range(len(l.stations) - 1)
                )
            ]
            shared_count = len(lines_on_edge) - 1
            offset_idx = lines_on_edge.index(line)
            offset_amount = (offset_idx - shared_count / 2.0) * 6
            (px1, py1), (px2, py2) = _offset_line_segment(p1, p2, offset_amount)
            pygame.draw.line(surface, color, (int(px1), int(py1)), (int(px2), int(py2)), thickness)


def _draw_stations(surface, sim, active_line: int | None = None, pending_first: int | None = None,
                   hovered: int | None = None):
    on_active = set()
    if active_line is not None and active_line < len(sim.lines):
        on_active = {id(s) for s in sim.lines[active_line].stations}
        active_color = LINE_COLORS[active_line % len(LINE_COLORS)]
    for i, station in enumerate(sim.stations):
        if i == pending_first:
            outline = (40, 180, 60)
        elif id(station) in on_active:
            outline = active_color  # type: ignore[name-defined]
        else:
            outline = STATION_OUTLINE
        _draw_station(surface, station, outline)
        if i == hovered:
            pygame.draw.circle(surface, (40, 180, 60), (station.x, station.y),
                               STATION_RADIUS + 6, 2)


def _nearest_endpoint(line, target_x: int, target_y: int) -> int:
    """Return 0 (start) or -1 (end) — whichever endpoint of `line` is closer to (target_x, target_y)."""
    start_s, end_s = line.stations[0], line.stations[-1]
    d_start = (start_s.x - target_x) ** 2 + (start_s.y - target_y) ** 2
    d_end = (end_s.x - target_x) ** 2 + (end_s.y - target_y) ** 2
    return 0 if d_start <= d_end else -1


def _draw_endpoint_markers(surface, sim, active_line: int | None):
    """Draw colored pips at the two endpoints of the selected line so it's obvious where it'll extend."""
    if active_line is None or active_line >= len(sim.lines):
        return
    line = sim.lines[active_line]
    if len(line.stations) == 0:
        return
    color = LINE_COLORS[active_line % len(LINE_COLORS)]
    for endpoint in (line.stations[0], line.stations[-1]):
        pygame.draw.circle(surface, color, (endpoint.x, endpoint.y), STATION_RADIUS + 8, 3)


def _draw_ghost_line(surface, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]):
    """Dashed-look ghost line for previewing extends / new-line second pick."""
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return
    steps = max(1, int(length / 14))
    for k in range(steps):
        if k % 2 == 1:
            continue
        t1 = k / steps
        t2 = min(1.0, (k + 1) / steps)
        sx, sy = x1 + dx * t1, y1 + dy * t1
        ex, ey = x1 + dx * t2, y1 + dy * t2
        pygame.draw.line(surface, color, (int(sx), int(sy)), (int(ex), int(ey)), 3)


def _draw_hud(surface, font, lines: list[str], color=HUD_COLOR):
    for i, text in enumerate(lines):
        surf = font.render(text, True, color)
        surface.blit(surf, (10, 10 + i * (FONT_SIZE + 4)))


def _can_extend(line, new_idx: int, sim: MiniMetroSim) -> bool:
    """Same rule as the agent: can extend if new station not already on this line."""
    new_s = sim.stations[new_idx]
    return new_s not in line.stations


def build_phase(sim: MiniMetroSim, surface, font, clock) -> bool:
    """Returns True if user wants to start sim, False if quit."""
    active_line: int | None = None
    pending_first: int | None = None  # first station picked for a NEW_LINE, awaiting second
    add_train_mode = False
    force_new_line = False
    message = ""

    def _reset_pending():
        nonlocal pending_first, active_line, add_train_mode, force_new_line
        pending_first = None
        add_train_mode = False
        force_new_line = False

    while True:
        mouse_pos = pygame.mouse.get_pos()
        hovered_st = _station_at(sim, mouse_pos)
        hovered_ln = _line_at(sim, mouse_pos) if hovered_st is None else None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if sim.lines:
                        return True
                    message = "Build at least one line first."
                elif event.key == pygame.K_t:
                    add_train_mode = not add_train_mode
                    pending_first = None
                    message = "ADD TRAIN: click a line." if add_train_mode else ""
                elif event.key == pygame.K_n:
                    force_new_line = True
                    active_line = None
                    pending_first = None
                    message = "NEW LINE on next click."
                elif event.key in (pygame.K_d, pygame.K_BACKSPACE, pygame.K_DELETE):
                    if active_line is not None and active_line < len(sim.lines):
                        idx = active_line
                        # Drop the line and any trains on it. Re-index remaining lines.
                        sim.lines.pop(idx)
                        for i, l in enumerate(sim.lines):
                            l.index = i
                        active_line = None
                        message = f"Deleted line {idx}."
                    else:
                        message = "Select a line first (click it)."

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # right-click: cancel/deselect
                    if pending_first is not None or add_train_mode or force_new_line:
                        _reset_pending()
                        message = ""
                    else:
                        active_line = None
                        message = ""
                    continue
                if event.button != 1:
                    continue
                pos = event.pos
                st_idx = _station_at(sim, pos)
                ln_idx = _line_at(sim, pos) if st_idx is None else None

                if add_train_mode:
                    if ln_idx is not None:
                        total_trains = sum(len(l.trains) for l in sim.lines)
                        if total_trains >= MAX_TRAINS:
                            message = f"Train pool full ({MAX_TRAINS})."
                        elif len(sim.lines[ln_idx].trains) >= MAX_TRAINS_PER_LINE:
                            message = f"Line {ln_idx} at max trains."
                        else:
                            apply_action(sim, (ActionType.ADD_TRAIN, ln_idx))
                            message = f"Added train to line {ln_idx}."
                    add_train_mode = False
                    continue

                if st_idx is not None:
                    if pending_first is not None or active_line is None or force_new_line:
                        # build a new line...
                        if pending_first is None:
                            if len(sim.lines) >= MAX_LINES:
                                message = f"Max lines reached ({MAX_LINES})."
                                force_new_line = False
                                continue
                            pending_first = st_idx
                            message = "Pick second station for new line."
                        else:
                            a, b = pending_first, st_idx
                            pending_first = None
                            force_new_line = False
                            if a == b:
                                message = "Same station — cancelled."
                            else:
                                apply_action(sim, (ActionType.NEW_LINE, a, b))
                                active_line = len(sim.lines) - 1
                                message = f"Created line {active_line}."
                    else:
                        line = sim.lines[active_line]
                        if not _can_extend(line, st_idx, sim):
                            message = "Station already on this line."
                        else:
                            target = sim.stations[st_idx]
                            endpoint_idx = _nearest_endpoint(line, target.x, target.y)
                            apply_action(sim, (ActionType.EXTEND_LINE, active_line, endpoint_idx, st_idx))
                            message = f"Extended line {active_line}."
                elif ln_idx is not None:
                    active_line = ln_idx
                    pending_first = None
                    message = f"Selected line {ln_idx}."
                else:
                    message = ""

        # ---- Draw ----
        surface.fill(BG_COLOR)
        _draw_lines(surface, sim, highlight_line=active_line)

        # Ghost preview: pending new-line second leg
        if pending_first is not None:
            first_s = sim.stations[pending_first]
            target_x, target_y = (sim.stations[hovered_st].x, sim.stations[hovered_st].y) \
                if hovered_st is not None else mouse_pos
            line_color_idx = len(sim.lines) % len(LINE_COLORS)
            _draw_ghost_line(surface, first_s.x, first_s.y, target_x, target_y,
                             LINE_COLORS[line_color_idx])

        # Ghost preview: extension from the endpoint we'd actually extend from
        elif (active_line is not None and not add_train_mode and not force_new_line
              and active_line < len(sim.lines) and len(sim.lines[active_line].stations) > 0):
            line = sim.lines[active_line]
            if hovered_st is not None and _can_extend(line, hovered_st, sim):
                target = sim.stations[hovered_st]
                ep = _nearest_endpoint(line, target.x, target.y)
                endpoint_s = line.stations[ep]
                _draw_ghost_line(surface, endpoint_s.x, endpoint_s.y, target.x, target.y,
                                 LINE_COLORS[active_line % len(LINE_COLORS)])
            elif hovered_st is None:
                ep = _nearest_endpoint(line, mouse_pos[0], mouse_pos[1])
                endpoint_s = line.stations[ep]
                _draw_ghost_line(surface, endpoint_s.x, endpoint_s.y, mouse_pos[0], mouse_pos[1],
                                 LINE_COLORS[active_line % len(LINE_COLORS)])

        _draw_stations(surface, sim, active_line=active_line,
                       pending_first=pending_first, hovered=hovered_st)
        _draw_endpoint_markers(surface, sim, active_line)

        # Trains
        for li, line in enumerate(sim.lines):
            color = LINE_COLORS[li % len(LINE_COLORS)]
            for train in line.trains:
                tx, ty = _train_screen_pos(train)
                rect = pygame.Rect(int(tx) - TRAIN_W // 2, int(ty) - TRAIN_H // 2, TRAIN_W, TRAIN_H)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (0, 0, 0), rect, 2)

        # Hovered line preview (helps user see what L-click would select)
        if hovered_ln is not None and hovered_ln != active_line:
            line = sim.lines[hovered_ln]
            color = LINE_COLORS[hovered_ln % len(LINE_COLORS)]
            for endpoint in (line.stations[0], line.stations[-1]):
                pygame.draw.circle(surface, color, (endpoint.x, endpoint.y),
                                   STATION_RADIUS + 4, 1)

        mode = "ADD TRAIN" if add_train_mode else ("NEW LINE next" if force_new_line else "BUILD")
        hud = [
            "BUILD MODE",
            f"mode:    {mode}",
            f"active:  {'line ' + str(active_line) if active_line is not None else 'none'}",
            f"lines:   {len(sim.lines)} / {MAX_LINES}",
            f"trains:  {sum(len(l.trains) for l in sim.lines)} / {MAX_TRAINS}",
            "",
            "L-click station: new/extend line",
            "L-click line: select it",
            "T then L-click line: add train",
            "N: force new line",
            "D / Backspace: delete selected line",
            "R-click: cancel selection / pending",
            "SPACE/ENTER: start sim   ESC: quit",
            "",
            message,
        ]
        _draw_hud(surface, font, hud)
        pygame.display.flip()
        clock.tick(60)


def play_phase(sim: MiniMetroSim, surface, font, clock) -> None:
    paused = False
    sim_speed = 1
    while not sim.failed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_UP:
                    sim_speed = min(sim_speed + 1, 10)
                if event.key == pygame.K_DOWN:
                    sim_speed = max(sim_speed - 1, 1)

        if not paused:
            for _ in range(sim_speed):
                sim.step()
                if sim.failed:
                    break

        surface.fill(BG_COLOR)
        _draw_lines(surface, sim)
        for station in sim.stations:
            _draw_station(surface, station, STATION_OUTLINE)
            waiting_list = list(station.waiting)
            for idx, passenger in enumerate(waiting_list[:12]):
                angle = 2 * math.pi * idx / max(len(waiting_list), 1)
                px = station.x + (STATION_RADIUS + 10) * math.cos(angle)
                py = station.y + (STATION_RADIUS + 10) * math.sin(angle)
                _draw_shape(surface, px, py, passenger.target_type,
                            fill_color=(200, 200, 200), outline_color=(100, 100, 100),
                            r=PASSENGER_R, outline_width=1)

        for li, line in enumerate(sim.lines):
            color = LINE_COLORS[li % len(LINE_COLORS)]
            for train in line.trains:
                tx, ty = _train_screen_pos(train)
                rect = pygame.Rect(int(tx) - TRAIN_W // 2, int(ty) - TRAIN_H // 2, TRAIN_W, TRAIN_H)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (0, 0, 0), rect, 2)
                for idx, passenger in enumerate(train.passengers):
                    grid_x = idx % 3
                    grid_y = idx // 3
                    pxi = tx - TRAIN_W // 2 + 8 + grid_x * 12
                    pyi = ty - TRAIN_H // 2 + 8 + grid_y * 10
                    _draw_shape(surface, pxi, pyi, passenger.target_type,
                                fill_color=(220, 180, 100), outline_color=(120, 80, 40),
                                r=4, outline_width=1)

        hud = [
            "PLAY MODE" + ("  (paused)" if paused else ""),
            f"tick:      {sim.time}",
            f"delivered: {sim.delivered}",
            f"speed:     {sim_speed}x",
            "",
            f"lines:     {len(sim.lines)}",
            f"stations:  {[len(l.stations) for l in sim.lines]}",
            f"trains:    {[len(l.trains) for l in sim.lines]}",
            "",
            "SPACE: pause   UP/DOWN: speed   ESC: quit",
        ]
        _draw_hud(surface, font, hud)
        pygame.display.flip()
        clock.tick(60)

    # Game over screen
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                return
        surface.fill(BG_COLOR)
        _draw_lines(surface, sim)
        for station in sim.stations:
            _draw_station(surface, station, STATION_OUTLINE)
        hud = [
            "GAME OVER — station overcrowded",
            f"survived {sim.time} ticks",
            f"delivered {sim.delivered} passengers",
            "",
            "any key to exit",
        ]
        _draw_hud(surface, font, hud, color=(180, 40, 40))
        pygame.display.flip()
        clock.tick(60)


def main():
    pygame.init()
    surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mini Metro — Build Your Own")
    font = pygame.font.SysFont("monospace", FONT_SIZE)
    clock = pygame.time.Clock()

    map_seed = 1

    rng_state = random.getstate()
    random.seed(map_seed)
    sim = MiniMetroSim(NUM_STATIONS)
    random.setstate(rng_state)
    if build_phase(sim, surface, font, clock):
        play_phase(sim, surface, font, clock)

    pygame.quit()


if __name__ == "__main__":
    main()
