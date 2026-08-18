import math
import pygame
from objects import StationType
from simulator import MiniMetroSim, current_spawn_interval, current_spawn_chance


LINE_COLORS = [
    (220,  50,  50),  # red
    ( 50, 130, 220),  # blue
    ( 50, 180,  80),  # green
    (230, 160,  30),  # orange
    (160,  60, 200),  # purple
    ( 40, 190, 200),  # teal
    (220, 100, 160),  # pink
]

STATION_RADIUS = 14
TRAIN_W, TRAIN_H = 40, 28
PASSENGER_R = 5
HUD_COLOR = (50, 50, 50)
BG_COLOR = (245, 245, 240)
STATION_OUTLINE = (30, 30, 30)
FONT_SIZE = 18


def _lerp(a, b, t):
    return a + (b - a) * t


def _offset_line_segment(p1: tuple, p2: tuple, offset: float) -> tuple:
    """Offset a line segment perpendicular to its direction."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length < 1:
        return (p1, p2)
    # Perpendicular direction (rotated 90°)
    px, py = -dy / length, dx / length
    return (
        (x1 + px * offset, y1 + py * offset),
        (x2 + px * offset, y2 + py * offset),
    )


def _train_screen_pos(train) -> tuple[float, float]:
    stations = train.line.stations
    pos = train.position
    direction = train.direction
    next_pos = pos + direction
    next_pos = max(0, min(len(stations) - 1, next_pos))
    s_cur = stations[pos]
    s_next = stations[next_pos]
    t = train.progress
    x = _lerp(s_cur.x, s_next.x, t)
    y = _lerp(s_cur.y, s_next.y, t)
    return (x, y)


def _draw_shape(surface, x, y, stype, fill_color=(255, 255, 255), outline_color=(0, 0, 0), r=STATION_RADIUS, outline_width=2):
    """Draw a station shape at (x, y) with given colors and radius."""
    x, y = int(x), int(y)

    if stype == StationType.CIRCLE:
        pygame.draw.circle(surface, fill_color, (x, y), r)
        pygame.draw.circle(surface, outline_color, (x, y), r, outline_width)

    elif stype == StationType.SQUARE:
        rect = pygame.Rect(x - r, y - r, r * 2, r * 2)
        pygame.draw.rect(surface, fill_color, rect)
        pygame.draw.rect(surface, outline_color, rect, outline_width)

    elif stype == StationType.TRIANGLE:
        pts = [
            (x, y - r),
            (x - int(r * 0.866), y + r // 2),
            (x + int(r * 0.866), y + r // 2),
        ]
        pygame.draw.polygon(surface, fill_color, pts)
        pygame.draw.polygon(surface, outline_color, pts, outline_width)

    elif stype == StationType.DIAMOND:
        pts = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
        pygame.draw.polygon(surface, fill_color, pts)
        pygame.draw.polygon(surface, outline_color, pts, outline_width)

    elif stype == StationType.CROSS:
        t = r // 3
        rects = [
            pygame.Rect(x - t, y - r, t * 2, r * 2),
            pygame.Rect(x - r, y - t, r * 2, t * 2),
        ]
        for rect in rects:
            pygame.draw.rect(surface, fill_color, rect)
            pygame.draw.rect(surface, outline_color, rect, outline_width)

    elif stype == StationType.PENTAGON:
        pts = [
            (x + int(r * math.sin(math.radians(i * 72))) ,
             y - int(r * math.cos(math.radians(i * 72))))
            for i in range(5)
        ]
        pygame.draw.polygon(surface, fill_color, pts)
        pygame.draw.polygon(surface, outline_color, pts, outline_width)

    elif stype == StationType.STAR:
        pts = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            rad = r if i % 2 == 0 else r // 2
            pts.append((x + int(rad * math.cos(angle)), y + int(rad * math.sin(angle))))
        pygame.draw.polygon(surface, fill_color, pts)
        pygame.draw.polygon(surface, outline_color, pts, outline_width)

    else:
        pygame.draw.circle(surface, fill_color, (x, y), r)
        pygame.draw.circle(surface, outline_color, (x, y), r, outline_width)


def _draw_station(surface, station, outline_color):
    _draw_shape(surface, station.x, station.y, station.type, fill_color=(255, 255, 255),
                outline_color=outline_color, r=STATION_RADIUS, outline_width=3)


class Visualizer:
    def __init__(self, width: int, height: int, caption: str = "Mini Metro Sim"):
        pygame.init()
        self.surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption(caption)
        self.font = pygame.font.SysFont("monospace", FONT_SIZE)
        self.small_font = pygame.font.SysFont("monospace", 13)
        self.clock = pygame.time.Clock()
        self.episode = 0
        self.last_reward = 0
        self.game_seed = None

    def draw(self, sim: MiniMetroSim, epsilon: float = 0.0):
        self.events()

        self.surface.fill(BG_COLOR)

        # Lines — group by shared edges and offset parallel lines
        for i, line in enumerate(sim.lines):
            color = LINE_COLORS[i % len(LINE_COLORS)]
            stations = line.stations

            for seg_idx in range(len(stations) - 1):
                s1, s2 = stations[seg_idx], stations[seg_idx + 1]

                # Canonical orientation by station id — ensures both directions of
                # the same edge yield the same perpendicular vector.
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

                pygame.draw.line(self.surface, color, (int(px1), int(py1)), (int(px2), int(py2)), 4)

        # Stations
        for station in sim.stations:
            _draw_station(self.surface, station, STATION_OUTLINE)

            # Waiting passengers as small shapes around the station
            waiting_list = list(station.waiting)
            for idx, passenger in enumerate(waiting_list[:12]):  # max 12 visible
                angle = 2 * math.pi * idx / max(len(waiting_list), 1)
                px = station.x + (STATION_RADIUS + 10) * math.cos(angle)
                py = station.y + (STATION_RADIUS + 10) * math.sin(angle)
                _draw_shape(self.surface, px, py, passenger.target_type,
                           fill_color=(200, 200, 200), outline_color=(100, 100, 100),
                           r=PASSENGER_R, outline_width=1)

        # Trains
        for line_idx, line in enumerate(sim.lines):
            color = LINE_COLORS[line_idx % len(LINE_COLORS)]
            for train in line.trains:
                tx, ty = _train_screen_pos(train)
                rect = pygame.Rect(int(tx) - TRAIN_W // 2, int(ty) - TRAIN_H // 2,
                                   TRAIN_W, TRAIN_H)
                pygame.draw.rect(self.surface, color, rect)
                pygame.draw.rect(self.surface, (0, 0, 0), rect, 2)

                # Draw passenger shapes inside the train
                train_pax = train.passengers
                for idx, passenger in enumerate(train_pax):
                    grid_x = idx % 3
                    grid_y = idx // 3
                    px = tx - TRAIN_W // 2 + 8 + grid_x * 12
                    py = ty - TRAIN_H // 2 + 8 + grid_y * 10
                    _draw_shape(self.surface, px, py, passenger.target_type,
                               fill_color=(220, 180, 100), outline_color=(120, 80, 40),
                               r=4, outline_width=1)

        # HUD
        hud_lines = [
            f"ep:        {self.episode}",
            f"tick:      {sim.time}",
            f"delivered: {sim.delivered}",
            f"epsilon:   {epsilon:.3f}",
            "",
            f"lines:     {len(sim.lines)}",
            "",
            f"last reward: {self.last_reward}",
        ]
        y = 10
        for text in hud_lines:
            surf = self.font.render(text, True, HUD_COLOR)
            self.surface.blit(surf, (10, y))
            y += FONT_SIZE + 4

        # Line color legend
        y += 4
        surf = self.font.render("legend:", True, HUD_COLOR)
        self.surface.blit(surf, (10, y))
        y += FONT_SIZE + 4
        for i, line in enumerate(sim.lines):
            color = LINE_COLORS[i % len(LINE_COLORS)]
            pygame.draw.rect(self.surface, color, pygame.Rect(10, y, 18, FONT_SIZE - 2))
            pygame.draw.rect(self.surface, HUD_COLOR, pygame.Rect(10, y, 18, FONT_SIZE - 2), 1)
            label = f"  line {i}  s:{len(line.stations)}  t:{len(line.trains)}"
            surf = self.font.render(label, True, HUD_COLOR)
            self.surface.blit(surf, (28, y))
            y += FONT_SIZE + 4

        # Spawn debug
        y += 4
        spawn_interval = current_spawn_interval(sim.time)
        spawn_chance = current_spawn_chance(sim.time)
        debug_lines = [
            "spawn debug:",
            f"  interval: {spawn_interval}",
            f"  chance:   {spawn_chance:.3f}",
        ]
        for text in debug_lines:
            surf = self.font.render(text, True, HUD_COLOR)
            self.surface.blit(surf, (10, y))
            y += FONT_SIZE + 4

        pygame.display.flip()
        self.clock.tick(60)

    def draw_design_step(self, sim: MiniMetroSim, action_label: str, step_idx: int,
                         total_steps: int | None = None,
                         structural_reward: float | None = None,
                         was_random: bool = False):
        """Draw the post-action state during Phase A and wait for a keypress to advance.

        Press any key (except QUIT) to step forward; ESC to skip the remaining design steps.
        Returns True if user requested skip.
        """
        self.surface.fill(BG_COLOR)

        for i, line in enumerate(sim.lines):
            color = LINE_COLORS[i % len(LINE_COLORS)]
            stations = line.stations
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
                pygame.draw.line(self.surface, color, (int(px1), int(py1)), (int(px2), int(py2)), 4)

        for station in sim.stations:
            _draw_station(self.surface, station, STATION_OUTLINE)

        for line_idx, line in enumerate(sim.lines):
            color = LINE_COLORS[line_idx % len(LINE_COLORS)]
            for train in line.trains:
                tx, ty = _train_screen_pos(train)
                rect = pygame.Rect(int(tx) - TRAIN_W // 2, int(ty) - TRAIN_H // 2, TRAIN_W, TRAIN_H)
                pygame.draw.rect(self.surface, color, rect)
                pygame.draw.rect(self.surface, (0, 0, 0), rect, 2)

        step_label = f"step {step_idx}" + (f" / {total_steps}" if total_steps else "")
        selection = "RANDOM" if was_random else "policy"
        reward_str = (f"{structural_reward:+.2f}" if structural_reward is not None else "n/a")
        hud_lines = [
            "DESIGN PHASE",
            step_label,
            "",
            f"action:   {action_label}",
            f"selected: {selection}",
            f"shaping:  {reward_str}",
            "",
            f"lines:    {len(sim.lines)}",
            "",
            "any key: next   ESC: skip to play-out",
        ]
        y = 10
        for text in hud_lines:
            surf = self.font.render(text, True, HUD_COLOR)
            self.surface.blit(surf, (10, y))
            y += FONT_SIZE + 4

        # Line color legend
        y += 4
        surf = self.font.render("legend:", True, HUD_COLOR)
        self.surface.blit(surf, (10, y))
        y += FONT_SIZE + 4
        for i, line in enumerate(sim.lines):
            color = LINE_COLORS[i % len(LINE_COLORS)]
            pygame.draw.rect(self.surface, color, pygame.Rect(10, y, 18, FONT_SIZE - 2))
            pygame.draw.rect(self.surface, HUD_COLOR, pygame.Rect(10, y, 18, FONT_SIZE - 2), 1)
            label = f"  line {i}  s:{len(line.stations)}  t:{len(line.trains)}"
            surf = self.font.render(label, True, HUD_COLOR)
            self.surface.blit(surf, (28, y))
            y += FONT_SIZE + 4

        # Spawn debug
        y += 4
        spawn_interval = current_spawn_interval(sim.time)
        spawn_chance = current_spawn_chance(sim.time)
        debug_lines = [
            "spawn debug:",
            f"  interval: {spawn_interval}",
            f"  chance:   {spawn_chance:.3f}",
        ]
        for text in debug_lines:
            surf = self.font.render(text, True, HUD_COLOR)
            self.surface.blit(surf, (10, y))
            y += FONT_SIZE + 4

        # Seed
        if self.game_seed is not None:
            surf = self.font.render(f"Map seed: {self.game_seed}", True, HUD_COLOR)
            x = self.surface.get_width() - surf.get_width() - 10
            self.surface.blit(surf, (x, 10))

        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return True
                    return False
            self.clock.tick(60)

    def next_episode(self):
        self.episode += 1

    def update_last_reward(self, reward):
        self.last_reward = reward
    
    def set_game_seed(self, seed):
        self.game_seed = seed

    def hold(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                if event.type == pygame.KEYDOWN:
                    return
    
    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def close(self):
        pygame.quit()
