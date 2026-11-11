"""Interactive build-your-own-network mode.

Run: python play.py [--map SEED] [--level LEVEL]

Build phase:
  Left-click a station    : start a NEW line, or EXTEND the selected line from its nearer end
  Click an existing line  : select it (its endpoints become extend targets)
  Right-click             : cancel the pending pick / deselect
  T + left-click on line  : add a train to that line
  N                       : start a new line on the next click
  D / Backspace           : delete the selected line
  L                       : station load factors (tint) and indices on / off
  SPACE / ENTER           : play the design
  ESC                     : quit

The preview turns red when a move breaks a rule (rules.py), e.g. a segment that would
cross another line. Play phase: the viewer's keys (H shows them); R replays with new
passengers and N comes back here to edit the design.
"""

import argparse
import copy
import random

import pygame

from config import MAX_LINES, MAX_TRAINS, MAX_TRAINS_PER_LINE
from maps import LEVELS
from metrics import station_load_factors
from rules import ActionType, apply_action, crosses_network, is_legal
from simulator import MiniMetroSim
from visualizer import (BG_COLOR, GHOST, LINE_COLORS, MUTED, STATION_RADIUS, Visualizer, draw_dashed,
                        draw_network, draw_panel, line_color)

ILLEGAL = (210, 60, 60)
PICKED = (40, 180, 60)


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
            dx, dy = s2.x - s1.x, s2.y - s1.y
            length_sq = dx * dx + dy * dy
            if length_sq == 0:
                continue
            t = max(0.0, min(1.0, ((px - s1.x) * dx + (py - s1.y) * dy) / length_sq))
            if (s1.x + t * dx - px) ** 2 + (s1.y + t * dy - py) ** 2 <= tol * tol:
                return li
    return None


def _nearest_endpoint(line, x: int, y: int) -> int:
    """0 (start) or -1 (end): whichever endpoint of `line` is closer to (x, y)."""
    start, end = line.stations[0], line.stations[-1]
    return 0 if (start.x - x) ** 2 + (start.y - y) ** 2 <= (end.x - x) ** 2 + (end.y - y) ** 2 else -1


def _why_illegal(sim: MiniMetroSim, action: tuple) -> str:
    """Explain why rules.is_legal rejects `action` ('' if it is legal)."""
    if is_legal(sim, action):
        return ""
    t = action[0]
    if t == ActionType.ADD_TRAIN:
        line = sim.lines[action[1]]
        if sum(len(l.trains) for l in sim.lines) >= MAX_TRAINS:
            return f"All {MAX_TRAINS} trains are in use."
        if len(line.trains) >= MAX_TRAINS_PER_LINE:
            return f"Line {action[1]} already has the maximum of {MAX_TRAINS_PER_LINE} trains."
        return f"Line {action[1]} needs another segment first (at most one train per segment)."
    if t == ActionType.NEW_LINE:
        if len(sim.lines) >= MAX_LINES:
            return f"All {MAX_LINES} lines are in use."
        if action[1] == action[2]:
            return "Pick two different stations."
        if sum(len(l.trains) for l in sim.lines) >= MAX_TRAINS:
            return f"A new line needs a train, and all {MAX_TRAINS} are in use."
    if t == ActionType.EXTEND_LINE and sim.stations[action[3]] in sim.lines[action[1]].stations:
        return "That station is already on this line."
    return "Lines may not cross; they can only meet at a station."


def build_phase(sim: MiniMetroSim, vis: Visualizer) -> bool:
    """Edit the design until the user starts the game (True) or quits (False)."""
    surface, font = vis.surface, vis.font
    active_line: int | None = None
    pending_first: int | None = None  # first station picked for a NEW_LINE, awaiting second
    add_train_mode = False
    force_new_line = False
    message = ""

    def attempt(action, success):
        nonlocal message
        reason = _why_illegal(sim, action)
        if reason:
            message = f"Not allowed: {reason[0].lower()}{reason[1:]}"
            return False
        apply_action(sim, action)
        message = success
        return True

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
                    active_line = pending_first = None
                    message = "NEW LINE on next click."
                elif event.key == pygame.K_l:
                    vis.dev = not vis.dev
                elif event.key in (pygame.K_d, pygame.K_BACKSPACE, pygame.K_DELETE):
                    if active_line is not None and active_line < len(sim.lines):
                        sim.lines.pop(active_line)  # with its trains; re-index the rest
                        for i, l in enumerate(sim.lines):
                            l.index = i
                        message = f"Deleted line {active_line}."
                        active_line = None
                    else:
                        message = "Select a line first (click it)."

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # right-click: cancel pending pick, else deselect
                    if pending_first is not None or add_train_mode or force_new_line:
                        pending_first, add_train_mode, force_new_line = None, False, False
                    else:
                        active_line = None
                    message = ""
                    continue
                if event.button != 1:
                    continue
                st_idx = _station_at(sim, event.pos)
                ln_idx = _line_at(sim, event.pos) if st_idx is None else None

                if add_train_mode:
                    if ln_idx is not None:
                        attempt((ActionType.ADD_TRAIN, ln_idx), f"Added a train to line {ln_idx}.")
                    add_train_mode = False
                elif st_idx is not None:
                    if pending_first is not None:
                        a, pending_first, force_new_line = pending_first, None, False
                        if attempt((ActionType.NEW_LINE, a, st_idx), f"Created line {len(sim.lines)}."):
                            active_line = len(sim.lines) - 1
                    elif active_line is None or force_new_line:
                        if len(sim.lines) >= MAX_LINES:
                            message = f"Not allowed: all {MAX_LINES} lines are in use."
                            force_new_line = False
                        else:
                            pending_first = st_idx
                            message = "Pick the second station for the new line."
                    else:
                        line = sim.lines[active_line]
                        target = sim.stations[st_idx]
                        end = _nearest_endpoint(line, target.x, target.y)
                        attempt((ActionType.EXTEND_LINE, active_line, end, st_idx), f"Extended line {active_line}.")
                elif ln_idx is not None:
                    active_line, pending_first = ln_idx, None
                    message = f"Selected line {ln_idx}."
                else:
                    message = ""

        # ---- draw ----
        surface.fill(BG_COLOR)
        outlines = {}
        if active_line is not None and active_line < len(sim.lines):
            outlines = {sim.stations.index(s): line_color(active_line) for s in sim.lines[active_line].stations}
        if pending_first is not None:
            outlines[pending_first] = PICKED
        draw_network(surface, sim, vis.small_font, highlight=active_line, outlines=outlines,
                     loads=station_load_factors(sim) if vis.dev else None, labels=vis.dev)

        # Preview of the next segment: in the new line's color, or red if the rules forbid it.
        preview = None
        if pending_first is not None:
            preview = (pending_first, hovered_st, LINE_COLORS[len(sim.lines) % len(LINE_COLORS)])
        elif (active_line is not None and active_line < len(sim.lines) and not add_train_mode
              and not force_new_line):
            line = sim.lines[active_line]
            end = _nearest_endpoint(line, *mouse_pos) if hovered_st is None else \
                _nearest_endpoint(line, sim.stations[hovered_st].x, sim.stations[hovered_st].y)
            preview = (sim.stations.index(line.stations[end]), hovered_st, line_color(active_line))
        if preview is not None:
            src, dst, color = preview
            start = (sim.stations[src].x, sim.stations[src].y)
            if dst is None:
                draw_dashed(surface, start, mouse_pos, GHOST)
            else:
                if dst == src or crosses_network(sim, src, dst):
                    color = ILLEGAL
                draw_dashed(surface, start, (sim.stations[dst].x, sim.stations[dst].y), color, width=4)

        if hovered_ln is not None and hovered_ln != active_line:
            line = sim.lines[hovered_ln]
            for endpoint in (line.stations[0], line.stations[-1]):
                pygame.draw.circle(surface, line_color(hovered_ln), (endpoint.x, endpoint.y), STATION_RADIUS + 6, 2)

        mode = "ADD TRAIN" if add_train_mode else ("NEW LINE next" if force_new_line else "BUILD")
        draw_panel(surface, font, [
            f"BUILD MODE   map {vis.info.get('map')}",
            f"mode {mode}   selected {'line ' + str(active_line) if active_line is not None else 'none'}"
            f"   lines {len(sim.lines)}/{MAX_LINES}   trains {sum(len(l.trains) for l in sim.lines)}/{MAX_TRAINS}",
            (message, ILLEGAL if message.startswith("Not allowed") else MUTED),
        ], (10, 10))
        draw_panel(vis.surface, vis.small_font, [(row, MUTED) for row in (
            "L-click station: new line / extend selected line",
            "L-click line: select   R-click: cancel",
            "T then click line: add train   N: new line",
            "D / Backspace: delete selected line   L: load factors on / off",
            "SPACE / ENTER: play   ESC: quit",
        )], (10, vis.height - 10), anchor="bottomleft")
        pygame.display.flip()
        vis.clock.tick(60)


def main():
    parser = argparse.ArgumentParser(description="Build a network by hand and play it.")
    parser.add_argument("--map", type=int, default=1, help="map seed")
    parser.add_argument("--level", choices=list(LEVELS), default="classic", help="kind of map (maps.LEVELS)")
    args = parser.parse_args()

    vis = Visualizer(mode="dev", caption="Mini Metro - Build Your Own")
    vis.auto_next = None
    vis.next_label = "edit the design"
    spec = LEVELS[args.level].spec(args.map)
    vis.new_map(map=args.map, stations=f"{spec.stations} {spec.layout}", policy="you")
    sim = MiniMetroSim(args.map, map_spec=spec)
    rng = random.Random()
    try:
        while build_phase(sim, vis):
            while True:
                game = copy.deepcopy(sim)
                game.set_passenger_seed(rng.randrange(2 ** 31))
                outcome = vis.play(game)
                print(f"ticks {game.time} | delivered {game.delivered}")
                if outcome != "replay":
                    break
    except SystemExit:
        pass
    finally:
        vis.close()


if __name__ == "__main__":
    main()
