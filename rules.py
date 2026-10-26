"""Design rules: the actions that build a network, and which of them are legal.

A design is built at tick 0, before any passenger spawns. These are the game's rules,
shared by everything that builds a design (the agent, search.py, the GA, play.py):

  - at most `lines.max` lines, each visiting a station at most once;
  - at most `trains.total` trains, `trains.max_per_line` per line, and no more trains on
    a line than it has segments;
  - a line's trains run evenly spaced in time (objects.Line.space_trains);
  - no two segments may cross. Lines meet only at shared stations, where passengers can
    change trains. (A segment passing exactly over a station or along another segment is
    not a crossing.)
"""

from enum import Enum

from simulator import MiniMetroSim, config, max_lines as MAX_LINES

MAX_TRAINS = config["trains"]["total"]
MAX_TRAINS_PER_LINE = config["trains"]["max_per_line"]


class ActionType(Enum):
    EXTEND_LINE = 0     # (EXTEND_LINE, line_idx, endpoint_idx, new_station_idx)
    NEW_LINE = 1        # (NEW_LINE, station_a, station_b)
    NOOP = 2            # (NOOP,)
    ADD_TRAIN = 3       # (ADD_TRAIN, line_idx)
    INSERT_STATION = 4  # (INSERT_STATION, line_idx, segment_idx, new_station_idx): detour the
                        # segment line[segment_idx] -> line[segment_idx + 1] through the station.
                        # Only used to connect stations no other legal move can reach.


def line_segments(sim: MiniMetroSim) -> list[tuple[int, int]]:
    """Every built segment as a (station index, station index) pair."""
    segments = []
    for line in sim.lines:
        idxs = [sim.stations.index(s) for s in line.stations]
        for i in range(len(idxs) - 1):
            segments.append((idxs[i], idxs[i + 1]))
    return segments


def orient(a, b, c) -> int:
    """Turn direction a -> b -> c: +1 counter-clockwise, -1 clockwise, 0 collinear."""
    v = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    return (v > 0) - (v < 0)


def segments_cross(a, b, c, d) -> bool:
    """True if segments ab and cd properly cross. Shared endpoints, touches, and collinear
    overlaps don't count. (A strict-inequality ccw test counted a station sitting on another
    segment's line as a crossing in one orientation but not in its mirror image.)"""
    if a is c or a is d or b is c or b is d:
        return False
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def crosses_network(sim: MiniMetroSim, a_idx: int, b_idx: int,
                    segments: list[tuple[int, int]] | None = None) -> bool:
    """Would a segment between stations a_idx and b_idx cross a built segment?
    Pass `segments` (from line_segments) to check against a design that isn't built."""
    a, b = sim.stations[a_idx].pos, sim.stations[b_idx].pos
    for c_idx, d_idx in line_segments(sim) if segments is None else segments:
        if segments_cross(a, b, sim.stations[c_idx].pos, sim.stations[d_idx].pos):
            return True
    return False


def action_segment(sim: MiniMetroSim, action: tuple) -> tuple[int, int] | None:
    """The segment an EXTEND_LINE or NEW_LINE would add, as station indices (None otherwise).
    The new station (EXTEND) or station_a (NEW_LINE) comes second / first respectively."""
    if action[0] == ActionType.EXTEND_LINE:
        _, line_idx, endpoint_idx, new_idx = action
        endpoint = sim.lines[line_idx].stations[endpoint_idx]
        return sim.stations.index(endpoint), new_idx
    if action[0] == ActionType.NEW_LINE:
        return action[1], action[2]
    return None


def action_segments(sim: MiniMetroSim, action: tuple) -> list[tuple[int, int]]:
    """Every segment a line-building action would add (an insertion adds two)."""
    if action[0] == ActionType.INSERT_STATION:
        _, line_idx, k, new_idx = action
        stations = sim.lines[line_idx].stations
        a, b = sim.stations.index(stations[k]), sim.stations.index(stations[k + 1])
        return [(a, new_idx), (new_idx, b)]
    seg = action_segment(sim, action)
    return [] if seg is None else [seg]


def can_add_train(sim: MiniMetroSim, line_idx: int) -> bool:
    line = sim.lines[line_idx]
    return (sum(len(l.trains) for l in sim.lines) < MAX_TRAINS
            and len(line.trains) < MAX_TRAINS_PER_LINE
            and len(line.stations) > len(line.trains) + 1)


def is_legal(sim: MiniMetroSim, action: tuple) -> bool:
    """Whether `action` is allowed in `sim` under the rules above."""
    t = action[0]
    if t == ActionType.NOOP:
        return True
    if t == ActionType.ADD_TRAIN:
        return can_add_train(sim, action[1])
    if t == ActionType.NEW_LINE:
        if len(sim.lines) >= MAX_LINES or action[1] == action[2]:
            return False
    elif t in (ActionType.EXTEND_LINE, ActionType.INSERT_STATION):
        if sim.stations[action[3]] in sim.lines[action[1]].stations:
            return False
    # The segment an insertion replaces shares an end with each new one, so never "crosses" them.
    segments = line_segments(sim)
    return not any(crosses_network(sim, a, b, segments) for a, b in action_segments(sim, action))


def apply_add_train(sim: MiniMetroSim, line_idx: int):
    from objects import Train
    if not can_add_train(sim, line_idx):
        return None
    line = sim.lines[line_idx]
    train = Train(line)
    line.add_train(train)
    line.space_trains()
    return train


def apply_extend(sim: MiniMetroSim, line_idx: int, endpoint_idx: int, new_idx: int):
    line = sim.lines[line_idx]
    new_station = sim.stations[new_idx]
    if endpoint_idx == 0:
        line.stations.insert(0, new_station)
    else:
        line.stations.append(new_station)
    line.space_trains()
    return line


def apply_insert(sim: MiniMetroSim, line_idx: int, segment_idx: int, new_idx: int):
    line = sim.lines[line_idx]
    line.stations.insert(segment_idx + 1, sim.stations[new_idx])
    line.space_trains()
    return line


def apply_action(sim: MiniMetroSim, action: tuple) -> dict:
    """Apply an action and return the entities it created/modified.

    Returns: {"line": Line | None, "train": Train | None, "extended": bool}
      - NEW_LINE: line + auto-added train, extended=False
      - EXTEND_LINE: the extended line, no new train, extended=True
      - ADD_TRAIN: the new train (or None if capped), no new line
      - NOOP: all None
    """
    t = action[0]
    result = {"line": None, "train": None, "extended": False}
    if t == ActionType.EXTEND_LINE:
        result["line"] = apply_extend(sim, action[1], action[2], action[3])
        result["extended"] = True
    elif t == ActionType.NEW_LINE:
        prev_n = len(sim.lines)
        sim.add_line([action[1], action[2]])
        if len(sim.lines) > prev_n:
            new_line = sim.lines[-1]
            result["line"] = new_line
            result["train"] = new_line.trains[0] if new_line.trains else None
    elif t == ActionType.ADD_TRAIN:
        result["train"] = apply_add_train(sim, action[1])
    elif t == ActionType.INSERT_STATION:
        result["line"] = apply_insert(sim, action[1], action[2], action[3])
        result["extended"] = True
    elif t == ActionType.NOOP:
        pass
    else:
        raise ValueError(f"Unknown action type: {t}")
    return result


def action_label(action: tuple, sim: MiniMetroSim) -> str:
    """Short description of `action`, read against the state before it is applied."""
    def name(i):
        return f"{sim.stations[i].type.name.lower()} #{i}"

    t = action[0]
    if t == ActionType.EXTEND_LINE:
        _, li, ei, ni = action
        return f"extend L{li} {'front' if ei == 0 else 'back'} to {name(ni)}"
    if t == ActionType.NEW_LINE:
        return f"new line L{len(sim.lines)}: {name(action[1])} - {name(action[2])}"
    if t == ActionType.ADD_TRAIN:
        return f"train {len(sim.lines[action[1]].trains) + 1} on L{action[1]}"
    if t == ActionType.INSERT_STATION:
        _, li, k, ni = action
        return f"detour L{li} through {name(ni)}"
    return "done"
