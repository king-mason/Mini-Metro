"""Design rules: the actions that build a network, which are legal, and how they apply.

A design is built at tick 0, before any passenger spawns:
  - at most MAX_LINES lines, each visiting a station at most once;
  - at most MAX_TRAINS trains, MAX_TRAINS_PER_LINE per line, and no more trains on a line
    than it has segments; a new line comes with a train, so it needs one free;
  - a line's trains run evenly spaced in time (objects.Line.space_trains);
  - no two segments may cross: lines meet only at shared stations.
"""

from enum import Enum

from config import MAX_LINES, MAX_TRAINS, MAX_TRAINS_PER_LINE
from objects import Train
from simulator import MiniMetroSim


class ActionType(Enum):
    EXTEND_LINE = 0     # (EXTEND_LINE, line_idx, end, station): end is 0 (front) or -1 (back)
    NEW_LINE = 1        # (NEW_LINE, station_a, station_b)
    NOOP = 2            # (NOOP,): stop the current design phase
    ADD_TRAIN = 3       # (ADD_TRAIN, line_idx)
    INSERT_STATION = 4  # (INSERT_STATION, line_idx, k, station): detour segment k -> k+1 through
                        # the station. Only for connecting stations nothing else can reach.


NOOP = (ActionType.NOOP,)


# ---- geometry ----

def line_segments(sim: MiniMetroSim) -> list[tuple[int, int]]:
    """Every built segment as a pair of station indices."""
    index = {id(s): i for i, s in enumerate(sim.stations)}
    return [(index[id(a)], index[id(b)]) for line in sim.lines for a, b in zip(line.stations, line.stations[1:])]


def orient(a, b, c) -> int:
    """Turn direction a -> b -> c: +1 counter-clockwise, -1 clockwise, 0 collinear."""
    v = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    return (v > 0) - (v < 0)


def segments_cross(a, b, c, d) -> bool:
    """True if segments ab and cd properly cross; shared ends, touches and collinear overlaps
    don't count."""
    if a is c or a is d or b is c or b is d:
        return False
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def crosses_network(sim: MiniMetroSim, a: int, b: int, segments: list[tuple[int, int]] | None = None) -> bool:
    """Would a segment between stations a and b cross a built one (or one of `segments`)?"""
    p, q = sim.stations[a].pos, sim.stations[b].pos
    return any(segments_cross(p, q, sim.stations[c].pos, sim.stations[d].pos)
               for c, d in (line_segments(sim) if segments is None else segments))


def action_segment(sim: MiniMetroSim, action: tuple) -> tuple[int, int] | None:
    """The segment an EXTEND_LINE (line end, new station) or NEW_LINE would add."""
    if action[0] == ActionType.EXTEND_LINE:
        _, li, end, new = action
        return sim.stations.index(sim.lines[li].stations[end]), new
    if action[0] == ActionType.NEW_LINE:
        return action[1], action[2]
    return None


def action_segments(sim: MiniMetroSim, action: tuple) -> list[tuple[int, int]]:
    """Every segment a line-building action would add (an insertion adds two)."""
    if action[0] == ActionType.INSERT_STATION:
        _, li, k, new = action
        stations = sim.lines[li].stations
        a, b = sim.stations.index(stations[k]), sim.stations.index(stations[k + 1])
        return [(a, new), (new, b)]
    seg = action_segment(sim, action)
    return [] if seg is None else [seg]


# ---- legality ----

def trains_in_use(sim: MiniMetroSim) -> int:
    return sum(len(line.trains) for line in sim.lines)


def can_add_train(sim: MiniMetroSim, line_idx: int) -> bool:
    line = sim.lines[line_idx]
    return (trains_in_use(sim) < MAX_TRAINS and len(line.trains) < MAX_TRAINS_PER_LINE
            and len(line.stations) > len(line.trains) + 1)


def is_legal(sim: MiniMetroSim, action: tuple) -> bool:
    """Whether `action` is allowed in `sim` under the rules above."""
    kind = action[0]
    if kind == ActionType.NOOP:
        return True
    if kind == ActionType.ADD_TRAIN:
        return can_add_train(sim, action[1])
    if kind == ActionType.NEW_LINE:
        if len(sim.lines) >= MAX_LINES or action[1] == action[2] or trains_in_use(sim) >= MAX_TRAINS:
            return False
    elif sim.stations[action[3]] in sim.lines[action[1]].stations:   # EXTEND_LINE, INSERT_STATION
        return False
    segments = line_segments(sim)
    return not any(crosses_network(sim, a, b, segments) for a, b in action_segments(sim, action))


# ---- applying actions ----

def apply_add_train(sim: MiniMetroSim, line_idx: int) -> Train | None:
    """Add a train if the line can take one (None otherwise)."""
    if not can_add_train(sim, line_idx):
        return None
    line = sim.lines[line_idx]
    train = Train(line)
    line.add_train(train)
    line.space_trains()
    return train


def apply_extend(sim: MiniMetroSim, line_idx: int, end: int, new_idx: int):
    line = sim.lines[line_idx]
    if end == 0:
        line.stations.insert(0, sim.stations[new_idx])
    else:
        line.stations.append(sim.stations[new_idx])
    line.space_trains()


def apply_insert(sim: MiniMetroSim, line_idx: int, k: int, new_idx: int):
    line = sim.lines[line_idx]
    line.stations.insert(k + 1, sim.stations[new_idx])
    line.space_trains()


def apply_action(sim: MiniMetroSim, action: tuple):
    """Apply a legal action to `sim` (check is_legal first)."""
    kind = action[0]
    if kind == ActionType.EXTEND_LINE:
        apply_extend(sim, *action[1:])
    elif kind == ActionType.NEW_LINE:
        sim.add_line([action[1], action[2]])
    elif kind == ActionType.ADD_TRAIN:
        apply_add_train(sim, action[1])
    elif kind == ActionType.INSERT_STATION:
        apply_insert(sim, *action[1:])
    elif kind != ActionType.NOOP:
        raise ValueError(f"unknown action type: {kind}")


def connect_isolated(sim: MiniMetroSim, on_apply=None) -> tuple[int, int]:
    """Connect every station left on no line, cheapest added track first: extend a line end,
    else start a new line, else detour a segment through it. on_apply(action) is called
    before each. Returns (connections made, stations nothing legal could reach)."""
    def dist(a, b):
        return sim.stations[a].pos.get_distance(sim.stations[b].pos)

    def added_track(action):
        new = action_segments(sim, action)
        if action[0] != ActionType.INSERT_STATION:
            return dist(*new[0])
        (a, s), (_, b) = new
        return dist(a, s) + dist(s, b) - dist(a, b)

    made = stranded = 0
    for i, station in enumerate(sim.stations):
        if any(station in line.stations for line in sim.lines):
            continue
        extends = [(ActionType.EXTEND_LINE, li, end, i) for li in range(len(sim.lines)) for end in (0, -1)]
        new_lines = [(ActionType.NEW_LINE, i, j) for j in range(len(sim.stations)) if j != i]
        inserts = [(ActionType.INSERT_STATION, li, k, i)
                   for li, line in enumerate(sim.lines) for k in range(len(line.stations) - 1)]
        for options in (extends, new_lines, inserts):
            legal = [a for a in options if is_legal(sim, a)]
            if legal:
                action = min(legal, key=added_track)
                if on_apply is not None:
                    on_apply(action)
                apply_action(sim, action)
                made += 1
                break
        else:
            stranded += 1
    return made, stranded


def action_label(action: tuple, sim: MiniMetroSim) -> str:
    """Short description of `action`, read against the state before it is applied."""
    def name(i):
        return f"{sim.stations[i].type.name.lower()} #{i}"

    kind = action[0]
    if kind == ActionType.EXTEND_LINE:
        _, li, end, new = action
        return f"extend L{li} {'front' if end == 0 else 'back'} to {name(new)}"
    if kind == ActionType.NEW_LINE:
        return f"new line L{len(sim.lines)}: {name(action[1])} - {name(action[2])}"
    if kind == ActionType.ADD_TRAIN:
        return f"train {len(sim.lines[action[1]].trains) + 1} on L{action[1]}"
    if kind == ActionType.INSERT_STATION:
        return f"detour L{action[1]} through {name(action[3])}"
    return "done"
