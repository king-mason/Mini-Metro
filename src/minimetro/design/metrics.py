"""Measurements of a design, computed before any passenger spawns: reachability, load
factors, geometry. Shared by the prior (heuristic.py), the network encoding (model.py) and
the viewers."""

import numpy as np

from minimetro.config import MAX_DIST, TRAIN_CAPACITY, TRAIN_SPEED
from minimetro.game.objects import StationType
from minimetro.game.rules import ActionType
from minimetro.game.simulator import MiniMetroSim, current_spawn_chance, current_spawn_interval

STATION_TYPES = list(StationType)
NEIGHBOR_K = 6         # candidate connections join each station only to its k nearest neighbors

# Passengers per tick spawned at each station late in the game, when overcrowding happens.
LAMBDA_REF = current_spawn_chance(10_000) / current_spawn_interval(10_000)
RHO_CLAMP = 5.0        # cap on load factors (an unserved station's is infinite)


def on_a_line(sim: MiniMetroSim, i: int) -> bool:
    return any(sim.stations[i] in line.stations for line in sim.lines)


def line_components(sim: MiniMetroSim) -> tuple[list[int], dict[int, set], dict[int, list[int]]]:
    """Groups of stations joined by lines (a station on no line is its own group).
    Returns (group of each station, station types in each group, members of each group)."""
    n = len(sim.stations)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for line in sim.lines:
        idxs = [sim.stations.index(s) for s in line.stations]
        for k in idxs[1:]:
            ra, rb = find(idxs[0]), find(k)
            if ra != rb:
                parent[ra] = rb

    comp_of = [find(k) for k in range(n)]
    comp_types: dict[int, set] = {}
    comp_members: dict[int, list[int]] = {}
    for k, comp in enumerate(comp_of):
        comp_types.setdefault(comp, set()).add(sim.stations[k].type)
        comp_members.setdefault(comp, []).append(k)
    return comp_of, comp_types, comp_members


def expected_unreachable_demand(sim: MiniMetroSim, i: int, reachable: set) -> float:
    """Share of station i's passengers whose destination type isn't in `reachable`."""
    return sum(p for t, p in sim.target_type_probs(i).items() if t not in reachable)


def nearest_type_distance(sim: MiniMetroSim, i: int, types: set) -> float:
    """Normalized distance from station i to the nearest other station of one of `types`."""
    if not types:
        return 0.0
    here = sim.stations[i].pos
    best = min((here.get_distance(s.pos) for k, s in enumerate(sim.stations) if k != i and s.type in types),
               default=MAX_DIST)
    return best / MAX_DIST


def close_neighbor_scarcity(sim: MiniMetroSim, i: int, k: int = 3) -> float:
    """Mean distance to station i's k nearest neighbors, normalized: high for awkward stations
    that are best connected early."""
    if len(sim.stations) <= 1:
        return 0.0
    here = sim.stations[i].pos
    dists = sorted(here.get_distance(s.pos) for j, s in enumerate(sim.stations) if j != i)
    return min(sum(dists[:k]) / min(k, len(dists)) / MAX_DIST, 1.0)


def station_load_factors(sim: MiniMetroSim) -> dict[int, float]:
    """Load factor rho = arrivals / train capacity passing, per station (rho >= 1: it will
    overcrowd). A line of length L with T trains visits a middle station T * speed / L
    times per tick and an end half as often. Unserved stations get infinity."""
    index = {id(s): i for i, s in enumerate(sim.stations)}
    service = [0.0] * len(sim.stations)
    for line in sim.lines:
        m, length = len(line.stations), line.length()
        if m < 2 or not line.trains or length <= 0:
            continue
        for pos, station in enumerate(line.stations):
            visits = len(line.trains) * TRAIN_SPEED / length
            if pos == 0 or pos == m - 1:
                visits *= 0.5
            service[index[id(station)]] += visits * TRAIN_CAPACITY
    return {i: LAMBDA_REF / s if s > 1e-9 else float("inf") for i, s in enumerate(service)}


def endpoint_load_factor(length: float, n_trains: int) -> float:
    """Load factor at a line's end, its worst-served station."""
    if length <= 0 or n_trains <= 0:
        return float("inf")
    service = 0.5 * n_trains * TRAIN_SPEED / length * TRAIN_CAPACITY
    return LAMBDA_REF / service if service > 1e-9 else float("inf")


def bend_cost(sim: MiniMetroSim, action: tuple) -> float:
    """How sharply an EXTEND_LINE turns at the old end: ~0 below 90 degrees, ~1 beyond
    (a steep sigmoid). 0 for other actions."""
    if action[0] != ActionType.EXTEND_LINE:
        return 0.0
    _, li, end, new = action
    stations = sim.lines[li].stations
    if len(stations) < 2:
        return 0.0
    tip, inner = stations[end], (stations[1] if end == 0 else stations[-2])
    target = sim.stations[new]
    dx1, dy1 = tip.x - inner.x, tip.y - inner.y
    dx2, dy2 = target.x - tip.x, target.y - tip.y
    len1 = (dx1 * dx1 + dy1 * dy1) ** 0.5
    len2 = (dx2 * dx2 + dy2 * dy2) ** 0.5
    if len1 == 0 or len2 == 0:
        return 0.0
    cos_a = min(max((dx1 * dx2 + dy1 * dy2) / (len1 * len2), -1.0), 1.0)
    return 1 / (1 + np.exp(-50 * (np.arccos(cos_a) - np.pi / 2)))


def expected_demand_gain(sim: MiniMetroSim, i: int, j: int) -> tuple[float, float, float]:
    """What joining stations i and j makes reachable: (share of demand newly served on i's
    side, the same on j's side, share of station types newly reachable)."""
    comp_of, comp_types, comp_members = line_components(sim)
    ci, cj = comp_of[i], comp_of[j]
    if ci == cj:
        return 0.0, 0.0, 0.0
    new_for_i = comp_types[cj] - comp_types[ci]
    new_for_j = comp_types[ci] - comp_types[cj]

    def gained(members, new_types):
        return sum(sum(p for t, p in sim.target_type_probs(k).items() if t in new_types) for k in members)

    norm = max(len(sim.stations), 1)
    coverage = (len(new_for_i) + len(new_for_j)) / (2 * len(STATION_TYPES))
    return gained(comp_members[ci], new_for_i) / norm, gained(comp_members[cj], new_for_j) / norm, coverage


def neighbor_sets(sim: MiniMetroSim, k: int = NEIGHBOR_K) -> list[set[int]]:
    """Each station's k nearest neighbors, made symmetric (j is i's neighbor if either counts
    the other). Cached on the sim: stations never move."""
    cache = getattr(sim, "_neighbor_cache", None)
    if cache is not None and cache[0] == k:
        return cache[1]
    n = len(sim.stations)
    sym: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        here = sim.stations[i].pos
        nearest = sorted((j for j in range(n) if j != i), key=lambda j: (here.get_distance(sim.stations[j].pos), j))
        for j in nearest[:k]:
            sym[i].add(j)
            sym[j].add(i)
    sim._neighbor_cache = (k, sym)
    return sym
