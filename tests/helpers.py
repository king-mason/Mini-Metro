"""Builders shared by the test modules."""

import copy
import random
from functools import lru_cache

import numpy as np

from config import MAX_LINES, NUM_STATIONS
from rules import ActionType, apply_action, is_legal, line_segments, segments_cross
from simulator import MiniMetroSim


def fresh_map(map_seed: int) -> MiniMetroSim:
    return MiniMetroSim(NUM_STATIONS, map_seed=map_seed)


def random_action(sim: MiniMetroSim, rng: random.Random) -> tuple:
    """Any action at all, legal or not: stations are drawn from the whole map."""
    n = len(sim.stations)
    kind = rng.choice([ActionType.NEW_LINE, ActionType.EXTEND_LINE, ActionType.ADD_TRAIN, ActionType.INSERT_STATION])
    if kind == ActionType.NEW_LINE or not sim.lines:
        return (ActionType.NEW_LINE, rng.randrange(n), rng.randrange(n))
    li = rng.randrange(len(sim.lines))
    if kind == ActionType.EXTEND_LINE:
        return (ActionType.EXTEND_LINE, li, rng.choice((0, -1)), rng.randrange(n))
    if kind == ActionType.INSERT_STATION:
        return (ActionType.INSERT_STATION, li, rng.randrange(len(sim.lines[li].stations) - 1), rng.randrange(n))
    return (ActionType.ADD_TRAIN, li)


def random_legal_design(map_seed: int, seed: int, attempts: int = 200) -> MiniMetroSim:
    """A design built from random actions, applying only those rules.is_legal allows."""
    sim, rng = fresh_map(map_seed), random.Random(seed)
    for _ in range(attempts):
        action = random_action(sim, rng)
        if is_legal(sim, action):
            apply_action(sim, action)
    return sim


@lru_cache(maxsize=None)
def heuristic_agent():
    """The structural heuristic (Q off), shared by all tests. Imports torch on first use."""
    from agent import Agent
    return Agent(use_q=False)


@lru_cache(maxsize=64)
def _heuristic_design_cached(map_seed: int):
    return heuristic_agent().design(fresh_map(map_seed))


def heuristic_design(map_seed: int):
    """The structural heuristic's DesignResult for a map (cached; returns a copy)."""
    return copy.deepcopy(_heuristic_design_cached(map_seed))


def crossing_pairs(sim: MiniMetroSim) -> list:
    pos = [s.pos for s in sim.stations]
    segs = line_segments(sim)
    return [(s, t) for k, s in enumerate(segs) for t in segs[k + 1:]
            if segments_cross(pos[s[0]], pos[s[1]], pos[t[0]], pos[t[1]])]


def headway_gaps(line) -> list[list[int]]:
    """For each (station, direction) on `line`, the gaps in ticks between consecutive trains
    over one round trip from the trains' current state. Movement only."""
    period = line.round_trip_ticks()
    trains = [copy.copy(t) for t in line.trains]
    visits: dict = {}
    for tick in range(2 * period):
        for train in trains:
            was = train.at_station
            train.advance(1)
            if train.at_station and not was and tick >= period:
                visits.setdefault((train.position, train.direction), []).append(tick)
    return [list(np.diff(sorted(t) + [sorted(t)[0] + period])) for t in visits.values()]


__all__ = ["NUM_STATIONS", "MAX_LINES", "fresh_map", "random_action", "random_legal_design",
           "heuristic_agent", "heuristic_design", "crossing_pairs", "headway_gaps"]
