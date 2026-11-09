"""rules.py: crossing geometry, legality, and what applying actions does."""

import random

import pytest
from hypothesis import given, strategies as st

from point import Point
from rules import (MAX_LINES, MAX_TRAINS, MAX_TRAINS_PER_LINE, ActionType, action_segments, apply_action,
                   can_add_train, crosses_network, is_legal, line_segments, segments_cross)
from tests.helpers import crossing_pairs, fresh_map, random_action, random_legal_design

coord = st.integers(min_value=-50, max_value=50)
points = st.builds(Point, coord, coord)
seeds = st.integers(min_value=0, max_value=10 ** 6)


# ---- geometry ----

def test_crossing_examples():
    a, b, c, d = Point(0, 0), Point(10, 10), Point(0, 10), Point(10, 0)
    assert segments_cross(a, b, c, d)                                   # an X
    assert not segments_cross(Point(0, 0), Point(10, 0), Point(5, 0), Point(5, 10))  # T: touches, no cross
    assert not segments_cross(Point(0, 0), Point(10, 0), Point(5, 0), Point(15, 0))  # collinear overlap
    assert not segments_cross(a, b, a, d)                               # shared station
    assert not segments_cross(a, Point(1, 1), c, d)                     # apart


@given(points, points, points, points)
def test_crossing_is_symmetric(a, b, c, d):
    r = segments_cross(a, b, c, d)
    assert r == segments_cross(b, a, c, d) == segments_cross(a, b, d, c) == segments_cross(c, d, a, b)


@given(points, points, points, points, coord, coord)
def test_crossing_survives_mirroring_and_translation(a, b, c, d, dx, dy):
    r = segments_cross(a, b, c, d)
    mirror = [Point(-p.x, p.y) for p in (a, b, c, d)]
    moved = [Point(p.x + dx, p.y + dy) for p in (a, b, c, d)]
    assert segments_cross(*mirror) == r
    assert segments_cross(*moved) == r


@given(points, points, points)
def test_segments_sharing_an_end_never_cross(a, b, c):
    assert not segments_cross(a, b, a, c)


# ---- legality ----

@given(seeds, seeds)
def test_legal_actions_never_build_a_crossing(map_seed, seed):
    sim = random_legal_design(map_seed % 10_000, seed)
    assert crossing_pairs(sim) == []
    assert len(sim.lines) <= MAX_LINES
    assert sum(len(l.trains) for l in sim.lines) <= MAX_TRAINS
    for line in sim.lines:
        assert len(set(map(id, line.stations))) == len(line.stations)          # no repeated station
        assert 1 <= len(line.trains) <= min(MAX_TRAINS_PER_LINE, len(line.stations) - 1)


@given(seeds, seeds)
def test_is_legal_rejects_exactly_the_crossings(map_seed, seed):
    """For line-building actions that respect the other rules, illegal means crossing."""
    sim, rng = random_legal_design(map_seed % 10_000, seed), random.Random(seed)
    for _ in range(40):
        action = random_action(sim, rng)
        if action[0] == ActionType.ADD_TRAIN:
            assert is_legal(sim, action) == can_add_train(sim, action[1])
            continue
        if action[0] == ActionType.NEW_LINE and (len(sim.lines) >= MAX_LINES or action[1] == action[2]
                                                 or sum(len(l.trains) for l in sim.lines) >= MAX_TRAINS):
            assert not is_legal(sim, action)
            continue
        if action[0] != ActionType.NEW_LINE and sim.stations[action[3]] in sim.lines[action[1]].stations:
            assert not is_legal(sim, action)
            continue
        crosses = any(crosses_network(sim, a, b) for a, b in action_segments(sim, action))
        assert is_legal(sim, action) == (not crosses)


def test_line_and_train_limits():
    sim = fresh_map(3)
    pairs = [(i, j) for i in range(20) for j in range(i + 1, 20)]
    for a, b in pairs:
        if len(sim.lines) == MAX_LINES:
            break
        if is_legal(sim, (ActionType.NEW_LINE, a, b)):
            apply_action(sim, (ActionType.NEW_LINE, a, b))
    assert len(sim.lines) == MAX_LINES
    assert not is_legal(sim, (ActionType.NEW_LINE, *next(p for p in pairs)))
    # a two-station line has one segment, so it can hold only its first train
    assert not can_add_train(sim, 0)
    assert apply_action(sim, (ActionType.ADD_TRAIN, 0))["train"] is None


# ---- applying actions ----

def test_new_line_starts_with_one_train_at_its_first_station():
    sim = fresh_map(5)
    a, b = next((i, j) for i in range(20) for j in range(i + 1, 20) if is_legal(sim, (ActionType.NEW_LINE, i, j)))
    apply_action(sim, (ActionType.NEW_LINE, a, b))
    (line,) = sim.lines
    assert [sim.stations.index(s) for s in line.stations] == [a, b]
    assert len(line.trains) == 1 and line.trains[0].position == 0


@given(seeds, seeds)
def test_insert_detours_one_segment(map_seed, seed):
    sim, rng = random_legal_design(map_seed % 10_000, seed), random.Random(seed)
    inserts = [(ActionType.INSERT_STATION, li, k, j) for li, line in enumerate(sim.lines)
               for k in range(len(line.stations) - 1) for j in range(len(sim.stations))]
    legal = [a for a in inserts if is_legal(sim, a)]
    if not legal:
        return
    _, li, k, j = action = rng.choice(legal)
    before = list(sim.lines[li].stations)
    apply_action(sim, action)
    assert sim.lines[li].stations == before[:k + 1] + [sim.stations[j]] + before[k + 1:]
    assert crossing_pairs(sim) == []


def test_segments_listed_by_station_index():
    sim = fresh_map(8)
    legal = next((i, j) for i in range(20) for j in range(i + 1, 20) if is_legal(sim, (ActionType.NEW_LINE, i, j)))
    apply_action(sim, (ActionType.NEW_LINE, *legal))
    assert line_segments(sim) == [legal]


@pytest.mark.parametrize("action", [(ActionType.NOOP,)])
def test_noop_is_always_legal(action):
    assert is_legal(fresh_map(0), action)
