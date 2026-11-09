"""objects.py: train movement and even spacing."""

from hypothesis import given, strategies as st

from objects import Train
from rules import ActionType, apply_action, can_add_train, is_legal
from tests.helpers import headway_gaps, random_legal_design

seeds = st.integers(min_value=0, max_value=10 ** 6)


def lines_with_trains(sim, at_least=1):
    return [l for l in sim.lines if len(l.trains) >= at_least]


def fill_trains(sim):
    """Add trains wherever the rules allow, so lines carry 2-4 trains."""
    for li in range(len(sim.lines)):
        while can_add_train(sim, li):
            apply_action(sim, (ActionType.ADD_TRAIN, li))
    return sim


@given(seeds, seeds)
def test_round_trip_returns_a_train_to_its_start(map_seed, seed):
    for line in lines_with_trains(random_legal_design(map_seed % 10_000, seed)):
        probe = Train(line)
        probe.advance(line.round_trip_ticks())
        assert (probe.position, probe.direction, probe.progress, probe.at_station) == (0, 1, 0.0, True)


@given(seeds, seeds)
def test_advance_matches_step_without_passengers(map_seed, seed):
    sim = random_legal_design(map_seed % 10_000, seed)
    for line in lines_with_trains(sim):
        a, b = Train(line), Train(line)
        for _ in range(97):
            a.advance(1)
            b.step(sim.lines)   # stations are empty at tick 0, so step only moves
        assert (a.position, a.direction, a.progress, a.at_station) == (b.position, b.direction, b.progress, b.at_station)


@given(seeds, seeds)
def test_trains_are_evenly_spaced(map_seed, seed):
    """Every station sees a train in each direction every round_trip / T ticks (to the tick)."""
    sim = fill_trains(random_legal_design(map_seed % 10_000, seed))
    for line in lines_with_trains(sim, at_least=2):
        for gaps in headway_gaps(line):
            assert max(gaps) - min(gaps) <= 1, gaps


@given(seeds, seeds, st.integers(min_value=50, max_value=600))
def test_spacing_holds_during_play(map_seed, seed, ticks):
    """A stop always takes one tick, however many board, so play keeps the spacing."""
    sim = fill_trains(random_legal_design(map_seed % 10_000, seed))
    sim.set_passenger_seed(seed)
    start = {id(l): headway_gaps(l) for l in sim.lines}
    for _ in range(ticks):
        sim.step()
    for line in lines_with_trains(sim, at_least=2):
        assert sorted(map(sorted, headway_gaps(line))) == sorted(map(sorted, start[id(line)]))


@given(seeds, seeds)
def test_extending_a_line_respaces_its_trains(map_seed, seed):
    sim = fill_trains(random_legal_design(map_seed % 10_000, seed))
    for li, line in enumerate(sim.lines):
        if len(line.trains) < 2:
            continue
        others = [j for j in range(len(sim.stations)) if sim.stations[j] not in line.stations]
        for j in others:
            if is_legal(sim, (ActionType.EXTEND_LINE, li, -1, j)):
                apply_action(sim, (ActionType.EXTEND_LINE, li, -1, j))
                for gaps in headway_gaps(line):
                    assert max(gaps) - min(gaps) <= 1
                break


def test_train_capacity_is_respected():
    sim = fill_trains(random_legal_design(11, 4))
    sim.set_passenger_seed(1)
    for _ in range(3000):
        sim.step()
        assert all(len(t.passengers) <= t.capacity for l in sim.lines for t in l.trains)
        if sim.failed:
            break
