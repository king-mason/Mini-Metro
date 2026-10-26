"""heuristic.py: candidate moves, metrics, and the heuristic's finished designs."""

from hypothesis import given, strategies as st

from heuristic import get_candidate_actions, neighbor_sets, station_load_factors
from rules import MAX_TRAINS, ActionType, apply_action, is_legal
from tests.helpers import crossing_pairs, fresh_map, heuristic_design, random_legal_design

seeds = st.integers(min_value=0, max_value=10 ** 6)


@given(seeds, seeds, st.sampled_from(["lines", "trains"]))
def test_candidates_are_legal_and_distinct(map_seed, seed, phase):
    sim = random_legal_design(map_seed % 10_000, seed, attempts=seed % 60)
    candidates = get_candidate_actions(sim, phase=phase)
    assert candidates, "there is always something to do, even if it is NOOP"
    assert len(set(candidates)) == len(candidates)
    assert all(is_legal(sim, a) for a in candidates)


@given(seeds, seeds)
def test_noop_once_every_station_is_served(map_seed, seed):
    sim = random_legal_design(map_seed % 10_000, seed)
    served = all(any(s in l.stations for l in sim.lines) for s in sim.stations)
    if served:
        assert (ActionType.NOOP,) in get_candidate_actions(sim, phase="lines")


@given(seeds)
def test_neighbor_sets_are_symmetric(map_seed):
    nbrs = neighbor_sets(fresh_map(map_seed % 10_000), 6)
    assert all(i in nbrs[j] for i, ns in enumerate(nbrs) for j in ns)
    assert all(len(ns) >= 6 for ns in nbrs)


@given(seeds, seeds)
def test_more_trains_never_raise_a_load_factor(map_seed, seed):
    sim = random_legal_design(map_seed % 10_000, seed)
    for li in range(len(sim.lines)):
        if is_legal(sim, (ActionType.ADD_TRAIN, li)):
            before = station_load_factors(sim)
            apply_action(sim, (ActionType.ADD_TRAIN, li))
            after = station_load_factors(sim)
            assert all(after[i] <= before[i] + 1e-12 for i in before)


@given(st.integers(min_value=0, max_value=99))
def test_heuristic_designs_are_complete_and_legal(map_index):
    result = heuristic_design(10_000 + map_index)
    sim = result.sim
    assert result.n_stranded == 0
    assert all(any(s in l.stations for l in sim.lines) for s in sim.stations)
    assert crossing_pairs(sim) == []
    assert sum(len(l.trains) for l in sim.lines) == MAX_TRAINS
