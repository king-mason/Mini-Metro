"""simulator.py: map generation, passenger randomness, play-outs, and the failure rule."""

import copy

from hypothesis import given, strategies as st

from simulator import MiniMetroSim, max_overcrowd_ticks, max_waiting, run_playout
from tests.helpers import NUM_STATIONS, heuristic_design

map_seeds = st.integers(min_value=0, max_value=10 ** 6)


@given(map_seeds)
def test_maps_are_reproducible_and_well_formed(map_seed):
    a, b = MiniMetroSim(NUM_STATIONS, map_seed=map_seed), MiniMetroSim(NUM_STATIONS, map_seed=map_seed)
    assert [(s.x, s.y, s.type) for s in a.stations] == [(s.x, s.y, s.type) for s in b.stations]
    assert len(a.stations) == NUM_STATIONS
    assert len({(s.x, s.y) for s in a.stations}) == NUM_STATIONS      # distinct grid points
    assert len({s.type for s in a.stations}) >= 2                     # passengers always have a target


@given(map_seeds, st.integers(min_value=0, max_value=NUM_STATIONS - 1))
def test_target_probabilities(map_seed, i):
    sim = MiniMetroSim(NUM_STATIONS, map_seed=map_seed)
    probs = sim.target_type_probs(i)
    assert abs(sum(probs.values()) - 1) < 1e-9
    assert probs[sim.stations[i].type] == 0


@given(st.integers(min_value=0, max_value=50), st.integers(min_value=0, max_value=2 ** 31 - 1))
def test_playouts_are_reproducible_and_leave_the_design_untouched(map_index, passenger_seed):
    design = heuristic_design(10_000 + map_index).sim
    snapshot = copy.deepcopy(design)
    first = run_playout(design, passenger_seed, 800)
    assert run_playout(design, passenger_seed, 800) == first
    assert design.time == 0 and design.delivered == 0
    assert [len(s.waiting) for s in design.stations] == [len(s.waiting) for s in snapshot.stations]


def test_passenger_seed_is_independent_of_global_random():
    import random
    design = heuristic_design(10_001).sim
    random.seed(1)
    a = run_playout(design, 42, 1000)
    random.seed(2)
    assert run_playout(design, 42, 1000) == a


def test_failure_rule():
    """The game ends once the overcrowded station-ticks, summed over the game, pass the limit."""
    sim = MiniMetroSim(NUM_STATIONS, map_seed=7, passenger_seed=0)   # no lines: everyone waits
    last = 0
    while not sim.failed:
        sim.step()
        assert sim.overcrowded_ticks >= last
        last = sim.overcrowded_ticks
        assert sim.failed == (sim.overcrowded_ticks > max_overcrowd_ticks)
    assert any(len(s.waiting) > max_waiting for s in sim.stations)
