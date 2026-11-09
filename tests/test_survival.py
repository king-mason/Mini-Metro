"""survival.py: the analytic survival estimate agrees with the game it models.

Checked against the simulator directly: who passengers want to reach, how long a round trip
takes, and how many riders each train carries out of each station.
"""

import math
from collections import Counter, defaultdict

import numpy as np
from hypothesis import given, strategies as st

from search import spec_of
from survival import CAP_TICKS, SPAWN_RATE, SurvivalModel, target_probs
from tests.helpers import fresh_map, heuristic_design, random_legal_design

map_seeds = st.integers(min_value=0, max_value=10 ** 6)
seeds = st.integers(min_value=0, max_value=10 ** 6)


def test_target_probs_match_the_spawner():
    """The model's destination odds are the ones MiniMetroSim.step actually draws: a map with
    no lines (nobody leaves), passengers pooled by the type of the station they spawned at."""
    sim = fresh_map(3)                   # 11 circles, 5 squares, 2 triangles, a lemon, a rhombus
    sim.set_passenger_seed(0)
    for _ in range(CAP_TICKS):
        sim.step()
    for origin in {s.type for s in sim.stations}:
        at = [i for i, s in enumerate(sim.stations) if s.type == origin]
        seen = Counter(p.target_type for i in at for p in sim.stations[i].waiting)
        total = sum(seen.values())
        assert set(seen) <= set(target_probs(sim, at[0]))
        for t, p in target_probs(sim, at[0]).items():
            assert abs(seen[t] / total - p) < 4 * math.sqrt(p * (1 - p) / total) + 1e-3


@given(map_seeds, seeds)
def test_train_gaps_match_the_round_trip(map_seed, seed):
    sim = random_legal_design(map_seed, seed)
    spec = spec_of(sim)
    gap = SurvivalModel(sim).flows(spec.lines, spec.trains)["gap"]
    for line, g in zip(sim.lines, gap):
        assert math.isclose(g * len(line.trains), line.round_trip_ticks())


@given(map_seeds, seeds)
def test_every_rider_is_routed_or_stuck(map_seed, seed):
    """Every spawned rider boards somewhere or is counted unreachable; each transfer boards again."""
    sim = random_legal_design(map_seed, seed)
    spec = spec_of(sim)
    f = SurvivalModel(sim).flows(spec.lines, spec.trains)
    demand = sum(sum(target_probs(sim, i).values()) for i in range(len(sim.stations)))
    assert math.isclose(sum(f["board"].values()) + f["unreachable"].sum(), demand + f["transfers"])
    assert all(r > 0 for r in f["ride"].values())


@given(map_seeds, seeds)
def test_estimates_are_deterministic_and_in_range(map_seed, seed):
    sim = random_legal_design(map_seed, seed)
    spec, model = spec_of(sim), SurvivalModel(sim)
    a = model.analyze(spec.lines, spec.trains)
    assert 1 <= a["ticks"] <= CAP_TICKS
    assert a == model.analyze(spec.lines, spec.trains)
    assert math.isclose(model.log_ticks(spec), math.log(a["ticks"]))


def test_train_loads_match_a_played_game():
    """Riders leaving each station on each line and direction, per unit spawn rate: model vs
    one game's ticks 400-1100 (a heuristic design that survives them)."""
    design = heuristic_design(0).sim
    spec = spec_of(design)
    ride = SurvivalModel(design).flows(spec.lines, spec.trains)["ride"]
    sim = design
    sim.set_passenger_seed(1)
    index = {id(s): i for i, s in enumerate(sim.stations)}
    seen = defaultdict(float)
    while sim.time < 1100:
        stopping = [(li, t) for li, line in enumerate(sim.lines) for t in line.trains if t._at_station]
        sim.step()
        assert not sim.failed
        if sim.time > 400:
            for li, t in stopping:
                seen[(index[id(t.line.stations[t.position])], li, t.direction)] += len(t.passengers)
    spawned = SPAWN_RATE[400:1100].sum()
    keys = sorted(set(ride) | set(seen))
    model = np.array([ride.get(k, 0.0) for k in keys])
    game = np.array([seen.get(k, 0.0) / spawned for k in keys])
    assert np.corrcoef(model, game)[0, 1] > 0.85
    assert abs(model.sum() / game.sum() - 1) < 0.15
