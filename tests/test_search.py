"""search.py: design specs, edit proposals, and the hill-climb's acceptance rule.

The hill-climb is tested with fake scorers (no games), so its logic is checked exactly:
what it plays each round, when it accepts, and what it reports.
"""

import random

import numpy as np
from hypothesis import given, strategies as st

from bench import BENCH_SEEDS
from heuristic import NEIGHBOR_K_DEFAULT, neighbor_sets
from search import Spec, build, improve, propose, spec_of, valid
from simulator import run_playout
from tests.helpers import crossing_pairs, fresh_map, heuristic_design

map_indices = st.integers(min_value=0, max_value=99)
seeds = st.integers(min_value=0, max_value=10 ** 6)


def map_and_design(map_index):
    m = 10_000 + map_index
    return fresh_map(m), heuristic_design(m).sim


# ---- specs and edits ----

@given(map_indices)
def test_build_reproduces_a_design(map_index):
    base, design = map_and_design(map_index)
    rebuilt = build(base, spec_of(design))
    assert spec_of(rebuilt) == spec_of(design)
    assert run_playout(rebuilt, 3, 1500) == run_playout(design, 3, 1500)


@given(map_indices, seeds)
def test_proposals_are_valid_legal_edits(map_index, seed):
    base, design = map_and_design(map_index)
    spec, rng, nbrs = spec_of(design), random.Random(seed), neighbor_sets(base, NEIGHBOR_K_DEFAULT)
    for _ in range(30):
        p = propose(spec, base, nbrs, rng)
        if p is None:
            continue
        new, edit = p
        assert new != spec and valid(new, base)
        assert crossing_pairs(build(base, new)) == []


def test_valid_rejects_broken_specs():
    base, design = map_and_design(0)
    spec = spec_of(design)
    assert valid(spec, base)
    missing = spec.copy()
    missing.lines[0] = missing.lines[0][:-1] if len(missing.lines[0]) > 2 else missing.lines[0]
    if len({j for l in missing.lines for j in l}) < len(base.stations):
        assert not valid(missing, base)
    too_many = spec.copy()
    too_many.trains[0] += 5
    assert not valid(too_many, base)
    pos = [s.pos for s in base.stations]
    from rules import segments_cross
    crossing = next(((a, b, c, d) for a in range(20) for b in range(20) for c in range(20) for d in range(20)
                     if len({a, b, c, d}) == 4 and segments_cross(pos[a], pos[b], pos[c], pos[d])))
    a, b, c, d = crossing
    rest = [j for j in range(20) if j not in crossing]
    assert not valid(Spec([[a, b], [c, d] + rest], [1, 1]), base)


# ---- the hill-climb, with fake scorers ----

class FakeScorer:
    """Scores a design as value(spec) * exp(noise); records every call."""

    def __init__(self, value, noise=0.0, seed=0):
        self.value, self.noise, self.rng = value, noise, np.random.default_rng(seed)
        self.calls = []

    def __call__(self, sims, seeds):
        self.calls.append(([spec_of(s) for s in sims], list(seeds)))
        base = np.array([[self.value(spec_of(s))] * len(seeds) for s in sims], dtype=float)
        return base * np.exp(self.noise * self.rng.standard_normal(base.shape))


def trains(spec):
    return 1000.0 + 100 * sum(spec.trains) - 10 * sum(len(l) for l in spec.lines)


@given(map_indices, seeds)
def test_budget_and_fresh_seeds(map_index, seed):
    base, design = map_and_design(map_index)
    scorer = FakeScorer(trains, noise=0.05, seed=seed)
    rounds, batch = 5, 8
    _, stats = improve(base, design, scorer, rounds=rounds, batch=batch, rng=random.Random(seed),
                       seed_rng=random.Random(seed))
    assert sum(len(sims) for sims, _ in scorer.calls) <= rounds * batch + 1
    seen = [tuple(s) for _, s in scorer.calls]
    assert len(set(seen)) == len(seen), "every round draws new seeds"
    assert not {x for s in seen for x in s} & set(BENCH_SEEDS)
    assert stats["rounds"] == rounds


@given(map_indices, seeds)
def test_each_round_replays_the_current_design_first(map_index, seed):
    """Round r+1 starts from round r's current design or from one of its edits."""
    base, design = map_and_design(map_index)
    scorer = FakeScorer(trains, noise=0.05, seed=seed)
    result, _ = improve(base, design, scorer, rounds=4, rng=random.Random(seed))
    rounds = scorer.calls[:-1]                       # the last call re-measures the result
    assert rounds[0][0][0] == spec_of(design)
    for (sims, _), (next_sims, _) in zip(rounds, rounds[1:]):
        assert next_sims[0] in sims
    assert spec_of(result) in rounds[-1][0]
    assert scorer.calls[-1][0] == [spec_of(result)]
    assert valid(spec_of(result), base)


@given(map_indices, seeds)
def test_no_signal_means_no_change(map_index, seed):
    """When every design scores the same, nothing is accepted however lucky the noise."""
    base, design = map_and_design(map_index)
    result, stats = improve(base, design, FakeScorer(lambda spec: 2000.0), rounds=4, rng=random.Random(seed))
    assert stats["accepted"] == {} and spec_of(result) == spec_of(design)


def test_the_noise_gate_tightens_with_z():
    """Pure noise (every design equal): how often the luckiest edit is taken falls with z."""
    def null_accept_rate(z):
        taken = 0
        for mi in range(8):
            base, design = map_and_design(mi)
            _, stats = improve(base, design, FakeScorer(lambda spec: 2000.0, noise=0.08, seed=mi),
                               rounds=10, z=z, rng=random.Random(mi))
            taken += sum(stats["accepted"].values())
        return taken / 80

    rates = [null_accept_rate(z) for z in (0.0, 1.0, 3.0)]
    assert rates[0] > rates[1] > rates[2] and rates[2] < 0.1


@given(map_indices, seeds)
def test_clear_gains_are_found_and_reported_honestly(map_index, seed):
    base, design = map_and_design(map_index)
    noise = 0.02
    result, stats = improve(base, design, FakeScorer(trains, noise=noise, seed=seed), rounds=10,
                            rng=random.Random(seed))
    true_start, true_final = trains(spec_of(design)), trains(spec_of(result))
    assert true_final >= true_start * 0.97   # edits within the noise may be taken
    # "final" is a fresh measurement of the returned design: within noise of its true value
    assert abs(np.log(stats["final"] / true_final)) < 4 * noise


def test_on_round_can_stop_the_search():
    base, design = map_and_design(1)
    calls = []
    _, stats = improve(base, design, FakeScorer(trains), rounds=10,
                       on_round=lambda *a: calls.append(a[0]) or False)
    assert calls == [1] and stats["rounds"] == 1
