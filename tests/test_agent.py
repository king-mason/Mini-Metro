"""agent.py: the design loop, forced connections (rules.py), the viewing hook, weights."""

import copy

import pytest
from hypothesis import given, strategies as st

from agent import Agent
from bench import mirrored
from rules import connect_isolated, is_legal
from tests.helpers import crossing_pairs, fresh_map, heuristic_agent, heuristic_design

pytestmark = pytest.mark.slow
map_indices = st.integers(min_value=0, max_value=99)


@pytest.fixture(scope="module")
def untrained():
    return Agent(seed=0)


@given(st.integers(min_value=0, max_value=20))
def test_untrained_agent_plays_the_heuristic(untrained, map_index):
    """Q's last layer starts at zero, so a fresh agent makes the heuristic's choices."""
    m = 10_000 + map_index
    assert untrained.design(fresh_map(m)).actions == heuristic_design(m).actions


@given(map_indices, st.integers(min_value=0, max_value=30))
def test_replaying_a_plan_reproduces_the_design(map_index, cut):
    agent = heuristic_agent()
    ref = heuristic_design(10_000 + map_index)
    again = agent.design(fresh_map(10_000 + map_index), plan=ref.actions[:cut])
    assert again.actions == ref.actions


@given(map_indices)
def test_heuristic_is_mirror_invariant(map_index):
    agent = heuristic_agent()
    sim = fresh_map(10_000 + map_index)
    ref = heuristic_design(10_000 + map_index).actions
    assert agent.design(mirrored(sim, True, False)).actions == ref
    assert agent.design(mirrored(sim, False, True)).actions == ref


@given(map_indices, st.integers(min_value=0, max_value=6))
def test_forced_connections_serve_every_station_legally(map_index, n_lines):
    """Stop line building early; the forced connections must still finish legally."""
    sim = copy.deepcopy(heuristic_design(10_000 + map_index).sim)
    sim.lines = sim.lines[:n_lines % (len(sim.lines) + 1)]
    for i, line in enumerate(sim.lines):
        line.index = i
    shown = []
    forced, stranded = connect_isolated(sim, shown.append)
    assert len(shown) == forced
    assert crossing_pairs(sim) == []
    served = sum(any(s in l.stations for l in sim.lines) for s in sim.stations)
    assert served == len(sim.stations) - stranded


def test_on_step_sees_every_action_before_it_is_applied():
    agent = heuristic_agent()
    seen = []

    def on_step(sim, action, label, scored):
        assert is_legal(sim, action), label
        if scored is not None:
            assert action in [a for a, _, _ in scored]
            totals = [p + q for _, p, q in scored]
            assert totals == sorted(totals, reverse=True)
        seen.append(action)

    result = agent.design(fresh_map(10_004), on_step=on_step)
    assert len(seen) == len(result.actions) + result.n_forced


def test_on_step_returning_false_stops_the_calls():
    agent = heuristic_agent()
    calls = []
    agent.design(fresh_map(10_005), on_step=lambda *a: calls.append(1) or False)
    assert calls == [1]


def test_checkpoint_round_trip(tmp_path, untrained):
    untrained.save(str(tmp_path / "ckpt"))
    other = Agent()
    assert other.load(str(tmp_path / "ckpt"))
    sim = fresh_map(10_006)
    assert other.design(sim).actions == untrained.design(sim).actions
    assert not other.load(str(tmp_path / "missing"))
