"""imitation.py: the right moves lead to the teacher's design, whichever of them is taken."""

import random

from hypothesis import given, settings, strategies as st

from agent import Agent
from imitation import Imitator, right_moves, walk
from rules import ActionType, apply_action
from search import spec_of
from tests.helpers import fresh_map, heuristic_design


def canonical(spec):
    """A design up to line order and direction."""
    return sorted((tuple(l) if l[0] < l[-1] else tuple(l[::-1]), t) for l, t in zip(spec.lines, spec.trains))


@settings(max_examples=10, deadline=None)
@given(st.integers(0, 40), st.integers(0, 10 ** 6))
def test_any_order_of_right_moves_rebuilds_the_target(map_index, seed):
    map_seed = 10_000 + map_index
    target = spec_of(heuristic_design(map_seed).sim)
    rng = random.Random(seed)
    built = Agent(use_q=False).design(fresh_map(map_seed),
                                      guide=lambda sim, phase, c: rng.choice(right_moves(sim, target, phase) or [None]))
    assert canonical(spec_of(built.sim)) == canonical(target)


def test_no_right_moves_once_off_the_path():
    target = spec_of(heuristic_design(10_000).sim)
    sim = fresh_map(10_000)
    on_target = {(min(a, b), max(a, b)) for l in target.lines for a, b in zip(l, l[1:])}
    a, b = next((a, b) for a in range(len(sim.stations)) for b in range(a + 1, len(sim.stations))
                if (a, b) not in on_target and not any(a in l and b in l for l in target.lines))
    apply_action(sim, (ActionType.NEW_LINE, a, b))
    assert right_moves(sim, target, "lines") == []


def test_a_learning_step_runs_and_follows_the_teacher():
    target = spec_of(heuristic_design(10_001).sim)
    agent = Agent(use_q=True, k_candidates=25, override_margin=0.0, seed=0)
    stats = Imitator(agent, seed=0).learn_map(fresh_map(10_001), target)
    assert stats["steps"] > 0 and stats["lost"] == 0 and stats["loss"] == stats["loss"]   # not NaN
    steps, _ = walk(agent, fresh_map(10_001), target)
    assert all(step.right.any() for step in steps)
