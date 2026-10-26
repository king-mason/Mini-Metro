"""visualizer.py: off-screen frames and the controls (SDL dummy driver, see conftest)."""

import copy

import pygame
import pytest

from simulator import config
from tests.helpers import fresh_map, heuristic_agent, heuristic_design
from visualizer import SPEEDS, NextMap, Visualizer


def press(key):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode="", scancode=0))


@pytest.fixture
def vis():
    v = Visualizer(*config["resolution"], mode="dev")
    v.new_map(map=10_000, policy="test")
    yield v
    v.close()


def drawn(vis) -> bool:
    """Something other than the background was drawn."""
    return len({vis.surface.get_at((x, y))[:3] for x in range(0, vis.width, 37) for y in range(0, vis.height, 37)}) > 3


def test_design_steps_draw_and_advance(vis):
    steps = []

    def on_step(*args):
        press(pygame.K_RIGHT)
        steps.append(vis.show_design_step(*args))
        assert drawn(vis)
        return steps[-1]

    result = heuristic_agent().design(fresh_map(10_000), on_step=on_step)
    assert all(steps) and len(steps) == len(result.actions) + result.n_forced


def test_enter_skips_the_rest_of_the_design(vis):
    calls = []
    press(pygame.K_RETURN)
    heuristic_agent().design(fresh_map(10_001), on_step=lambda *a: calls.append(1) or vis.show_design_step(*a))
    assert calls == [1]


def test_n_asks_for_the_next_map(vis):
    press(pygame.K_n)
    with pytest.raises(NextMap):
        heuristic_agent().design(fresh_map(10_002), on_step=vis.show_design_step)


def test_play_runs_to_game_over_and_reports(vis):
    sim = copy.deepcopy(heuristic_design(10_003).sim)
    sim.set_passenger_seed(1)
    vis.speed = len(SPEEDS) - 1
    vis.auto_next = 0.0
    assert vis.play(sim, max_ticks=1500) == "next"
    assert sim.time == 1500 or sim.failed
    assert drawn(vis)


def test_play_keys(vis):
    sim = copy.deepcopy(heuristic_design(10_003).sim)
    press(pygame.K_LEFT)
    press(pygame.K_r)
    start = vis.speed
    assert vis.play(sim) == "replay"
    assert vis.speed == start - 1


def test_escape_quits(vis):
    press(pygame.K_ESCAPE)
    with pytest.raises(SystemExit):
        vis.play(copy.deepcopy(heuristic_design(10_003).sim))


def test_screenshot(vis, tmp_path, monkeypatch):
    monkeypatch.setitem(config["paths"], "screenshots", str(tmp_path))
    vis.show_search(0, 3, heuristic_design(10_004).sim, None)
    path = vis.screenshot()
    assert path.startswith(str(tmp_path)) and pygame.image.load(path).get_size() == (vis.width, vis.height)
