"""Showing a policy at work: design each map on screen, optionally search, then play it."""

import copy
import random

from minimetro import bench
from minimetro.design.agent import Agent
from minimetro.game.maps import LEVELS
from minimetro.game.simulator import MiniMetroSim
from minimetro.viewer import NextMap, Visualizer


def episode(agent: Agent, vis: Visualizer, map_seed: int, level: str, search_rounds: int, rng: random.Random):
    """Design one map on screen, polish it with search, and play it until the viewer moves on."""
    sim0 = MiniMetroSim(map_seed, map_spec=LEVELS[level].spec(map_seed))
    design = agent.design(sim0, on_step=vis.show_design_step).sim
    if search_rounds > 0:
        vis.show_search(0, search_rounds, design, None)
        design, stats = agent.improve(sim0, design, rounds=search_rounds, seed_rng=bench.search_seed_rng(map_seed),
                                      rng=random.Random(map_seed), on_round=vis.show_search)
        print(f"  search: {stats['start']:.0f} -> {stats['final']:.0f} ticks (fresh seeds), kept {stats['accepted']}")
    while True:
        passenger_seed = rng.randrange(2 ** 31)
        vis.set_info(passengers=passenger_seed)
        sim = copy.deepcopy(design)
        sim.set_passenger_seed(passenger_seed)
        outcome = vis.play(sim)
        print(f"map {map_seed} | passengers {passenger_seed} | ticks {sim.time:6d} | delivered {sim.delivered:6d}")
        if outcome != "replay":
            return


def watch(agent: Agent, vis: Visualizer, maps, level: str = "classic", search_rounds: int = 0, policy: str = ""):
    """Show `maps` one after another (N skips ahead). Returns when they run out; ESC or
    closing the window raises SystemExit."""
    rng = random.Random()
    for map_seed in maps:
        spec = LEVELS[level].spec(map_seed)
        vis.new_map(map=map_seed, stations=f"{spec.stations} {spec.layout}", policy=policy)
        try:
            episode(agent, vis, map_seed, level, search_rounds, rng)
        except NextMap:
            pass
