"""Watch the agent design networks and play them out.

    python run_visual.py                         # showcase: checkpoint_v4_best, else the heuristic
    python run_visual.py --dev                   # step through decisions and their scores
    python run_visual.py --map 10003 --search 0  # one map, no play-out search
    python run_visual.py --heuristic --dev

Press H in the window for the keys (also listed in visualizer.py). Maps given with --map
play in order from that seed; otherwise they are random. The design search draws its seeds
from bench.search_seed_rng(map), so a map always gets the same design.
"""

import argparse
import copy
import random

import bench
from agent import Agent
from simulator import MiniMetroSim
from visualizer import NextMap, Visualizer


def episode(agent: Agent, vis: Visualizer, map_seed: int, search_rounds: int, rng: random.Random):
    """Design one map on screen, polish it with search, and play it until the viewer moves on."""
    sim0 = MiniMetroSim(map_seed=map_seed)
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


def watch(agent: Agent, vis: Visualizer, maps, search_rounds: int = 0, policy: str = ""):
    """Show `maps` one after another (N skips ahead). Returns when they run out; ESC or
    closing the window raises SystemExit."""
    rng = random.Random()
    for map_seed in maps:
        vis.new_map(map=map_seed, policy=policy)
        try:
            episode(agent, vis, map_seed, search_rounds, rng)
        except NextMap:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev", action="store_true", help="development view (manual design steps, scores)")
    parser.add_argument("--map", type=int, help="first map seed (default: random maps)")
    parser.add_argument("--maps", type=int, default=500, help="how many maps to show")
    parser.add_argument("--search", type=int, default=10, metavar="ROUNDS",
                        help="rounds of play-out search to polish each design (0 = off)")
    parser.add_argument("--ckpt", default="checkpoint_v4_best", help="checkpoint to load")
    parser.add_argument("--heuristic", action="store_true", help="structural prior only (no learned Q)")
    args = parser.parse_args()

    agent = Agent(use_q=not args.heuristic,
                  workers=bench.default_workers() if args.search else 0)
    policy = "heuristic"
    if not args.heuristic:
        if agent.load(args.ckpt):
            policy = args.ckpt
        else:
            print(f"No checkpoint at {args.ckpt}/ - showing the structural heuristic.")
    if args.search:
        policy += f" + search {args.search}"

    rng = random.Random()
    maps = ([args.map + i for i in range(args.maps)] if args.map is not None
            else [rng.randrange(10_000) for _ in range(args.maps)])
    vis = Visualizer(mode="dev" if args.dev else "showcase")
    try:
        watch(agent, vis, maps, args.search, policy)
    except SystemExit:
        pass
    finally:
        agent.close()
        vis.close()


if __name__ == "__main__":
    main()
