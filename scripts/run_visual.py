"""Watch the agent design networks and play them out.

    python scripts/run_visual.py                         # showcase: default checkpoint, else the heuristic
    python scripts/run_visual.py --dev                   # step through decisions and their scores
    python scripts/run_visual.py --map 10003 --search 0  # one map, no play-out search
    python scripts/run_visual.py --heuristic --dev
    python scripts/run_visual.py --level 30              # a bench difficulty level (maps.LEVELS)

Press H in the window for the keys (also listed in viewer.py). Maps given with --map
play in order from that seed; otherwise they are random. The design search draws its seeds
from bench.search_seed_rng(map), so a map always gets the same design.
"""

import argparse
import random

from minimetro import bench
from minimetro.config import DEFAULT_CHECKPOINT
from minimetro.design.agent import Agent
from minimetro.game.maps import LEVELS
from minimetro.showcase import watch
from minimetro.viewer import Visualizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev", action="store_true", help="development view (manual design steps, scores)")
    parser.add_argument("--map", type=int, help="first map seed (default: random maps)")
    parser.add_argument("--maps", type=int, default=500, help="how many maps to show")
    parser.add_argument("--level", choices=list(LEVELS), default="classic", help="kind of map (maps.LEVELS)")
    parser.add_argument("--search", type=int, default=10, metavar="ROUNDS",
                        help="rounds of play-out search to polish each design (0 = off)")
    parser.add_argument("--ckpt", default=DEFAULT_CHECKPOINT, help="checkpoint to load")
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
        watch(agent, vis, maps, args.level, args.search, policy)
    except SystemExit:
        pass
    finally:
        agent.close()
        vis.close()


if __name__ == "__main__":
    main()
