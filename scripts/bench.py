"""Benchmark a designer: how long do its designs survive, compared with the heuristic?

    python scripts/bench.py heuristic                          # the structural prior alone
    python scripts/bench.py agent                              # the default checkpoint
    python scripts/bench.py agent --ckpt checkpoints/v4_best --mcts 30
    python scripts/bench.py heuristic --polish 30              # then polished by the survival estimate
    python scripts/bench.py heuristic --search 10 --maps 20    # then polished by play-out search
    python scripts/bench.py heuristic --polish 30 --search 10 --maps 40 --name teacher
    python scripts/bench.py heuristic --suite classic          # the original 20-station maps

Results go to results/[suite/]<name>.json and are compared, map by map, with that folder's
heuristic.json, and on the test suite with teacher.json as share of the teacher's gain.
The default name ends in _m<N> when --maps is given, so a partial run never replaces a full one.
"""

import argparse
import os
import time

from minimetro.bench import (CAPS, PolicySpec, default_name, default_workers, invariance, load, report,
                             results_dir, run, save, suite)
from minimetro.config import DEFAULT_CHECKPOINT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy", choices=["heuristic", "agent"])
    parser.add_argument("--ckpt", default=DEFAULT_CHECKPOINT, help="checkpoint dir for policy=agent")
    parser.add_argument("--mcts", type=int, default=0, help="MCTS simulations per decision (0 = greedy)")
    parser.add_argument("--name", help="results name (default: derived from the policy)")
    parser.add_argument("--suite", choices=["test", "val", "classic"], default="test")
    parser.add_argument("--maps", type=int, help="use the first N maps of each level")
    parser.add_argument("--teacher", help="results to measure share of gain against "
                                          "(default: the suite's teacher.json)")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--compare", help="results file to compare against, paired by map "
                                          "(default: the suite's heuristic.json)")
    parser.add_argument("--invariance", action="store_true",
                        help="also check whether mirrored maps get the same design")
    parser.add_argument("--polish", type=int, default=0, metavar="ROUNDS",
                        help="polish each design on the survival estimate (no play-outs)")
    parser.add_argument("--restarts", type=int, default=1, help="polish climbs per map, keeping the best-rated")
    parser.add_argument("--search", type=int, default=0, metavar="ROUNDS",
                        help="then polish it by play-out search (slow)")
    parser.add_argument("--plain-search", action="store_true",
                        help="search plays random edits, not the ones the survival estimate rates best")
    args = parser.parse_args()

    spec = PolicySpec(args.policy, args.ckpt if args.policy == "agent" else None, args.mcts,
                      polish_rounds=args.polish, polish_restarts=args.restarts,
                      search_rounds=args.search, screen_edits=not args.plain_search)
    name = args.name or default_name(spec, args.maps)
    maps = suite(args.suite, args.maps)
    directory = results_dir(args.suite)
    compare_path = args.compare or os.path.join(directory, "heuristic.json")
    teacher_path = args.teacher or os.path.join(directory, "teacher.json")
    start = time.time()
    results = run(spec, maps=maps, cap=CAPS[args.suite], workers=args.workers)
    path = save(results, name, directory)
    reference = teacher = None
    if os.path.exists(compare_path) and os.path.abspath(compare_path) != os.path.abspath(path):
        reference = load(compare_path)
    if os.path.exists(teacher_path) and os.path.abspath(teacher_path) != os.path.abspath(path):
        teacher = load(teacher_path)
    print(report(name, results, reference, os.path.splitext(os.path.basename(compare_path))[0], teacher))
    if args.invariance:
        inv = invariance(spec, maps=maps, workers=args.workers)
        print(f"  same design on mirrored map: left-right {inv['left_right']:.0%}, "
              f"top-bottom {inv['top_bottom']:.0%}, both {inv['both']:.0%}")
    print(f"  saved {path} ({time.time() - start:.0f}s)")



if __name__ == "__main__":
    main()
