"""The teacher's designs, for imitation (learning/imitation.py): the heuristic's design on each
training map, polished on the survival estimate, then optionally searched with play-outs.

    python scripts/teach.py --maps 1000                  # polish30 on 1000 maps per training level
    python scripts/teach.py --maps 100 --search 10       # polish, then screened search

One JSON line per map goes to data/teacher_<name>.jsonl: the map and the design's lines and
trains. Maps already in the file are skipped, so a run can be resumed.
"""

import argparse
import json
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor

from minimetro import bench
from minimetro.config import DATA_DIR
from minimetro.design.search import spec_of
from minimetro.learning.imitation import load_teacher

_agent = None


def _init(spec: bench.PolicySpec):
    global _agent
    import torch
    torch.set_num_threads(1)
    _agent = bench.make_agent(spec)


def teach(spec: bench.PolicySpec, bench_map: bench.BenchMap) -> dict:
    sim0 = bench_map.sim()
    design = _agent.design(sim0).sim
    polish = search = None
    if spec.polish_rounds:
        design, polish = _agent.polish(sim0, design, rounds=spec.polish_rounds, restarts=spec.polish_restarts,
                                       rng=random.Random(bench_map.seed))
    if spec.search_rounds:
        design, search = _agent.improve(sim0, design, rounds=spec.search_rounds,
                                        seed_rng=bench.search_seed_rng(bench_map.seed),
                                        rng=random.Random(bench_map.seed))
    s = spec_of(design)
    return {"level": bench_map.level, "map": bench_map.seed, "lines": s.lines, "trains": s.trains,
            "polish": polish, "search": search}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", type=int, default=1000, help="training maps per level")
    parser.add_argument("--polish", type=int, default=30, metavar="ROUNDS")
    parser.add_argument("--restarts", type=int, default=1, help="polish climbs per map, keeping the best-rated")
    parser.add_argument("--search", type=int, default=0, metavar="ROUNDS")
    parser.add_argument("--workers", type=int, default=bench.default_workers())
    parser.add_argument("--name", help="default: polish<R>[_search<R>]")
    args = parser.parse_args()

    spec = bench.PolicySpec(polish_rounds=args.polish, polish_restarts=args.restarts, search_rounds=args.search)
    name = args.name or bench.default_name(spec).removeprefix("heuristic_")
    path = os.path.join(DATA_DIR, f"teacher_{name}.jsonl")
    os.makedirs(DATA_DIR, exist_ok=True)
    done = {(r["level"], r["map"]) for r in load_teacher(path)} if os.path.exists(path) else set()
    maps = [m for m in bench.suite("train", args.maps) if (m.level, m.seed) not in done]
    print(f"{len(maps)} maps to teach ({len(done)} already in {path})")
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init, initargs=(spec,)) as pool, open(path, "a") as out:
        for k, row in enumerate(pool.map(teach, [spec] * len(maps), maps, chunksize=4), 1):
            out.write(json.dumps(row) + "\n")
            if k % 100 == 0:
                out.flush()
                print(f"  {k}/{len(maps)}", flush=True)


if __name__ == "__main__":
    main()
