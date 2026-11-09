"""Collect search logs: every edit the design search tries, with its measured result.

    python collect_edits.py --maps 0:500                 # appends to data/search_logs.jsonl
    python collect_edits.py --maps 0:500 --out other.jsonl

One JSON line per map: the heuristic design, every search round (see search.improve's
`record`), the final design, and both designs played on 8 more seeds ("eval": heuristic
row, then final). Maps already in the file are skipped, so an interrupted run resumes.
Training maps come from 0..9999 (bench.py's maps are 10000+). Seeds are drawn >= 1000, so
never the bench's or validation's.
"""

import argparse
import json
import os
import random
import time

import bench
from agent import Agent
from search import spec_of
from simulator import MiniMetroSim

EVAL_SEEDS = 8


def collect_map(agent: Agent, map_seed: int, rounds: int) -> dict:
    start = time.time()
    sim0 = MiniMetroSim(map_seed=map_seed)
    heuristic = agent.design(sim0).sim
    seed_rng = random.Random(10_000_000 + map_seed)
    log: list = []
    final, stats = agent.improve(sim0, heuristic, rounds=rounds, seed_rng=seed_rng,
                                 rng=random.Random(map_seed), record=log)
    eval_seeds = [seed_rng.randrange(1_000, 2 ** 31) for _ in range(EVAL_SEEDS)]
    return {
        "map": map_seed,
        "heuristic": [spec_of(heuristic).lines, spec_of(heuristic).trains],
        "final": [spec_of(final).lines, spec_of(final).trains],
        "rounds": log,
        "stats": stats,
        "eval": agent.playouts.ticks([heuristic, final], eval_seeds).tolist(),
        "secs": time.time() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", default="0:500", help="map seed range FIRST:LAST (last excluded)")
    parser.add_argument("--out", default=os.path.join("data", "search_logs.jsonl"))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=bench.default_workers())
    args = parser.parse_args()

    first, last = map(int, args.maps.split(":"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["map"] for line in f if line.strip()}
    todo = [m for m in range(first, last) if m not in done]
    print(f"{len(done)} maps already in {args.out}; collecting {len(todo)}", flush=True)

    agent = Agent(use_q=False, workers=args.workers, screen_edits=False)   # random edits: unbiased data
    try:
        with open(args.out, "a") as f:
            for i, m in enumerate(todo, 1):
                row = collect_map(agent, m, args.rounds)
                f.write(json.dumps(row) + "\n")
                f.flush()
                ev = row["eval"]
                gain = sum(ev[1]) / max(sum(ev[0]), 1)
                print(f"[{i}/{len(todo)}] map {m}: {row['secs']:.1f}s, kept {row['stats']['accepted']},"
                      f" eval x{gain:.2f}", flush=True)
    finally:
        agent.close()


if __name__ == "__main__":
    main()
