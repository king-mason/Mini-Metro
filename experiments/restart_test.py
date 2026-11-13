"""Does the start matter? Polish each map from several starting designs and play the results.

    python experiments/restart_test.py            # resumable; writes experiments/restart_test.jsonl
    python experiments/restart_test.py --report   # summary only

Arms per map (all polished until 15 rounds pass without a gain, unless noted):
    ref30   heuristic start, polish 30 rounds (the current teacher)
    top     heuristic start, climbed to the top
    dice    heuristic start, different random edits (control for luck)
    kick    heuristic start + 6 random edits
    noisy   heuristic built with noise on its move scores
"""

import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

from minimetro import bench
from minimetro.design.agent import Agent
from minimetro.design.metrics import neighbor_sets
from minimetro.design.search import build, polish, propose, spec_of
from minimetro.design.survival import SurvivalModel
from minimetro.game.maps import LEVELS
from minimetro.game.simulator import run_playout

OUT = os.path.join("experiments", "restart_test.jsonl")
SEEDS = list(range(5))
CAP = bench.CAPS["val"]
REPEATS = 3
KICK_EDITS = 6
NOISE = 1.0          # std of Gumbel noise on move scores, in prior units
PATIENCE, MAX_ROUNDS = 15, 300


def maps(per_level: int) -> list[bench.BenchMap]:
    # val maps (30/35 stations), plus unused 40-station maps (test uses 10000+, these 30000+)
    return bench.suite("val", per_level) + [bench.BenchMap("40", 30_000 + i) for i in range(per_level)]


def segments(spec) -> set:
    return {(min(a, b), max(a, b)) for l in spec.lines for a, b in zip(l, l[1:])}


def overlap(a, b) -> float:
    sa, sb = segments(a), segments(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def climb(sim0, start_sim, value, rng, rounds=None):
    """Polish; rounds=None means until PATIENCE rounds pass without a gain."""
    if rounds is not None:
        return polish(sim0, start_sim, value, rounds=rounds, rng=rng)
    last = {"final": None, "since": 0}

    def stop(r, n, sim, stats):
        if stats["final"] == last["final"]:
            last["since"] += 1
        else:
            last.update(final=stats["final"], since=0)
        return last["since"] < PATIENCE

    return polish(sim0, start_sim, value, rounds=MAX_ROUNDS, rng=rng, on_round=stop)


def kicked(sim0, design, rng):
    spec, nbrs = spec_of(design), neighbor_sets(sim0)
    for _ in range(KICK_EDITS):
        for _ in range(50):
            p = propose(spec, sim0, nbrs, rng)
            if p is not None:
                spec = p[0]
                break
    return build(sim0, spec)


def noisy_design(agent, sim0, rng):
    def guide(sim, phase, candidates):
        scores = agent.scores(sim, phase, candidates)
        return max(candidates, key=lambda a: scores[a] - NOISE * math.log(-math.log(rng.random() + 1e-12)))
    return agent.design(sim0, guide=guide).sim


def run_map(bm: bench.BenchMap) -> dict:
    sim0 = bm.sim()
    agent = Agent(use_q=False)
    value = SurvivalModel(sim0).log_ticks
    heur = agent.design(sim0).sim
    heur_spec = spec_of(heur)
    arms = []

    def add(arm, start, rng, rounds=None):
        design, stats = climb(sim0, start, value, rng, rounds)
        spec = spec_of(design)
        ticks = [run_playout(design, s, CAP)[0] for s in SEEDS]
        arms.append({"arm": arm, "rounds": stats["rounds"], "est_start": stats["start"], "est": stats["final"],
                     "ticks": ticks, "start_overlap": overlap(spec_of(start), heur_spec),
                     "end_overlap": overlap(spec, heur_spec), "lines": spec.lines, "trains": spec.trains})

    add("ref30", heur, random.Random(bm.seed), rounds=30)
    add("top", heur, random.Random(bm.seed))
    for k in range(REPEATS):
        add("dice", heur, random.Random(10_000 * (k + 1) + bm.seed))
        add("kick", kicked(sim0, heur, random.Random(20_000 * (k + 1) + bm.seed)), random.Random(bm.seed + k))
        add("noisy", noisy_design(agent, sim0, random.Random(30_000 * (k + 1) + bm.seed)), random.Random(bm.seed + k))
    heur_ticks = [run_playout(heur, s, CAP)[0] for s in SEEDS]
    return {"level": bm.level, "map": bm.seed, "stations": len(sim0.stations), "heuristic": heur_ticks, "arms": arms}


def logmean(ticks) -> float:
    return float(np.mean(np.log(ticks)))


def report(rows: list[dict]):
    """Everything in log ticks relative to ref30 on the same map (+0.10 ~ 10% longer survival)."""
    print(f"{len(rows)} maps\n")
    groups = {"all": rows}
    for st in sorted({r["stations"] for r in rows}):
        groups[f"{st} stations"] = [r for r in rows if r["stations"] == st]
    for name, rs in groups.items():
        print(f"== {name} ({len(rs)} maps) ==")
        rel = {arm: [] for arm in ("top", "dice", "kick", "noisy")}
        spread, best_any, pick_est, ovl = [], [], [], {a: [] for a in rel}
        for r in rs:
            ref = next(a for a in r["arms"] if a["arm"] == "ref30")
            base = logmean(ref["ticks"])
            climbs = [a for a in r["arms"] if a["arm"] != "ref30"]
            for a in climbs:
                rel[a["arm"]].append(logmean(a["ticks"]) - base)
                ovl[a["arm"]].append(a["end_overlap"])
            real = [logmean(a["ticks"]) for a in climbs]
            spread.append(max(real) - min(real))
            best_any.append(max(real) - base)
            pick_est.append(real[int(np.argmax([a["est"] for a in climbs]))] - base)
        heur_gap = np.mean([logmean(next(a for a in r["arms"] if a["arm"] == "ref30")["ticks"]) - logmean(r["heuristic"])
                            for r in rs])
        print(f"  ref30 vs heuristic: x{math.exp(heur_gap):.3f}")
        for arm, v in rel.items():
            print(f"  {arm:6s} vs ref30: mean x{math.exp(np.mean(v)):.3f} "
                  f"(network overlap with heuristic {np.mean(ovl[arm]):.0%})")
        print(f"  spread between climbs on a map (best/worst): x{math.exp(np.mean(spread)):.3f}")
        print(f"  best climb per map vs ref30 (needs play-outs to pick): x{math.exp(np.mean(best_any)):.3f}")
        print(f"  climb the estimate rates best vs ref30 (no play-outs): x{math.exp(np.mean(pick_est)):.3f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-level", type=int, default=10)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    done = [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
    if not args.report:
        todo = [m for m in maps(args.per_level) if (m.level, m.seed) not in {(r["level"], r["map"]) for r in done}]
        print(f"{len(todo)} maps to run, {len(done)} done", flush=True)
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(run_map, todo):
                with open(OUT, "a") as f:
                    f.write(json.dumps(row) + "\n")
                done.append(row)
                print(f"done {row['level']} {row['map']} ({len(done)})", flush=True)
    report(done)


if __name__ == "__main__":
    main()
