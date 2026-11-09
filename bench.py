"""Fixed benchmark: how long do a policy's designs survive, compared with the heuristic?

    python bench.py heuristic                          # the structural prior alone
    python bench.py agent --ckpt checkpoint_v4_best    # a trained checkpoint
    python bench.py agent --ckpt checkpoint_v4_best --mcts 30
    python bench.py heuristic --search 10 --maps 20    # then polished by play-out search
    python bench.py heuristic --polish 30              # then polished by the survival estimate

Every run plays the same held-out maps and passenger seeds. Results go to
bench_results/<name>.json and are compared, map by map, with bench_results/heuristic.json.
"""

import argparse
import copy
import json
import math
import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from config import HEIGHT, MAX_TRAINS, WIDTH
from playouts import CAP
from point import Point
from simulator import MiniMetroSim, run_playout

BENCH_MAPS = list(range(10_000, 10_100))  # held out: training uses maps 0..9999
BENCH_SEEDS = list(range(5))              # passenger seeds, shared by every run
RESULTS_DIR = "bench_results"


@dataclass
class PolicySpec:
    kind: str = "heuristic"     # "heuristic" (prior only) or "agent" (prior + learned Q)
    ckpt: str | None = None
    mcts_sims: int = 0
    polish_rounds: int = 0      # then hill-climb on the survival estimate
    search_rounds: int = 0      # then play-out search
    screen_edits: bool = True   # search plays the edits the survival estimate rates best


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def make_agent(spec: PolicySpec):
    from agent import Agent
    agent = Agent(use_q=spec.kind == "agent", mcts_sims=spec.mcts_sims, screen_edits=spec.screen_edits)
    if spec.kind == "agent" and not agent.load(spec.ckpt):
        raise FileNotFoundError(f"no compatible checkpoint at {spec.ckpt}/")
    return agent


def search_seed_rng(map_seed: int) -> random.Random:
    """Seeds a search uses on this map: reproducible, and >= 1000 (search.improve), so never
    the bench's or validation's."""
    return random.Random(1_000_000 + map_seed)


def evaluate_map(agent, spec: PolicySpec, map_seed: int, seeds: list[int], cap: int) -> dict:
    """Design one map (then polish / search, as `spec` says) and play it once per seed."""
    sim0 = MiniMetroSim(map_seed=map_seed)
    result = agent.design(sim0)
    design, polish, search = result.sim, None, None
    if spec.polish_rounds:
        design, polish = agent.polish(sim0, design, rounds=spec.polish_rounds, rng=random.Random(map_seed))
    if spec.search_rounds:
        design, search = agent.improve(sim0, design, rounds=spec.search_rounds, seed_rng=search_seed_rng(map_seed),
                                       rng=random.Random(map_seed))
    runs = [run_playout(design, seed, cap) for seed in seeds]
    return {
        "map": map_seed,
        "ticks": [t for t, _ in runs],
        "delivered": [d for _, d in runs],
        "trains": sum(len(l.trains) for l in design.lines),
        "forced": result.n_forced,
        "stranded": result.n_stranded,
        "search": search,   # the search's own measurements (start / final ticks, edits kept)
        "polish": polish,   # the survival estimate's prediction (start / final ticks, edits kept)
    }


_worker_agent = None


def _init_worker(spec: PolicySpec):
    global _worker_agent
    import torch
    torch.set_num_threads(1)  # one process per core
    _worker_agent = make_agent(spec)


def _eval_map_in_worker(spec: PolicySpec, map_seed: int, seeds: list[int], cap: int) -> dict:
    return evaluate_map(_worker_agent, spec, map_seed, seeds, cap)


def _pool(spec: PolicySpec, workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_init_worker, initargs=(spec,))


def mirrored(sim: MiniMetroSim, flip_x: bool = True, flip_y: bool = False) -> MiniMetroSim:
    """An undesigned map reflected across its center lines, station order kept: a policy
    that learned concepts rather than positions designs it the same way."""
    m = copy.deepcopy(sim)
    for s in m.stations:
        s.pos = Point(WIDTH - s.x if flip_x else s.x, HEIGHT - s.y if flip_y else s.y)
    return m


def _mirror_check_in_worker(map_seed: int) -> tuple[bool, bool]:
    sim = MiniMetroSim(map_seed=map_seed)
    base = _worker_agent.design(sim).actions
    return (_worker_agent.design(mirrored(sim, True, False)).actions == base,
            _worker_agent.design(mirrored(sim, False, True)).actions == base)


def invariance(spec: PolicySpec, maps=BENCH_MAPS, workers: int | None = None) -> dict:
    """Share of maps whose design is unchanged by left-right / top-bottom mirroring."""
    with _pool(spec, max(workers or default_workers(), 1)) as pool:
        checks = list(pool.map(_mirror_check_in_worker, list(maps)))
    return {
        "left_right": float(np.mean([lr for lr, _ in checks])),
        "top_bottom": float(np.mean([tb for _, tb in checks])),
        "both": float(np.mean([lr and tb for lr, tb in checks])),
    }


def run(spec: PolicySpec, maps=BENCH_MAPS, seeds=BENCH_SEEDS, cap: int = CAP, workers: int | None = None) -> dict:
    """Evaluate `spec` on every map. workers=0 runs in this process (for debugging)."""
    maps, seeds = list(maps), list(seeds)
    workers = default_workers() if workers is None else workers
    if workers == 0:
        agent = make_agent(spec)
        rows = [evaluate_map(agent, spec, m, seeds, cap) for m in maps]
    else:
        with _pool(spec, workers) as pool:
            rows = list(pool.map(_eval_map_in_worker, [spec] * len(maps), maps, [seeds] * len(maps),
                                 [cap] * len(maps)))
    return {"spec": asdict(spec), "cap": cap, "seeds": seeds, "maps": rows}


def map_scores(results: dict) -> dict[int, float]:
    """Per-map score: mean ticks survived over the passenger seeds (each capped)."""
    return {row["map"]: float(np.mean(row["ticks"])) for row in results["maps"]}


def summarize(results: dict) -> dict:
    rows = results["maps"]
    scores = np.array(list(map_scores(results).values()))
    return {
        "maps": len(rows),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "p10": float(np.percentile(scores, 10)),
        "delivered": float(np.mean([np.mean(r["delivered"]) for r in rows])),
        "trains": float(np.mean([r["trains"] for r in rows])),
        "all_trains": float(np.mean([r["trains"] == MAX_TRAINS for r in rows])),
        "forced": float(np.mean([r["forced"] for r in rows])),
        "stranded": float(np.mean([r.get("stranded", 0) > 0 for r in rows])),
        "reached_cap": float(np.mean([min(r["ticks"]) >= results["cap"] for r in rows])),
    }


def compare(results: dict, reference: dict) -> dict:
    """Geometric-mean survival ratio vs `reference`, paired by map, with a 95% CI. A map is
    a win / loss when its ratio is more than 1% from 1. The runs must share seeds and cap."""
    if results["seeds"] != reference["seeds"] or results["cap"] != reference["cap"]:
        raise ValueError("results and reference used different passenger seeds or caps")
    a, b = map_scores(results), map_scores(reference)
    common = sorted(set(a) & set(b))
    d = np.log([a[m] / b[m] for m in common])
    half = 1.96 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {
        "n": len(common),
        "ratio": float(math.exp(d.mean())),
        "lo": float(math.exp(d.mean() - half)),
        "hi": float(math.exp(d.mean() + half)),
        "wins": float(np.mean(d > 0.01)),
        "losses": float(np.mean(d < -0.01)),
    }


def save(results: dict, name: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f)
    return path


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def report(name: str, results: dict, reference: dict | None = None, reference_name: str = "") -> str:
    s = summarize(results)
    lines = [
        f"{name}: {s['maps']} maps x {len(results['seeds'])} passenger seeds, cap {results['cap']} ticks",
        f"  ticks survived  mean {s['mean']:6.0f}   median {s['median']:6.0f}   p10 {s['p10']:6.0f}",
        f"  delivered {s['delivered']:6.0f}   trains {s['trains']:.2f} (all {MAX_TRAINS} on {s['all_trains']:.0%} of maps)"
        f"   forced connections {s['forced']:.2f}   reached cap on {s['reached_cap']:.0%}",
        f"  a station left unconnected (no legal move) on {s['stranded']:.0%} of maps",
    ]
    if reference is not None:
        c = compare(results, reference)
        lines.append(
            f"  vs {reference_name}: x{c['ratio']:.3f} geometric mean [95% CI {c['lo']:.3f}, {c['hi']:.3f}]"
            f"   better on {c['wins']:.0%}, worse on {c['losses']:.0%} of {c['n']} maps")
    return "\n".join(lines)


def default_name(spec: PolicySpec) -> str:
    name = "heuristic" if spec.kind == "heuristic" else os.path.basename(os.path.normpath(spec.ckpt))
    if spec.mcts_sims:
        name += f"_mcts{spec.mcts_sims}"
    if spec.polish_rounds:
        name += f"_polish{spec.polish_rounds}"
    if spec.search_rounds:
        name += f"_{'' if spec.screen_edits else 'plain'}search{spec.search_rounds}"
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy", choices=["heuristic", "agent"])
    parser.add_argument("--ckpt", default="checkpoint_v4_best", help="checkpoint dir for policy=agent")
    parser.add_argument("--mcts", type=int, default=0, help="MCTS simulations per decision (0 = greedy)")
    parser.add_argument("--name", help="results name (default: derived from the policy)")
    parser.add_argument("--maps", type=int, default=len(BENCH_MAPS), help="use the first N bench maps")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--compare", default=os.path.join(RESULTS_DIR, "heuristic.json"),
                        help="results file to compare against, paired by map")
    parser.add_argument("--invariance", action="store_true",
                        help="also check whether mirrored maps get the same design")
    parser.add_argument("--polish", type=int, default=0, metavar="ROUNDS",
                        help="polish each design on the survival estimate (no play-outs)")
    parser.add_argument("--search", type=int, default=0, metavar="ROUNDS",
                        help="then polish it by play-out search (slow)")
    parser.add_argument("--plain-search", action="store_true",
                        help="search plays random edits, not the ones the survival estimate rates best")
    args = parser.parse_args()

    spec = PolicySpec(args.policy, args.ckpt if args.policy == "agent" else None, args.mcts,
                      polish_rounds=args.polish, search_rounds=args.search, screen_edits=not args.plain_search)
    name = args.name or default_name(spec)
    start = time.time()
    results = run(spec, maps=BENCH_MAPS[:args.maps], workers=args.workers)
    path = save(results, name)
    reference = None
    if os.path.exists(args.compare) and os.path.abspath(args.compare) != os.path.abspath(path):
        reference = load(args.compare)
    print(report(name, results, reference, os.path.splitext(os.path.basename(args.compare))[0]))
    if args.invariance:
        inv = invariance(spec, maps=BENCH_MAPS[:args.maps], workers=args.workers)
        print(f"  same design on mirrored map: left-right {inv['left_right']:.0%}, "
              f"top-bottom {inv['top_bottom']:.0%}, both {inv['both']:.0%}")
    print(f"  saved {path} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
