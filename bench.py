"""Fixed benchmark: how long do a policy's designs survive, compared with the heuristic?

    python bench.py heuristic                          # the structural prior alone
    python bench.py agent --ckpt checkpoint_v4_best    # a trained checkpoint
    python bench.py agent --ckpt checkpoint_v4_best --mcts 30
    python bench.py heuristic --search 10 --maps 20    # then polished by play-out search
    python bench.py heuristic --polish 30              # then polished by the survival estimate
    python bench.py heuristic --suite classic          # the original 20-station maps

Every run plays the same held-out maps and passenger seeds: --maps N per difficulty level
(maps.LEVELS). Results go to bench_results/<name>.json and are compared, map by map, with
bench_results/heuristic.json (classic suite: bench_results/classic/).
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
from maps import LEVELS, MapSpec
from playouts import CAP
from point import Point
from simulator import MiniMetroSim, run_playout

BENCH_LEVELS = ("30", "35", "40")         # keys of maps.LEVELS, easiest first
LEVEL_MAPS = 100                          # maps per level
CLASSIC_MAPS = list(range(10_000, 10_100))
BENCH_SEEDS = list(range(5))              # passenger seeds, shared by every run
RESULTS_DIR = "bench_results"


@dataclass(frozen=True)
class BenchMap:
    level: str      # a key of maps.LEVELS
    seed: int

    @property
    def spec(self) -> MapSpec:
        return LEVELS[self.level].spec(self.seed)

    def sim(self):
        return MiniMetroSim(self.seed, map_spec=self.spec)


def suite(name: str = "levels", per_level: int | None = None) -> list[BenchMap]:
    """The bench maps, held out (training uses seeds 0..9999). "levels": the first
    `per_level` maps of each level, each level its own block of seeds. "classic": the
    original 20-station maps."""
    if name == "classic":
        return [BenchMap("classic", m) for m in CLASSIC_MAPS[:per_level]]
    return [BenchMap(level, 10_000 + 1_000 * j + i)
            for j, level in enumerate(BENCH_LEVELS) for i in range(per_level or LEVEL_MAPS)]


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


def evaluate_map(agent, spec: PolicySpec, bench_map: BenchMap, seeds: list[int], cap: int) -> dict:
    """Design one map (then polish / search, as `spec` says) and play it once per seed."""
    map_seed = bench_map.seed
    sim0 = bench_map.sim()
    result = agent.design(sim0)
    design, polish, search = result.sim, None, None
    if spec.polish_rounds:
        design, polish = agent.polish(sim0, design, rounds=spec.polish_rounds, rng=random.Random(map_seed))
    if spec.search_rounds:
        design, search = agent.improve(sim0, design, rounds=spec.search_rounds, seed_rng=search_seed_rng(map_seed),
                                       rng=random.Random(map_seed))
    runs = [run_playout(design, seed, cap) for seed in seeds]
    return {
        "level": bench_map.level,
        "layout": bench_map.spec.layout,
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


def _eval_map_in_worker(spec: PolicySpec, bench_map: BenchMap, seeds: list[int], cap: int) -> dict:
    return evaluate_map(_worker_agent, spec, bench_map, seeds, cap)


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


def _mirror_check_in_worker(bench_map: BenchMap) -> tuple[bool, bool]:
    sim = bench_map.sim()
    base = _worker_agent.design(sim).actions
    return (_worker_agent.design(mirrored(sim, True, False)).actions == base,
            _worker_agent.design(mirrored(sim, False, True)).actions == base)


def invariance(spec: PolicySpec, maps: list[BenchMap], workers: int | None = None) -> dict:
    """Share of maps whose design is unchanged by left-right / top-bottom mirroring."""
    with _pool(spec, max(workers or default_workers(), 1)) as pool:
        checks = list(pool.map(_mirror_check_in_worker, list(maps)))
    return {
        "left_right": float(np.mean([lr for lr, _ in checks])),
        "top_bottom": float(np.mean([tb for _, tb in checks])),
        "both": float(np.mean([lr and tb for lr, tb in checks])),
    }


def run(spec: PolicySpec, maps: list[BenchMap] | None = None, seeds=BENCH_SEEDS, cap: int = CAP,
        workers: int | None = None) -> dict:
    """Evaluate `spec` on every map (default: the levels suite). workers=0 runs in this
    process (for debugging)."""
    maps, seeds = list(maps or suite()), list(seeds)
    workers = default_workers() if workers is None else workers
    if workers == 0:
        agent = make_agent(spec)
        rows = [evaluate_map(agent, spec, m, seeds, cap) for m in maps]
    else:
        with _pool(spec, workers) as pool:
            rows = list(pool.map(_eval_map_in_worker, [spec] * len(maps), maps, [seeds] * len(maps),
                                 [cap] * len(maps)))
    return {"spec": asdict(spec), "cap": cap, "seeds": seeds, "maps": rows}


def map_key(row: dict) -> tuple[str, int]:
    """Runs pair up by level and map seed (results from before levels are classic)."""
    return row.get("level", "classic"), row["map"]


def map_scores(results: dict, where=None) -> dict[tuple[str, int], float]:
    """Per-map score: mean ticks survived over the passenger seeds (each capped). where(row)
    picks a subset of maps."""
    return {map_key(r): float(np.mean(r["ticks"])) for r in results["maps"] if where is None or where(r)}


def summarize(results: dict, where=None) -> dict:
    rows = [r for r in results["maps"] if where is None or where(r)]
    scores = np.array(list(map_scores(results, where).values()))
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


def compare(results: dict, reference: dict, where=None) -> dict:
    """Geometric-mean survival ratio vs `reference`, paired by map, with a 95% CI. A map is
    a win / loss when its ratio is more than 1% from 1. The runs must share seeds and cap;
    where(row) compares a subset of maps."""
    if results["seeds"] != reference["seeds"] or results["cap"] != reference["cap"]:
        raise ValueError("results and reference used different passenger seeds or caps")
    a, b = map_scores(results, where), map_scores(reference)
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


def results_dir(suite_name: str) -> str:
    return RESULTS_DIR if suite_name == "levels" else os.path.join(RESULTS_DIR, suite_name)


def save(results: dict, name: str, directory: str = RESULTS_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.json")
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
    for field in ("level", "layout"):
        values = sorted({r[field] for r in results["maps"] if field in r}, key=lambda v: (len(v), v))
        if len(values) < 2:
            continue
        lines.append(f"  by {field:7s} maps   mean  reached cap" + (f"   vs {reference_name}" if reference else ""))
        for value in values:
            def where(r):
                return r.get(field) == value
            g = summarize(results, where)
            row = f"    {value:10s} {g['maps']:4d} {g['mean']:6.0f}  {g['reached_cap']:9.0%}"
            if reference is not None:
                c = compare(results, reference, where)
                row += f"     x{c['ratio']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]" if c["n"] else "     (not in reference)"
            lines.append(row)
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
    parser.add_argument("--suite", choices=["levels", "classic"], default="levels")
    parser.add_argument("--maps", type=int, help="use the first N maps of each level")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--compare", help="results file to compare against, paired by map "
                                          "(default: the suite's heuristic.json)")
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
    maps = suite(args.suite, args.maps)
    directory = results_dir(args.suite)
    compare_path = args.compare or os.path.join(directory, "heuristic.json")
    start = time.time()
    results = run(spec, maps=maps, workers=args.workers)
    path = save(results, name, directory)
    reference = None
    if os.path.exists(compare_path) and os.path.abspath(compare_path) != os.path.abspath(path):
        reference = load(compare_path)
    print(report(name, results, reference, os.path.splitext(os.path.basename(compare_path))[0]))
    if args.invariance:
        inv = invariance(spec, maps=maps, workers=args.workers)
        print(f"  same design on mirrored map: left-right {inv['left_right']:.0%}, "
              f"top-bottom {inv['top_bottom']:.0%}, both {inv['both']:.0%}")
    print(f"  saved {path} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
