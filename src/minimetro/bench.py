"""Fixed benchmark: how long do a policy's designs survive, compared with the heuristic?

Suites: "test" has the first N maps of each size, all layouts, including sizes and layouts
no learner trains on; "val" is for tuning, on training-like maps; "classic" is the original
bench. Every run of a suite plays the same maps and passenger seeds, so runs compare map by
map. Run it with scripts/bench.py.
"""

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

from minimetro.config import HEIGHT, MAX_TRAINS, RESULTS_DIR, WIDTH
from minimetro.game.maps import LEVELS, MapSpec, novelty
from minimetro.game.point import Point
from minimetro.game.simulator import MiniMetroSim, run_playout

# Map seeds: training 0..9999 (up to 1000 per level), test 10_000.., validation 20_000.. .
SUITE_LEVELS = {"test": ("30", "35", "40"), "val": ("train30", "train35"),
                "train": ("train30", "train35")}                            # keys of maps.LEVELS
SUITE_SEED_START = {"test": 10_000, "val": 20_000, "train": 0}
LEVEL_MAPS = 100                          # maps per level
CLASSIC_MAPS = list(range(10_000, 10_100))
CAPS = {"test": 10_000, "val": 10_000, "classic": 6_000}   # classic keeps its historical cap
BENCH_SEEDS = list(range(5))              # passenger seeds, shared by every run


@dataclass(frozen=True)
class BenchMap:
    level: str      # a key of maps.LEVELS
    seed: int

    @property
    def spec(self) -> MapSpec:
        return LEVELS[self.level].spec(self.seed)

    def sim(self):
        return MiniMetroSim(self.seed, map_spec=self.spec)


def suite(name: str = "test", per_level: int | None = None) -> list[BenchMap]:
    """A suite's maps: the first `per_level` of each level, each level its own block of
    seeds ("classic": the original 20-station maps)."""
    if name == "classic":
        return [BenchMap("classic", m) for m in CLASSIC_MAPS[:per_level]]
    start = SUITE_SEED_START[name]
    if name == "train" and (per_level or LEVEL_MAPS) > 1_000:
        raise ValueError("at most 1000 training maps per level")
    return [BenchMap(level, start + 1_000 * j + i)
            for j, level in enumerate(SUITE_LEVELS[name]) for i in range(per_level or LEVEL_MAPS)]


@dataclass
class PolicySpec:
    kind: str = "heuristic"     # "heuristic" (prior only) or "agent" (prior + learned Q)
    ckpt: str | None = None
    mcts_sims: int = 0
    polish_rounds: int = 0      # then hill-climb on the survival estimate
    polish_restarts: int = 1    # climbs per map; the one the estimate rates best is kept
    search_rounds: int = 0      # then play-out search
    screen_edits: bool = True   # search plays the edits the survival estimate rates best


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def make_agent(spec: PolicySpec):
    from minimetro.design.agent import Agent
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
        design, polish = agent.polish(sim0, design, rounds=spec.polish_rounds, restarts=spec.polish_restarts,
                                      rng=random.Random(map_seed))
    if spec.search_rounds:
        design, search = agent.improve(sim0, design, rounds=spec.search_rounds, seed_rng=search_seed_rng(map_seed),
                                       rng=random.Random(map_seed))
    runs = [run_playout(design, seed, cap) for seed in seeds]
    return {
        "level": bench_map.level,
        "layout": bench_map.spec.layout,
        "novelty": novelty(bench_map.spec),
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


def run(spec: PolicySpec, maps: list[BenchMap] | None = None, seeds=BENCH_SEEDS, cap: int = CAPS["test"],
        workers: int | None = None) -> dict:
    """Evaluate `spec` on every map (default: the test suite). workers=0 runs in this
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


def share_of_gain(results: dict, teacher: dict, base: dict, where=None) -> dict:
    """How much of the teacher's gain over `base` a run gets: mean log(run / base) over
    mean log(teacher / base), on maps all three share. 1 = as good as the teacher. The CI
    resamples maps."""
    a, t, b = map_scores(results, where), map_scores(teacher), map_scores(base)
    common = sorted(set(a) & set(t) & set(b))
    if not common:
        return {"n": 0, "share": float("nan"), "lo": float("nan"), "hi": float("nan")}
    gain = np.log([a[m] / b[m] for m in common])
    teacher_gain = np.log([t[m] / b[m] for m in common])
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(common), size=(2000, len(common)))
    boot = gain[picks].mean(axis=1) / teacher_gain[picks].mean(axis=1)
    return {"n": len(common), "share": float(gain.mean() / teacher_gain.mean()),
            "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}


def results_dir(suite_name: str) -> str:
    return RESULTS_DIR if suite_name == "test" else os.path.join(RESULTS_DIR, suite_name)


def save(results: dict, name: str, directory: str = RESULTS_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f)
    return path


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def report(name: str, results: dict, reference: dict | None = None, reference_name: str = "",
           teacher: dict | None = None) -> str:
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
    teacher = teacher if reference is not None else None
    if teacher is not None:
        g = share_of_gain(results, teacher, reference)
        lines.append(f"  share of the teacher's gain: {g['share']:.0%} [{g['lo']:.0%}, {g['hi']:.0%}] on {g['n']} maps")
    for field in ("level", "layout", "novelty"):
        values = sorted({r[field] for r in results["maps"] if field in r}, key=lambda v: (len(v), v))
        if len(values) < 2:
            continue
        lines.append(f"  by {field:7s} maps   mean  reached cap" + (f"   vs {reference_name}" if reference else "")
                     + ("              teacher's gain" if teacher is not None else ""))
        for value in values:
            def where(r):
                return r.get(field) == value
            g = summarize(results, where)
            row = f"    {value:10s} {g['maps']:4d} {g['mean']:6.0f}  {g['reached_cap']:9.0%}"
            if reference is not None:
                c = compare(results, reference, where)
                row += f"     x{c['ratio']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]" if c["n"] else "     (not in reference)"
            if teacher is not None:
                g = share_of_gain(results, teacher, reference, where)
                row += f"   {g['share']:4.0%} [{g['lo']:.0%}, {g['hi']:.0%}]" if g["n"] else ""
            lines.append(row)
    return "\n".join(lines)


def default_name(spec: PolicySpec, maps: int | None = None) -> str:
    """Results name for a run; `maps` (per level) marks a partial run."""
    name = "heuristic" if spec.kind == "heuristic" else os.path.basename(os.path.normpath(spec.ckpt))
    if spec.mcts_sims:
        name += f"_mcts{spec.mcts_sims}"
    if spec.polish_rounds:
        name += f"_polish{spec.polish_rounds}" + (f"x{spec.polish_restarts}" if spec.polish_restarts > 1 else "")
    if spec.search_rounds:
        name += f"_{'' if spec.screen_edits else 'plain'}search{spec.search_rounds}"
    if maps:
        name += f"_m{maps}"
    return name
