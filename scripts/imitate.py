"""Train the student to build like the teacher (learning/imitation.py).

    python scripts/imitate.py                                   # data/teacher_polish30.jsonl, 3 epochs
    python scripts/imitate.py --epochs 1 --maps 100             # a quick trial

Each training map: the student walks to the teacher's design and takes one gradient step.
Every --eval-every maps it is benchmarked on validation maps with real play-outs, scored as
its share of the teacher's gain over the heuristic there; the best weights are kept.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from minimetro import bench
from minimetro.config import CHECKPOINT_DIR, DATA_DIR
from minimetro.design.agent import Agent
from minimetro.design.search import Spec
from minimetro.learning.imitation import Imitator, accuracy, load_teacher

STUDENT_K = 25            # candidates per decision: a right move is among them ~99% of the time
VAL_SEEDS = [0, 1, 2]


def validation_reference(spec: bench.PolicySpec, maps, workers: int, path: str) -> dict:
    """A validation run, cached in `path` (the maps and seeds never change)."""
    if os.path.exists(path):
        return bench.load(path)
    results = bench.run(spec, maps, VAL_SEEDS, cap=bench.CAPS["val"], workers=workers)
    bench.save(results, os.path.splitext(os.path.basename(path))[0], os.path.dirname(path))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", default=os.path.join(DATA_DIR, "teacher_polish30.jsonl"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--maps", type=int, help="use only the first N teacher maps")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.1, help="softmax temperature over move scores")
    parser.add_argument("--eval-every", type=int, default=200, help="training maps between validations")
    parser.add_argument("--val-maps", type=int, default=30, help="validation maps per level")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=os.path.join(CHECKPOINT_DIR, "student"))
    parser.add_argument("--init", help="start from these weights")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_teacher(args.teacher)[:args.maps]
    torch.manual_seed(args.seed)
    agent = Agent(use_q=True, k_candidates=STUDENT_K, override_margin=0.0, seed=args.seed)
    if args.init and not agent.load(args.init):
        raise FileNotFoundError(args.init)
    imitator = Imitator(agent, lr=args.lr, tau=args.tau, seed=args.seed)

    val_maps = bench.suite("val", args.val_maps)
    val_dir = os.path.join(bench.RESULTS_DIR, "val")
    base = validation_reference(bench.PolicySpec(), val_maps, args.workers, os.path.join(val_dir, f"imitation_heuristic_{args.val_maps}.json"))
    teacher = validation_reference(bench.PolicySpec(polish_rounds=30), val_maps, args.workers,
                                   os.path.join(val_dir, f"imitation_teacher_{args.val_maps}.json"))
    best_ckpt, log_path = args.out + "_best", args.out + "_log.jsonl"
    best = float("-inf")

    def validate(maps_seen: int, window: list[dict]):
        nonlocal best
        agent.save(args.out)
        start = time.time()
        results = bench.run(bench.PolicySpec("agent", ckpt=args.out), val_maps, VAL_SEEDS, cap=bench.CAPS["val"],
                            workers=args.workers)
        g = bench.share_of_gain(results, teacher, base)
        c = bench.compare(results, base)
        acc = accuracy(window) if window else {}
        entry = {"maps": maps_seen, "term_weights": agent.term_weights(), "share": g["share"], "lo": g["lo"], "hi": g["hi"], "ratio": c["ratio"],
                 "agreed": acc.get("agreed"), "loss": float(np.nanmean([s["loss"] for s in window])) if window else None}
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[{maps_seen:5d} maps] val: x{c['ratio']:.3f} vs heuristic, {g['share']:.0%} of the teacher's gain "
              f"[{g['lo']:.0%}, {g['hi']:.0%}]" + (f" | train: picks right {acc['agreed']:.0%}, loss {entry['loss']:.3f}"
                                                   if window else "") + f" ({time.time() - start:.0f}s)", flush=True)
        if window:
            print("    prior term weights: " + ", ".join(f"{k} {v:.2f}" for k, v in agent.term_weights().items()), flush=True)
        if g["share"] > best:
            best = g["share"]
            agent.save(best_ckpt)

    rng = random.Random(args.seed)
    validate(0, [])
    seen, window = 0, []
    for epoch in range(args.epochs):
        for row in rng.sample(rows, len(rows)):
            sim0 = bench.BenchMap(row["level"], row["map"]).sim()
            window.append(imitator.learn_map(sim0, Spec(row["lines"], row["trains"])))
            seen += 1
            if seen % args.eval_every == 0:
                validate(seen, window)
                window = []
    if window:
        validate(seen, window)
    print(f"best: {best:.0%} of the teacher's gain, saved in {best_ckpt}/")


if __name__ == "__main__":
    main()
