"""Train the design agent (training.py), validating against the heuristic.

Every episode is a new map. Every EVAL_EVERY episodes the weights are scored on VAL_MAPS
against the heuristic, and the best are kept in BEST_CKPT. VAL_MAPS are separate from the
training and bench maps, so picking the best checkpoint doesn't inflate the bench.
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

import bench
from agent import Agent
from training import Trainer

NUM_EPISODES = 500
ROLLING_WINDOW = 50
EVAL_EVERY = 25
CKPT = "checkpoint_v4"
BEST_CKPT = "checkpoint_v4_best"
VAL_MAPS = list(range(20_000, 20_040))
VAL_SEEDS = list(range(3))
WORKERS = bench.default_workers()


def plot(history: list[dict], evals: list[tuple[int, dict]], path: str = "training_summary.png"):
    fig, (ax_train, ax_val) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    episodes = np.arange(1, len(history) + 1)
    greedy = np.array([h["ticks_greedy"] for h in history])
    ax_train.plot(episodes, greedy, color="#39a6f5", linewidth=0.7, label="greedy design on the training map")
    if len(greedy) >= ROLLING_WINDOW:
        rolling = np.convolve(greedy, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid")
        ax_train.plot(np.arange(ROLLING_WINDOW, len(greedy) + 1), rolling, color="#1f77b4",
                      linewidth=2, label=f"rolling mean ({ROLLING_WINDOW})")
    ax_train.set_ylabel("ticks survived (capped)")
    ax_train.legend(loc="best")
    ax_train.grid(alpha=0.3)

    if evals:
        x = [e for e, _ in evals]
        ratio = np.array([c["ratio"] for _, c in evals])
        lo = np.array([c["lo"] for _, c in evals])
        hi = np.array([c["hi"] for _, c in evals])
        ax_val.errorbar(x, ratio, yerr=[ratio - lo, hi - ratio], color="#2ca02c", marker="o",
                        capsize=3, linewidth=2, label="validation survival vs heuristic (geo-mean, 95% CI)")
    ax_val.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="heuristic")
    ax_val.set_ylabel("ratio vs heuristic")
    ax_val.set_xlabel("episode")
    ax_val.legend(loc="best")
    ax_val.grid(alpha=0.3)

    fig.suptitle(f"Training summary ({len(history)} episodes, branch-and-compare)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")
    plt.show()


def train(trainer: Trainer, num_episodes: int, ckpt: str = CKPT, best_ckpt: str = BEST_CKPT):
    """Run training episodes on fresh maps, validating every EVAL_EVERY. Returns (history, evals)."""
    heuristic_val = bench.run(bench.PolicySpec("heuristic"), VAL_MAPS, VAL_SEEDS, workers=WORKERS)
    print(bench.report("heuristic on validation maps", heuristic_val))

    map_rng = random.Random()
    history: list[dict] = []
    evals: list[tuple[int, dict]] = []
    for episode in range(num_episodes):
        stats = trainer.episode(map_seed=map_rng.randrange(10_000))
        history.append(stats)
        print(f"Episode {episode + 1:4d} | Map {stats['map_seed']:4d} | greedy {stats['ticks_greedy']:6.0f}"
              f" | best found {stats['ticks_best']:6.0f} | accepted {stats['accepted']}/{trainer.config.branch_rounds}"
              f" | mean |adv| {stats['mean_abs_adv']:.3f} | loss {stats['loss']:.4f}", flush=True)

        if (episode + 1) % EVAL_EVERY == 0:
            trainer.save(ckpt)
            val = bench.run(bench.PolicySpec("agent", ckpt=ckpt), VAL_MAPS, VAL_SEEDS, workers=WORKERS)
            c = bench.compare(val, heuristic_val)
            evals.append((episode + 1, c))
            print(f"  >> VAL vs heuristic ({c['n']} maps): x{c['ratio']:.3f} [95% CI {c['lo']:.3f}, {c['hi']:.3f}]"
                  f" | better on {c['wins']:.0%}, worse on {c['losses']:.0%}", flush=True)
            trainer.save_if_best(math.log(c["ratio"]), path=best_ckpt)
    return history, evals


def main():
    agent = Agent(workers=WORKERS)
    trainer = Trainer(agent)
    trainer.load(CKPT)
    try:
        history, evals = train(trainer, NUM_EPISODES)
    finally:
        agent.close()

    plot(history, evals)

    # ---- final visualized run with the best checkpoint (more: python run_visual.py) ----
    from run_visual import watch
    from visualizer import Visualizer
    agent.load(BEST_CKPT)
    vis = Visualizer()
    try:
        watch(agent, vis, [random.randrange(10_000)], policy=BEST_CKPT)
    except SystemExit:
        pass
    finally:
        vis.close()


if __name__ == "__main__":
    main()
