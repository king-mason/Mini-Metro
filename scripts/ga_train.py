"""Train the genetic algorithm (learning/ga.py), plot its progress to results/ga_training_summary.png, and
show the best genome. Fitness is mean ticks survived, as in train.py."""

import os

import matplotlib.pyplot as plt
import numpy as np

from minimetro.config import RESULTS_DIR
from minimetro.game.maps import MapSpec
from minimetro.game.simulator import MiniMetroSim
from minimetro.learning.ga import GeneticAlgorithm, apply_genome
from minimetro.viewer import Visualizer

NUM_GENERATIONS = 100
ROLLING_WINDOW = 20

ga = GeneticAlgorithm(
    num_stations=20,
    population_size=30,
    elite_size=3,
    tournament_size=5,
    mutation_rate=0.3,
    crossover_rate=0.7,
    num_playouts=3,
    map_block_size=10,
    min_neighbors=4,       # minimum neighbors per station in the graph
    max_dist_fraction=0.35, # also include stations within this fraction of MAX_DIST
    structural_weight=1.0,  # scale for RL structural reward added to ticks fitness
)
# ga.load()  # uncomment to resume from checkpoints/ga/
ga.initialize()

best_history: list[float] = []
mean_history: list[float] = []

for _ in range(NUM_GENERATIONS):
    record = ga.step()
    best_history.append(record["best"])
    mean_history.append(record["mean"])
    print(
        f"Gen {record['generation']:4d} | Map {record['map_seed']:3d} | "
        f"Best: {record['best']:7.1f} | Mean: {record['mean']:7.1f}"
    )
    if record["generation"] % 20 == 0:
        ga.save()

ga.save()

# ---- summary plot ----
gens = np.arange(1, NUM_GENERATIONS + 1)

plt.figure(figsize=(14, 6))
plt.plot(gens, best_history, color="#39a6f5", linewidth=0.8, label="best (ticks)")
plt.plot(gens, mean_history, color="#ee637a", linewidth=0.8, alpha=0.7, label="mean (ticks)")
if NUM_GENERATIONS >= ROLLING_WINDOW:
    rolling = np.convolve(best_history, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid")
    plt.plot(
        np.arange(ROLLING_WINDOW, NUM_GENERATIONS + 1),
        rolling,
        color="#1f77b4",
        linewidth=2,
        label=f"rolling best ({ROLLING_WINDOW})",
    )
plt.xlabel("generation")
plt.ylabel("ticks survived")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.suptitle(
    f"GA training ({NUM_GENERATIONS} gens, pop={ga.population_size}, {ga.num_stations} stations, "
    f"k≥{ga.min_neighbors} neighbors, struct_w={ga.structural_weight})"
)
plt.tight_layout()
PLOT = os.path.join(RESULTS_DIR, "ga_training_summary.png")
plt.savefig(PLOT, dpi=120)
print(f"Saved {PLOT}")
plt.show()

# ---- final visualized run with best genome ----
assert ga.best_genome is not None, "No best genome found — did training run?"
sim = MiniMetroSim(0, map_spec=MapSpec(ga.num_stations))
apply_genome(sim, ga.best_genome)

print("Running final visualized simulation with best genome...")
vis = Visualizer()
vis.new_map(map=0, policy="GA best genome")
try:
    vis.play(sim)
    print(f"Final Run | Ticks: {sim.time}")
except SystemExit:
    pass
finally:
    vis.close()
