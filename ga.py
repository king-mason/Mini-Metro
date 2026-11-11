"""Genetic algorithm for network design: an alternative to the agent (run via ga_main.py).

Genomes are lines (random walks on a neighbor graph) plus trains per line. Fitness is ticks
survived in play-outs plus the structural prior of the genome's moves; moves that break
the rules (rules.py) are skipped.
"""

from __future__ import annotations

import copy
import os
import pickle
import random
from dataclasses import dataclass, field

import numpy as np

from config import MAX_DIST, MAX_TRAINS, MAX_TRAINS_PER_LINE
from config import MAX_LINES as max_lines
from heuristic import structural_reward
from maps import MapSpec
from rules import ActionType, apply_action, is_legal
from simulator import MiniMetroSim, run_playout


# ---------------------------------------------------------------------------
# Neighbor graph
# ---------------------------------------------------------------------------

def build_neighbor_graph(
    stations: list,
    min_neighbors: int = 4,
    max_dist_fraction: float = 0.35,
) -> dict[int, set[int]]:
    """Build a set of valid station-to-station connections.

    Station j is reachable from station i when either:
    - j is among the min_neighbors closest stations to i, OR
    - distance(i, j) <= max_dist_fraction * MAX_DIST.

    The result is a symmetric adjacency dict.
    """
    n = len(stations)
    thresh = max_dist_fraction * MAX_DIST
    neighbors: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        by_dist = sorted(
            (stations[i].pos.get_distance(stations[j].pos), j)
            for j in range(n) if j != i
        )
        for k, (d, j) in enumerate(by_dist):
            if k < min_neighbors or d <= thresh:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------

@dataclass
class Genome:
    """A metro network design.

    lines[i]  — ordered station-index list for line i (length >= 2)
    trains[i] — desired total train count for line i (1..MAX_TRAINS_PER_LINE)
    fitness   — cached evaluation result (None = needs evaluation)
    """
    lines: list[list[int]] = field(default_factory=list)
    trains: list[int] = field(default_factory=list)
    fitness: float | None = None

    def clone(self) -> "Genome":
        return Genome(lines=[l[:] for l in self.lines], trains=self.trains[:])


def _dedupe(line: list[int]) -> list[int]:
    seen: set[int] = set()
    out = []
    for x in line:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# Genome construction
# ---------------------------------------------------------------------------

def random_genome(
    num_stations: int,
    neighbors: dict[int, set[int]] | None = None,
) -> Genome:
    if neighbors is not None:
        return _random_genome_from_graph(num_stations, neighbors)
    return _random_genome_unconstrained(num_stations)


def _random_genome_unconstrained(num_stations: int) -> Genome:
    n_lines = random.randint(1, max_lines)
    lines, trains, total = [], [], 0
    for _ in range(n_lines):
        size = random.randint(2, max(2, num_stations // 2))
        lines.append(random.sample(range(num_stations), min(size, num_stations)))
        budget = max(1, min(MAX_TRAINS_PER_LINE, MAX_TRAINS - total))
        t = random.randint(1, budget)
        trains.append(t)
        total += t
    return Genome(lines=lines, trains=trains)


def _random_genome_from_graph(
    num_stations: int,
    neighbors: dict[int, set[int]],
) -> Genome:
    """Build lines as random walks through the neighbor graph."""
    n_lines = random.randint(1, max_lines)
    lines, trains, total = [], [], 0
    for _ in range(n_lines):
        start = random.randrange(num_stations)
        target_len = random.randint(2, max(2, num_stations // 2))
        line = [start]
        for _ in range(target_len - 1):
            current = line[-1]
            opts = [j for j in neighbors.get(current, set()) if j not in line]
            if not opts:
                break
            line.append(random.choice(opts))
        if len(line) >= 2:
            lines.append(line)
            budget = max(1, min(MAX_TRAINS_PER_LINE, MAX_TRAINS - total))
            t = random.randint(1, budget)
            trains.append(t)
            total += t

    if not lines:
        for i in range(num_stations):
            opts = list(neighbors.get(i, set()))
            if opts:
                lines.append([i, opts[0]])
                trains.append(1)
                break
        if not lines:
            a, b = random.sample(range(num_stations), 2)
            lines.append([a, b])
            trains.append(1)

    return Genome(lines=lines, trains=trains)


# ---------------------------------------------------------------------------
# Apply genome to simulator
# ---------------------------------------------------------------------------

def genome_actions(sim: MiniMetroSim, genome: Genome):
    """Build `genome` on `sim` (stations, no lines) as design actions, yielding each action
    just before it is applied. Steps the rules forbid are skipped: a line whose first
    segment would cross the network, a station whose segment would, extra trains."""
    for i, raw_line in enumerate(genome.lines):
        if len(sim.lines) >= max_lines:
            break
        unique = _dedupe([idx for idx in raw_line if 0 <= idx < len(sim.stations)])
        if len(unique) < 2:
            continue
        action = (ActionType.NEW_LINE, unique[0], unique[1])
        if not is_legal(sim, action):
            continue
        yield action
        apply_action(sim, action)
        line_idx = len(sim.lines) - 1
        for j in unique[2:]:
            action = (ActionType.EXTEND_LINE, line_idx, -1, j)
            if is_legal(sim, action):
                yield action
                apply_action(sim, action)
        desired = genome.trains[i] if i < len(genome.trains) else 1
        for _ in range(desired - 1):
            action = (ActionType.ADD_TRAIN, line_idx)
            if not is_legal(sim, action):
                break
            yield action
            apply_action(sim, action)


def apply_genome(sim: MiniMetroSim, genome: Genome) -> None:
    """Apply a genome to a freshly-created (lines-empty) sim in-place."""
    for _ in genome_actions(sim, genome):
        pass


# ---------------------------------------------------------------------------
# Structural fitness component
# ---------------------------------------------------------------------------

def compute_structural_reward(template_sim: MiniMetroSim, genome: Genome) -> float:
    """Replay genome as RL design actions and accumulate structural reward.

    Reuses structural_reward from heuristic.py verbatim, so the GA sees the same
    bend/isolation/reach/coverage signals the agent's prior uses. The template_sim must
    have stations but no lines yet.
    """
    sim = copy.deepcopy(template_sim)
    return sum(structural_reward(sim, action) for action in genome_actions(sim, genome))


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def evaluate(
    genome: Genome,
    num_stations: int,
    num_playouts: int = 3,
    map_seed: int | None = None,
    max_ticks: int = 50_000,
    structural_weight: float = 1.0,
) -> float:
    """Combined fitness: mean ticks survived + structural_weight * structural reward."""
    template = MiniMetroSim(map_seed, map_spec=MapSpec(num_stations))

    s_reward = compute_structural_reward(template, genome)

    design = copy.deepcopy(template)
    apply_genome(design, genome)
    total_ticks = sum(run_playout(design, random.randrange(2 ** 31), max_ticks)[0] for _ in range(num_playouts))

    return total_ticks / num_playouts + structural_weight * s_reward


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _edges_valid(line: list[int], neighbors: dict[int, set[int]]) -> bool:
    return all(line[k + 1] in neighbors.get(line[k], set()) for k in range(len(line) - 1))


def crossover(p1: Genome, p2: Genome) -> tuple[Genome, Genome]:
    """Uniform line-slot crossover: each of the max_lines slots is drawn
    independently from either parent."""
    l1 = p1.lines + [[] for _ in range(max_lines - len(p1.lines))]
    l2 = p2.lines + [[] for _ in range(max_lines - len(p2.lines))]
    t1 = p1.trains + [1] * (max_lines - len(p1.trains))
    t2 = p2.trains + [1] * (max_lines - len(p2.trains))

    c1l, c1t, c2l, c2t = [], [], [], []
    for i in range(max_lines):
        if random.random() < 0.5:
            c1l.append(l1[i][:]); c1t.append(t1[i])
            c2l.append(l2[i][:]); c2t.append(t2[i])
        else:
            c1l.append(l2[i][:]); c1t.append(t2[i])
            c2l.append(l1[i][:]); c2t.append(t1[i])

    def _clean(lines, trains):
        pairs = [(l, t) for l, t in zip(lines, trains) if len(l) >= 2]
        return ([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else ([], [])

    cl1, ct1 = _clean(c1l, c1t)
    cl2, ct2 = _clean(c2l, c2t)
    if not cl1:
        cl1, ct1 = ([p1.lines[0][:]] if p1.lines else [[0, 1]]), [1]
    if not cl2:
        cl2, ct2 = ([p2.lines[0][:]] if p2.lines else [[0, 1]]), [1]
    return Genome(lines=cl1, trains=ct1), Genome(lines=cl2, trains=ct2)


def mutate(
    genome: Genome,
    num_stations: int,
    neighbors: dict[int, set[int]] | None = None,
    rate: float = 0.3,
) -> Genome:
    g = genome.clone()
    if not g.lines:
        if neighbors is not None:
            return _random_genome_from_graph(num_stations, neighbors)
        a, b = random.sample(range(num_stations), 2)
        return Genome(lines=[[a, b]], trains=[1])

    for i in range(len(g.lines)):
        if random.random() > rate:
            continue
        line = g.lines[i]
        op = random.choices(
            ["insert", "remove", "swap", "reverse", "train_up", "train_down"],
            weights=[3, 2, 3, 1, 2, 2],
        )[0]

        if op == "insert":
            # Neighbor constraint: new station must be a valid neighbor of the
            # stations it would sit between (or adjacent to at endpoints).
            # swap/reverse ops may produce non-neighbor edges, which is acceptable
            # — the structural reward will naturally penalize them.
            pos = random.randint(0, len(line))
            if neighbors is not None:
                if pos == 0:
                    valid = neighbors.get(line[0], set()) - set(line)
                elif pos == len(line):
                    valid = neighbors.get(line[-1], set()) - set(line)
                else:
                    valid = (
                        neighbors.get(line[pos - 1], set())
                        & neighbors.get(line[pos], set())
                    ) - set(line)
            else:
                valid = set(range(num_stations)) - set(line)
            if valid:
                line.insert(pos, random.choice(list(valid)))

        elif op == "remove" and len(line) > 2:
            if neighbors is not None:
                # Endpoints can always be removed; interior removal creates a new
                # edge between the flanking stations — only allow if that edge is valid.
                valid_pos = [0, len(line) - 1] + [
                    p for p in range(1, len(line) - 1)
                    if line[p + 1] in neighbors.get(line[p - 1], set())
                ]
            else:
                valid_pos = list(range(len(line)))
            if valid_pos:
                line.pop(random.choice(valid_pos))

        elif op == "swap" and len(line) >= 2:
            a, b = sorted(random.sample(range(len(line)), 2))
            candidate = line[:]
            candidate[a], candidate[b] = line[b], line[a]
            if neighbors is None or _edges_valid(candidate, neighbors):
                g.lines[i] = candidate

        elif op == "reverse":
            candidate = line[::-1]
            if neighbors is None or _edges_valid(candidate, neighbors):
                g.lines[i] = candidate

        elif op == "train_up":
            g.trains[i] = min(MAX_TRAINS_PER_LINE, g.trains[i] + 1)

        elif op == "train_down":
            g.trains[i] = max(1, g.trains[i] - 1)

    # Add a new line via a fresh graph walk
    if len(g.lines) < max_lines and random.random() < rate:
        seed_genome = random_genome(num_stations, neighbors)
        if seed_genome.lines:
            g.lines.append(seed_genome.lines[0])
            g.trains.append(seed_genome.trains[0])

    # Remove a line
    if len(g.lines) > 1 and random.random() < rate * 0.3:
        idx = random.randrange(len(g.lines))
        g.lines.pop(idx)
        g.trains.pop(idx)

    # Clamp total trains
    while sum(g.trains) > MAX_TRAINS:
        over = [i for i, t in enumerate(g.trains) if t > 1]
        if not over:
            break
        g.trains[random.choice(over)] -= 1

    return g


# ---------------------------------------------------------------------------
# GA
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    def __init__(
        self,
        num_stations: int = 20,
        population_size: int = 30,
        elite_size: int = 3,
        tournament_size: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        num_playouts: int = 3,
        map_block_size: int = 10,
        min_neighbors: int = 4,
        max_dist_fraction: float = 0.35,
        structural_weight: float = 1.0,
    ):
        self.num_stations = num_stations
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_playouts = num_playouts
        self.map_block_size = map_block_size
        self.min_neighbors = min_neighbors
        self.max_dist_fraction = max_dist_fraction
        self.structural_weight = structural_weight

        self.population: list[Genome] = []
        self.generation: int = 0
        self.best_genome: Genome | None = None
        self.best_fitness: float = float("-inf")
        self.history: list[dict] = []

        self._cached_seed: int | None = None
        self._neighbors: dict[int, set[int]] | None = None

    def _get_neighbors(self, map_seed: int) -> dict[int, set[int]]:
        """Return the neighbor graph for map_seed, rebuilding if the seed changed."""
        if map_seed == self._cached_seed and self._neighbors is not None:
            return self._neighbors
        ref = MiniMetroSim(map_seed, map_spec=MapSpec(self.num_stations))
        self._neighbors = build_neighbor_graph(
            ref.stations, self.min_neighbors, self.max_dist_fraction
        )
        self._cached_seed = map_seed
        return self._neighbors

    def initialize(self) -> None:
        neighbors = self._get_neighbors(0)
        self.population = [
            random_genome(self.num_stations, neighbors)
            for _ in range(self.population_size)
        ]

    def _select(self) -> Genome:
        competitors = random.sample(self.population, self.tournament_size)
        return max(competitors, key=lambda g: g.fitness or 0.0)

    def step(self) -> dict:
        """Evaluate, select, reproduce, and mutate. Returns generation stats."""
        map_seed = self.generation // self.map_block_size
        neighbors = self._get_neighbors(map_seed)

        for g in self.population:
            if g.fitness is None:
                g.fitness = evaluate(
                    g,
                    self.num_stations,
                    self.num_playouts,
                    map_seed,
                    structural_weight=self.structural_weight,
                )

        self.population.sort(key=lambda g: g.fitness or 0.0, reverse=True)

        if (self.population[0].fitness or 0.0) > self.best_fitness:
            self.best_fitness = self.population[0].fitness  # type: ignore[assignment]
            self.best_genome = self.population[0].clone()

        fitnesses = [g.fitness or 0.0 for g in self.population]
        record = {
            "generation": self.generation + 1,
            "map_seed": map_seed,
            "best": max(fitnesses),
            "mean": float(np.mean(fitnesses)),
            "worst": min(fitnesses),
        }
        self.history.append(record)

        next_gen: list[Genome] = [g.clone() for g in self.population[: self.elite_size]]
        while len(next_gen) < self.population_size:
            p1 = self._select()
            if random.random() < self.crossover_rate:
                c1, c2 = crossover(p1, self._select())
            else:
                c1, c2 = p1.clone(), p1.clone()
            c1 = mutate(c1, self.num_stations, neighbors, self.mutation_rate)
            c2 = mutate(c2, self.num_stations, neighbors, self.mutation_rate)
            c1.fitness = c2.fitness = None
            next_gen.append(c1)
            if len(next_gen) < self.population_size:
                next_gen.append(c2)

        self.population = next_gen
        self.generation += 1
        return record

    def save(self, path: str = "checkpoint_ga") -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "ga_state.pkl"), "wb") as f:
            pickle.dump({
                "generation": self.generation,
                "best_genome": self.best_genome,
                "best_fitness": self.best_fitness,
                "history": self.history,
                "population": self.population,
            }, f)
        print(f"Saved to {path}/")

    def load(self, path: str = "checkpoint_ga") -> None:
        p = os.path.join(path, "ga_state.pkl")
        if not os.path.exists(p):
            return
        with open(p, "rb") as f:
            s = pickle.load(f)
        self.generation = s["generation"]
        self.best_genome = s["best_genome"]
        self.best_fitness = s["best_fitness"]
        self.history = s["history"]
        self.population = s["population"]
        print(f"Loaded from {path}/ (gen {self.generation}, best {self.best_fitness:.1f})")
