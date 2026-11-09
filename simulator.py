"""The game: a map of stations, the lines and trains built on it (rules.py), and passengers
spawning faster and faster. A design survives until too many stations have overcrowded
for too long (MAX_OVERCROWD_TICKS)."""

import copy
import math
import random
from collections import Counter

from config import (HEIGHT, MAX_LINES, MAX_OVERCROWD_TICKS, NUM_STATIONS, PASSENGERS, STATION_CAPACITY, STATION_RADIUS,
                    TIME_SCALE, WIDTH)
from objects import Line, Passenger, Station, StationType, Train
from point import Point

_CHANCE_START = PASSENGERS["spawn_chance_init"]
_CHANCE_MAX = PASSENGERS["spawn_chance_max"]
_CHANCE_GROWTH = max(1, PASSENGERS["spawn_chance_growth"] // TIME_SCALE)
_INTERVAL_MAX = max(1, PASSENGERS["spawn_interval_max"] // TIME_SCALE)
_INTERVAL_MIN = max(1, PASSENGERS["spawn_interval_min"] // TIME_SCALE)
_INTERVAL_DECAY = PASSENGERS["spawn_decay"] * TIME_SCALE

COMMON_TYPES = (StationType.CIRCLE, StationType.SQUARE, StationType.TRIANGLE)
COMMON_WEIGHTS = (0.5, 0.35, 0.15)
RARE_TYPES = (StationType.CROSS, StationType.DIAMOND, StationType.PENTAGON, StationType.RHOMBUS,
              StationType.DROPLET, StationType.STAR, StationType.LEMON)


def current_spawn_interval(t: int) -> float:
    """Ticks between spawn rounds: falls logistically from the max to the min, centered after
    the spawn chance has mostly grown."""
    diff = _INTERVAL_MAX - _INTERVAL_MIN
    center = 2 * _CHANCE_GROWTH
    rate = 4 * _INTERVAL_DECAY / diff
    interval = _INTERVAL_MAX + diff * (1 / (1 + math.exp(rate * (t - center))) - 1)
    return max(1, round(interval, 2))


def current_spawn_chance(t: int) -> float:
    """Chance that a station spawns a passenger in a spawn round: grows logistically."""
    diff = _CHANCE_MAX - _CHANCE_START
    return _CHANCE_START + diff * (2 / (1 + math.exp(-2 * t / _CHANCE_GROWTH)) - 1)


def random_stations(n: int, rng) -> list[Station]:
    """n stations on distinct points of a grid (spacing 4 radii) over the middle of the
    screen. Every 10th station gets a rare type, and there are always at least two types."""
    spacing = 4 * STATION_RADIUS
    free = [Point(x, y) for x in range(WIDTH // 10, 9 * WIDTH // 10 + 1, spacing)
            for y in range(HEIGHT // 5, 4 * HEIGHT // 5 + 1, spacing)]
    stations = []
    for k in range(1, min(n, len(free)) + 1):
        pos = rng.choice(free)
        free.remove(pos)
        stype = rng.choice(RARE_TYPES) if k % 10 == 0 else rng.choices(COMMON_TYPES, COMMON_WEIGHTS)[0]
        stations.append(Station(stype, pos))
    types = {s.type for s in stations}
    if len(types) < 2 and len(stations) >= 2:
        stations[-1].type = next(t for t in COMMON_TYPES if t not in types)
    return stations


class MiniMetroSim:
    def __init__(self, num_stations: int = NUM_STATIONS, map_seed: int | None = None, passenger_seed: int | None = None):
        """map_seed fixes the station layout, passenger_seed the spawning. Each sim owns its
        RNGs, so other random draws (exploration, search) can't shift the passenger stream."""
        self.stations = random_stations(num_stations, random.Random(map_seed) if map_seed is not None else random)
        self.rng = random.Random(passenger_seed)
        self.lines: list[Line] = []
        self.time = 0
        self.spawn_interval = current_spawn_interval(0)
        self.last_spawned_at = 0
        self.delivered = 0
        self.overcrowded_ticks = 0
        self.failed = False

    def set_passenger_seed(self, seed: int):
        """Reseed spawning. Deep copies share RNG state, so reseed each play-out copy."""
        self.rng.seed(seed)

    def target_type_probs(self, station_idx: int) -> dict[StationType, float]:
        """Where passengers spawned here want to go: each other type in proportion to its
        number of stations."""
        own = self.stations[station_idx].type
        counts = Counter(s.type for s in self.stations if s.type != own)
        total = sum(counts.values())
        return {t: counts[t] / total for t in StationType} if total else {}

    def add_line(self, station_ids: list[int]):
        """A new line through these stations, with one train."""
        if len(self.lines) >= MAX_LINES:
            raise ValueError(f"all {MAX_LINES} lines are in use")
        line = Line(len(self.lines), [self.stations[i] for i in station_ids])
        line.add_train(Train(line))
        self.lines.append(line)

    def step(self):
        """One tick: spawn passengers (every spawn_interval ticks), move trains, count
        overcrowded stations."""
        self.time += 1
        if self.time - self.last_spawned_at >= self.spawn_interval:
            self.last_spawned_at = self.time
            self.spawn_interval = current_spawn_interval(self.time)
            for i, station in enumerate(self.stations):
                if self.rng.random() < current_spawn_chance(self.time):
                    probs = self.target_type_probs(i)
                    if probs:
                        target = self.rng.choices(list(probs), weights=list(probs.values()))[0]
                        station.waiting.append(Passenger(target))

        for line in self.lines:
            for train in line.trains:
                self.delivered += train.step(self.lines)

        for station in self.stations:
            if len(station.waiting) > STATION_CAPACITY:
                self.overcrowded_ticks += 1
                if self.overcrowded_ticks > MAX_OVERCROWD_TICKS:
                    self.failed = True


def run_playout(sim: MiniMetroSim, passenger_seed: int, max_ticks: int) -> tuple[int, int]:
    """Play a design forward on a copy of `sim`: (ticks survived, passengers delivered).
    Module-level so worker processes can run it."""
    s = copy.deepcopy(sim)
    s.set_passenger_seed(passenger_seed)
    while s.time < max_ticks and not s.failed:
        s.step()
    return s.time, s.delivered
