import copy
import math
import random
from collections import Counter

import yaml

from objects import Station, Line, Train, StationType
from point import Point

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
width, height = config["resolution"]
max_waiting = config["stations"]["capacity"]
_time_scale = config.get("time_scale", 1)
_spawn_chance_start = config["passengers"]["spawn_chance_init"]
_spawn_chance_max = config["passengers"]["spawn_chance_max"]
_spawn_chance_growth = max(1, config["passengers"]["spawn_chance_growth"] // _time_scale)
_spawn_max = max(1, config["passengers"]["spawn_interval_max"] // _time_scale)
_spawn_min = max(1, config["passengers"]["spawn_interval_min"] // _time_scale)
_spawn_decay = config["passengers"]["spawn_decay"] * _time_scale
max_lines = config["lines"]["max"]
# The game is lost once this many station-ticks over capacity have accumulated (all
# stations together, never reset).
max_overcrowd_ticks = config["stations"]["max_overcrowd_ticks"] // _time_scale


def current_spawn_interval(sim_time: int) -> float:
    """Ticks between spawn rounds: a logistic decay from spawn_interval_max to
    spawn_interval_min, centered after spawn chance has mostly grown."""
    diff = _spawn_max - _spawn_min
    k = 2 * _spawn_chance_growth   # set max decay time to after spawn chance has grown
    r = 4 * _spawn_decay / diff
    interval = _spawn_max + diff * (1 / (1 + math.exp(r * (sim_time - k))) - 1)
    return max(1, round(interval, 2))

def current_spawn_chance(sim_time: int) -> float:
    """Chance that a station spawns a passenger in a spawn round: grows logistically from
    spawn_chance_init toward spawn_chance_max."""
    diff = _spawn_chance_max - _spawn_chance_start
    return _spawn_chance_start + diff * (2 / (1 + math.exp(-2 * sim_time / _spawn_chance_growth)) - 1)


class MiniMetroSim:
    def __init__(self, num_stations=5, map_seed=None, passenger_seed=None):
        """map_seed fixes the station layout; passenger_seed fixes spawning.

        Each sim owns its RNGs, so nothing else that draws random numbers (exploration,
        search) can shift the passenger stream. map_seed=None falls back to the global
        `random` module for the layout, matching the old random.seed(...) call pattern.
        """
        map_rng = random.Random(map_seed) if map_seed is not None else random
        self.stations: list[Station] = self.get_new_stations(num_stations, rng=map_rng)
        self.rng = random.Random(passenger_seed)
        self.lines: list[Line] = []
        self.time = 0
        self.spawn_interval = current_spawn_interval(0)
        self.last_spawned_at = 0
        self.delivered = 0
        self.overcrowded_ticks = 0
        self.failed = False

    def set_passenger_seed(self, seed):
        """Reseed spawning. Deep copies share RNG state, so reseed each play-out copy."""
        self.rng.seed(seed)

    def get_new_stations(self, num_stations: int, rng=random):
        """Place stations on distinct points of a grid (spacing 4 station radii) covering the
        middle 80% x 60% of the screen. Every 10th station gets a rare shape."""
        station_positions = []
        station_radius = config["stations"]["radius"]
        for i in range(width//10, 9*width//10 + 1, 4*station_radius):
            for j in range(height//5, 4*height//5 + 1, 4*station_radius):
                station_positions.append(Point(i, j))

        stations = []
        shape_choices = [StationType.CIRCLE, StationType.SQUARE, StationType.TRIANGLE]
        shape_weights = [.5, .35, .15]
        rare_shape_choices = [StationType.CROSS, StationType.DIAMOND, StationType.PENTAGON, 
                              StationType.RHOMBUS, StationType.DROPLET, StationType.STAR, 
                              StationType.LEMON]
        
        # Create stations
        for _ in range(num_stations):
            if len(station_positions) == 0:
                return stations  # no more options
            station_pos = rng.choice(station_positions)
            station_positions.remove(station_pos)
            if (len(stations) + 1) % 10 == 0:
                shape = rng.choice(rare_shape_choices)
            else:
                shape = rng.choices(shape_choices, shape_weights, k=1)[0]
            station = Station(shape, station_pos)
            stations.append(station)

        # Ensure at least two distinct types so passengers always have a valid target
        types_present = {s.type for s in stations}
        if len(types_present) < 2 and len(stations) >= 2:
            current_type = next(iter(types_present))
            fallback = next(t for t in shape_choices if t != current_type)
            stations[-1].type = fallback

        return stations
    
    def target_type_probs(self, station_idx) -> dict[StationType, float]:
        """Target distribution for passengers spawned at station_idx: each other type in
        proportion to its number of stations, so common types are more likely targets.
        step() draws targets from exactly this.
        """
        station = self.stations[station_idx]
        counts = Counter(s.type for s in self.stations if s.type != station.type)
        total = sum(counts.values())
        if total == 0:
            return {}
        return {t: counts[t] / total for t in StationType}

    def add_line(self, station_ids):
        line_idx = len(self.lines)
        if line_idx >= max_lines:
            print('error trying to connect: max lines already')
            return  # max lines already reached
        line = Line(line_idx, [self.stations[i] for i in station_ids])
        train = Train(line)
        line.trains.append(train)
        self.lines.append(line)

    def step(self):
        """Advance one tick: spawn passengers, move trains, check for overcrowding."""
        self.time += 1
        station_types = [st.type for st in self.stations]

        # Resting passengers (re-entering after delivery) are switched off:
        # for s in self.stations:
        #     s.tick_resting(self.time, station_types)

        # Passenger spawning — interval shrinks over time toward spawn_interval_min.
        if self.time - self.last_spawned_at >= self.spawn_interval:
            self.last_spawned_at = self.time
            self.spawn_interval = current_spawn_interval(self.time)
            for i, s in enumerate(self.stations):
                if self.rng.random() < current_spawn_chance(self.time):
                    probs = self.target_type_probs(i)
                    types = list(probs)
                    s.spawn_passenger(types, weights=[probs[t] for t in types], current_time=self.time,
                                      rng=self.rng)

        for line in self.lines:
            for train in line.trains:
                self.delivered += train.step(self.lines, current_time=self.time)

        for s in self.stations:
            if len(s.waiting) > max_waiting:
                self.overcrowded_ticks += 1
                if self.overcrowded_ticks > max_overcrowd_ticks:
                    self.failed = True


def run_playout(sim: MiniMetroSim, passenger_seed: int, max_ticks: int) -> tuple[int, int]:
    """Play a finished design forward on a copy of `sim`. Returns (ticks survived, delivered).

    Module-level so it can be sent to worker processes.
    """
    s = copy.deepcopy(sim)
    s.set_passenger_seed(passenger_seed)
    while s.time < max_ticks and not s.failed:
        s.step()
    return s.time, s.delivered
