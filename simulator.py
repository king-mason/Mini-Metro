import random
from collections import Counter
from objects import Station, Line, Train, Passenger, StationType
from point import Point

import yaml

import math

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
_overcrowd_penalty = config["rewards"]["overcrowd_penalty"]
_max_overcrowd_ticks = config["stations"]["max_overcrowd_ticks"] // _time_scale

# def current_spawn_interval(sim_time: int) -> int:
#     """Exponential decay toward a minimum: interval = min + (start - min) * exp(-t / decay)."""
#     interval = _spawn_min + (_spawn_start - _spawn_min) * math.exp(-sim_time / _spawn_decay)
#     return max(1, round(interval))

def current_spawn_interval(sim_time: int) -> float:
    """Logistic decay toward a minimum: interval = """
    diff = _spawn_max - _spawn_min
    k = 2 * _spawn_chance_growth   # set max decay time to after spawn chance has grown
    r = 4 * _spawn_decay / diff
    interval = _spawn_max + diff * (1 / (1 + math.exp(r * (sim_time - k))) - 1)
    return max(1, round(interval, 2))

def current_spawn_chance(sim_time: int) -> float:
    """Logistic growth toward a maximum: chance = """
    diff = _spawn_chance_max - _spawn_chance_start
    return _spawn_chance_start + diff * (2 / (1 + math.exp(-2 * sim_time / _spawn_chance_growth)) - 1)


# TODO



class MiniMetroSim:
    def __init__(self, num_stations=5):
        self.stations: list[Station] = self.get_new_stations(num_stations)
        self.lines: list[Line] = []
        self.time = 0
        self.spawn_interval = current_spawn_interval(0)
        self.last_spawned_at = 0
        self.delivered = 0
        self.overcrowing_ticks = 0
        self.failed = False
    
    def get_new_stations(self, num_stations: int):
        station_positions = []
        station_radius = config["stations"]["radius"]
        # for i in range(4*station_radius, int(width - 3*station_radius), 4*station_radius):
        for i in range(width//10, 9*width//10 + 1, 4*station_radius):
            row = []
            # for j in range(4*station_radius, int(height - 3*station_radius), 4*station_radius):
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
            station_pos = random.choice(station_positions)
            station_positions.remove(station_pos)
            if (len(stations) + 1) % 10 == 0:
                shape = random.choice(rare_shape_choices)
            else:
                shape = random.choices(shape_choices, shape_weights, k=1)[0]
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
        """Expected target distribution for passengers spawned at station_idx.

        Station.spawn_passenger samples from the station type list with duplicates,
        excluding the origin type, so common station types are more likely targets.
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
        reward = 0
        self.time += 1
        delivered_passengers_all: list = []
        station_types = [st.type for st in self.stations]

        # Re-activate resting passengers before spawning so the queue state is current.
        # for s in self.stations:
        #     s.tick_resting(self.time, station_types)

        # Passenger spawning — interval shrinks over time toward spawn_interval_min.
        if self.time - self.last_spawned_at >= self.spawn_interval:
            self.last_spawned_at = self.time
            self.spawn_interval = current_spawn_interval(self.time)
            for i, s in enumerate(self.stations):
                if random.random() < current_spawn_chance(self.time):
                    probs = self.target_type_probs(i)
                    weights = [probs[t] for t in station_types]
                    s.spawn_passenger(station_types, weights=weights, current_time=self.time)
            
        # Move trains and handle passengers
        for line in self.lines:
            for train in line.trains:
                delivered, speed_bonus, delivered_passengers = train.step(self.lines, current_time=self.time)
                reward += delivered + speed_bonus
                self.delivered += delivered
                delivered_passengers_all.extend(delivered_passengers)

        # Overcrowd penalty
        for s in self.stations:
            if len(s.waiting) > max_waiting:
                self.overcrowing_ticks += 1
                if self.overcrowing_ticks > _max_overcrowd_ticks:
                    self.failed = True
                reward -= _overcrowd_penalty
            elif len(s.waiting) > max_waiting / 2:
                reward -= _overcrowd_penalty / 2

        return self._get_state(), reward, self.failed, delivered_passengers_all

    def _get_state(self):
        # Flattened representation for RL agent
        state = []
        for s in self.stations:
            state.append(tuple([s.pos, s.type, len(s.waiting)]))
        return state
    
    def run(self):
        for _ in range(1000):
            self.step()
    


if __name__ == "__main__":

    random.seed(2026)
    sim = MiniMetroSim()
    sim.run()
    print(sim._get_state())

    








