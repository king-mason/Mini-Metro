import heapq
import random
from collections import deque
from point import Point
from enum import Enum, auto

import yaml as _yaml
with open("config.yaml") as _f:
    _cfg = _yaml.safe_load(_f)
_TRAIN_SPEED = _cfg["trains"]["speed"] * _cfg.get("time_scale", 1)
_TIME_SCALE = _cfg.get("time_scale", 1)
_REST_DURATION_MEAN = max(1, _cfg["passengers"]["rest_duration"] // _TIME_SCALE)
_REST_DURATION_SD = max(1, _cfg["passengers"]["rest_duration_randomness"] // _TIME_SCALE)


class StationType(Enum):
    CIRCLE = auto()
    TRIANGLE = auto()
    SQUARE = auto()
    CROSS = auto()
    DIAMOND = auto()
    PENTAGON = auto()
    RHOMBUS = auto()
    DROPLET = auto()
    STAR = auto()
    LEMON = auto()


class Passenger:
    def __init__(self, target_type: StationType, journey_start_time: int = 0):
        self.target_type = target_type
        self.journey_start_time = journey_start_time
        self.lines_ridden = set()
        self.trains_ridden = set()
        self.rest_until: int = -1  # tick when passenger re-enters waiting; -1 = active
        # Cached shortest-path distances to nearest target-type station, computed lazily
        # on first boarding decision. Maps id(station) → distance. {} means not computed;
        # {} after _plan_route() means target unreachable.
        self._dist_to_target: dict | None = None

    def reset_for_new_journey(self, new_target: StationType, current_time: int):
        """Called when passenger re-emerges from rest with a new destination."""
        self.target_type = new_target
        self.journey_start_time = current_time
        self.lines_ridden = set()
        self.trains_ridden = set()
        self.rest_until = -1
        self._dist_to_target = None  # invalidate route for new target

    def _plan_route(self, all_lines: list):
        """Compute and cache, per journey:
          - dist_to_target[id(s)]: shortest geometric distance from s to nearest target-type station.
          - expected_wait[id(s)]:  estimated wait *in distance units* for the optimal first-move
                                   train at s, computed as line_length / num_trains (half-period).
        Train speed is constant across the sim, so comparing distances ≡ comparing times.
        """
        target = self.target_type
        adj: dict = {}
        targets_seen = set()
        targets = []
        for line in all_lines:
            for s in line.stations:
                if s.type == target and id(s) not in targets_seen:
                    targets_seen.add(id(s))
                    targets.append(s)
            for i in range(len(line.stations) - 1):
                a, b = line.stations[i], line.stations[i + 1]
                d = a.pos.get_distance(b.pos)
                adj.setdefault(id(a), []).append((id(b), d))
                adj.setdefault(id(b), []).append((id(a), d))

        dist = {id(t): 0.0 for t in targets}
        pq = [(0.0, id(t)) for t in targets]
        heapq.heapify(pq)
        while pq:
            d, sid = heapq.heappop(pq)
            if d > dist.get(sid, float("inf")):
                continue
            for nbid, edge_d in adj.get(sid, []):
                nd = d + edge_d
                if nd < dist.get(nbid, float("inf")):
                    dist[nbid] = nd
                    heapq.heappush(pq, (nd, nbid))
        self._dist_to_target = dist

        # Expected wait at each station = avg distance until the next train arrives in
        # the optimal direction. With N trains shuttling on a line of length L, the
        # average wait at any point is L / N (half a round-trip period).
        wait: dict = {}
        for line in all_lines:
            if not line.trains or len(line.stations) < 2:
                continue
            line_length = sum(
                line.stations[i].pos.get_distance(line.stations[i + 1].pos)
                for i in range(len(line.stations) - 1)
            )
            avg_wait = line_length / len(line.trains)
            for i, s in enumerate(line.stations):
                cost_s = dist.get(id(s), float("inf"))
                if cost_s == float("inf"):
                    continue
                # Is (line, direction) an optimal first move from s? Check both directions.
                for direction in (1, -1):
                    nb_pos = i + direction
                    if 0 <= nb_pos < len(line.stations):
                        nb = line.stations[nb_pos]
                        edge = s.pos.get_distance(nb.pos)
                        cost_nb = dist.get(id(nb), float("inf"))
                        if abs((edge + cost_nb) - cost_s) < 1e-6:
                            # This line is one of the optimal options at s.
                            wait[id(s)] = min(wait.get(id(s), float("inf")), avg_wait)
                            break
        self._expected_wait = wait

    def is_train_useful(self, current_station, next_station, all_lines: list) -> bool:
        """True if boarding this train is the fastest option from `current_station`.

        Always board if the train is on a shortest path (extra travel = 0).
        Board a suboptimal train if the extra travel distance is less than the
        expected wait for the optimal alternative.
        """
        if self._dist_to_target is None:
            self._plan_route(all_lines)
        cost_cur = self._dist_to_target.get(id(current_station), float("inf"))
        if cost_cur == float("inf"):
            return False
        cost_next = self._dist_to_target.get(id(next_station), float("inf"))
        edge = current_station.pos.get_distance(next_station.pos)
        extra = (edge + cost_next) - cost_cur
        if extra <= 1e-6:
            return True  # on a shortest path — always take it
        if cost_next >= cost_cur:
            return False  # moving away from target — never board
        # Makes net progress but not optimal — accept if extra travel < expected wait.
        return extra <= self._expected_wait.get(id(current_station), float("inf"))

    def __str__(self):
        return str(self.target_type)

    def __repr__(self):
        return str(self.target_type)


class Station:
    def __init__(self, stype: StationType, pos: Point):
        self.type = stype
        self.pos = pos
        self.x = pos.x
        self.y = pos.y
        self.waiting: deque = deque()
        self.resting: deque = deque()  # passengers serving their rest period after delivery
        self.overcrowding_ticks = 0

    def spawn_passenger(self, station_type_options: list, weights: list | None = None, current_time: int = 0):
        other_type_options = []
        if weights is None:
            weights = [1 for _ in range(len(station_type_options))]
        new_weights = []
        for t, w in zip(station_type_options, weights):
            if t != self.type:
                other_type_options.append(t)
                new_weights.append(w)
        if not other_type_options:
            return
        target = random.choices(other_type_options, new_weights)[0]
        self.waiting.append(Passenger(target, journey_start_time=current_time))

    def tick_resting(self, current_time: int, station_options: list):
        """Move passengers whose rest period has ended back into the waiting queue."""
        still_resting = deque()
        for p in self.resting:
            if p.rest_until <= current_time:
                options = [t for t in station_options if t != self.type]
                if options:
                    p.reset_for_new_journey(random.choice(options), current_time)
                    self.waiting.append(p)
                # If no valid target (only one station type), passenger stays resting
                # until the map has more diversity — rare but safe to skip.
            else:
                still_resting.append(p)
        self.resting = still_resting

    def __str__(self):
        return f"Station ({self.type}) @ {self.pos}"

    def __repr__(self):
        return f"Station ({self.type}) @ {self.pos}"


class Line:
    def __init__(self, line_index: int, stations: list[Station]):
        self.index = line_index
        self.stations: list[Station] = stations
        self.trains: list[Train] = []

    def add_train(self, train):
        self.trains.append(train)
        train.line = self
    
    def __str__(self):
        return f"Line {self.index}"

    def __repr__(self):
        return f"<Line {self.index}>"


class Train:
    def __init__(self, line: Line, capacity=6, speed=_TRAIN_SPEED):
        self.line = line
        self.capacity = capacity
        self.passengers = []
        self.position = 0
        self.direction = 1
        self.progress = 0.0
        self.speed = speed
        self._at_station = True

    def move(self) -> bool:
        """Advance toward the next station. Returns True when a new station is reached."""
        if len(self.line.stations) < 2:
            return False
        from_s = self.line.stations[self.position]
        to_idx = self.position + self.direction
        to_s = self.line.stations[to_idx]
        dist = from_s.pos.get_distance(to_s.pos)
        self.progress += self.speed / max(dist, 1.0)
        if self.progress >= 1.0:
            self.progress = 0.0
            self.position = to_idx
            if self.position >= len(self.line.stations) - 1:
                self.direction = -1
            elif self.position <= 0:
                self.direction = 1
            return True
        return False

    def move_to(self, station_idx, direction):
        self.position = station_idx
        self.direction = direction
        self.progress = 0.0
        self._at_station = True

    def board_passenger(self, passenger):
        self.passengers.append(passenger)

    SPEED_BONUS_CAP = 300

    def stop_at_station(self, all_lines: list[Line], current_time: int = 0):
        station = self.line.stations[self.position]

        delivered_passengers: list[Passenger] = []
        speed_bonus = 0.0
        remaining = []
        transfers = []
        # The station this train will visit next (after the current stop).
        next_pos = self.position + self.direction
        n = len(self.line.stations)
        next_train_station = self.line.stations[next_pos] if 0 <= next_pos < n else None

        # Disembark passengers whose route disagrees with this train's next stop.
        for p in self.passengers:
            if p.target_type == station.type:
                delivered_passengers.append(p)
                wait = current_time - p.journey_start_time
                speed_bonus += max(0.0, 1.0 - wait / self.SPEED_BONUS_CAP)
            elif (next_train_station is not None
                  and p.is_train_useful(station, next_train_station, all_lines)):
                remaining.append(p)  # train continues toward target — stay on
            else:
                transfers.append(p)  # transfer off, wait for a better train
        self.passengers = remaining

        # Boarding: only board if this train's next stop is on the passenger's
        # shortest path to target.
        remaining_waiting = deque()
        for p in station.waiting:
            if (len(self.passengers) < self.capacity
                    and next_train_station is not None
                    and p.is_train_useful(station, next_train_station, all_lines)):
                p.lines_ridden.add(self.line)
                p.trains_ridden.add(self)
                self.passengers.append(p)
            else:
                remaining_waiting.append(p)
        station.waiting = remaining_waiting

        for p in transfers:
            station.waiting.append(p)

        return len(delivered_passengers), speed_bonus, delivered_passengers        

    def step(self, all_lines: list, current_time: int = 0):
        if self._at_station:
            self._at_station = False
            return self.stop_at_station(all_lines, current_time)
        arrived = self.move()
        if arrived:
            self._at_station = True
        return 0, 0.0, []

    def __str__(self):
        return f"<Train on {self.line} @ Station {self.position} with Direction {self.direction}>"
