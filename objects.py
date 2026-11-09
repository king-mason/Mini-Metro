"""Stations, lines, trains and passengers, and the rules passengers ride by."""

import heapq
from collections import deque
from enum import Enum, auto

from config import TRAIN_CAPACITY, TRAIN_SPEED
from point import Point

INF = float("inf")


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
    """Wants to reach any station of `target_type`. Routes are planned on first use and kept
    for the journey (the design doesn't change during play)."""

    def __init__(self, target_type: StationType):
        self.target_type = target_type
        self._dist_to_target: dict | None = None   # id(station) -> distance along lines
        self._expected_wait: dict = {}             # id(station) -> wait for its best train, in distance

    def _plan_route(self, lines: list["Line"]):
        """Shortest distance along the lines from every station to the nearest target, and the
        expected wait at each station for a train on a shortest path (line length / trains,
        in distance units: speed is the same everywhere, so distance stands in for time)."""
        adj: dict = {}
        targets = {}
        for line in lines:
            for s in line.stations:
                if s.type == self.target_type:
                    targets[id(s)] = s
            for a, b in zip(line.stations, line.stations[1:]):
                d = a.pos.get_distance(b.pos)
                adj.setdefault(id(a), []).append((id(b), d))
                adj.setdefault(id(b), []).append((id(a), d))

        dist = {sid: 0.0 for sid in targets}
        heap = [(0.0, sid) for sid in targets]
        heapq.heapify(heap)
        while heap:
            d, sid = heapq.heappop(heap)
            if d > dist.get(sid, INF):
                continue
            for nb, edge in adj.get(sid, []):
                if d + edge < dist.get(nb, INF):
                    dist[nb] = d + edge
                    heapq.heappush(heap, (d + edge, nb))
        self._dist_to_target = dist

        wait: dict = {}
        for line in lines:
            if not line.trains or len(line.stations) < 2:
                continue
            avg_wait = line.length() / len(line.trains)
            for i, s in enumerate(line.stations):
                cost = dist.get(id(s), INF)
                if cost == INF:
                    continue
                for nb_pos in (i + 1, i - 1):
                    if 0 <= nb_pos < len(line.stations):
                        nb = line.stations[nb_pos]
                        if abs(s.pos.get_distance(nb.pos) + dist.get(id(nb), INF) - cost) < 1e-6:
                            wait[id(s)] = min(wait.get(id(s), INF), avg_wait)
                            break
        self._expected_wait = wait

    def is_train_useful(self, current: "Station", next_stop: "Station", lines: list["Line"]) -> bool:
        """Board (or stay on) a train heading to `next_stop`? Yes if it's on a shortest path;
        also if it gets closer and the detour is shorter than the wait for a better train."""
        if self._dist_to_target is None:
            self._plan_route(lines)
        cost_here = self._dist_to_target.get(id(current), INF)
        if cost_here == INF:
            return False
        cost_next = self._dist_to_target.get(id(next_stop), INF)
        extra = current.pos.get_distance(next_stop.pos) + cost_next - cost_here
        if extra <= 1e-6:
            return True
        if cost_next >= cost_here:
            return False
        return extra <= self._expected_wait.get(id(current), INF)

    def __repr__(self):
        return str(self.target_type)


class Station:
    def __init__(self, stype: StationType, pos: Point):
        self.type = stype
        self.pos = pos
        self.waiting: deque[Passenger] = deque()

    @property
    def x(self) -> int:
        return self.pos.x

    @property
    def y(self) -> int:
        return self.pos.y

    def __repr__(self):
        return f"Station({self.type.name}, {self.pos})"


class Line:
    def __init__(self, index: int, stations: list[Station]):
        self.index = index
        self.stations = stations
        self.trains: list[Train] = []

    def add_train(self, train: "Train"):
        self.trains.append(train)
        train.line = self

    def length(self) -> float:
        """One-way length along the stations."""
        return sum(a.pos.get_distance(b.pos) for a, b in zip(self.stations, self.stations[1:]))

    def round_trip_ticks(self) -> int:
        """Ticks for one train to go end to end and back, stops included."""
        if len(self.stations) < 2:
            return 0
        probe = Train(self)
        ticks = 0
        while True:
            probe.advance(1)
            ticks += 1
            if probe.position == 0 and probe.at_station:
                return ticks

    def space_trains(self):
        """Spread the trains evenly in time around the round trip, so every station sees a
        train every round_trip / trains ticks in each direction. Design time only."""
        period = self.round_trip_ticks()
        for k, train in enumerate(self.trains):
            train.move_to(0, 1)
            train.advance(round(k * period / len(self.trains)))

    def __repr__(self):
        return f"<Line {self.index}>"


class Train:
    """Shuttles end to end. Each stop takes one tick, then it travels at `speed`."""

    def __init__(self, line: Line, capacity: int = TRAIN_CAPACITY, speed: float = TRAIN_SPEED):
        self.line = line
        self.capacity = capacity
        self.speed = speed
        self.passengers: list[Passenger] = []
        self.position = 0          # index of the station it is at or last left
        self.direction = 1
        self.progress = 0.0        # fraction of the current segment travelled
        self.at_station = True

    def next_stop(self) -> Station | None:
        """The station after the current one (None at an end, before turning)."""
        k = self.position + self.direction
        return self.line.stations[k] if 0 <= k < len(self.line.stations) else None

    def move(self) -> bool:
        """Travel one tick. Returns True on reaching the next station (turning at the ends)."""
        stations = self.line.stations
        if len(stations) < 2:
            return False
        to_idx = self.position + self.direction
        dist = stations[self.position].pos.get_distance(stations[to_idx].pos)
        self.progress += self.speed / max(dist, 1.0)
        if self.progress < 1.0:
            return False
        self.progress = 0.0
        self.position = to_idx
        if self.position >= len(stations) - 1:
            self.direction = -1
        elif self.position <= 0:
            self.direction = 1
        return True

    def move_to(self, station_idx: int, direction: int):
        self.position, self.direction, self.progress, self.at_station = station_idx, direction, 0.0, True

    def advance(self, ticks: int):
        """Move for `ticks` ticks without serving passengers (same timing as step)."""
        for _ in range(ticks):
            if self.at_station:
                self.at_station = False
            elif self.move():
                self.at_station = True

    def stop_at_station(self, lines: list[Line]) -> int:
        """Let passengers off and on. Returns how many reached their destination."""
        station = self.line.stations[self.position]
        next_stop = self.next_stop()

        def useful(p):
            return next_stop is not None and p.is_train_useful(station, next_stop, lines)

        delivered, staying, transfers = 0, [], []
        for p in self.passengers:
            if p.target_type == station.type:
                delivered += 1
            elif useful(p):
                staying.append(p)
            else:
                transfers.append(p)
        self.passengers = staying

        still_waiting = deque()
        for p in station.waiting:
            if len(self.passengers) < self.capacity and useful(p):
                self.passengers.append(p)
            else:
                still_waiting.append(p)
        station.waiting = still_waiting
        station.waiting.extend(transfers)
        return delivered

    def step(self, lines: list[Line]) -> int:
        """One tick: stop (if at a station) or travel. Returns passengers delivered."""
        if self.at_station:
            self.at_station = False
            return self.stop_at_station(lines)
        if self.move():
            self.at_station = True
        return 0

    def __repr__(self):
        return f"<Train on {self.line} at station {self.position}, direction {self.direction}>"
