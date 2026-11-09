"""How long will a finished design survive? An analytic estimate, no play-outs (~5 ms).

Passengers spawn at every station at the game's rate, which ramps up over time. Each
passenger follows the game's own rules (objects.Passenger): they wait for the first train
whose next stop gets them closer, and stay aboard while the train keeps getting them closer.
Following those rules gives the riders per unit spawn rate who board and ride each train at each
station. From those:

  - between trains, a station's queue grows like a sawtooth whose peak is the arrival rate
    times the gap between useful trains (round trip / trains on the line);
  - riders a full train leaves behind pile up for good, as do riders with no route at all.

The game is lost once station-ticks over capacity pass max_overcrowd_ticks; the estimate
adds up the expected overcrowded time as spawning ramps up and returns the tick it runs out.

On 500 training maps it ranked the designs a search tries by real survival with spearman
0.93, and the edits within one search round at 0.77 (the measured noise ceiling is ~0.95),
with the best edit in its top 3 88% of the time (measured before simulator.step drew
targets by plain type share; then a type's odds went as its station count squared). The
absolute ticks are only roughly right; use it to compare designs on one map.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

import numpy as np

from heuristic import TRAIN_CAPACITY, TRAIN_SPEED
from simulator import (MiniMetroSim, current_spawn_chance, current_spawn_interval, max_overcrowd_ticks,
                       max_waiting)

CAP_TICKS = 6000
SPAWN_RATE = np.array([current_spawn_chance(t) / current_spawn_interval(t) for t in range(1, CAP_TICKS + 1)])
CUM_SPAWNED = np.cumsum(SPAWN_RATE)


def target_probs(sim: MiniMetroSim, i: int) -> dict:
    """The chance that a passenger spawned at station i wants each station type, as
    MiniMetroSim.step draws it (target_type_probs), without the types it never picks."""
    return {t: p for t, p in sim.target_type_probs(i).items() if p}


def segment_ticks(dist: float) -> int:
    """Ticks to travel one segment plus the stop at its end, with objects.Train.move's own
    float sums: a segment of exactly k steps (420 px at speed 30) can need k + 1."""
    ticks, progress = 0, 0.0
    while progress < 1.0:
        progress += TRAIN_SPEED / max(dist, 1.0)
        ticks += 1
    return ticks + 1


class SurvivalModel:
    """Survival estimates for designs on one map (`base`: the map, lines not needed)."""

    def __init__(self, base: MiniMetroSim):
        self.n = len(base.stations)
        self.types = [s.type for s in base.stations]
        self.probs = [target_probs(base, i) for i in range(self.n)]
        pos = np.array([(s.pos.x, s.pos.y) for s in base.stations], dtype=float)
        self.dist = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))
        self.seg_ticks = [[segment_ticks(d) for d in row] for row in self.dist]

    def log_ticks(self, spec) -> float:
        """log of the estimated ticks survived by a search.Spec (higher is better)."""
        return math.log(self.analyze(spec.lines, spec.trains)["ticks"])

    def flows(self, lines: list[list[int]], trains: list[int]) -> dict:
        """Riders per unit spawn rate for the design (station indices per line, trains per line):
        "ride": {(station, line, direction): riders leaving the station on that line's trains},
        "board": the same for riders boarding there, "queues": {(station, frozenset of useful
        (line, direction)): riders waiting there for one of those trains}, "unreachable" per
        station, "transfers" in total, and "gap": ticks between a line's trains."""
        n, D = self.n, self.dist
        adj = [set() for _ in range(n)]
        where = [dict() for _ in range(n)]           # station -> {line: index on the line}
        for li, l in enumerate(lines):
            for k, j in enumerate(l):
                where[j][li] = k
            for a, b in zip(l, l[1:]):
                adj[a].add(b); adj[b].add(a)
        length = [sum(D[a, b] for a, b in zip(l, l[1:])) for l in lines]
        gap = [2 * sum(self.seg_ticks[a][b] for a, b in zip(l, l[1:])) / t for l, t in zip(lines, trains)]
        freq = [1 / g for g in gap]

        def next_stop(li, k, dr):
            return lines[li][k + dr] if 0 <= k + dr < len(lines[li]) else None

        ride, board, queues = defaultdict(float), defaultdict(float), defaultdict(float)
        unreachable, transfers = np.zeros(n), 0.0
        for target in set(self.types):
            # Passenger._plan_route: shortest distance along the lines to the nearest target ...
            dist = [0.0 if self.types[i] == target else math.inf for i in range(n)]
            heap = [(0.0, i) for i in range(n) if dist[i] == 0]
            heapq.heapify(heap)
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                for v in adj[u]:
                    if d + D[u, v] < dist[v] - 1e-9:
                        dist[v] = d + D[u, v]
                        heapq.heappush(heap, (dist[v], v))
            # ... and the expected wait at each station: L / N of the lines with an optimal move there
            wait = [math.inf] * n
            for s in (j for j in range(n) if dist[j] < math.inf):
                for li, k in where[s].items():
                    if any(nb is not None and abs(D[s, nb] + dist[nb] - dist[s]) < 1e-6
                           for nb in (next_stop(li, k, 1), next_stop(li, k, -1))):
                        wait[s] = min(wait[s], length[li] / trains[li])

            def useful(s, li, dr):
                """Passenger.is_train_useful for a train on line li leaving s in direction dr."""
                nb = next_stop(li, where[s][li], dr)
                if nb is None:
                    return False
                extra = D[s, nb] + dist[nb] - dist[s]
                return extra <= 1e-6 or (dist[nb] < dist[s] and extra <= wait[s])

            waiting = np.zeros(n)
            onboard = defaultdict(float)             # (station, line, direction arrived in) -> riders
            for i in range(n):
                p = self.probs[i].get(target, 0.0)
                if p and self.types[i] != target:
                    if dist[i] == math.inf:
                        unreachable[i] += p
                    else:
                        waiting[i] += p
            # every useful hop lowers dist, so far-to-near order has seen all of a station's inflow
            for s in sorted((j for j in range(n) if 0 < dist[j] < math.inf), key=lambda j: -dist[j]):
                for li, k in where[s].items():
                    for dr in (1, -1):
                        r = onboard.pop((s, li, dr), 0.0)
                        if not r:
                            continue
                        dr = dr if next_stop(li, k, dr) is not None else -dr    # trains turn at the ends
                        if useful(s, li, dr):
                            ride[(s, li, dr)] += r
                            onboard[(next_stop(li, k, dr), li, dr)] += r
                        else:
                            waiting[s] += r
                            transfers += r
                if not waiting[s]:
                    continue
                options = [(li, dr) for li in where[s] for dr in (1, -1) if useful(s, li, dr)]
                if not options:
                    unreachable[s] += waiting[s]
                    continue
                # waiting riders take the first useful train, so they split by train frequency
                queues[(s, frozenset(options))] += waiting[s]
                total = sum(freq[li] for li, _ in options)
                for li, dr in options:
                    share = waiting[s] * freq[li] / total
                    board[(s, li, dr)] += share
                    ride[(s, li, dr)] += share
                    onboard[(next_stop(li, where[s][li], dr), li, dr)] += share
        return {"ride": dict(ride), "board": dict(board), "queues": dict(queues),
                "unreachable": unreachable, "transfers": transfers, "gap": gap}

    def analyze(self, lines: list[list[int]], trains: list[int]) -> dict:
        """Estimated "ticks" survived (capped at CAP_TICKS), plus what limits it: "peak", the
        largest sawtooth peak per unit spawn rate; "max_util", the busiest train's riders per seat
        per unit spawn rate; "unreachable" and "transfers", riders per unit spawn rate."""
        n = self.n
        f = self.flows(lines, trains)
        gap, freq = f["gap"], [1 / g for g in f["gap"]]

        # sawtooth peak per station: each group of riders waiting for the same trains is cleared
        # by the first of them; different groups' peaks don't line up, so the largest counts in
        # full and the rest at their mean (half)
        groups = defaultdict(list)
        for (s, options), q in f["queues"].items():
            groups[s].append(q / sum(freq[li] for li, _ in options))
        peak = np.zeros(n)
        for s, sizes in groups.items():
            sizes.sort(reverse=True)
            peak[s] = sizes[0] + 0.5 * sum(sizes[1:])

        # seats: riders on (line, direction) at s want rate * ride of the 6 / gap seats per tick;
        # riders already aboard keep their seats, so a shortfall falls on those boarding at s
        keys = list(f["ride"])
        station = np.array([k[0] for k in keys], dtype=int)
        ride = np.array([f["ride"][k] for k in keys])
        board = np.array([f["board"].get(k, 0.0) for k in keys])
        seats = np.array([TRAIN_CAPACITY / gap[k[1]] for k in keys])
        short = np.clip(SPAWN_RATE[:, None] * ride[None] - seats[None], 0, None) * (board / ride)[None]
        left_behind = short @ (station[:, None] == np.arange(n)[None])          # (ticks, stations)

        stuck = CUM_SPAWNED[:, None] * f["unreachable"][None] + np.cumsum(left_behind, axis=0)
        room = max_waiting - stuck
        top = SPAWN_RATE[:, None] * peak[None]
        # share of each tick a station spends over capacity, with the sawtooth's queue uniform on [0, top]
        over = np.where(room <= 0, 1.0, np.clip(1 - room / np.maximum(top, 1e-9), 0, 1))
        ticks = min(int(np.searchsorted(np.cumsum(over.sum(1)), max_overcrowd_ticks)) + 1, CAP_TICKS)
        return {"ticks": float(ticks), "peak": float(peak.max()),
                "max_util": float((ride / seats).max()) if keys else 0.0,
                "unreachable": float(f["unreachable"].sum()), "transfers": float(f["transfers"])}
