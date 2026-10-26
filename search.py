"""Play-out-verified local search over finished designs.

Start from any complete design (e.g. the agent's), propose small edits — move a train, add
a train, drop / extend / shift a line end, reverse a stretch of a line — score them by real
play-outs on a fixed set of passenger seeds, and keep the best edit if it beats the current
design. Nothing is predicted: every accepted edit was measured to help on this map. This is
how the agent adapts to a particular map; it costs play-outs, not training.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import dataclass

from heuristic import NEIGHBOR_K_DEFAULT, neighbor_sets
from rules import MAX_TRAINS, MAX_TRAINS_PER_LINE, apply_add_train, apply_extend, segments_cross
from simulator import MiniMetroSim

EDITS = ("move_train", "add_train", "drop_end", "extend", "shift_end", "reverse")


@dataclass
class Spec:
    """A design as plain data: station indices per line, and trains per line."""
    lines: list[list[int]]
    trains: list[int]

    def copy(self) -> "Spec":
        return Spec([l[:] for l in self.lines], self.trains[:])


def spec_of(sim: MiniMetroSim) -> Spec:
    return Spec([[sim.stations.index(s) for s in l.stations] for l in sim.lines],
                [len(l.trains) for l in sim.lines])


def build(base: MiniMetroSim, spec: Spec) -> MiniMetroSim:
    """A fresh design on a copy of `base` (a sim with stations and no lines)."""
    sim = copy.deepcopy(base)
    sim.lines = []
    for li, stations in enumerate(spec.lines):
        sim.add_line([stations[0], stations[1]])
        for j in stations[2:]:
            apply_extend(sim, li, -1, j)
    for li, t in enumerate(spec.trains):
        for _ in range(t - 1):
            apply_add_train(sim, li)
    return sim


def valid(spec: Spec, base: MiniMetroSim) -> bool:
    """Every station served, lines of 2+ distinct stations, train limits respected, and no
    two segments crossing (rules.py)."""
    if len({j for l in spec.lines for j in l}) < len(base.stations):
        return False
    if any(len(l) < 2 or len(set(l)) != len(l) for l in spec.lines):
        return False
    if sum(spec.trains) > MAX_TRAINS:
        return False
    if not all(1 <= t <= min(MAX_TRAINS_PER_LINE, len(l) - 1) for l, t in zip(spec.lines, spec.trains)):
        return False
    pos = [s.pos for s in base.stations]
    segs = [(l[i], l[i + 1]) for l in spec.lines for i in range(len(l) - 1)]
    return not any(segments_cross(pos[a], pos[b], pos[c], pos[d])
                   for k, (a, b) in enumerate(segs) for c, d in segs[k + 1:])


def propose(spec: Spec, base: MiniMetroSim, nbrs: list[set[int]], rng: random.Random) -> tuple[Spec, str] | None:
    """One random small edit of `spec`, or None if the drawn edit doesn't apply."""
    new = spec.copy()
    lines, trains = new.lines, new.trains
    edit = rng.choice(EDITS)
    li = rng.randrange(len(lines))
    end = rng.choice((0, -1))
    if edit == "move_train":
        lj = rng.randrange(len(lines))
        if lj == li:
            return None
        trains[li] -= 1
        trains[lj] += 1
    elif edit == "add_train":
        trains[li] += 1
    elif edit == "drop_end":
        if len(lines[li]) < 3:
            return None
        lines[li].pop(end)
        trains[li] = min(trains[li], len(lines[li]) - 1)
    elif edit == "extend":
        options = sorted(j for j in nbrs[lines[li][end]] if j not in lines[li])
        if not options:
            return None
        j = rng.choice(options)
        lines[li].insert(0, j) if end == 0 else lines[li].append(j)
    elif edit == "shift_end":
        lj = rng.randrange(len(lines))
        if lj == li or len(lines[li]) < 3:
            return None
        s = lines[li].pop(end)
        trains[li] = min(trains[li], len(lines[li]) - 1)
        if s in lines[lj]:
            return None
        # attach at whichever end of the receiving line is closer
        pos = base.stations[s].pos
        d_front = pos.get_distance(base.stations[lines[lj][0]].pos)
        d_back = pos.get_distance(base.stations[lines[lj][-1]].pos)
        lines[lj].insert(0, s) if d_front <= d_back else lines[lj].append(s)
    elif edit == "reverse":
        if len(lines[li]) < 4:
            return None
        i, j = sorted(rng.sample(range(len(lines[li])), 2))
        lines[li][i:j + 1] = lines[li][i:j + 1][::-1]
    if new == spec or not valid(new, base):
        return None
    return new, edit


def improve(base: MiniMetroSim, start: MiniMetroSim, score, rounds: int = 10, batch: int = 8,
            min_gain: float = 0.01, rng: random.Random | None = None,
            on_round=None) -> tuple[MiniMetroSim, dict]:
    """Batched hill-climb from the design `start` on the map `base`.

    score(list of sims) -> list of mean ticks survived, all on the same passenger seeds.
    Each round scores `batch` distinct edits of the current design together and keeps the
    best one if it beats the current score by more than `min_gain` (relative).
    on_round(round, rounds, design, stats), if given, is called after each round with the
    current best design; returning False stops the search early.
    Returns (best design, stats).
    """
    rng = rng or random.Random()
    nbrs = neighbor_sets(base, NEIGHBOR_K_DEFAULT)
    current = spec_of(start)
    current_score = start_score = score([build(base, current)])[0]
    accepted = Counter()
    for round_idx in range(rounds):
        proposals: list[tuple[Spec, str]] = []
        for _ in range(batch * 20):  # most draws apply; cap the attempts anyway
            if len(proposals) == batch:
                break
            p = propose(current, base, nbrs, rng)
            if p is not None and all(p[0] != q[0] for q in proposals):
                proposals.append(p)
        if not proposals:
            break
        scores = score([build(base, spec) for spec, _ in proposals])
        best = max(range(len(proposals)), key=scores.__getitem__)
        if scores[best] > current_score * (1 + min_gain):
            current, current_score = proposals[best][0], scores[best]
            accepted[proposals[best][1]] += 1
        if on_round is not None:
            stats = {"start": start_score, "final": current_score, "accepted": dict(accepted)}
            if on_round(round_idx + 1, rounds, build(base, current), stats) is False:
                break
    return build(base, current), {"start": start_score, "final": current_score, "accepted": dict(accepted)}
