"""Local search over finished designs.

Start from any complete design (e.g. the agent's) and propose small edits — move a train,
add a train, drop / extend / shift a line end, reverse a stretch of a line.

improve: score edits by real play-outs on fresh passenger seeds each round, and keep the best
    edit if it beats the current design by more than the play-out noise, so every accepted
    edit was measured to help on this map. A cheap estimate (survival.py) can pre-screen
    which edits get played. This is how the agent adapts to a particular map; it costs
    play-outs, not training.
polish: hill-climb on a cheap estimate alone, with no play-outs.
"""

from __future__ import annotations

import copy
import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

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


def draw_edits(spec: Spec, base: MiniMetroSim, nbrs: list[set[int]], rng: random.Random, n: int,
               attempts: int | None = None) -> list[tuple[Spec, str]]:
    """Up to n distinct edits of `spec` from `attempts` draws (default 20 per edit wanted;
    most draws apply)."""
    edits: list[tuple[Spec, str]] = []
    for _ in range(attempts or n * 20):
        if len(edits) == n:
            break
        p = propose(spec, base, nbrs, rng)
        if p is not None and all(p[0] != q[0] for q in edits):
            edits.append(p)
    return edits


def improve(base: MiniMetroSim, start: MiniMetroSim, score, rounds: int = 10, batch: int = 8,
            n_seeds: int = 4, min_gain: float = 0.01, z: float = 1.0,
            seed_rng: random.Random | None = None, rng: random.Random | None = None,
            on_round=None, record: list | None = None, prescreen=None,
            pool: int = 40) -> tuple[MiniMetroSim, dict]:
    """Batched hill-climb from the design `start` on the map `base`.

    score(sims, seeds) -> array (len(sims), len(seeds)) of ticks survived.

    Each round plays the current design and batch - 1 distinct edits of it on n_seeds fresh
    passenger seeds (from seed_rng), and compares every edit with the current design seed by
    seed, in log ticks. The best edit replaces the current design only if its mean gain is
    more than min_gain (relative) and more than z standard errors, the noise estimated from
    all of the round's edits together. Fresh seeds every round keep the search from fitting
    a few seeds, and re-scoring the current design means a lucky winner is not trusted at
    its lucky score.

    z = 1 is calibrated on real games (16 maps x 10 rounds, each round's best edit re-measured
    on 16 more seeds): the luckiest-of-7 bias is small (+0.007 log), so most best edits are
    real gains and a strict gate mostly rejects them. Taking every best edit over 1% took 58%
    of rounds, 8% of them truly worse; z = 1 takes 47%, 4% worse, for ~4% less total gain;
    z = 2 takes none that are worse but gives up ~10% of the gain.

    Cost: rounds * batch designs, plus the starting design scored once more at the end.
    stats["start"] / stats["final"] are mean ticks of the starting / returned design, each
    on its own fresh seeds, so final / start is an honest (if noisy) estimate of the gain.

    on_round(round, rounds, design, stats), if given, is called after each round with the
    current design; returning False stops the search early.

    record, if given, gets one dict per round (training data for an edit-value model):
    "current" and each of "proposals" as [lines, trains] (+ edit name for proposals),
    "ticks" (current first, then proposals; one row per design, one column per seed),
    and "accepted" (index into proposals, or None).

    prescreen(spec) -> float, if given, is a cheap estimate of a design's value (higher is
    better): each round draws up to `pool` distinct edits and plays only the batch - 1 it
    rates highest, so the play-out cost is unchanged.
    """
    rng = rng or random.Random()
    seed_rng = seed_rng or random.Random(rng.random())
    nbrs = neighbor_sets(base, NEIGHBOR_K_DEFAULT)
    current = spec_of(start)
    accepted = Counter()
    stats = {"start": None, "final": None, "accepted": {}, "rounds": 0}

    def fresh_seeds():
        return [seed_rng.randrange(1_000, 2 ** 31) for _ in range(n_seeds)]

    for round_idx in range(rounds):
        if prescreen is None:
            proposals = draw_edits(current, base, nbrs, rng, batch - 1, attempts=batch * 20)
        else:
            proposals = draw_edits(current, base, nbrs, rng, max(pool, batch - 1))
            proposals = sorted(proposals, key=lambda p: -prescreen(p[0]))[:batch - 1]
        if not proposals:
            break
        ticks = np.asarray(score([build(base, current)] + [build(base, spec) for spec, _ in proposals],
                                 fresh_seeds()), dtype=float)
        if stats["start"] is None:
            stats["start"] = float(ticks[0].mean())
        gains = np.log(ticks[1:]) - np.log(ticks[0])            # (edits, seeds), paired by seed
        mean = gains.mean(axis=1)
        noise = math.sqrt(float(gains.var(axis=1, ddof=1).mean())) if n_seeds > 1 else 0.0
        best = int(np.argmax(mean))
        take = mean[best] > max(math.log1p(min_gain), z * noise / math.sqrt(n_seeds))
        if record is not None:
            record.append({"current": [current.lines, current.trains],
                           "proposals": [[spec.lines, spec.trains, edit] for spec, edit in proposals],
                           "ticks": ticks.tolist(), "accepted": best if take else None})
        if take:
            current = proposals[best][0]
            accepted[proposals[best][1]] += 1
        # provisional; replaced by an unbiased re-measurement after the last round
        stats.update(accepted=dict(accepted), rounds=round_idx + 1,
                     final=float(ticks[1 + best].mean() if take else ticks[0].mean()))
        if on_round is not None and on_round(round_idx + 1, rounds, build(base, current), dict(stats)) is False:
            break

    design = build(base, current)
    if stats["rounds"]:
        stats["final"] = float(np.mean(score([design], fresh_seeds())[0]))
    return design, stats


def polish(base: MiniMetroSim, start: MiniMetroSim, value, rounds: int = 30, pool: int = 40,
           rng: random.Random | None = None, on_round=None) -> tuple[MiniMetroSim, dict]:
    """Hill-climb from `start` on value(spec) -> float (higher is better) alone: each round
    draws up to `pool` edits and keeps the best if it beats the current design. No play-outs,
    so nothing here is measured; with survival.SurvivalModel(base).log_ticks as `value` it
    gained x1.70 over the heuristic on 48 bench maps (play-out search, 320 games: x1.58).

    stats["start"] / stats["final"] are value() of the starting / returned design.
    on_round works as in improve."""
    rng = rng or random.Random()
    nbrs = neighbor_sets(base, NEIGHBOR_K_DEFAULT)
    current = spec_of(start)
    best = value(current)
    accepted = Counter()
    stats = {"start": best, "final": best, "accepted": {}, "rounds": 0}
    for round_idx in range(rounds):
        edits = draw_edits(current, base, nbrs, rng, pool)
        if not edits:
            break
        values = [value(spec) for spec, _ in edits]
        i = int(np.argmax(values))
        if values[i] > best + 1e-9:
            current, best = edits[i][0], values[i]
            accepted[edits[i][1]] += 1
        stats.update(final=best, accepted=dict(accepted), rounds=round_idx + 1)
        if on_round is not None and on_round(round_idx + 1, rounds, build(base, current), dict(stats)) is False:
            break
    return build(base, current), stats
