"""Learning to build like a teacher.

A teacher (polish, or polish then search) turns the heuristic's design into a better one on
each training map. The student then builds that map step by step. At each step, every move
that keeps the teacher's design reachable counts as right, and the student learns to score
the right moves above the others. The student picks among the right moves itself, so it
learns the teacher's design, not one fixed order of building it.
"""

import json
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Batch

from minimetro.config import MAX_LINES
from minimetro.design.agent import Agent
from minimetro.design.heuristic import prior_terms
from minimetro.design.search import Spec
from minimetro.game.rules import NOOP, ActionType, is_legal
from minimetro.game.simulator import MiniMetroSim
from minimetro.learning.model import IDENTITY_AUG, action_aux_features, action_station_indices, build_graph


def _place(built: list[int], target: list[int]) -> int | None:
    """Where `built` sits inside `target` as a contiguous stretch, or None."""
    m = len(built)
    for s in range(len(target) - m + 1):
        if target[s:s + m] == built:
            return s
    return None


def match_lines(sim: MiniMetroSim, target: Spec) -> dict[int, tuple[int, list[int], int]] | None:
    """For each built line: (target line index, its stations oriented the built line's way,
    where the built line starts in them). None if some built line is part of no target line."""
    index = {id(s): i for i, s in enumerate(sim.stations)}
    used, matched = set(), {}
    for li, line in enumerate(sim.lines):
        built = [index[id(s)] for s in line.stations]
        for ti, stations in enumerate(target.lines):
            if ti in used:
                continue
            for oriented in (stations, stations[::-1]):
                start = _place(built, oriented)
                if start is not None:
                    matched[li] = (ti, oriented, start)
                    used.add(ti)
                    break
            if li in matched:
                break
        if li not in matched:
            return None
    return matched


def right_moves(sim: MiniMetroSim, target: Spec, phase: str) -> list[tuple]:
    """The legal moves that keep `target` reachable from `sim` ([] once it isn't)."""
    matched = match_lines(sim, target)
    if matched is None:
        return []
    moves = []
    if phase == "trains":
        moves = [(ActionType.ADD_TRAIN, li) for li, (ti, _, _) in matched.items()
                 if len(sim.lines[li].trains) < target.trains[ti]]
    else:
        for li, (ti, oriented, start) in matched.items():
            end = start + len(sim.lines[li].stations)
            if start > 0:
                moves.append((ActionType.EXTEND_LINE, li, 0, oriented[start - 1]))
            if end < len(oriented):
                moves.append((ActionType.EXTEND_LINE, li, -1, oriented[end]))
        started = {ti for ti, _, _ in matched.values()}
        if len(sim.lines) < MAX_LINES:
            for ti, stations in enumerate(target.lines):
                if ti not in started:
                    moves += [(ActionType.NEW_LINE, min(a, b), max(a, b)) for a, b in zip(stations, stations[1:])]
    moves = list(dict.fromkeys(a for a in moves if is_legal(sim, a)))
    return moves or ([NOOP] if _complete(sim, target, matched, phase) else [])


def _complete(sim, target, matched, phase) -> bool:
    if len(matched) != len(target.lines):
        return False
    if phase == "trains":
        return all(len(sim.lines[li].trains) >= target.trains[ti] for li, (ti, _, _) in matched.items())
    return all(len(sim.lines[li].stations) == len(target.lines[ti]) for li, (ti, _, _) in matched.items())


@dataclass
class Step:
    """One decision to learn from: the state, the moves scored, which are right, and each
    move's prior terms (scaled, before the agent's learned weights)."""
    data: object
    globals_: torch.Tensor
    stations: list[tuple[int, int]]
    aux: torch.Tensor
    prior: torch.Tensor
    right: torch.Tensor          # bool per move


def walk(agent: Agent, sim0: MiniMetroSim, target: Spec) -> tuple[list[Step], dict]:
    """Build `target` with the student choosing among the right moves (one from outside its
    candidates when none is among them); return the steps and
    stats: steps on the path, whether the student's own pick was right, and whether a right
    move was among its candidates (it can only reproduce the teacher if so)."""
    steps, stats = [], {"steps": 0, "agreed": 0, "covered": 0, "lost": 0}

    def guide(sim, phase, candidates):
        right = right_moves(sim, target, phase)
        if not right:
            stats["lost"] += 1
            return None
        prior, q = agent.prior_and_q(sim, phase, right + [a for a in candidates if a not in right])
        score = {a: prior[a] + q[a] for a in prior}
        own = max(candidates, key=score.get)
        covered = any(a in candidates for a in right)
        stats["steps"] += 1
        stats["agreed"] += own in right
        stats["covered"] += covered
        # Learn only from the moves the student can actually make: a right move added from
        # outside its candidates would be right by construction, an easy tell.
        moves = candidates
        if covered and len(moves) > 1:
            data, globals_ = build_graph(sim, phase, agent.aug)
            steps.append(Step(data, globals_, [action_station_indices(sim, a) for a in moves],
                              torch.stack([action_aux_features(sim, a) for a in moves]),
                              torch.tensor([prior_terms(sim, a) for a in moves], dtype=torch.float32) * agent.prior_scale,
                              torch.tensor([a in right for a in moves])))
        return own if own in right else max(right, key=score.get)

    agent.design(sim0, guide=guide)
    return steps, stats


class Imitator:
    """Trains the agent so that its reweighted prior + Q ranks the teacher's moves first: a
    softmax over each step's moves (temperature tau), maximizing the probability of the right
    ones. Learning a weight per prior term lets the student trust one term (say, bends) less
    without discarding the rest."""

    def __init__(self, agent: Agent, lr: float = 3e-4, tau: float = 0.1, augment: bool = True,
                 seed: int | None = None):
        self.agent, self.tau, self.augment = agent, tau, augment
        self.rng = random.Random(seed)
        self.optimizer = optim.Adam([
            {"params": list(agent.state_encoder.parameters()) + list(agent.q_net.parameters())},
            {"params": [agent.log_term_weights], "lr": 1e-2},    # a few numbers: let them move faster
        ], lr=lr)

    def learn_map(self, sim0: MiniMetroSim, target: Spec) -> dict:
        """Walk one map and take one gradient step on all its decisions."""
        if self.augment:
            self.agent.aug = (self.rng.choice((1.0, -1.0)), self.rng.choice((1.0, -1.0)), self.rng.random() < 0.5)
        try:
            steps, stats = walk(self.agent, sim0, target)
        finally:
            self.agent.aug = IDENTITY_AUG
        stats["loss"] = self.update(steps) if steps else float("nan")
        return stats

    def update(self, steps: list[Step]) -> float:
        agent = self.agent
        graphs = Batch.from_data_list([s.data for s in steps])
        node_embeds, g = agent.state_encoder(graphs, torch.stack([s.globals_ for s in steps]))
        offsets = graphs.ptr[:-1].tolist()
        zero = torch.zeros(node_embeds.shape[1])
        losses = []
        for k, step in enumerate(steps):
            h_p = torch.stack([node_embeds[offsets[k] + p] if p >= 0 else zero for p, _ in step.stations])
            h_s = torch.stack([node_embeds[offsets[k] + s] if s >= 0 else zero for _, s in step.stations])
            q = agent.q_net(h_p, h_s, g[k].unsqueeze(0).expand(len(step.stations), -1), step.aux)
            logits = ((step.prior * agent.log_term_weights.exp()).sum(-1) + q) / self.tau
            losses.append(torch.logsumexp(logits, 0) - torch.logsumexp(logits[step.right], 0))
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


def accuracy(stats: list[dict]) -> dict:
    """Share of path decisions where the student's pick was right / a right move was a candidate."""
    steps = max(sum(s["steps"] for s in stats), 1)
    return {"agreed": sum(s["agreed"] for s in stats) / steps, "covered": sum(s["covered"] for s in stats) / steps,
            "lost": float(np.mean([s["lost"] > 0 for s in stats]))}


def load_teacher(path: str) -> list[dict]:
    """Teacher designs written by scripts/teach.py, one dict per map."""
    with open(path) as f:
        return [json.loads(line) for line in f]
