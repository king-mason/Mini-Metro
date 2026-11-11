"""Learning Q by branch-and-compare.

Build the greedy design on a map, then edit it by swapping one decision for an alternative
while keeping the rest. Play both on the same passenger seeds: the log-ratio of survival
is the target for score(alternative) - score(chosen). The map's difficulty cancels out, and
each target is about one decision.
"""

import math
import os
import pickle
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch

from agent import Agent, DesignResult
from maps import LEVELS
from model import IDENTITY_AUG, action_aux_features, action_station_indices, build_graph
from simulator import MiniMetroSim


@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 32
    memory_size: int = 10_000
    branch_rounds: int = 2           # rounds per episode
    num_branches: int = 6            # single-decision edits per round
    accept_margin: float = 0.08      # log gain for a branch to become the incumbent (~ noise at 4 seeds)
    branch_window: float = 2.0       # alternatives must score within this of the best candidate
    branch_uniform: float = 0.3      # chance to pick the alternative uniformly, else by softmax
    branch_temperature: float = 0.25
    replays_per_episode: int = 4
    augment: bool = True             # encode each training map under a random symmetry
    level: str = "classic"           # kind of training map (maps.LEVELS)


class _NoAlternative(Exception):
    """The decision being branched has no alternative worth testing."""


class Trainer:
    def __init__(self, agent: Agent, config: TrainConfig = TrainConfig(), seed: int | None = None):
        self.agent = agent
        self.config = config
        self.rng = random.Random(seed)
        self.memory: deque = deque(maxlen=config.memory_size)
        self.optimizer = optim.Adam(list(agent.state_encoder.parameters()) + list(agent.q_net.parameters()),
                                    lr=config.lr)
        self.steps_done = 0
        self.episodes_done = 0
        self.best_score = float("-inf")

    def episode(self, map_seed: int | None = None, learn: bool = True) -> dict:
        """One branch-and-compare episode on one map. Returns stats for logging."""
        if map_seed is None:
            map_seed = self.rng.randrange(10_000)
        if self.config.augment:
            self.agent.aug = (self.rng.choice((1.0, -1.0)), self.rng.choice((1.0, -1.0)), self.rng.random() < 0.5)
        try:
            return self._episode(map_seed, learn)
        finally:
            self.agent.aug = IDENTITY_AUG

    def _episode(self, map_seed: int, learn: bool) -> dict:
        """Each round scores the incumbent and up to num_branches single-decision edits on the
        same fresh seeds; the best edit replaces the incumbent if it wins by accept_margin."""
        cfg = self.config
        sim0 = MiniMetroSim(map_seed, map_spec=LEVELS[cfg.level].spec(map_seed))
        incumbent = self.agent.design(sim0)
        samples, ticks_greedy, ticks_best, accepted = [], None, 0.0, 0
        for _ in range(cfg.branch_rounds):
            seeds = [self.rng.randrange(2 ** 31) for _ in range(self.agent.playout_seeds)]
            eligible = [t for t, n in enumerate(incumbent.n_candidates) if n >= 2]
            branches, pending = [], []
            for t in self.rng.sample(eligible, len(eligible)):
                if len(branches) == cfg.num_branches:
                    break
                branch, sample = self._branch(sim0, incumbent, t)
                if branch is not None:
                    branches.append(branch)
                    pending.append(sample)

            ticks = self.agent.playouts.ticks([incumbent.sim] + [b.sim for b in branches], seeds).mean(axis=1)
            if ticks_greedy is None:
                ticks_greedy = ticks[0]
            ticks_best = max(ticks_best, *ticks)
            advantages = [math.log(y / ticks[0]) for y in ticks[1:]]
            for sample, adv in zip(pending, advantages):
                sample["advantage"] = adv
                samples.append(sample)
            if advantages and max(advantages) > cfg.accept_margin:
                incumbent = branches[int(np.argmax(advantages))]
                accepted += 1

        losses = []
        if learn:
            for sample in samples:
                self.remember(sample)
            for _ in range(cfg.replays_per_episode):
                loss = self.replay()
                if loss is not None:
                    losses.append(loss)
            self.episodes_done += 1
        return {
            "map_seed": map_seed,
            "ticks_greedy": float(ticks_greedy),
            "ticks_best": float(ticks_best),
            "branches": len(samples),
            "accepted": accepted,
            "mean_abs_adv": float(np.mean([abs(s["advantage"]) for s in samples])) if samples else 0.0,
            "loss": float(np.mean(losses)) if losses else float("nan"),
        }

    def _branch(self, sim0: MiniMetroSim, incumbent: DesignResult, t: int):
        """Swap decision t of `incumbent` for a sampled alternative, keeping its later decisions
        wherever they still fit. Returns (design, sample without its advantage), or
        (None, None) if there is no alternative within branch_window."""
        cfg, agent = self.config, self.agent
        chosen = incumbent.actions[t]
        sample: dict = {}

        def chooser(sim, phase, candidates):
            assert chosen in candidates, "replayed plan diverged from the incumbent"
            scores = agent.scores(sim, phase, candidates)
            floor = max(scores.values()) - cfg.branch_window
            others = [a for a in candidates if a != chosen and scores[a] >= floor]
            if not others:
                raise _NoAlternative
            if self.rng.random() < cfg.branch_uniform:
                alt = self.rng.choice(others)
            else:
                s = np.array([scores[a] for a in others], dtype=float)
                weights = np.exp((s - s.max()) / max(cfg.branch_temperature, 1e-6))
                alt = others[self.rng.choices(range(len(others)), weights=weights.tolist())[0]]
            sample["data"], sample["globals"] = build_graph(sim, phase, agent.aug)
            for key, action in (("chosen", chosen), ("alt", alt)):
                sample[key] = (*action_station_indices(sim, action), action_aux_features(sim, action),
                               agent.prior(sim, action))
            return alt

        rest = list(zip(incumbent.actions[t + 1:], incumbent.phases[t + 1:]))
        follow = {phase: [a for a, p in rest if p == phase] for phase in ("lines", "trains")}
        try:
            return agent.design(sim0, plan=incumbent.actions[:t], chooser=chooser, follow=follow), sample
        except _NoAlternative:
            return None, None

    def remember(self, sample: dict):
        """Store a comparison. Q(alt) - Q(chosen) is trained toward the measured advantage minus
        the prior's own difference, so the full scores match the outcome."""
        p0, s0, aux0, prior0 = sample["chosen"]
        p1, s1, aux1, prior1 = sample["alt"]
        target = sample["advantage"] - (prior1 - prior0)
        self.memory.append((sample["data"], sample["globals"], p0, s0, aux0, p1, s1, aux1, target))

    def replay(self) -> float | None:
        """One pairwise update on a sampled batch. None until the memory holds a batch."""
        if len(self.memory) < self.config.batch_size:
            return None
        batch = self.rng.sample(self.memory, self.config.batch_size)
        datas, globals_, p0, s0, aux0, p1, s1, aux1, targets = zip(*batch)
        graphs = Batch.from_data_list(list(datas))
        node_embeds, g = self.agent.state_encoder(graphs, torch.stack(globals_))
        offsets = graphs.ptr[:-1]          # first node of each graph in the batch

        def gather(idxs):
            idx = torch.as_tensor(idxs, dtype=torch.long)
            present = (idx >= 0).float().unsqueeze(-1)     # -1: no station (NOOP)
            return node_embeds[offsets + idx.clamp(min=0)] * present

        q_chosen = self.agent.q_net(gather(p0), gather(s0), g, torch.stack(aux0))
        q_alt = self.agent.q_net(gather(p1), gather(s1), g, torch.stack(aux1))
        # Huber: a few large targets (overriding a big prior penalty) shouldn't dominate
        loss = F.smooth_l1_loss(q_alt - q_chosen, torch.as_tensor(targets, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps_done += 1
        return float(loss)

    # ---- checkpoints: the agent's weights plus training progress ----

    def save(self, path: str):
        self.agent.save(path)
        with open(os.path.join(path, "params.pkl"), "wb") as f:
            pickle.dump({"steps_done": self.steps_done, "episodes_done": self.episodes_done,
                         "best_score": self.best_score}, f)

    def save_if_best(self, score: float, path: str) -> bool:
        """Save to `path` if `score` beats the best so far."""
        if score <= self.best_score:
            return False
        self.best_score = score
        self.save(path)
        print(f"  new best: {score:+.4f}")
        return True

    def load(self, path: str) -> bool:
        if not self.agent.load(path):
            return False
        params_path = os.path.join(path, "params.pkl")
        if os.path.exists(params_path):
            with open(params_path, "rb") as f:
                params = pickle.load(f)
            self.steps_done = params.get("steps_done", 0)
            self.episodes_done = params.get("episodes_done", 0)
            self.best_score = params.get("best_score", float("-inf"))
        print(f"Resuming from {path}/ (episodes {self.episodes_done}, best {self.best_score:+.4f})")
        return True
