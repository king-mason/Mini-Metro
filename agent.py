"""The design agent: scores candidate moves (prior + learned Q), builds a whole network
greedily (optionally with MCTS look-ahead), and polishes it for one map.

Training lives in training.py, play-outs in playouts.py, edit search in search.py.
"""

import copy
import json
import math
import os
import random
from dataclasses import dataclass

import torch

from config import MAX_LINES, MAX_TRAINS
from heuristic import K_CANDIDATES, PRIOR, PRIOR_TERMS, get_candidate_actions, prior_terms, structural_reward
from mcts import MCTS
from metrics import NEIGHBOR_K, neighbor_sets
from model import (ACTION_AUX_DIM, IDENTITY_AUG, STATE_FEAT_SIZE, GNNStateEncoder, PairScoringNet,
                   action_aux_features, action_station_indices, build_graph)
from playouts import CAP, Playouts
from rules import NOOP, ActionType, action_label, apply_action, connect_isolated
from simulator import MiniMetroSim

KICK_EDITS = 6       # random edits that restart a polish climb elsewhere
BUILD_NOISE = 1.0    # Gumbel noise on move scores, in prior units, for a rebuilt start

@dataclass
class DesignResult:
    """A finished design and the decisions that built it (index-aligned lists). Forced
    connections are not decisions, so they are only counted."""
    sim: MiniMetroSim
    actions: list
    phases: list
    n_candidates: list
    n_forced: int
    n_stranded: int = 0   # stations no legal move could connect


class Agent:
    """Scores each candidate move as prior_scale * structural(s, a) + Q(s, a), where Q is a
    learned correction in log(ticks survived). Q starts at zero, so an untrained agent plays
    the structural heuristic; use_q=False always does."""

    def __init__(self, use_q: bool = True, k_candidates: int = K_CANDIDATES, k_neighbors: int = NEIGHBOR_K,
                 max_design_actions: int | None = None, prior_scale: float = PRIOR.prior_scale,
                 override_margin: float = 0.1, playout_seeds: int = 4, playout_cap: int = CAP,
                 workers: int = 0, mcts_sims: int = 0, mcts_c: float = 1.4, mcts_horizon: int = 6,
                 screen_edits: bool = True, seed: int | None = None):
        """override_margin: Q must rate another move this much better (log units) to overrule
        the prior. playout_seeds: passenger seeds per design in search and training.
        workers > 0 plays out in that many processes. mcts_sims > 0 turns on look-ahead.
        screen_edits: the survival estimate picks which edits search plays."""
        self.use_q = use_q
        self.k = k_candidates
        self.k_neighbors = k_neighbors
        self.max_design_actions = max_design_actions
        self.prior_scale = prior_scale
        self.override_margin = override_margin
        self.playout_seeds = playout_seeds
        self.screen_edits = screen_edits
        self.playouts = Playouts(playout_cap, workers)
        self.rng = random.Random(seed)
        self.aug = IDENTITY_AUG            # symmetry states are encoded under (training varies it)

        self.state_encoder = GNNStateEncoder()
        self.q_net = PairScoringNet(STATE_FEAT_SIZE, STATE_FEAT_SIZE, ACTION_AUX_DIM)
        # learned reweighting of the prior's terms (imitation); all 0 = the heuristic's prior
        self.log_term_weights = torch.nn.Parameter(torch.zeros(len(PRIOR_TERMS)))
        self.mcts = MCTS(self.scores, self.prior, self.candidates, mcts_sims, mcts_c, mcts_horizon) if mcts_sims else None

    def close(self):
        self.playouts.close()

    # ---- scoring ----

    def candidates(self, sim: MiniMetroSim, phase: str) -> list:
        return get_candidate_actions(sim, self.k, phase=phase, neighbor_k=self.k_neighbors)

    def prior(self, sim: MiniMetroSim, action: tuple) -> float:
        if not self.log_term_weights.any():
            return self.prior_scale * structural_reward(sim, action)
        weights = self.log_term_weights.detach().exp().tolist()
        return self.prior_scale * sum(w * t for w, t in zip(weights, prior_terms(sim, action)))

    def term_weights(self) -> dict[str, float]:
        """How much the agent trusts each term of the prior (1 = as hand-written)."""
        return dict(zip(PRIOR_TERMS, self.log_term_weights.detach().exp().tolist()))

    def prior_and_q(self, sim: MiniMetroSim, phase: str, candidates: list) -> tuple[dict, dict]:
        """({action: prior}, {action: Q}) for the candidates, from one state encoding."""
        prior = {a: self.prior(sim, a) for a in candidates}
        if not self.use_q:
            return prior, dict.fromkeys(candidates, 0.0)
        data, globals_ = build_graph(sim, phase, self.aug)
        with torch.no_grad():
            node_embeds, g = self.state_encoder(data, globals_)
            zero = torch.zeros(STATE_FEAT_SIZE)
            rows = [action_station_indices(sim, a) for a in candidates]
            h_p = torch.stack([node_embeds[p] if p >= 0 else zero for p, _ in rows])
            h_s = torch.stack([node_embeds[s] if s >= 0 else zero for _, s in rows])
            aux = torch.stack([action_aux_features(sim, a) for a in candidates])
            q = self.q_net(h_p, h_s, g.unsqueeze(0).expand(len(candidates), -1), aux)
        return prior, dict(zip(candidates, q.tolist()))

    def scores(self, sim: MiniMetroSim, phase: str, candidates: list) -> dict:
        """{action: prior + Q}."""
        prior, q = self.prior_and_q(sim, phase, candidates)
        return {a: prior[a] + q[a] for a in candidates}

    def explain(self, sim: MiniMetroSim, phase: str, candidates: list) -> list[tuple]:
        """(action, prior, Q) per candidate, best total first. For viewing."""
        prior, q = self.prior_and_q(sim, phase, candidates)
        return sorted(((a, prior[a], q[a]) for a in candidates), key=lambda r: -(r[1] + r[2]))

    def select_action(self, sim: MiniMetroSim, phase: str = "lines", candidates: list | None = None):
        """The policy's move (None if there are none): the prior's choice unless Q rates another
        better by override_margin, or the MCTS plan when look-ahead is on."""
        if candidates is None:
            candidates = self.candidates(sim, phase)
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        if self.mcts is not None:
            action = self.mcts.plan(sim, phase)
            if action is not None:
                return action
        prior, q = self.prior_and_q(sim, phase, candidates)
        best = max(candidates, key=lambda a: prior[a] + q[a])
        default = max(prior, key=prior.get)
        if prior[best] + q[best] - (prior[default] + q[default]) < self.override_margin:
            return default
        return best

    # ---- design ----

    def design(self, sim0: MiniMetroSim, plan: list = (), chooser=None, follow: dict | None = None,
               guide=None, on_step=None) -> DesignResult:
        """Build a design on a copy of sim0: lines, forced connections, then trains.

        plan: replay these first decisions. chooser(sim, phase, candidates) -> action makes the
        next one instead of the policy. follow = {"lines": [...], "trains": [...]}: later
        decisions take the next listed action that is still a candidate, then stop building
        lines / fall back to the policy once a list runs out. guide(sim, phase, candidates) ->
        action or None steers every other decision (None: the policy decides); its action
        need not be a candidate.

        on_step(sim, action, label, scored) is called before each action is applied, for
        viewing; `scored` is explain()'s list, or None for a forced connection. Returning
        False stops the calls.
        """
        sim = copy.deepcopy(sim0)
        actions, phases, n_candidates = [], [], []
        queues = {phase: list(acts) for phase, acts in follow.items()} if follow is not None else None
        watching = on_step is not None

        def show(action, scored=None):
            nonlocal watching
            if watching:
                label = action_label(action, sim)
                watching = on_step(sim, action, label if scored is not None else f"forced: {label}", scored) is not False

        def followed(phase, candidates):
            while queues[phase]:
                action = queues[phase].pop(0)
                if action in candidates:
                    return action
            return NOOP if phase == "lines" and NOOP in candidates else None

        def decide(phase):
            candidates = self.candidates(sim, phase)
            if not candidates:
                return None
            k = len(actions)
            if k < len(plan):
                action = plan[k]
            elif k == len(plan) and chooser is not None:
                action = chooser(sim, phase, candidates)
            else:
                action = followed(phase, candidates) if queues is not None else None
                if action is None and guide is not None:
                    action = guide(sim, phase, candidates)
                if action is None:
                    action = self.select_action(sim, phase, candidates)
            actions.append(action)
            phases.append(phase)
            n_candidates.append(len(candidates))
            show(action, self.explain(sim, phase, candidates) if watching else None)
            return action

        def build(phase, limit):
            for _ in range(limit):
                action = decide(phase)
                if action is None or action[0] == ActionType.NOOP:
                    return
                apply_action(sim, action)

        # no line visits a station twice, so MAX_LINES * stations decisions always suffice
        build("lines", self.max_design_actions or MAX_LINES * len(sim.stations))
        n_forced, n_stranded = connect_isolated(sim, show)
        build("trains", MAX_TRAINS + 1)
        return DesignResult(sim, actions, phases, n_candidates, n_forced, n_stranded)

    # ---- polishing a finished design ----

    def improve(self, sim0: MiniMetroSim, design: MiniMetroSim, rounds: int = 10, batch: int = 8,
                seed_rng: random.Random | None = None, rng: random.Random | None = None,
                on_round=None, record: list | None = None) -> tuple[MiniMetroSim, dict]:
        """Play-out search over small edits (search.improve). Pass a seed_rng that can't draw
        the seeds the result will be judged on (bench.search_seed_rng)."""
        from search import improve
        from survival import SurvivalModel
        return improve(sim0, design, self.playouts.ticks, rounds=rounds, batch=batch, n_seeds=self.playout_seeds,
                       seed_rng=seed_rng or random.Random(self.rng.random()), rng=rng or self.rng,
                       on_round=on_round, record=record,
                       prescreen=SurvivalModel(sim0).log_ticks if self.screen_edits else None)

    def noisy_design(self, sim0: MiniMetroSim, rng: random.Random, noise: float = BUILD_NOISE) -> MiniMetroSim:
        """A design built with Gumbel noise of scale `noise` on every move's score: a
        different network that is still a sensible one."""
        def guide(sim, phase, candidates):
            scores = self.scores(sim, phase, candidates)
            return max(candidates, key=lambda a: scores[a] - noise * math.log(-math.log(rng.random() + 1e-12)))
        return self.design(sim0, guide=guide).sim

    def polish(self, sim0: MiniMetroSim, design: MiniMetroSim, rounds: int = 30, restarts: int = 1,
               rng: random.Random | None = None, on_round=None) -> tuple[MiniMetroSim, dict]:
        """Hill-climb on the survival estimate alone (search.polish): no play-outs.

        Each climb ends on a different local top. With restarts > 1 it climbs again from
        other starts (the same design, a kicked copy, a noisy rebuild, in turn) and keeps
        the result the estimate rates best. stats["start"] / ["final"]: estimated ticks of
        `design` / the result; stats["climb"]: which climb was kept (0 = the first)."""
        from search import build, kick, polish, spec_of
        from survival import SurvivalModel

        rng = rng or self.rng
        value = SurvivalModel(sim0).log_ticks
        nbrs = neighbor_sets(sim0)
        report = None if on_round is None else lambda r, n, sim, stats: on_round(r, n, sim, in_ticks(stats))

        def in_ticks(stats):
            return {**stats, "start": math.exp(stats["start"]), "final": math.exp(stats["final"])}

        def start(k):
            kind = "same" if k == 0 else ("same", "kick", "noisy")[(k - 1) % 3]
            if kind == "kick":
                return build(sim0, kick(spec_of(design), sim0, nbrs, rng, KICK_EDITS))
            return self.noisy_design(sim0, rng) if kind == "noisy" else design

        best, first_start = None, None
        for k in range(restarts):
            result, stats = polish(sim0, start(k), value, rounds=rounds, rng=rng, on_round=report)
            first_start = stats["start"] if first_start is None else first_start
            if best is None or stats["final"] > best[1]["final"]:
                best = (result, {**stats, "climb": k})
        result, stats = best
        return result, in_ticks({**stats, "start": first_start, "restarts": restarts})

    # ---- weights ----

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_encoder.state_dict(), os.path.join(path, "state_encoder.pth"))
        torch.save(self.q_net.state_dict(), os.path.join(path, "q_net.pth"))
        with open(os.path.join(path, "settings.json"), "w") as f:
            json.dump({"k_candidates": self.k, "override_margin": self.override_margin,
                       "log_term_weights": self.log_term_weights.tolist()}, f)

    def load(self, path: str) -> bool:
        """Load weights saved by save(), and the candidate count and margin they were trained
        with. False (and nothing loaded) if they are missing or from a different architecture."""
        enc_path, q_path = os.path.join(path, "state_encoder.pth"), os.path.join(path, "q_net.pth")
        if not (os.path.exists(enc_path) and os.path.exists(q_path)):
            return False
        encoder, q_net = GNNStateEncoder(), PairScoringNet(STATE_FEAT_SIZE, STATE_FEAT_SIZE, ACTION_AUX_DIM)
        try:
            encoder.load_state_dict(torch.load(enc_path, weights_only=True))
            q_net.load_state_dict(torch.load(q_path, weights_only=True))
        except RuntimeError as e:
            print(f"Skipping checkpoint at {path}/: architecture changed ({e})")
            return False
        self.state_encoder.load_state_dict(encoder.state_dict())   # in place: optimizers keep their parameters
        self.q_net.load_state_dict(q_net.state_dict())
        settings_path = os.path.join(path, "settings.json")
        if os.path.exists(settings_path):
            with open(settings_path) as f:
                settings = json.load(f)
            self.k, self.override_margin = settings["k_candidates"], settings["override_margin"]
            with torch.no_grad():
                self.log_term_weights.copy_(torch.tensor(settings.get("log_term_weights", [0.0] * len(PRIOR_TERMS))))
        return True
