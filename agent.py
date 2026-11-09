"""The design agent: builds a network greedily from prior + learned Q (optionally with MCTS
look-ahead), polishes it with play-out search, and learns Q by branch-and-compare.
"""

import copy
import math
import multiprocessing
import os
import pickle
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch

from heuristic import K_CANDIDATES_DEFAULT, NEIGHBOR_K_DEFAULT, R, get_candidate_actions, structural_reward
from model import (ACTION_AUX_DIM, IDENTITY_AUG, STATE_FEAT_SIZE, GNNStateEncoder, PairScoringNet,
                   action_aux_features, action_station_indices, build_graph)
from rules import MAX_TRAINS, ActionType, action_label, action_segments, apply_action, is_legal
from simulator import MiniMetroSim, run_playout


# ---- MCTS over design actions ----

class _MCTSNode:
    """A design state in the search tree. Transitions (apply_action) are deterministic,
    so this is a plain tree; the only stochasticity (the play-out) is bootstrapped by
    the scorer at leaves rather than rolled out."""
    __slots__ = ("sim", "phase", "depth", "action", "struct",
                 "children", "untried", "priors", "value0", "is_terminal", "N", "W")

    def __init__(self, sim, phase, depth, action=None, struct=0.0):
        self.sim = sim
        self.phase = phase
        self.depth = depth
        self.action = action      # action taken from the parent to reach this node
        self.struct = struct      # scaled structural (edge) reward of that action
        self.children: dict = {}
        self.untried: list = []   # candidate actions not yet expanded, best-prior first
        self.priors: dict = {}    # action -> score(s, a)
        self.value0 = 0.0         # bootstrap state value
        self.is_terminal = False
        self.N = 0
        self.W = 0.0


# ---- agent ----

@dataclass
class DesignResult:
    """A finished design plus the agent decisions that built it (index-aligned lists).
    Forced connections are deterministic, so they are not decisions and not listed."""
    sim: MiniMetroSim
    actions: list
    phases: list
    n_candidates: list
    n_forced: int
    n_stranded: int = 0   # stations no legal move could connect


class _NoPlausibleAlternative(Exception):
    """Raised inside a branch when the decision has no alternative worth testing."""


def _softmax_choice(rng: random.Random, items: list, scores: list[float], temperature: float):
    s = np.array(scores, dtype=float)
    p = np.exp((s - s.max()) / max(temperature, 1e-6))
    return items[rng.choices(range(len(items)), weights=p.tolist())[0]]


class Agent:
    """Design-first agent: builds the whole network (lines, then trains) before play-out.

    Each candidate is scored  prior_scale * structural(s, a) + Q(s, a).  The structural
    terms are the hand-written priors in config.yaml; Q is a learned correction in units of
    log(ticks survived). Q's output starts at zero, so an untrained agent plays exactly the
    structural heuristic.

    Training is branch-and-compare (see train_episode): build the greedy design, edit it by
    swapping one decision for an alternative while keeping the rest, and play both out on
    the same passenger seeds. The measured log-ratio of survival is the target for
    score(s, alternative) - score(s, chosen). The map's difficulty cancels in the ratio, and
    each target is about one decision.
    """
    def __init__(
        self,
        num_stations: int = 20,
        memory_size: int = 10_000,
        batch_size: int = 32,
        lr: float = 3e-4,
        k_candidates: int = K_CANDIDATES_DEFAULT,
        k_neighbors: int = NEIGHBOR_K_DEFAULT,
        max_design_actions: int = 50,
        use_q: bool = True,
        prior_scale: float = R.PRIOR_SCALE,
        override_margin: float = 0.1,
        num_branches: int = 6,
        branch_rounds: int = 2,
        playout_seeds: int = 4,
        playout_cap: int = 6000,
        accept_margin: float = 0.08,
        branch_uniform: float = 0.3,
        branch_temperature: float = 0.25,
        branch_window: float = 2.0,
        replays_per_episode: int = 4,
        augment: bool = True,
        workers: int = 0,
        mcts_sims: int = 0,
        mcts_c: float = 1.4,
        mcts_horizon: int = 6,
        screen_edits: bool = True,
        seed: int | None = None,
    ):
        """use_q=False scores with the structural prior alone (the heuristic baseline).

        override_margin: the greedy policy keeps the prior's choice unless Q predicts some
        other candidate is better by at least this much (log units; 0.1 ≈ 10% longer
        survival). The prior's top-2 gap is usually tiny (median ~0.05), so without a
        margin, small errors in Q flip decisions at random: measured, Q overrode the prior
        on 45% of decisions and validation survival fell to ~0.75x the heuristic.

        Each episode runs branch_rounds rounds of num_branches single-decision branches.
        Every design is scored on playout_seeds passenger seeds, capped at playout_cap ticks
        (spawning plateaus by ~3000). A branch becomes the incumbent if it wins by more than
        accept_margin in log units (~8% is the noise floor at 4 seeds). Alternatives come
        from candidates scoring within branch_window of the best, drawn uniformly with
        probability branch_uniform, otherwise by softmax over scores. The window keeps out
        moves the prior all but forbids (U-turns cost 1000 * prior_scale = 10):
        measured, those branches averaged an advantage of -0.02, so their targets are mostly
        "cancel the penalty" and would swamp the subtler corrections. Revisiting those
        penalties is a prior question (plan Phase 2), not something to learn around here.

        augment: each training episode encodes the map under a random reflection/rotation
        (the sim only depends on distances, so it is the same problem); off at evaluation.

        workers > 0 runs play-outs in that many processes; the entry script then needs an
        `if __name__ == "__main__":` guard. mcts_sims > 0 enables look-ahead, off by default:
        the pairwise-trained Q only orders actions within a state, while MCTS compares
        values across states.

        screen_edits: improve lets the survival estimate (survival.py) pick which edits to
        play each round, 7 of 40 drawn, at the same play-out cost: measured x1.20 [1.10, 1.33]
        over unscreened search on 48 bench maps. Off when the search should try edits at
        random (e.g. collecting unbiased training data).
        """
        self.num_stations = num_stations
        self.k = k_candidates
        self.k_neighbors = k_neighbors
        self.MAX_DESIGN_ACTIONS = max_design_actions
        self.use_q = use_q
        self.prior_scale = prior_scale
        self.override_margin = override_margin
        self.state_size = STATE_FEAT_SIZE

        self.num_branches = num_branches
        self.branch_rounds = branch_rounds
        self.playout_seeds = playout_seeds
        self.playout_cap = playout_cap
        self.screen_edits = screen_edits
        self.accept_margin = accept_margin
        self.branch_uniform = branch_uniform
        self.branch_temperature = branch_temperature
        self.branch_window = branch_window
        self.replays_per_episode = replays_per_episode
        self.augment = augment
        self._aug = IDENTITY_AUG  # symmetry used to encode states; random only while training
        self.workers = workers
        self._pool: ProcessPoolExecutor | None = None

        self.mcts_sims = mcts_sims
        self.mcts_c = mcts_c
        self.mcts_horizon = mcts_horizon

        self.rng = random.Random(seed)
        self.memory: deque = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.steps_done = 0
        self.episodes_done = 0
        self.last_episode_ticks = 0.0
        self.best_score = float("-inf")

        self.state_encoder = GNNStateEncoder()
        self.q_net = PairScoringNet(self.state_size, self.state_size, ACTION_AUX_DIM)
        self.optimizer = optim.Adam(
            list(self.state_encoder.parameters()) + list(self.q_net.parameters()),
            lr=lr,
        )

    # ---- scoring ----

    def _action_inputs(self, sim: MiniMetroSim, node_embeds: torch.Tensor, actions: list):
        """Stacked (primary embedding, secondary embedding, aux) rows for `actions`."""
        zero = torch.zeros(self.state_size)
        h_p, h_s, aux = [], [], []
        for action in actions:
            p_idx, s_idx = action_station_indices(sim, action)
            h_p.append(node_embeds[p_idx] if p_idx >= 0 else zero)
            h_s.append(node_embeds[s_idx] if s_idx >= 0 else zero)
            aux.append(action_aux_features(sim, action))
        return torch.stack(h_p), torch.stack(h_s), torch.stack(aux)

    def _prior_and_q(self, sim: MiniMetroSim, phase: str, candidates: list) -> tuple[dict, dict]:
        """Per-candidate (prior_scale * structural(s, a), Q(s, a)). One state encode per call."""
        prior = {a: self.prior_scale * structural_reward(sim, a) for a in candidates}
        if not self.use_q:
            return prior, dict.fromkeys(candidates, 0.0)
        data, globals_ = build_graph(sim, phase, self._aug)
        with torch.no_grad():
            node_embeds, g = self.state_encoder(data, globals_)
            h_p, h_s, aux = self._action_inputs(sim, node_embeds, candidates)
            q = self.q_net(h_p, h_s, g.unsqueeze(0).expand(len(candidates), -1), aux)
        return prior, dict(zip(candidates, q.tolist()))

    def _candidate_scores(self, sim: MiniMetroSim, phase: str, candidates: list) -> dict:
        """score(s, a) = prior_scale * structural(s, a) + Q(s, a) for each candidate.
        Used for branch sampling and as the MCTS prior."""
        prior, q = self._prior_and_q(sim, phase, candidates)
        return {a: prior[a] + q[a] for a in candidates}

    def select_action(self, sim: MiniMetroSim, phase: str = "lines",
                      candidates: list | None = None) -> tuple | None:
        """Greedy choice, or MCTS look-ahead when mcts_sims > 0. None if no candidates.

        Greedy keeps the prior's choice unless Q rates another candidate better by at least
        override_margin, so Q only steers where it has learned a large correction."""
        if candidates is None:
            candidates = get_candidate_actions(sim, self.k, phase=phase, neighbor_k=self.k_neighbors)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        if self.mcts_sims > 0:
            action = self._mcts_search(sim, phase)
            if action is not None:
                return action
        prior, q = self._prior_and_q(sim, phase, candidates)
        scores = {a: prior[a] + q[a] for a in candidates}
        best = max(scores, key=scores.get)
        default = max(prior, key=prior.get)
        if scores[best] - scores[default] < self.override_margin:
            return default
        return best

    # ---- MCTS ----

    def _mcts_make_node(self, sim, phase, depth, action=None, struct=0.0,
                        terminal_value: float | None = None) -> "_MCTSNode":
        """Create a node and populate its priors, bootstrap value, expansion order,
        and terminal flag. One state encode per node."""
        node = _MCTSNode(sim, phase, depth, action=action, struct=struct)
        if terminal_value is not None:
            # NOOP ends this phase, so the design is finished here: its value is what the
            # parent scored for stopping, not the best continuation of a state that won't
            # continue. (Using the continuation made stopping tie with adding a train.)
            node.is_terminal = True
            node.value0 = terminal_value
            return node
        if depth >= self.mcts_horizon:
            node.is_terminal = True
        candidates = get_candidate_actions(sim, self.k, phase=phase, neighbor_k=self.k_neighbors)
        if not candidates:
            return node
        node.priors = self._candidate_scores(sim, phase, candidates)
        # State value = value of the best continuation = max_a score(s, a).
        node.value0 = max(node.priors.values())
        if not node.is_terminal:
            node.untried = sorted(node.priors, key=node.priors.get, reverse=True)
        return node

    def _mcts_search(self, root_sim: MiniMetroSim, phase: str):
        """Plan the next design action by simulating from a deep-copied root. Returns
        the most-visited root action (None if there are no candidates)."""
        root = self._mcts_make_node(copy.deepcopy(root_sim), phase, depth=0)
        if not root.priors:
            return None
        if len(root.priors) == 1:
            return next(iter(root.priors))  # only one legal move (e.g. a forced rescue)

        vmin, vmax = float("inf"), float("-inf")  # min-max normalize returns for UCB

        for _ in range(self.mcts_sims):
            node = root
            path = [root]
            # SELECTION: descend fully-expanded interior nodes via normalized UCB.
            while not node.is_terminal and not node.untried and node.children:
                node = self._ucb_select(node, vmin, vmax)
                path.append(node)
            # EXPANSION: open the best untried action at this node.
            if not node.is_terminal and node.untried:
                action = node.untried.pop(0)
                struct = self.prior_scale * structural_reward(node.sim, action)
                if action[0] == ActionType.NOOP:
                    child = self._mcts_make_node(
                        node.sim, phase, node.depth + 1, action=action, struct=struct,
                        terminal_value=node.priors[action] - struct,
                    )
                else:
                    child_sim = copy.deepcopy(node.sim)
                    apply_action(child_sim, action)
                    child = self._mcts_make_node(child_sim, phase, node.depth + 1,
                                                 action=action, struct=struct)
                node.children[action] = child
                path.append(child)
                node = child
            # EVALUATION + BACKUP: episodic return = Σ edge structural + bootstrap leaf value.
            g = sum(nd.struct for nd in path[1:]) + node.value0
            vmin, vmax = min(vmin, g), max(vmax, g)
            for nd in path:
                nd.N += 1
                nd.W += g

        # Robust choice: the most-visited root child (ties broken by mean value).
        best = max(root.children.values(), key=lambda c: (c.N, c.W / max(c.N, 1)))
        return best.action

    def _ucb_select(self, node: "_MCTSNode", vmin: float, vmax: float) -> "_MCTSNode":
        span = vmax - vmin if vmax > vmin else 1.0
        log_n = math.log(node.N + 1)
        best, best_score = None, float("-inf")
        for child in node.children.values():
            q_norm = ((child.W / child.N) - vmin) / span
            u = self.mcts_c * math.sqrt(log_n / child.N)
            score = q_norm + u
            if score > best_score:
                best_score, best = score, child
        return best

    # ---- design ----

    def design(self, sim0: MiniMetroSim, plan: list = (), chooser=None, follow: dict | None = None,
               on_step=None) -> DesignResult:
        """Build a complete design on a copy of sim0: lines, forced connections, then trains.

        The first len(plan) decisions replay `plan`. If `chooser` is given, it makes the next
        decision instead of the policy: chooser(sim, phase, candidates) -> action.

        follow = {"lines": [...], "trains": [...]} holds the rest of a design being edited.
        Each later decision takes the next action of its phase that is still a candidate
        (actions that no longer fit are dropped); once a phase's list runs out, line building
        stops as soon as NOOP is allowed and remaining trains are placed greedily. Without
        `follow`, all other decisions are greedy (select_action).

        on_step(sim, action, label, scored) is called before each action is applied, for
        viewing: `scored` lists (action, prior, Q) for every candidate, best first, or is None
        for a forced connection. Returning False stops the calls for the rest of the design.
        """
        sim = copy.deepcopy(sim0)
        actions, phases, n_candidates = [], [], []
        watching = on_step is not None
        queues = {phase: list(acts) for phase, acts in follow.items()} if follow is not None else None

        def followed(phase, candidates):
            queue = queues[phase]
            while queue:
                action = queue.pop(0)
                if action in candidates:
                    return action
            if phase == "lines" and (ActionType.NOOP,) in candidates:
                return (ActionType.NOOP,)
            return None

        def decide(phase):
            candidates = get_candidate_actions(sim, self.k, phase=phase, neighbor_k=self.k_neighbors)
            if not candidates:
                return None
            k = len(actions)
            if k < len(plan):
                action = plan[k]
            elif k == len(plan) and chooser is not None:
                action = chooser(sim, phase, candidates)
            else:
                action = followed(phase, candidates) if queues is not None else None
                if action is None:
                    action = self.select_action(sim, phase, candidates)
            actions.append(action)
            phases.append(phase)
            n_candidates.append(len(candidates))
            show(action, self.explain(sim, phase, candidates) if watching else None)
            return action

        def show(action, scored=None):
            nonlocal watching
            if watching:
                label = action_label(action, sim) if scored is not None else f"forced: {action_label(action, sim)}"
                watching = on_step(sim, action, label, scored) is not False

        # Phase A1: line design. NEW_LINE / EXTEND_LINE / NOOP only.
        for _ in range(self.MAX_DESIGN_ACTIONS):
            action = decide("lines")
            if action is None or action[0] == ActionType.NOOP:
                break
            apply_action(sim, action)

        n_forced, n_stranded = self._connect_isolated(sim, show)

        # Phase A2: train allocation over the final lines.
        for _ in range(MAX_TRAINS + 1):
            action = decide("trains")
            if action is None or action[0] == ActionType.NOOP:
                break
            apply_action(sim, action)

        return DesignResult(sim, actions, phases, n_candidates, n_forced, n_stranded)

    def explain(self, sim: MiniMetroSim, phase: str, candidates: list) -> list[tuple]:
        """(action, prior, Q) for each candidate, best total score first. For viewing."""
        prior, q = self._prior_and_q(sim, phase, candidates)
        return sorted(((a, prior[a], q[a]) for a in candidates), key=lambda r: -(r[1] + r[2]))

    @staticmethod
    def _connect_isolated(sim: MiniMetroSim, show) -> tuple[int, int]:
        """Connect stations the agent left isolated. Not agent choices: for each, extend the
        nearest line end that reaches it without a crossing, or else start a new line to the
        nearest station it can reach, or else detour the segment that needs the least extra
        track through it. A station none of these reach stays unconnected, and its queue
        grows until it overcrowds. Returns (forced connections, stations left stranded)."""
        def added_track(action):
            dist = lambda a, b: sim.stations[a].pos.get_distance(sim.stations[b].pos)
            new = action_segments(sim, action)
            if action[0] != ActionType.INSERT_STATION:
                return dist(*new[0])
            (a, s), (_, b) = new
            return dist(a, s) + dist(s, b) - dist(a, b)

        forced = stranded = 0
        for iso_idx, station in enumerate(sim.stations):
            if any(station in line.stations for line in sim.lines):
                continue
            extends = [(ActionType.EXTEND_LINE, li, end, iso_idx)
                       for li in range(len(sim.lines)) for end in (0, -1)]
            new_lines = [(ActionType.NEW_LINE, iso_idx, j) for j in range(len(sim.stations)) if j != iso_idx]
            inserts = [(ActionType.INSERT_STATION, li, k, iso_idx)
                       for li, line in enumerate(sim.lines) for k in range(len(line.stations) - 1)]
            for options in (extends, new_lines, inserts):
                legal = [a for a in options if is_legal(sim, a)]
                if legal:
                    action = min(legal, key=added_track)
                    show(action)
                    apply_action(sim, action)
                    forced += 1
                    break
            else:
                stranded += 1
        return forced, stranded

    # ---- play-outs ----

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"))
        return self._pool

    def close(self):
        """Shut down the play-out worker pool, if one was started."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def playout_ticks(self, sims: list[MiniMetroSim], seeds: list[int]) -> np.ndarray:
        """Ticks survived (capped at playout_cap), shape (len(sims), len(seeds))."""
        jobs = [(sim, seed) for sim in sims for seed in seeds]
        job_sims, job_seeds = zip(*jobs)
        if self.workers > 0:
            results = list(self._get_pool().map(run_playout, job_sims, job_seeds, repeat(self.playout_cap)))
        else:
            results = [run_playout(sim, seed, self.playout_cap) for sim, seed in jobs]
        return np.array([t for t, _ in results], dtype=float).reshape(len(sims), len(seeds))

    def score_designs(self, sims: list[MiniMetroSim], seeds: list[int]) -> list[float]:
        """Mean ticks survived (capped at playout_cap) of each design over the same seeds."""
        return self.playout_ticks(sims, seeds).mean(axis=1).tolist()

    def improve(self, sim0: MiniMetroSim, design: MiniMetroSim, rounds: int = 10, batch: int = 8,
                seed_rng: random.Random | None = None, rng: random.Random | None = None,
                on_round=None, record: list | None = None) -> tuple[MiniMetroSim, dict]:
        """Polish a finished design for this map with play-out-verified local edits (search.py).

        Each round draws playout_seeds fresh passenger seeds from seed_rng (default: from the
        agent's rng); when the result will be evaluated on known seeds, pass a seed_rng that
        can't produce them (bench.search_seed_rng). on_round and record are passed to
        search.improve. With screen_edits, survival.py picks which edits are played."""
        from search import improve
        from survival import SurvivalModel
        prescreen = SurvivalModel(sim0).log_ticks if self.screen_edits else None
        return improve(sim0, design, self.playout_ticks, rounds=rounds, batch=batch,
                       n_seeds=self.playout_seeds, seed_rng=seed_rng or random.Random(self.rng.random()),
                       rng=rng or self.rng, on_round=on_round, record=record, prescreen=prescreen)

    def polish(self, sim0: MiniMetroSim, design: MiniMetroSim, rounds: int = 30,
               rng: random.Random | None = None, on_round=None) -> tuple[MiniMetroSim, dict]:
        """Polish a finished design with the survival estimate alone (search.polish): no
        play-outs, a few seconds per map. stats["start"] / ["final"] are estimated ticks."""
        from search import polish
        from survival import SurvivalModel

        def in_ticks(stats):
            return {**stats, "start": math.exp(stats["start"]), "final": math.exp(stats["final"])}

        report = None if on_round is None else lambda r, n, sim, stats: on_round(r, n, sim, in_ticks(stats))
        design, stats = polish(sim0, design, SurvivalModel(sim0).log_ticks, rounds=rounds,
                               rng=rng or self.rng, on_round=report)
        return design, in_ticks(stats)

    # ---- training ----

    def _branch(self, sim0: MiniMetroSim, incumbent: DesignResult, t: int):
        """Edit `incumbent`: swap decision t for a sampled alternative and keep the rest of
        the design wherever it still fits (see design's `follow`). Letting the policy redraw
        everything after t instead made one early swap change ~31 later decisions, so the
        measured advantage was a whole new network's and no per-decision feature could
        predict it; and train swaps re-converged to the same allocation (advantage 0).

        Returns (branch design, training sample still missing its "advantage"), or
        (None, None) if decision t has no alternative inside branch_window."""
        chosen = incumbent.actions[t]
        sample: dict = {}

        def chooser(sim, phase, candidates):
            assert chosen in candidates, "replayed plan diverged from the incumbent"
            scores = self._candidate_scores(sim, phase, candidates)
            floor = max(scores.values()) - self.branch_window
            others = [a for a in candidates if a != chosen and scores[a] >= floor]
            if not others:
                raise _NoPlausibleAlternative
            if self.rng.random() < self.branch_uniform:
                alt = self.rng.choice(others)
            else:
                alt = _softmax_choice(self.rng, others, [scores[a] for a in others],
                                      self.branch_temperature)
            sample["data"], sample["globals"] = build_graph(sim, phase, self._aug)
            for key, action in (("chosen", chosen), ("alt", alt)):
                p_idx, s_idx = action_station_indices(sim, action)
                prior = self.prior_scale * structural_reward(sim, action)
                sample[key] = (p_idx, s_idx, action_aux_features(sim, action), prior)
            return alt

        rest = list(zip(incumbent.actions[t + 1:], incumbent.phases[t + 1:]))
        follow = {phase: [a for a, p in rest if p == phase] for phase in ("lines", "trains")}
        try:
            branch = self.design(sim0, plan=incumbent.actions[:t], chooser=chooser, follow=follow)
        except _NoPlausibleAlternative:
            return None, None
        return branch, sample

    def train_episode(self, map_seed: int | None = None, learn: bool = True) -> dict:
        """One branch-and-compare episode on one map. Returns stats for logging.

        Each round scores the incumbent design on fresh passenger seeds, plus num_branches
        branches that each change exactly one of its decisions, on the same seeds. Every
        branch yields one target, log(ticks_branch / ticks_incumbent): the map's difficulty
        cancels in the ratio, and other decisions' noise can't leak in. The best branch, if it
        wins by more than accept_margin, is the next round's incumbent; new seeds each round
        mean an incumbent that won on lucky seeds is re-scored fairly.
        """
        if map_seed is None:
            map_seed = self.rng.randrange(10_000)
        if self.augment:
            self._aug = (self.rng.choice((1.0, -1.0)), self.rng.choice((1.0, -1.0)),
                         self.rng.random() < 0.5)
        try:
            return self._train_on_map(map_seed, learn)
        finally:
            self._aug = IDENTITY_AUG

    def _train_on_map(self, map_seed: int, learn: bool) -> dict:
        """Body of train_episode, run under the episode's symmetry (self._aug)."""
        sim0 = MiniMetroSim(self.num_stations, map_seed=map_seed)
        incumbent = self.design(sim0)

        samples: list[dict] = []
        ticks_greedy = None
        ticks_best = 0.0
        accepted = 0
        for _ in range(self.branch_rounds):
            seeds = [self.rng.randrange(2 ** 31) for _ in range(self.playout_seeds)]
            eligible = [t for t, n in enumerate(incumbent.n_candidates) if n >= 2]
            branches, pending = [], []
            for t in self.rng.sample(eligible, len(eligible)):  # random order, no repeats
                if len(branches) == self.num_branches:
                    break
                branch, sample = self._branch(sim0, incumbent, t)
                if branch is not None:
                    branches.append(branch)
                    pending.append(sample)

            ticks = self.score_designs([incumbent.sim] + [b.sim for b in branches], seeds)
            if ticks_greedy is None:
                ticks_greedy = ticks[0]
            ticks_best = max(ticks_best, *ticks)
            advantages = [math.log(y / ticks[0]) for y in ticks[1:]]
            for sample, adv in zip(pending, advantages):
                sample["advantage"] = adv
                samples.append(sample)
            if advantages and max(advantages) > self.accept_margin:
                incumbent = branches[int(np.argmax(advantages))]
                accepted += 1

        losses = []
        if learn:
            for sample in samples:
                self.remember(sample)
            for _ in range(self.replays_per_episode):
                loss = self.replay()
                if loss is not None:
                    losses.append(loss)
            self.episodes_done += 1

        self.last_episode_ticks = ticks_greedy
        return {
            "map_seed": map_seed,
            "ticks_greedy": ticks_greedy,
            "ticks_best": ticks_best,
            "branches": len(samples),
            "accepted": accepted,
            "mean_abs_adv": float(np.mean([abs(s["advantage"]) for s in samples])) if samples else 0.0,
            "loss": float(np.mean(losses)) if losses else float("nan"),
        }

    def remember(self, sample: dict):
        """Store one branch comparison. The target for Q(alt) - Q(chosen) is the measured
        advantage minus the prior's own difference, so the full scores match the outcome."""
        p0, s0, aux0, prior0 = sample["chosen"]
        p1, s1, aux1, prior1 = sample["alt"]
        target = sample["advantage"] - (prior1 - prior0)
        self.memory.append((sample["data"], sample["globals"], p0, s0, aux0, p1, s1, aux1, target))

    def replay(self) -> float | None:
        """One pairwise update: Q(s, alt) - Q(s, chosen) toward the stored target."""
        if len(self.memory) < self.batch_size:
            return None
        batch = self.rng.sample(self.memory, self.batch_size)
        datas, globals_list, p0, s0, aux0, p1, s1, aux1, targets = zip(*batch)

        # Batch graphs; Batch.ptr[b] is the starting node offset of graph b in the
        # concatenated node tensor, so a stored station index i maps to ptr[b] + i.
        batched_data = Batch.from_data_list(list(datas))
        node_embeds, g = self.state_encoder(batched_data, torch.stack(globals_list))
        offsets = batched_data.ptr[:-1]

        def gather(idxs):
            idx_t = torch.as_tensor(idxs, dtype=torch.long)
            valid = (idx_t >= 0).float().unsqueeze(-1)   # -1 = no station (NOOP): zeroed
            return node_embeds[offsets + idx_t.clamp(min=0)] * valid

        q_chosen = self.q_net(gather(p0), gather(s0), g, torch.stack(aux0))
        q_alt = self.q_net(gather(p1), gather(s1), g, torch.stack(aux1))
        # Huber: a branch that overrides a large structural penalty yields a big target;
        # don't let a few of those dominate the gradient.
        loss = F.smooth_l1_loss(q_alt - q_chosen, torch.as_tensor(targets, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps_done += 1
        return float(loss)

    # ---- checkpoints ----

    def save(self, path: str = "checkpoint_v4"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_encoder.state_dict(), os.path.join(path, "state_encoder.pth"))
        torch.save(self.q_net.state_dict(), os.path.join(path, "q_net.pth"))
        with open(os.path.join(path, "params.pkl"), "wb") as f:
            pickle.dump({
                "steps_done": self.steps_done,
                "episodes_done": self.episodes_done,
                "best_score": self.best_score,
            }, f)
        print(f"Saved to {path}/")

    def save_if_best(self, score: float, path: str = "checkpoint_v4_best") -> bool:
        """Save to `path` only if `score` exceeds the agent's best-so-far. Returns True if saved."""
        if score <= self.best_score:
            return False
        self.best_score = score
        self.save(path)
        print(f"  ↑ new best: {score:+.4f}")
        return True

    def load(self, path: str = "checkpoint_v4", verbose: bool = True) -> bool:
        """Load weights from `path` if a compatible checkpoint is there. All-or-nothing:
        a checkpoint from another architecture is skipped entirely, never half-loaded."""
        q_path = os.path.join(path, "q_net.pth")
        se_path = os.path.join(path, "state_encoder.pth")
        p_path = os.path.join(path, "params.pkl")
        if not (os.path.exists(q_path) and os.path.exists(se_path)):
            return False
        try:
            encoder = GNNStateEncoder()
            encoder.load_state_dict(torch.load(se_path, weights_only=True))
            q_net = PairScoringNet(self.state_size, self.state_size, ACTION_AUX_DIM)
            q_net.load_state_dict(torch.load(q_path, weights_only=True))
        except RuntimeError as e:
            print(f"Skipping checkpoint at {path}/ — architecture changed: {e}")
            return False
        self.state_encoder.load_state_dict(encoder.state_dict())
        self.q_net.load_state_dict(q_net.state_dict())
        if os.path.exists(p_path):
            with open(p_path, "rb") as f:
                params = pickle.load(f)
            self.steps_done = params.get("steps_done", 0)
            self.episodes_done = params.get("episodes_done", 0)
            self.best_score = params.get("best_score", float("-inf"))
        if verbose:
            print(f"Loaded weights from {path}/  (episodes={self.episodes_done}, "
                  f"steps={self.steps_done}, best={self.best_score:+.4f})")
        return True
