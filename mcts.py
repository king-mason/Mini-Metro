"""Monte Carlo tree search over design actions: optional look-ahead for the policy.

Transitions are deterministic, so this is a plain tree. A path's return is the sum of the
prior's rewards along it plus the policy's best score at the leaf (no play-outs).
"""

import copy
import math

from rules import ActionType, apply_action


class _Node:
    __slots__ = ("sim", "depth", "reward", "action", "children", "untried", "scores", "value",
                 "terminal", "visits", "total")

    def __init__(self, sim, depth, action=None, reward=0.0):
        self.sim = sim
        self.depth = depth
        self.action = action        # the action that led here
        self.reward = reward        # the prior's reward for that action
        self.children: dict = {}
        self.untried: list = []     # actions not expanded yet, best first
        self.scores: dict = {}      # action -> policy score
        self.value = 0.0            # bootstrap value: the best score from here
        self.terminal = False
        self.visits = 0
        self.total = 0.0


class MCTS:
    """scores(sim, phase, candidates) -> {action: score}; reward(sim, action) -> the prior's
    part of a score; candidates(sim, phase) -> actions."""

    def __init__(self, scores, reward, candidates, sims: int, c: float = 1.4, horizon: int = 6):
        self.scores, self.reward, self.candidates = scores, reward, candidates
        self.sims, self.c, self.horizon = sims, c, horizon

    def plan(self, sim, phase: str):
        """The most-visited first action from `sim` (None if there are no candidates)."""
        root = self._node(copy.deepcopy(sim), phase, 0)
        if not root.scores:
            return None
        if len(root.scores) == 1:
            return next(iter(root.scores))
        lo, hi = float("inf"), float("-inf")          # range of returns, to normalize UCB
        for _ in range(self.sims):
            node, path = root, [root]
            while not node.terminal and not node.untried and node.children:
                node = self._select(node, lo, hi)
                path.append(node)
            if not node.terminal and node.untried:
                node = self._expand(node, phase)
                path.append(node)
            g = sum(n.reward for n in path[1:]) + node.value
            lo, hi = min(lo, g), max(hi, g)
            for n in path:
                n.visits += 1
                n.total += g
        best = max(root.children.values(), key=lambda n: (n.visits, n.total / max(n.visits, 1)))
        return best.action

    def _node(self, sim, phase, depth, action=None, reward=0.0, stop_value=None) -> _Node:
        node = _Node(sim, depth, action, reward)
        if stop_value is not None:
            # NOOP finishes the design: its value is the parent's score for stopping
            node.terminal, node.value = True, stop_value
            return node
        node.terminal = depth >= self.horizon
        candidates = self.candidates(sim, phase)
        if candidates:
            node.scores = self.scores(sim, phase, candidates)
            node.value = max(node.scores.values())
            if not node.terminal:
                node.untried = sorted(node.scores, key=node.scores.get, reverse=True)
        return node

    def _expand(self, node: _Node, phase: str) -> _Node:
        action = node.untried.pop(0)
        reward = self.reward(node.sim, action)
        if action[0] == ActionType.NOOP:
            child = self._node(node.sim, phase, node.depth + 1, action, reward,
                               stop_value=node.scores[action] - reward)
        else:
            sim = copy.deepcopy(node.sim)
            apply_action(sim, action)
            child = self._node(sim, phase, node.depth + 1, action, reward)
        node.children[action] = child
        return child

    def _select(self, node: _Node, lo: float, hi: float) -> _Node:
        span = hi - lo if hi > lo else 1.0
        log_n = math.log(node.visits + 1)
        return max(node.children.values(),
                   key=lambda n: (n.total / n.visits - lo) / span + self.c * math.sqrt(log_n / n.visits))
