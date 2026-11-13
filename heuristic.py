"""The structural heuristic: which moves are worth considering, and a hand-written prior
that scores them (weights in config.yaml under `rewards:`). Nothing here learns; this is
the baseline every learned policy is benchmarked against."""

from dataclasses import dataclass

from config import MAX_DIST, MAX_LINES, MAX_TRAINS, PRIOR_WEIGHTS
from metrics import (NEIGHBOR_K, RHO_CLAMP, bend_cost, close_neighbor_scarcity, endpoint_load_factor,
                     expected_demand_gain, neighbor_sets, on_a_line)
from rules import NOOP, ActionType, action_segment, can_add_train, crosses_network, line_segments, trains_in_use
from simulator import MiniMetroSim

K_CANDIDATES = 10      # shortest extensions and new lines offered per decision


@dataclass(frozen=True)
class PriorWeights:
    prior_scale: float          # structural reward -> log-ticks units
    congestion_penalty: float
    congestion_relief: float
    isolation_bonus: float
    reach_demand: float
    reach_coverage: float
    scarcity_bonus: float
    bend_penalty: float
    connection_cost_scale: float
    connection_cost_curvature: float


PRIOR = PriorWeights(**PRIOR_WEIGHTS)


# ---- candidate moves ----

def get_candidate_actions(sim: MiniMetroSim, k: int = K_CANDIDATES, phase: str = "lines",
                          neighbor_k: int = NEIGHBOR_K) -> list[tuple]:
    """The moves a policy chooses between.

    "lines": the k shortest legal extensions and k shortest new lines (between near
        neighbors only), plus the shortest of each that reaches every station still on no
        line. NOOP is offered once no unconnected station can be reached.
    "trains": ADD_TRAIN for each line that can take one, longest first; NOOP only when none
        can (a spare train only adds capacity).
    """
    if phase == "trains":
        lines = sorted((li for li in range(len(sim.lines)) if can_add_train(sim, li)),
                       key=lambda li: -len(sim.lines[li].stations))
        return [(ActionType.ADD_TRAIN, li) for li in lines] or [NOOP]

    isolated = [i for i in range(len(sim.stations)) if not on_a_line(sim, i)]
    neighbors = neighbor_sets(sim, neighbor_k)
    segments = line_segments(sim)
    candidates: list[tuple] = []

    def offer(action):
        if action not in candidates:
            candidates.append(action)

    def offer_shortest(scored, touches):
        """For each unconnected station, the shortest option in `scored` reaching it."""
        for i in isolated:
            best = min((row for row in scored if touches(row, i)), key=lambda row: row[0], default=None)
            if best is not None:
                offer(best[1])

    extensions = []    # (length, action)
    for li, line in enumerate(sim.lines):
        on_line = {id(s) for s in line.stations}
        for end in ((0, -1) if len(line.stations) > 1 else (0,)):
            tip = sim.stations.index(line.stations[end])
            for new in neighbors[tip]:
                if id(sim.stations[new]) not in on_line and not crosses_network(sim, tip, new, segments):
                    d = line.stations[end].pos.get_distance(sim.stations[new].pos)
                    extensions.append((d, (ActionType.EXTEND_LINE, li, end, new)))
    extensions.sort(key=lambda row: (row[0], row[1][1:]))     # ties: lowest line, end, station
    for _, action in extensions[:k]:
        offer(action)
    offer_shortest(extensions, lambda row, i: row[1][3] == i)

    if len(sim.lines) < MAX_LINES and trains_in_use(sim) < MAX_TRAINS:   # a new line needs a free train
        share_a_line = {(min(a, b), max(a, b)) for line in sim.lines
                        for a in map(sim.stations.index, line.stations)
                        for b in map(sim.stations.index, line.stations) if a != b}
        new_lines = []
        for a in range(len(sim.stations)):
            for b in range(a + 1, len(sim.stations)):
                if ((a, b) not in share_a_line and b in neighbors[a]
                        and not crosses_network(sim, a, b, segments)):
                    d = sim.stations[a].pos.get_distance(sim.stations[b].pos)
                    new_lines.append((d, (ActionType.NEW_LINE, a, b)))
        new_lines.sort(key=lambda row: (row[0], row[1][1:]))
        for _, action in new_lines[:k]:
            offer(action)
        offer_shortest(new_lines, lambda row, i: i in row[1][1:])

    reached = {i for action in candidates for i in action_segment(sim, action)}
    if not reached & set(isolated):
        candidates.insert(0, NOOP)
    return candidates


# ---- the prior ----

def connection_cost(d: float) -> float:
    x = d / MAX_DIST
    return PRIOR.connection_cost_scale * (x + PRIOR.connection_cost_curvature * x * x)


def congestion_cost(length: float, n_trains: int) -> float:
    """Penalty for a line end loaded past capacity (load factor above 1)."""
    rho = endpoint_load_factor(length, n_trains)
    return PRIOR.congestion_penalty * max(0.0, min(rho, RHO_CLAMP) - 1.0)


# The prior's score is a sum of these terms (penalties negative); a student can reweight them.
PRIOR_TERMS = ("isolation", "reach", "scarcity", "track", "bend", "congestion", "relief")


def joining_terms(sim: MiniMetroSim, a: int, b: int, d: float) -> tuple[float, float, float, float]:
    """(isolation, reach, scarcity, track) for a new segment a-b of length d: connecting
    stations on no line (awkward ones especially), making demand reachable, and the cost
    of the track."""
    isolated = [s for s in (a, b) if not on_a_line(sim, s)]
    gain_a, gain_b, coverage = expected_demand_gain(sim, a, b)
    reach = PRIOR.reach_demand * (gain_a + gain_b) + PRIOR.reach_coverage * coverage
    scarcity = sum(PRIOR.scarcity_bonus * close_neighbor_scarcity(sim, s) for s in isolated)
    return PRIOR.isolation_bonus * len(isolated), reach, scarcity, -connection_cost(d)


def prior_terms(sim: MiniMetroSim, action: tuple) -> tuple[float, ...]:
    """The prior's score for `action` in parts (PRIOR_TERMS), read against the state before
    it is applied."""
    kind = action[0]
    if kind == ActionType.EXTEND_LINE:
        _, li, end, new = action
        line = sim.lines[li]
        tip = sim.stations.index(line.stations[end])
        d = line.stations[end].pos.get_distance(sim.stations[new].pos)
        return (*joining_terms(sim, new, tip, d), -PRIOR.bend_penalty * bend_cost(sim, action),
                -congestion_cost(line.length() + d, len(line.trains)), 0.0)
    if kind == ActionType.NEW_LINE:
        _, a, b = action
        d = sim.stations[a].pos.get_distance(sim.stations[b].pos)
        return (*joining_terms(sim, a, b, d), 0.0, -congestion_cost(d, 1), 0.0)
    if kind == ActionType.ADD_TRAIN:
        # how much the train relieves the line's worst load factor
        line = sim.lines[action[1]]
        relief = (endpoint_load_factor(line.length(), len(line.trains))
                  - endpoint_load_factor(line.length(), len(line.trains) + 1))
        return (0.0,) * 6 + (PRIOR.congestion_relief * min(RHO_CLAMP if relief == float("inf") else relief, RHO_CLAMP),)
    return (0.0,) * len(PRIOR_TERMS)


def structural_reward(sim: MiniMetroSim, action: tuple) -> float:
    """The prior's score for `action`: its terms added in order."""
    return sum(prior_terms(sim, action))
