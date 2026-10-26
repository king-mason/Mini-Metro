"""The structural heuristic: how a design is measured, which moves are considered, and the
hand-written prior that scores them.

Nothing here learns. The prior (structural_reward, coefficients in config.yaml under
`rewards:`) is the baseline policy every learned policy is benchmarked against, and the
metrics double as the features the network sees (model.py).
"""

import numpy as np

from objects import StationType
from rules import MAX_LINES, ActionType, action_segment, can_add_train, crosses_network, line_segments
from simulator import MiniMetroSim, config, current_spawn_chance, current_spawn_interval

WIDTH, HEIGHT = config["resolution"]
MAX_WAITING = config["stations"]["capacity"]
MAX_DIST = (WIDTH ** 2 + HEIGHT ** 2) ** 0.5

STATION_TYPES = list(StationType)
NUM_STATION_TYPES = len(STATION_TYPES)

# --- congestion model (analytic queueing estimate, used at design time) ---
TRAIN_SPEED = config["trains"]["speed"] * config.get("time_scale", 1)
TRAIN_CAPACITY = 6                       # matches Train default capacity in objects.py
_LOAD_REF_TIME = 10_000                  # representative late-game tick (spawning near its hard asymptote)
# Per-station passenger arrival rate (passengers/tick): each station spawns with
# probability spawn_chance every spawn_interval ticks. Evaluated near end-game,
# which is when overcrowd failures actually happen.
LAMBDA_REF = current_spawn_chance(_LOAD_REF_TIME) / current_spawn_interval(_LOAD_REF_TIME)
RHO_CLAMP = 5.0                          # cap the load-factor feature (unserved stations are ~inf)


class R:
    """Structural prior coefficients. Edit values in config.yaml under `rewards:`.

    These bias which designs the agent builds and tries; what it learns from is ticks
    survived in play-outs (see Agent.train_episode). Access via R.<name>. Read once at
    module load — change config.yaml and re-run.
    """
    _cfg = config["rewards"]
    # structural reward -> learned-score units (log ticks survived)
    PRIOR_SCALE               = _cfg["prior_scale"]
    # shaping signals
    ISOLATION_BONUS           = _cfg["isolation_bonus"]
    REACH_DEMAND              = _cfg["reach_demand"]
    REACH_COVERAGE            = _cfg["reach_coverage"]
    SCARCITY_BONUS            = _cfg["scarcity_bonus"]
    BEND_PENALTY              = _cfg["bend_penalty"]
    CONNECTION_COST_SCALE     = _cfg["connection_cost_scale"]
    CONNECTION_COST_CURVATURE = _cfg["connection_cost_curvature"]
    CONGESTION_PENALTY        = _cfg["congestion_penalty"]
    CONGESTION_RELIEF         = _cfg["congestion_relief"]


K_CANDIDATES_DEFAULT = 10
NEIGHBOR_K_DEFAULT = 6   # candidate connections are limited to each station's k nearest neighbors


# ---- state / action encoding ----

def line_components(sim: MiniMetroSim) -> tuple[list[int], dict[int, set], dict[int, list[int]]]:
    """Components induced by shared line membership.

    Isolated stations are their own component. Stations appearing on multiple lines
    naturally merge those lines through the shared station.
    """
    n = len(sim.stations)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for line in sim.lines:
        idxs = [sim.stations.index(s) for s in line.stations]
        for k in range(1, len(idxs)):
            union(idxs[0], idxs[k])

    comp_of = [find(k) for k in range(n)]
    comp_types: dict[int, set] = {}
    comp_members: dict[int, list[int]] = {}
    for k, comp in enumerate(comp_of):
        comp_types.setdefault(comp, set()).add(sim.stations[k].type)
        comp_members.setdefault(comp, []).append(k)
    return comp_of, comp_types, comp_members


def expected_unreachable_demand(sim: MiniMetroSim, station_idx: int, reachable: set) -> float:
    probs = sim.target_type_probs(station_idx)
    return sum(prob for t, prob in probs.items() if t not in reachable)


def nearest_type_distance(sim: MiniMetroSim, station_idx: int, target_types: set) -> float:
    if not target_types:
        return 0.0
    station = sim.stations[station_idx]
    best = min(
        (
            station.pos.get_distance(other.pos)
            for k, other in enumerate(sim.stations)
            if k != station_idx and other.type in target_types
        ),
        default=MAX_DIST,
    )
    return best / MAX_DIST


def close_neighbor_scarcity(sim: MiniMetroSim, station_idx: int, k: int = 3) -> float:
    """Higher means the station has few short geometric options.

    This is static map information. It helps the design-first agent avoid leaving
    awkward stations until the end, where only cross-back connections remain.
    """
    if len(sim.stations) <= 1:
        return 0.0
    station = sim.stations[station_idx]
    dists = sorted(
        station.pos.get_distance(other.pos)
        for j, other in enumerate(sim.stations)
        if j != station_idx
    )
    avg_near = sum(dists[:k]) / min(k, len(dists))
    return min(avg_near / MAX_DIST, 1.0)


def line_length(line) -> float:
    """One-way geometric length of a line (sum of consecutive segment distances)."""
    return sum(
        line.stations[i].pos.get_distance(line.stations[i + 1].pos)
        for i in range(len(line.stations) - 1)
    )


def station_load_factors(sim: MiniMetroSim) -> dict[int, float]:
    """Analytic queueing load factor ρ = arrival_rate / service_rate per station.

    Service rate at a station = (train-visit rate) × train capacity, summed over the
    lines through it. A train on a line of one-way length L with T trains visits an
    interior station at rate T·speed/L and an endpoint at half that (visited once per
    round trip, not twice). ρ ≥ 1 means passengers arrive faster than trains can clear
    them → that station will overcrowd. This is exactly the throughput judgment a human
    makes by eye, made available at design time. Stations with no train service get ρ = ∞.
    """
    n = len(sim.stations)
    idx = {id(s): i for i, s in enumerate(sim.stations)}
    service = [0.0] * n
    for line in sim.lines:
        m = len(line.stations)
        if m < 2 or not line.trains:
            continue
        length = line_length(line)
        if length <= 0:
            continue
        t = len(line.trains)
        for pos, st in enumerate(line.stations):
            is_end = pos == 0 or pos == m - 1
            visit_rate = t * TRAIN_SPEED / length
            if is_end:
                visit_rate *= 0.5
            service[idx[id(st)]] += visit_rate * TRAIN_CAPACITY
    return {
        i: (LAMBDA_REF / service[i] if service[i] > 1e-9 else float("inf"))
        for i in range(n)
    }


def endpoint_load_factor(length: float, n_trains: int) -> float:
    """Load factor at a line endpoint (its worst-served station).

    Endpoints are visited once per round trip, so visit_rate = 0.5·T·speed/length.
    Used for analytic, deepcopy-free congestion shaping at design time.
    """
    if length <= 0 or n_trains <= 0:
        return float("inf")
    service = 0.5 * n_trains * TRAIN_SPEED / length * TRAIN_CAPACITY
    return LAMBDA_REF / service if service > 1e-9 else float("inf")


def bend_cost(sim: MiniMetroSim, action: tuple) -> float:
    """Bend at the junction when extending. 0 if straight, 1.0 if U-turn.

    The new station is appended at the start or end of the line, so the bend
    forms at the previous endpoint between its inner neighbor and the new
    station. Returns 0 if the line had only one station (no prior direction).
    """
    if action[0] != ActionType.EXTEND_LINE:
        return 0.0
    _, line_idx, endpoint_idx, new_idx = action
    stations = sim.lines[line_idx].stations
    if len(stations) < 2:
        return 0.0
    endpoint = stations[endpoint_idx]
    inner = stations[1] if endpoint_idx == 0 else stations[-2]
    new_s = sim.stations[new_idx]
    dx1 = endpoint.pos.x - inner.pos.x
    dy1 = endpoint.pos.y - inner.pos.y
    dx2 = new_s.pos.x - endpoint.pos.x
    dy2 = new_s.pos.y - endpoint.pos.y
    len1 = (dx1 * dx1 + dy1 * dy1) ** 0.5
    len2 = (dx2 * dx2 + dy2 * dy2) ** 0.5
    if len1 == 0 or len2 == 0:
        return 0.0
    cos_a = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
    if cos_a > 1.0:
        cos_a = 1.0
    elif cos_a < -1.0:
        cos_a = -1.0
    # return (1.0 - cos_a) / 2.0
    a = np.arccos(cos_a)
    return 1 / (1 + np.exp(-50 * (a - np.pi / 2)))    # steep sigmoid centered at 90 deg


def expected_demand_gain(sim: MiniMetroSim, i: int, j: int) -> tuple[float, float, float]:
    """Expected demand newly made reachable by connecting station i and j.

    Returns (gain_i_side, gain_j_side, coverage_gain). Works before passengers
    exist (uses target-type probabilities), so it is useful for design-first training.
    """
    comp_of, comp_types, comp_members = line_components(sim)
    ci, cj = comp_of[i], comp_of[j]
    if ci == cj:
        return 0.0, 0.0, 0.0

    types_i = comp_types[ci]
    types_j = comp_types[cj]
    new_for_i = types_j - types_i
    new_for_j = types_i - types_j

    gain_i = sum(
        sum(prob for t, prob in sim.target_type_probs(station_idx).items() if t in new_for_i)
        for station_idx in comp_members[ci]
    )
    gain_j = sum(
        sum(prob for t, prob in sim.target_type_probs(station_idx).items() if t in new_for_j)
        for station_idx in comp_members[cj]
    )
    norm = max(len(sim.stations), 1)
    coverage_gain = (len(new_for_i) + len(new_for_j)) / (2 * NUM_STATION_TYPES)
    return gain_i / norm, gain_j / norm, coverage_gain


def neighbor_sets(sim: MiniMetroSim, neighbor_k: int) -> list[set[int]]:
    """Symmetric k-nearest-neighbor adjacency over station positions.

    Restricts the candidate move graph to geometrically local edges: a connection
    is only offered between stations that are near neighbors, so neither the policy
    nor random exploration can propose an edge spanning the whole map. Symmetric so
    an edge is allowed if *either* endpoint considers the other a neighbor.
    Cached on the sim — station positions never change within an episode.
    """
    cache = getattr(sim, "_neighbor_cache", None)
    if cache is not None and cache[0] == neighbor_k:
        return cache[1]
    n = len(sim.stations)
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        order = sorted(
            (j for j in range(n) if j != i),
            key=lambda j: sim.stations[i].pos.get_distance(sim.stations[j].pos),
        )
        nbrs[i].update(order[:neighbor_k])
    sym = [set(s) for s in nbrs]
    for i in range(n):
        for j in nbrs[i]:
            sym[j].add(i)
    sim._neighbor_cache = (neighbor_k, sym)
    return sym


def get_candidate_actions(sim: MiniMetroSim, k: int = K_CANDIDATES_DEFAULT,
                          phase: str = "lines",
                          neighbor_k: int = NEIGHBOR_K_DEFAULT) -> list[tuple]:
    """Candidate action set for the current sub-phase.

    phase="lines": the k shortest legal EXTEND_LINEs and NEW_LINEs, both restricted to the
        k-nearest-neighbor move graph, plus the shortest of each for every isolated
        station. NOOP is offered once every station is on a line, or once no isolated
        station can be reached by a legal move (forced connections then try further).
    phase="trains": one ADD_TRAIN candidate per line that can take a train (longest
        lines first). NOOP is offered only when no line can: a spare train costs nothing
        in the sim and only adds capacity, so stopping early is never right.
    """
    n = len(sim.stations)
    connected_ids = {id(s) for line in sim.lines for s in line.stations}
    isolated_idxs = [i for i, s in enumerate(sim.stations) if id(s) not in connected_ids]
    neighbors = neighbor_sets(sim, neighbor_k)
    segments = line_segments(sim)
    candidates: list[tuple] = []

    def add_candidate(action: tuple):
        if action not in candidates:
            candidates.append(action)

    if phase == "trains":
        # Train allocation: ADD_TRAIN for each line with capacity; NOOP only once none has.
        open_lines = [li for li in range(len(sim.lines)) if can_add_train(sim, li)]
        for line_idx in sorted(open_lines, key=lambda li: -len(sim.lines[li].stations)):
            add_candidate((ActionType.ADD_TRAIN, line_idx))
        if not candidates:
            add_candidate((ActionType.NOOP,))
        return candidates

    # EXTEND_LINE: for each line, each endpoint, each station not on that line.
    extend_scored: list = []
    for line_idx, line in enumerate(sim.lines):
        if not line.stations:
            continue
        line_set = {id(s) for s in line.stations}
        seen_endpoints: set[int] = set()
        for endpoint_idx in (0, -1):
            ep_id = id(line.stations[endpoint_idx])
            if ep_id in seen_endpoints:
                continue  # 1-station line: front == back
            seen_endpoints.add(ep_id)
            endpoint = line.stations[endpoint_idx]
            endpoint_real = sim.stations.index(endpoint)
            for new_idx in neighbors[endpoint_real]:
                if id(sim.stations[new_idx]) in line_set:
                    continue
                if crosses_network(sim, endpoint_real, new_idx, segments):
                    continue
                dist = endpoint.pos.get_distance(sim.stations[new_idx].pos)
                extend_scored.append((dist, line_idx, endpoint_idx, new_idx))
    extend_scored.sort(key=lambda x: x[0])
    for _, li, ei, ni in extend_scored[:k]:
        add_candidate((ActionType.EXTEND_LINE, li, ei, ni))

    # Rescue candidates: keep top-k distance priority, but always expose the
    # closest extension for each isolated station so the learner can avoid
    # stranding awkward stations until only cross-back routes remain.
    for station_idx in isolated_idxs:
        best = min(
            (row for row in extend_scored if row[3] == station_idx),
            key=lambda x: x[0],
            default=None,
        )
        if best is not None:
            _, li, ei, ni = best
            add_candidate((ActionType.EXTEND_LINE, li, ei, ni))

    # NEW_LINE: only if room for another line. Skip pairs already sharing a line.
    if len(sim.lines) < MAX_LINES:
        same_line_pairs: set[tuple[int, int]] = set()
        for line in sim.lines:
            ids = [sim.stations.index(s) for s in line.stations]
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    lo, hi = sorted((ids[a], ids[b]))
                    same_line_pairs.add((lo, hi))
        new_scored: list = []
        for a in range(n):
            for b in range(a + 1, n):
                if (a, b) in same_line_pairs:
                    continue
                if b not in neighbors[a]:   # only connect spatial neighbors
                    continue
                if crosses_network(sim, a, b, segments):
                    continue
                dist = sim.stations[a].pos.get_distance(sim.stations[b].pos)
                new_scored.append((dist, a, b))
        new_scored.sort(key=lambda x: x[0])
        for _, a, b in new_scored[:k]:
            add_candidate((ActionType.NEW_LINE, a, b))

        # Same rescue rule for starting a new line: every isolated station should
        # have its closest available seed-pair candidate represented.
        for station_idx in isolated_idxs:
            best = min(
                (row for row in new_scored if row[1] == station_idx or row[2] == station_idx),
                key=lambda x: x[0],
                default=None,
            )
            if best is not None:
                _, a, b = best
                add_candidate((ActionType.NEW_LINE, a, b))

    touched = {i for action in candidates for i in action_segment(sim, action)}
    if not any(i in touched for i in isolated_idxs):
        candidates.insert(0, (ActionType.NOOP,))
    return candidates


def connection_cost(dist: float) -> float:
    norm = dist / MAX_DIST
    return R.CONNECTION_COST_SCALE * (norm + R.CONNECTION_COST_CURVATURE * norm * norm)


def structural_reward(sim: MiniMetroSim, action: tuple) -> float:
    """Immediate, intrinsic reward for an action.

    Captures useful static design priors for the design-first setting (crossings are
    illegal, see rules.py, so they need no penalty): keep links short, connect isolated stations, improve expected reachability, and avoid
    leaving scarce/awkward stations until the end.
    Must be called BEFORE apply_action (reads the pre-apply state).
    """
    t = action[0]
    if t == ActionType.EXTEND_LINE:
        _, line_idx, endpoint_idx, new_idx = action
        endpoint_s = sim.lines[line_idx].stations[endpoint_idx]
        new_s = sim.stations[new_idx]
        endpoint_idx_real = sim.stations.index(endpoint_s)
        was_isolated_new = all(new_s not in l.stations for l in sim.lines)
        isolation_bonus = R.ISOLATION_BONUS * was_isolated_new
        gain_a, gain_b, coverage_gain = expected_demand_gain(sim, new_idx, endpoint_idx_real)
        reach_bonus = R.REACH_DEMAND * (gain_a + gain_b) + R.REACH_COVERAGE * coverage_gain
        scarcity_bonus = R.SCARCITY_BONUS * close_neighbor_scarcity(sim, new_idx) * was_isolated_new
        bend_penalty = R.BEND_PENALTY * bend_cost(sim, action)
        dist = endpoint_s.pos.get_distance(new_s.pos)
        action_cost = connection_cost(dist)
        # Congestion: lengthening a line raises its endpoint headway → more starvation.
        line = sim.lines[line_idx]
        resulting_rho = endpoint_load_factor(line_length(line) + dist, len(line.trains))
        congestion_penalty = R.CONGESTION_PENALTY * max(0.0, min(resulting_rho, RHO_CLAMP) - 1.0)
        return (isolation_bonus + reach_bonus + scarcity_bonus
                - action_cost - bend_penalty - congestion_penalty)
    if t == ActionType.NEW_LINE:
        _, a, b = action
        s_a, s_b = sim.stations[a], sim.stations[b]
        was_isolated_a = all(s_a not in l.stations for l in sim.lines)
        was_isolated_b = all(s_b not in l.stations for l in sim.lines)
        isolation_bonus = R.ISOLATION_BONUS * (was_isolated_a + was_isolated_b)
        gain_a, gain_b, coverage_gain = expected_demand_gain(sim, a, b)
        reach_bonus = R.REACH_DEMAND * (gain_a + gain_b) + R.REACH_COVERAGE * coverage_gain
        scarcity_bonus = (
            R.SCARCITY_BONUS * close_neighbor_scarcity(sim, a) * was_isolated_a
            + R.SCARCITY_BONUS * close_neighbor_scarcity(sim, b) * was_isolated_b
        )
        dist = s_a.pos.get_distance(s_b.pos)
        action_cost = connection_cost(dist)
        # A new 2-station line starts with one train; penalize if even that is starved.
        resulting_rho = endpoint_load_factor(dist, 1)
        congestion_penalty = R.CONGESTION_PENALTY * max(0.0, min(resulting_rho, RHO_CLAMP) - 1.0)
        return (isolation_bonus + reach_bonus + scarcity_bonus
                - action_cost - congestion_penalty)
    if t == ActionType.ADD_TRAIN:
        _, line_idx = action
        line = sim.lines[line_idx]
        # Reward proportional to how much this train relieves the line's worst headway.
        length = line_length(line)
        t_now = len(line.trains)
        relief = endpoint_load_factor(length, t_now) - endpoint_load_factor(length, t_now + 1)
        if relief == float("inf"):
            relief = RHO_CLAMP
        return R.CONGESTION_RELIEF * min(relief, RHO_CLAMP)
    return 0.0  # NOOP
