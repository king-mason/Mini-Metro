from simulator import MiniMetroSim, max_lines, current_spawn_chance, current_spawn_interval
from visualizer import Visualizer
from objects import StationType
from enum import Enum
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
import numpy as np
import random
import math
import os
import pickle
from collections import deque, Counter
import yaml


# TODO
# lines should not bend more than 90 degrees at a station ideally (more difficult for agent to see)


class ActionType(Enum):
    EXTEND_LINE = 0     # (EXTEND_LINE, line_idx, endpoint_idx, new_station_idx)
    NEW_LINE = 1        # (NEW_LINE, station_a, station_b)
    NOOP = 2            # (NOOP,)
    ADD_TRAIN = 3       # (ADD_TRAIN, line_idx)

NUM_ACTION_TYPES = len(ActionType)


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

WIDTH, HEIGHT = config["resolution"]
MAX_WAITING = config["stations"]["capacity"]
MAX_DIST = (WIDTH ** 2 + HEIGHT ** 2) ** 0.5
MAX_TRAINS = config["trains"]["total"]
MAX_TRAINS_PER_LINE = config["trains"]["max_per_line"]

STATION_TYPES = list(StationType)
NUM_STATION_TYPES = len(STATION_TYPES)

SPEED_BONUS_CAP = 300

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
    """Reward / cost coefficients. Edit values in config.yaml under `rewards:`.

    Access via R.<name>. Read once at module load — change config.yaml and re-run.
    """
    _cfg = config["rewards"]
    # return signals
    DELIVERY                  = _cfg["delivery"]
    SPEED_BONUS_MAX           = _cfg["speed_bonus_max"]
    FAILURE_BLAME             = _cfg["failure_blame"]
    OVERCROWD_PENALTY         = _cfg["overcrowd_penalty"]
    OVERCROWD_BLAME           = _cfg["overcrowd_blame"]
    TRANSFER_PENALTY          = _cfg["transfer_penalty"]
    # shaping signals
    ISOLATION_BONUS           = _cfg["isolation_bonus"]
    REACH_DEMAND              = _cfg["reach_demand"]
    REACH_COVERAGE            = _cfg["reach_coverage"]
    SCARCITY_BONUS            = _cfg["scarcity_bonus"]
    CROSSING_PENALTY          = _cfg["crossing_penalty"]
    BEND_PENALTY              = _cfg["bend_penalty"]
    CONNECTION_COST_SCALE     = _cfg["connection_cost_scale"]
    CONNECTION_COST_CURVATURE = _cfg["connection_cost_curvature"]
    ADD_TRAIN_BONUS     = _cfg["add_train_bonus"]
    CONGESTION_PENALTY        = _cfg["congestion_penalty"]
    CONGESTION_RELIEF         = _cfg["congestion_relief"]

# State features: GNN-encoded graph embedding + projected global signals.
# - Per-station nodes carry: type one-hot, per-target-type waiting, total waiting,
#   line-membership count, is-endpoint, normalized position (x, y), reachable
#   destination types for its current component, expected unreachable demand,
#   nearest unreachable destination distance, close-neighbor scarcity, and the
#   analytic load factor (predicted arrival vs. train-service rate).
# - Per-line-segment edges carry: distance, line one-hot, train count, load, at-capacity.
# - 8 global features (lines_used, disconnected, total_waiting, avg_line_len,
#   expected unreachable demand, close-option scarcity, crossings, max load factor)
#   feed the final projection alongside the pooled graph embedding.
NODE_FEAT_DIM = NUM_STATION_TYPES + NUM_STATION_TYPES + 1 + 1 + 1 + 2 + NUM_STATION_TYPES + 3 + 1
EDGE_FEAT_DIM = 1 + max_lines + 1 + 1 + 1                              # = 4 + max_lines
GLOBAL_FEAT_DIM = 8
STATE_FEAT_SIZE = 64  # GNN output dimension

# Action representation (factorized scoring):
#   An action is scored from the GNN node embeddings of the two stations it touches
#   plus a small "aux" vector of action-level signals that are NOT per-node properties
#   (and so aren't already captured by the node embeddings). Per-station context —
#   type, waiting, reachability, scarcity — lives in the node embeddings, so we do NOT
#   re-encode it here.
# Station indices used (primary, secondary), via _action_endpoints:
#   EXTEND_LINE: primary = new station, secondary = line endpoint being extended
#   NEW_LINE:    primary = station_a,  secondary = station_b
#   ADD_TRAIN:   primary = line's first station, secondary = line's last station
#   NOOP:        no stations (both embeddings zeroed)
# Aux layout:
#   action_type one-hot                                                     (= NUM_ACTION_TYPES)
#   distance                                                                (= 1)
#   bend_cost (EXTEND only)                                                 (= 1)
#   crossing_risk                                                           (= 1)
#   resulting_line_load, extends_line_at_capacity                           (= 2)
#   creates_transfer_station                                                (= 1)
#   expected_demand_gain, coverage_gain, bridges_components                 (= 3)
ACTION_AUX_DIM = NUM_ACTION_TYPES + 9

K_CANDIDATES_DEFAULT = 10
NEIGHBOR_K_DEFAULT = 6   # candidate connections are limited to each station's k nearest neighbors


# ---- state / action encoding ----

def _line_components(sim: MiniMetroSim) -> tuple[list[int], dict[int, set], dict[int, list[int]]]:
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


def _expected_unreachable_demand(sim: MiniMetroSim, station_idx: int, reachable: set) -> float:
    probs = sim.target_type_probs(station_idx)
    return sum(prob for t, prob in probs.items() if t not in reachable)


def _nearest_type_distance(sim: MiniMetroSim, station_idx: int, target_types: set) -> float:
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


def _close_neighbor_scarcity(sim: MiniMetroSim, station_idx: int, k: int = 3) -> float:
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


def _line_length(line) -> float:
    """One-way geometric length of a line (sum of consecutive segment distances)."""
    return sum(
        line.stations[i].pos.get_distance(line.stations[i + 1].pos)
        for i in range(len(line.stations) - 1)
    )


def _station_load_factors(sim: MiniMetroSim) -> dict[int, float]:
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
        length = _line_length(line)
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


def _endpoint_load_factor(length: float, n_trains: int) -> float:
    """Load factor at a line endpoint (its worst-served station).

    Endpoints are visited once per round trip, so visit_rate = 0.5·T·speed/length.
    Used for analytic, deepcopy-free congestion shaping at design time.
    """
    if length <= 0 or n_trains <= 0:
        return float("inf")
    service = 0.5 * n_trains * TRAIN_SPEED / length * TRAIN_CAPACITY
    return LAMBDA_REF / service if service > 1e-9 else float("inf")


def _segments(sim: MiniMetroSim) -> list[tuple[int, int]]:
    segments = []
    for line in sim.lines:
        idxs = [sim.stations.index(s) for s in line.stations]
        for i in range(len(idxs) - 1):
            a, b = idxs[i], idxs[i + 1]
            segments.append((a, b))
    return segments


def _ccw(a, b, c) -> bool:
    return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)


def _segments_cross(a, b, c, d) -> bool:
    if a is c or a is d or b is c or b is d:
        return False
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def _candidate_segment(sim: MiniMetroSim, action: tuple) -> tuple[int, int] | None:
    if action[0] == ActionType.EXTEND_LINE:
        _, line_idx, endpoint_idx, new_idx = action
        endpoint = sim.lines[line_idx].stations[endpoint_idx]
        return sim.stations.index(endpoint), new_idx
    if action[0] == ActionType.NEW_LINE:
        return action[1], action[2]
    return None


def _crossing_count_for_action(sim: MiniMetroSim, action: tuple) -> int:
    seg = _candidate_segment(sim, action)
    if seg is None:
        return 0
    a_idx, b_idx = seg
    a, b = sim.stations[a_idx].pos, sim.stations[b_idx].pos
    count = 0
    for c_idx, d_idx in _segments(sim):
        c, d = sim.stations[c_idx].pos, sim.stations[d_idx].pos
        if _segments_cross(a, b, c, d):
            count += 1
    return count


def _bend_cost_for_extend(sim: MiniMetroSim, action: tuple) -> float:
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


def _build_graph(sim: MiniMetroSim) -> tuple[Data, torch.Tensor]:
    """Construct a torch_geometric Data object representing the current sim state.

    Returns (data, global_feats) where:
      - data.x: (n_stations, NODE_FEAT_DIM)
      - data.edge_index: (2, 2*n_segments) [undirected: each segment added twice]
      - data.edge_attr: (2*n_segments, EDGE_FEAT_DIM)
      - global_feats: (GLOBAL_FEAT_DIM,)
    """
    n = max(len(sim.stations), 1)
    cap = MAX_WAITING

    # Identify which stations are endpoints of any line (for is_endpoint feature)
    endpoint_ids: set = set()
    on_line_ids: set = set()
    for line in sim.lines:
        if line.stations:
            endpoint_ids.add(id(line.stations[0]))
            endpoint_ids.add(id(line.stations[-1]))
            on_line_ids.update(id(s) for s in line.stations)
    comp_of, comp_types, _ = _line_components(sim)
    load_factors = _station_load_factors(sim)

    # Node features
    node_feats = []
    node_unreachable = []
    node_scarcity = []
    node_rho = []
    for station_idx, s in enumerate(sim.stations):
        type_oh = [0.0] * NUM_STATION_TYPES
        type_oh[STATION_TYPES.index(s.type)] = 1.0

        target_counter = Counter(p.target_type for p in s.waiting)
        target_feats = [target_counter[t] / cap for t in STATION_TYPES]
        target_feats = [min(v, 1.0) for v in target_feats]

        waiting_norm = min(len(s.waiting) / cap, 1.0)
        lines_count = sum(1 for l in sim.lines if s in l.stations) / max_lines
        is_endpoint = 1.0 if id(s) in endpoint_ids else 0.0
        pos_x = s.x / WIDTH
        pos_y = s.y / HEIGHT
        reachable = comp_types[comp_of[station_idx]]
        reachable_oh = [1.0 if t in reachable else 0.0 for t in STATION_TYPES]
        unreachable_types = set(STATION_TYPES) - reachable - {s.type}
        expected_unreachable = _expected_unreachable_demand(sim, station_idx, reachable)
        nearest_unreachable = _nearest_type_distance(sim, station_idx, unreachable_types)
        scarcity = _close_neighbor_scarcity(sim, station_idx)
        rho = min(load_factors[station_idx], RHO_CLAMP) / RHO_CLAMP  # normalized load factor
        node_unreachable.append(expected_unreachable)
        node_scarcity.append(scarcity)
        node_rho.append(rho)

        node_feats.append(
            type_oh
            + target_feats
            + [waiting_norm, lines_count, is_endpoint, pos_x, pos_y]
            + reachable_oh
            + [expected_unreachable, nearest_unreachable, scarcity, rho]
        )

    # Edge features: per segment, in both directions
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_attrs: list[list[float]] = []
    for line_idx, line in enumerate(sim.lines):
        line_oh = [0.0] * max_lines
        line_oh[line_idx % max_lines] = 1.0
        n_trains_norm = len(line.trains) / MAX_TRAINS_PER_LINE
        line_waiting = sum(len(st.waiting) for st in line.stations)
        load_raw = line_waiting / max(len(line.trains) * 6, 1)
        load_norm = min(load_raw, 5.0) / 5.0
        at_cap = 1.0 if load_raw >= 1.0 else 0.0

        idxs = [sim.stations.index(s) for s in line.stations]
        for k in range(len(idxs) - 1):
            i1, i2 = idxs[k], idxs[k + 1]
            dist_norm = sim.stations[i1].pos.get_distance(sim.stations[i2].pos) / MAX_DIST
            attr = [dist_norm] + line_oh + [n_trains_norm, load_norm, at_cap]
            edge_src.append(i1); edge_dst.append(i2); edge_attrs.append(attr)
            edge_src.append(i2); edge_dst.append(i1); edge_attrs.append(attr)

    x = torch.FloatTensor(node_feats)
    if edge_src:
        edge_index = torch.LongTensor([edge_src, edge_dst])
        edge_attr = torch.FloatTensor(edge_attrs)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEAT_DIM))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Global features
    disconnected = n - len(on_line_ids)
    total_waiting = sum(len(s.waiting) for s in sim.stations)
    avg_line_len = (sum(len(l.stations) for l in sim.lines) / max(len(sim.lines), 1)) / n
    crossing_pairs = 0
    segments = _segments(sim)
    for i, (a_idx, b_idx) in enumerate(segments):
        a, b = sim.stations[a_idx].pos, sim.stations[b_idx].pos
        for c_idx, d_idx in segments[i + 1:]:
            c, d = sim.stations[c_idx].pos, sim.stations[d_idx].pos
            crossing_pairs += int(_segments_cross(a, b, c, d))
    # Worst connected-station load factor: how close the most-starved served station
    # is to overcrowding. Ignores ∞ (unconnected) so it reflects throughput, not coverage.
    connected_rho = [
        load_factors[i] for i in range(len(sim.stations))
        if load_factors[i] != float("inf")
    ]
    max_rho = min(max(connected_rho), RHO_CLAMP) / RHO_CLAMP if connected_rho else 0.0
    global_feats = torch.FloatTensor([
        len(sim.lines) / max_lines,
        disconnected / n,
        min(total_waiting / (n * cap), 1.0),
        min(avg_line_len, 1.0),
        sum(node_unreachable) / max(n, 1),
        sum(node_scarcity) / max(n, 1),
        min(crossing_pairs / max(len(segments), 1), 1.0),
        max_rho,
    ])

    return data, global_feats


# Backwards-compat alias: encode_state returns the same (data, globals) tuple.
encode_state = _build_graph


def _resulting_line_stations(sim: MiniMetroSim, action: tuple):
    """Read-only: returns (target_line_or_None, resulting_stations)."""
    t = action[0]
    if t == ActionType.EXTEND_LINE:
        _, line_idx, endpoint_idx, new_idx = action
        line = sim.lines[line_idx]
        new_station = sim.stations[new_idx]
        if endpoint_idx == 0:
            return line, [new_station] + list(line.stations)
        return line, list(line.stations) + [new_station]
    if t == ActionType.NEW_LINE:
        _, a, b = action
        return None, [sim.stations[a], sim.stations[b]]
    if t == ActionType.ADD_TRAIN:
        _, line_idx = action
        line = sim.lines[line_idx]
        return line, list(line.stations)
    raise ValueError(f"_resulting_line_stations on {t}")


def _action_endpoints(sim: MiniMetroSim, action: tuple) -> tuple[int, int | None]:
    """Returns (primary_idx, secondary_idx_or_None).
    EXTEND_LINE: primary = new station, secondary = endpoint being extended.
    NEW_LINE:    primary = station_a,  secondary = station_b.
    ADD_TRAIN:   primary = first station of line, secondary = last station of line."""
    t = action[0]
    if t == ActionType.EXTEND_LINE:
        _, line_idx, endpoint_idx, new_idx = action
        endpoint_station = sim.lines[line_idx].stations[endpoint_idx]
        return new_idx, sim.stations.index(endpoint_station)
    if t == ActionType.NEW_LINE:
        return action[1], action[2]
    if t == ActionType.ADD_TRAIN:
        _, line_idx = action
        line = sim.lines[line_idx]
        return sim.stations.index(line.stations[0]), sim.stations.index(line.stations[-1])
    raise ValueError(f"_action_endpoints on {t}")


def _type_covered(sim: MiniMetroSim, t: StationType) -> bool:
    return any(s.type == t for line in sim.lines for s in line.stations)


def _expected_demand_gain(sim: MiniMetroSim, i: int, j: int) -> tuple[float, float, float]:
    """Expected demand newly made reachable by connecting station i and j.

    Returns (gain_i_side, gain_j_side, coverage_gain). Works before passengers
    exist (uses target-type probabilities), so it is useful for design-first training.
    """
    comp_of, comp_types, comp_members = _line_components(sim)
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


def _action_station_indices(sim: MiniMetroSim, action: tuple) -> tuple[int, int]:
    """(primary_idx, secondary_idx) used to look up node embeddings.

    Returns (-1, -1) for NOOP (no stations — embeddings are zeroed at scoring time).
    For all other actions both indices are real station indices (see _action_endpoints).
    """
    if action[0] == ActionType.NOOP:
        return -1, -1
    return _action_endpoints(sim, action)


def _action_aux_features(sim: MiniMetroSim, action: tuple) -> torch.Tensor:
    """Small action-level feature vector (ACTION_AUX_DIM).

    Only signals that are NOT per-station properties — those already live in the
    node embeddings and must not be re-encoded. What remains is the action type,
    geometry of the proposed segment, line-level load, and the *pairwise* reachability
    interaction (which is an exact computation worth handing the net directly).
    """
    t = action[0]
    type_oh = [0.0] * NUM_ACTION_TYPES
    type_oh[t.value] = 1.0
    if t == ActionType.NOOP:
        return torch.FloatTensor(type_oh + [0.0] * (ACTION_AUX_DIM - NUM_ACTION_TYPES))

    p_idx, s_idx = _action_endpoints(sim, action)
    p, s = sim.stations[p_idx], sim.stations[s_idx]

    dist = p.pos.get_distance(s.pos) / MAX_DIST
    bend = _bend_cost_for_extend(sim, action) if t == ActionType.EXTEND_LINE else 0.0
    crossing_risk = min(_crossing_count_for_action(sim, action) / max(max_lines, 1), 1.0)

    # Resulting-line load + currently-at-capacity.
    target_line, resulting_stations = _resulting_line_stations(sim, action)
    if resulting_stations:
        waiting_resulting = sum(len(st.waiting) for st in resulting_stations)
        n_trains = len(target_line.trains) if target_line is not None else 1
        load = min(waiting_resulting / max(n_trains * 6, 1), 5.0) / 5.0
    else:
        load = 0.0
    if target_line is None:
        at_cap = 0.0
    else:
        cur_load = sum(len(st.waiting) for st in target_line.stations) / max(len(target_line.trains) * 6, 1)
        at_cap = 1.0 if cur_load >= 1.0 else 0.0

    on_p = any(p in l.stations for l in sim.lines)
    on_s = any(s in l.stations for l in sim.lines)
    if t == ActionType.EXTEND_LINE:
        creates_transfer = 1.0 if on_p else 0.0
    elif t == ActionType.NEW_LINE:
        creates_transfer = 1.0 if (on_p or on_s) else 0.0
    else:  # ADD_TRAIN
        creates_transfer = 0.0

    if t == ActionType.ADD_TRAIN:
        demand_gain = coverage_gain = bridges = 0.0
    else:
        gain_p, gain_s, coverage_gain = _expected_demand_gain(sim, p_idx, s_idx)
        demand_gain = gain_p + gain_s
        comp_of, _, _ = _line_components(sim)
        bridges = 1.0 if comp_of[p_idx] != comp_of[s_idx] else 0.0

    return torch.FloatTensor(type_oh + [
        dist, bend, crossing_risk, load, at_cap,
        creates_transfer, demand_gain, coverage_gain, bridges,
    ])


def _neighbor_sets(sim: MiniMetroSim, neighbor_k: int) -> list[set[int]]:
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

    phase="lines": top-k EXTEND_LINE + top-k NEW_LINE, both restricted to the
        k-nearest-neighbor move graph; NOOP available once all stations are
        connected to at least one line.
    phase="trains": one ADD_TRAIN candidate per under-trained line (load-sorted),
        plus NOOP (always available — agent can stop allocating early).
    """
    n = len(sim.stations)
    connected_ids = {id(s) for line in sim.lines for s in line.stations}
    isolated_idxs = [i for i, s in enumerate(sim.stations) if id(s) not in connected_ids]
    neighbors = _neighbor_sets(sim, neighbor_k)
    candidates: list[tuple] = []

    def add_candidate(action: tuple):
        if action not in candidates:
            candidates.append(action)

    if phase == "trains":
        # Train allocation: NOOP always available; ADD_TRAIN for each line with capacity.
        add_candidate((ActionType.NOOP,))
        if sum(len(l.trains) for l in sim.lines) < MAX_TRAINS:
            train_scored: list = []
            for line_idx, line in enumerate(sim.lines):
                if len(line.trains) < MAX_TRAINS_PER_LINE and len(line.stations) > len(line.trains) + 1:
                    train_scored.append((-len(line.stations), line_idx))
            train_scored.sort()
            for _, line_idx in train_scored:
                add_candidate((ActionType.ADD_TRAIN, line_idx))
        return candidates

    if len(connected_ids) >= n:
        add_candidate((ActionType.NOOP,))

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
    if len(sim.lines) < max_lines:
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

    return candidates


def _apply_add_train(sim: MiniMetroSim, line_idx: int):
    from objects import Train
    if sum([len(l.trains) for l in sim.lines]) >= MAX_TRAINS:
        return None
    line = sim.lines[line_idx]
    if len(line.trains) >= MAX_TRAINS_PER_LINE:
        return None
    if len(line.stations) <= len(line.trains) + 1:
        return None
    train = Train(line)
    # Space the new train as far as possible from existing trains along the line.
    n = len(line.stations)
    if n >= 2 and line.trains:
        existing = [t.position for t in line.trains]
        best_pos = max(range(n), key=lambda p: min(abs(p - e) for e in existing))
        direction = 1 if best_pos < n - 1 else -1
        train.move_to(best_pos, direction)
    line.trains.append(train)
    return train


def _apply_extend(sim: MiniMetroSim, line_idx: int, endpoint_idx: int, new_idx: int):
    line = sim.lines[line_idx]
    new_station = sim.stations[new_idx]
    if endpoint_idx == 0:
        line.stations.insert(0, new_station)
        for train in line.trains:
            train.position += 1
    else:
        line.stations.append(new_station)
    return line


def _connection_cost(dist: float) -> float:
    norm = dist / MAX_DIST
    return R.CONNECTION_COST_SCALE * (norm + R.CONNECTION_COST_CURVATURE * norm * norm)


def _compute_structural_reward(sim: MiniMetroSim, action: tuple) -> float:
    """Immediate, intrinsic reward for an action.

    Captures useful static design priors for the design-first setting: keep links
    short, connect isolated stations, improve expected reachability, and avoid
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
        gain_a, gain_b, coverage_gain = _expected_demand_gain(sim, new_idx, endpoint_idx_real)
        reach_bonus = R.REACH_DEMAND * (gain_a + gain_b) + R.REACH_COVERAGE * coverage_gain
        scarcity_bonus = R.SCARCITY_BONUS * _close_neighbor_scarcity(sim, new_idx) * was_isolated_new
        crossing_penalty = R.CROSSING_PENALTY * _crossing_count_for_action(sim, action)
        bend_penalty = R.BEND_PENALTY * _bend_cost_for_extend(sim, action)
        dist = endpoint_s.pos.get_distance(new_s.pos)
        action_cost = _connection_cost(dist)
        # Congestion: lengthening a line raises its endpoint headway → more starvation.
        line = sim.lines[line_idx]
        resulting_rho = _endpoint_load_factor(_line_length(line) + dist, len(line.trains))
        congestion_penalty = R.CONGESTION_PENALTY * max(0.0, min(resulting_rho, RHO_CLAMP) - 1.0)
        return (isolation_bonus + reach_bonus + scarcity_bonus
                - action_cost - crossing_penalty - bend_penalty - congestion_penalty)
    if t == ActionType.NEW_LINE:
        _, a, b = action
        s_a, s_b = sim.stations[a], sim.stations[b]
        was_isolated_a = all(s_a not in l.stations for l in sim.lines)
        was_isolated_b = all(s_b not in l.stations for l in sim.lines)
        isolation_bonus = R.ISOLATION_BONUS * (was_isolated_a + was_isolated_b)
        gain_a, gain_b, coverage_gain = _expected_demand_gain(sim, a, b)
        reach_bonus = R.REACH_DEMAND * (gain_a + gain_b) + R.REACH_COVERAGE * coverage_gain
        scarcity_bonus = (
            R.SCARCITY_BONUS * _close_neighbor_scarcity(sim, a) * was_isolated_a
            + R.SCARCITY_BONUS * _close_neighbor_scarcity(sim, b) * was_isolated_b
        )
        crossing_penalty = R.CROSSING_PENALTY * _crossing_count_for_action(sim, action)
        dist = s_a.pos.get_distance(s_b.pos)
        action_cost = _connection_cost(dist)
        # A new 2-station line starts with one train; penalize if even that is starved.
        resulting_rho = _endpoint_load_factor(dist, 1)
        congestion_penalty = R.CONGESTION_PENALTY * max(0.0, min(resulting_rho, RHO_CLAMP) - 1.0)
        return (isolation_bonus + reach_bonus + scarcity_bonus
                - action_cost - crossing_penalty - congestion_penalty)
    if t == ActionType.ADD_TRAIN:
        _, line_idx = action
        line = sim.lines[line_idx]
        # Reward proportional to how much this train relieves the line's worst headway.
        length = _line_length(line)
        t_now = len(line.trains)
        relief = _endpoint_load_factor(length, t_now) - _endpoint_load_factor(length, t_now + 1)
        if relief == float("inf"):
            relief = RHO_CLAMP
        return R.CONGESTION_RELIEF * min(relief, RHO_CLAMP)
    return 0.0  # NOOP


def apply_action(sim: MiniMetroSim, action: tuple) -> dict:
    """Apply an action and return the entities it created/modified.

    Returns: {"line": Line | None, "train": Train | None, "extended": bool}
      - NEW_LINE: line + auto-added train, extended=False
      - EXTEND_LINE: the extended line, no new train, extended=True
      - ADD_TRAIN: the new train (or None if capped), no new line
      - NOOP: all None
    """
    t = action[0]
    result = {"line": None, "train": None, "extended": False}
    if t == ActionType.EXTEND_LINE:
        result["line"] = _apply_extend(sim, action[1], action[2], action[3])
        result["extended"] = True
    elif t == ActionType.NEW_LINE:
        prev_n = len(sim.lines)
        sim.add_line([action[1], action[2]])
        if len(sim.lines) > prev_n:
            new_line = sim.lines[-1]
            result["line"] = new_line
            result["train"] = new_line.trains[0] if new_line.trains else None
    elif t == ActionType.ADD_TRAIN:
        result["train"] = _apply_add_train(sim, action[1])
    elif t == ActionType.NOOP:
        pass
    else:
        raise ValueError(f"Unknown action type: {t}")
    return result


def action_label(action: tuple, sim: MiniMetroSim) -> str:
    t = action[0]
    if t == ActionType.EXTEND_LINE:
        _, li, ei, ni = action
        side = "front" if ei == 0 else "back"
        new_s = sim.stations[ni]
        return f"EXTEND L{li}-{side} +{new_s.type.name}({ni})"
    if t == ActionType.NEW_LINE:
        _, a, b = action
        s_a, s_b = sim.stations[a], sim.stations[b]
        return f"NEW_LINE {s_a.type.name}<->{s_b.type.name} ({a},{b})"
    if t == ActionType.ADD_TRAIN:
        _, li = action
        line = sim.lines[li]
        return f"ADD_TRAIN L{li} ({len(line.trains)} trains)"
    if t == ActionType.NOOP:
        return "NOOP"
    return t.name


# ---- GNN state encoder ----

class GNNStateEncoder(nn.Module):
    """Encodes the station graph into per-station embeddings + a global context vector.

    Two GATv2Conv layers with edge features produce one embedding per station (NOT
    pooled — the per-node resolution is what the pair scorer indexes into). The global
    design features are projected separately into a same-sized context vector.
    """
    def __init__(
        self,
        node_dim: int = NODE_FEAT_DIM,
        edge_dim: int = EDGE_FEAT_DIM,
        global_dim: int = GLOBAL_FEAT_DIM,
        hidden: int = 64,
        out_dim: int = STATE_FEAT_SIZE,
    ):
        super().__init__()
        self.conv1 = GATv2Conv(node_dim, hidden, edge_dim=edge_dim, heads=2, concat=True)
        self.conv2 = GATv2Conv(hidden * 2, out_dim, edge_dim=edge_dim, heads=1, concat=False)
        self.global_proj = nn.Linear(global_dim, out_dim)

    def forward(self, data, global_feats: torch.Tensor):
        """data is a single Data or batched Batch. global_feats is (B, GLOBAL_FEAT_DIM)
        for a Batch, or (GLOBAL_FEAT_DIM,) for a single Data.

        Returns (node_embeds, g):
          - node_embeds: (n_nodes, out_dim) — one row per station (batched: all graphs
            concatenated; recover per-graph offsets via Batch.ptr).
          - g: (out_dim,) for a single Data, or (B, out_dim) for a Batch.
        """
        h = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        h = F.relu(self.conv2(h, data.edge_index, data.edge_attr))
        g = F.relu(self.global_proj(global_feats))
        return h, g


# ---- pair-scoring Q-network ----

class PairScoringNet(nn.Module):
    """Scores an action from the embeddings of the two stations it touches.

    Inputs are the primary/secondary node embeddings (zeroed when an action has no
    such station, e.g. NOOP), the global context vector, and a small action-level
    aux vector. Output is a scalar playout Q-value.
    """
    def __init__(self, node_dim: int = STATE_FEAT_SIZE, global_dim: int = STATE_FEAT_SIZE,
                 aux_dim: int = ACTION_AUX_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim * 2 + global_dim + aux_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_p: torch.Tensor, h_s: torch.Tensor,
                g: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_p, h_s, g, aux], dim=-1)
        return self.net(x).squeeze(-1)


# ---- MCTS over design actions ----

class _MCTSNode:
    """A design state in the search tree. Transitions (apply_action) are deterministic,
    so this is a plain tree; the only stochasticity (the play-out) is bootstrapped by
    the Q-net at leaves rather than rolled out."""
    __slots__ = ("sim", "phase", "depth", "action", "struct",
                 "children", "untried", "priors", "value0", "is_terminal", "N", "W")

    def __init__(self, sim, phase, depth, action=None, struct=0.0):
        self.sim = sim
        self.phase = phase
        self.depth = depth
        self.action = action      # action taken from the parent to reach this node
        self.struct = struct      # structural (edge) reward of that action
        self.children: dict = {}
        self.untried: list = []   # candidate actions not yet expanded, best-prior first
        self.priors: dict = {}    # action -> heuristic value  struct(a) + Q(s,a)
        self.value0 = 0.0         # bootstrap state value = max prior (causal value-to-go)
        self.is_terminal = False
        self.N = 0
        self.W = 0.0


# ---- agent ----

class Agent:
    def __init__(
        self,
        num_stations: int = 5,
        memory_size: int = 10_000,
        batch_size: int = 32,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        lr: float = 1e-3,
        target_update_freq: int = 100,
        k_candidates: int = K_CANDIDATES_DEFAULT,
        k_neighbors: int = NEIGHBOR_K_DEFAULT,
        replays_per_step: int = 4,
        max_design_actions: int = 50,
        num_playouts: int = 3,
        mcts_sims: int = 30,
        mcts_c: float = 1.4,
        mcts_horizon: int = 6,
    ):
        self.num_stations = num_stations
        self.k = k_candidates
        self.k_neighbors = k_neighbors
        self.state_size = STATE_FEAT_SIZE
        self.aux_size = ACTION_AUX_DIM
        # MCTS planning over the design phase. mcts_sims=0 falls back to greedy argmax.
        self.mcts_sims = mcts_sims
        self.mcts_c = mcts_c
        self.mcts_horizon = mcts_horizon

        self.memory: deque = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.replays_per_step = replays_per_step
        self.MAX_DESIGN_ACTIONS = max_design_actions
        self.num_playouts = num_playouts
        self.steps_done = 0
        self.last_episode_ticks = 0
        self.best_score = float("-inf")

        self.state_encoder = GNNStateEncoder()
        self.target_state_encoder = GNNStateEncoder()
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())
        self.target_state_encoder.eval()

        self.q_net = PairScoringNet(self.state_size, self.state_size, self.aux_size)
        self.target_net = PairScoringNet(self.state_size, self.state_size, self.aux_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            list(self.state_encoder.parameters()) + list(self.q_net.parameters()),
            lr=lr,
        )

    def _candidate_heuristics(self, sim: MiniMetroSim, phase: str, candidates: list) -> dict:
        """Map each candidate action to its one-step value estimate: the exact
        structural reward plus the Q-net's playout (causal) value-to-go. This is the
        greedy score AND the MCTS prior / leaf-value source — one encode per state."""
        data, globals_ = _build_graph(sim)
        scores: dict = {}
        with torch.no_grad():
            node_embeds, g = self.state_encoder(data, globals_)
            zero = torch.zeros(self.state_size)
            for action in candidates:
                p_idx, s_idx = _action_station_indices(sim, action)
                h_p = node_embeds[p_idx] if p_idx >= 0 else zero
                h_s = node_embeds[s_idx] if s_idx >= 0 else zero
                aux = _action_aux_features(sim, action)
                playout_q = self.q_net(
                    h_p.unsqueeze(0), h_s.unsqueeze(0), g.unsqueeze(0), aux.unsqueeze(0)
                ).item()
                scores[action] = playout_q + _compute_structural_reward(sim, action)
        return scores

    def select_action(self, sim: MiniMetroSim, phase: str = "lines") -> tuple[tuple | None, bool]:
        """Returns (action, was_random). was_random=True when epsilon-sampled."""
        candidates = get_candidate_actions(sim, self.k, phase=phase, neighbor_k=self.k_neighbors)
        if not candidates:
            return None, False
        if random.random() < self.epsilon:
            return random.choice(candidates), True

        # Planned choice: MCTS look-ahead when enabled, else greedy one-step argmax.
        if self.mcts_sims > 0:
            action = self._mcts_search(sim, phase)
            if action is not None:
                return action, False

        scores = self._candidate_heuristics(sim, phase, candidates)
        return max(scores, key=scores.get), False

    # ---- MCTS ----

    def _mcts_make_node(self, sim, phase, depth, action=None, struct=0.0) -> "_MCTSNode":
        """Create a node and populate its priors, bootstrap value, expansion order,
        and terminal flag. One state encode per node."""
        node = _MCTSNode(sim, phase, depth, action=action, struct=struct)
        # Terminal: the design (for this phase) ends after a NOOP, or we hit the
        # look-ahead horizon. Terminal nodes are still valued but never expanded.
        if (action is not None and action[0] == ActionType.NOOP) or depth >= self.mcts_horizon:
            node.is_terminal = True
        candidates = get_candidate_actions(sim, self.k, phase=phase, neighbor_k=self.k_neighbors)
        if not candidates:
            return node
        node.priors = self._candidate_heuristics(sim, phase, candidates)
        # State value = value of the best continuation = max_a (struct(a) + Q(s,a)).
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
                child_sim = copy.deepcopy(node.sim)
                apply_action(child_sim, action)
                child = self._mcts_make_node(
                    child_sim, phase, node.depth + 1,
                    action=action, struct=_compute_structural_reward(node.sim, action),
                )
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

    def remember(self, data, globals_, p_idx, s_idx, aux, return_):
        """Store one Monte-Carlo transition. The graph is re-encoded at replay time
        and the two station embeddings are looked up by (p_idx, s_idx); -1 means
        'no station' (zeroed). return_ is the full MC return (playout component only)."""
        self.memory.append((data, globals_, p_idx, s_idx, aux, return_))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        datas, globals_list, p_idxs, s_idxs, auxs, returns = zip(*batch)

        # Batch graphs; Batch.ptr[b] is the starting node offset of graph b in the
        # concatenated node tensor, so a stored station index i maps to ptr[b] + i.
        batched_data = Batch.from_data_list(list(datas))
        globals_t = torch.stack(globals_list)
        node_embeds, g = self.state_encoder(batched_data, globals_t)  # (sum_n, d), (B, d)

        offsets = batched_data.ptr[:-1]                  # (B,)
        p_idx_t = torch.as_tensor(p_idxs, dtype=torch.long)
        s_idx_t = torch.as_tensor(s_idxs, dtype=torch.long)
        p_valid = (p_idx_t >= 0).float().unsqueeze(-1)   # zero out 'no station' rows
        s_valid = (s_idx_t >= 0).float().unsqueeze(-1)
        h_p = node_embeds[offsets + p_idx_t.clamp(min=0)] * p_valid   # (B, d)
        h_s = node_embeds[offsets + s_idx_t.clamp(min=0)] * s_valid   # (B, d)
        aux_t = torch.stack(auxs)

        current_q = self.q_net(h_p, h_s, g, aux_t)
        target_q = torch.as_tensor(returns, dtype=torch.float32)  # pure MC: target = return

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())

    def train_episode(self, max_ticks: int = 50_000, visualizer: Visualizer | None=None, 
                      map_seed: int | None = None, learn: bool = True) -> float:
        """Run one design + play-out episode. When learn=False (evaluation), the
        transition buffer, replay updates, and epsilon decay are all skipped — the
        episode only measures performance on the given map."""
        if map_seed is None:
            map_seed = random.randint(0, 10000)
        if visualizer:
            visualizer.set_game_seed(map_seed)
        
        if map_seed is not None:
            rng_state = random.getstate()
            random.seed(map_seed)
            sim = MiniMetroSim(self.num_stations)
            random.setstate(rng_state)
        else:
            sim = MiniMetroSim(self.num_stations)
        self.last_episode_ticks = 0

        # Action-id bookkeeping lives directly on the sim objects:
        #   line.extension_action_ids — list of agent action_ids that built this line
        #   train.creator_action_id  — the action_id that created this train
        # Deep-copy preserves these attributes, so Phase B reads them with no remapping.

        pending_actions: list = []
        design_skip = False  # set True if user presses ESC during design viz

        # ---- Phase A1: Line design. NEW_LINE / EXTEND_LINE / NOOP only. ----
        for action_id in range(self.MAX_DESIGN_ACTIONS):
            data, globals_ = _build_graph(sim)
            action, was_random = self.select_action(sim, phase="lines")
            if action is None or action[0] == ActionType.NOOP:
                pending_actions.append({
                    "data": data, "globals": globals_,
                    "p_idx": -1, "s_idx": -1,
                    "aux": _action_aux_features(sim, (ActionType.NOOP,)),
                    "action_id": action_id,
                    "structural_reward": 0.0,
                    "causal_reward": 0.0,
                })
                if visualizer and not design_skip:
                    design_skip = visualizer.draw_design_step(
                        sim, "NOOP (end design)", action_id + 1,
                        structural_reward=0.0, was_random=was_random,
                    )
                break

            structural = _compute_structural_reward(sim, action)
            p_idx, s_idx = _action_station_indices(sim, action)
            aux = _action_aux_features(sim, action)
            created = apply_action(sim, action)

            if created["train"] is not None:
                created["train"].creator_action_id = action_id
            if created["line"] is not None:
                if not created["extended"]:
                    created["line"].extension_action_ids = [action_id]
                else:
                    created["line"].extension_action_ids.append(action_id)

            pending_actions.append({
                "data": data, "globals": globals_,
                "p_idx": p_idx, "s_idx": s_idx, "aux": aux,
                "action_id": action_id,
                "structural_reward": structural,
                "causal_reward": 0.0,
            })

            if visualizer and not design_skip:
                design_skip = visualizer.draw_design_step(
                    sim, action_label(action, sim), action_id + 1,
                    structural_reward=structural, was_random=was_random,
                )

        # ---- Connectivity enforcement: force-connect any stations the agent left isolated. ----
        # Runs after Phase A (budget exhausted or NOOP). Greedily extends the nearest
        # existing endpoint to each isolated station. Not agent choices, so they don't
        # enter pending_actions; the line's existing credit map covers any deliveries.
        connected_ids = {id(s) for l in sim.lines for s in l.stations}
        isolated = [i for i, s in enumerate(sim.stations) if id(s) not in connected_ids]
        for iso_idx in isolated:
            if not sim.lines:
                break
            best = min(
                (
                    sim.lines[li].stations[ep_pos].pos.get_distance(sim.stations[iso_idx].pos),
                    li,
                    ep_pos,
                )
                for li, line in enumerate(sim.lines)
                for ep_pos in (0, len(line.stations) - 1)
                if id(sim.stations[iso_idx]) not in {id(s) for s in line.stations}
            )
            _, li, ep_pos = best
            endpoint_idx = 0 if ep_pos == 0 else -1
            forced_action = (ActionType.EXTEND_LINE, li, endpoint_idx, iso_idx)
            apply_action(sim, forced_action)
            if visualizer and not design_skip:
                design_skip = visualizer.draw_design_step(
                    sim, f"FORCED: {action_label(forced_action, sim)}", len(pending_actions) + 1,
                    structural_reward=None, was_random=False,
                )

        # ---- Phase A2: Train allocation. Only ADD_TRAIN / NOOP candidates. ----
        # Lines are final; agent distributes the train pool across them.
        # Budget cap: enough room to fill the entire train pool plus one stop signal.
        train_budget = MAX_TRAINS + 1
        for _ in range(train_budget):
            data, globals_ = _build_graph(sim)
            action, was_random = self.select_action(sim, phase="trains")
            action_id = len(pending_actions)
            if action is None or action[0] == ActionType.NOOP:
                pending_actions.append({
                    "data": data, "globals": globals_,
                    "p_idx": -1, "s_idx": -1,
                    "aux": _action_aux_features(sim, (ActionType.NOOP,)),
                    "action_id": action_id,
                    "structural_reward": 0.0,
                    "causal_reward": 0.0,
                })
                if visualizer and not design_skip:
                    design_skip = visualizer.draw_design_step(
                        sim, "NOOP (end trains)", action_id + 1,
                        structural_reward=0.0, was_random=was_random,
                    )
                break

            structural = _compute_structural_reward(sim, action)
            p_idx, s_idx = _action_station_indices(sim, action)
            aux = _action_aux_features(sim, action)
            created = apply_action(sim, action)

            if created["train"] is not None:
                created["train"].creator_action_id = action_id

            pending_actions.append({
                "data": data, "globals": globals_,
                "p_idx": p_idx, "s_idx": s_idx, "aux": aux,
                "action_id": action_id,
                "structural_reward": structural,
                "causal_reward": 0.0,
            })

            if visualizer and not design_skip:
                design_skip = visualizer.draw_design_step(
                    sim, action_label(action, sim), action_id + 1,
                    structural_reward=structural, was_random=was_random,
                )

        # ---- Phase B: Multiple play-outs. Average causal credit across runs. ----
        # Each play-out gets a fresh deep-copy of the post-design sim.
        # Action-ids live on Line/Train objects, so deep-copy preserves them — no remapping.
        id_to_idx = {p["action_id"]: i for i, p in enumerate(pending_actions)}
        total_ticks = 0

        for play_idx in range(self.num_playouts):
            play_sim = copy.deepcopy(sim)
            ticks = 0
            lost = False
            run_causal: dict[int, float] = {}

            # Lines are fixed during a play-out, so precompute station → serving
            # action_ids once for progressive overcrowd blame. Deep-copy preserves
            # reference identity within play_sim, so id(station) matches across
            # play_sim.stations and play_sim.lines[*].stations.
            # Include the line's TRAIN-creating actions, not just its extensions: an
            # overcrowded station is usually a throughput failure (too few trains), so
            # ADD_TRAIN / under-allocation must share the blame — otherwise the only
            # lesson learned is "build this line less", never "give it more trains".
            station_serving: dict[int, set[int]] = {}
            for line in play_sim.lines:
                aids = {aid for aid in getattr(line, "extension_action_ids", ()) if aid in id_to_idx}
                aids |= {getattr(t, "creator_action_id", None) for t in line.trains}
                aids = {aid for aid in aids if aid is not None and aid in id_to_idx}
                if not aids:
                    continue
                for st in line.stations:
                    station_serving.setdefault(id(st), set()).update(aids)

            while ticks < max_ticks and not lost:
                _, _, lost, delivered_passengers = play_sim.step()
                for psg in delivered_passengers:
                    # Train credit set: action_ids that created the trains this passenger rode.
                    train_credit = {getattr(t, "creator_action_id", None) for t in psg.trains_ridden}
                    train_credit = {aid for aid in train_credit if aid is not None and aid in id_to_idx}

                    # Line credit set: action_ids that built the lines this passenger rode.
                    line_credit = set()
                    for line in psg.lines_ridden:
                        for aid in getattr(line, "extension_action_ids", ()):
                            if aid in id_to_idx:
                                line_credit.add(aid)

                    if not train_credit and not line_credit:
                        continue

                    wait = ticks - psg.journey_start_time
                    speed_bonus = R.SPEED_BONUS_MAX * max(0.0, 1.0 - wait / SPEED_BONUS_CAP)
                    # Each line-change is a transfer; discourage transfer-heavy routes.
                    n_transfers = max(0, len(psg.lines_ridden) - 1)
                    pool = max(0.0, R.DELIVERY + speed_bonus - R.TRANSFER_PENALTY * n_transfers)

                    # Two independent pools: trains split one pool among themselves,
                    # line-extensions split another pool among themselves.
                    if train_credit:
                        share = pool / len(train_credit)
                        for aid in train_credit:
                            run_causal[aid] = run_causal.get(aid, 0.0) + share
                    if line_credit:
                        share = pool / len(line_credit)
                        for aid in line_credit:
                            run_causal[aid] = run_causal.get(aid, 0.0) + share

                # Progressive overcrowd blame: as queues build, charge the lines that
                # serve each crowded station a small per-tick penalty — so the design
                # decisions responsible for chronic crowding are credited directly,
                # rather than only on terminal failure.
                for station in play_sim.stations:
                    nw = len(station.waiting)
                    if nw > MAX_WAITING:
                        pen = R.OVERCROWD_BLAME
                    elif nw > MAX_WAITING / 2:
                        pen = R.OVERCROWD_BLAME * 0.5
                    else:
                        continue
                    serving = station_serving.get(id(station))
                    if not serving:
                        continue
                    share = pen / len(serving)
                    for aid in serving:
                        run_causal[aid] = run_causal.get(aid, 0.0) - share

                ticks += 1
                if visualizer and play_idx == self.num_playouts - 1:
                    visualizer.draw(play_sim, epsilon=self.epsilon)

            total_ticks += ticks

            # Accumulate averaged causal reward.
            for aid, cr in run_causal.items():
                if aid in id_to_idx:
                    pending_actions[id_to_idx[aid]]["causal_reward"] += cr / self.num_playouts

            if lost:
                blame_weights: dict[int, float] = {}
                noop_id = pending_actions[-1]["action_id"]
                for station in play_sim.stations:
                    n_waiting = len(station.waiting)
                    if n_waiting == 0:
                        continue
                    serving_aids = station_serving.get(id(station), set())
                    if serving_aids:
                        per_aid = n_waiting / len(serving_aids)
                        for aid in serving_aids:
                            blame_weights[aid] = blame_weights.get(aid, 0.0) + per_aid
                    else:
                        blame_weights[noop_id] = blame_weights.get(noop_id, 0.0) + n_waiting

                total_weight = sum(blame_weights.values())
                if total_weight > 0:
                    for aid, weight in blame_weights.items():
                        if aid in id_to_idx:
                            pending_actions[id_to_idx[aid]]["causal_reward"] += (
                                -R.FAILURE_BLAME * (weight / total_weight) / self.num_playouts
                            )
                else:
                    blame = -R.FAILURE_BLAME / max(1, len(pending_actions)) / self.num_playouts
                    for p in pending_actions:
                        p["causal_reward"] += blame

        # ---- Phase C: Monte Carlo returns; flush to replay buffer. ----
        # The Q-network learns only playout (causal) value. Structural reward is
        # computed exactly at selection time and added to the Q-score there, so it
        # must NOT also be baked into the learned target — that would double-count it.
        G = 0.0
        for i in reversed(range(len(pending_actions))):
            r_i = pending_actions[i]["causal_reward"]   # structural handled at selection time
            G = r_i + self.gamma * G
            pending_actions[i]["return"] = G

        if learn:
            for p in pending_actions:
                self.remember(
                    p["data"], p["globals"], p["p_idx"], p["s_idx"], p["aux"], p["return"],
                )

        total_reward = sum(p["structural_reward"] + p["causal_reward"] for p in pending_actions)

        if visualizer:
            visualizer.update_last_reward(round(total_reward, 2))

        if learn:
            for _ in range(max(1, len(pending_actions) * self.replays_per_step)):
                self.replay()

        self.last_episode_ticks = total_ticks // self.num_playouts

        if learn and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_reward

    def evaluate(self, test_seeds: list[int]) -> tuple[float, float]:
        """Greedy (epsilon=0) performance on held-out maps the agent never trains on.

        No buffer writes, replay, or epsilon decay — purely a measurement. Returns
        (mean_reward, mean_ticks) across the given seeds."""
        saved_eps = self.epsilon
        self.epsilon = 0.0
        rewards, ticks = [], []
        try:
            for seed in test_seeds:
                rewards.append(self.train_episode(map_seed=seed, learn=False))
                ticks.append(self.last_episode_ticks)
        finally:
            self.epsilon = saved_eps
        return float(np.mean(rewards)), float(np.mean(ticks))

    def save(self, path: str = "checkpoint_v3"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_encoder.state_dict(), os.path.join(path, "state_encoder.pth"))
        torch.save(self.target_state_encoder.state_dict(), os.path.join(path, "target_state_encoder.pth"))
        torch.save(self.q_net.state_dict(), os.path.join(path, "q_net.pth"))
        torch.save(self.target_net.state_dict(), os.path.join(path, "target_net.pth"))
        with open(os.path.join(path, "params.pkl"), "wb") as f:
            pickle.dump({
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
                "best_score": getattr(self, "best_score", float("-inf")),
            }, f)
        print(f"Saved to {path}/")

    def save_if_best(self, score: float, path: str = "checkpoint_v3_best") -> bool:
        """Save to `path` only if `score` exceeds the agent's best-so-far. Returns True if saved."""
        if not hasattr(self, "best_score"):
            self.best_score = float("-inf")
        if score <= self.best_score:
            return False
        self.best_score = score
        self.save(path)
        print(f"  ↑ new best: {score:.1f}")
        return True

    def load(self, path: str = "checkpoint_v3"):
        q_path = os.path.join(path, "q_net.pth")
        t_path = os.path.join(path, "target_net.pth")
        se_path = os.path.join(path, "state_encoder.pth")
        tse_path = os.path.join(path, "target_state_encoder.pth")
        p_path = os.path.join(path, "params.pkl")
        if os.path.exists(q_path):
            try:
                self.q_net.load_state_dict(torch.load(q_path, weights_only=True))
                self.target_net.load_state_dict(torch.load(t_path, weights_only=True))
                if os.path.exists(se_path):
                    self.state_encoder.load_state_dict(torch.load(se_path, weights_only=True))
                    self.target_state_encoder.load_state_dict(torch.load(tse_path, weights_only=True))
                print(f"Loaded weights from {path}/")
            except RuntimeError as e:
                print(f"Skipping checkpoint at {path}/ — architecture changed (delete to retrain): {e}")
                return
        if os.path.exists(p_path):
            with open(p_path, "rb") as f:
                params = pickle.load(f)
            self.epsilon = params.get("epsilon", self.epsilon)
            self.steps_done = params.get("steps_done", 0)
            self.best_score = params.get("best_score", float("-inf"))
            print(f"  epsilon={self.epsilon:.4f}  steps={self.steps_done}  best={self.best_score:.1f}")
