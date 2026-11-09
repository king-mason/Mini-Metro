"""What the network sees and how it scores: the graph encoding of a design state, the
per-action features, and the two torch modules (GNN state encoder, pair scorer)."""

import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from config import MAX_DIST, MAX_LINES, MAX_TRAINS, MAX_TRAINS_PER_LINE, STATION_CAPACITY, TRAIN_CAPACITY
from metrics import (RHO_CLAMP, STATION_TYPES, bend_cost, close_neighbor_scarcity, expected_demand_gain,
                     expected_unreachable_demand, line_components, nearest_type_distance, neighbor_sets,
                     station_load_factors)
from rules import ActionType
from simulator import MiniMetroSim

NUM_ACTION_TYPES = len(ActionType)

# The encoding has no screen coordinates and no type labels, only quantities that mean the
# same on every map (the game depends only on distances and type counts).
#   nodes:   type share, inbound demand, waiting, lines through it, is an end, distance to
#            the centroid, reachable types, unreachable demand and its distance, scarcity,
#            load factor, distance to the nearest same-type station
#   edges:   built segments and k-nearest-neighbor links, both directions: distance,
#            displacement, is a segment, line one-hot, trains, load, at capacity, and type
#            relations (same type, sender's type unreachable from receiver, same component)
#   globals: lines used, unconnected stations, waiting, mean line length, unreachable demand,
#            scarcity, worst load factor, design phase (so the two NOOPs differ), trains left
NODE_FEAT_DIM = 12
EDGE_FEAT_DIM = 1 + 2 + 1 + MAX_LINES + 3 + 3
GLOBAL_FEAT_DIM = 9
# A symmetry of the square applied to edge displacements: (x sign, y sign, swap axes).
IDENTITY_AUG = (1.0, 1.0, False)
STATE_FEAT_SIZE = 64  # GNN output dimension

# An action is scored from the embeddings of the two stations it touches (see
# action_endpoints) plus a few action-level features no single station carries: type
# one-hot, distance, bend, resulting line load, at capacity, makes a transfer station,
# demand and coverage gained, joins two components.
ACTION_AUX_DIM = NUM_ACTION_TYPES + 8


def build_graph(sim: MiniMetroSim, phase: str = "lines",
                 aug: tuple = IDENTITY_AUG) -> tuple[Data, torch.Tensor]:
    """The state as a graph: (Data with x, edge_index, edge_attr; global features).
    `aug` reflects / rotates the edge displacements, the only orientation-dependent features."""
    n = max(len(sim.stations), 1)
    cap = STATION_CAPACITY

    endpoint_ids: set = set()
    on_line_ids: set = set()
    for line in sim.lines:
        if line.stations:
            endpoint_ids.add(id(line.stations[0]))
            endpoint_ids.add(id(line.stations[-1]))
            on_line_ids.update(id(s) for s in line.stations)
    comp_of, comp_types, _ = line_components(sim)
    load_factors = station_load_factors(sim)
    type_counts = Counter(s.type for s in sim.stations)
    centroid_x = sum(s.x for s in sim.stations) / n
    centroid_y = sum(s.y for s in sim.stations) / n

    # every station sends one unit of demand, split over target types and shared by each
    # type's stations
    inbound = Counter()
    for station_idx in range(len(sim.stations)):
        for t, prob in sim.target_type_probs(station_idx).items():
            inbound[t] += prob
    inbound_per_station = {t: inbound[t] / type_counts[t] for t in type_counts}

    # Node features
    node_feats = []
    node_unreachable = []
    node_scarcity = []
    node_rho = []
    for station_idx, s in enumerate(sim.stations):
        waiting_norm = min(len(s.waiting) / cap, 1.0)
        lines_count = sum(1 for l in sim.lines if s in l.stations) / MAX_LINES
        is_endpoint = 1.0 if id(s) in endpoint_ids else 0.0
        centrality = math.hypot(s.x - centroid_x, s.y - centroid_y) / MAX_DIST
        reachable = comp_types[comp_of[station_idx]]
        unreachable_types = set(STATION_TYPES) - reachable - {s.type}
        expected_unreachable = expected_unreachable_demand(sim, station_idx, reachable)
        nearest_unreachable = nearest_type_distance(sim, station_idx, unreachable_types)
        scarcity = close_neighbor_scarcity(sim, station_idx)
        rho = min(load_factors[station_idx], RHO_CLAMP) / RHO_CLAMP
        nearest_same_type = nearest_type_distance(sim, station_idx, {s.type})
        node_unreachable.append(expected_unreachable)
        node_scarcity.append(scarcity)
        node_rho.append(rho)

        node_feats.append([
            type_counts[s.type] / n,
            inbound_per_station[s.type],
            waiting_norm, lines_count, is_endpoint, centrality,
            len(reachable) / len(type_counts),
            expected_unreachable, nearest_unreachable, scarcity, rho, nearest_same_type,
        ])

    # messages flow src -> dst, so each edge describes the sender as seen from the receiver
    sx, sy, swap = aug
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_attrs: list[list[float]] = []

    def add_edge(src: int, dst: int, is_line: float, line_part: list[float]):
        a, b = sim.stations[src], sim.stations[dst]
        dx = sx * (a.x - b.x) / MAX_DIST
        dy = sy * (a.y - b.y) / MAX_DIST
        if swap:
            dx, dy = dy, dx
        edge_src.append(src)
        edge_dst.append(dst)
        edge_attrs.append(
            [a.pos.get_distance(b.pos) / MAX_DIST, dx, dy, is_line]
            + line_part
            + [
                float(a.type == b.type),
                float(a.type not in comp_types[comp_of[dst]]),
                float(comp_of[src] == comp_of[dst]),
            ]
        )

    for line_idx, line in enumerate(sim.lines):
        line_oh = [0.0] * MAX_LINES
        line_oh[line_idx % MAX_LINES] = 1.0
        n_trains_norm = len(line.trains) / MAX_TRAINS_PER_LINE
        line_waiting = sum(len(st.waiting) for st in line.stations)
        load_raw = line_waiting / max(len(line.trains) * TRAIN_CAPACITY, 1)
        load_norm = min(load_raw, 5.0) / 5.0
        at_cap = 1.0 if load_raw >= 1.0 else 0.0
        line_part = line_oh + [n_trains_norm, load_norm, at_cap]

        idxs = [sim.stations.index(s) for s in line.stations]
        for k in range(len(idxs) - 1):
            add_edge(idxs[k], idxs[k + 1], 1.0, line_part)
            add_edge(idxs[k + 1], idxs[k], 1.0, line_part)

    no_line = [0.0] * (MAX_LINES + 3)
    for dst, nbrs in enumerate(neighbor_sets(sim)):
        for src in sorted(nbrs):
            add_edge(src, dst, 0.0, no_line)

    x = torch.FloatTensor(node_feats)
    if edge_src:
        edge_index = torch.LongTensor([edge_src, edge_dst])
        edge_attr = torch.FloatTensor(edge_attrs)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEAT_DIM))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    disconnected = n - len(on_line_ids)
    total_waiting = sum(len(s.waiting) for s in sim.stations)
    avg_line_len = (sum(len(l.stations) for l in sim.lines) / max(len(sim.lines), 1)) / n
    # worst load factor among served stations (throughput, not coverage)
    connected_rho = [
        load_factors[i] for i in range(len(sim.stations))
        if load_factors[i] != float("inf")
    ]
    max_rho = min(max(connected_rho), RHO_CLAMP) / RHO_CLAMP if connected_rho else 0.0
    global_feats = torch.FloatTensor([
        len(sim.lines) / MAX_LINES,
        disconnected / n,
        min(total_waiting / (n * cap), 1.0),
        min(avg_line_len, 1.0),
        sum(node_unreachable) / max(n, 1),
        sum(node_scarcity) / max(n, 1),
        max_rho,
        1.0 if phase == "trains" else 0.0,
        (MAX_TRAINS - sum(len(l.trains) for l in sim.lines)) / MAX_TRAINS,
    ])

    return data, global_feats


def resulting_line_stations(sim: MiniMetroSim, action: tuple):
    """(line the action changes or None, its stations afterwards)."""
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
    raise ValueError(f"resulting_line_stations on {t}")


def action_endpoints(sim: MiniMetroSim, action: tuple) -> tuple[int, int | None]:
    """(primary, secondary) station: EXTEND_LINE (new station, line end), NEW_LINE (a, b),
    ADD_TRAIN (the line's first and last stations)."""
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
    raise ValueError(f"action_endpoints on {t}")


def action_station_indices(sim: MiniMetroSim, action: tuple) -> tuple[int, int]:
    """Stations whose embeddings score the action; (-1, -1) for NOOP (zero embeddings)."""
    if action[0] == ActionType.NOOP:
        return -1, -1
    return action_endpoints(sim, action)


def action_aux_features(sim: MiniMetroSim, action: tuple) -> torch.Tensor:
    """Action-level features (ACTION_AUX_DIM); per-station ones are in the embeddings."""
    t = action[0]
    type_oh = [0.0] * NUM_ACTION_TYPES
    type_oh[t.value] = 1.0
    if t == ActionType.NOOP:
        return torch.FloatTensor(type_oh + [0.0] * (ACTION_AUX_DIM - NUM_ACTION_TYPES))

    p_idx, s_idx = action_endpoints(sim, action)
    p, s = sim.stations[p_idx], sim.stations[s_idx]

    dist = p.pos.get_distance(s.pos) / MAX_DIST
    bend = bend_cost(sim, action) if t == ActionType.EXTEND_LINE else 0.0

    target_line, resulting_stations = resulting_line_stations(sim, action)
    if resulting_stations:
        waiting_resulting = sum(len(st.waiting) for st in resulting_stations)
        n_trains = len(target_line.trains) if target_line is not None else 1
        load = min(waiting_resulting / max(n_trains * TRAIN_CAPACITY, 1), 5.0) / 5.0
    else:
        load = 0.0
    if target_line is None:
        at_cap = 0.0
    else:
        cur_load = sum(len(st.waiting) for st in target_line.stations) / max(len(target_line.trains) * TRAIN_CAPACITY, 1)
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
        gain_p, gain_s, coverage_gain = expected_demand_gain(sim, p_idx, s_idx)
        demand_gain = gain_p + gain_s
        comp_of, _, _ = line_components(sim)
        bridges = 1.0 if comp_of[p_idx] != comp_of[s_idx] else 0.0

    return torch.FloatTensor(type_oh + [
        dist, bend, load, at_cap,
        creates_transfer, demand_gain, coverage_gain, bridges,
    ])


# ---- GNN state encoder ----

class GNNStateEncoder(nn.Module):
    """Two GATv2 layers give one embedding per station; a context vector combines the global
    features with mean- and max-pooled embeddings, for network-wide choices."""
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
        self.global_proj = nn.Linear(global_dim + 2 * out_dim, out_dim)

    def forward(self, data, global_feats: torch.Tensor):
        """(per-station embeddings, context). For a Batch, embeddings are concatenated
        (offsets in Batch.ptr) and the context is (B, out_dim)."""
        h = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        h = F.relu(self.conv2(h, data.edge_index, data.edge_attr))
        single = data.batch is None
        batch = torch.zeros(h.shape[0], dtype=torch.long) if single else data.batch
        if single:
            global_feats = global_feats.unsqueeze(0)
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        g = F.relu(self.global_proj(torch.cat([global_feats, pooled], dim=-1)))
        return h, (g.squeeze(0) if single else g)


# ---- pair-scoring Q-network ----

class PairScoringNet(nn.Module):
    """Q(s, a): a learned correction to the prior, in log(ticks survived), from the two
    stations' embeddings, the context and the action features. Trained on differences
    between actions in the same state only."""
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
        # zero output at first: an untrained agent plays exactly the heuristic
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_p: torch.Tensor, h_s: torch.Tensor,
                g: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_p, h_s, g, aux], dim=-1)
        return self.net(x).squeeze(-1)
