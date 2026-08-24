"""
Clean VRPTW environment for pure RL — no zones, no LLM, no events.

MDP:
  - Multi-vehicle sequential control: active vehicle = argmin(cur_time) among non-retired
  - Action 0         : retire active vehicle (return to depot)
  - Action 1..N      : visit customer c
  - State            : per-stop features for ALL N+1 nodes + vehicle state + global state
  - Feasibility mask : checked every step (TW + capacity + depot-return constraint)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ============================================================
# Instance
# ============================================================
@dataclass
class VRPTWInstance:
    name: str
    coords: np.ndarray        # [n_nodes, 2]  node 0 = depot
    demands: np.ndarray       # [n_nodes]
    tw_open: np.ndarray       # [n_nodes]
    tw_close: np.ndarray      # [n_nodes]
    service_time: np.ndarray  # [n_nodes]
    vehicle_capacity: float
    vehicle_limit: int
    n_customers: int          # n_nodes - 1


def load_solomon(path: str) -> VRPTWInstance:
    """Parse standard Solomon VRPTW benchmark text format."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    name = lines[0]

    v_idx = next(i for i, l in enumerate(lines) if l.upper().startswith("VEHICLE"))
    vparts = lines[v_idx + 2].split() if lines[v_idx + 1].split()[0].isalpha() else lines[v_idx + 1].split()
    # handle optional header row after VEHICLE
    for offset in (1, 2):
        parts = lines[v_idx + offset].split()
        try:
            vehicle_limit = int(parts[0])
            vehicle_capacity = float(parts[1])
            break
        except (ValueError, IndexError):
            continue

    c_idx = next(i for i, l in enumerate(lines) if l.upper().startswith("CUSTOMER"))
    rows: List[List[float]] = []
    for ln in lines[c_idx + 1:]:
        if any(ln.upper().startswith(k) for k in ("EVENTS", "EVENT", "VEHICLE")):
            break
        parts = ln.split()
        if len(parts) >= 7:
            try:
                rows.append([float(x) for x in parts[:7]])
            except ValueError:
                continue  # skip header rows like "CUST NO. XCOORD ..."

    data = np.array(rows, dtype=np.float64)
    n_nodes = len(data)

    return VRPTWInstance(
        name=name,
        coords=data[:, 1:3],
        demands=data[:, 3],
        tw_open=data[:, 4],
        tw_close=data[:, 5],
        service_time=data[:, 6],
        vehicle_capacity=vehicle_capacity,
        vehicle_limit=vehicle_limit,
        n_customers=n_nodes - 1,
    )


def build_tt(inst: VRPTWInstance, speed: float = 1.0) -> np.ndarray:
    xy = inst.coords.astype(np.float64)
    n = xy.shape[0]
    tt = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dx = xy[:, 0] - xy[i, 0]
        dy = xy[:, 1] - xy[i, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        tt[i] = (np.floor(dist / max(speed, 1e-8) * 10.0) / 10.0).astype(np.float32)
    return tt


# ============================================================
# Observation layout constants
# ============================================================
# Per stop: x, y, tw_open, tw_close, service_time, demand,
#           served, is_cur_node, travel_time, end_slack, depot_return_slack
STOP_FEAT_DIM = 11

# Vehicle: cur_time, remaining_cap, vehicles_left_ratio
VEHICLE_FEAT_DIM = 3

# Global: time_remaining_ratio, served_ratio
GLOBAL_FEAT_DIM = 2


# ============================================================
# Vehicle state
# ============================================================
@dataclass
class VehicleState:
    cur_node: int = 0
    cur_time: float = 0.0
    remaining_cap: float = 0.0
    route: List[int] = field(default_factory=lambda: [0])
    retired: bool = False
    activated: bool = False


# ============================================================
# Environment
# ============================================================
class VRPTWEnv(gym.Env):
    """
    Pure VRPTW gym environment.

    Vehicles are controlled one at a time in order of minimum current time
    (matches how a real dispatcher would allocate the next free vehicle).
    The agent issues one action per step for the active vehicle.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        inst: VRPTWInstance,
        tt: np.ndarray,
        visit_reward: float = 2.0,
        unserved_penalty: float = 20.0,
        vehicle_use_penalty: float = 10.0,
        dist_weight: float = 3.0,
    ):
        super().__init__()
        self.inst = inst
        self.tt = np.asarray(tt, dtype=np.float32)
        self.N = inst.n_customers
        self.max_vehicles = inst.vehicle_limit

        self.visit_reward = float(visit_reward)
        self.unserved_penalty = float(unserved_penalty)
        self.vehicle_use_penalty = float(vehicle_use_penalty)
        self.dist_weight = float(dist_weight)

        self._T = float(inst.tw_close[0])                # depot close = time horizon
        self._cap = float(inst.vehicle_capacity)
        self._coord_scale = float(max(np.max(np.abs(inst.coords)), 1.0))

        n_stops = self.N + 1
        obs_dim = n_stops * STOP_FEAT_DIM + VEHICLE_FEAT_DIM + GLOBAL_FEAT_DIM

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.N + 1)

        # initialized in reset()
        self.vehicles: List[VehicleState] = []
        self.served: np.ndarray = np.zeros(self.N + 1, dtype=np.uint8)
        self.active_idx: int = 0
        self.total_distance: float = 0.0
        self.late_count: int = 0

        self.reset()

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    def _nt(self, v: float) -> float:
        return float(v / self._T)

    def _nc(self, v: float) -> float:
        return float(v / self._coord_scale)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _av(self) -> VehicleState:
        return self.vehicles[self.active_idx]

    def _next_active(self) -> int:
        best, best_t = -1, float("inf")
        for i, v in enumerate(self.vehicles):
            if not v.retired and v.cur_time < best_t:
                best, best_t = i, v.cur_time
        return best

    def _retire(self, idx: int) -> float:
        v = self.vehicles[idx]
        travel = 0.0
        if v.cur_node != 0:
            travel = float(self.tt[v.cur_node, 0])
            v.cur_time += travel
            v.cur_node = 0
            v.route.append(0)
            self.total_distance += travel
        v.retired = True
        return travel

    def _all_served(self) -> bool:
        return bool(np.all(self.served[1:] == 1))

    @property
    def vehicles_used(self) -> int:
        return sum(1 for v in self.vehicles if v.activated)

    # ------------------------------------------------------------------
    # Feasibility mask
    # ------------------------------------------------------------------
    def get_feasible_mask(self) -> np.ndarray:
        """
        Boolean mask of shape [N+1].
        True  → action is feasible for the active vehicle.
        mask[0] (depot/retire) is always True.
        """
        av = self._av()
        mask = np.zeros(self.N + 1, dtype=bool)
        mask[0] = True  # depot always available

        for c in range(1, self.N + 1):
            if self.served[c]:
                continue
            if self.inst.demands[c] > av.remaining_cap + 1e-8:
                continue
            travel = float(self.tt[av.cur_node, c])
            arrival = av.cur_time + travel
            if arrival > float(self.inst.tw_close[c]) + 1e-6:
                continue
            depart = (
                max(arrival, float(self.inst.tw_open[c]))
                + float(self.inst.service_time[c])
            )
            if depart + float(self.tt[c, 0]) > self._T + 1e-6:
                continue
            mask[c] = True

        return mask

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        av = self._av()
        n_stops = self.N + 1
        stop_feats = np.zeros((n_stops, STOP_FEAT_DIM), dtype=np.float32)

        tt_row = self.tt[av.cur_node]
        tt_to_depot = self.tt[:, 0]

        for i in range(n_stops):
            travel = float(tt_row[i])
            arrival = av.cur_time + travel
            tw_o = float(self.inst.tw_open[i])
            tw_c = float(self.inst.tw_close[i])
            svc = float(self.inst.service_time[i])
            depart = max(arrival, tw_o) + svc
            end_slack = tw_c - arrival
            depot_return_slack = self._T - depart - float(tt_to_depot[i])

            stop_feats[i, 0]  = self._nc(float(self.inst.coords[i, 0]))   # x
            stop_feats[i, 1]  = self._nc(float(self.inst.coords[i, 1]))   # y
            stop_feats[i, 2]  = self._nt(tw_o)                             # tw_open
            stop_feats[i, 3]  = self._nt(tw_c)                             # tw_close
            stop_feats[i, 4]  = self._nt(svc)                              # service_time
            stop_feats[i, 5]  = float(self.inst.demands[i]) / self._cap    # demand
            stop_feats[i, 6]  = float(self.served[i])                      # served
            stop_feats[i, 7]  = 1.0 if i == av.cur_node else 0.0          # is_cur_node
            stop_feats[i, 8]  = self._nt(travel)                           # travel from cur pos
            stop_feats[i, 9]  = self._nt(end_slack)                        # arrival deadline slack
            stop_feats[i, 10] = self._nt(depot_return_slack)               # depot-return slack

        n_alive = sum(1 for v in self.vehicles if not v.retired)
        vehicle_feat = np.array([
            self._nt(av.cur_time),
            av.remaining_cap / self._cap,
            n_alive / self.max_vehicles,
        ], dtype=np.float32)

        global_feat = np.array([
            self._nt(self._T - av.cur_time),
            float(np.sum(self.served[1:])) / self.N,
        ], dtype=np.float32)

        return np.concatenate([stop_feats.reshape(-1), vehicle_feat, global_feat])

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.served = np.zeros(self.N + 1, dtype=np.uint8)
        self.served[0] = 1
        self.vehicles = [
            VehicleState(
                cur_node=0, cur_time=0.0, remaining_cap=self._cap,
                route=[0], retired=False, activated=False,
            )
            for _ in range(self.max_vehicles)
        ]
        self.active_idx = 0
        self.total_distance = 0.0
        self.total_late = 0.0
        self.late_count = 0
        self.last_end_reason = "unknown"
        return self._build_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        av = self._av()

        # ── Action 0: retire current vehicle ──────────────────────────
        if action == 0:
            travel = self._retire(self.active_idx)
            reward -= self.dist_weight * self._nt(travel)

            next_idx = self._next_active()
            if next_idx == -1:
                if self._all_served():
                    terminated = True
                    info["reason"] = "all_served"
                    self.last_end_reason = "done"
                else:
                    unserved = [c for c in range(1, self.N + 1) if not self.served[c]]
                    reward -= self.unserved_penalty * len(unserved)
                    truncated = True
                    info["reason"] = "all_vehicles_retired"
                    info["unserved"] = len(unserved)
                    self.last_end_reason = "truncated"
            else:
                self.active_idx = next_idx
                info["reason"] = "vehicle_retired"

            return self._build_obs(), reward, terminated, truncated, info

        # ── Action c: visit customer ───────────────────────────────────
        c = int(action)
        travel = float(self.tt[av.cur_node, c])
        arrival = av.cur_time + travel
        service_start = max(arrival, float(self.inst.tw_open[c]))
        depart = service_start + float(self.inst.service_time[c])

        if not av.activated:
            reward -= self.vehicle_use_penalty   # first dispatch cost (primary objective)
        av.cur_node = c
        av.cur_time = depart
        av.remaining_cap -= float(self.inst.demands[c])
        av.route.append(c)
        av.activated = True
        self.served[c] = 1
        self.total_distance += travel

        reward -= self.dist_weight * self._nt(travel)
        reward += self.visit_reward

        if self._all_served():
            for i in range(self.max_vehicles):
                if not self.vehicles[i].retired:
                    self._retire(i)
            terminated = True
            info["reason"] = "all_served"
            self.last_end_reason = "done"
            return self._build_obs(), reward, terminated, truncated, info

        # Force retire if depot return is no longer feasible
        if av.cur_time + float(self.tt[c, 0]) > self._T + 1e-6:
            t = self._retire(self.active_idx)
            reward -= self.dist_weight * self._nt(t)
            next_idx = self._next_active()
            if next_idx == -1:
                unserved = [c for c in range(1, self.N + 1) if not self.served[c]]
                if unserved:
                    reward -= self.unserved_penalty * len(unserved)
                    truncated = True
                    info["reason"] = "all_vehicles_retired"
                    self.last_end_reason = "truncated"
                else:
                    terminated = True
                    info["reason"] = "all_served"
                    self.last_end_reason = "done"
            else:
                self.active_idx = next_idx

        return self._build_obs(), reward, terminated, truncated, info
