from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ============================================================
# 0) Dynamic Events
# ============================================================

class EventType(Enum):
    ACCIDENT = "accident"   # 특정 edge(들) travel time 증가
    RAIN     = "rain"       # 특정 zone 전체 edge travel time 증가


@dataclass
class DynamicEvent:
    event_type: EventType
    trigger_time: float          # 이 시각 이후 첫 step에서 적용
    duration: float              # 이 시간 동안 지속 (end_time = trigger_time + duration)
    multiplier: float            # travel time 배율 (>1.0, 환경 내부용 — LLM에 미전달)
    affected_nodes: List[int]    # 영향받는 노드 목록
                                 # ACCIDENT: [src, dst] 양방향
                                 # RAIN: zone에 속한 모든 고객 노드
    rainfall_mm: float = 0.0     # 강수량 mm/h (RAIN 전용, LLM용 자연어 정보)
    probability: float = 100.0   # 강수확률 % (RAIN 전용, LLM용 자연어 정보)
    severity: str = ""           # ACCIDENT 전용: "low" / "medium" / "high"
    description: str = ""        # LLM용 자연어 설명
    active: bool = False         # 현재 적용 중인지
    recovered: bool = False      # 이미 복구됐는지

    @property
    def end_time(self) -> float:
        return self.trigger_time + self.duration


_ACCIDENT_SEVERITY = {"low": 2.0, "medium": 3.5, "high": 5.0}  # aligned with ortools_vrptw.py


# ============================================================
# 1) Solomon VRPTW parsing
# ============================================================

@dataclass
class WasteInstance:
    name: str
    coords: np.ndarray          # [n_nodes, 2], depot = 0
    demands: np.ndarray         # [n_nodes]
    tw_open: np.ndarray         # [n_nodes]
    tw_close: np.ndarray        # [n_nodes]
    service_time: np.ndarray    # [n_nodes]
    vehicle_capacity: float
    vehicle_limit: int
    n_customers: int
    preset_events: List["DynamicEvent"] = None  # EVENTS 섹션에서 로드된 고정 이벤트


def _read_nonempty_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_waste_benchmark_txt(path: str) -> WasteInstance:
    lines = _read_nonempty_lines(path)
    if not lines:
        raise ValueError(f"Empty dataset file: {path}")

    name = lines[0]
    vehicle_limit: Optional[int] = None
    vehicle_capacity: Optional[float] = None
    rows: List[Tuple[int, float, float, float, float, float, float]] = []
    event_rows: List[List[str]] = []

    i = 0
    while i < len(lines):
        line = lines[i].upper()

        if line == "VEHICLE":
            if i + 2 >= len(lines):
                raise ValueError("Malformed VEHICLE section.")
            vals = lines[i + 2].split()
            if len(vals) < 2:
                raise ValueError("Malformed VEHICLE values.")
            vehicle_limit = int(float(vals[0]))
            vehicle_capacity = float(vals[1])
            i += 3
            continue

        if line == "CUSTOMER":
            i += 1
            while i < len(lines):
                toks = lines[i].split()
                if toks and toks[0].lstrip("-").isdigit():
                    break
                i += 1

            while i < len(lines):
                toks = lines[i].split()
                if len(toks) < 7 or not toks[0].lstrip("-").isdigit():
                    break

                cust_no = int(toks[0])
                x = float(toks[1])
                y = float(toks[2])
                demand = float(toks[3])
                ready = float(toks[4])
                due = float(toks[5])
                service = float(toks[6])
                rows.append((cust_no, x, y, demand, ready, due, service))
                i += 1
            continue

        if line == "EVENTS":
            i += 1
            while i < len(lines):
                toks = lines[i].split()
                if not toks or toks[0].startswith("#"):
                    i += 1
                    continue
                if toks[0].upper() in ("RAIN", "ACCIDENT"):
                    event_rows.append(toks)
                    i += 1
                else:
                    break
            continue

        i += 1

    if vehicle_limit is None or vehicle_capacity is None:
        raise ValueError("Failed to parse VEHICLE section.")
    if not rows:
        raise ValueError("Failed to parse CUSTOMER section.")

    rows.sort(key=lambda x: x[0])
    if rows[0][0] != 0:
        raise ValueError("Expected depot id 0 for Solomon format.")

    n_nodes = len(rows)
    coords = np.zeros((n_nodes, 2), dtype=np.float32)
    demands = np.zeros(n_nodes, dtype=np.float32)
    tw_open = np.zeros(n_nodes, dtype=np.float32)
    tw_close = np.zeros(n_nodes, dtype=np.float32)
    service_time = np.zeros(n_nodes, dtype=np.float32)

    for idx, (cust_no, x, y, demand, ready, due, service) in enumerate(rows):
        if cust_no != idx:
            raise ValueError(
                f"Customer ids must be contiguous 0..N. Got cust_no={cust_no}, idx={idx}"
            )
        coords[idx] = [x, y]
        demands[idx] = demand
        tw_open[idx] = ready
        tw_close[idx] = due
        service_time[idx] = service

    preset_events = []
    for toks in event_rows:
        keyword = toks[0].upper()
        trigger = float(toks[1])
        duration = float(toks[2])
        if keyword == "RAIN":
            multiplier = float(toks[3])
            rainfall_mm = float(toks[4])
            # Generated rain files have format: RAIN trigger duration multiplier rainfall nodes...
            # (same as C+R / OR-Tools — no probability field).
            # Always fire at 100%; toks[5:] are the affected nodes.
            probability = 100.0
            nodes = [int(x) for x in toks[5:]]
            severity = ""
            etype = EventType.RAIN
        else:  # ACCIDENT
            severity = toks[3].lower()
            multiplier = (_ACCIDENT_SEVERITY[severity] if severity in _ACCIDENT_SEVERITY
                          else float(severity))
            # Accident files don't include a probability field — always 100%.
            # (Attempting to parse toks[4] as probability would misread node IDs,
            #  since node IDs and probability values are in the same 0-100 range.)
            probability = 100.0
            nodes = [int(x) for x in toks[4:]]
            rainfall_mm = 0.0
            etype = EventType.ACCIDENT
        preset_events.append(DynamicEvent(
            event_type=etype,
            trigger_time=trigger,
            duration=duration,
            multiplier=multiplier,
            affected_nodes=nodes,
            rainfall_mm=rainfall_mm,
            probability=probability,
            severity=severity,
        ))

    return WasteInstance(
        name=name,
        coords=coords,
        demands=demands,
        tw_open=tw_open,
        tw_close=tw_close,
        service_time=service_time,
        vehicle_capacity=float(vehicle_capacity),
        vehicle_limit=int(vehicle_limit),
        n_customers=n_nodes - 1,
        preset_events=preset_events if preset_events else None,
    )


def build_travel_time_matrix(inst: WasteInstance, speed: float = 1.0) -> np.ndarray:
    """Build travel time matrix using int(round(dist)) — matches Solomon benchmark standard
    and OR-Tools evaluation, enabling direct comparison."""
    xy = inst.coords.astype(np.float64)
    n = xy.shape[0]
    tt = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        dx = xy[:, 0] - xy[i, 0]
        dy = xy[:, 1] - xy[i, 1]
        dist = np.sqrt(dx * dx + dy * dy) / max(speed, 1e-8)
        tt[i] = np.round(dist).astype(np.float32)   # int(round(dist)) — Solomon standard

    return tt


# ============================================================
# 2) Route env
# ============================================================

@dataclass
class RouteLog:
    vehicle_index: int
    zone_index: int
    nodes: List[int] = field(default_factory=list)
    end_time: float = 0.0
    load: float = 0.0


@dataclass
class VehicleState:
    cur_node: int = 0
    cur_time: float = 0.0
    remaining_cap: float = 0.0
    route_nodes: List[int] = field(default_factory=list)
    route_load: float = 0.0
    active_zone_idx: int = 0
    retired: bool = False
    activated: bool = False  # True after visiting first customer


class WasteFleetEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        inst: WasteInstance,
        tt: np.ndarray,
        max_vehicles: Optional[int] = None,
        late_count_penalty: float = 20.0,
        late_penalty: float = 150.0,
        unserved_penalty: float = 500.0,
        max_scope_size: Optional[int] = None,
    ):
        super().__init__()

        self.inst = inst
        self.tt_base = np.asarray(tt, dtype=np.float32)   # 원본 (불변)
        self.tt = self.tt_base.copy()                       # 현재 적용 중인 tt
        self.n_nodes = len(inst.coords)
        self.N = inst.n_customers

        self.max_vehicles = int(max_vehicles) if max_vehicles is not None else int(inst.vehicle_limit)
        self.late_count_penalty = float(late_count_penalty)
        self.late_penalty = float(late_penalty)
        self.unserved_penalty = float(unserved_penalty)

        self.customer_to_zone = np.zeros(self.n_nodes, dtype=np.int32)
        self.total_zones = 1

        self.max_scope_size: int = int(max_scope_size) if max_scope_size is not None else self.N
        self.adjacent_zones: Dict[int, Set[int]] = {}

        self.active_events: List[DynamicEvent] = []

        # Allow subclasses to pre-set dims before calling super().__init__()
        # (needed because reset() → _build_obs() is called inside super().__init__())
        self.stop_feat_dim    = getattr(self, 'stop_feat_dim',    8)
        self.vehicle_feat_dim = getattr(self, 'vehicle_feat_dim', 3)
        self.global_feat_dim  = getattr(self, 'global_feat_dim',  1)
        obs_dim = (
            self.max_scope_size * self.stop_feat_dim
            + self.vehicle_feat_dim
            + self.global_feat_dim
        )

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.N + 1)

        self.reset()

    # ------------------------------------------------------------------
    # Properties for backward compatibility with train_dqn_waste.py
    # (delegate to the currently active vehicle)
    # ------------------------------------------------------------------

    @property
    def cur_node(self) -> int:
        return self.vehicles[self.active_vehicle_idx].cur_node

    @property
    def cur_time(self) -> float:
        return self.vehicles[self.active_vehicle_idx].cur_time

    @property
    def remaining_cap(self) -> float:
        return self.vehicles[self.active_vehicle_idx].remaining_cap

    @property
    def active_zone_idx(self) -> int:
        return self.vehicles[self.active_vehicle_idx].active_zone_idx

    @property
    def vehicles_used(self) -> int:
        """Number of vehicles that have visited at least one customer."""
        return sum(1 for v in self.vehicles if v.activated)

    def set_zone_assignment(
        self,
        customer_to_zone: Dict[int, int],
        total_zones: int,
        adjacent_zones: Optional[Dict[int, Set[int]]] = None,
    ) -> None:
        zone_arr = np.zeros(self.n_nodes, dtype=np.int32)
        for cust, z in customer_to_zone.items():
            if 1 <= cust <= self.N:
                zone_arr[cust] = int(z)
        self.customer_to_zone = zone_arr
        self.total_zones = int(max(1, total_zones))
        self.adjacent_zones = adjacent_zones if adjacent_zones is not None else {}
        # Pre-assign zones to vehicles (vehicle i → zone i % total_zones)
        for i, v in enumerate(self.vehicles):
            v.active_zone_idx = i % self.total_zones

    def set_active_zone(self, active_zone_idx: int) -> None:
        """Set the active vehicle's assigned zone."""
        self.vehicles[self.active_vehicle_idx].active_zone_idx = int(max(0, active_zone_idx))

    def _get_active_vehicle_idx(self) -> int:
        """Return index of non-retired vehicle with minimum cur_time. -1 if all retired."""
        min_time = float("inf")
        min_idx = -1
        for i, v in enumerate(self.vehicles):
            if not v.retired and v.cur_time < min_time:
                min_time = v.cur_time
                min_idx = i
        return min_idx

    def _all_served(self) -> bool:
        return bool(np.all(self.served[1:] == 1))

    def _retire_vehicle(self, veh_idx: int) -> None:
        """Send vehicle back to depot and mark retired. Records RouteLog."""
        v = self.vehicles[veh_idx]
        if v.cur_node != 0:
            travel = float(self.tt[v.cur_node, 0])
            v.cur_time += travel
            self.total_distance += travel
            v.cur_node = 0
            v.route_nodes.append(0)
        self.route_history.append(RouteLog(
            vehicle_index=veh_idx,
            zone_index=v.active_zone_idx,
            nodes=v.route_nodes.copy(),
            end_time=v.cur_time,
            load=v.route_load,
        ))
        v.retired = True

    def _time_scale(self) -> float:
        return max(float(np.max(self.inst.tw_close)), 1.0)

    def _norm_time(self, value: float) -> float:
        return float(value / self._time_scale())

    def _coord_scale(self) -> float:
        return max(float(np.max(self.inst.coords)), 1.0)

    def _norm_coord(self, value: float) -> float:
        return float(value / self._coord_scale())

    def _norm_zone(self, zone_idx: int) -> float:
        if self.total_zones <= 1:
            return 0.0
        return float(zone_idx / max(self.total_zones - 1, 1))

    def _build_obs(self) -> np.ndarray:
        depot_due = float(self.inst.tw_close[0])
        av = self.vehicles[self.active_vehicle_idx]

        scope_custs = self._get_scope_customers()
        stop_feats = np.zeros((self.max_scope_size, self.stop_feat_dim), dtype=np.float32)
        tt_curr_row = self.tt[av.cur_node]
        tt_to_depot = self.tt[:, 0]

        # Position 0: depot (Kool VRP Appendix C.2 — separate depot embedding)
        depot_travel = float(tt_curr_row[0])
        depot_arrival = av.cur_time + depot_travel
        stop_feats[0, 0] = 0.0                                                        # zone = 0
        stop_feats[0, 1] = self._norm_time(float(self.inst.tw_open[0]) - depot_arrival)
        stop_feats[0, 2] = self._norm_time(depot_due - depot_arrival)
        stop_feats[0, 3] = self._norm_time(depot_travel)
        stop_feats[0, 4] = 0.0                                                        # already at depot
        stop_feats[0, 5] = 0.0                                                        # not a customer
        stop_feats[0, 6] = 1.0 if av.cur_node == 0 else 0.0                          # is_cur_node
        stop_feats[0, 7] = self._norm_coord(float(self.inst.coords[0, 0]))            # x
        stop_feats[0, 8] = self._norm_coord(float(self.inst.coords[0, 1]))            # y

        # Positions 1..max_scope_size-1: zone customers
        for i, cust in enumerate(scope_custs[:self.max_scope_size - 1]):
            travel = float(tt_curr_row[cust])
            arrival = av.cur_time + travel
            start_slack = float(self.inst.tw_open[cust]) - arrival
            depart = max(arrival, float(self.inst.tw_open[cust])) + float(self.inst.service_time[cust])
            end_slack = float(self.inst.tw_close[cust]) - arrival  # arrival deadline slack (service extends after)
            depot_return_slack = depot_due - depart - float(tt_to_depot[cust])

            stop_feats[i + 1, 0] = self._norm_zone(int(self.customer_to_zone[cust]))
            stop_feats[i + 1, 1] = self._norm_time(start_slack)
            stop_feats[i + 1, 2] = self._norm_time(end_slack)
            stop_feats[i + 1, 3] = self._norm_time(travel)
            stop_feats[i + 1, 4] = self._norm_time(depot_return_slack)
            stop_feats[i + 1, 5] = float(self.served[cust])
            stop_feats[i + 1, 6] = 1.0 if cust == av.cur_node else 0.0              # is_cur_node
            stop_feats[i + 1, 7] = self._norm_coord(float(self.inst.coords[cust, 0]))  # x
            stop_feats[i + 1, 8] = self._norm_coord(float(self.inst.coords[cust, 1]))  # y

        # Active vehicle: [assigned_zone, retired, cur_time]
        # Zone-fixed assignment → other vehicles irrelevant; only active vehicle state needed
        vehicle_feat = np.array(
            [
                self._norm_zone(int(av.active_zone_idx)),
                float(av.retired),
                self._norm_time(float(av.cur_time)),
            ],
            dtype=np.float32,
        )

        # Global: time remaining until depot closes
        global_feat = np.array(
            [self._norm_time(depot_due - av.cur_time)],
            dtype=np.float32,
        )

        parts = [stop_feats.reshape(-1), vehicle_feat, global_feat]
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _get_scope_customers(self) -> List[int]:
        """Active zone 고객만 반환 (zone-scoped, hard constraint)."""
        av = self.vehicles[self.active_vehicle_idx]
        active_zone = int(av.active_zone_idx)
        return [
            c for c in range(1, self.N + 1)
            if self.customer_to_zone[c] == active_zone
        ]

    def _can_serve(self, cust: int) -> bool:
        if cust <= 0 or cust > self.N:
            return False
        if self.served[cust] == 1:
            return False
        av = self.vehicles[self.active_vehicle_idx]
        if self.inst.demands[cust] > av.remaining_cap + 1e-8:
            return False
        return True

    def get_visit_timing(self, cust: int) -> Tuple[float, float, float, float]:
        av = self.vehicles[self.active_vehicle_idx]
        travel = float(self.tt[av.cur_node, cust])
        arrival = av.cur_time + travel
        service_start = max(arrival, float(self.inst.tw_open[cust]))
        late = max(0.0, service_start - float(self.inst.tw_close[cust]))
        depart = service_start + float(self.inst.service_time[cust])
        return travel, arrival, service_start, depart

    def is_on_time_if_visit(self, cust: int) -> bool:
        if not self._can_serve(cust):
            return False
        _, _, service_start, _ = self.get_visit_timing(cust)
        return service_start <= float(self.inst.tw_close[cust]) + 1e-8

    def can_return_to_depot_if_visit(self, cust: int) -> bool:
        if not self._can_serve(cust):
            return False

        depot_due = float(self.inst.tw_close[0])
        _, _, _, depart = self.get_visit_timing(cust)
        back_to_depot = float(self.tt[cust, 0])

        return (depart + back_to_depot) <= depot_due + 1e-8

    def get_depot_return_info(self, cust: int) -> Tuple[float, float, float, float]:
        """방문 후 depot 복귀 관련 수치를 반환한다.

        Returns:
            (travel_to_depot, return_arrival, return_slack, depot_due)
            - travel_to_depot : cust → depot 이동 시간
            - return_arrival  : 복귀 예상 도착 시각 (depart + travel_to_depot)
            - return_slack    : depot_due - return_arrival (음수면 위반)
            - depot_due       : depot time window 마감 시각
        """
        depot_due = float(self.inst.tw_close[0])
        _, _, _, depart = self.get_visit_timing(cust)
        travel_to_depot = float(self.tt[cust, 0])
        return_arrival = depart + travel_to_depot
        return travel_to_depot, return_arrival, depot_due - return_arrival, depot_due

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        self.served = np.zeros(self.n_nodes, dtype=np.uint8)
        self.served[0] = 1

        self.total_distance = 0.0
        self.total_late = 0.0
        self.late_count = 0
        self.route_history: List[RouteLog] = []

        # Initialize all vehicles at depot
        self.vehicles: List[VehicleState] = [
            VehicleState(
                cur_node=0,
                cur_time=0.0,
                remaining_cap=float(self.inst.vehicle_capacity),
                route_nodes=[0],
                route_load=0.0,
                active_zone_idx=i % max(self.total_zones, 1),
                retired=False,
                activated=False,
            )
            for i in range(self.max_vehicles)
        ]
        self.active_vehicle_idx: int = 0  # vehicle with min cur_time (all tied at 0 → pick 0)

        # Reset travel time and sample events for this episode
        self.tt = self.tt_base.copy()
        if self.inst.preset_events:
            import copy as _copy
            import random as _random
            self.active_events = []
            for src in self.inst.preset_events:
                e = _copy.copy(src)
                e.active = False
                e.recovered = False
                if _random.random() * 100.0 >= e.probability:
                    e.recovered = True  # cancelled — never triggers
                self.active_events.append(e)
        else:
            self.active_events = []

        obs = self._build_obs()
        info = {
            "instance_name": self.inst.name,
            "n_customers": self.N,
            "vehicle_capacity": self.inst.vehicle_capacity,
            "vehicle_limit": self.max_vehicles,
        }
        return obs, info

    def _apply_event(self, event: DynamicEvent) -> None:
        """이벤트를 tt에 반영."""
        nodes = event.affected_nodes
        m = event.multiplier
        if event.event_type == EventType.ACCIDENT:
            # 양방향 edge
            if len(nodes) >= 2:
                src, dst = nodes[0], nodes[1]
                self.tt[src, dst] = self.tt_base[src, dst] * m
                self.tt[dst, src] = self.tt_base[dst, src] * m
        elif event.event_type == EventType.RAIN:
            # edge (a→b) is rain-affected only if both a and b are in the rain zone
            for a in nodes:
                for b in nodes:
                    self.tt[a, b] = self.tt_base[a, b] * m
                    self.tt[b, a] = self.tt_base[b, a] * m
        event.active = True

    def _recover_event(self, event: DynamicEvent) -> None:
        """이벤트 복구: 원래 tt로 되돌림."""
        nodes = event.affected_nodes
        if event.event_type == EventType.ACCIDENT:
            if len(nodes) >= 2:
                src, dst = nodes[0], nodes[1]
                self.tt[src, dst] = self.tt_base[src, dst]
                self.tt[dst, src] = self.tt_base[dst, src]
        elif event.event_type == EventType.RAIN:
            for a in nodes:
                for b in nodes:
                    self.tt[a, b] = self.tt_base[a, b]
                    self.tt[b, a] = self.tt_base[b, a]
        event.active = False
        event.recovered = True

    def _update_events(self, cur_time: float) -> List[DynamicEvent]:
        """현재 시각 기준으로 이벤트 적용/복구. 새로 활성화된 이벤트 목록 반환."""
        newly_triggered = []
        for event in self.active_events:
            if event.recovered:
                continue
            if not event.active and cur_time >= event.trigger_time:
                self._apply_event(event)
                newly_triggered.append(event)
            elif event.active and cur_time >= event.end_time:
                self._recover_event(event)
        return newly_triggered

    def step(self, action: int, external_penalty: float = 0.0):

        reward = 0.0
        terminated = False
        truncated = False
        veh = self.vehicles[self.active_vehicle_idx]

        # 현재 vehicle의 시각 기준으로 이벤트 적용/복구
        # newly_triggered: LLM 연동 시 이벤트 정보 전달에 활용
        newly_triggered = self._update_events(veh.cur_time)

        info: Dict[str, Any] = {
            "action": int(action),
            "served_count": int(np.sum(self.served[1:])),
            "late_count": int(self.late_count),
            "vehicles_used": int(self.vehicles_used),
            "active_zone_idx": int(veh.active_zone_idx),
            "active_vehicle_idx": int(self.active_vehicle_idx),
            "newly_triggered_events": [e.description for e in newly_triggered],
            "active_events": [e.description for e in self.active_events if e.active],
        }

        # ── action = 0: retire active vehicle (return to depot) ──────────
        if action == 0:
            self._retire_vehicle(self.active_vehicle_idx)

            next_idx = self._get_active_vehicle_idx()
            if next_idx == -1:
                # All vehicles retired — episode ends, compute reward
                terminated = self._all_served()
                truncated  = not terminated
                unserved_custs = [c for c in range(1, self.N + 1) if self.served[c] == 0]
                reward = self._episode_reward(len(unserved_custs))
                info["reason"] = "all_served" if terminated else "all_vehicles_retired"
                if unserved_custs:
                    info["unserved_count"] = len(unserved_custs)
                info["total_distance"] = float(self.total_distance)
                info["total_late"]     = float(self.total_late)
                return self._build_obs(), reward, terminated, truncated, info

            self.active_vehicle_idx = next_idx
            info["reason"] = "vehicle_retired"
            return self._build_obs(), reward, terminated, truncated, info

        # ── action = customer: visit that customer ────────────────────────
        cust = int(action)

        if not veh.activated:
            veh.activated = True

        travel, arrival, service_start, depart = self.get_visit_timing(cust)
        late = max(0.0, service_start - float(self.inst.tw_close[cust]))

        veh.cur_time = depart
        veh.cur_node = cust
        veh.remaining_cap -= float(self.inst.demands[cust])
        veh.route_nodes.append(cust)
        veh.route_load += float(self.inst.demands[cust])
        self.served[cust] = 1
        self.total_distance += travel
        self.total_late += late
        if late > 0:
            self.late_count += 1
            info["late"] = float(late)

        info["arrival"]       = float(arrival)
        info["service_start"] = float(service_start)
        info["depart"]        = float(depart)
        info["remaining_cap"] = float(veh.remaining_cap)

        if self._all_served():
            for i, v in enumerate(self.vehicles):
                if not v.retired:
                    self._retire_vehicle(i)
            terminated = True
            reward = self._episode_reward(0)
            info["reason"]         = "all_served"
            info["total_distance"] = float(self.total_distance)
            info["total_late"]     = float(self.total_late)
            return self._build_obs(), reward, terminated, truncated, info

        # ── Hard VRPTW depot constraint: force retire if can't return in time ──
        depot_due = float(self.inst.tw_close[0])
        if veh.cur_time + float(self.tt[cust, 0]) > depot_due:
            self._retire_vehicle(self.active_vehicle_idx)
            info["reason"] = "depot_forced_retire"

            next_idx = self._get_active_vehicle_idx()
            if next_idx == -1:
                unserved_custs = [c for c in range(1, self.N + 1) if self.served[c] == 0]
                if unserved_custs:
                    truncated = True
                    reward = self._episode_reward(len(unserved_custs))
                    info["reason"]       = "all_vehicles_retired"
                    info["unserved_count"] = len(unserved_custs)
                else:
                    terminated = True
                    reward = self._episode_reward(0)
                    info["reason"] = "all_served"
                info["total_distance"] = float(self.total_distance)
                info["total_late"]     = float(self.total_late)
                return self._build_obs(), reward, terminated, truncated, info

            self.active_vehicle_idx = next_idx
            return self._build_obs(), reward, terminated, truncated, info

        # Advance to next vehicle with minimum cur_time
        self.active_vehicle_idx = self._get_active_vehicle_idx()

        return self._build_obs(), reward, terminated, truncated, info

    def _episode_reward(self, n_unserved: int) -> float:
        """Normalized episode-level reward.

        reward = -(total_distance/T
                   + late_count_penalty * n_late/N
                   + late_penalty * total_late/T
                   + unserved_penalty * n_unserved/N)

        All components normalized to comparable magnitude.
        Priority: unserved > late_count > late_magnitude > distance
        """
        T = self._time_scale()
        N = max(self.N, 1)
        return -(self.total_distance / T
                 + self.late_count_penalty * self.late_count / N
                 + self.late_penalty       * self.total_late  / T
                 + self.unserved_penalty   * n_unserved       / N)

    def render(self):
        av = self.vehicles[self.active_vehicle_idx]
        print(
            f"[{self.inst.name}] "
            f"active_veh={self.active_vehicle_idx}, "
            f"node={av.cur_node}, "
            f"time={av.cur_time:.1f}, "
            f"cap={av.remaining_cap:.1f}, "
            f"served={int(np.sum(self.served[1:]))}/{self.N}, "
            f"veh_used={self.vehicles_used}/{self.max_vehicles}, "
            f"zone={av.active_zone_idx}"
        )