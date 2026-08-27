"""
vrptw_env.py — VRPTW shared foundation.
  - Solomon data loading (load_solomon, make_batch)
  - VRPTWEnv (RL environment: reset/step/reward)
  - Shared utilities (plotting, solution printing, AverageMeter)
"""
from __future__ import annotations

import os
import re
import numpy as np
import torch
from dataclasses import dataclass


# ── Solomon parsing ───────────────────────────────────────────────────────────

# Named severity labels (old format) and 1-10 integer scale (new format).
# Integer scale: severity stored in event files, multiplier applied internally.
# LLM and RL see only the integer label, never the true multiplier.
_ACCIDENT_SEVERITY = {
    # n-car description keys (new format)
    "3-car-collision": 5.0,
    "4-car-pile-up":   8.5,
    "5-car-pile-up":  13.0,
    # legacy named levels
    "low": 1.5, "medium": 2.0, "high": 3.0,
    # legacy 1–10 integer scale
    "1":  1.5,  "2":  2.0,  "3":  2.8,  "4":  3.5,  "5":  5.0,
    "6":  6.5,  "7":  8.5,  "8": 10.5,  "9": 13.0, "10": 15.0,
}


def load_solomon(path: str) -> dict:
    """
    Parse Solomon VRPTW benchmark file (with optional EVENTS section).
    Returns dict of normalized tensors (all via T or max_coord).
    preset_events: list of dicts — type/trigger_time/duration/multiplier/
                   rainfall_mm/probability/nodes (raw time units, 1-based node ids).
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    name = lines[0]

    v_idx = next(i for i, l in enumerate(lines) if l.upper().startswith("VEHICLE"))
    vehicle_limit, vehicle_capacity = None, None
    for offset in (1, 2):
        parts = lines[v_idx + offset].split()
        try:
            vehicle_limit    = int(parts[0])
            vehicle_capacity = float(parts[1])
            break
        except (ValueError, IndexError):
            continue

    c_idx = next(i for i, l in enumerate(lines) if l.upper().startswith("CUSTOMER"))
    rows = []
    for ln in lines[c_idx + 1:]:
        parts = ln.split()
        if len(parts) >= 7:
            try:
                rows.append([float(x) for x in parts[:7]])
            except ValueError:
                continue

    data = np.array(rows, dtype=np.float64)
    T         = float(data[0, 5])
    max_coord = float(max(np.max(np.abs(data[:, 1])), np.max(np.abs(data[:, 2])), 1.0))

    coords_raw = data[:, 1:3].astype(np.float32)
    n          = len(coords_raw)
    diff       = coords_raw[:, None, :] - coords_raw[None, :, :]
    tt_raw     = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)
    tt_norm    = tt_raw / T

    coords_norm   = (coords_raw / max_coord).astype(np.float32)
    demand_norm   = (data[:, 3] / vehicle_capacity).astype(np.float32)
    tw_open_norm  = (data[:, 4] / T).astype(np.float32)
    tw_close_norm = (data[:, 5] / T).astype(np.float32)
    svc_norm      = (data[:, 6] / T).astype(np.float32)

    preset_events = []
    try:
        ev_start = next(i for i, l in enumerate(lines) if l.upper() == "EVENTS")
        for ln in lines[ev_start + 1:]:
            toks = ln.split()
            if not toks or toks[0].startswith('#'):
                continue
            kw = toks[0].upper()
            if kw not in ("RAIN", "ACCIDENT"):
                continue
            trigger  = float(toks[1])
            duration = float(toks[2])
            vehicles_involved = None
            if kw == "RAIN":
                multiplier  = float(toks[3])
                rainfall_mm = float(toks[4])
                nodes       = [int(x) for x in toks[5:]]
                severity    = None
            else:
                raw = toks[3]
                try:
                    multiplier = float(raw)
                    # Current format: ACCIDENT trigger duration mult vehicles nodes...
                    # vehicles_involved is recorded at generation time (mirrors
                    # rainfall_mm) -- never reverse-mapped from multiplier, so a
                    # future re-tuning of mu/severity_weights can't silently
                    # desync this observation value from what was actually drawn.
                    vehicles_involved = int(toks[4])
                    nodes = [int(x) for x in toks[5:]]
                except ValueError:
                    # Legacy NAMED-severity format only (no vehicles field):
                    # ACCIDENT trigger duration <low|medium|high|...> nodes...
                    # NOTE: legacy files with a BARE-FLOAT multiplier and no
                    # vehicles field (e.g. old *_acc_A/_B, *_rand_acc_* files)
                    # are NOT distinguishable from the current format here --
                    # toks[4] would be misread as vehicles_involved, dropping
                    # the true first node. Those files are not referenced by
                    # this project's actual generation/training pipeline
                    # (generate_pool_events / trainer scripts only load the
                    # migrated *_acc.txt files), so this is accepted dead-path
                    # breakage rather than a live bug.
                    severity_key = raw.lower()
                    multiplier = _ACCIDENT_SEVERITY.get(severity_key, 5.0)
                    nodes = [int(x) for x in toks[4:]]
                severity    = None
                rainfall_mm = 0.0
            preset_events.append(dict(
                type=kw, trigger_time=trigger, duration=duration,
                multiplier=multiplier, rainfall_mm=rainfall_mm,
                vehicles_involved=vehicles_involved,
                severity=severity,   # str for accident, None for rain
                nodes=nodes,
            ))
    except StopIteration:
        pass

    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32)

    return dict(
        name             = name,
        n_customers      = n - 1,
        vehicle_limit    = vehicle_limit,
        vehicle_capacity = vehicle_capacity,
        T                = T,
        max_coord        = max_coord,
        preset_events    = preset_events,
        depot_xy         = _t(coords_norm[[0]]),
        node_xy          = _t(coords_norm[1:]),
        depot_demand     = _t(demand_norm[[0]]),
        node_demand      = _t(demand_norm[1:]),
        depot_tw_open    = _t(tw_open_norm[[0]]),
        depot_tw_close   = _t(tw_close_norm[[0]]),
        depot_service    = _t(svc_norm[[0]]),
        node_tw_open     = _t(tw_open_norm[1:]),
        node_tw_close    = _t(tw_close_norm[1:]),
        node_service     = _t(svc_norm[1:]),
        tt               = _t(tt_norm),
    )


def make_batch(inst: dict, batch_size: int, device: torch.device) -> dict:
    """Repeat a single Solomon instance batch_size times."""
    keys_2d = ['depot_xy', 'node_xy']
    keys_1d = ['depot_demand', 'node_demand', 'depot_tw_open', 'depot_tw_close',
               'depot_service', 'node_tw_open', 'node_tw_close', 'node_service']
    out = {}
    for k in keys_2d:
        t = inst[k].to(device)
        out[k] = t.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    for k in keys_1d:
        t = inst[k].to(device)
        out[k] = t.unsqueeze(0).expand(batch_size, -1).contiguous()
    out['tt'] = inst['tt'].to(device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    out['vehicle_limit'] = inst.get('vehicle_limit', None)  # scalar int or None
    out['events'] = inst.get('preset_events', [])  # raw dicts; env normalizes by T internally
    out['T']      = inst.get('T', None)            # raw Solomon T, for normalizing event timing
    return out


from generate_vrptw import (  # noqa: F401 (re-exported)
    generate_c_type_instance,
    generate_r_type_instance,
    generate_rc_type_instance,
    generate_unified_instance,
    generate_rf_instance,
)


def augment_xy_data_by_8_fold(xy_data: torch.Tensor) -> torch.Tensor:
    """8-fold geometric augmentation. xy_data: (batch, N, 2) -> (8*batch, N, 2)"""
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    return torch.cat([
        torch.cat((x,     y    ), dim=2),
        torch.cat((1 - x, y    ), dim=2),
        torch.cat((x,     1 - y), dim=2),
        torch.cat((1 - x, 1 - y), dim=2),
        torch.cat((y,     x    ), dim=2),
        torch.cat((1 - y, x    ), dim=2),
        torch.cat((y,     1 - x), dim=2),
        torch.cat((1 - y, 1 - x), dim=2),
    ], dim=0)


# ── Environment dataclasses ───────────────────────────────────────────────────

@dataclass
class Reset_State:
    depot_xy:           torch.Tensor = None   # (batch, 1, 2)
    node_xy:            torch.Tensor = None   # (batch, N, 2)
    node_demand:        torch.Tensor = None   # (batch, N)
    node_tw_open:       torch.Tensor = None   # (batch, N)
    node_tw_close:      torch.Tensor = None   # (batch, N)
    node_service_time:  torch.Tensor = None   # (batch, N)
    depot_tw_close:     torch.Tensor = None   # (batch, 1)
    rain_tokens: torch.Tensor = None   # (batch, R, 4) [node/N, mm/100, t_s/T, t_e/T] [EVT]
    acc_tokens:  torch.Tensor = None   # (batch, A, 5) [na/N, nb/N, sev, t_s/T, t_e/T] [EVT]


@dataclass
class Step_State:
    BATCH_IDX:       torch.Tensor = None   # (batch, pomo)
    POMO_IDX:        torch.Tensor = None   # (batch, pomo)
    selected_count:  int          = None
    load:            torch.Tensor = None   # (batch, pomo)
    current_node:    torch.Tensor = None   # (batch, pomo)
    ninf_mask:       torch.Tensor = None   # (batch, pomo, N+1)
    current_time:    torch.Tensor = None   # (batch, pomo)


# ── Environment ───────────────────────────────────────────────────────────────

class VRPTWEnv:
    def __init__(self, **env_params):
        self.env_params      = env_params
        self.problem_size    = env_params['problem_size']
        self.pomo_size       = env_params['pomo_size']
        self.late_count_penalty = env_params.get('late_count_penalty', 0.0)
        self.late_penalty       = env_params.get('late_penalty',       1.0)
        self.vehicle_penalty    = env_params.get('vehicle_penalty',    0.0)
        self.unserved_penalty   = env_params.get('unserved_penalty',   500.0)  # [VL] per unserved customer / N
        self.D_max              = env_params.get('D_max',  1.0)   # 1.0 = no normalization (backward compat)
        self.Lt_max             = env_params.get('Lt_max', 1.0)   # 1.0 = no normalization
        self.k_excess_alpha     = env_params.get('k_excess_alpha', 0.0)  # extra penalty per vehicle above K_min
        self.hard_tw            = env_params.get('hard_tw', False)  # True = mask TW-infeasible nodes in ninf_mask

        self.k_min        = None   # (batch,) computed in load_problems from normalized demand
        self.vehicle_limit = None  # [VL] set by load_problems() from instance dict (None = unlimited)

        self.batch_size          = None
        self.BATCH_IDX           = None
        self.POMO_IDX            = None
        self.depot_node_xy       = None
        self.depot_node_demand   = None
        self.depot_node_tw_open  = None
        self.depot_node_tw_close = None
        self.depot_node_service  = None
        self.tt                  = None
        self._ev_trigger_t       = None   # (E,) normalized trigger time, or None if no events
        self._ev_end_t           = None   # (E,) normalized trigger+duration
        self._ev_mult_t          = None   # (E,) multiplier
        self._ev_zone_t          = None   # (E, N+1) bool, node in event zone

        self.selected_count     = None
        self.current_node       = None
        self.selected_node_list = None
        self.at_the_depot       = None
        self.load               = None
        self.visited_ninf_flag  = None
        self.ninf_mask          = None
        self.finished           = None
        self.current_time       = None
        self.total_late         = None
        self.n_late_stops       = None
        self.depot_visit_count  = None
        self.accident_feat      = None

        self.reset_state = Reset_State()
        self.step_state  = Step_State()

    def load_problems(self, batch_dict: dict):
        depot_xy      = batch_dict['depot_xy']
        node_xy       = batch_dict['node_xy']
        node_demand   = batch_dict['node_demand']
        node_tw_open  = batch_dict['node_tw_open']
        node_tw_close = batch_dict['node_tw_close']
        node_service  = batch_dict['node_service']

        def _ensure_2d(t):
            return t.unsqueeze(-1) if t.dim() == 1 else t

        depot_tw_open  = _ensure_2d(batch_dict['depot_tw_open'])
        depot_tw_close = _ensure_2d(batch_dict['depot_tw_close'])
        depot_service  = _ensure_2d(batch_dict['depot_service'])

        self.batch_size = depot_xy.size(0)
        dev = depot_xy.device

        self.depot_node_xy       = torch.cat((depot_xy, node_xy), dim=1)
        depot_dem                = torch.zeros(self.batch_size, 1, device=dev)
        self.depot_node_demand   = torch.cat((depot_dem, node_demand), dim=1)
        self.depot_node_tw_open  = torch.cat((depot_tw_open,  node_tw_open),  dim=1)
        self.depot_node_tw_close = torch.cat((depot_tw_close, node_tw_close), dim=1)
        self.depot_node_service  = torch.cat((depot_service,  node_service),  dim=1)
        self.tt = batch_dict['tt'].contiguous()
        self._setup_events(batch_dict.get('events', []), batch_dict.get('T', None))

        # [VL] vehicle_limit: scalar int from Solomon instance (None = unlimited)
        self.vehicle_limit = batch_dict.get('vehicle_limit', None)

        # K_min = ceil(total_normalized_demand): demand is already divided by capacity
        if self.k_excess_alpha > 0:
            self.k_min = node_demand.sum(dim=1).ceil().float()  # (batch,)
        else:
            self.k_min = None

        self.BATCH_IDX = torch.arange(self.batch_size, device=dev)[:, None] \
                             .expand(self.batch_size, self.pomo_size)
        self.POMO_IDX  = torch.arange(self.pomo_size, device=dev)[None, :] \
                             .expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy          = depot_xy
        self.reset_state.node_xy           = node_xy
        self.reset_state.node_demand       = node_demand
        self.reset_state.node_tw_open      = node_tw_open
        self.reset_state.node_tw_close     = node_tw_close
        self.reset_state.node_service_time = node_service
        self.reset_state.depot_tw_close = depot_tw_close
        self.reset_state.rain_tokens    = None   # caller sets before pre_forward
        self.reset_state.acc_tokens     = None

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX  = self.POMO_IDX

    def reset(self):
        dev = self.depot_node_xy.device

        self.selected_count     = 0
        self.current_node       = None
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0), dtype=torch.long, device=dev)

        self.at_the_depot      = torch.ones( (self.batch_size, self.pomo_size), dtype=torch.bool,  device=dev)
        self.load              = torch.ones( (self.batch_size, self.pomo_size),                    device=dev)
        self.visited_ninf_flag = torch.zeros((self.batch_size, self.pomo_size, self.problem_size+1), device=dev)
        self.ninf_mask         = torch.zeros((self.batch_size, self.pomo_size, self.problem_size+1), device=dev)
        self.finished          = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.bool,  device=dev)
        self.current_time      = torch.zeros((self.batch_size, self.pomo_size), device=dev)
        self.total_late        = torch.zeros((self.batch_size, self.pomo_size), device=dev)
        self.n_late_stops      = torch.zeros((self.batch_size, self.pomo_size), device=dev)
        self.depot_visit_count = torch.zeros((self.batch_size, self.pomo_size), device=dev)
        self._unserved_count   = torch.zeros((self.batch_size, self.pomo_size), device=dev)  # [VL]

        return self.reset_state, None, False

    def pre_step(self):
        self.step_state.selected_count  = self.selected_count
        self.step_state.load            = self.load
        self.step_state.current_node    = self.current_node
        self.step_state.ninf_mask       = self.ninf_mask
        self.step_state.finished        = self.finished
        self.step_state.current_time = self.current_time
        return self.step_state, None, False

    def step(self, selected):
        dev = selected.device
        N1  = self.problem_size + 1

        previous_node = self.current_node

        self.selected_count += 1
        self.current_node    = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected[:, :, None]), dim=2)

        self.at_the_depot = (selected == 0)
        demand_list  = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, N1)
        selected_dem = demand_list.gather(dim=2, index=selected[:, :, None]).squeeze(2)
        self.load   -= selected_dem
        self.load[self.at_the_depot] = 1.0

        if previous_node is None:
            travel_time = torch.zeros(self.batch_size, self.pomo_size, device=dev)
        else:
            base_tt     = self.tt[self.BATCH_IDX, previous_node, selected]
            travel_time = self._event_multiplier_arc(previous_node, selected, self.current_time) * base_tt

        arrival      = self.current_time + travel_time
        tw_open_sel  = self.depot_node_tw_open [self.BATCH_IDX, selected]
        tw_close_sel = self.depot_node_tw_close[self.BATCH_IDX, selected]
        service_sel  = self.depot_node_service [self.BATCH_IDX, selected]
        self.current_time = torch.max(arrival, tw_open_sel) + service_sel
        self.current_time[self.at_the_depot] = 0.0

        lateness = torch.clamp(arrival - tw_close_sel, min=0.0)
        lateness[self.at_the_depot] = 0.0
        self.total_late   = self.total_late   + lateness
        self.n_late_stops = self.n_late_stops + (lateness > 0).float()

        # Count only genuine route-endings: arriving at depot from a customer node
        if previous_node is not None:
            came_from_customer = (previous_node != 0)
            self.depot_visit_count += (self.at_the_depot & came_from_customer).float()

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0

        self.ninf_mask = self.visited_ninf_flag.clone()
        demand_too_large = self.load[:, :, None] + 1e-5 < demand_list
        self.ninf_mask[demand_too_large] = float('-inf')

        if self.hard_tw:
            # Block nodes whose TW closes before the vehicle can arrive.
            # current_node may be None only on the very first call (pre_step), but
            # step() is never called with current_node=None, so this is safe.
            travel = self.tt[self.BATCH_IDX, self.current_node]  # (B, P, N+1)
            travel = self._event_multiplier_row(self.current_node, self.current_time) * travel
            future_arrival = self.current_time[:, :, None] + travel  # (B, P, N+1)
            tw_violated = future_arrival > self.depot_node_tw_close[:, None, :]  # (B, P, N+1)
            tw_violated[:, :, 0] = False  # depot (index 0) always reachable
            self.ninf_mask[tw_violated] = float('-inf')

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished  = self.finished | newly_finished

        # [VL] vehicle_limit: when the last vehicle returns to depot → terminate immediately
        # Any remaining unvisited customers receive unserved_penalty.
        if self.vehicle_limit is not None:
            exhausted = (self.depot_visit_count >= self.vehicle_limit) & self.at_the_depot & ~self.finished
            if exhausted.any():
                unserved = (self.visited_ninf_flag[:, :, 1:] != float('-inf')).float().sum(dim=2)
                self._unserved_count[exhausted] = unserved[exhausted]
                self.finished = self.finished | exhausted

        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load           = self.load
        self.step_state.current_node   = self.current_node
        self.step_state.ninf_mask      = self.ninf_mask
        self.step_state.current_time   = self.current_time

        done = self.finished.all()
        if done:
            reward = (-self._get_travel_distance()                        / self.D_max
                      - self.late_count_penalty * self.n_late_stops       / self.problem_size
                      - self.late_penalty       * self.total_late         / self.Lt_max
                      - self.vehicle_penalty    * self.depot_visit_count  / self.problem_size
                      - self.unserved_penalty   * self._unserved_count    / self.problem_size)  # [VL]
            if self.k_excess_alpha > 0 and self.k_min is not None:
                # extra penalty for vehicles beyond K_min (theoretical minimum)
                excess_k = (self.depot_visit_count - self.k_min.unsqueeze(1)).clamp(min=0)
                reward = reward - self.k_excess_alpha * excess_k / self.problem_size
        else:
            reward = None
        return self.step_state, reward, done

    # ── Event physics (internal) ──────────────────────────────────────────────
    # The environment owns event simulation: preset_events (RAIN/ACCIDENT, with
    # trigger_time/duration/multiplier/nodes) are ingested once at load_problems()
    # and applied per-step as a function of each rollout's own current_time and
    # the arc's endpoints — never by external mutation of self.tt. self.tt stays
    # the static base matrix for the whole episode.

    def _setup_events(self, events: list, T_raw: float | None):
        dev = self.tt.device
        N1  = self.problem_size + 1
        T   = T_raw if T_raw else 1.0

        triggers, ends, mults, zones = [], [], [], []
        for e in events:
            mult = e.get('multiplier')
            nodes = e.get('nodes') or []
            dur   = e.get('duration', 0.0)
            if not nodes or mult is None or mult <= 1.0 or dur <= 0:
                continue
            zone = torch.zeros(N1, dtype=torch.bool, device=dev)
            idx = [n for n in nodes if 0 <= n <= self.problem_size]
            if not idx:
                continue
            zone[idx] = True
            triggers.append(e['trigger_time'] / T)
            ends.append((e['trigger_time'] + dur) / T)
            mults.append(float(mult))
            zones.append(zone)

        if triggers:
            self._ev_trigger_t = torch.tensor(triggers, device=dev)          # (E,)
            self._ev_end_t     = torch.tensor(ends,     device=dev)          # (E,)
            self._ev_mult_t    = torch.tensor(mults,    device=dev)          # (E,)
            self._ev_zone_t    = torch.stack(zones, dim=0)                   # (E, N1)
        else:
            self._ev_trigger_t = self._ev_end_t = self._ev_mult_t = self._ev_zone_t = None

    def _event_multiplier_arc(self, node_a: torch.Tensor, node_b: torch.Tensor,
                               at_time: torch.Tensor) -> torch.Tensor:
        """node_a/node_b/at_time: (batch, pomo). Returns (batch, pomo) combined multiplier
        (max across simultaneously-active events) for the arc node_a->node_b at at_time."""
        if self._ev_trigger_t is None:
            return torch.ones_like(at_time)
        active  = (at_time[None] >= self._ev_trigger_t[:, None, None]) & \
                  (at_time[None] <  self._ev_end_t[:, None, None])              # (E,B,P)
        in_zone = self._ev_zone_t[:, node_a] | self._ev_zone_t[:, node_b]       # (E,B,P)
        hit     = active & in_zone
        mult    = torch.where(hit, self._ev_mult_t[:, None, None],
                               torch.ones((), device=at_time.device, dtype=self._ev_mult_t.dtype))
        return mult.max(dim=0).values

    def _event_multiplier_row(self, node_a: torch.Tensor, at_time: torch.Tensor) -> torch.Tensor:
        """node_a/at_time: (batch, pomo). Returns (batch, pomo, N+1) combined multiplier
        for travel from node_a to every destination, at at_time."""
        if self._ev_trigger_t is None:
            return torch.ones(*at_time.shape, self.problem_size + 1, device=at_time.device)
        active   = (at_time[None] >= self._ev_trigger_t[:, None, None]) & \
                   (at_time[None] <  self._ev_end_t[:, None, None])             # (E,B,P)
        zone_a   = self._ev_zone_t[:, node_a]                                   # (E,B,P)
        zone_all = self._ev_zone_t                                              # (E,N1)
        in_zone  = zone_a[..., None] | zone_all[:, None, None, :]               # (E,B,P,N1)
        hit      = active[..., None] & in_zone                                  # (E,B,P,N1)
        mult     = torch.where(hit, self._ev_mult_t[:, None, None, None],
                                torch.ones((), device=at_time.device, dtype=self._ev_mult_t.dtype))
        return mult.max(dim=0).values

    def total_late_violations(self) -> int:
        return int((self.total_late > 0).sum().item())

    def _get_travel_distance(self):
        # Use tt (normalized by T) for consistent scaling with total_late.
        # For Solomon, travel_time = travel_distance (speed=1), so this equals
        # total_travel_time / T — interpretable as fraction of time horizon.
        n   = self.selected_node_list                        # (batch, pomo, steps)
        nxt = n.roll(shifts=-1, dims=2)                     # cyclic: last→first
        B, P, S = n.shape
        bi  = torch.arange(B, device=n.device)[:, None, None].expand(B, P, S)
        return self.tt[bi, n, nxt].sum(dim=2)


# ── Shared utilities ──────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def _extract_routes(node_list):
    routes, current = [], [0]
    for n in map(int, node_list):
        if n == 0:
            if len(current) > 1:
                current.append(0)
                routes.append(current)
            current = [0]
        else:
            current.append(n)
    if len(current) > 1:
        current.append(0)
        routes.append(current)
    return routes


def _event_travel_mult(t_depart: float, node_a: int, node_b: int,
                        preset_events: list) -> float:
    """Return the worst-case multiplier for edge (a→b) given vehicle departure time."""
    mult = 1.0
    for ev in preset_events:
        trigger = float(ev['trigger_time'])
        end     = trigger + float(ev['duration'])
        if not (trigger <= t_depart < end):
            continue
        if ev['type'] == 'ACCIDENT':
            a, b = ev['nodes'][0], ev['nodes'][1]
            if (node_a == a and node_b == b) or (node_a == b and node_b == a):
                mult = max(mult, float(ev['multiplier']))
        elif ev['type'] == 'RAIN':
            zone = set(ev['nodes'])
            if node_a in zone and node_b in zone:
                mult = max(mult, float(ev['multiplier']))
    return mult


def _solution_stats(inst, routes):
    """Compute stats dict from routes for reuse in print/save.

    Timing uses event-aware travel times (via inst['preset_events']) so that
    late_count / total_late reflect the actual disruption experienced during
    the episode.  Total distance uses base TT (geometric distance).
    """
    T   = float(inst['T'])
    cap = float(inst['vehicle_capacity'])
    mc  = float(inst['max_coord'])

    coords  = np.vstack([inst['depot_xy'].numpy() * mc, inst['node_xy'].numpy() * mc])
    # Solomon VRPTW standard: exact Euclidean (no rounding)
    _diff = coords[:, None, :] - coords[None, :, :]
    tt    = np.sqrt((_diff ** 2).sum(-1))
    tw_open = np.round(np.concatenate([inst['depot_tw_open'].numpy() * T, inst['node_tw_open'].numpy() * T]))
    tw_close= np.round(np.concatenate([inst['depot_tw_close'].numpy() * T, inst['node_tw_close'].numpy() * T]))
    service = np.round(np.concatenate([inst['depot_service'].numpy() * T, inst['node_service'].numpy() * T]))
    demands = np.concatenate([inst['depot_demand'].numpy() * cap, inst['node_demand'].numpy() * cap])
    preset_events = inst.get('preset_events', [])

    total_dist   = sum(tt[r[j], r[j+1]] for r in routes for j in range(len(r)-1))
    n_served     = sum(len(r) - 2 for r in routes)
    late_count   = 0
    total_late   = 0.0

    per_vehicle = []
    for route in routes:
        stops = route[1:-1]
        rdist = sum(tt[route[j], route[j+1]] for j in range(len(route)-1))
        load  = sum(demands[s] for s in stops)
        stop_details = []
        t, prev = 0.0, 0
        for s in stops:
            base_tt   = float(tt[prev, s])
            ev_mult   = _event_travel_mult(t, prev, s, preset_events)
            travel    = base_tt * ev_mult
            arrival   = t + travel
            svc_start = max(arrival, float(tw_open[s]))
            depart    = svc_start + float(service[s])
            late      = max(0.0, arrival - float(tw_close[s]))
            if late >= 0.5:
                late_count += 1
                total_late += late
            stop_details.append(dict(
                prev=prev, node=s,
                x=float(coords[s, 0]), y=float(coords[s, 1]),
                arrival=arrival, wait=max(0.0, float(tw_open[s]) - arrival),
                svc_start=svc_start, depart=depart,
                tw_open=float(tw_open[s]), tw_close=float(tw_close[s]),
                demand=float(demands[s]), late=late,
            ))
            t, prev = depart, s
        ret_dist = float(tt[prev, 0])
        per_vehicle.append(dict(
            stops=stops, rdist=rdist, load=load,
            stop_details=stop_details,
            prev_last=prev, ret_dist=ret_dist, arr_depot=t + ret_dist,
        ))

    return dict(
        T=T, cap=cap, tt=tt, coords=coords,
        tw_open=tw_open, tw_close=tw_close, service=service, demands=demands,
        total_dist=total_dist, n_served=n_served,
        late_count=late_count, total_late=total_late,
        per_vehicle=per_vehicle,
    )


def _print_solution(inst, routes, total_reward, sol_routes=None):
    stats  = _solution_stats(inst, routes)
    T      = stats['T']
    cap    = stats['cap']
    tt     = stats['tt']
    n_cust = inst['n_customers']
    W      = 75

    print("\n" + "=" * W)
    print("  BEST SOLUTION  (greedy decode)")
    print("=" * W)
    print(f"  vehicles={len(routes):2d}  served={stats['n_served']:3d}/{n_cust}"
          f"  dist={stats['total_dist']:8.2f}  reward={total_reward:.4f}")
    print(f"  late_stops={stats['late_count']}  total_late={stats['total_late']:.2f}")

    if sol_routes is not None:
        ref  = _solution_stats(inst, sol_routes)
        hdr  = f"  {'':16s}  {'veh':>4}  {'served':>10}  {'dist':>9}  {'late_stops':>10}  {'total_late':>10}"
        sep  = "  " + "-" * 65
        mrow = (f"  {'Model':16s}  {len(routes):>4}  {stats['n_served']:>4}/{n_cust:<5}"
                f"  {stats['total_dist']:>9.2f}  {stats['late_count']:>10}  {stats['total_late']:>10.2f}")
        rrow = (f"  {'Reference':16s}  {len(sol_routes):>4}  {ref['n_served']:>4}/{n_cust:<5}"
                f"  {ref['total_dist']:>9.2f}  {ref['late_count']:>10}  {ref['total_late']:>10.2f}")
        dv   = len(routes) - len(sol_routes)
        dd   = stats['total_dist'] - ref['total_dist']
        dlc  = stats['late_count'] - ref['late_count']
        dtl  = stats['total_late'] - ref['total_late']
        drow = (f"  {'Delta (M-Ref)':16s}  {dv:>+4}  {'':>10}  {dd:>+9.2f}  {dlc:>+10}  {dtl:>+10.2f}")
        print(hdr)
        print(sep)
        print(mrow)
        print(rrow)
        print(sep)
        print(drow)
    print()

    for vi, vdata in enumerate(stats['per_vehicle'], 1):
        print(f"  -- Vehicle {vi:2d}  [{len(vdata['stops']):3d} stops | dist={vdata['rdist']:7.1f}"
              f" | load={vdata['load']:.0f}/{cap:.0f}]")
        for sd in vdata['stop_details']:
            tw_ok = " LATE" if sd['late'] > 1e-6 else ""
            print(f"      {sd['prev']:3d} -> {sd['node']:3d}"
                  f"  ({sd['x']:5.1f},{sd['y']:5.1f})"
                  f"  arr={sd['arrival']:6.1f}  wait={sd['wait']:4.1f}  dep={sd['depart']:6.1f}"
                  f"  TW=[{sd['tw_open']:5.1f},{sd['tw_close']:5.1f}]{tw_ok}"
                  f"  dem={sd['demand']:3.0f}")
        print(f"      {vdata['prev_last']:3d} ->   0  depot"
              f"  dist={vdata['ret_dist']:.1f}  arr={vdata['arr_depot']:.1f}/{T:.0f}")
        print()
    print("=" * W)


def _save_solution(inst, routes, total_reward, save_path: str, sol_routes=None):
    """Save visit order and timing to a text file."""
    stats  = _solution_stats(inst, routes)
    T      = stats['T']
    cap    = stats['cap']
    tt     = stats['tt']
    n_cust = inst['n_customers']
    W      = 75

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * W + "\n")
        f.write(f"  SOLUTION: {inst['name']}\n")
        f.write("=" * W + "\n")
        f.write(f"  vehicles={len(routes):2d}  served={stats['n_served']:3d}/{n_cust}"
                f"  dist={stats['total_dist']:8.2f}  reward={total_reward:.4f}\n")
        f.write(f"  late_stops={stats['late_count']}  total_late={stats['total_late']:.2f}\n")

        if sol_routes is not None:
            ref  = _solution_stats(inst, sol_routes)
            hdr  = f"  {'':16s}  {'veh':>4}  {'served':>10}  {'dist':>9}  {'late_stops':>10}  {'total_late':>10}\n"
            sep  = "  " + "-" * 65 + "\n"
            mrow = (f"  {'Model':16s}  {len(routes):>4}  {stats['n_served']:>4}/{n_cust:<5}"
                    f"  {stats['total_dist']:>9.2f}  {stats['late_count']:>10}  {stats['total_late']:>10.2f}\n")
            rrow = (f"  {'Reference':16s}  {len(sol_routes):>4}  {ref['n_served']:>4}/{n_cust:<5}"
                    f"  {ref['total_dist']:>9.2f}  {ref['late_count']:>10}  {ref['total_late']:>10.2f}\n")
            dv   = len(routes) - len(sol_routes)
            dd   = stats['total_dist'] - ref['total_dist']
            dlc  = stats['late_count'] - ref['late_count']
            dtl  = stats['total_late'] - ref['total_late']
            drow = (f"  {'Delta (M-Ref)':16s}  {dv:>+4}  {'':>10}  {dd:>+9.2f}  {dlc:>+10}  {dtl:>+10.2f}\n")
            f.write(hdr)
            f.write(sep)
            f.write(mrow)
            f.write(rrow)
            f.write(sep)
            f.write(drow)
        f.write("\n")

        # Visit order summary per vehicle
        f.write("  VISIT ORDER\n")
        f.write("-" * W + "\n")
        for vi, vdata in enumerate(stats['per_vehicle'], 1):
            order = " -> ".join(str(sd['node']) for sd in vdata['stop_details'])
            f.write(f"  V{vi:02d}: 0 -> {order} -> 0"
                    f"  [dist={vdata['rdist']:.1f}, load={vdata['load']:.0f}/{cap:.0f}]\n")
        f.write("\n")

        # Detailed timing per vehicle
        f.write("  DETAILED TIMING\n")
        f.write("-" * W + "\n")
        for vi, vdata in enumerate(stats['per_vehicle'], 1):
            f.write(f"\n  -- Vehicle {vi:2d}  [{len(vdata['stops']):3d} stops |"
                    f" dist={vdata['rdist']:7.1f} | load={vdata['load']:.0f}/{cap:.0f}]\n")
            for sd in vdata['stop_details']:
                tw_ok = " LATE" if sd['late'] > 1e-6 else ""
                f.write(f"      {sd['prev']:3d} -> {sd['node']:3d}"
                        f"  ({sd['x']:5.1f},{sd['y']:5.1f})"
                        f"  arr={sd['arrival']:6.1f}  wait={sd['wait']:4.1f}"
                        f"  dep={sd['depart']:6.1f}"
                        f"  TW=[{sd['tw_open']:5.1f},{sd['tw_close']:5.1f}]{tw_ok}"
                        f"  dem={sd['demand']:3.0f}\n")
            f.write(f"      {vdata['prev_last']:3d} ->   0  depot"
                    f"  dist={vdata['ret_dist']:.1f}  arr={vdata['arr_depot']:.1f}/{T:.0f}\n")
        f.write("\n" + "=" * W + "\n")

    print(f"[Solution] saved -> {save_path}")


def _load_sol(data_path: str, n_customers: int):
    base = os.path.splitext(data_path)[0]
    candidates = [
        base + ".sol", base + ".best",
        os.path.join(os.path.dirname(data_path), "solutions",
                     os.path.basename(base) + ".txt"),
        os.path.join(os.path.dirname(data_path), "solutions",
                     os.path.basename(base) + ".sol"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        routes = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if re.search(r'\d+\.\d+', line):
                    continue
                nums = list(map(int, re.findall(r'\b\d+\b', line)))
                cust = [n for n in nums if 1 <= n <= n_customers]
                if len(cust) >= 2:
                    routes.append([0] + cust + [0])
        if routes:
            print(f"[Solution] loaded from {path}")
            return routes
    return None


def plot_routes(inst, routes, total_reward: float, save_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    mc     = float(inst['max_coord'])
    coords = np.vstack([inst['depot_xy'].numpy() * mc, inst['node_xy'].numpy() * mc])
    depot  = coords[0]
    colors = plt.cm.tab10.colors

    _, ax = plt.subplots(figsize=(12, 10))
    for vi, route in enumerate(routes):
        stops = route[1:-1]
        if not stops:
            continue
        color = colors[vi % len(colors)]
        xs = [coords[n][0] for n in route]
        ys = [coords[n][1] for n in route]
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.8)
        for j in range(len(route) - 1):
            x0, y0 = coords[route[j]]
            x1, y1 = coords[route[j+1]]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
        ax.scatter([coords[s][0] for s in stops],
                   [coords[s][1] for s in stops], color=color, s=20, zorder=3)
        for s in stops:
            ax.text(coords[s][0]+0.3, coords[s][1]+0.3, str(s), fontsize=6, color=color)

    served = {s for r in routes for s in r[1:-1]}
    unvis  = [n for n in range(1, inst['n_customers']+1) if n not in served]
    if unvis:
        ax.scatter([coords[n][0] for n in unvis], [coords[n][1] for n in unvis],
                   color="red", s=40, zorder=4, marker="x",
                   label=f"Unserved ({len(unvis)})")

    ax.scatter(depot[0], depot[1], marker="*", s=300, color="black", zorder=5, label="Depot")
    ax.set_title(f"{inst['name']} -- served={len(served)}/{inst['n_customers']}"
                 f"  vehicles={len(routes)}  reward={total_reward:.4f}", fontsize=12)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] route map saved -> {save_path}")


def _save_curve(epochs, values, color, ylabel, title, save_path,
                sparse=False, window=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not values:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    if sparse:
        ax.plot(epochs, values, color=color, linewidth=1.5, marker="o", markersize=4)
    else:
        if window is None:
            window = min(50, max(1, len(values) // 20))
        mavg = [sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window)
                for i in range(len(values))]
        ax.plot(epochs, values, color="lightgray", linewidth=0.6, label="raw")
        ax.plot(epochs, mavg,   color=color,       linewidth=2.0,
                label=f"moving avg (w={window})")
        ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    train_scores: list,
    loss_history:  list,
    plots_dir:     str,
    test_scores:   list | None = None,  # [(epoch, score), ...]
) -> None:
    """Save train_score.png, test_score.png, loss.png separately into plots_dir."""
    eps = list(range(1, len(train_scores) + 1))

    _save_curve(eps, train_scores, "steelblue",
                "Reward (higher = better)", "Train Reward",
                os.path.join(plots_dir, "train_reward.png"))

    if loss_history:
        _save_curve(eps[:len(loss_history)], loss_history, "darkorange",
                    "Loss", "Training Loss",
                    os.path.join(plots_dir, "loss.png"))

    if test_scores:
        te, ts = zip(*test_scores)
        _save_curve(list(te), list(ts), "tomato",
                    "Reward (higher = better)", "Test Reward",
                    os.path.join(plots_dir, "test_reward.png"),
                    sparse=True)
    print(f"[Plot] curves saved -> {plots_dir}")
