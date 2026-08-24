"""
VRPTW Environment — based on POMO CVRPEnv, extended with Time Window constraints.

Changes from original CVRPEnv:
  [TW-1] Reset_State: + node_tw_open, node_tw_close, node_service_time, depot_tw_close
  [TW-2] Step_State:  + current_time
  [TW-3] load_problems(): receives Solomon batch dict (not random generation)
  [TW-4] reset():  initialise current_time = 0, total_late = 0, depot_visit_count = 0
  [TW-5] step():   update current_time after each visit
  [TW-6] step():   accumulate lateness = max(0, arrival - tw_close) per customer visit
  [TW-7] step():   count depot visits (= vehicles used)
  [TW-8] reward = -distance - late_penalty * total_late - vehicle_penalty * n_vehicles
  [EVT]  Both rain and accident embedded as encoder node features (8-D total):
           rain:     known at episode start → set in reset_state before pre_forward
           accident: known when triggered   → update reset_state, re-call pre_forward
"""
from dataclasses import dataclass
import torch


@dataclass
class Reset_State:
    depot_xy:           torch.Tensor = None   # (batch, 1, 2)
    node_xy:            torch.Tensor = None   # (batch, N, 2)
    node_demand:        torch.Tensor = None   # (batch, N)
    node_tw_open:       torch.Tensor = None   # (batch, N)   [TW-1]
    node_tw_close:      torch.Tensor = None   # (batch, N)   [TW-1]
    node_service_time:  torch.Tensor = None   # (batch, N)   [TW-1]
    depot_tw_close:     torch.Tensor = None   # (batch, 1)   [TW-1]
    rain_tokens: torch.Tensor = None   # (batch, R, 4) [node/N, mm/100, t_s/T, t_e/T] [EVT]
    acc_tokens:  torch.Tensor = None   # (batch, A, 5) [na/N, nb/N, sev, t_s/T, t_e/T] [EVT]


@dataclass
class Step_State:
    BATCH_IDX:      torch.Tensor = None   # (batch, pomo)
    POMO_IDX:       torch.Tensor = None   # (batch, pomo)
    selected_count: int          = None
    load:           torch.Tensor = None   # (batch, pomo)
    current_node:   torch.Tensor = None   # (batch, pomo)
    ninf_mask:      torch.Tensor = None   # (batch, pomo, N+1)
    current_time:   torch.Tensor = None   # (batch, pomo)  [TW-2]


class VRPTWEnv:
    def __init__(self, **env_params):
        self.env_params   = env_params
        self.problem_size = env_params['problem_size']   # N (customers only)
        self.pomo_size    = env_params['pomo_size']
        self.late_penalty     = env_params.get('late_penalty',     1.0)   # [TW-8]
        self.vehicle_penalty  = env_params.get('vehicle_penalty',  0.0)  # [TW-8]
        self.unserved_penalty = env_params.get('unserved_penalty', 500.0) # [VL] per unserved customer
        self.D_max            = env_params.get('D_max',            1.0)  # [TW-8] reward normalizer
        self.Lt_max           = env_params.get('Lt_max',           1.0)  # [TW-8] reward normalizer
        self.vehicle_limit    = None  # set by load_problems() from instance dict

        # set by load_problems()
        self.batch_size          = None
        self.BATCH_IDX           = None   # (batch, pomo)
        self.POMO_IDX            = None   # (batch, pomo)
        self.depot_node_xy       = None   # (batch, N+1, 2)
        self.depot_node_demand   = None   # (batch, N+1)
        self.depot_node_tw_open  = None   # (batch, N+1)  [TW]
        self.depot_node_tw_close = None   # (batch, N+1)  [TW]
        self.depot_node_service  = None   # (batch, N+1)  [TW]
        self.tt                  = None   # (batch, N+1, N+1) travel-time matrix [TW]

        # dynamic state
        self.selected_count    = None
        self.current_node      = None   # (batch, pomo)
        self.selected_node_list = None  # (batch, pomo, steps)
        self.at_the_depot      = None   # (batch, pomo)
        self.load              = None   # (batch, pomo)
        self.visited_ninf_flag = None   # (batch, pomo, N+1)
        self.ninf_mask         = None   # (batch, pomo, N+1)
        self.finished          = None   # (batch, pomo)
        self.current_time      = None   # (batch, pomo)  [TW-2]

        self.reset_state = Reset_State()
        self.step_state  = Step_State()

    # ------------------------------------------------------------------
    def load_problems(self, batch_dict: dict):
        """
        Load a batched Solomon instance.
        batch_dict produced by VRPTWProblemDef.make_batch().
        """
        depot_xy      = batch_dict['depot_xy']       # (batch, 1, 2)
        node_xy       = batch_dict['node_xy']        # (batch, N, 2)
        node_demand   = batch_dict['node_demand']    # (batch, N)
        node_tw_open  = batch_dict['node_tw_open']   # (batch, N)  [TW]
        node_tw_close = batch_dict['node_tw_close']  # (batch, N)  [TW]
        node_service  = batch_dict['node_service']   # (batch, N)  [TW]
        depot_tw_open  = batch_dict['depot_tw_open'].unsqueeze(-1) if batch_dict['depot_tw_open'].dim()==1 \
                          else batch_dict['depot_tw_open']   # (batch, 1)
        depot_tw_close = batch_dict['depot_tw_close'].unsqueeze(-1) if batch_dict['depot_tw_close'].dim()==1 \
                          else batch_dict['depot_tw_close']  # (batch, 1)
        depot_service  = batch_dict['depot_service'].unsqueeze(-1) if batch_dict['depot_service'].dim()==1 \
                          else batch_dict['depot_service']   # (batch, 1)

        self.batch_size = depot_xy.size(0)
        dev = depot_xy.device

        self.depot_node_xy     = torch.cat((depot_xy, node_xy), dim=1)   # (batch, N+1, 2)
        depot_dem = torch.zeros(self.batch_size, 1, device=dev)
        self.depot_node_demand = torch.cat((depot_dem, node_demand), dim=1)  # (batch, N+1)

        # [TW] TW tensors: depot at index 0, customers at 1..N
        self.depot_node_tw_open  = torch.cat((depot_tw_open,  node_tw_open),  dim=1)  # (batch, N+1)
        self.depot_node_tw_close = torch.cat((depot_tw_close, node_tw_close), dim=1)  # (batch, N+1)
        self.depot_node_service  = torch.cat((depot_service,  node_service),  dim=1)  # (batch, N+1)

        # [TW] travel time matrix
        self.tt = batch_dict['tt']   # (batch, N+1, N+1)

        # [VL] vehicle limit (scalar int or None = unlimited)
        self.vehicle_limit = batch_dict.get('vehicle_limit', None)

        self.BATCH_IDX = torch.arange(self.batch_size, device=dev)[:, None] \
                             .expand(self.batch_size, self.pomo_size)
        self.POMO_IDX  = torch.arange(self.pomo_size, device=dev)[None, :] \
                             .expand(self.batch_size, self.pomo_size)

        # [EVT] event tokens — caller sets before pre_forward
        self.reset_state.rain_tokens = None
        self.reset_state.acc_tokens  = None

        # populate Reset_State
        self.reset_state.depot_xy          = depot_xy
        self.reset_state.node_xy           = node_xy
        self.reset_state.node_demand       = node_demand
        self.reset_state.node_tw_open      = node_tw_open       # [TW-1]
        self.reset_state.node_tw_close     = node_tw_close      # [TW-1]
        self.reset_state.node_service_time = node_service       # [TW-1]
        self.reset_state.depot_tw_close    = depot_tw_close     # [TW-1]

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX  = self.POMO_IDX

    # ------------------------------------------------------------------
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
        self.current_time      = torch.zeros((self.batch_size, self.pomo_size), device=dev)  # [TW-4]
        self.total_late        = torch.zeros((self.batch_size, self.pomo_size), device=dev)  # [TW-4]
        self.depot_visit_count = torch.zeros((self.batch_size, self.pomo_size), device=dev)  # [TW-4]
        self._unserved_count   = torch.zeros((self.batch_size, self.pomo_size), device=dev)  # [VL]

        return self.reset_state, None, False

    # ------------------------------------------------------------------
    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load           = self.load
        self.step_state.current_node   = self.current_node
        self.step_state.ninf_mask      = self.ninf_mask
        self.step_state.current_time   = self.current_time    # [TW-2]
        return self.step_state, None, False

    # ------------------------------------------------------------------
    def step(self, selected):
        # selected.shape: (batch, pomo)
        dev = selected.device
        N1  = self.problem_size + 1

        # ── save previous node for travel-time computation ────────────
        previous_node = self.current_node   # None on first step

        # ── Dynamic-1: position & history ────────────────────────────
        self.selected_count += 1
        self.current_node    = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected[:, :, None]), dim=2)

        # ── Dynamic-2: capacity ───────────────────────────────────────
        self.at_the_depot = (selected == 0)
        demand_list   = self.depot_node_demand[:, None, :].expand(
                            self.batch_size, self.pomo_size, N1)
        gather_idx    = selected[:, :, None]
        selected_dem  = demand_list.gather(dim=2, index=gather_idx).squeeze(2)
        self.load    -= selected_dem
        self.load[self.at_the_depot] = 1.0   # refill at depot

        # ── Dynamic-3: time tracking [TW-5] ──────────────────────────
        if previous_node is None:
            travel_time = torch.zeros(self.batch_size, self.pomo_size, device=dev)
        else:
            travel_time = self.tt[self.BATCH_IDX, previous_node, selected]  # (batch, pomo)

        arrival      = self.current_time + travel_time
        tw_open_sel  = self.depot_node_tw_open [self.BATCH_IDX, selected]   # (batch, pomo)
        tw_close_sel = self.depot_node_tw_close[self.BATCH_IDX, selected]   # (batch, pomo)
        service_sel  = self.depot_node_service [self.BATCH_IDX, selected]   # (batch, pomo)
        self.current_time = torch.max(arrival, tw_open_sel) + service_sel
        self.current_time[self.at_the_depot] = 0.0   # reset clock when returning to depot

        # [TW-6] accumulate lateness (soft penalty — NOT masked) ──────
        lateness = torch.clamp(arrival - tw_close_sel, min=0.0)   # (batch, pomo)
        lateness[self.at_the_depot] = 0.0                         # depot has no TW penalty
        self.total_late = self.total_late + lateness

        # [TW-7] count vehicles: depot return from a customer only
        if previous_node is not None:
            came_from_customer = (previous_node != 0)
            self.depot_visit_count += (self.at_the_depot & came_from_customer).float()

        # ── Visited flag (same as original) ──────────────────────────
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0   # depot stays open unless AT depot

        # ── Build ninf_mask: visited + capacity only (same as original CVRP) ──
        self.ninf_mask = self.visited_ninf_flag.clone()
        demand_too_large = self.load[:, :, None] + 1e-5 < demand_list
        self.ninf_mask[demand_too_large] = float('-inf')

        # [VL] vehicle_limit hard masking: block depot when limit is reached
        if self.vehicle_limit is not None:
            limit_reached = self.depot_visit_count >= self.vehicle_limit  # (batch, pomo)
            self.ninf_mask[:, :, 0][limit_reached & ~self.finished] = float('-inf')

        # ── Finished check ────────────────────────────────────────────
        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished  = self.finished | newly_finished

        # [VL] Deadlock: all nodes masked but rollout not finished (vehicle_limit + capacity clash)
        if self.vehicle_limit is not None:
            deadlocked = (self.ninf_mask == float('-inf')).all(dim=2) & ~self.finished
            if deadlocked.any():
                unserved = (self.visited_ninf_flag[:, :, 1:] != float('-inf')).float().sum(dim=2)
                self._unserved_count[deadlocked] = unserved[deadlocked]
                self.finished = self.finished | deadlocked

        self.ninf_mask[:, :, 0][self.finished] = 0   # depot always reachable for finished rollouts

        # ── Step_State update ─────────────────────────────────────────
        self.step_state.selected_count  = self.selected_count
        self.step_state.load            = self.load
        self.step_state.current_node    = self.current_node
        self.step_state.ninf_mask       = self.ninf_mask
        self.step_state.current_time    = self.current_time    # [TW-2]

        done = self.finished.all()
        if done:
            # [TW-8] reward = -(D/D_max) - late*(Lt/Lt_max) - vehicle*(K/N) - unserved*(U/N)
            reward = (-self._get_travel_distance() / self.D_max
                      - self.late_penalty      * self.total_late         / self.Lt_max
                      - self.vehicle_penalty   * self.depot_visit_count  / self.problem_size
                      - self.unserved_penalty  * self._unserved_count    / self.problem_size)
        else:
            reward = None
        return self.step_state, reward, done

    # ------------------------------------------------------------------
    # Event TT modification helpers
    # ------------------------------------------------------------------

    def apply_rain(self, rain_nodes: list, multiplier: float):
        """
        Slow intra-rain-zone travel times (both-endpoint rule).
        Call AFTER load_problems(). Modifies self.tt in-place.
        Effect: vehicles take longer to traverse rain-affected edges
                -> current_time increases -> TW violations rise.
        """
        if not rain_nodes or multiplier <= 1.0:
            return
        dev = self.tt.device
        idx = torch.tensor(rain_nodes, device=dev, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx <= self.problem_size)]
        if idx.numel() < 2:
            return
        # All (i,j) pairs with i != j
        r = idx.unsqueeze(1).expand(-1, idx.numel())   # (k, k)
        c = idx.unsqueeze(0).expand(idx.numel(), -1)   # (k, k)
        mask = (r != c)
        self.tt[:, r[mask], c[mask]] *= multiplier

    def apply_accident(self, node_a: int, node_b: int, multiplier: float):
        """
        Slow the accident segment (bidirectional). Saves originals for restoration.
        Call at trigger_step during rollout.
        """
        if multiplier <= 1.0:
            return
        self._acc_nodes  = (node_a, node_b)
        self._acc_orig_ab = self.tt[:, node_a, node_b].clone()
        self._acc_orig_ba = self.tt[:, node_b, node_a].clone()
        self.tt[:, node_a, node_b] = self._acc_orig_ab * multiplier
        self.tt[:, node_b, node_a] = self._acc_orig_ba * multiplier

    def restore_accident(self):
        """Restore TT after accident duration expires."""
        if hasattr(self, '_acc_nodes') and self._acc_nodes is not None:
            a, b = self._acc_nodes
            self.tt[:, a, b] = self._acc_orig_ab
            self.tt[:, b, a] = self._acc_orig_ba
            self._acc_nodes = None

    def total_late_violations(self) -> int:
        """Number of (batch, pomo) rollouts with any lateness > 0."""
        return int((self.total_late > 0).sum().item())

    # ------------------------------------------------------------------
    def _get_travel_distance(self):
        """Compute total travel distance for each (batch, pomo) rollout."""
        idx = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        seq    = all_xy.gather(dim=2, index=idx)          # (batch, pomo, steps, 2)
        rolled = seq.roll(dims=2, shifts=-1)
        dists  = ((seq - rolled) ** 2).sum(3).sqrt()      # (batch, pomo, steps)
        return dists.sum(2)                               # (batch, pomo)
