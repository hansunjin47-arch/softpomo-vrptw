"""
Ontology-based simultaneous clustering + routing for pure VRPTW.

No pre-clustering. At each step the ontology scores all feasible customers
using domain rules and selects the highest-scoring action.
Routes form implicitly through sequential decisions.

Ontology concepts
  UrgentCustomer    : tw_close - service_start ≤ URGENCY_THRESHOLD
  EarlyCustomer     : must wait (arrival < tw_open)
  SavingsCandidate  : Clarke-Wright savings > 0

Scoring rules (weighted combination)
  1. TimeWindowUrgency   : prefer tight-deadline customers
  2. GeographicProximity : prefer nearby customers (route compactness)
  3. ClarkeWrightSavings : prefer customers that save vs depot detour
  4. WaitPenalty         : penalize long waits (early customers)
"""
from __future__ import annotations

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import argparse
import os
import time
from typing import List

import numpy as np

from CR_env import VRPTWEnv, load_solomon, build_tt

# ── Ontology thresholds ───────────────────────────────────────────────────────
URGENCY_THRESHOLD = 30.0   # tw_slack ≤ this → UrgentCustomer

# ── Scoring weights ───────────────────────────────────────────────────────────
W_URGENCY   = 3.0   # TimeWindowUrgency rule weight
W_PROXIMITY = 1.0   # GeographicProximity rule weight
W_SAVINGS   = 1.5   # ClarkeWrightSavings rule weight
W_WAIT      = 0.5   # WaitPenalty weight (subtracted)
W_UTIL      = 0.5   # CapacityUtilization rule weight
DEPOT_BIAS  = 0.05  # Base score for depot return — customer score must exceed this


# ============================================================
# Ontology
# ============================================================

class VRPTWPureOntology:
    """
    Rule-based ontology for simultaneous clustering + routing.

    Concepts (TBox)
    ---------------
    UrgentCustomer  : remaining time window ≤ URGENCY_THRESHOLD after arrival
    EarlyCustomer   : vehicle arrives before tw_open (must wait)
    SavingsCandidate: Clarke-Wright savings > 0 vs depot detour

    Properties (ABox, computed per step)
    -------------------------------------
    tw_slack      : tw_close - max(arrival, tw_open)
    wait_time     : max(0, tw_open - arrival)
    cw_savings    : tt[cur,depot] + tt[depot,c] - tt[cur,c]
    proximity     : 1 / (travel + wait + 1)
    util_ratio    : demand / remaining_cap

    Rules (scoring)
    ---------------
    score(c) = W_URGENCY / (tw_slack + 1)
             + W_PROXIMITY / (travel + wait + 1)
             + W_SAVINGS * max(0, savings) / (|savings| + 1)
             + W_UTIL * demand / (remaining_cap + 1e-6)
             - W_WAIT * wait / (T + 1)
    """

    def __init__(self, env: VRPTWEnv):
        self.env = env

    # ── Concept classifiers ──────────────────────────────────────────────────

    def is_urgent(self, c: int, arrival: float) -> bool:
        tw_slack = float(self.env.inst.tw_close[c]) - max(
            arrival, float(self.env.inst.tw_open[c])
        )
        return tw_slack <= URGENCY_THRESHOLD

    def is_early(self, arrival: float, tw_open: float) -> bool:
        return arrival < tw_open

    def cw_savings(self, c: int, cur_node: int) -> float:
        return (
            float(self.env.tt[cur_node, 0])
            + float(self.env.tt[0, c])
            - float(self.env.tt[cur_node, c])
        )

    # ── Scoring ──────────────────────────────────────────────────────────────

    def score_candidates(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute ontology scores for all actions.
        mask[0] = depot (always True), mask[1..N] = customers.
        Returns float array of shape [N+1].
        """
        env = self.env
        av  = env.vehicles[env.active_idx]
        T   = env._T
        scores = np.full(env.N + 1, -np.inf)

        # Depot return: low base score — prefer visiting customers
        scores[0] = DEPOT_BIAS

        for c in range(1, env.N + 1):
            if not mask[c]:
                continue

            travel  = float(env.tt[av.cur_node, c])
            arrival = av.cur_time + travel
            tw_o    = float(env.inst.tw_open[c])
            tw_c    = float(env.inst.tw_close[c])
            demand  = float(env.inst.demands[c])
            wait    = max(0.0, tw_o - arrival)
            svc_start = max(arrival, tw_o)
            tw_slack  = tw_c - svc_start

            # Rule 1: TimeWindowUrgency
            urgency_score = W_URGENCY / (tw_slack + 1.0)

            # Rule 2: GeographicProximity (with wait penalty built in)
            proximity_score = W_PROXIMITY / (travel + wait + 1.0)

            # Rule 3: ClarkeWrightSavings
            s = self.cw_savings(c, av.cur_node)
            savings_score = W_SAVINGS * max(0.0, s) / (abs(s) + 1.0)

            # Rule 4: CapacityUtilization — prefer customers that fill remaining cap efficiently
            util_score = W_UTIL * demand / (av.remaining_cap + 1e-6)

            # Rule 5: WaitPenalty (subtract)
            wait_penalty = W_WAIT * wait / (T + 1.0)

            scores[c] = urgency_score + proximity_score + savings_score + util_score - wait_penalty

        return scores

    def select_action(self, mask: np.ndarray) -> int:
        scores = self.score_candidates(mask)
        scores[~mask] = -np.inf
        return int(np.argmax(scores))


# ============================================================
# Episode runner
# ============================================================

def run_episode(env: VRPTWEnv, ontology: VRPTWPureOntology, max_steps: int) -> float:
    env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        mask = env.get_feasible_mask()
        if not mask.any():
            break
        action = ontology.select_action(mask)
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


def _fmt(ep: int, reward: float, env: VRPTWEnv, ep_time: float) -> str:
    served   = int(np.sum(env.served[1:]))
    late_cnt = env.late_count
    on_time  = served - late_cnt
    unserved = env.N - served
    late_avg = env.total_late / late_cnt if late_cnt > 0 else 0.0
    return (
        f"[ONTOLOGY EP {ep:4d}]"
        f" reward={reward:10.2f}"
        f"  served={on_time:3d}"
        f"  tw_late={late_cnt:3d}"
        f"  unserved={unserved:3d}/{env.N}"
        f"  veh={env.vehicles_used:2d}/{env.max_vehicles:2d}"
        f"  dist={env.total_distance:10.2f}"
        f"  late={env.total_late:9.2f}"
        f"  late_avg={late_avg:8.2f}"
        f"  ep_time={ep_time:.1f}s"
        f"  done={env.last_end_reason}"
    )


# ============================================================
# Main
# ============================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Ontology-based simultaneous C+R for pure VRPTW",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("instance", nargs="?", default="c101",
                   help="Solomon instance name (e.g. c101) or full path")
    p.add_argument("--data-dir",  default=os.path.join(_ROOT, "data", "Solomon"))
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--max-steps",  type=int, default=600)
    # env rewards
    p.add_argument("--visit-reward",        type=float, default=2.0)
    p.add_argument("--late-penalty-fixed",  type=float, default=4.0)
    p.add_argument("--late-penalty-scale",  type=float, default=5.0)
    p.add_argument("--unserved-penalty",    type=float, default=8.0)
    p.add_argument("--vehicle-use-penalty", type=float, default=10.0)
    # ontology weights
    p.add_argument("--w-urgency",   type=float, default=W_URGENCY)
    p.add_argument("--w-proximity", type=float, default=W_PROXIMITY)
    p.add_argument("--w-savings",   type=float, default=W_SAVINGS)
    p.add_argument("--w-util",      type=float, default=W_UTIL)
    p.add_argument("--w-wait",      type=float, default=W_WAIT)
    return p.parse_args()


def main():
    args = _parse_args()

    inst_arg = args.instance
    if os.path.isfile(inst_arg):
        data_path = inst_arg
    else:
        name = inst_arg if inst_arg.endswith(".txt") else inst_arg + ".txt"
        data_path = os.path.join(args.data_dir, name)

    inst = load_solomon(data_path)
    tt   = build_tt(inst)
    env  = VRPTWEnv(
        inst=inst, tt=tt,
        visit_reward        = args.visit_reward,
        late_penalty_fixed  = args.late_penalty_fixed,
        late_penalty_scale  = args.late_penalty_scale,
        unserved_penalty    = args.unserved_penalty,
        vehicle_use_penalty = args.vehicle_use_penalty,
    )

    # Apply CLI weight overrides
    global W_URGENCY, W_PROXIMITY, W_SAVINGS, W_UTIL, W_WAIT
    W_URGENCY   = args.w_urgency
    W_PROXIMITY = args.w_proximity
    W_SAVINGS   = args.w_savings
    W_UTIL      = args.w_util
    W_WAIT      = args.w_wait

    ontology = VRPTWPureOntology(env)

    print(f"[ONTOLOGY] instance={inst.name}  N={inst.n_customers}  vehicles={inst.vehicle_limit}")
    print(f"[ONTOLOGY] weights: urgency={W_URGENCY}  proximity={W_PROXIMITY}"
          f"  savings={W_SAVINGS}  util={W_UTIL}  wait={W_WAIT}\n")

    best_reward = -1e9
    best_served = -1
    best_line   = ""
    t0 = time.time()

    for ep in range(1, args.n_episodes + 1):
        ep_t0   = time.time()
        reward  = run_episode(env, ontology, args.max_steps)
        ep_time = time.time() - ep_t0
        line    = _fmt(ep, reward, env, ep_time)

        served = int(np.sum(env.served[1:]))
        is_best = served > best_served or (served == best_served and reward > best_reward)
        if is_best:
            best_served, best_reward = served, reward
            best_line = line
            print("[Best EP] " + line)

        print(line)

    total = time.time() - t0
    print(f"\n[Time] total={total:.1f}s  best_served={best_served}")
    if best_line:
        print("[Best EP] " + best_line)


if __name__ == "__main__":
    main()
