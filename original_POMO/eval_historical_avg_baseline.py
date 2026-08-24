"""
eval_historical_avg_baseline.py

Historical-average TT baseline for C101 using OR-Tools.

Planning strategy (deterministic, no real-time info):
  - C101 BASE / ACC scenarios → plan with M_base  (historical avg from C102-C109 base)
  - C101 RAIN scenarios       → plan with M_rain_avg (avg from C102-C109 RAIN events)

After planning, the FIXED route is simulated on each actual C101 scenario
(with real-time events: rain/accident TT slows edges at the moment of traversal).
This demonstrates the limitation of static planning vs. dynamic RL adaptation.

Outputs Config F reward: -(D/D_max + 4*Lc/N + 4*Lt/Lt_max + K/N)
  where D_max = 10000 (fixed reference), Lt_max = T*N, K = vehicles used
"""
from __future__ import annotations

import math
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from ortools_vrptw import load_solomon_raw, solve, resimulate_with_events

DATA_DIR  = os.path.join(_ROOT, "data", "Solomon")
AVG_DIR   = os.path.join(DATA_DIR, "historical_avg")

# Config F reward parameters (match train_soft_cluster.py)
D_MAX_REF  = 10_000.0
LT_MAX_MUL = 1.0          # Lt_max = T * N (T is in raw time units)
W_D        = 1.0
W_LC       = 4.0
W_LT       = 4.0
W_K        = 1.0


def load_avg_tt(path: str) -> list:
    """Load numpy TT matrix and convert to OR-Tools int list."""
    M = np.load(path)
    n = M.shape[0]
    return [[int(round(M[i, j])) for j in range(n)] for i in range(n)]


def compute_reward_f(metrics: dict, N: int, T: float) -> float:
    """Config F reward: -(D/D_max + 4*Lc/N + 4*Lt/Lt_max + K/N)"""
    D  = metrics["total_distance"]
    Lc = metrics["n_late_stops"]
    Lt = metrics["total_late"]
    K  = metrics.get("vehicles_used", 0)
    Lt_max = T * N
    return -(D / D_MAX_REF + W_LC * Lc / N + W_LT * Lt / Lt_max + W_K * K / N)


def simulate_plan_on_scenario(routes: list, scenario_path: str) -> dict:
    """Simulate a fixed plan (routes) on a scenario with actual events."""
    inst = load_solomon_raw(scenario_path)
    result = resimulate_with_events(routes, inst)
    # Count vehicles used = non-empty routes
    result["vehicles_used"] = sum(1 for r in routes if r)
    result["n_customers"]   = inst["n_customers"]
    result["T"]             = inst["T"]
    return result


def plan_with_tt(base_inst: dict, tt_matrix: list, time_limit: int = 60) -> list:
    """Solve VRPTW with historical-avg TT matrix. Returns routes (list of lists)."""
    result = solve(base_inst, tt_matrix, time_limit_sec=time_limit)
    print(f"  OR-Tools: {result['status']}  vehicles={result['vehicles_used']}"
          f"  served={result['served']}  late={result['served_late']}"
          f"  dist={result['total_distance']:.1f}  ({result['elapsed']:.1f}s)")
    return result["routes"]


def main():
    # ── Load historical avg TT matrices ───────────────────────────────────────
    print("Loading historical avg TT matrices...")
    M_base_path = os.path.join(AVG_DIR, "M_base.npy")
    M_rain_path = os.path.join(AVG_DIR, "M_rain_avg.npy")
    if not os.path.isfile(M_base_path):
        print(f"[ERROR] Missing {M_base_path}. Run compute_historical_avg_tt.py first.")
        sys.exit(1)
    tt_base = load_avg_tt(M_base_path)
    tt_rain = load_avg_tt(M_rain_path)
    print(f"  M_base:     {len(tt_base)}×{len(tt_base)}")
    print(f"  M_rain_avg: {len(tt_rain)}×{len(tt_rain)}")

    # ── Load C101 base instance for planning ──────────────────────────────────
    c101_path = os.path.join(DATA_DIR, "c101.txt")
    c101_base = load_solomon_raw(c101_path)
    N = c101_base["n_customers"]
    T = c101_base["T"]
    print(f"\nC101: N={N} customers, T={T}, vehicles_limit={c101_base['vehicle_limit']}")

    # ── Plan routes (once) using historical avg TT ────────────────────────────
    print("\n[Plan A] BASE/ACC plan → M_base")
    routes_base = plan_with_tt(c101_base, tt_base, time_limit=60)

    print("\n[Plan B] RAIN plan → M_rain_avg")
    routes_rain = plan_with_tt(c101_base, tt_rain, time_limit=60)

    # ── Define test scenarios ─────────────────────────────────────────────────
    scenarios = [
        ("C101_BASE",   "c101.txt",       routes_base, "M_base"),
        ("C101_RAIN_A", "c101_rain_A.txt", routes_rain, "M_rain"),
        ("C101_RAIN_B", "c101_rain_B.txt", routes_rain, "M_rain"),
        ("C101_ACC_A",  "c101_acc_A.txt",  routes_base, "M_base"),
        ("C101_ACC_B",  "c101_acc_B.txt",  routes_base, "M_base"),
    ]

    # ── Simulate and collect results ──────────────────────────────────────────
    print("\n" + "="*75)
    print("  HISTORICAL-AVERAGE BASELINE RESULTS (Config F)")
    print("="*75)
    header = f"{'Scenario':<16} {'Plan':>8} {'K':>4} {'Lc':>5} {'Lt':>9} {'D':>8} {'R_F':>8}"
    print(header)
    print("-"*75)

    results = []
    for name, fname, routes, plan_label in scenarios:
        spath = os.path.join(DATA_DIR, fname)
        m = simulate_plan_on_scenario(routes, spath)
        n_late = m["n_late_stops"]
        total_late = m["total_late"]
        dist = m["total_distance"]
        K = m["vehicles_used"]
        reward = compute_reward_f(m, N, T)
        results.append((name, plan_label, K, n_late, total_late, dist, reward))
        print(f"{name:<16} {plan_label:>8} {K:>4d} {n_late:>5d} {total_late:>9.1f} {dist:>8.1f} {reward:>8.4f}")

    print("-"*75)
    avg_r = sum(r[-1] for r in results) / len(results)
    print(f"{'Average':>16} {'':>8} {'':>4} {'':>5} {'':>9} {'':>8} {avg_r:>8.4f}")
    print("="*75)
    print()
    print("Note: Plan is fixed at planning time (using historical avg TT).")
    print("      Simulation uses actual event TT — vehicle re-routing NOT possible.")
    print(f"      D_max={D_MAX_REF:.0f}, Lt_max=T*N={T:.0f}*{N}={T*N:.0f}")


if __name__ == "__main__":
    main()
