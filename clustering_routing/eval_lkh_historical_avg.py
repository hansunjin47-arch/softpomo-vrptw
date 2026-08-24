"""
eval_lkh_historical_avg.py

LKH-3 static baselines for C1 / RC1 / R1.

Two planning strategies:
  - M_base      : base Euclidean TT (no event info)
  - M_true_hist : time-weighted expected TT averaged over
                  (base + all 4 event files) × training instances
                  E[TT(i,j)] = base × (1 + (mult-1) × active_fraction)
                  This is the best a static planner can do with historical data.

Simulation: fixed plan executed on actual test scenarios with
            time-dependent real event TT.

Usage:
  python eval_lkh_historical_avg.py --benchmark c1
  python eval_lkh_historical_avg.py --benchmark rc1
  python eval_lkh_historical_avg.py --benchmark r1
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from R_lkh import (
    load_solomon,
    solve_vrptw_lkh,
    parse_rain_events,
    evaluate_routes_with_rain,
)

DATA_DIR = os.path.join(_ROOT, "data", "Solomon")

D_MAX_REF = 10_000.0
W_D, W_LC, W_LT, W_K = 1.0, 4.0, 4.0, 1.0
TIME_LIMIT = 60

BENCHMARKS = {
    "c1":  dict(
        train=[f"c10{i}" for i in range(2, 10)],
        test_scenarios=["c101", "c101_rain_A", "c101_rain_B", "c101_acc_A", "c101_acc_B"],
    ),
    "rc1": dict(
        train=[f"rc10{i}" for i in range(2, 9)],
        test_scenarios=["rc101", "rc101_rain_A", "rc101_rain_B", "rc101_acc_A", "rc101_acc_B"],
    ),
    "r1":  dict(
        train=[f"r1{i:02d}" for i in range(2, 13)],
        test_scenarios=["r101", "r101_rain_A", "r101_rain_B", "r101_acc_A", "r101_acc_B"],
    ),
}

EVENT_SUFFIXES = ["rain_A", "rain_B", "acc_A", "acc_B"]


# ── Build M_true_hist ──────────────────────────────────────────────────────────

def euclidean_tt(inst: dict) -> np.ndarray:
    coords = inst["coords"]
    n = len(coords)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            M[i, j] = math.sqrt(dx * dx + dy * dy)
    return M


def _apply_event_factors(base_tt: np.ndarray, events: List[Tuple], T: float,
                         time_weighted: bool) -> np.ndarray:
    """
    Compute event-adjusted TT matrix for a single scenario.
    time_weighted=False : TT(i,j) = base × mult  (full multiplier, no dilution)
    time_weighted=True  : TT(i,j) = base × (1 + (mult-1) × active_duration/T)
                          — dilutes the event impact by its fraction of the horizon
    """
    n = base_tt.shape[0]
    factor = np.ones((n, n), dtype=np.float64)
    for trigger, duration, mult, nodes in events:
        t_s = max(0.0, trigger)
        t_e = min(T, trigger + duration)
        if t_e <= t_s:
            continue
        w = ((t_e - t_s) / T) if time_weighted else 1.0
        delta = (float(mult) - 1.0) * w
        if len(nodes) == 2:
            a, b = nodes[0], nodes[1]
            if a < n and b < n:
                factor[a, b] += delta
                factor[b, a] += delta
        else:
            node_set = set(nodes)
            for i in node_set:
                if i >= n:
                    continue
                for j in node_set:
                    if j >= n or i == j:
                        continue
                    factor[i, j] += delta
    return base_tt * factor


def build_hist_tt(base_tt: np.ndarray, train_instances: List[str], T: float,
                  time_weighted: bool) -> np.ndarray:
    """
    Build historical average TT by averaging over all training event scenarios.

    time_weighted=True  → M_hist_diluted:
        Each event's TT contribution is weighted by active_duration/T.
        Represents average expected TT across all time.

    time_weighted=False → M_hist_full:
        Each event applies its full multiplier (no time dilution).
        Represents expected TT assuming event is always fully active when it occurs.

    Both average across all training instances × (1 base + 4 event) scenarios.
    """
    matrices = []
    for inst_name in train_instances:
        matrices.append(base_tt.copy())  # base (no-event) scenario
        for suffix in EVENT_SUFFIXES:
            ev_path = os.path.join(DATA_DIR, f"{inst_name}_{suffix}.txt")
            if not os.path.isfile(ev_path):
                continue
            events = parse_rain_events(ev_path)
            matrices.append(_apply_event_factors(base_tt, events, T, time_weighted))

    if not matrices:
        return base_tt.copy()
    M = np.mean(matrices, axis=0)
    mask = base_tt > 0
    pct = float(np.mean((M[mask] / base_tt[mask] - 1.0) * 100))
    label = "M_hist_diluted" if time_weighted else "M_hist_full"
    n_base = len(train_instances)
    n_ev   = len(matrices) - n_base
    print(f"  {label}: averaged over {n_base} base + {n_ev} event scenarios  "
          f"(+{pct:.2f}% vs base)")
    return M


def numpy_to_lkh_tt(M: np.ndarray) -> List[List[int]]:
    n = M.shape[0]
    return [[int(round(float(M[i, j]))) for j in range(n)] for i in range(n)]


# ── Simulation & reward ────────────────────────────────────────────────────────

def simulate_fixed_plan(routes, inst, events) -> dict:
    ev = evaluate_routes_with_rain(routes, inst, events)
    return {
        "total_distance": ev["total_distance"],
        "total_late":     ev["total_late"],
        "n_late_stops":   ev["n_late_stops"],
        "vehicles_used":  sum(1 for r in routes if r),
    }


def reward_f(m: dict, N: int, T: float) -> float:
    return -(W_D * m["total_distance"] / D_MAX_REF
             + W_LC * m["n_late_stops"] / N
             + W_LT * m["total_late"] / (T * N)
             + W_K  * m["vehicles_used"] / N)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="c1", choices=list(BENCHMARKS))
    parser.add_argument("--time-limit", type=int, default=TIME_LIMIT)
    args = parser.parse_args()

    cfg         = BENCHMARKS[args.benchmark]
    train_insts = cfg["train"]
    test_name   = cfg["test_scenarios"][0]

    inst_test = load_solomon(os.path.join(DATA_DIR, f"{test_name}.txt"))
    N = inst_test["n_customers"]
    T = inst_test["tw_close"][0]
    print(f"[{args.benchmark.upper()}]  test={test_name}  N={N}  T={T}  "
          f"veh_limit={inst_test['vehicle_limit']}")

    base_tt_np = euclidean_tt(inst_test)

    print(f"\nBuilding M_hist_diluted (time-weighted) ...")
    diluted_tt_np = build_hist_tt(base_tt_np, train_insts, T, time_weighted=True)

    print(f"Building M_hist_full (full multiplier) ...")
    full_tt_np    = build_hist_tt(base_tt_np, train_insts, T, time_weighted=False)

    tt_diluted = numpy_to_lkh_tt(diluted_tt_np)
    tt_full    = numpy_to_lkh_tt(full_tt_np)

    print(f"\n[Plan: M_hist_diluted] (time_limit={args.time_limit}s)...")
    res_d = solve_vrptw_lkh(inst_test, tt_diluted, time_limit_sec=args.time_limit)
    routes_d = res_d["routes"]
    print(f"  K={res_d['vehicles_used']}  dist={res_d['total_distance']:.1f}"
          f"  late={res_d['served_late']}  ({res_d['elapsed']:.1f}s)")

    print(f"\n[Plan: M_hist_full]    (time_limit={args.time_limit}s)...")
    res_f = solve_vrptw_lkh(inst_test, tt_full, time_limit_sec=args.time_limit)
    routes_f = res_f["routes"]
    print(f"  K={res_f['vehicles_used']}  dist={res_f['total_distance']:.1f}"
          f"  late={res_f['served_late']}  ({res_f['elapsed']:.1f}s)")

    print("\n" + "=" * 90)
    print(f"  LKH-3 BASELINES [{args.benchmark.upper()}]")
    print("=" * 90)
    print(f"{'Scenario':<22} {'Plan':<16} {'K':>4} {'Lc':>5} {'Lt':>9} {'D':>8} {'R_F':>8}")
    print("-" * 90)

    rewards_d, rewards_f = [], []
    for sc_name in cfg["test_scenarios"]:
        spath   = os.path.join(DATA_DIR, f"{sc_name}.txt")
        inst_sc = load_solomon(spath)
        events  = parse_rain_events(spath)

        md = simulate_fixed_plan(routes_d, inst_sc, events)
        rd = reward_f(md, N, T); rewards_d.append(rd)
        print(f"{sc_name:<22} {'M_hist_diluted':<16} {md['vehicles_used']:>4d}"
              f" {md['n_late_stops']:>5d} {md['total_late']:>9.2f}"
              f" {md['total_distance']:>9.2f} {rd:>8.4f}")

        mf = simulate_fixed_plan(routes_f, inst_sc, events)
        rf = reward_f(mf, N, T); rewards_f.append(rf)
        print(f"{'':22} {'M_hist_full':<16} {mf['vehicles_used']:>4d}"
              f" {mf['n_late_stops']:>5d} {mf['total_late']:>9.2f}"
              f" {mf['total_distance']:>9.2f} {rf:>8.4f}")
        print()

    print("-" * 90)
    print(f"{'Average':>22} {'M_hist_diluted':<16}"
          f"{'':>5}{'':>6}{'':>10}{'':>9} {sum(rewards_d)/len(rewards_d):>8.4f}")
    print(f"{'':>22} {'M_hist_full':<16}"
          f"{'':>5}{'':>6}{'':>10}{'':>9} {sum(rewards_f)/len(rewards_f):>8.4f}")
    print("=" * 90)
    print(f"\nD_max={D_MAX_REF:.0f}  Lt_max=T*N={T:.0f}x{N}={T*N:.0f}")


if __name__ == "__main__":
    main()
