"""Step 1: STD dissimilarity + k-medoids, p fixed at p_min (= kim_cap's N),
plus a simple capacity-repair rule. Compare zone-composition quality
(NN-tour length, TW-feasibility violations) against Kim2006(capped)."""
from __future__ import annotations
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from _tmp_dri_clusters import (
    DATA_DIR, _kim2006_clusters_capped, _std_dissimilarity_matrix, _init_medoids, _kmedoids,
)


def _std_kmedoids_fixed(inst, tt):
    custs = list(range(1, inst.n_customers + 1))
    Q = float(inst.vehicle_capacity)
    demands = {c: float(inst.demands[c]) for c in custs}
    total_demand = sum(demands.values())
    p_min = max(2, int(total_demand // Q) + 1)

    Sbar = _std_dissimilarity_matrix(inst, tt)
    ranked_init = _init_medoids(Sbar, custs)
    clusters, medoids = _kmedoids(Sbar, custs, p_min, ranked_init, seed=42)

    # capacity repair: move worst-fit member out of any over-capacity cluster
    cl_demand = {m: sum(demands[c] for c in members) for m, members in clusters.items()}
    changed = True
    while changed:
        changed = False
        for m, members in clusters.items():
            if cl_demand[m] <= Q + 1e-8:
                continue
            cand = [c for c in members if c != m]
            if not cand:
                break
            worst = max(cand, key=lambda c: Sbar[c, m])
            others = [(m2, Sbar[worst, m2]) for m2 in medoids
                      if m2 != m and cl_demand[m2] + demands[worst] <= Q + 1e-8]
            if not others:
                break
            target = min(others, key=lambda x: x[1])[0]
            members.remove(worst)
            clusters[target].append(worst)
            cl_demand[m] -= demands[worst]
            cl_demand[target] += demands[worst]
            changed = True
            break

    return [members for members in clusters.values()]


def _nn_tour_length(zone, tt):
    if not zone:
        return 0.0
    unvisited = set(zone)
    cur = 0  # depot
    total = 0.0
    while unvisited:
        nxt = min(unvisited, key=lambda c: tt[cur, c])
        total += float(tt[cur, nxt])
        cur = nxt
        unvisited.remove(nxt)
    total += float(tt[cur, 0])
    return total


def _tw_feasible(zone, inst, tt):
    if not zone:
        return True
    seq = sorted(zone, key=lambda x: float(inst.tw_close[x]))
    cur_time, cur_node = 0.0, 0
    for c in seq:
        arr = cur_time + float(tt[cur_node, c])
        if arr > float(inst.tw_close[c]) + 1e-8:
            return False
        cur_time = max(arr, float(inst.tw_open[c])) + float(inst.service_time[c])
        cur_node = c
    depot_close = float(inst.tw_close[0])
    return cur_time + float(tt[cur_node, 0]) <= depot_close + 1e-8


groups = {
    "C1": [f"c1{i:02d}" for i in range(1, 10)],
    "RC1": [f"rc1{i:02d}" for i in range(1, 9)],
    "R1": [f"r1{i:02d}" for i in range(1, 13)],
}

for gname, names in groups.items():
    print(f"\n=== {gname} ===")
    sum_nn_cap, sum_nn_std = 0.0, 0.0
    sum_infeas_cap, sum_infeas_std = 0, 0
    for name in names:
        path = os.path.join(DATA_DIR, name + '.txt')
        if not os.path.exists(path):
            continue
        inst = load_waste_benchmark_txt(path)
        tt = build_travel_time_matrix(inst)

        zones_cap = _kim2006_clusters_capped(inst, tt)
        zones_std = _std_kmedoids_fixed(inst, tt)

        nn_cap = sum(_nn_tour_length(z, tt) for z in zones_cap)
        nn_std = sum(_nn_tour_length(z, tt) for z in zones_std)
        infeas_cap = sum(0 if _tw_feasible(z, inst, tt) else 1 for z in zones_cap)
        infeas_std = sum(0 if _tw_feasible(z, inst, tt) else 1 for z in zones_std)

        sum_nn_cap += nn_cap
        sum_nn_std += nn_std
        sum_infeas_cap += infeas_cap
        sum_infeas_std += infeas_std

        print(f"{name}  n_cap={len(zones_cap):2d}  n_std={len(zones_std):2d}  "
              f"NN_cap={nn_cap:8.1f}  NN_std={nn_std:8.1f}  "
              f"infeas_cap={infeas_cap}  infeas_std={infeas_std}")

    n = len(names)
    print(f"  {gname} totals: NN_cap={sum_nn_cap:9.1f}  NN_std={sum_nn_std:9.1f}  "
          f"(std/cap ratio={sum_nn_std/sum_nn_cap:.3f})  "
          f"infeas_cap={sum_infeas_cap}  infeas_std={sum_infeas_std}")
