"""Step1 + fallback: increment q while any zone is TW-infeasible (Kim2006-style retry)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from R_utils import infer_solution_path, parse_solution_routes, validate_solution_routes
from _tmp_dri_clusters import (
    DATA_DIR, _kim2006_clusters_capped, _std_dissimilarity_matrix, _init_medoids, _kmedoids,
)
from _tmp_dri_step1 import _nn_tour_length, _tw_feasible


def _capacity_repair(Sbar, clusters, medoids, demands, Q):
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
    return clusters


def _std_kmedoids_fallback(inst, tt, max_q):
    custs = list(range(1, inst.n_customers + 1))
    Q = float(inst.vehicle_capacity)
    demands = {c: float(inst.demands[c]) for c in custs}
    total_demand = sum(demands.values())
    p_min = max(2, int(total_demand // Q) + 1)

    Sbar = _std_dissimilarity_matrix(inst, tt)
    ranked_init = _init_medoids(Sbar, custs)

    q = p_min
    while True:
        clusters, medoids = _kmedoids(Sbar, custs, q, ranked_init, seed=42)
        clusters = _capacity_repair(Sbar, clusters, medoids, demands, Q)
        zones = [members for members in clusters.values()]
        infeas = sum(0 if _tw_feasible(z, inst, tt) else 1 for z in zones)
        if infeas == 0 or q >= max_q:
            return zones, q, infeas, p_min
        q += 1


groups = {
    "C1": [f"c1{i:02d}" for i in range(1, 10)],
    "RC1": [f"rc1{i:02d}" for i in range(1, 9)],
    "R1": [f"r1{i:02d}" for i in range(1, 13)],
}

MAX_Q = 40

for gname, names in groups.items():
    print(f"\n=== {gname} ===")
    diffs = []
    nn_ratios = []
    for name in names:
        path = os.path.join(DATA_DIR, name + '.txt')
        if not os.path.exists(path):
            continue
        inst = load_waste_benchmark_txt(path)
        tt = build_travel_time_matrix(inst)

        sol_path = infer_solution_path(path, None)
        routes = parse_solution_routes(sol_path)
        validate_solution_routes(inst, routes)
        n_opt = len(routes)

        zones_cap = _kim2006_clusters_capped(inst, tt)
        nn_cap = sum(_nn_tour_length(z, tt) for z in zones_cap)

        zones_std, q_final, infeas, p_min = _std_kmedoids_fallback(inst, tt, MAX_Q)
        nn_std = sum(_nn_tour_length(z, tt) for z in zones_std)

        diffs.append(q_final - n_opt)
        nn_ratios.append(nn_std / nn_cap)

        flag = "" if infeas == 0 else f"  (HIT MAX_Q, still infeas={infeas})"
        print(f"{name}  p_min={p_min:2d}  q_final={q_final:2d}  n_opt={n_opt:2d}  "
              f"diff={q_final-n_opt:+3d}  NN_cap={nn_cap:7.1f}  NN_std={nn_std:7.1f}  "
              f"ratio={nn_std/nn_cap:.3f}{flag}")

    avg_diff = sum(diffs) / len(diffs)
    avg_ratio = sum(nn_ratios) / len(nn_ratios)
    print(f"  {gname}: avg(q_final-n_opt)={avg_diff:+.2f}  avg NN ratio(std/cap)={avg_ratio:.3f}")
