"""Quick test: DRI (Kerscher & Minner 2025) decomposition phase -- STD dissimilarity
+ k-medoids + silhouette-based cluster count -- vs Kim2006 (old/capped), optimal, LKH-3."""
from __future__ import annotations
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from R_utils import (
    _kim2006_clusters, infer_solution_path, parse_solution_routes, validate_solution_routes,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')


# ------------------------------------------------------------
# Kim2006 capped (N fixed at capacity lower bound, no retry)
# ------------------------------------------------------------
def _kim2006_clusters_capped(inst, tt):
    coords = inst.coords
    Q = float(inst.vehicle_capacity)
    n = inst.n_customers
    custs = list(range(1, n + 1))
    depot_close = float(inst.tw_close[0])

    def _d(i, j):
        return float(tt[i, j])

    def _cdist(cx, cy, c):
        return math.sqrt((float(coords[c][0]) - cx) ** 2 + (float(coords[c][1]) - cy) ** 2)

    def _tw_feasible(cluster):
        if not cluster:
            return True
        seq = sorted(cluster, key=lambda x: float(inst.tw_close[x]))
        cur_time, cur_node = 0.0, 0
        for c in seq:
            arr = cur_time + _d(cur_node, c)
            if arr > float(inst.tw_close[c]) + 1e-8:
                return False
            cur_time = max(arr, float(inst.tw_open[c])) + float(inst.service_time[c])
            cur_node = c
        return cur_time + _d(cur_node, 0) <= depot_close + 1e-8

    def _can_add(c, cluster):
        if sum(float(inst.demands[x]) for x in cluster) + float(inst.demands[c]) > Q + 1e-8:
            return False
        return _tw_feasible(cluster + [c])

    total_demand = sum(float(inst.demands[c]) for c in custs)
    N = max(1, int(total_demand // Q) + 1)

    random.seed(42)
    seeds = random.sample(custs, min(N, len(custs)))
    centroid_pos = [[float(coords[s][0]), float(coords[s][1])] for s in seeds]
    clusters = [[] for _ in range(N)]
    prev_assign = None

    for _ in range(50):
        gc_x = sum(cp[0] for cp in centroid_pos) / N
        gc_y = sum(cp[1] for cp in centroid_pos) / N
        sorted_custs = sorted(
            custs,
            key=lambda c: math.sqrt((float(coords[c][0]) - gc_x) ** 2 + (float(coords[c][1]) - gc_y) ** 2),
            reverse=True,
        )
        new_clusters = [[] for _ in range(N)]
        for c in sorted_custs:
            order = sorted(range(N), key=lambda i: _cdist(centroid_pos[i][0], centroid_pos[i][1], c))
            placed = False
            for i in order:
                if _can_add(c, new_clusters[i]):
                    new_clusters[i].append(c)
                    placed = True
                    break
            if not placed:
                new_clusters[order[0]].append(c)
        for i in range(N):
            if new_clusters[i]:
                centroid_pos[i] = [
                    float(np.mean([coords[c][0] for c in new_clusters[i]])),
                    float(np.mean([coords[c][1] for c in new_clusters[i]])),
                ]
        curr_assign = [tuple(sorted(cl)) for cl in new_clusters]
        clusters = new_clusters
        if curr_assign == prev_assign:
            break
        prev_assign = curr_assign

    for _ in range(50):
        cp = [
            [float(np.mean([coords[c][0] for c in cl])), float(np.mean([coords[c][1] for c in cl]))]
            if cl else centroid_pos[i]
            for i, cl in enumerate(clusters)
        ]
        moved = False
        for i, cl in enumerate(clusters):
            for c in list(cl):
                d_own = _cdist(cp[i][0], cp[i][1], c)
                for j, cj in enumerate(clusters):
                    if j == i:
                        continue
                    if _cdist(cp[j][0], cp[j][1], c) >= d_own:
                        continue
                    if not _can_add(c, cj):
                        continue
                    cl.remove(c)
                    cj.append(c)
                    moved = True
                    break
        if not moved:
            break

    return [cl for cl in clusters if cl]


# ------------------------------------------------------------
# DRI decomposition phase (Kerscher & Minner 2025): STD dissimilarity
# + k-medoids + silhouette-based cluster count
# ------------------------------------------------------------
def _std_dissimilarity_matrix(inst, tt):
    n = inst.n_customers
    coords = inst.coords
    x0, y0 = float(coords[0][0]), float(coords[0][1])

    theta = np.zeros(n + 1)
    for i in range(1, n + 1):
        dx = float(coords[i][0]) - x0
        dy = float(coords[i][1]) - y0
        theta[i] = math.atan2(dy, dx)  # matches Eq.(8)+(9) range convention

    lam = sum(float(coords[i][0]) + float(coords[i][1]) for i in range(1, n + 1)) / (2 * n)

    Q = float(inst.vehicle_capacity)
    l0 = float(inst.tw_close[0])
    e0 = float(inst.tw_open[0])
    horizon = l0 - e0

    S = np.zeros((n + 1, n + 1))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue
            dx = float(coords[j][0]) - float(coords[i][0])
            dy = float(coords[j][1]) - float(coords[i][1])
            dth = theta[j] - theta[i]
            s_spatial = math.sqrt(dx * dx + dy * dy + lam * dth * dth)

            f_ij = float(inst.tw_close[j]) - (float(inst.tw_open[i]) + float(inst.service_time[i]) + float(tt[i, j]))
            h_ij = max(float(inst.tw_open[j]) - (float(inst.tw_close[i]) + float(inst.service_time[i]) + float(tt[i, j])), 0.0)

            penalty = 2.0 - (f_ij - h_ij) / horizon + (float(inst.demands[i]) + float(inst.demands[j])) / Q
            S[i, j] = s_spatial * penalty

    Sbar = np.zeros((n + 1, n + 1))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue
            Sbar[i, j] = min(S[i, j], S[j, i])
    return Sbar


def _init_medoids(Sbar, custs):
    # Eq.16: m_i = sum_j Sbar[i,j] / sum_l Sbar[j,l]; pick top-p by m_i
    denom = {j: float(np.sum([Sbar[j, l] for l in custs if l != j])) for j in custs}
    m = {i: float(np.sum([Sbar[i, j] / denom[j] for j in custs if j != i])) for i in custs}
    return sorted(custs, key=lambda i: m[i], reverse=True)


def _kmedoids(Sbar, custs, p, ranked_init, seed=42):
    medoids = ranked_init[:p]

    for _ in range(20):
        clusters = {m: [] for m in medoids}
        for c in custs:
            best_m = min(medoids, key=lambda m: Sbar[c, m])
            clusters[best_m].append(c)

        new_medoids = []
        for m, members in clusters.items():
            best = min(members, key=lambda i: sum(Sbar[i, j] for j in members))
            new_medoids.append(best)

        if sorted(new_medoids) == sorted(medoids):
            break
        medoids = new_medoids

    clusters = {m: [] for m in medoids}
    for c in custs:
        best_m = min(medoids, key=lambda m: Sbar[c, m])
        clusters[best_m].append(c)
    return clusters, medoids


def _silhouette(Sbar, clusters, medoids):
    nearest = {}
    for m in medoids:
        others = [o for o in medoids if o != m]
        nearest[m] = min(others, key=lambda o: Sbar[m, o])

    total, count = 0.0, 0
    for m, members in clusters.items():
        m_prime = nearest[m]
        for i in members:
            a, b = Sbar[i, m], Sbar[i, m_prime]
            denom = max(a, b)
            if denom == 0:
                continue
            total += (b - a) / denom
            count += 1
    return total / count if count > 0 else -1.0


def _dri_clusters(inst, tt):
    n = inst.n_customers
    custs = list(range(1, n + 1))
    Q = float(inst.vehicle_capacity)
    total_demand = sum(float(inst.demands[c]) for c in custs)
    p_min = max(2, int(total_demand // Q) + 1)
    p_max = p_min + 15

    Sbar = _std_dissimilarity_matrix(inst, tt)
    ranked_init = _init_medoids(Sbar, custs)

    best_p, best_sil, best_clusters = None, -1e9, None
    for p in range(p_min, p_max + 1):
        if p >= n:
            break
        clusters, medoids = _kmedoids(Sbar, custs, p, ranked_init, seed=42)
        sil = _silhouette(Sbar, clusters, medoids)
        if sil > best_sil:
            best_sil, best_p, best_clusters = sil, p, clusters

    return [members for members in best_clusters.values()]


# ------------------------------------------------------------
def _lkh3_base_veh(name):
    path = os.path.join(DATA_DIR, f'lkh_eval_{name}.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if '(base)' in line:
                toks = line.split()
                return int(toks[6])
    return None


groups = {
    "C1": [f"c1{i:02d}" for i in range(1, 10)],
    "RC1": [f"rc1{i:02d}" for i in range(1, 9)],
    "R1": [f"r1{i:02d}" for i in range(1, 13)],
}

for gname, names in groups.items():
    print(f"\n=== {gname} ===")
    diffs_old, diffs_cap, diffs_dri = [], [], []
    for name in names:
        path = os.path.join(DATA_DIR, name + '.txt')
        if not os.path.exists(path):
            print(f"{name}  (no data file)")
            continue
        inst = load_waste_benchmark_txt(path)
        tt = build_travel_time_matrix(inst)

        sol_path = infer_solution_path(path, None)
        if not sol_path:
            print(f"{name}  (no _sol file, skipped)")
            continue
        routes = parse_solution_routes(sol_path)
        validate_solution_routes(inst, routes)
        n_opt = len(routes)

        n_old = len(_kim2006_clusters(inst, tt))
        n_cap = len(_kim2006_clusters_capped(inst, tt))
        n_dri = len(_dri_clusters(inst, tt))
        lkh3 = _lkh3_base_veh(name)
        lkh3_s = f"{lkh3:3d}" if lkh3 is not None else "  -"

        diffs_old.append(n_old - n_opt)
        diffs_cap.append(n_cap - n_opt)
        diffs_dri.append(n_dri - n_opt)

        print(f"{name}  kim_old={n_old:3d}  kim_cap={n_cap:3d}  dri={n_dri:3d}  optimal={n_opt:3d}  lkh3={lkh3_s}")

    if diffs_old:
        avg_old = sum(diffs_old) / len(diffs_old)
        avg_cap = sum(diffs_cap) / len(diffs_cap)
        avg_dri = sum(diffs_dri) / len(diffs_dri)
        print(f"  {gname} avg diff: old={avg_old:+.2f}  capped={avg_cap:+.2f}  dri={avg_dri:+.2f}")
