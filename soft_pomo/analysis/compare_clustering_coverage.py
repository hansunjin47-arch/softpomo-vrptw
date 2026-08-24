"""
compare_clustering_coverage.py

K-means vs Kim(2006) clustering coverage 비교.
Coverage = fraction of cluster nodes that appear in the best-matching optimal route.

Usage:
    python compare_clustering_coverage.py --benchmark c1
    python compare_clustering_coverage.py --benchmark rc1
    python compare_clustering_coverage.py --benchmark r1
    python compare_clustering_coverage.py  # runs all three
    python compare_clustering_coverage.py --coverage   # coverage + K table for all benchmarks
"""
import sys, os, json, math, random, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'original_POMO'))
from vrptw_env import load_solomon

HERE     = os.path.dirname(__file__)
SOL_DIR  = os.path.join(HERE, '..', '..', 'data', 'Solomon')
DATA_DIR = SOL_DIR

BENCHMARKS = {
    'c1':  [f'c{i:03d}' for i in range(101, 110)],
    'c2':  [f'c{i:03d}' for i in range(201, 209)],
    'r1':  [f'r{i:03d}' for i in range(101, 113)],
    'r2':  [f'r{i:03d}' for i in range(201, 212)],
    'rc1': [f'rc{i:03d}' for i in range(101, 109)],
    'rc2': [f'rc{i:03d}' for i in range(201, 209)],
}


def load_sol(path):
    routes = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('Route'):
                nodes = list(map(int, s.split(':')[1].split()))
                routes.append(nodes)
    return routes


# ── K-means clustering ────────────────────────────────────────────────────────

def kmeans_cluster(inst: dict, K: int) -> list:
    coords = inst['node_xy'].numpy()
    N = len(coords)
    K = min(K, N)
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=10, random_state=0)
        labels = km.fit_predict(coords)
    except ImportError:
        from scipy.cluster.vq import kmeans2
        np.random.seed(0)
        _, labels = kmeans2(coords, K, minit='points')
    clusters = [[] for _ in range(K)]
    for i, lbl in enumerate(labels):
        clusters[int(lbl)].append(i + 1)
    return [c for c in clusters if c]


# ── Kim(2006) clustering ──────────────────────────────────────────────────────

def kim2006_cluster(inst: dict, K: int, fixed_k: bool = False) -> list:
    import math as _math

    T        = float(inst['T'])
    import torch
    tt_raw   = inst['tt'].numpy() * T
    tw_open  = np.concatenate([inst['depot_tw_open'].numpy() * T,
                               inst['node_tw_open'].numpy()  * T])
    tw_close = np.concatenate([inst['depot_tw_close'].numpy() * T,
                               inst['node_tw_close'].numpy()  * T])
    service  = np.concatenate([inst['depot_service'].numpy() * T,
                               inst['node_service'].numpy()  * T])
    coords   = inst['node_xy'].numpy()
    demands  = inst['node_demand'].numpy()
    N        = inst['n_customers']
    custs    = list(range(1, N + 1))
    Q        = 1.0
    depot_close = float(tw_close[0])

    def _d(i, j):
        return float(tt_raw[i, j])

    def _cdist(cx, cy, c):
        return _math.sqrt((float(coords[c - 1][0]) - cx) ** 2 +
                          (float(coords[c - 1][1]) - cy) ** 2)

    def _tw_feasible(cluster):
        if not cluster:
            return True
        seq = sorted(cluster, key=lambda x: float(tw_close[x]))
        cur_time, cur_node = 0.0, 0
        for c in seq:
            arr = cur_time + _d(cur_node, c)
            if arr > float(tw_close[c]) + 1e-8:
                return False
            cur_time = max(arr, float(tw_open[c])) + float(service[c])
            cur_node = c
        return cur_time + _d(cur_node, 0) <= depot_close + 1e-8

    def _can_add(c, cluster):
        if sum(float(demands[x - 1]) for x in cluster) + float(demands[c - 1]) > Q + 1e-8:
            return False
        return _tw_feasible(cluster + [c])

    K_try    = max(1, K)
    clusters = []

    for _inc in range(30):
        random.seed(42 + _inc)
        seeds        = random.sample(custs, min(K_try, len(custs)))
        centroid_pos = [[float(coords[s - 1][0]), float(coords[s - 1][1])] for s in seeds]
        clusters     = [[] for _ in range(K_try)]
        prev_assign  = None

        for _ in range(50):
            gc_x = sum(cp[0] for cp in centroid_pos) / K_try
            gc_y = sum(cp[1] for cp in centroid_pos) / K_try
            sorted_custs = sorted(
                custs,
                key=lambda c: _math.sqrt((float(coords[c - 1][0]) - gc_x) ** 2 +
                                         (float(coords[c - 1][1]) - gc_y) ** 2),
                reverse=True,
            )
            new_clusters = [[] for _ in range(K_try)]
            for c in sorted_custs:
                order = sorted(range(K_try),
                               key=lambda i: _cdist(centroid_pos[i][0], centroid_pos[i][1], c))
                placed = False
                for i in order:
                    if _can_add(c, new_clusters[i]):
                        new_clusters[i].append(c)
                        placed = True
                        break
                if not placed:
                    new_clusters[order[0]].append(c)

            for i in range(K_try):
                if new_clusters[i]:
                    centroid_pos[i] = [
                        float(np.mean([coords[c - 1][0] for c in new_clusters[i]])),
                        float(np.mean([coords[c - 1][1] for c in new_clusters[i]])),
                    ]

            curr_assign = [tuple(sorted(cl)) for cl in new_clusters]
            clusters    = new_clusters
            if curr_assign == prev_assign:
                break
            prev_assign = curr_assign

        for _ in range(50):
            cp = [
                [float(np.mean([coords[c - 1][0] for c in cl])),
                 float(np.mean([coords[c - 1][1] for c in cl]))]
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
                        cp[i] = ([float(np.mean([coords[x - 1][0] for x in cl])),
                                  float(np.mean([coords[x - 1][1] for x in cl]))]
                                 if cl else cp[i])
                        cp[j] = [float(np.mean([coords[x - 1][0] for x in cj])),
                                 float(np.mean([coords[x - 1][1] for x in cj]))]
                        moved = True
                        break
            if not moved:
                break

        raw = [cl for cl in clusters if cl]
        if fixed_k or all(_tw_feasible(cl) for cl in raw):
            break
        K_try += 1

    return [cl for cl in clusters if cl], K_try


# ── Kim(2006) paper-exact clustering ─────────────────────────────────────────

def kim2006_paper_exact(inst: dict, K: int) -> tuple:
    """
    Kim(2006) algorithm as described in the paper.

    Differences from kim2006_cluster() (which was used in R_utils.py):
      - Single fixed seed (42), NO 30-restart outer loop
      - Assignment / move-improvement loops run until full convergence (no 50-iter cap)
      - K increment: only when TW-infeasible, then restart with same seed on K+1

    This corresponds to the algorithm in Section 3 of Kim et al. (2006).
    """
    import math as _math

    T        = float(inst['T'])
    tt_raw   = inst['tt'].numpy() * T
    tw_open  = np.concatenate([inst['depot_tw_open'].numpy() * T,
                               inst['node_tw_open'].numpy()  * T])
    tw_close = np.concatenate([inst['depot_tw_close'].numpy() * T,
                               inst['node_tw_close'].numpy()  * T])
    service  = np.concatenate([inst['depot_service'].numpy() * T,
                               inst['node_service'].numpy()  * T])
    coords   = inst['node_xy'].numpy()
    demands  = inst['node_demand'].numpy()
    N_custs  = inst['n_customers']
    custs    = list(range(1, N_custs + 1))
    Q        = 1.0
    depot_close = float(tw_close[0])

    def _d(i, j):
        return float(tt_raw[i, j])

    def _cdist(cx, cy, c):
        return _math.sqrt((float(coords[c - 1][0]) - cx) ** 2 +
                          (float(coords[c - 1][1]) - cy) ** 2)

    def _tw_feasible(cluster):
        if not cluster:
            return True
        seq = sorted(cluster, key=lambda x: float(tw_close[x]))
        cur_time, cur_node = 0.0, 0
        for c in seq:
            arr = cur_time + _d(cur_node, c)
            if arr > float(tw_close[c]) + 1e-8:
                return False
            cur_time = max(arr, float(tw_open[c])) + float(service[c])
            cur_node = c
        return cur_time + _d(cur_node, 0) <= depot_close + 1e-8

    def _can_add(c, cluster):
        if sum(float(demands[x - 1]) for x in cluster) + float(demands[c - 1]) > Q + 1e-8:
            return False
        return _tw_feasible(cluster + [c])

    def _run_once(K_try):
        # Paper step 1: random initialisation (single seed — no restarts)
        random.seed(42)
        seeds = random.sample(custs, min(K_try, len(custs)))
        centroid_pos = [[float(coords[s - 1][0]), float(coords[s - 1][1])] for s in seeds]
        clusters = [[] for _ in range(K_try)]
        prev_assign = None

        # Paper steps 2-4: grand-centroid farthest-first assignment until stable
        for _ in range(200):
            gc_x = sum(cp[0] for cp in centroid_pos) / K_try
            gc_y = sum(cp[1] for cp in centroid_pos) / K_try
            sorted_custs = sorted(
                custs,
                key=lambda c: _math.sqrt((float(coords[c - 1][0]) - gc_x) ** 2 +
                                         (float(coords[c - 1][1]) - gc_y) ** 2),
                reverse=True,
            )
            new_clusters = [[] for _ in range(K_try)]
            for c in sorted_custs:
                order = sorted(range(K_try),
                               key=lambda i: _cdist(centroid_pos[i][0], centroid_pos[i][1], c))
                placed = False
                for i in order:
                    if _can_add(c, new_clusters[i]):
                        new_clusters[i].append(c)
                        placed = True
                        break
                if not placed:
                    new_clusters[order[0]].append(c)

            for i in range(K_try):
                if new_clusters[i]:
                    centroid_pos[i] = [
                        float(np.mean([coords[c - 1][0] for c in new_clusters[i]])),
                        float(np.mean([coords[c - 1][1] for c in new_clusters[i]])),
                    ]

            curr_assign = [tuple(sorted(cl)) for cl in new_clusters]
            clusters = new_clusters
            if curr_assign == prev_assign:
                break
            prev_assign = curr_assign

        # Paper step 5: move-improvement until stable
        for _ in range(200):
            cp = [
                [float(np.mean([coords[c - 1][0] for c in cl])),
                 float(np.mean([coords[c - 1][1] for c in cl]))]
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
                        cp[i] = ([float(np.mean([coords[x - 1][0] for x in cl])),
                                  float(np.mean([coords[x - 1][1] for x in cl]))]
                                 if cl else cp[i])
                        cp[j] = [float(np.mean([coords[x - 1][0] for x in cj])),
                                 float(np.mean([coords[x - 1][1] for x in cj]))]
                        moved = True
                        break
            if not moved:
                break

        return clusters

    # Paper step 6: if TW-infeasible, increment K and restart (deterministically)
    K_try = max(1, K)
    clusters = []
    for _ in range(N_custs):  # safety upper bound
        clusters = _run_once(K_try)
        raw = [cl for cl in clusters if cl]
        if all(_tw_feasible(cl) for cl in raw):
            break
        K_try += 1

    return [cl for cl in clusters if cl], K_try


# ── Coverage metric ───────────────────────────────────────────────────────────

def compute_coverage(clusters: list, routes: list) -> float:
    """Average fraction of cluster nodes found in the best-matching optimal route."""
    if not clusters or not routes:
        return 0.0
    covs = []
    for cset_list in clusters:
        cset = set(cset_list)
        if not cset:
            continue
        best = max(routes, key=lambda r: len(set(r) & cset))
        overlap = len(set(best) & cset)
        covs.append(overlap / len(cset))
    return sum(covs) / len(covs) if covs else 0.0


# ── Main comparison ───────────────────────────────────────────────────────────

def run_benchmark(bm_name: str):
    instances = BENCHMARKS[bm_name]
    print(f'\n=== {bm_name.upper()} : K_init / K_final(Kim2006) / K_final(paper-exact) / K_opt ===')
    print(f'{"Instance":<12} {"K_init":>7} {"K_final":>8} {"K_paper":>8} {"K_opt":>7} '
          f'{"Kf/Ko":>7} {"Kp/Ko":>7}')
    print('-' * 65)

    k_inits, k_finals, k_papers, k_opts = [], [], [], []
    for name in instances:
        inst_path = os.path.join(DATA_DIR, f'{name.upper()}.txt')
        sol_path  = os.path.join(SOL_DIR,  f'{name}_sol.txt')
        if not os.path.exists(inst_path):
            print(f'{name:<12}  (no instance file)')
            continue
        if not os.path.exists(sol_path):
            print(f'{name:<12}  (no solution file)')
            continue

        inst   = load_solomon(inst_path)
        routes = load_sol(sol_path)
        K_init = math.ceil(float(inst['node_demand'].sum().item()))
        K_opt  = len(routes)

        _, K_final = kim2006_cluster(inst, K_init, fixed_k=False)
        _, K_paper = kim2006_paper_exact(inst, K_init)

        ratio_final = K_final / K_opt
        ratio_paper = K_paper / K_opt

        k_inits.append(K_init); k_finals.append(K_final)
        k_papers.append(K_paper); k_opts.append(K_opt)
        print(f'{name:<12} {K_init:>7} {K_final:>8} {K_paper:>8} {K_opt:>7} '
              f'{ratio_final:>7.2f}x {ratio_paper:>7.2f}x')

    if k_opts:
        print('-' * 65)
        avg_rf = sum(k_finals) / sum(k_opts)
        avg_rp = sum(k_papers) / sum(k_opts)
        print(f'{"avg":<12} {sum(k_inits)//len(k_inits):>7} '
              f'{sum(k_finals)//len(k_finals):>8} {sum(k_papers)//len(k_papers):>8} '
              f'{sum(k_opts)//len(k_opts):>7} '
              f'{avg_rf:>7.2f}x {avg_rp:>7.2f}x')
    print()


def run_coverage_comparison(bm_name: str):
    """Coverage comparison: K-means vs Kim2006(current) vs Kim2006(paper-exact)."""
    instances = BENCHMARKS[bm_name]
    print(f'\n=== {bm_name.upper()} : Coverage comparison ===')
    print(f'{"Instance":<12} {"K_init":>7} {"K_opt":>7} {"Kmeans":>8} '
          f'{"Kim-cur":>9} {"Kim-ppr":>9} {"Kf_cur":>8} {"Kf_ppr":>8}')
    print('-' * 75)

    cov_km, cov_cur, cov_ppr = [], [], []
    for name in instances:
        inst_path = os.path.join(DATA_DIR, f'{name.upper()}.txt')
        sol_path  = os.path.join(SOL_DIR,  f'{name}_sol.txt')
        if not os.path.exists(inst_path) or not os.path.exists(sol_path):
            print(f'{name:<12}  (missing file)')
            continue

        inst   = load_solomon(inst_path)
        routes = load_sol(sol_path)
        K_init = math.ceil(float(inst['node_demand'].sum().item()))
        K_opt  = len(routes)

        cl_km          = kmeans_cluster(inst, K_init)
        cl_cur, K_fc   = kim2006_cluster(inst, K_init, fixed_k=False)
        cl_ppr, K_fp   = kim2006_paper_exact(inst, K_init)

        cv_km  = compute_coverage(cl_km,  routes)
        cv_cur = compute_coverage(cl_cur, routes)
        cv_ppr = compute_coverage(cl_ppr, routes)

        cov_km.append(cv_km); cov_cur.append(cv_cur); cov_ppr.append(cv_ppr)
        print(f'{name:<12} {K_init:>7} {K_opt:>7} {cv_km:>8.1%} '
              f'{cv_cur:>9.1%} {cv_ppr:>9.1%} {K_fc:>8} {K_fp:>8}')

    if cov_km:
        print('-' * 75)
        print(f'{"avg":<12} {"":>7} {"":>7} {sum(cov_km)/len(cov_km):>8.1%} '
              f'{sum(cov_cur)/len(cov_cur):>9.1%} {sum(cov_ppr)/len(cov_ppr):>9.1%}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=list(BENCHMARKS.keys()), default=None)
    parser.add_argument('--coverage', action='store_true',
                        help='Run coverage comparison (K-means vs Kim-current vs Kim-paper-exact)')
    args = parser.parse_args()

    if args.coverage:
        bm_list = [args.benchmark] if args.benchmark else ['c1', 'c2', 'r1', 'r2', 'rc1', 'rc2']
        for bm in bm_list:
            run_coverage_comparison(bm)
    elif args.benchmark:
        run_benchmark(args.benchmark)
    else:
        for bm in ['c1', 'c2', 'r1', 'r2', 'rc1', 'rc2']:
            run_benchmark(bm)
