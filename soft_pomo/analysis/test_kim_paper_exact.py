"""
test_kim_paper_exact.py

Kim(2006) paper-exact algorithm — while True 수렴 테스트.
MAX_ITER 초과 시 *** LOOP *** 표시.

Usage:
    python test_kim_paper_exact.py
"""
import sys, os, math, random, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vrptw_env import load_solomon

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Solomon')

INSTANCES = (
    [f'C{i:03d}' for i in range(101, 110)] +
    [f'RC{i:03d}' for i in range(101, 109)] +
    [f'R{i:03d}' for i in range(101, 113)]
)

MAX_ITER = 2000  # flag as potential infinite loop if exceeded


def kim_paper_exact_counted(inst: dict, K: int):
    """
    Paper-exact Kim(2006): while True loops with iteration counter.
    Returns (clusters, K_final, total_assign_iters, total_move_iters, K_increments).
    If either loop exceeds MAX_ITER → move_iters = -1 (did not converge).
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
        random.seed(42)
        seeds = random.sample(custs, min(K_try, len(custs)))
        centroid_pos = [[float(coords[s - 1][0]), float(coords[s - 1][1])] for s in seeds]
        clusters = [[] for _ in range(K_try)]
        prev_assign = None
        assign_iters = 0

        # Paper: repeat until stable (no 50-iter cap)
        while True:
            assign_iters += 1
            if assign_iters > MAX_ITER:
                return clusters, assign_iters, -1  # did not converge

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

        # Paper: move-improvement until no improvement
        move_iters = 0
        while True:
            move_iters += 1
            if move_iters > MAX_ITER:
                return clusters, assign_iters, -1

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

        return clusters, assign_iters, move_iters

    K_try = max(1, K)
    total_assign, total_move, k_inc = 0, 0, 0
    for _ in range(N_custs):
        clusters, ai, mi = _run_once(K_try)
        total_assign += ai
        if mi == -1:
            return None, K_try, total_assign, -1, k_inc
        total_move += mi
        raw = [cl for cl in clusters if cl]
        if all(_tw_feasible(cl) for cl in raw):
            break
        K_try += 1
        k_inc += 1

    return [cl for cl in clusters if cl], K_try, total_assign, total_move, k_inc


if __name__ == '__main__':
    hdr = f"{'Instance':<12} {'K_init':>7} {'K_final':>8} {'K_inc':>6} {'AssignIter':>11} {'MoveIter':>9} {'Time(s)':>8}  Status"
    print(hdr)
    print('-' * len(hdr))

    for name in INSTANCES:
        path = os.path.join(DATA_DIR, name + '.txt')
        if not os.path.exists(path):
            print(f'{name:<12}  (missing)')
            continue
        inst = load_solomon(path)
        K_init = math.ceil(float(inst['node_demand'].sum().item()))
        t0 = time.time()
        clusters, K_final, ai, mi, k_inc = kim_paper_exact_counted(inst, K_init)
        elapsed = time.time() - t0
        status = 'CONVERGED' if mi != -1 else '*** LOOP ***'
        print(f'{name:<12} {K_init:>7} {K_final:>8} {k_inc:>6} {ai:>11} {mi:>9} {elapsed:>8.3f}  {status}')
