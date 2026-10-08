"""Kim-style grouping; unchanged algorithm extracted from the shared trainer."""

def cluster_customers(inst: dict, K: int) -> list[list[int]]:
    """Kim(2006) capacitated TW-feasible clustering → lists of customer indices (1-indexed).

    Adapts the R_utils._kim2006_clusters() logic to the normalized inst dict format:
    - demands are normalized (capacity Q=1.0)
    - tw/service/tt denormalized by multiplying with inst['T']
    - coords are in real scale (no normalization needed)
    - customer c → coords[c-1], demands[c-1], tw_open[c], tw_close[c], service[c]
    """
    import numpy as np
    import math as _math
    import random as _random

    T        = float(inst['T'])
    tt_raw   = inst['tt'].cpu().numpy() * T                         # (N+1, N+1)
    tw_open  = np.concatenate([inst['depot_tw_open'].numpy() * T,
                               inst['node_tw_open'].numpy()  * T])  # (N+1,)
    tw_close = np.concatenate([inst['depot_tw_close'].numpy() * T,
                               inst['node_tw_close'].numpy()  * T]) # (N+1,)
    service  = np.concatenate([inst['depot_service'].numpy() * T,
                               inst['node_service'].numpy()  * T])  # (N+1,)
    coords   = inst['node_xy'].cpu().numpy()   # (N, 2)  customer 1→coords[0]
    demands  = inst['node_demand'].cpu().numpy()          # (N,) normalized; Q=1.0
    N        = inst['n_customers']
    custs    = list(range(1, N + 1))           # 1-indexed customer IDs
    Q        = 1.0                             # normalized capacity
    depot_close = float(tw_close[0])

    def _d(i: int, j: int) -> float:
        return float(tt_raw[i, j])

    def _cdist(cx: float, cy: float, c: int) -> float:
        return _math.sqrt((float(coords[c - 1][0]) - cx) ** 2 +
                          (float(coords[c - 1][1]) - cy) ** 2)

    def _tw_feasible(cluster: list) -> bool:
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

    def _can_add(c: int, cluster: list) -> bool:
        if sum(float(demands[x - 1]) for x in cluster) + float(demands[c - 1]) > Q + 1e-8:
            return False
        return _tw_feasible(cluster + [c])

    K_limit = min(N, int(inst.get('vehicle_limit') or 25), 25)
    K_try    = max(1, K)
    if K_try > K_limit:
        raise ValueError(f'Kim initial K={K_try} exceeds vehicle limit {K_limit}')
    clusters: list[list[int]] = []

    for _inc in range(30):
        _random.seed(42 + _inc)
        seeds        = _random.sample(custs, min(K_try, len(custs)))
        centroid_pos = [[float(coords[s - 1][0]), float(coords[s - 1][1])] for s in seeds]
        clusters     = [[] for _ in range(K_try)]
        prev_assign  = None

        # Main assignment loop (grand-centroid farthest-first)
        for _ in range(50):
            gc_x = sum(cp[0] for cp in centroid_pos) / K_try
            gc_y = sum(cp[1] for cp in centroid_pos) / K_try

            sorted_custs = sorted(
                custs,
                key=lambda c: _math.sqrt((float(coords[c - 1][0]) - gc_x) ** 2 +
                                         (float(coords[c - 1][1]) - gc_y) ** 2),
                reverse=True,
            )
            new_clusters: list[list[int]] = [[] for _ in range(K_try)]
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

        # Move improvement: relocate node to closer feasible cluster
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
        # Forced assignments above may violate capacity even after relocation.
        # Allow float32 normalization noise, but never return an overloaded group.
        capacity_ok = all(sum(float(demands[c - 1]) for c in cl) <= Q + 1e-6
                          for cl in raw)
        if capacity_ok and (all(_tw_feasible(cl) for cl in raw) or K_try >= K_limit):
            break    # Preserve the existing soft-TW policy at the vehicle limit.
        if K_try >= K_limit:
            raise RuntimeError(
                f'Kim clustering for {inst["name"]} still exceeds capacity '
                f'at vehicle limit {K_limit}; no valid clustering returned')
        K_try += 1   # add one more cluster and retry
    else:
        raise RuntimeError(f'Kim clustering for {inst["name"]} exhausted retries')

    result = [cl for cl in clusters if cl]
    print(f'  [Kim2006] {inst["name"]}: {len(result)} clusters '
          f'(K_init={K}, K_final={K_try})', flush=True)
    return result
