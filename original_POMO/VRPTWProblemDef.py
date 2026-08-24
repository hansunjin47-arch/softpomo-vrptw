"""
VRPTW Problem Definition
- load_solomon(): parse Solomon benchmark file → normalized tensors
- generate_c_type_instance(): random C-type VRPTW instance (Solomon 1987 method)
- augment_xy_data_by_8_fold(): 8-fold geometric augmentation (from original POMO)
"""
import numpy as np
import torch


_ACCIDENT_SEVERITY = {"low": 1.5, "medium": 2.0, "high": 3.0}


def load_solomon(path: str) -> dict:
    """
    Parse Solomon VRPTW benchmark file (with optional EVENTS section).
    Returns dict of normalized tensors (all in [0,1] scale via T or max_coord).
    preset_events: list of dicts with keys type/trigger_time/duration/multiplier/
                   rainfall_mm/probability/nodes — raw time units, 1-based node ids.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    name = lines[0]

    # --- vehicle info ---
    v_idx = next(i for i, l in enumerate(lines) if l.upper().startswith("VEHICLE"))
    vehicle_limit, vehicle_capacity = None, None
    for offset in (1, 2):
        parts = lines[v_idx + offset].split()
        try:
            vehicle_limit = int(parts[0])
            vehicle_capacity = float(parts[1])
            break
        except (ValueError, IndexError):
            continue

    # --- customer data ---
    c_idx = next(i for i, l in enumerate(lines) if l.upper().startswith("CUSTOMER"))
    rows = []
    for ln in lines[c_idx + 1:]:
        parts = ln.split()
        if len(parts) >= 7:
            try:
                rows.append([float(x) for x in parts[:7]])
            except ValueError:
                continue

    data = np.array(rows, dtype=np.float64)
    # cols: cust_no, x, y, demand, tw_open, tw_close, service_time
    # row 0 = depot

    T        = float(data[0, 5])   # depot tw_close = time horizon
    max_coord = float(max(np.max(np.abs(data[:, 1])), np.max(np.abs(data[:, 2])), 1.0))

    coords_raw = data[:, 1:3].astype(np.float32)

    # --- precompute travel time matrix (Solomon: Euclidean distance) ---
    n = len(coords_raw)
    diff = coords_raw[:, None, :] - coords_raw[None, :, :]   # (n, n, 2)
    tt_raw = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)  # (n, n)
    tt_norm = tt_raw / T                                       # normalize by T

    # --- normalize features ---
    coords_norm   = (coords_raw / max_coord).astype(np.float32)   # [0,1]
    demand_norm   = (data[:, 3] / vehicle_capacity).astype(np.float32)
    tw_open_norm  = (data[:, 4] / T).astype(np.float32)
    tw_close_norm = (data[:, 5] / T).astype(np.float32)
    svc_norm      = (data[:, 6] / T).astype(np.float32)

    # --- EVENTS section (optional) ---
    preset_events = []
    try:
        ev_start = next(i for i, l in enumerate(lines) if l.upper() == "EVENTS")
        for ln in lines[ev_start + 1:]:
            toks = ln.split()
            if not toks or toks[0].startswith('#'):
                continue
            kw = toks[0].upper()
            if kw not in ("RAIN", "ACCIDENT"):
                continue
            trigger  = float(toks[1])
            duration = float(toks[2])
            if kw == "RAIN":
                multiplier  = float(toks[3])
                rainfall_mm = float(toks[4])
                probability = float(toks[5])
                nodes       = [int(x) for x in toks[6:]]
            else:  # ACCIDENT
                severity    = toks[3].lower()
                multiplier  = _ACCIDENT_SEVERITY.get(severity)
                if multiplier is None:
                    multiplier = float(toks[3])   # numeric form e.g. "3.0"
                probability = float(toks[4])
                nodes       = [int(x) for x in toks[5:]]
                rainfall_mm = 0.0
            preset_events.append(dict(
                type=kw, trigger_time=trigger, duration=duration,
                multiplier=multiplier, rainfall_mm=rainfall_mm,
                probability=probability, nodes=nodes,
            ))
    except StopIteration:
        pass

    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32)

    return dict(
        name             = name,
        n_customers      = n - 1,
        vehicle_limit    = vehicle_limit,
        vehicle_capacity = vehicle_capacity,
        T                = T,
        max_coord        = max_coord,
        preset_events    = preset_events,   # [] if no EVENTS section
        # depot: index 0, customers: index 1..n
        depot_xy         = _t(coords_norm[[0]]),          # (1, 2)
        node_xy          = _t(coords_norm[1:]),           # (N, 2)
        depot_demand     = _t(demand_norm[[0]]),          # (1,)
        node_demand      = _t(demand_norm[1:]),           # (N,)
        depot_tw_open    = _t(tw_open_norm[[0]]),         # (1,)
        depot_tw_close   = _t(tw_close_norm[[0]]),        # (1,)  = 1.0
        depot_service    = _t(svc_norm[[0]]),             # (1,)
        node_tw_open     = _t(tw_open_norm[1:]),          # (N,)
        node_tw_close    = _t(tw_close_norm[1:]),         # (N,)
        node_service     = _t(svc_norm[1:]),              # (N,)
        tt               = _t(tt_norm),                   # (N+1, N+1)
    )


def generate_c_type_instance(
    n_customers: int = 100,
    n_clusters:  int = 10,
    T:           float = 1236.0,
    capacity:    float = 200.0,
    service_time: float = 90.0,
    tw_sigma:    float = None,
    rng = None,
) -> dict:
    """
    Generate a random C-type VRPTW instance following Solomon (1987).

    Coordinates in [0, 100]² (clustered Gaussian).
    TW center = arrival time from nearest-neighbor tour per cluster.
    TW half-width = |N(0, tw_sigma)|.
    tw_sigma=None → sampled uniformly in [45, 230] to span C101–C109 tightness range.

    Returns the same dict format as load_solomon() (normalized tensors, no events).
    """
    if rng is None:
        rng = np.random.default_rng()
    if tw_sigma is None:
        tw_sigma = float(rng.uniform(45.0, 230.0))

    max_coord = 100.0

    # ── 1. Cluster centers and customer coordinates ────────────────────────────
    centers = rng.uniform(15, 85, size=(n_clusters, 2))

    # distribute customers evenly across clusters
    base, extra = divmod(n_customers, n_clusters)
    cluster_ids = np.repeat(np.arange(n_clusters), base)
    cluster_ids = np.concatenate([cluster_ids, np.arange(extra)])
    rng.shuffle(cluster_ids)

    cluster_sigma = 8.0
    coords = np.zeros((n_customers, 2))
    for i, cid in enumerate(cluster_ids):
        coords[i] = np.clip(centers[cid] + rng.normal(0, cluster_sigma, 2), 0, max_coord)

    # depot: random in centre region (matching Solomon C1 depot at (40,50))
    depot = rng.uniform(20, 80, size=(1, 2))

    # ── 2. Demands and service times ───────────────────────────────────────────
    demands = rng.integers(10, 51, size=n_customers).astype(np.float64)
    svc     = np.full(n_customers, service_time, dtype=np.float64)

    # ── 3. Travel-time matrix (Euclidean) ─────────────────────────────────────
    all_xy  = np.vstack([depot, coords])          # (N+1, 2), row 0 = depot
    diff    = all_xy[:, None, :] - all_xy[None, :, :]
    tt      = np.sqrt((diff ** 2).sum(-1))        # (N+1, N+1), raw units

    # ── 4. TW via nearest-neighbour tour per cluster ───────────────────────────
    tw_open  = np.zeros(n_customers, dtype=np.float64)
    tw_close = np.full(n_customers, T, dtype=np.float64)

    for cid in range(n_clusters):
        members = [i for i, c in enumerate(cluster_ids) if c == cid]
        if not members:
            continue

        unvisited = set(members)
        cur       = 0        # depot index in all_xy
        cur_t     = 0.0

        while unvisited:
            # nearest unvisited customer
            dists   = {j: tt[cur, j + 1] for j in unvisited}
            nxt     = min(dists, key=dists.get)
            travel  = dists[nxt]
            arrival = cur_t + travel

            # TW: center = arrival, half-width = |N(0, sigma)|
            half_w  = abs(float(rng.normal(0, tw_sigma)))
            tw_o    = max(0.0, arrival - half_w)
            tw_c    = min(T,   arrival + half_w)

            # ensure minimum feasibility: reachable from depot and not degenerate
            tw_o = max(tw_o, tt[0, nxt + 1])  # must be reachable from depot
            if tw_c - tw_o < 1.0:
                tw_c = min(T, tw_o + max(2 * half_w, 10.0))

            tw_open[nxt]  = tw_o
            tw_close[nxt] = tw_c

            # advance time: wait if early, then serve
            svc_start = max(arrival, tw_o)
            cur_t     = svc_start + service_time
            cur       = nxt + 1    # index in all_xy
            unvisited.remove(nxt)

    # ── 5. Normalize and package ───────────────────────────────────────────────
    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32)

    depot_xy_n  = (depot  / max_coord).astype(np.float32)
    node_xy_n   = (coords / max_coord).astype(np.float32)
    demand_n    = (demands / capacity).astype(np.float32)
    tw_open_n   = (tw_open  / T).astype(np.float32)
    tw_close_n  = (tw_close / T).astype(np.float32)
    svc_n       = (svc / T).astype(np.float32)
    tt_n        = (tt  / T).astype(np.float32)

    return dict(
        name             = f'rand_C_s{tw_sigma:.0f}',
        n_customers      = n_customers,
        vehicle_limit    = 25,
        vehicle_capacity = capacity,
        T                = T,
        max_coord        = max_coord,
        preset_events    = [],
        depot_xy         = _t(depot_xy_n),           # (1, 2)
        node_xy          = _t(node_xy_n),            # (N, 2)
        depot_demand     = _t(np.array([0.0], dtype=np.float32)),
        node_demand      = _t(demand_n),             # (N,)
        depot_tw_open    = _t(np.array([0.0], dtype=np.float32)),
        depot_tw_close   = _t(np.array([1.0], dtype=np.float32)),
        depot_service    = _t(np.array([0.0], dtype=np.float32)),
        node_tw_open     = _t(tw_open_n),            # (N,)
        node_tw_close    = _t(tw_close_n),           # (N,)
        node_service     = _t(svc_n),                # (N,)
        tt               = _t(tt_n),                 # (N+1, N+1)
    )


def make_batch(inst: dict, batch_size: int, device: torch.device) -> dict:
    """Repeat a single Solomon instance batch_size times (no memory copy via expand)."""
    keys_2d = ['depot_xy', 'node_xy']
    keys_1d = ['depot_demand', 'node_demand', 'depot_tw_open', 'depot_tw_close',
               'depot_service', 'node_tw_open', 'node_tw_close', 'node_service']

    out = {}
    for k in keys_2d:
        t = inst[k].to(device)   # (1,2) or (N,2)
        out[k] = t.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    for k in keys_1d:
        t = inst[k].to(device)   # (1,) or (N,)
        out[k] = t.unsqueeze(0).expand(batch_size, -1).contiguous()
    # tt: (N+1, N+1) → (batch, N+1, N+1)
    out['tt'] = inst['tt'].to(device).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    return out


def augment_xy_data_by_8_fold(xy_data: torch.Tensor) -> torch.Tensor:
    """8-fold geometric augmentation. xy_data: (batch, N, 2) → (8*batch, N, 2)"""
    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    return torch.cat([
        torch.cat((x,     y    ), dim=2),
        torch.cat((1 - x, y    ), dim=2),
        torch.cat((x,     1 - y), dim=2),
        torch.cat((1 - x, 1 - y), dim=2),
        torch.cat((y,     x    ), dim=2),
        torch.cat((1 - y, x    ), dim=2),
        torch.cat((y,     1 - x), dim=2),
        torch.cat((1 - y, 1 - x), dim=2),
    ], dim=0)
