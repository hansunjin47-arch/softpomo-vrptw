"""
generate_vrptw.py — Random VRPTW instance generator (Solomon 1987 method).

Supported types:
  C  : Gaussian-clustered customers, TW centered on NN-tour arrival times.
  R  : Uniform-random customers, TW center uniform in feasible interval.
  RC : Mixed (half clustered + half random), TW center uniform in feasible interval.

Per-type defaults (from actual Solomon benchmark data):
  Type  T     svc   sigma_min  sigma_max   demand_range
  C     1236   90      45        231        {10,20,30,40,50} geometric-decay (mean≈18)
  R      230   10       6         74        Uniform int [1,40] (mean≈20)
  RC     240   10      19         97        Uniform int [1,40] (mean≈20)

tw_sigma relationship: avg_TW_width ≈ 1.596 × tw_sigma
  C: c101 avg=72→sigma=45, c109 avg=369→sigma=231
  R: r101 avg=10→sigma=6,  r112 avg=118→sigma=74
  RC: rc101 avg=30→sigma=19, rc104 avg=155→sigma=97

n_clusters (C/RC only): derived as ceil(N × mean_demand / capacity), not hardcoded.

Usage:
  python generate_vrptw.py --type C  --n 1000 --out data/rand_C_1000.pt
  python generate_vrptw.py --type R  --n 1000 --out data/rand_R_1000.pt
  python generate_vrptw.py --type RC --n 1000 --out data/rand_RC_1000.pt
  python generate_vrptw.py --type C  --n 500  --sigma 45 --out data/rand_C_tight.pt

The saved .pt file is a list of inst dicts (same format as load_solomon()).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import numpy as np
import torch

# Demand distribution matching Solomon C1 empirical structure
# Values: {10,20,30,40,50}, probabilities ≈ geometric decay with ratio 0.5
_DEMAND_VALUES = np.array([10, 20, 30, 40, 50])
_DEMAND_PROBS  = np.array([0.52, 0.26, 0.13, 0.07, 0.02])
_DEMAND_MEAN   = float((_DEMAND_VALUES * _DEMAND_PROBS).sum())  # ≈ 18.1


# ── Shared helpers ────────────────────────────────────────────────────────────

def _sample_demands_c(n: int, rng: np.random.Generator) -> np.ndarray:
    """Demands for C-type: {10,20,30,40,50} geometric-decay (matches Solomon C1, mean≈18)."""
    return rng.choice(_DEMAND_VALUES, size=n, p=_DEMAND_PROBS).astype(np.float64)


def _sample_demands_r(n: int, rng: np.random.Generator) -> np.ndarray:
    """Demands for R/RC-type: Uniform integer [1,40] (matches Solomon R1/RC1, mean≈20)."""
    return rng.integers(1, 41, size=n).astype(np.float64)


def _n_clusters_from_capacity(n_customers: int, mean_demand: float, capacity: float) -> int:
    """
    Derive number of clusters from capacity constraint (Solomon's implicit approach).
    n_clusters = ceil(N × mean_demand / capacity)
    C-type (mean≈18): ceil(100 × 18 / 200) = 9 → 10 in practice.
    """
    return max(1, math.ceil(n_customers * mean_demand / capacity))


def _tt_matrix(coords: np.ndarray) -> np.ndarray:
    """Euclidean travel-time matrix. coords: (N+1, 2), row 0 = depot."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def _package(
    depot: np.ndarray, coords: np.ndarray,
    demands: np.ndarray, svc: np.ndarray,
    tw_open: np.ndarray, tw_close: np.ndarray,
    tt: np.ndarray,
    T: float, capacity: float, tw_sigma: float, name_prefix: str,
) -> dict:
    """Normalize and package into load_solomon()-compatible dict."""
    max_coord = 100.0

    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32)

    return dict(
        name             = f'rand_{name_prefix}_s{tw_sigma:.0f}',
        n_customers      = len(coords),
        vehicle_limit    = 25,
        vehicle_capacity = capacity,
        T                = T,
        max_coord        = max_coord,
        preset_events    = [],
        depot_xy         = _t((depot   / max_coord).astype(np.float32)),
        node_xy          = _t((coords  / max_coord).astype(np.float32)),
        depot_demand     = _t(np.zeros(1, dtype=np.float32)),
        node_demand      = _t((demands / capacity).astype(np.float32)),
        depot_tw_open    = _t(np.zeros(1, dtype=np.float32)),
        depot_tw_close   = _t(np.ones(1,  dtype=np.float32)),
        depot_service    = _t(np.zeros(1, dtype=np.float32)),
        node_tw_open     = _t((tw_open  / T).astype(np.float32)),
        node_tw_close    = _t((tw_close / T).astype(np.float32)),
        node_service     = _t((svc      / T).astype(np.float32)),
        tt               = _t((tt / T).astype(np.float32)),
    )


def _sample_tw_sigma(tw_sigma, tw_sigma_min, tw_sigma_max, rng) -> float:
    if tw_sigma is not None:
        return float(tw_sigma)
    return float(rng.uniform(tw_sigma_min, tw_sigma_max))


# ── C-type generator ──────────────────────────────────────────────────────────

def generate_c_type_instance(
    n_customers:   int   = 100,
    T:             float = 1236.0,
    capacity:      float = 200.0,
    tw_sigma:      float = None,
    tw_sigma_min:  float = 45.0,
    tw_sigma_max:  float = 230.0,
    cluster_sigma: float = 8.0,
    svc_min:       float = 70.0,
    svc_max:       float = 110.0,
    rng = None,
) -> dict:
    """
    C-type: Gaussian-clustered coordinates, TW centered on NN-tour arrival times.

    n_clusters derived automatically: ceil(N × mean_demand / capacity).
    Service times: Uniform(svc_min, svc_max) per customer (centered around 90 = Solomon C1).
    tw_sigma sampled per instance in [tw_sigma_min, tw_sigma_max].
    """
    if rng is None:
        rng = np.random.default_rng()
    tw_sigma = _sample_tw_sigma(tw_sigma, tw_sigma_min, tw_sigma_max, rng)

    n_clusters = _n_clusters_from_capacity(n_customers, _DEMAND_MEAN, capacity)

    # 1. Cluster centers and customer coordinates
    centers = rng.uniform(15, 85, size=(n_clusters, 2))
    base, extra = divmod(n_customers, n_clusters)
    cluster_ids = np.concatenate([
        np.repeat(np.arange(n_clusters), base),
        np.arange(extra),
    ])
    rng.shuffle(cluster_ids)

    coords = np.zeros((n_customers, 2))
    for i, cid in enumerate(cluster_ids):
        coords[i] = np.clip(centers[cid] + rng.normal(0, cluster_sigma, 2), 0, 100.0)

    depot = rng.uniform(20, 80, size=(1, 2))

    # 2. Demands and service times
    demands = _sample_demands_c(n_customers, rng)
    svc     = rng.uniform(svc_min, svc_max, size=n_customers)

    # 3. Travel-time matrix
    all_xy = np.vstack([depot, coords])
    tt     = _tt_matrix(all_xy)

    # 4. TW via nearest-neighbour tour per cluster
    tw_open  = np.zeros(n_customers, dtype=np.float64)
    tw_close = np.full(n_customers, T, dtype=np.float64)

    for cid in range(n_clusters):
        members   = [i for i, c in enumerate(cluster_ids) if c == cid]
        unvisited = set(members)
        cur, cur_t = 0, 0.0

        while unvisited:
            dists   = {j: tt[cur, j + 1] for j in unvisited}
            nxt     = min(dists, key=dists.get)
            arrival = cur_t + dists[nxt]

            half_w = abs(float(rng.normal(0, tw_sigma)))
            tw_o   = max(tt[0, nxt + 1], arrival - half_w)
            tw_c   = min(T, arrival + half_w)
            if tw_c - tw_o < 1.0:
                tw_c = min(T, tw_o + max(2 * half_w, 10.0))

            tw_open[nxt]  = tw_o
            tw_close[nxt] = tw_c

            svc_start = max(arrival, tw_o)
            cur_t     = svc_start + svc[nxt]
            cur       = nxt + 1
            unvisited.remove(nxt)

    return _package(depot, coords, demands, svc, tw_open, tw_close, tt,
                    T, capacity, tw_sigma, 'C')


# ── R-type generator ──────────────────────────────────────────────────────────

def generate_r_type_instance(
    n_customers:  int   = 100,
    T:            float = 230.0,
    capacity:     float = 200.0,
    tw_sigma:     float = None,
    tw_sigma_min: float = 6.0,
    tw_sigma_max: float = 74.0,
    svc_min:      float = 9.0,
    svc_max:      float = 11.0,
    rng = None,
) -> dict:
    """
    R-type: uniform random coordinates, TW center uniform in feasible interval.

    Per Solomon (1987): TW center ~ Uniform(e0 + t_{0,i}, l0 - t_{i,0} - s_i).
    TW half-width = |N(0, tw_sigma)| per customer.
    Defaults match Solomon R1 class: T=230, svc=10, sigma in [6,74] (r101→r112).
    tw_sigma sampled per instance in [tw_sigma_min, tw_sigma_max].
    """
    if rng is None:
        rng = np.random.default_rng()
    tw_sigma = _sample_tw_sigma(tw_sigma, tw_sigma_min, tw_sigma_max, rng)

    # 1. Random coordinates
    coords = rng.uniform(0, 100, size=(n_customers, 2))
    depot  = rng.uniform(20, 80, size=(1, 2))

    # 2. Demands and service times (R1: integer demands in [1,40], svc=10)
    demands = _sample_demands_r(n_customers, rng)
    svc     = rng.uniform(svc_min, svc_max, size=n_customers)

    # 3. Travel-time matrix
    all_xy = np.vstack([depot, coords])
    tt     = _tt_matrix(all_xy)

    # 4. TW: center = Uniform(t_{depot→i}, T - t_{i→depot} - svc_i)
    tw_open  = np.zeros(n_customers, dtype=np.float64)
    tw_close = np.full(n_customers, T, dtype=np.float64)

    for i in range(n_customers):
        t_from_depot = tt[0, i + 1]
        t_to_depot   = tt[i + 1, 0]
        feasible_lo  = t_from_depot
        feasible_hi  = T - t_to_depot - svc[i]

        if feasible_hi <= feasible_lo:
            # Customer unreachable within T — give [0, T] (unconstrained)
            tw_open[i], tw_close[i] = 0.0, T
            continue

        center = float(rng.uniform(feasible_lo, feasible_hi))
        half_w = abs(float(rng.normal(0, tw_sigma)))

        tw_o = max(feasible_lo, center - half_w)
        tw_c = min(T, center + half_w)
        if tw_c - tw_o < 1.0:
            tw_c = min(T, tw_o + max(2 * half_w, 10.0))

        tw_open[i]  = tw_o
        tw_close[i] = tw_c

    return _package(depot, coords, demands, svc, tw_open, tw_close, tt,
                    T, capacity, tw_sigma, 'R')


# ── RC-type generator ─────────────────────────────────────────────────────────

_RC_DEMAND_MEAN = 20.5  # mean of Uniform[1,40]


def generate_rc_type_instance(
    n_customers:   int   = 100,
    T:             float = 240.0,
    capacity:      float = 200.0,
    tw_sigma:      float = None,
    tw_sigma_min:  float = 19.0,
    tw_sigma_max:  float = 97.0,
    cluster_sigma: float = 8.0,
    clustered_frac: float = 0.5,
    svc_min:       float = 9.0,
    svc_max:       float = 11.0,
    rng = None,
) -> dict:
    """
    RC-type: mix of clustered and random coordinates (Solomon semi-clustered).

    clustered_frac of customers placed in Gaussian clusters, rest uniform random.
    TW generation follows R-type (center uniform in feasible interval).
    Defaults match Solomon RC1 class: T=240, svc=10, sigma in [19,97] (rc101→rc104).
    tw_sigma sampled per instance in [tw_sigma_min, tw_sigma_max].
    """
    if rng is None:
        rng = np.random.default_rng()
    tw_sigma = _sample_tw_sigma(tw_sigma, tw_sigma_min, tw_sigma_max, rng)

    n_clusters  = _n_clusters_from_capacity(n_customers, _RC_DEMAND_MEAN, capacity)
    n_clustered = int(round(n_customers * clustered_frac))
    n_random    = n_customers - n_clustered

    # 1. Clustered portion
    centers = rng.uniform(15, 85, size=(n_clusters, 2))
    base, extra = divmod(n_clustered, n_clusters)
    cluster_ids = np.concatenate([
        np.repeat(np.arange(n_clusters), base),
        np.arange(extra),
    ])
    rng.shuffle(cluster_ids)

    coords_c = np.zeros((n_clustered, 2))
    for i, cid in enumerate(cluster_ids):
        coords_c[i] = np.clip(centers[cid] + rng.normal(0, cluster_sigma, 2), 0, 100.0)

    # 2. Random portion
    coords_r = rng.uniform(0, 100, size=(n_random, 2))

    coords = np.vstack([coords_c, coords_r])
    rng.shuffle(coords)   # shuffle so clustered/random are intermixed

    depot = rng.uniform(20, 80, size=(1, 2))

    # 3. Demands and service times (RC: integer demands in [1,40], svc≈10)
    demands = _sample_demands_r(n_customers, rng)
    svc     = rng.uniform(svc_min, svc_max, size=n_customers)

    # 4. Travel-time matrix
    all_xy = np.vstack([depot, coords])
    tt     = _tt_matrix(all_xy)

    # 5. TW: R-type method (uniform center in feasible interval)
    tw_open  = np.zeros(n_customers, dtype=np.float64)
    tw_close = np.full(n_customers, T, dtype=np.float64)

    for i in range(n_customers):
        t_from_depot = tt[0, i + 1]
        t_to_depot   = tt[i + 1, 0]
        feasible_lo  = t_from_depot
        feasible_hi  = T - t_to_depot - svc[i]

        if feasible_hi <= feasible_lo:
            tw_open[i], tw_close[i] = 0.0, T
            continue

        center = float(rng.uniform(feasible_lo, feasible_hi))
        half_w = abs(float(rng.normal(0, tw_sigma)))

        tw_o = max(feasible_lo, center - half_w)
        tw_c = min(T, center + half_w)
        if tw_c - tw_o < 1.0:
            tw_c = min(T, tw_o + max(2 * half_w, 10.0))

        tw_open[i]  = tw_o
        tw_close[i] = tw_c

    return _package(depot, coords, demands, svc, tw_open, tw_close, tt,
                    T, capacity, tw_sigma, 'RC')


# ── Unified generator ─────────────────────────────────────────────────────────

def generate_unified_instance(
    n_customers:      int   = 100,
    capacity:         float = 200.0,
    T:                float = None,
    clustered_frac:   float = None,
    cluster_sigma:    float = 8.0,
    tw_sigma_rel:     float = None,
    svc_rel:          float = None,
    T_min:            float = 200.0,
    T_max:            float = 1300.0,
    tw_sigma_rel_min: float = 0.025,
    tw_tight_max:     float = 0.080,
    tw_medium_max:    float = 0.220,
    tw_sigma_rel_max: float = 0.404,
    svc_rel_min:      float = 0.040,
    svc_rel_max:      float = 0.080,
    t_corr_svc:       bool  = True,
    rng = None,
) -> dict:
    """
    Unified VRPTW generator — continuous parameter space over Solomon (1987).

    C/R/RC are special cases of two continuous dimensions:
      clustered_frac  in [0, 1]         : 0=all-random(R), 1=all-clustered(C)
      tw_sigma_rel    in [0.025, 0.40]  : sigma/T ratio (tight → wide TW)

    Per-instance sampling (Solomon empirical ranges):
      T             ~ Uniform[200, 1300]
      clustered_frac~ Uniform[0, 1]
      tw_sigma_rel  ~ Uniform[0.025, 0.404]  (TW tightness; sigma_rel already T-normalized)
      svc_rel       ~ T-correlated: 0.043 at T=230 (R/RC1) → 0.073 at T=1236 (C1)

    svc T-correlation (Solomon-aligned):
      R1/RC1 (T≈230): svc=10  → svc_rel≈0.043
      C1     (T≈1236): svc=90 → svc_rel≈0.073
      Linear interpolation between these two anchors.

    TW: uniform sampling over full range — sigma_rel is already T-normalized,
      so no T-correlation needed. C/R/RC differences are captured via clustered_frac.

    Solomon benchmark coverage:
      C-style : clustered_frac≈1.0, tw_sigma_rel in [0.036, 0.187], svc_rel≈0.073
      R-style : clustered_frac≈0.0, tw_sigma_rel in [0.026, 0.322], svc_rel≈0.043
      RC-style: clustered_frac≈0.5, tw_sigma_rel in [0.079, 0.404], svc_rel≈0.042

    TW generation follows Solomon R/RC method:
      center ~ Uniform(t_depot→i, T - t_i→depot - svc_i)
      half-width ~ |N(0, tw_sigma)|
    This is equivalent to C-type method when clustered_frac=1 and tw_sigma_rel is small
    (clustered coords → tight arrival spread → similar to NN-tour arrival TW).
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── per-instance parameter sampling ───────────────────────────────────────
    if T is None:
        T = float(rng.uniform(T_min, T_max))
    if clustered_frac is None:
        clustered_frac = float(rng.uniform(0.0, 1.0))
    if tw_sigma_rel is None:
        # Uniform sampling over full range — stratified(1/3) was tried but hurt R1 perf
        tw_sigma_rel = float(rng.uniform(tw_sigma_rel_min, tw_sigma_rel_max))
    if svc_rel is None:
        if t_corr_svc:
            # T-correlated: R1/RC1 (T≈230) → svc_rel≈0.043, C1 (T≈1236) → svc_rel≈0.073
            _svc_rel_base = 0.043 + (0.073 - 0.043) * (T - 230.0) / (1236.0 - 230.0)
            _svc_rel_base = max(svc_rel_min, min(svc_rel_max, _svc_rel_base))
            svc_rel = float(rng.uniform(_svc_rel_base * 0.9, _svc_rel_base * 1.1))
        else:
            # 구 generator: uniform sampling (T-correlation 없음)
            svc_rel = float(rng.uniform(svc_rel_min, svc_rel_max))

    tw_sigma = tw_sigma_rel * T
    svc_mean = svc_rel * T

    # ── coordinates: clustered_frac × N Gaussian clusters, rest uniform ───────
    n_clustered = int(round(n_customers * clustered_frac))
    n_random    = n_customers - n_clustered
    mean_demand = 20.5  # Uniform[1,40] mean
    n_clusters  = _n_clusters_from_capacity(n_customers, mean_demand, capacity)

    coords = np.zeros((n_customers, 2))

    if n_clustered > 0:
        centers = rng.uniform(15, 85, size=(n_clusters, 2))
        base, extra = divmod(n_clustered, n_clusters)
        cids = np.concatenate([
            np.repeat(np.arange(n_clusters), base),
            np.arange(extra),
        ])
        rng.shuffle(cids)
        for i, cid in enumerate(cids):
            coords[i] = np.clip(centers[cid] + rng.normal(0, cluster_sigma, 2), 0, 100.0)

    if n_random > 0:
        coords[n_clustered:] = rng.uniform(0, 100, size=(n_random, 2))

    rng.shuffle(coords)  # intermix clustered/random

    depot = rng.uniform(20, 80, size=(1, 2))

    # ── demands and service times ─────────────────────────────────────────────
    demands = _sample_demands_r(n_customers, rng)       # Uniform[1,40]
    svc_val = max(float(rng.uniform(svc_mean * 0.9, svc_mean * 1.1)), 0.1)
    svc     = np.full(n_customers, svc_val)

    # ── travel-time matrix ────────────────────────────────────────────────────
    all_xy = np.vstack([depot, coords])
    tt     = _tt_matrix(all_xy)

    # ── time windows (Solomon R/RC method: uniform center in feasible interval)
    tw_open  = np.zeros(n_customers, dtype=np.float64)
    tw_close = np.full(n_customers, T, dtype=np.float64)

    for i in range(n_customers):
        t_from_depot = tt[0, i + 1]
        t_to_depot   = tt[i + 1, 0]
        feasible_lo  = t_from_depot
        feasible_hi  = T - t_to_depot - svc[i]

        if feasible_hi <= feasible_lo:
            tw_open[i], tw_close[i] = 0.0, T
            continue

        center = float(rng.uniform(feasible_lo, feasible_hi))
        half_w = abs(float(rng.normal(0, tw_sigma)))

        tw_o = max(feasible_lo, center - half_w)
        tw_c = min(T, center + half_w)
        if tw_c - tw_o < 1.0:
            tw_c = min(T, tw_o + max(2 * half_w, svc_mean))

        tw_open[i]  = tw_o
        tw_close[i] = tw_c

    return _package(depot, coords, demands, svc, tw_open, tw_close, tt,
                    T, capacity, tw_sigma, 'U')


# ── RouteFinder-style generator ───────────────────────────────────────────────

def generate_rf_instance(
    n_customers: int = 100,
    T: float = 4.6,
    capacity: float = 50.0,
    seed: int = None,
    rng: np.random.Generator = None,
) -> dict:
    """
    RouteFinder-style CVRPTW instance.

    Reproduces the TW generation from ai4co/routefinder (TMLR 2025):
      - Depot and customer coords uniform in [0, 1]
      - service_time ~ U(0.15, 0.18)  [fraction of T, already normalized]
      - tw_length    ~ U(0.18, 0.20)  [fraction of T, already normalized]
      - tw_start: distance-based — TW centered around earliest feasible arrival
        from depot, with a random shift in [-tw_len/2, +tw_len/2]
      - T = 4.6, capacity = 50, demands ~ Uniform int [1, 9]
      - Hard TW masking expected at training time (--config RF sets hard_tw=True)

    Output dict has the same schema as load_solomon() / generate_*_instance():
      depot_xy (1,2), node_xy (N,2), depot/node demand/tw/service (scalars/1D),
      tt (N+1, N+1) — all normalized: coords in [0,1], TW/service/tt by /T.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    N = n_customers
    a, b, c = 0.15, 0.18, 0.20  # RouteFinder's service / TW-length parameters

    # Coordinates in [0, 1]
    depot_xy = rng.uniform(0.0, 1.0, size=(1, 2)).astype(np.float32)
    node_xy  = rng.uniform(0.0, 1.0, size=(N, 2)).astype(np.float32)

    # Demands: integer [1, 9] normalized by capacity
    demands = rng.integers(1, 10, size=N).astype(np.float32) / capacity

    # Service and TW length (already as fraction of T = normalized)
    service_norm = rng.uniform(a, b, size=N).astype(np.float32)
    tw_len_norm  = rng.uniform(b, c, size=N).astype(np.float32)

    # Distance from depot (in [0,1] space) normalized by T
    diff      = node_xy - depot_xy  # (N, 2)
    dist_norm = np.sqrt((diff ** 2).sum(-1)) / T  # (N,)

    # TW start: centered at earliest feasible arrival, random ±half-width shift
    half  = tw_len_norm / 2.0
    shift = rng.uniform(-half, half).astype(np.float32)
    tw_open_norm  = np.clip(dist_norm + shift, 0.0, 1.0 - tw_len_norm)
    tw_close_norm = np.clip(tw_open_norm + tw_len_norm, 0.0, 1.0)
    tw_open_norm  = np.clip(tw_close_norm - tw_len_norm, 0.0, 1.0)  # re-align if top-clipped

    # TT matrix (N+1 × N+1) normalized by T
    all_xy  = np.vstack([depot_xy[0], node_xy])          # (N+1, 2)
    diff2   = all_xy[:, None, :] - all_xy[None, :, :]   # (N+1, N+1, 2)
    tt_norm = np.sqrt((diff2 ** 2).sum(-1)).astype(np.float32) / T  # (N+1, N+1)

    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32)

    return dict(
        name             = 'rand_RF',
        n_customers      = N,
        vehicle_limit    = 25,
        vehicle_capacity = capacity,
        T                = T,
        max_coord        = 1.0,
        preset_events    = [],
        depot_xy         = _t(depot_xy),                          # (1, 2)
        node_xy          = _t(node_xy),                           # (N, 2)
        depot_demand     = _t(np.zeros(1, dtype=np.float32)),
        node_demand      = _t(demands),                           # (N,)
        depot_tw_open    = _t(np.zeros(1, dtype=np.float32)),
        depot_tw_close   = _t(np.ones(1,  dtype=np.float32)),
        depot_service    = _t(np.zeros(1, dtype=np.float32)),
        node_tw_open     = _t(tw_open_norm),                      # (N,)
        node_tw_close    = _t(tw_close_norm),                     # (N,)
        node_service     = _t(service_norm),                      # (N,)
        tt               = _t(tt_norm),                           # (N+1, N+1)
    )


# ── Dataset generation ────────────────────────────────────────────────────────

_GENERATORS = {
    'C':       generate_c_type_instance,
    'R':       generate_r_type_instance,
    'RC':      generate_rc_type_instance,
    'UNIFIED': generate_unified_instance,
    'RF':      generate_rf_instance,
}


def generate_dataset(
    gen_type:     str  = 'C',
    n_instances:  int  = 1000,
    seed:         int  = 42,
    **kwargs,
) -> list:
    """
    Generate n_instances random VRPTW instances of the given type.
    Per-type defaults (T, svc, tw_sigma range) are used unless overridden via kwargs.
    tw_sigma is sampled independently per instance.
    kwargs are forwarded directly to the per-type generator function.
    """
    gen_fn = _GENERATORS.get(gen_type.upper())
    if gen_fn is None:
        raise ValueError(f'Unknown gen_type "{gen_type}". Supported: {list(_GENERATORS)}')

    rng = np.random.default_rng(seed)
    return [gen_fn(rng=rng, **kwargs) for _ in range(n_instances)]


# ── Solomon-format conversion ─────────────────────────────────────────────────

def to_solomon_dict(inst: dict) -> dict:
    """Convert _package() normalized tensor dict → load_solomon()-compatible raw dict.

    The returned dict has the same keys/types as load_solomon():
      coords, demands, tw_open, tw_close, service_time  — Python lists of floats
      vehicle_limit, vehicle_capacity, n_customers       — int / float
    depot is index 0 in every list.
    """
    T   = float(inst['T'])
    cap = float(inst['vehicle_capacity'])
    mc  = float(inst.get('max_coord', 100.0))

    depot_xy = inst['depot_xy'].numpy() * mc          # (1, 2)
    node_xy  = inst['node_xy'].numpy()  * mc          # (n, 2)
    all_xy   = np.vstack([depot_xy, node_xy])         # (n+1, 2)

    demands_raw  = np.concatenate([[0.0], inst['node_demand'].numpy()  * cap])
    tw_open_raw  = np.concatenate([[0.0], inst['node_tw_open'].numpy() * T])
    tw_close_raw = np.concatenate([[T],   inst['node_tw_close'].numpy() * T])
    svc_raw      = np.concatenate([[0.0], inst['node_service'].numpy() * T])

    return dict(
        name             = inst.get('name', 'rand'),
        n_customers      = inst['n_customers'],
        vehicle_limit    = inst['vehicle_limit'],
        vehicle_capacity = cap,
        coords           = [(float(all_xy[i][0]), float(all_xy[i][1]))
                            for i in range(len(all_xy))],
        demands          = demands_raw.tolist(),
        tw_open          = tw_open_raw.tolist(),
        tw_close         = tw_close_raw.tolist(),
        service_time     = svc_raw.tolist(),
    )


def write_solomon_txt(raw: dict, path: str) -> None:
    """Write a single instance as a Solomon-format .txt file (same as c101.txt)."""
    n_cust = raw['n_customers']
    coords = raw['coords']
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{raw['name'].upper()}\n\n")
        f.write("VEHICLE\n")
        f.write("NUMBER     CAPACITY\n")
        f.write(f"  {raw['vehicle_limit']}         {int(raw['vehicle_capacity'])}\n\n")
        f.write("CUSTOMER\n")
        f.write("CUST NO.  XCOORD.   YCOORD.    DEMAND   "
                "READY TIME  DUE DATE   SERVICE   TIME\n \n")
        for i in range(n_cust + 1):
            f.write(
                f"  {i:4d}"
                f"  {round(coords[i][0]):7d}"
                f"  {round(coords[i][1]):7d}"
                f"  {round(raw['demands'][i]):9d}"
                f"  {round(raw['tw_open'][i]):11d}"
                f"  {round(raw['tw_close'][i]):10d}"
                f"  {round(raw['service_time'][i]):9d}\n"
            )


# ── LKH-3 evaluation helpers ─────────────────────────────────────────────────

def _to_lkh_inst(raw: dict) -> tuple:
    """Convert load_solomon()-compatible raw dict → (lkh_inst, tt_int) for LKH-3.

    Accepts either a to_solomon_dict() output or a load_solomon() output.
    Returns:
        lkh_inst  - the raw dict itself (already compatible)
        tt_int    - List[List[int]] integer travel-time matrix
    """
    coords = raw['coords']
    n = len(coords)
    tt_int = [
        [int(round(math.sqrt((coords[i][0] - coords[j][0])**2 +
                              (coords[i][1] - coords[j][1])**2)))
         for j in range(n)]
        for i in range(n)
    ]
    return raw, tt_int


def eval_with_lkh(
    dataset:    list,
    time_limit: int  = 30,
    runs:       int  = 1,
) -> list:
    """Run LKH-3 on each instance in dataset (Solomon raw dict format).

    dataset may contain either _package() normalized tensor dicts or
    to_solomon_dict() raw dicts — _to_lkh_inst() handles both.
    """
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _CLUSTER = os.path.join(os.path.dirname(_HERE), 'clustering_routing')
    if _CLUSTER not in sys.path:
        sys.path.insert(0, _CLUSTER)
    from R_lkh import solve_vrptw_lkh

    results = []
    for idx, inst in enumerate(dataset):
        raw, tt_int = _to_lkh_inst(inst)
        print(f'  [LKH-3] instance {idx+1}/{len(dataset)} '
              f'(n={raw["n_customers"]}) ...', end=' ', flush=True)
        r = solve_vrptw_lkh(raw, tt_int,
                             time_limit_sec=time_limit, runs=runs)
        print(f'status={r["status"]}  K={r["vehicles_used"]}  '
              f'dist={r["total_distance"]:.1f}  '
              f'late={r["served_late"]}  '
              f't={r["elapsed"]:.1f}s')
        r['instance_idx'] = idx
        r['instance_name'] = raw['name']
        results.append(r)
    return results


def print_lkh_summary(results: list, n_customers: int,
                       save_path: str = None) -> None:
    """Print (and optionally save) aggregate LKH-3 stats over all instances."""
    feasible = [r for r in results if r['status'] != 'no_solution']

    header = (f'{"#":<4} {"Name":<24} {"Status":<16} {"K":>4} '
              f'{"OnTime":>7} {"Late":>5} {"Dist":>9} {"T":>6}')
    sep    = '-' * len(header)

    lines = [
        f'LKH-3 summary  ({len(feasible)}/{len(results)} feasible)',
        header, sep,
    ]
    for r in results:
        lines.append(
            f'{r["instance_idx"]+1:<4} {r["instance_name"]:<24} '
            f'{r["status"]:<16} {r["vehicles_used"]:>4} '
            f'{r["served_on_time"]:>4}/{n_customers:<3} '
            f'{r["served_late"]:>5} '
            f'{r["total_distance"]:>9.1f} '
            f'{r["elapsed"]:>5.1f}s'
        )
    lines.append(sep)
    if feasible:
        avg_k    = sum(r['vehicles_used']   for r in feasible) / len(feasible)
        avg_dist = sum(r['total_distance']  for r in feasible) / len(feasible)
        avg_late = sum(r['served_late']     for r in feasible) / len(feasible)
        lines.append(f'Avg (feasible): K={avg_k:.1f}  dist={avg_dist:.1f}  '
                     f'late_stops={avg_late:.1f}')

    print(f'\n{"="*70}')
    for ln in lines:
        print(f'  {ln}')
    print()

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'[LKH-3] Results saved -> {save_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate random VRPTW training datasets (Solomon 1987 method).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Per-type defaults (override with --sigma-min/max, --svc-min/max as needed):
  C:  T=1236, svc=[70,110], sigma=[45,231]   (matches C1 class: c101→c109)
  R:  T=230,  svc=[9,11],   sigma=[6,74]     (matches R1 class: r101→r112)
  RC: T=240,  svc=[9,11],   sigma=[19,97]    (matches RC1 class: rc101→rc104)
        """
    )
    parser.add_argument('--type',           default='C', choices=list(_GENERATORS),
                        help='Instance type: C / R / RC (default: C)')
    parser.add_argument('--n',              type=int,   default=1000,
                        help='Number of instances (default: 1000)')
    parser.add_argument('--n-customers',    type=int,   default=100,
                        help='Number of customers per instance (default: 100)')
    parser.add_argument('--sigma',          type=float, default=None,
                        help='Fix tw_sigma for all instances (overrides --sigma-min/max)')
    parser.add_argument('--sigma-min',      type=float, default=None,
                        help='Override min tw_sigma (default: per-type)')
    parser.add_argument('--sigma-max',      type=float, default=None,
                        help='Override max tw_sigma (default: per-type)')
    parser.add_argument('--cluster-sigma',  type=float, default=8.0,
                        help='Gaussian spread around cluster centers, C/RC only (default: 8.0)')
    parser.add_argument('--clustered-frac', type=float, default=0.5,
                        help='Fraction of clustered customers for RC type (default: 0.5)')
    parser.add_argument('--svc-min',        type=float, default=None,
                        help='Override min service time (default: per-type)')
    parser.add_argument('--svc-max',        type=float, default=None,
                        help='Override max service time (default: per-type)')
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--out',            default=None,
                        help='Output .pt path (default: data/rand_{TYPE}_{N}.pt)')
    parser.add_argument('--lkh-eval',       action='store_true',
                        help='Run LKH-3 on generated instances after saving')
    parser.add_argument('--lkh-time-limit', type=int,   default=30,
                        help='LKH-3 time limit per instance in seconds (default: 30)')
    parser.add_argument('--lkh-runs',       type=int,   default=1,
                        help='LKH-3 number of independent runs (default: 1)')
    parser.add_argument('--save-txt',       action='store_true',
                        help='Also save each instance as a Solomon .txt file')
    parser.add_argument('--txt-dir',        default=None,
                        help='Directory to write Solomon .txt files '
                             '(default: <out>_txt/). Implies --save-txt.')
    args = parser.parse_args()

    extra = {}
    if args.sigma is not None:
        extra['tw_sigma_min'] = args.sigma
        extra['tw_sigma_max'] = args.sigma
    else:
        if args.sigma_min is not None:
            extra['tw_sigma_min'] = args.sigma_min
        if args.sigma_max is not None:
            extra['tw_sigma_max'] = args.sigma_max
    if args.svc_min is not None:
        extra['svc_min'] = args.svc_min
    if args.svc_max is not None:
        extra['svc_max'] = args.svc_max
    if args.type in ('C', 'RC'):
        extra['cluster_sigma'] = args.cluster_sigma
    if args.type == 'RC':
        extra['clustered_frac'] = args.clustered_frac

    if args.n_customers != 100:
        extra['n_customers'] = args.n_customers

    print(f'[generate_vrptw] type={args.type}  n={args.n}  n_customers={args.n_customers}  '
          f'seed={args.seed}  overrides={extra}')

    dataset = generate_dataset(
        gen_type    = args.type,
        n_instances = args.n,
        seed        = args.seed,
        **extra,
    )

    if args.out is None:
        os.makedirs('data', exist_ok=True)
        out_path = f'data/rand_{args.type}_{args.n}.pt'
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        out_path = args.out

    # Save as tensor dicts (compatible with make_batch in train_vrptw.py)
    torch.save(dataset, out_path)
    print(f'[generate_vrptw] Saved {len(dataset)} instances (tensor format) -> {out_path}')

    inst0 = dataset[0]
    tw_w  = (inst0['node_tw_close'] - inst0['node_tw_open']) * inst0['T']
    mean_d = float(_DEMAND_MEAN if args.type == 'C' else _RC_DEMAND_MEAN)
    n_clust = _n_clusters_from_capacity(inst0['n_customers'], mean_d, float(inst0['vehicle_capacity']))
    T0 = float(inst0['T'])
    print(f'  Sample[0]: {inst0["name"]}  T={T0:.0f}  n_clusters={n_clust}  '
          f'avg_tw_width={tw_w.mean():.1f}  '
          f'min={tw_w.min():.1f}  max={tw_w.max():.1f}')

    # Optionally write individual Solomon .txt files
    if args.save_txt or args.txt_dir:
        txt_dir = args.txt_dir if args.txt_dir else out_path.replace('.pt', '_txt')
        os.makedirs(txt_dir, exist_ok=True)
        type_tag = args.type.lower()   # c / r / rc
        for i, raw in enumerate(raw_dataset):
            inst_name = f'_{type_tag}{i+1:03d}'
            raw = dict(raw, name=inst_name)   # override name field
            write_solomon_txt(raw, os.path.join(txt_dir, f'{inst_name}.txt'))
        print(f'[generate_vrptw] Solomon .txt files -> {txt_dir}/')

    if args.lkh_eval:
        lkh_txt = out_path.replace('.pt', '_lkh.txt')
        print(f'\n[LKH-3] Evaluating {len(raw_dataset)} instances '
              f'(time_limit={args.lkh_time_limit}s, runs={args.lkh_runs}) ...')
        lkh_results = eval_with_lkh(
            raw_dataset,
            time_limit = args.lkh_time_limit,
            runs       = args.lkh_runs,
        )
        print_lkh_summary(lkh_results, raw0['n_customers'], save_path=lkh_txt)
