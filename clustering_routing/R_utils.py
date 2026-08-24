"""
Shared routing utilities: config, zone building, feasible actions, ontology pruning.

Imported by main.py, llm_baseline.py, rl_baseline.py.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from R_env import WasteFleetEnv, WasteInstance


# ============================================================
# 1) Config
# ============================================================

@dataclass
class TrainConfig:
    data_path: str = r"C:\Users\your_path\C101.txt"
    seed: int = 42

    num_episodes: int = 300
    gamma: float = 0.99
    lr: float = 2e-4
    max_steps_per_episode: int = 10_000

    estimated_steps_per_episode: int = 80

    # ── POMO hyperparameters ─────────────────────────────────────────────────
    pomo_k: int = 4

    # ── PPO hyperparameters ──────────────────────────────────────────────────
    ppo_epochs: int = 4
    ppo_batch_size: int = 64
    rollout_steps: int = 512
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    entropy_coef_min: float = 0.0
    gae_lambda: float = 0.95
    lr_min: float = 1e-5

    # ── Attention network (matching C+R model_params) ────────────────────────
    embed_dim: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 6

    # Episode-level reward (normalized by T and N)
    late_count_penalty: float = 20.0
    late_penalty: float = 150.0
    unserved_penalty: float = 500.0

    use_solution_zones: bool = False
    solution_path: Optional[str] = None

    use_fixed_solution_policy: bool = False
    fixed_policy_print_trace: bool = False

    cluster_random_restarts: int = 8
    kmeans_iters: int = 30
    cluster_repair_rounds: int = 10

    hull_adj_distance_threshold: float = 15.0
    prefer_on_time_candidates: bool = True

    # ontology = hard filter only
    use_ontology_pruning: bool = True
    ontology_filter_on_capacity: bool = False
    ontology_filter_on_return_feasible: bool = True
    ontology_filter_late_actions: bool = False
    ontology_max_allowed_tardiness: float = 30.0
    ontology_keep_depot_action: bool = True

    # ontology reasoning (dynamic reward)
    use_ontology_reasoning: bool = True
    ontology_penalty_threshold: float = 0.5
    ontology_penalty_scale_factor: float = 1.5
    ontology_penalty_max_multiplier: float = 3.0

    # HighUnservedStop chunk-based classification
    ontology_unserved_window_ratio: float = 0.1
    ontology_min_warmup_ratio: float = 0.5
    llm_recall_on_chunk: bool = True
    llm_use_cot: bool = True
    llm_temperature: float = 0.0

    # OWL classification thresholds
    ontology_urgent_slack_threshold: float = 30.0
    ontology_critical_zone_threshold: float = 0.5

    llm_model: str = "qwen3:32b"
    llm_confidence_lambda: float = 1.0
    # If True, lambda_t (LLM-confidence bias strength) is learned per-step via
    # the trust_gate network. If False, lambda_t is fixed at llm_confidence_lambda
    # — useful for isolating the effect of the confidence bias itself.
    use_learned_trust_gate: bool = True
    # Fraction of stops per cluster the LLM selects+scores (boost-only, (0,1]).
    # 0.0 = no stops scored (skip LLM call, all confidence=0).
    # 1.0 = all stops scored (legacy full-scoring prompt, 0.0-1.0 range).
    llm_confidence_topk_ratio: float = 1.0
    use_llm_event_response: bool = True
    use_llm_zone_confidence: bool = False

    # checkpoint
    checkpoint_path: Optional[str] = None
    checkpoint_interval: int = 50
    resume_from: Optional[str] = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2) Utils
# ============================================================

# Shared hyperparameters used by rl_baseline.py and main.py.
# Change once here → both scripts pick it up automatically.
SHARED_HYPERPARAMS: dict = dict(
    # PPO
    gamma=0.99,
    lr_min=1e-5,
    ppo_epochs=6,
    ppo_batch_size=64,
    rollout_steps=512,
    clip_eps=0.2,
    value_coef=1.0,
    entropy_coef=0.0,
    entropy_coef_min=0.0,
    gae_lambda=0.95,
    # Attention network (matching C+R model_params)
    embed_dim=128,
    n_heads=8,
    n_encoder_layers=6,
    # Reward shaping (episode-level, normalized by T and N)
    late_count_penalty=20.0,
    late_penalty=150.0,
    unserved_penalty=500.0,
    # POMO
    pomo_k=4,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 5) Zone plan
# ============================================================

@dataclass
class ZonePlan:
    zones: List[List[int]]
    customer_to_zone: Dict[int, int]
    zone_centroids: np.ndarray
    zone_demands: List[float]
    zone_capacity: float
    zone_hulls: List[np.ndarray]
    adjacent_zones: Dict[int, Set[int]]

    @property
    def n_zones(self) -> int:
        return len(self.zones)


def estimate_vehicle_count(inst: WasteInstance) -> int:
    total_demand = float(np.sum(inst.demands[1:]))
    q = float(inst.vehicle_capacity)
    n = int(math.ceil(total_demand / max(q, 1e-8)))
    n = max(1, n)
    n = min(n, max(inst.vehicle_limit, 1))
    return n


# ============================================================
# 6) Convex hull helpers
# ============================================================

def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points.copy()

    pts = np.unique(points.astype(np.float64), axis=0)
    if len(pts) <= 2:
        return pts.copy()

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull


def point_in_convex_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    if len(poly) == 0:
        return False
    if len(poly) == 1:
        return np.linalg.norm(point - poly[0]) <= 1e-8
    if len(poly) == 2:
        a, b = poly[0], poly[1]
        ab = b - a
        ap = point - a
        cross = abs(ab[0] * ap[1] - ab[1] * ap[0])
        if cross > 1e-8:
            return False
        dot = np.dot(ap, ab)
        if dot < -1e-8:
            return False
        if dot - np.dot(ab, ab) > 1e-8:
            return False
        return True

    sign = None
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        c = _cross(a, b, point)
        if abs(c) <= 1e-8:
            continue
        cur = c > 0
        if sign is None:
            sign = cur
        elif sign != cur:
            return False
    return True


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p, q, r):
        return _cross(p, q, r)

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - 1e-8 <= q[0] <= max(p[0], r[0]) + 1e-8
            and min(p[1], r[1]) - 1e-8 <= q[1] <= max(p[1], r[1]) + 1e-8
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True

    if abs(o1) <= 1e-8 and on_segment(a, c, b):
        return True
    if abs(o2) <= 1e-8 and on_segment(a, d, b):
        return True
    if abs(o3) <= 1e-8 and on_segment(c, a, d):
        return True
    if abs(o4) <= 1e-8 and on_segment(c, b, d):
        return True

    return False


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


def hulls_overlap_or_touch(h1: np.ndarray, h2: np.ndarray) -> bool:
    if len(h1) == 0 or len(h2) == 0:
        return False

    if len(h1) >= 2 and len(h2) >= 2:
        for i in range(len(h1)):
            a = h1[i]
            b = h1[(i + 1) % len(h1)]
            for j in range(len(h2)):
                c = h2[j]
                d = h2[(j + 1) % len(h2)]
                if segments_intersect(a, b, c, d):
                    return True

    if point_in_convex_polygon(h1[0], h2):
        return True
    if point_in_convex_polygon(h2[0], h1):
        return True

    return False


def hull_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    if len(h1) == 0 or len(h2) == 0:
        return float("inf")
    if hulls_overlap_or_touch(h1, h2):
        return 0.0

    if len(h1) == 1 and len(h2) == 1:
        return float(np.linalg.norm(h1[0] - h2[0]))
    if len(h1) == 1:
        best = float("inf")
        p = h1[0]
        for j in range(len(h2)):
            c = h2[j]
            d = h2[(j + 1) % len(h2)] if len(h2) > 1 else h2[j]
            best = min(best, point_segment_distance(p, c, d))
        return best
    if len(h2) == 1:
        best = float("inf")
        p = h2[0]
        for i in range(len(h1)):
            a = h1[i]
            b = h1[(i + 1) % len(h1)] if len(h1) > 1 else h1[i]
            best = min(best, point_segment_distance(p, a, b))
        return best

    best = float("inf")
    for i in range(len(h1)):
        a = h1[i]
        b = h1[(i + 1) % len(h1)]
        for j in range(len(h2)):
            c = h2[j]
            d = h2[(j + 1) % len(h2)]
            best = min(best, segment_distance(a, b, c, d))
    return best


def build_zone_hulls(inst: WasteInstance, zones: List[List[int]]) -> List[np.ndarray]:
    hulls: List[np.ndarray] = []
    for z in zones:
        pts = inst.coords[z]
        hulls.append(convex_hull(pts))
    return hulls


def build_hull_adjacency(
    hulls: List[np.ndarray],
    threshold: float,
) -> Dict[int, Set[int]]:
    n = len(hulls)
    adj: Dict[int, Set[int]] = {i: {i} for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            d = hull_distance(hulls[i], hulls[j])
            if d <= threshold + 1e-8:
                adj[i].add(j)
                adj[j].add(i)

    return adj


# ============================================================
# 7) Solution file based zones
# ============================================================

def infer_solution_path(data_path: str, solution_path: Optional[str]) -> Optional[Path]:
    if solution_path is not None:
        p = Path(solution_path)
        if p.exists():
            return p
        return None

    data_p = Path(data_path)
    stem = data_p.stem
    parent = data_p.parent

    candidates = [
        parent / f"{stem}_sol",
        parent / f"{stem}_sol.txt",
        parent / f"{stem.upper()}_sol",
        parent / f"{stem.upper()}_sol.txt",
        parent / f"{stem.lower()}_sol",
        parent / f"{stem.lower()}_sol.txt",
        parent / f"{stem.upper()}_SOL",
        parent / f"{stem.upper()}_SOL.txt",
        parent / f"{stem.lower()}_SOL",
        parent / f"{stem.lower()}_SOL.txt",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def parse_solution_routes(solution_path: Path) -> List[List[int]]:
    if not solution_path.exists():
        raise FileNotFoundError(f"Solution file not found: {solution_path}")

    routes: List[List[int]] = []
    with open(solution_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = re.match(r"^Route\s+\d+\s*:\s*(.*)$", line, flags=re.IGNORECASE)
            if m:
                rhs = m.group(1).strip()
                if rhs == "":
                    routes.append([])
                else:
                    route = [int(x) for x in rhs.split()]
                    routes.append(route)

    if not routes:
        raise ValueError(f"No route lines parsed from solution file: {solution_path}")

    return routes


def validate_solution_routes(inst: WasteInstance, routes: List[List[int]]) -> None:
    seen = []
    for r in routes:
        seen.extend(r)

    seen_sorted = sorted(seen)
    expected = list(range(1, inst.n_customers + 1))

    if seen_sorted != expected:
        missing = sorted(set(expected) - set(seen_sorted))
        dup = sorted([x for x in set(seen_sorted) if seen_sorted.count(x) > 1])
        raise ValueError(
            f"Solution routes do not match customer set. missing={missing}, duplicated={dup}"
        )

    for r in routes:
        demand = float(np.sum(inst.demands[r])) if len(r) > 0 else 0.0
        if demand > inst.vehicle_capacity + 1e-8:
            raise ValueError(
                f"Route demand exceeds capacity: demand={demand}, cap={inst.vehicle_capacity}, route={r}"
            )


def build_zone_plan_from_routes(
    inst: WasteInstance,
    routes: List[List[int]],
    cfg: TrainConfig,
) -> ZonePlan:
    zones = [list(r) for r in routes if len(r) > 0]

    customer_to_zone: Dict[int, int] = {}
    centroids = np.zeros((len(zones), 2), dtype=np.float32)
    demands = []

    for k, z in enumerate(zones):
        for c in z:
            customer_to_zone[c] = k
        centroids[k] = np.mean(inst.coords[z], axis=0)
        demands.append(float(np.sum(inst.demands[z])))

    hulls = build_zone_hulls(inst, zones)
    adjacent_zones = build_hull_adjacency(
        hulls=hulls,
        threshold=float(cfg.hull_adj_distance_threshold),
    )

    return ZonePlan(
        zones=zones,
        customer_to_zone=customer_to_zone,
        zone_centroids=centroids,
        zone_demands=demands,
        zone_capacity=float(inst.vehicle_capacity),
        zone_hulls=hulls,
        adjacent_zones=adjacent_zones,
    )


# ============================================================
# 8) Kim2006 clustering
# ============================================================

def finalize_zone_plan(inst: WasteInstance, zones: List[List[int]], cfg: TrainConfig) -> ZonePlan:
    non_empty = [list(z) for z in zones if len(z) > 0]

    customer_to_zone: Dict[int, int] = {}
    centroids = np.zeros((len(non_empty), 2), dtype=np.float32)
    demands = []

    for k, z in enumerate(non_empty):
        for c in z:
            customer_to_zone[c] = k
        centroids[k] = np.mean(inst.coords[z], axis=0)
        demands.append(float(np.sum(inst.demands[z])))

    hulls = build_zone_hulls(inst, non_empty)
    adjacent_zones = build_hull_adjacency(
        hulls=hulls,
        threshold=float(cfg.hull_adj_distance_threshold),
    )

    return ZonePlan(
        zones=non_empty,
        customer_to_zone=customer_to_zone,
        zone_centroids=centroids,
        zone_demands=demands,
        zone_capacity=float(inst.vehicle_capacity),
        zone_hulls=hulls,
        adjacent_zones=adjacent_zones,
    )


def _kim2006_clusters(inst: WasteInstance, tt: np.ndarray) -> List[List[int]]:
    import random as _random
    import math as _math

    coords = inst.coords
    Q = float(inst.vehicle_capacity)
    n = inst.n_customers
    custs = list(range(1, n + 1))
    depot_close = float(inst.tw_close[0])

    def _d(i: int, j: int) -> float:
        return float(tt[i, j])

    def _cdist(cx: float, cy: float, c: int) -> float:
        return _math.sqrt((float(coords[c][0]) - cx) ** 2 + (float(coords[c][1]) - cy) ** 2)

    def _tw_feasible(cluster: List[int]) -> bool:
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

    def _can_add(c: int, cluster: List[int]) -> bool:
        if sum(float(inst.demands[x]) for x in cluster) + float(inst.demands[c]) > Q + 1e-8:
            return False
        return _tw_feasible(cluster + [c])

    total_demand = sum(float(inst.demands[c]) for c in custs)
    N = max(1, int(total_demand // Q) + 1)
    clusters: List[List[int]] = []

    for _inc in range(30):
        _random.seed(42 + _inc)
        seeds = _random.sample(custs, min(N, len(custs)))
        centroid_pos = [[float(coords[s][0]), float(coords[s][1])] for s in seeds]
        clusters = [[] for _ in range(N)]
        prev_assign = None

        for _ in range(50):
            gc_x = sum(cp[0] for cp in centroid_pos) / N
            gc_y = sum(cp[1] for cp in centroid_pos) / N

            sorted_custs = sorted(
                custs,
                key=lambda c: _math.sqrt((float(coords[c][0]) - gc_x) ** 2 +
                                         (float(coords[c][1]) - gc_y) ** 2),
                reverse=True,
            )
            new_clusters: List[List[int]] = [[] for _ in range(N)]
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
                [float(np.mean([coords[c][0] for c in cl])),
                 float(np.mean([coords[c][1] for c in cl]))]
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
                        cp[i] = ([float(np.mean([coords[x][0] for x in cl])),
                                   float(np.mean([coords[x][1] for x in cl]))]
                                  if cl else cp[i])
                        cp[j] = [float(np.mean([coords[x][0] for x in cj])),
                                  float(np.mean([coords[x][1] for x in cj]))]
                        moved = True
                        break
            if not moved:
                break

        raw = [cl for cl in clusters if cl]
        if all(_tw_feasible(cl) for cl in raw):
            break
        N += 1

    return [cl for cl in clusters if cl]


def build_initial_zones(inst: WasteInstance, cfg: TrainConfig, tt: np.ndarray) -> ZonePlan:
    clusters = _kim2006_clusters(inst, tt)
    print(f"  [Kim 2006] generated {len(clusters)} clusters "
          f"covering {sum(len(c) for c in clusters)}/{inst.n_customers} customers")
    return finalize_zone_plan(inst, clusters, cfg)


# ============================================================
# 9) Feasible actions + zone control
# ============================================================

def get_active_zone_index_from_vehicle(env: WasteFleetEnv, zone_plan: ZonePlan) -> int:
    idx = int(env.vehicles[env.active_vehicle_idx].active_zone_idx)
    idx = max(0, min(idx, zone_plan.n_zones - 1))
    return idx


def get_global_feasible_customers(env: WasteFleetEnv) -> List[int]:
    feasible = []
    for cust in range(1, env.N + 1):
        if env._can_serve(cust) and env.can_return_to_depot_if_visit(cust):
            feasible.append(cust)
    return feasible


def split_on_time_and_late_customers(
    env: WasteFleetEnv,
    customers: List[int],
) -> Tuple[List[int], List[int]]:
    on_time = []
    late = []
    for c in customers:
        if env.is_on_time_if_visit(c):
            on_time.append(c)
        else:
            late.append(c)
    return on_time, late


def get_feasible_actions_with_zone_control(
    env: WasteFleetEnv,
    zone_plan: Optional[ZonePlan],
    cfg: TrainConfig,
) -> Tuple[List[int], Optional[int], bool]:
    if zone_plan is None:
        feasible = get_global_feasible_customers(env)
        if cfg.prefer_on_time_candidates:
            on_time, _ = split_on_time_and_late_customers(env, feasible)
            feasible = on_time  # hard constraint: no fallback to late

        if len(feasible) == 0:
            return [0], None, env._all_served()
        return feasible, None, env._all_served()

    if env._all_served():
        active_zone_idx = get_active_zone_index_from_vehicle(env, zone_plan)
        return [0], active_zone_idx, True

    active_zone_idx = get_active_zone_index_from_vehicle(env, zone_plan)
    active_zone = set(zone_plan.zones[active_zone_idx])

    feasible_actions = [
        c for c in active_zone
        if env._can_serve(c) and env.can_return_to_depot_if_visit(c)
    ]

    if len(feasible_actions) == 0:
        return [0], active_zone_idx, False

    if cfg.prefer_on_time_candidates:
        on_time, _ = split_on_time_and_late_customers(env, feasible_actions)
        feasible_actions = on_time  # hard constraint: no fallback to late

    if len(feasible_actions) == 0:
        return [0], active_zone_idx, False

    return feasible_actions, active_zone_idx, False


# ============================================================
# 10) Ontology candidate pruning
# ============================================================

@dataclass
class TimeWindowConstraint:
    ready_time: float
    due_time: float
    service_time: float
    travel_time: float
    arrival_time: float
    service_start_time: float
    depart_time: float
    wait_time: float
    slack: float
    tardiness: float
    satisfied: bool
    violation_reason: Optional[str]
    urgency: str


@dataclass
class CapacityConstraint:
    demand: float
    remaining_capacity: float
    satisfied: bool
    violation_reason: Optional[str]
    load_ratio: float


@dataclass
class DepotReturnConstraint:
    depart_time: float
    travel_to_depot: float
    depot_due: float
    return_arrival: float
    return_slack: float
    satisfied: bool
    violation_reason: Optional[str]
    risk_level: str


@dataclass
class ZoneContext:
    zone_index: int
    same_zone: bool
    distance_level: str


@dataclass
class ActionOntologyFact:
    action: int

    time_window: TimeWindowConstraint
    capacity: CapacityConstraint
    depot_return: DepotReturnConstraint
    zone: ZoneContext

    @property
    def is_hard_feasible(self) -> bool:
        return self.capacity.satisfied and self.depot_return.satisfied

    @property
    def violated_constraints(self) -> List[str]:
        violated = []
        if not self.capacity.satisfied:
            violated.append("CapacityConstraint")
        if not self.depot_return.satisfied:
            violated.append("DepotReturnConstraint")
        return violated

    def to_llm_dict(self) -> dict:
        return {
            "action": self.action,
            "violated_constraints": self.violated_constraints,
            "is_hard_feasible": self.is_hard_feasible,
            "ZoneContext": {
                "zone_index": self.zone.zone_index,
                "same_zone": self.zone.same_zone,
                "adjacent_zone": self.zone.adjacent_zone,
                "zone_penalty": self.zone.zone_penalty,
                "distance_level": self.zone.distance_level,
            },
            "TimeWindowConstraint": {
                "ready_time": self.time_window.ready_time,
                "due_time": self.time_window.due_time,
                "travel_time": self.time_window.travel_time,
                "arrival_time": self.time_window.arrival_time,
                "service_start_time": self.time_window.service_start_time,
                "wait_time": self.time_window.wait_time,
                "slack": self.time_window.slack,
                "tardiness": self.time_window.tardiness,
                "satisfied": self.time_window.satisfied,
                "urgency": self.time_window.urgency,
                "violation_reason": self.time_window.violation_reason,
            },
            "CapacityConstraint": {
                "demand": self.capacity.demand,
                "remaining_capacity": self.capacity.remaining_capacity,
                "load_ratio": round(self.capacity.load_ratio, 3),
                "satisfied": self.capacity.satisfied,
                "violation_reason": self.capacity.violation_reason,
            },
            "DepotReturnConstraint": {
                "depart_time": self.depot_return.depart_time,
                "travel_to_depot": self.depot_return.travel_to_depot,
                "return_arrival": self.depot_return.return_arrival,
                "depot_due": self.depot_return.depot_due,
                "return_slack": self.depot_return.return_slack,
                "satisfied": self.depot_return.satisfied,
                "risk_level": self.depot_return.risk_level,
                "violation_reason": self.depot_return.violation_reason,
            },
        }


CandidateFact = ActionOntologyFact


def build_candidate_facts(
    env: WasteFleetEnv,
    feasible: List[int],
    zone_plan: Optional[ZonePlan],
    active_zone_idx: Optional[int],
) -> List[ActionOntologyFact]:
    facts: List[ActionOntologyFact] = []

    for a in feasible:
        if a == 0:
            continue

        travel, arrival, service_start, depart = env.get_visit_timing(a)
        travel_to_depot, return_arrival, return_slack, depot_due = env.get_depot_return_info(a)

        ready = float(env.inst.tw_open[a])
        due = float(env.inst.tw_close[a])
        service_time_val = float(env.inst.service_time[a])
        demand = float(env.inst.demands[a])
        remaining_cap = float(env.remaining_cap)

        wait_time = max(0.0, ready - arrival)
        slack = due - service_start
        tardiness = max(0.0, service_start - due)

        tw_satisfied = tardiness <= 1e-8
        tw_violation = (
            None if tw_satisfied
            else f"service_start({service_start:.1f}) > due_time({due:.1f}) by {tardiness:.1f}"
        )
        if slack <= 30:
            urgency = "high"
        elif slack <= 120:
            urgency = "medium"
        else:
            urgency = "low"

        tw = TimeWindowConstraint(
            ready_time=ready,
            due_time=due,
            service_time=service_time_val,
            travel_time=float(travel),
            arrival_time=float(arrival),
            service_start_time=float(service_start),
            depart_time=float(depart),
            wait_time=float(wait_time),
            slack=float(slack),
            tardiness=float(tardiness),
            satisfied=tw_satisfied,
            violation_reason=tw_violation,
            urgency=urgency,
        )

        cap_satisfied = demand <= remaining_cap + 1e-8
        cap_violation = (
            None if cap_satisfied
            else f"demand({demand:.1f}) > remaining_capacity({remaining_cap:.1f})"
        )
        load_ratio = demand / max(remaining_cap, 1e-8)

        cap = CapacityConstraint(
            demand=demand,
            remaining_capacity=remaining_cap,
            satisfied=cap_satisfied,
            violation_reason=cap_violation,
            load_ratio=float(load_ratio),
        )

        dr_satisfied = return_slack >= -1e-8
        dr_violation = (
            None if dr_satisfied
            else (
                f"return_arrival({return_arrival:.1f}) > depot_due({depot_due:.1f})"
                f" by {-return_slack:.1f}"
            )
        )
        if return_slack <= 60:
            risk_level = "high"
        elif return_slack <= 180:
            risk_level = "medium"
        else:
            risk_level = "low"

        dr = DepotReturnConstraint(
            depart_time=float(depart),
            travel_to_depot=float(travel_to_depot),
            depot_due=float(depot_due),
            return_arrival=float(return_arrival),
            return_slack=float(return_slack),
            satisfied=dr_satisfied,
            violation_reason=dr_violation,
            risk_level=risk_level,
        )

        cand_zone = zone_plan.customer_to_zone[a] if zone_plan is not None else -1
        same_zone = (
            zone_plan is not None
            and active_zone_idx is not None
            and cand_zone == active_zone_idx
        )

        if travel <= 10:
            dist_level = "near"
        elif travel <= 30:
            dist_level = "medium"
        else:
            dist_level = "far"

        zone_ctx = ZoneContext(
            zone_index=int(cand_zone),
            same_zone=bool(same_zone),
            distance_level=dist_level,
        )

        facts.append(ActionOntologyFact(
            action=int(a),
            time_window=tw,
            capacity=cap,
            depot_return=dr,
            zone=zone_ctx,
        ))

    return facts


def prune_candidates_by_ontology(
    env: WasteFleetEnv,
    feasible: List[int],
    zone_plan: Optional[ZonePlan],
    active_zone_idx: Optional[int],
    cfg: TrainConfig,
) -> Tuple[List[int], List[ActionOntologyFact]]:
    if not cfg.use_ontology_pruning:
        return feasible, []

    facts = build_candidate_facts(env, feasible, zone_plan, active_zone_idx)

    if len(facts) == 0:
        return feasible, facts

    kept_actions: List[int] = []

    for f in facts:
        reject = False

        if cfg.ontology_filter_on_capacity and not f.capacity.satisfied:
            reject = True

        if cfg.ontology_filter_on_return_feasible and not f.depot_return.satisfied:
            reject = True

        if cfg.ontology_filter_late_actions and f.time_window.tardiness > cfg.ontology_max_allowed_tardiness + 1e-8:
            reject = True

        if not reject:
            kept_actions.append(f.action)

    if len(kept_actions) == 0:
        relaxed = [f.action for f in facts if not (cfg.ontology_filter_on_capacity and not f.capacity.satisfied)]
        kept_actions = relaxed

    if len(kept_actions) == 0 and cfg.ontology_keep_depot_action and 0 in feasible:
        kept_actions = [0]

    if len(kept_actions) == 0:
        kept_actions = feasible[:]

    return kept_actions, facts


# ============================================================
# 11) Heuristic fallback (shared with LLM module)
# ============================================================

def heuristic_topk_fallback(
    env: WasteFleetEnv,
    feasible: List[int],
    top_k: int,
) -> List[int]:
    if len(feasible) <= top_k:
        return feasible

    scored = []
    for a in feasible:
        if a == 0:
            score = 1e9
        else:
            travel, arrival, service_start, depart = env.get_visit_timing(a)
            due = float(env.inst.tw_close[a])
            wait = max(0.0, float(env.inst.tw_open[a]) - arrival)
            tardiness = max(0.0, service_start - due)
            score = travel + 0.04 * wait + 3.0 * tardiness + 0.01 * due
        scored.append((score, a))

    scored.sort(key=lambda x: x[0])
    return [a for _, a in scored[:top_k]]


# ============================================================
# 12) Fixed solution-order policy
# ============================================================

def select_fixed_solution_action(
    env: WasteFleetEnv,
    zone_plan: Optional[ZonePlan],
) -> int:
    if zone_plan is None:
        raise ValueError("Fixed solution policy requires solution-based zone_plan.")

    route_idx = int(env.active_zone_idx)

    if route_idx >= zone_plan.n_zones:
        return 0

    route = zone_plan.zones[route_idx]

    for cust in route:
        if env.served[cust] == 0:
            return cust

    return 0
