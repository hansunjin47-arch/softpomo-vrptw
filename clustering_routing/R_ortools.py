"""
OR-Tools VRPTW solver for Solomon benchmark instances.

Usage:
    python ortools_baseline.py --data data/Solomon/c101.txt
    python ortools_baseline.py --data data/Solomon/c101.txt --time_limit 60
    python ortools_baseline.py --data data/Solomon/c101.txt --zone_file data/Solomon/c101_sol.txt
"""
from __future__ import annotations

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# ============================================================
# Minimal Solomon parser (no gymnasium dependency)
# ============================================================

def load_solomon(path: str):
    """Parse Solomon benchmark txt. Returns dict of arrays."""
    coords, demands, tw_open, tw_close, service_time = [], [], [], [], []
    vehicle_limit = vehicle_capacity = None
    name = ""

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    name = lines[0]
    i = 0
    while i < len(lines):
        upper = lines[i].upper()
        if upper == "VEHICLE":
            vals = lines[i + 2].split()
            vehicle_limit = int(vals[0])
            vehicle_capacity = float(vals[1])
            i += 3
        elif upper.startswith("CUST"):
            i += 2  # skip header
        else:
            parts = lines[i].split()
            if len(parts) >= 7 and parts[0].isdigit():
                coords.append((float(parts[1]), float(parts[2])))
                demands.append(float(parts[3]))
                tw_open.append(float(parts[4]))
                tw_close.append(float(parts[5]))
                service_time.append(float(parts[6]))
            i += 1

    return {
        "name": name,
        "coords": coords,
        "demands": demands,
        "tw_open": tw_open,
        "tw_close": tw_close,
        "service_time": service_time,
        "vehicle_limit": vehicle_limit,
        "vehicle_capacity": vehicle_capacity,
        "n_customers": len(coords) - 1,
    }


# ============================================================
# Travel time matrix
# ============================================================

def build_tt_matrix(inst: dict, speed: float = 1.0) -> List[List[int]]:
    """Euclidean distance → integer travel time (OR-Tools requires integers)."""
    coords = inst["coords"]
    n = len(coords)
    tt = []
    for i in range(n):
        row = []
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist = math.sqrt(dx * dx + dy * dy) / speed
            row.append(int(round(dist)))
        tt.append(row)
    return tt


def build_rain_tt_matrix(
    inst: dict,
    rain_events: List[Tuple[float, float, float, List[int]]],
) -> List[List[int]]:
    """Rain-aware TT matrix: edges where both endpoints are in rain zone get multiplied.

    OR-Tools is a static solver, so we apply the max multiplier across all events
    that cover each edge. Re-evaluation uses time-dependent simulation for accuracy.
    """
    coords = inst["coords"]
    n = len(coords)
    tt = []
    for i in range(n):
        row = []
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            base = math.sqrt(dx * dx + dy * dy)
            mult = 1.0
            for _, _, m, nodes in rain_events:
                node_set = set(nodes)
                if i in node_set and j in node_set:
                    mult = max(mult, m)
            row.append(int(round(base * mult)))
        tt.append(row)
    return tt


# ============================================================
# Zone plan (optional: same zones as RL)
# ============================================================

def parse_solution_zones(sol_path: str) -> Optional[List[List[int]]]:
    """Parse Solomon solution file → list of routes (zones)."""
    try:
        routes = []
        with open(sol_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Route"):
                    parts = line.split(":")
                    if len(parts) == 2:
                        stops = [int(x) for x in parts[1].split()]
                        if stops:
                            routes.append(stops)
        return routes if routes else None
    except Exception:
        return None


# ============================================================
# OR-Tools VRPTW solver
# ============================================================

def solve_vrptw(
    inst: dict,
    tt: List[List[int]],
    time_limit_sec: int = 30,
    zone_routes: Optional[List[List[int]]] = None,
    n_vehicles_max: Optional[int] = None,
) -> Dict:
    """
    Solve VRPTW using OR-Tools.

    If zone_routes is provided, vehicles are pre-assigned to zones
    (same cluster-first setting as RL model).
    Otherwise, OR-Tools solves globally without zone restriction.

    n_vehicles_max: override vehicle count cap (e.g., to match unknown scenario).

    Returns result dict with routes, total_distance, total_late, served.
    """
    n_nodes = len(inst["coords"])
    n_vehicles = n_vehicles_max if n_vehicles_max is not None else inst["vehicle_limit"]
    depot = 0

    # -- Create routing model ----------------------------------
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # -- Travel time callback (arc cost: travel only) ---------
    def travel_callback(from_idx, to_idx):
        return tt[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    travel_cb_idx = routing.RegisterTransitCallback(travel_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(travel_cb_idx)

    # -- Time callback (travel + service at from-node) ---------
    def time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return tt[from_node][to_node] + int(inst["service_time"][from_node])

    time_cb_idx = routing.RegisterTransitCallback(time_callback)

    # -- Time window dimension ---------------------------------
    depot_due = int(inst["tw_close"][0])
    routing.AddDimension(
        time_cb_idx,
        slack_max=depot_due,
        capacity=depot_due,
        fix_start_cumul_to_zero=True,
        name="Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Set time windows for each node
    for node in range(n_nodes):
        idx = manager.NodeToIndex(node)
        tw_start = int(inst["tw_open"][node])
        tw_end = int(inst["tw_close"][node])
        time_dim.CumulVar(idx).SetRange(tw_start, tw_end)

    # -- Capacity dimension ------------------------------------
    def demand_callback(from_idx):
        return int(inst["demands"][manager.IndexToNode(from_idx)])

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        slack_max=0,
        vehicle_capacities=[int(inst["vehicle_capacity"])] * n_vehicles,
        fix_start_cumul_to_zero=True,
        name="Capacity",
    )

    # -- Zone restriction (cluster-first) ---------------------
    if zone_routes is not None:
        # Build customer→vehicle mapping from zone routes
        customer_to_vehicle: Dict[int, int] = {}
        for veh_idx, route in enumerate(zone_routes):
            if veh_idx >= n_vehicles:
                break
            for cust in route:
                customer_to_vehicle[cust] = veh_idx

        # Allow each customer only on its assigned vehicle
        for cust, veh_idx in customer_to_vehicle.items():
            if 1 <= cust < n_nodes:
                node_idx = manager.NodeToIndex(cust)
                allowed = [False] * n_vehicles
                allowed[veh_idx] = True
                allowed_list = [v for v, ok in enumerate(allowed) if ok]
                for v in range(n_vehicles):
                    if v not in allowed_list:
                        routing.VehicleVar(node_idx).RemoveValue(v)

    # -- Penalty for dropped nodes (allow dropping with high penalty) --
    penalty = 100_000
    for node in range(1, n_nodes):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # -- Search parameters -------------------------------------
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = time_limit_sec
    search_params.log_search = False

    # -- Solve -------------------------------------------------
    t0 = time.time()
    solution = routing.SolveWithParameters(search_params)
    elapsed = time.time() - t0

    if not solution:
        return {
            "status": "no_solution",
            "routes": [],
            "served": 0,
            "served_on_time": 0,
            "served_late": 0,
            "unserved": inst["n_customers"],
            "total_distance": 0.0,
            "total_late": 0.0,
            "vehicles_used": 0,
            "elapsed": elapsed,
        }

    # -- Extract solution --------------------------------------
    routes = []
    total_distance = 0.0
    total_late = 0.0
    served = 0
    served_on_time = 0
    served_late_count = 0
    vehicles_used = 0

    for veh in range(n_vehicles):
        if routing.IsVehicleUsed(solution, veh):
            vehicles_used += 1

        route = []
        idx = routing.Start(veh)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != depot:
                route.append(node)
                served += 1
            idx = solution.Value(routing.NextVar(idx))
        routes.append(route)

    # Compute actual distance and lateness
    for veh, route in enumerate(routes):
        path = [depot] + route + [depot]
        cur_time = 0.0
        coords = inst["coords"]
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            travel = math.sqrt(
                (coords[src][0] - coords[dst][0]) ** 2 +
                (coords[src][1] - coords[dst][1]) ** 2
            )
            total_distance += travel
            cur_time += travel
            if dst != depot:
                tw_start = float(inst["tw_open"][dst])
                tw_end = float(inst["tw_close"][dst])
                service = float(inst["service_time"][dst])
                cur_time = max(cur_time, tw_start)
                late = max(0.0, cur_time - tw_end)
                total_late += late
                if late > 0:
                    served_late_count += 1
                else:
                    served_on_time += 1
                cur_time += service

    return {
        "status": "optimal" if routing.status() == 1 else "feasible",
        "routes": routes,
        "served": served,
        "served_on_time": served_on_time,
        "served_late": served_late_count,
        "unserved": inst["n_customers"] - served,
        "total_distance": total_distance,
        "total_late": total_late,
        "vehicles_used": vehicles_used,
        "elapsed": elapsed,
    }


# ============================================================
# Pretty print
# ============================================================

def print_result(result: Dict, inst: dict, label: str = "") -> None:
    n = inst["n_customers"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Status          : {result['status']}")
    print(f"  Served on-time  : {result['served_on_time']:3d} / {n}")
    print(f"  TW violation    : {result['served_late']:3d} / {n}  (visited but late)")
    print(f"  Unserved (drop) : {result['unserved']:3d} / {n}")
    print(f"  Vehicles used   : {result['vehicles_used']} / {inst['vehicle_limit']}")
    print(f"  Total distance  : {result['total_distance']:.2f}")
    print(f"  Total late      : {result['total_late']:.2f}")
    print(f"  Solve time      : {result['elapsed']:.1f}s")
    print()
    for i, route in enumerate(result["routes"]):
        if route:
            print(f"  Vehicle {i+1:2d}: {route}")


def plot_result(result: Dict, inst: dict, label: str, save_path: str) -> None:
    """Visualize routes by color and save to save_path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not found - skipping")
        return

    coords = inst["coords"]
    depot = coords[0]
    n_cust = inst["n_customers"]

    _, ax = plt.subplots(figsize=(12, 10))

    used_routes = [r for r in result["routes"] if r]
    n_used = len(used_routes)
    cmap = plt.colormaps["tab20" if n_used <= 20 else "hsv"].resampled(max(n_used, 1))

    for idx, route in enumerate(used_routes):
        color = cmap(idx)
        full = [0] + route + [0]

        # route lines + direction arrows
        for k in range(len(full) - 1):
            x0, y0 = coords[full[k]]
            x1, y1 = coords[full[k + 1]]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.9))
        xs = [coords[n][0] for n in full]
        ys = [coords[n][1] for n in full]
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.7)

        # stop dots + labels
        for n in route:
            ax.scatter(coords[n][0], coords[n][1], color=color, s=25, zorder=3)
            ax.text(coords[n][0] + 0.4, coords[n][1] + 0.4, str(n),
                    fontsize=6, color=color)

    # unvisited stops
    visited = {n for r in used_routes for n in r}
    unvisited = [n for n in range(1, n_cust + 1) if n not in visited]
    if unvisited:
        ax.scatter([coords[n][0] for n in unvisited],
                   [coords[n][1] for n in unvisited],
                   color="red", s=50, zorder=4, marker="x",
                   label=f"Unvisited ({len(unvisited)})")
        for n in unvisited:
            ax.text(coords[n][0] + 0.4, coords[n][1] + 0.4, str(n),
                    fontsize=6, color="red")

    # depot
    ax.scatter(depot[0], depot[1], marker="*", s=300, color="black", zorder=5, label="Depot")
    ax.text(depot[0] + 0.4, depot[1] + 0.4, "Depot", fontsize=8, fontweight="bold")

    ax.set_title(
        f"{label}\nserved={result['served']}/{n_cust}  "
        f"vehicles={result['vehicles_used']}/{inst['vehicle_limit']}  "
        f"dist={result['total_distance']:.1f}  late={result['total_late']:.1f}",
        fontsize=11,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot] saved -> {save_path}")


# ============================================================
# Rain event parsing & route re-evaluation
# ============================================================

def parse_rain_events(path: str) -> List[Tuple[float, float, float, List[int]]]:
    """Parse RAIN / ACCIDENT events from scenario file.
    Returns list of (trigger_time, duration, multiplier, affected_nodes).
    """
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            upper = s.upper()
            parts = s.split()
            if upper.startswith("RAIN"):
                # RAIN trigger duration mult radius node1 node2 ...
                trigger  = float(parts[1])
                duration = float(parts[2])
                mult     = float(parts[3])
                nodes    = [int(x) for x in parts[5:]]   # parts[4]=radius
                events.append((trigger, duration, mult, nodes))
            elif upper.startswith("ACCIDENT"):
                # ACCIDENT trigger duration mult node1 node2
                trigger  = float(parts[1])
                duration = float(parts[2])
                mult     = float(parts[3])
                nodes    = [int(x) for x in parts[4:]]   # no radius field
                events.append((trigger, duration, mult, nodes))
    return events


def evaluate_routes_with_rain(
    routes: List[List[int]],
    inst: dict,
    rain_events: List[Tuple[float, float, float, List[int]]],
) -> Dict:
    """Re-simulate OR-Tools routes under rain conditions (time-dependent tt).

    For each edge traversal, checks if rain is active at departure time.
    Edge is rain-affected only if BOTH endpoints are in the rain zone.
    """
    coords = inst["coords"]
    depot = 0

    total_distance = 0.0
    total_late = 0.0
    late_count = 0
    served_count = 0

    def edge_travel(src: int, dst: int, cur_time: float) -> float:
        base = math.sqrt(
            (coords[src][0] - coords[dst][0]) ** 2 +
            (coords[src][1] - coords[dst][1]) ** 2
        )
        mult = 1.0
        for trigger, duration, m, nodes in rain_events:
            end = trigger + duration
            if cur_time >= trigger and cur_time < end:
                node_set = set(nodes)
                if src in node_set and dst in node_set:
                    mult = max(mult, m)
        return base * mult

    for route in routes:
        if not route:
            continue
        path = [depot] + route + [depot]
        cur_time = 0.0
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            travel = edge_travel(src, dst, cur_time)
            total_distance += travel
            cur_time += travel
            if dst != depot:
                tw_open = float(inst["tw_open"][dst])
                tw_close = float(inst["tw_close"][dst])
                service = float(inst["service_time"][dst])
                cur_time = max(cur_time, tw_open)
                late = max(0.0, cur_time - tw_close)
                if late > 0:
                    total_late += late
                    late_count += 1
                else:
                    served_count += 1
                cur_time += service

    return {
        "total_distance": total_distance,
        "total_late": total_late,
        "served_count": served_count,
        "late_count": late_count,
    }


# NOTE: OR-Tools is a static solver -it optimizes with full knowledge from the start.
# Therefore it CANNOT model dynamic/sudden events (accidents, rain) fairly.
# OR-Tools is used ONLY as a static no-event baseline for routing quality comparison.
#
# For dynamic event comparison, use:
#   RL-only  (use_llm_event_response=False) vs
#   RL+LLM   (use_llm_event_response=True)
# Both start from the same trained policy and encounter events mid-episode.


# ============================================================
# Main
# ============================================================

def _routing_phase(clusters: List[List[int]], inst: dict) -> List[List[int]]:
    """Section 3.2 (Gocken 2019): greedy nearest-neighbour routing within each cluster.

    For each cluster: start from the depot, add the nearest customer that satisfies
    TW and capacity. If no feasible customer remains for the current route, start a
    new route with the next nearest-to-depot un-routed customer.
    Returns TW- and capacity-feasible route list (may have more routes than clusters).
    """
    import math as _math

    coords = inst["coords"]
    Q = inst["vehicle_capacity"]

    def _d(i, j):
        return _math.sqrt((coords[i][0] - coords[j][0]) ** 2 +
                          (coords[i][1] - coords[j][1]) ** 2)

    all_routes: List[List[int]] = []

    for cluster in clusters:
        unrouted = sorted(cluster, key=lambda c: _d(0, c))
        while unrouted:
            # Start new route with nearest un-routed customer to depot
            route: List[int] = []
            demand = 0.0
            cur_node = 0
            cur_time = 0.0

            # First customer: nearest to depot
            first = unrouted[0]
            travel = _d(0, first)
            arr = cur_time + travel
            if arr <= inst["tw_close"][first] + 1e-8 and inst["demands"][first] <= Q + 1e-8:
                route.append(first)
                unrouted.remove(first)
                demand += inst["demands"][first]
                cur_time = max(arr, inst["tw_open"][first]) + inst["service_time"][first]
                cur_node = first
            else:
                # Can't start a route with this customer -assign alone and move on
                all_routes.append([first])
                unrouted.remove(first)
                continue

            # Keep adding nearest feasible customer
            while unrouted:
                best_c, best_d = None, float("inf")
                for c in unrouted:
                    dist_c = _d(cur_node, c)
                    arr_c = cur_time + dist_c
                    if arr_c > inst["tw_close"][c] + 1e-8:
                        continue
                    if demand + inst["demands"][c] > Q + 1e-8:
                        continue
                    if dist_c < best_d:
                        best_d, best_c = dist_c, c
                if best_c is None:
                    break
                route.append(best_c)
                unrouted.remove(best_c)
                demand += inst["demands"][best_c]
                arr_c = cur_time + _d(cur_node, best_c)
                cur_time = max(arr_c, inst["tw_open"][best_c]) + inst["service_time"][best_c]
                cur_node = best_c

            all_routes.append(route)

    return [r for r in all_routes if r]


# ============================================================
# Clustering algorithms -Gocken & Yaktubay (2019)
# Phase 1 (clustering): capacity constraint only.
# Phase 2 (routing):    _routing_phase applies TW-aware greedy routing (Section 3.2).
# ============================================================

def _kmeans_raw(inst: dict) -> List[List[int]]:
    """K-means Phase 1: 2D spatial clusters, K = floor(total_demand/Q) + 1."""
    import numpy as _np
    from sklearn.cluster import KMeans as _KMeans

    coords = inst["coords"]
    Q = inst["vehicle_capacity"]
    n = inst["n_customers"]
    total_demand = sum(inst["demands"][c] for c in range(1, n + 1))
    K = max(1, int(total_demand // Q) + 1)

    X = _np.array([[coords[c][0], coords[c][1]] for c in range(1, n + 1)])
    labels = _KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X)

    raw: dict = {}
    for idx, c in enumerate(range(1, n + 1)):
        raw.setdefault(int(labels[idx]), []).append(c)

    return list(raw.values())


def kmeans_zones(inst: dict) -> List[List[int]]:
    return _routing_phase(_kmeans_raw(inst), inst)


def _cbased_raw(inst: dict, max_iter: int = 10) -> List[List[int]]:
    """Centroid-based Phase 1: capacity-only clustering (Shin & Han 2011)."""
    import math as _math

    coords = inst["coords"]
    Q = inst["vehicle_capacity"]
    n = inst["n_customers"]

    def _edist(a, b):
        return _math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    unassigned = list(range(1, n + 1))
    clusters: List[List[int]] = []

    # Phase 1: cluster construction
    first_cust = max(unassigned, key=lambda c: _edist(coords[0], coords[c]))
    while unassigned:
        if not clusters:
            seed = first_cust
        else:
            seed = unassigned[0]
        unassigned.remove(seed)

        cluster = [seed]
        demand = inst["demands"][seed]
        centroid = list(coords[seed])

        while unassigned:
            closest = min(unassigned, key=lambda c: _edist(coords[c], centroid))
            d = inst["demands"][closest]
            if demand + d > Q + 1e-8:
                break
            cluster.append(closest)
            unassigned.remove(closest)
            demand += d
            centroid = [
                sum(coords[c][0] for c in cluster) / len(cluster),
                sum(coords[c][1] for c in cluster) / len(cluster),
            ]
        clusters.append(cluster)

    # Phase 2: cluster adjustment
    for _ in range(max_iter):
        centroids = [
            [sum(coords[c][0] for c in cl) / len(cl),
             sum(coords[c][1] for c in cl) / len(cl)]
            for cl in clusters
        ]
        changed = False
        for i, cl in enumerate(clusters):
            for c in list(cl):
                own_d = _edist(coords[c], centroids[i])
                for j, cj in enumerate(clusters):
                    if j == i:
                        continue
                    if _edist(coords[c], centroids[j]) < own_d:
                        dem_j = sum(inst["demands"][x] for x in cj)
                        if dem_j + inst["demands"][c] <= Q + 1e-8:
                            cl.remove(c)
                            cj.append(c)
                            centroids[i] = (
                                [sum(coords[x][0] for x in cl) / len(cl),
                                 sum(coords[x][1] for x in cl) / len(cl)]
                                if cl else centroids[i]
                            )
                            centroids[j] = [
                                sum(coords[x][0] for x in cj) / len(cj),
                                sum(coords[x][1] for x in cj) / len(cj),
                            ]
                            changed = True
                            break
        if not changed:
            break

    return [cl for cl in clusters if cl]


def cbased_zones(inst: dict) -> List[List[int]]:
    return _routing_phase(_cbased_raw(inst), inst)


def _dbscan_raw(inst: dict, eps: float = 12.0, min_pts: int = 4) -> List[List[int]]:
    """DBSCAN Phase 1: capacity-only clustering -Table III in Gocken 2019."""
    import math as _math

    coords = inst["coords"]
    Q = inst["vehicle_capacity"]
    n = inst["n_customers"]
    custs = list(range(1, n + 1))

    def _d(i, j):
        return _math.sqrt((coords[i][0] - coords[j][0]) ** 2 +
                          (coords[i][1] - coords[j][1]) ** 2)

    nbrs = {c: [o for o in custs if o != c and _d(c, o) <= eps] for c in custs}
    core = {c for c in custs if len(nbrs[c]) >= min_pts}

    visited: set = set()
    clusters: List[List[int]] = []

    for seed in core:
        if seed in visited:
            continue
        cluster, demand = [], 0.0
        queue = [seed]
        visited.add(seed)
        while queue:
            p = queue.pop(0)
            dp = inst["demands"][p]
            if demand + dp > Q + 1e-8:
                visited.discard(p)
                continue
            cluster.append(p)
            demand += dp
            if p in core:
                for nb in nbrs[p]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
        if cluster:
            clusters.append(cluster)

    # assign noise/unvisited to nearest cluster with capacity
    assigned = {c for cl in clusters for c in cl}
    for c in custs:
        if c in assigned:
            continue
        best_cl, best_d = None, float("inf")
        for cl in clusters:
            dem = sum(inst["demands"][x] for x in cl)
            if dem + inst["demands"][c] > Q + 1e-8:
                continue
            cx = sum(coords[x][0] for x in cl) / len(cl)
            cy = sum(coords[x][1] for x in cl) / len(cl)
            dist = _math.sqrt((coords[c][0] - cx) ** 2 + (coords[c][1] - cy) ** 2)
            if dist < best_d:
                best_d, best_cl = dist, cl
        if best_cl is not None:
            best_cl.append(c)
        else:
            clusters.append([c])

    return clusters


def dbscan_zones(inst: dict) -> List[List[int]]:
    return _routing_phase(_dbscan_raw(inst), inst)


def _snn_raw(inst: dict, k: int = 40, eps: float = 12.0,
             T: int = 1, min_pts: int = 5) -> List[List[int]]:
    """SNN Phase 1: capacity-only clustering -Table IV in Gocken 2019.

    k     : number of nearest neighbours per point
    eps   : Euclidean radius for neighbourhood search
    T     : minimum SNN similarity threshold for density count
    min_pts: minimum density to be a core point
    """
    import math as _math

    coords = inst["coords"]
    Q = inst["vehicle_capacity"]
    n = inst["n_customers"]
    custs = list(range(1, n + 1))

    def _d(i, j):
        return _math.sqrt((coords[i][0] - coords[j][0]) ** 2 +
                          (coords[i][1] - coords[j][1]) ** 2)

    # Step 1: k nearest neighbours
    nn: dict = {}
    for c in custs:
        dists = sorted([(o, _d(c, o)) for o in custs if o != c], key=lambda x: x[1])
        nn[c] = {o for o, _ in dists[:k]}

    # Step 2: revise to mutual neighbours
    nn_mutual = {c: {o for o in nn[c] if c in nn[o]} for c in custs}

    # Step 3: SNN similarity = |NN(i) ∩ NN(j)| for j in mutual NN of i
    def _snn_sim(i, j):
        return len(nn_mutual[i] & nn_mutual[j])

    # Step 4-5: Eps neighbourhood and SNN density
    eps_nbrs = {c: [o for o in custs if o != c and _d(c, o) <= eps] for c in custs}
    snn_density = {
        c: sum(1 for o in eps_nbrs[c] if _snn_sim(c, o) >= T)
        for c in custs
    }

    # Step 6: core points
    core = {c for c in custs if snn_density[c] >= min_pts}

    visited: set = set()
    clusters: List[List[int]] = []

    for seed in core:
        if seed in visited:
            continue
        cluster, demand = [], 0.0
        queue = [seed]
        visited.add(seed)
        while queue:
            p = queue.pop(0)
            dp = inst["demands"][p]
            if demand + dp > Q + 1e-8:
                visited.discard(p)
                continue
            cluster.append(p)
            demand += dp
            if p in core:
                for nb in eps_nbrs[p]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
        if cluster:
            clusters.append(cluster)

    # assign unvisited/noise
    assigned = {c for cl in clusters for c in cl}
    for c in custs:
        if c in assigned:
            continue
        best_cl, best_d = None, float("inf")
        for cl in clusters:
            dem = sum(inst["demands"][x] for x in cl)
            if dem + inst["demands"][c] > Q + 1e-8:
                continue
            cx = sum(coords[x][0] for x in cl) / len(cl)
            cy = sum(coords[x][1] for x in cl) / len(cl)
            dist = _math.sqrt((coords[c][0] - cx) ** 2 + (coords[c][1] - cy) ** 2)
            if dist < best_d:
                best_d, best_cl = dist, cl
        if best_cl is not None:
            best_cl.append(c)
        else:
            clusters.append([c])

    return clusters


def snn_zones(inst: dict) -> List[List[int]]:
    return _routing_phase(_snn_raw(inst), inst)


def _kim2006_raw(inst: dict) -> List[List[int]]:
    """Capacitated K-means clustering -Kim et al. (2006) Section 5, adapted for Solomon.

    Removed vs. paper (not present in Solomon VRPTW):
      - Disposal trips / landfill visits
      - Driver lunch break
      - Route capacity (max stops/day) -Solomon has no per-vehicle daily stop limit

    Retained exactly as paper:
      - N estimated from total volume / vehicle capacity
      - Random initial centroids (paper: "N initial centroid seed stops selected randomly")
      - Grand centroid: stops sorted descending by distance from centroid-of-centroids;
        farthest stop assigned first to nearest feasible cluster
      - Cluster capacity = volume + TSP-estimated route time (stops-per-day removed)
      - TW conflict check: EDF greedy simulation
      - Convergence: repeat until no assignment change
      - Move improvement: relocate stop to nearer centroid when capacity + TW allow
      - N-increment loop: N += 1 and restart if routing phase leaves unrouted stops
    """
    import math as _math
    import random as _random

    coords = inst["coords"]
    Q = inst["vehicle_capacity"]
    n = inst["n_customers"]
    custs = list(range(1, n + 1))
    depot_close = float(inst["tw_close"][0])

    def _d(i, j):
        return _math.sqrt((coords[i][0] - coords[j][0]) ** 2 +
                          (coords[i][1] - coords[j][1]) ** 2)

    def _nn_route_time(cluster):
        """Nearest-neighbour TSP estimate: total travel time + service time."""
        if not cluster:
            return 0.0
        unvisited = list(cluster)
        cur, total = 0, 0.0
        while unvisited:
            nxt = min(unvisited, key=lambda c: _d(cur, c))
            total += _d(cur, nxt) + inst["service_time"][nxt]
            cur = nxt
            unvisited.remove(nxt)
        return total + _d(cur, 0)

    def _tw_feasible(cluster):
        """EDF greedy: returns True if all stops in cluster can be served within TW."""
        if not cluster:
            return True
        seq = sorted(cluster, key=lambda x: inst["tw_close"][x])
        cur_time, cur_node = 0.0, 0
        for c in seq:
            arr = cur_time + _d(cur_node, c)
            if arr > inst["tw_close"][c] + 1e-8:
                return False
            cur_time = max(arr, inst["tw_open"][c]) + inst["service_time"][c]
            cur_node = c
        return cur_time + _d(cur_node, 0) <= depot_close + 1e-8

    def _can_add(c, cluster):
        """Cluster capacity check: volume + TSP route time + TW conflict."""
        if sum(inst["demands"][x] for x in cluster) + inst["demands"][c] > Q + 1e-8:
            return False
        if _nn_route_time(cluster + [c]) > depot_close + 1e-8:
            return False
        return _tw_feasible(cluster + [c])

    total_demand = sum(inst["demands"][c] for c in custs)
    N = max(1, int(total_demand // Q) + 1)

    clusters: List[List[int]] = []

    for _inc in range(30):  # N-increment loop
        # -- Random initial centroids (paper: "selected randomly") ------------
        _random.seed(42 + _inc)
        seeds = _random.sample(custs, min(N, len(custs)))
        centroid_pos = [[coords[s][0], coords[s][1]] for s in seeds]

        clusters = [[] for _ in range(N)]
        prev_assign: Optional[List] = None

        # -- K-means with grand centroid (repeat until no change) -------------
        for _ in range(50):
            # Grand centroid = centroid of the N cluster centroids
            gc_x = sum(cx for cx, _ in centroid_pos) / N
            gc_y = sum(cy for _, cy in centroid_pos) / N

            # Sort customers descending by distance from grand centroid
            sorted_custs = sorted(
                custs,
                key=lambda c: _math.sqrt((coords[c][0] - gc_x) ** 2 +
                                         (coords[c][1] - gc_y) ** 2),
                reverse=True,
            )

            new_clusters: List[List[int]] = [[] for _ in range(N)]

            for c in sorted_custs:
                order = sorted(
                    range(N),
                    key=lambda i: _math.sqrt((coords[c][0] - centroid_pos[i][0]) ** 2 +
                                             (coords[c][1] - centroid_pos[i][1]) ** 2),
                )
                placed = False
                for i in order:
                    if _can_add(c, new_clusters[i]):
                        new_clusters[i].append(c)
                        placed = True
                        break
                if not placed:
                    new_clusters[order[0]].append(c)  # force to nearest

            # Update centroids
            for i in range(N):
                if new_clusters[i]:
                    centroid_pos[i] = [
                        sum(coords[c][0] for c in new_clusters[i]) / len(new_clusters[i]),
                        sum(coords[c][1] for c in new_clusters[i]) / len(new_clusters[i]),
                    ]

            curr_assign = [tuple(sorted(cl)) for cl in new_clusters]
            clusters = new_clusters
            if curr_assign == prev_assign:
                break
            prev_assign = curr_assign

        # -- Move improvement: relocate stop to nearer centroid if capacity+TW ok --
        for _ in range(50):
            cp = [
                [sum(coords[c][0] for c in cl) / len(cl),
                 sum(coords[c][1] for c in cl) / len(cl)]
                if cl else centroid_pos[i]
                for i, cl in enumerate(clusters)
            ]
            moved = False
            for i, cl in enumerate(clusters):
                for c in list(cl):
                    d_own = _math.sqrt((coords[c][0] - cp[i][0]) ** 2 +
                                       (coords[c][1] - cp[i][1]) ** 2)
                    for j, cj in enumerate(clusters):
                        if j == i:
                            continue
                        d_other = _math.sqrt((coords[c][0] - cp[j][0]) ** 2 +
                                             (coords[c][1] - cp[j][1]) ** 2)
                        if d_other >= d_own:
                            continue
                        if not _can_add(c, cj):
                            continue
                        cl.remove(c)
                        cj.append(c)
                        cp[i] = ([sum(coords[x][0] for x in cl) / len(cl),
                                   sum(coords[x][1] for x in cl) / len(cl)]
                                  if cl else cp[i])
                        cp[j] = [sum(coords[x][0] for x in cj) / len(cj),
                                  sum(coords[x][1] for x in cj) / len(cj)]
                        moved = True
                        break
            if not moved:
                break

        # -- N-increment check -------------------------------------------------
        # All clusters must be TW-feasible; if any fails, add one more vehicle.
        raw = [cl for cl in clusters if cl]
        if all(_tw_feasible(cl) for cl in raw):
            break
        N += 1

    clusters = [cl for cl in clusters if cl]

    # -- Convex-hull-overlap merge post-processing ----------------------------
    # Neighbor criterion: two clusters share a Delaunay triangulation edge
    # (threshold-free geographic adjacency).
    # For each neighboring pair:
    #   1. Try direct merge (demand + TW feasible)
    #   2. Try removing one stop → merge rest → relocate stop elsewhere
    import numpy as _np
    from scipy.spatial import Delaunay as _Delaunay

    def _cluster_feasible(cl):
        """Merge-phase feasibility: demand + TW only (OR-Tools handles actual routing)."""
        if sum(inst["demands"][x] for x in cl) > Q + 1e-8:
            return False
        return _tw_feasible(cl)

    def _can_relocate(s, cl):
        if sum(inst["demands"][x] for x in cl) + inst["demands"][s] > Q + 1e-8:
            return False
        return _tw_feasible(cl + [s])

    def _build_neighbors(clusters):
        """Return set of (i,j) cluster index pairs connected by a Delaunay edge."""
        all_pts = _np.array([[coords[c][0], coords[c][1]] for cl in clusters for c in cl])
        node_to_cluster = {c: ci for ci, cl in enumerate(clusters) for c in cl}
        stop_order = [c for cl in clusters for c in cl]
        tri = _Delaunay(all_pts)
        neighbors = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    ci = node_to_cluster[stop_order[simplex[a]]]
                    cj = node_to_cluster[stop_order[simplex[b]]]
                    if ci != cj:
                        neighbors.add((min(ci, cj), max(ci, cj)))
        return neighbors

    merged_any = True
    while merged_any:
        merged_any = False
        neighbors = _build_neighbors(clusters)
        for (i, j) in sorted(neighbors):
            if i >= len(clusters) or j >= len(clusters):
                break
            combined = clusters[i] + clusters[j]

            # Case 1: direct merge
            if _cluster_feasible(combined):
                clusters[i] = combined
                clusters.pop(j)
                merged_any = True
                break

            # Case 2: remove one stop → merge rest → relocate stop elsewhere
            for s in combined:
                rest = [x for x in combined if x != s]
                if not _cluster_feasible(rest):
                    continue
                for k in range(len(clusters)):
                    if k in (i, j):
                        continue
                    if _can_relocate(s, clusters[k]):
                        clusters[k].append(s)
                        clusters[i] = rest
                        clusters.pop(j)
                        merged_any = True
                        break
                if merged_any:
                    break
            if merged_any:
                break

    return clusters


def random_zones(inst: dict) -> List[List[int]]:
    """Random-based route construction with capacity + TW -Table V in Gocken 2019."""
    import math as _math
    import random as _random

    coords = inst["coords"]
    Q = inst["vehicle_capacity"]
    n = inst["n_customers"]
    custs = list(range(1, n + 1))
    depot_close = float(inst["tw_close"][0])

    def _d(i, j):
        return _math.sqrt((coords[i][0] - coords[j][0]) ** 2 +
                          (coords[i][1] - coords[j][1]) ** 2)

    # Step 1: start with closest customer to depot
    unrouted = sorted(custs, key=lambda c: _d(0, c))
    routes: List[List[int]] = []
    route: List[int] = []
    demand = 0.0
    cur_time = 0.0
    cur_node = 0

    _random.seed(42)
    remaining = list(unrouted)

    while remaining:
        candidates = list(remaining)
        _random.shuffle(candidates)
        assigned = False
        for c in candidates:
            travel = _d(cur_node, c)
            arr = cur_time + travel
            # TW check: must arrive before tw_close, and can return to depot in time
            if arr > inst["tw_close"][c] + 1e-8:
                continue
            # Capacity check
            if demand + inst["demands"][c] > Q + 1e-8:
                continue
            # Can return to depot after service?
            depart = max(arr, inst["tw_open"][c]) + inst["service_time"][c]
            if depart + _d(c, 0) > depot_close + 1e-8:
                continue
            route.append(c)
            demand += inst["demands"][c]
            cur_time = depart
            cur_node = c
            remaining.remove(c)
            assigned = True
            break
        if not assigned:
            if route:
                routes.append(route)
            route, demand, cur_time, cur_node = [], 0.0, 0.0, 0
            if remaining:
                # start next route with closest unrouted to depot
                remaining.sort(key=lambda c: _d(0, c))

    if route:
        routes.append(route)
    return routes


if __name__ == "__main__":
    import argparse
    import os as _os

    parser = argparse.ArgumentParser(description="OR-Tools VRPTW solver")
    parser.add_argument("scenario", type=str,
                        help="Dataset name, e.g. c101 or c101_rain_A")
    parser.add_argument("--data-dir", default=None,
                        help="Directory containing Solomon .txt files (default: ./data/Solomon)")
    parser.add_argument("--time-limit", type=int, default=30,
                        help="OR-Tools time limit per experiment in seconds (default: 30)")
    parser.add_argument("--zone-mode",
                        choices=["all", "clustering", "solution", "global"],
                        default="all",
                        help="Zone assignment mode:\n"
                             "  clustering → Kim2006 capacitated K-means\n"
                             "  solution   → pre-computed solution file zones\n"
                             "  global     → no zone restriction (OR-Tools decides)\n"
                             "  all        → run all three (default)")
    args = parser.parse_args()

    DATA_FILE  = args.scenario
    TIME_LIMIT = args.time_limit
    ZONE_MODE  = args.zone_mode

    INSTANCE      = DATA_FILE.split("_rain_")[0] if "_rain_" in DATA_FILE else DATA_FILE
    RAIN_SCENARIO = DATA_FILE if "_rain_" in DATA_FILE else None

    _BASE_DIR    = _os.path.dirname(_os.path.abspath(__file__))
    _SOLOMON_DIR = args.data_dir if args.data_dir else _os.path.join(_BASE_DIR, "data", "Solomon")
    data_path    = _os.path.join(_SOLOMON_DIR, DATA_FILE + ".txt")
    sol_path     = _os.path.join(_SOLOMON_DIR, INSTANCE + "_sol.txt")
    _PLOT_DIR    = _os.path.join(_BASE_DIR, "plots", INSTANCE)

    inst      = load_solomon(data_path)
    tt_base   = build_tt_matrix(inst)
    sol_zones = parse_solution_zones(sol_path)

    print(f"[Instance]  {inst['name']}  customers={inst['n_customers']}  "
          f"vehicles={inst['vehicle_limit']}  capacity={inst['vehicle_capacity']}")
    print(f"[ZoneMode]  {ZONE_MODE}")

    rain_events: List[Tuple[float, float, float, List[int]]] = []
    if RAIN_SCENARIO:
        rain_path   = _os.path.join(_SOLOMON_DIR, RAIN_SCENARIO + ".txt")
        rain_events = parse_rain_events(rain_path)
        print(f"[Rain]      {RAIN_SCENARIO}  events={len(rain_events)}")
    print(f"[TimeLimit] {TIME_LIMIT}s per experiment\n")

    def _run_one(label: str, zone_routes, tt: List[List[int]],
                 plot_name: str, n_vehicles_max: Optional[int] = None) -> Tuple[str, Dict]:
        print(f"{'-'*65}")
        print(f"{label}")
        if n_vehicles_max is not None:
            print(f"  [Vehicles] capped at {n_vehicles_max} (matched to unknown)")
        if zone_routes is not None:
            sizes = sorted([len(c) for c in zone_routes], reverse=True)
            print(f"  [Zones] {len(zone_routes)} zones | sizes={sizes}")
        r = solve_vrptw(inst, tt, time_limit_sec=TIME_LIMIT,
                        zone_routes=zone_routes, n_vehicles_max=n_vehicles_max)
        n = inst["n_customers"]
        print(f"  [OR-Tools] {r['vehicles_used']} veh  "
              f"on_time={r['served_on_time']}/{n}  "
              f"tw_viol={r['served_late']}  "
              f"unserved={r['unserved']}  "
              f"dist={r['total_distance']:.1f}  late={r['total_late']:.1f}")
        plot_result(r, inst, f"{label} — {INSTANCE}",
                    _os.path.join(_PLOT_DIR, plot_name))
        return (label, r)

    def _run_experiments(
        tt: List[List[int]],
        n_vehicles_override: Optional[Dict[str, int]] = None,
        plot_prefix: str = "",
    ) -> List[Tuple[str, Dict]]:
        res = []
        run_all = (ZONE_MODE == "all")

        def _nveh(label):
            return n_vehicles_override.get(label) if n_vehicles_override else None

        if run_all or ZONE_MODE == "clustering":
            label = "Kim2006 clustering (POSTECH)"
            clusters_h = _kim2006_raw(inst)
            res.append(_run_one(label, clusters_h, tt,
                                f"{plot_prefix}H_kim2006.png", _nveh(label)))

        if (run_all or ZONE_MODE == "solution") and sol_zones is not None:
            label = "Solution file zones"
            res.append(_run_one(label, sol_zones, tt,
                                f"{plot_prefix}F_solution.png", _nveh(label)))
        elif ZONE_MODE == "solution" and sol_zones is None:
            print(f"[Warning] solution file not found: {sol_path}")

        if run_all or ZONE_MODE == "global":
            label = "Global (no zone restriction)"
            res.append(_run_one(label, None, tt,
                                f"{plot_prefix}G_global.png", _nveh(label)))

        return res

    def _print_summary(results: List[Tuple[str, Dict]], title: str) -> None:
        n = inst["n_customers"]
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
        print(f"  {'Experiment':<32} {'Veh':>5} {'OnTime':>7} {'TWViol':>7} "
              f"{'Unserved':>9} {'Dist':>8} {'Late':>8} {'T':>6}")
        print(f"  {'-'*32} {'-'*5} {'-'*7} {'-'*7} {'-'*9} {'-'*8} {'-'*8} {'-'*6}")
        for label, r in results:
            print(f"  {label:<32} "
                  f"{r['vehicles_used']:>2}/{inst['vehicle_limit']:<3} "
                  f"{r['served_on_time']:>4}/{n:<3} "
                  f"{r['served_late']:>4}/{n:<3} "
                  f"{r['unserved']:>6}/{n:<3} "
                  f"{r['total_distance']:>8.1f} "
                  f"{r['total_late']:>8.2f} "
                  f"{r['elapsed']:>5.1f}s")

    def _print_rain_eval(results: List[Tuple[str, Dict]], title: str) -> None:
        n = inst["n_customers"]
        print(f"\n{'='*80}")
        print(f"  {title}  (time-dependent rain re-simulation)")
        print(f"{'='*80}")
        print(f"  {'Experiment':<32} {'OnTime':>7} {'TWViol':>7} "
              f"{'Unserved':>9} {'Dist':>8} {'Late':>8}")
        print(f"  {'-'*32} {'-'*7} {'-'*7} {'-'*9} {'-'*8} {'-'*8}")
        for label, r in results:
            if not r["routes"]:
                continue
            ev = evaluate_routes_with_rain(r["routes"], inst, rain_events)
            unserved = n - ev["served_count"] - ev["late_count"]
            print(f"  {label:<32} "
                  f"{ev['served_count']:>4}/{n:<3} "
                  f"{ev['late_count']:>4}/{n:<3} "
                  f"{unserved:>6}/{n:<3} "
                  f"{ev['total_distance']:>8.1f} "
                  f"{ev['total_late']:>8.2f}")

    unknown_results = _run_experiments(tt_base, plot_prefix="unknown_")
    _print_summary(unknown_results, "UNKNOWN — planned with base TT")

    if RAIN_SCENARIO:
        tt_rain = build_rain_tt_matrix(inst, rain_events)
        vehicles_used_map = {label: r["vehicles_used"] for label, r in unknown_results}
        print(f"\n[Known] fixing vehicle caps to match unknown: "
              f"{ {k: v for k, v in vehicles_used_map.items()} }")
        known_results = _run_experiments(tt_rain, n_vehicles_override=vehicles_used_map,
                                         plot_prefix="known_")
        _print_summary(known_results, "KNOWN — planned with rain TT (same vehicle cap)")

        print()
        _print_rain_eval(unknown_results, "UNKNOWN — re-simulated under actual rain")
        _print_rain_eval(known_results,   "KNOWN   — re-simulated under actual rain")
