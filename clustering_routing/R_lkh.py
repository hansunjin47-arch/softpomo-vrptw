"""
LKH-3 VRPTW solver for Solomon benchmark instances.

Requires LKH-3 compiled in WSL at /usr/local/bin/LKH.
Build:
    wsl -d Ubuntu -u root -- bash -c \
        "cd /opt/LKH-3.0.6 && make CFLAGS='-O3 -Wall -IINCLUDE -DTWO_LEVEL_TREE -g -lm -fcommon'"
    wsl -d Ubuntu -u root -- cp /opt/LKH-3.0.6/LKH /usr/local/bin/LKH

Usage:
    python R_lkh.py c101
    python R_lkh.py c101_rain_A --time-limit 60
    python R_lkh.py c101 --zone-mode global
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

WSL_LKH_BIN = "/usr/local/bin/LKH"


# ============================================================
# Solomon parser  (same as R_ortools.py)
# ============================================================

def load_solomon(path: str) -> dict:
    coords, demands, tw_open, tw_close, service_time = [], [], [], [], []
    vehicle_limit = vehicle_capacity = None

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
            i += 2
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
# Travel time matrices
# ============================================================

def build_tt_matrix(inst: dict, speed: float = 1.0) -> List[List[int]]:
    coords = inst["coords"]
    n = len(coords)
    return [
        [int(round(math.sqrt((coords[i][0] - coords[j][0])**2 +
                              (coords[i][1] - coords[j][1])**2) / speed))
         for j in range(n)]
        for i in range(n)
    ]


def build_rain_tt_matrix(
    inst: dict,
    rain_events: List[Tuple[float, float, float, List[int]]],
) -> List[List[int]]:
    coords = inst["coords"]
    n = len(coords)
    tt = []
    for i in range(n):
        row = []
        for j in range(n):
            base = math.sqrt((coords[i][0] - coords[j][0])**2 +
                             (coords[i][1] - coords[j][1])**2)
            mult = 1.0
            for _, _, m, nodes in rain_events:
                node_set = set(nodes)
                if i in node_set and j in node_set:
                    mult = max(mult, m)
            row.append(int(round(base * mult)))
        tt.append(row)
    return tt


# ============================================================
# Zone plan parser
# ============================================================

def parse_solution_zones(sol_path: str) -> Optional[List[List[int]]]:
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
# Rain event parser
# ============================================================

def parse_rain_events(path: str) -> List[Tuple[float, float, float, List[int]]]:
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
                nodes    = [int(x) for x in parts[5:]]   # parts[4]=radius, nodes from parts[5]
                events.append((trigger, duration, mult, nodes))
            elif upper.startswith("ACCIDENT"):
                # ACCIDENT trigger duration <desc|mult> node1 node2
                trigger  = float(parts[1])
                duration = float(parts[2])
                _ACC_MULT = {
                    "3-car-collision": 5.0, "4-car-pile-up": 8.5, "5-car-pile-up": 13.0,
                    "low": 1.5, "medium": 2.0, "high": 3.0,
                }
                raw = parts[3]
                try:
                    mult = float(raw)
                except ValueError:
                    mult = _ACC_MULT.get(raw.lower(), 5.0)
                nodes    = [int(x) for x in parts[4:]]
                events.append((trigger, duration, mult, nodes))
    return events


# ============================================================
# WSL path helpers
# ============================================================

def win_to_wsl(win_path: str) -> str:
    """Convert C:\\foo\\bar  →  /mnt/c/foo/bar"""
    p = Path(win_path)
    drive = p.drive.rstrip(":").lower()
    rest = str(p.relative_to(p.anchor)).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


# ============================================================
# LKH-3 VRPTW file writers
# ============================================================

def _write_vrp_file(
    inst: dict,
    tt: List[List[int]],
    vrp_path: str,
    n_vehicles: int,
    zone_routes: Optional[List[List[int]]] = None,
    distance_limit: Optional[int] = None,
) -> None:
    """Write VRPLIB VRPTW problem file for LKH-3."""
    n_nodes = len(inst["coords"])
    depot_close = int(inst["tw_close"][0])
    # DISTANCE = max route duration. For rain TT, caller can pass a larger value.
    dist = distance_limit if distance_limit is not None else depot_close

    with open(vrp_path, "w") as f:
        f.write(f"NAME: problem\n")
        f.write(f"COMMENT: VRPTW\n")
        f.write(f"TYPE: CVRPTW\n")
        f.write(f"DIMENSION: {n_nodes}\n")
        f.write(f"VEHICLES: {n_vehicles}\n")
        f.write(f"CAPACITY: {int(inst['vehicle_capacity'])}\n")
        f.write(f"DISTANCE: {dist}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in tt:
            f.write(" ".join(str(v) for v in row) + "\n")
        f.write("DEMAND_SECTION\n")
        for i in range(n_nodes):
            f.write(f"{i + 1} {int(inst['demands'][i])}\n")
        f.write("TIME_WINDOW_SECTION\n")
        for i in range(n_nodes):
            f.write(f"{i + 1} {int(inst['tw_open'][i])} {int(inst['tw_close'][i])}\n")
        f.write("SERVICE_TIME_SECTION\n")
        for i in range(n_nodes):
            f.write(f"{i + 1} {int(inst['service_time'][i])}\n")

        # Zone restriction: fix each customer to its vehicle via PICKUP_AND_DELIVERY workaround
        # LKH-3 does not support hard zone assignment natively; we skip soft zone hints here.
        # (Zone-based ordering is handled by initial tour hinting if needed.)

        f.write("DEPOT_SECTION\n1\n-1\nEOF\n")


def _write_par_file(
    vrp_wsl: str,
    tour_wsl: str,
    par_path: str,
    time_limit: int,
    runs: int,
    max_trials: int,
    mtsp_objective: Optional[str] = None,
) -> None:
    """Write LKH-3 parameter file (Windows path, content uses WSL paths)."""
    with open(par_path, "w") as f:
        f.write(f"PROBLEM_FILE = {vrp_wsl}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_wsl}\n")
        f.write(f"TIME_LIMIT = {time_limit}\n")
        f.write(f"MAX_TRIALS = {max_trials}\n")
        f.write(f"RUNS = {runs}\n")
        f.write("TRACE_LEVEL = 0\n")
        f.write("SEED = 42\n")
        if mtsp_objective:
            f.write(f"MTSP_OBJECTIVE = {mtsp_objective}\n")


def _parse_tour(tour_path: str, n_customers: int) -> List[List[int]]:
    """Parse LKH-3 CVRPTW tour file → list of 0-indexed customer routes.

    LKH-3 adds V depot copies for V vehicles, so DIMENSION = n_customers + V.
    Depot copies have node index 1 or > n_customers + 1.
    """
    if not os.path.exists(tour_path):
        return []
    n_cust_nodes = n_customers + 1  # depot=1, customers=2..n_customers+1
    routes: List[List[int]] = []
    current: List[int] = []
    in_tour = False
    with open(tour_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                in_tour = True
                continue
            if not in_tour:
                continue
            if line in ("-1", "EOF"):
                break
            node = int(line)
            is_depot = (node == 1 or node > n_cust_nodes)
            if is_depot:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(node - 1)  # 0-indexed: customers become 1..n_customers
    if current:
        routes.append(current)
    return [r for r in routes if r]


# ============================================================
# Main solver
# ============================================================

def _solve_once(
    inst: dict,
    tt: List[List[int]],
    time_limit_sec: int,
    zone_routes: Optional[List[List[int]]],
    n_vehicles_max: Optional[int],
    runs: int,
    max_trials: int,
    mtsp_objective: Optional[str] = None,
) -> Dict:
    """Run LKH-3 once and parse the resulting tour (if one is written).

    Returns status="no_solution" (routes=[]) if LKH-3 wrote no tour file at
    all. For CVRPTW, LKH-3 (WriteTour.c) only writes a tour file when its
    best solution has zero time-window penalty, unless MTSP_OBJECTIVE is
    set -- see solve_vrptw_lkh for the retry tiers that handle this.
    """
    n_vehicles = n_vehicles_max if n_vehicles_max is not None else inst["vehicle_limit"]

    # DISTANCE = max allowed route duration for LKH-3.
    # For rain TT the values are larger than base; depot time windows still enforce
    # the real deadline, so we just set DISTANCE generously (3x depot close) to
    # avoid blocking LKH-3 from finding any route at all.
    depot_close    = int(inst["tw_close"][0])
    distance_limit = depot_close * 3

    with tempfile.TemporaryDirectory() as tmp_win:
        vrp_win  = os.path.join(tmp_win, "problem.vrp")
        par_win  = os.path.join(tmp_win, "problem.par")
        tour_win = os.path.join(tmp_win, "problem.tour")

        vrp_wsl  = win_to_wsl(vrp_win)
        par_wsl  = win_to_wsl(par_win)
        tour_wsl = win_to_wsl(tour_win)

        _write_vrp_file(inst, tt, vrp_win, n_vehicles, zone_routes,
                        distance_limit=distance_limit)
        _write_par_file(vrp_wsl, tour_wsl, par_win, time_limit_sec, runs, max_trials,
                        mtsp_objective=mtsp_objective)

        t0 = time.time()
        ok = False
        try:
            proc = subprocess.run(
                ["wsl", "-d", "Ubuntu", "-u", "root", "--",
                 WSL_LKH_BIN, par_wsl],
                timeout=time_limit_sec + 30,
                capture_output=True,
                text=True,
            )
            ok = proc.returncode == 0
            if not ok:
                print(f"  [LKH-3 error] returncode={proc.returncode}")
        except subprocess.TimeoutExpired:
            ok = False
        elapsed = time.time() - t0

        routes = _parse_tour(tour_win, inst["n_customers"]) if ok and os.path.exists(tour_win) else []

    if not routes:
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

    # Evaluate routes
    coords = inst["coords"]
    depot = 0
    total_distance = 0.0
    total_late = 0.0
    served = 0
    served_on_time = 0
    served_late_count = 0
    vehicles_used = sum(1 for r in routes if r)

    for route in routes:
        if not route:
            continue
        path = [depot] + route + [depot]
        cur_time = 0.0
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            # TSPLIB/Solomon standard: nint(Euclidean) — matches LKH-3 internal distances
            travel = round(math.sqrt((coords[src][0] - coords[dst][0])**2 +
                                     (coords[src][1] - coords[dst][1])**2))
            total_distance += travel
            cur_time += travel
            if dst != depot:
                tw_start = float(inst["tw_open"][dst])
                tw_end   = float(inst["tw_close"][dst])
                service  = float(inst["service_time"][dst])
                cur_time = max(cur_time, tw_start)
                late = max(0.0, cur_time - tw_end)
                total_late += late
                if late > 0:
                    served_late_count += 1
                else:
                    served_on_time += 1
                served += 1
                cur_time += service

    return {
        "status": "feasible",
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


def solve_vrptw_lkh(
    inst: dict,
    tt: List[List[int]],
    time_limit_sec: int = 60,
    zone_routes: Optional[List[List[int]]] = None,
    n_vehicles_max: Optional[int] = None,
    runs: int = 1,
    max_trials: int = 10000,
) -> Dict:
    """Solve VRPTW using LKH-3 via WSL.

    LKH-3 (WriteTour.c) only writes a tour file for CVRPTW when its best
    solution has zero time-window penalty. If the default-budget solve
    below comes back with no tour at all, retry at 3x the time budget; if
    that still fails, retry once more at 3x with MTSP_OBJECTIVE=MINSUM,
    which only relaxes LKH-3's tour-writing gate (verified to not change
    the search itself) so its best-effort (penalty>0) solution can still
    be reported.

    status: "feasible"          - default budget, zero TW penalty
            "feasible_extended" - 3x budget needed, zero TW penalty
            "best_effort"       - 3x budget + MINSUM, residual TW penalty>0
            "no_solution"       - LKH-3 produced no tour at all
    """
    result = _solve_once(inst, tt, time_limit_sec, zone_routes, n_vehicles_max, runs, max_trials)
    if result["status"] != "no_solution":
        return result

    result = _solve_once(inst, tt, time_limit_sec * 3, zone_routes, n_vehicles_max, runs, max_trials)
    if result["status"] != "no_solution":
        result["status"] = "feasible_extended"
        return result

    result = _solve_once(inst, tt, time_limit_sec * 3, zone_routes, n_vehicles_max, runs, max_trials,
                          mtsp_objective="MINSUM")
    if result["routes"]:
        result["status"] = "best_effort"
    return result


# ============================================================
# Rain re-evaluation  (same logic as R_ortools.py)
# ============================================================

def evaluate_routes_with_rain(
    routes: List[List[int]],
    inst: dict,
    rain_events: List[Tuple[float, float, float, List[int]]],
) -> Dict:
    coords = inst["coords"]
    depot = 0
    total_distance = 0.0
    total_late = 0.0
    late_count = 0
    served_count = 0

    def edge_travel(src, dst, cur_time):
        # TSPLIB/Solomon standard: nint(Euclidean) as base, then multiply by event
        base = round(math.sqrt((coords[src][0] - coords[dst][0])**2 +
                               (coords[src][1] - coords[dst][1])**2))
        mult = 1.0
        for trigger, duration, m, nodes in rain_events:
            if cur_time >= trigger and cur_time < trigger + duration:
                if src in set(nodes) and dst in set(nodes):
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
                cur_time = max(cur_time, float(inst["tw_open"][dst]))
                late = max(0.0, cur_time - float(inst["tw_close"][dst]))
                if late > 0:
                    total_late += late
                    late_count += 1
                else:
                    served_count += 1
                cur_time += float(inst["service_time"][dst])

    return {
        "total_distance": total_distance,
        "total_late": total_late,
        "served_count": served_count,
        "late_count": late_count,
        # aliases expected by generate_events.py
        "served_on_time": served_count,
        "n_late_stops": late_count,
    }


# ============================================================
# Pretty print / plot  (same as R_ortools.py)
# ============================================================

def print_result(result: Dict, inst: dict, label: str = "") -> None:
    n = inst["n_customers"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Status          : {result['status']}")
    print(f"  Served on-time  : {result['served_on_time']:3d} / {n}")
    print(f"  TW violation    : {result['served_late']:3d} / {n}")
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
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not found - skipping")
        return

    coords = inst["coords"]
    depot  = coords[0]
    n_cust = inst["n_customers"]
    used_routes = [r for r in result["routes"] if r]
    n_used = len(used_routes)
    cmap = plt.colormaps["tab20" if n_used <= 20 else "hsv"].resampled(max(n_used, 1))

    _, ax = plt.subplots(figsize=(12, 10))
    for idx, route in enumerate(used_routes):
        color = cmap(idx)
        full = [0] + route + [0]
        for k in range(len(full) - 1):
            x0, y0 = coords[full[k]]
            x1, y1 = coords[full[k + 1]]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.9))
        xs = [coords[n][0] for n in full]
        ys = [coords[n][1] for n in full]
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.7)
        for n in route:
            ax.scatter(coords[n][0], coords[n][1], color=color, s=25, zorder=3)
            ax.text(coords[n][0] + 0.4, coords[n][1] + 0.4, str(n),
                    fontsize=6, color=color)

    visited = {n for r in used_routes for n in r}
    unvisited = [n for n in range(1, n_cust + 1) if n not in visited]
    if unvisited:
        ax.scatter([coords[n][0] for n in unvisited],
                   [coords[n][1] for n in unvisited],
                   color="red", s=50, zorder=4, marker="x",
                   label=f"Unvisited ({len(unvisited)})")

    ax.scatter(depot[0], depot[1], marker="*", s=300, color="black", zorder=5, label="Depot")
    ax.set_title(
        f"{label}\nserved={result['served']}/{n_cust}  "
        f"veh={result['vehicles_used']}/{inst['vehicle_limit']}  "
        f"dist={result['total_distance']:.1f}  late={result['total_late']:.1f}",
        fontsize=11,
    )
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    import matplotlib.pyplot as plt2
    plt2.tight_layout()
    plt2.savefig(save_path, dpi=150)
    plt2.close()
    print(f"  [Plot] saved -> {save_path}")


# ============================================================
# Summary printers
# ============================================================

def _print_summary(results: List[Tuple[str, Dict]], inst: dict, title: str) -> None:
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


def _print_rain_eval(
    results: List[Tuple[str, Dict]],
    inst: dict,
    rain_events: List,
    title: str,
) -> None:
    n = inst["n_customers"]
    print(f"\n{'='*80}")
    print(f"  {title}  (time-dependent rain re-simulation)")
    print(f"{'='*80}")
    print(f"  {'Experiment':<32} {'Veh':>5} {'OnTime':>7} {'TWViol':>7} "
          f"{'Unserved':>9} {'Dist':>8} {'Late':>8}")
    print(f"  {'-'*32} {'-'*5} {'-'*7} {'-'*7} {'-'*9} {'-'*8} {'-'*8}")
    for label, r in results:
        if not r["routes"]:
            print(f"  {label:<32} {'N/A':>5} {'N/A':>7} {'N/A':>7} {'N/A':>9}")
            continue
        ev = evaluate_routes_with_rain(r["routes"], inst, rain_events)
        unserved = n - ev["served_count"] - ev["late_count"]
        print(f"  {label:<32} "
              f"{r['vehicles_used']:>2}/{inst['vehicle_limit']:<3} "
              f"{ev['served_count']:>4}/{n:<3} "
              f"{ev['late_count']:>4}/{n:<3} "
              f"{unserved:>6}/{n:<3} "
              f"{ev['total_distance']:>8.1f} "
              f"{ev['total_late']:>8.2f}")


# ============================================================
# Main
# ============================================================

_C1_BASE   = [f"c10{i}" for i in range(1, 10)]           # c101-c109
_C2_BASE   = [f"c20{i}" for i in range(1, 9)]            # c201-c208
_EVENTS    = ["rain_A", "rain_B", "acc_A", "acc_B"]


def run_batch(
    base_instances: List[str],
    solomon_dir: str,
    time_limit: int = 60,
    runs: int = 1,
    result_path: Optional[str] = None,
) -> None:
    """Run LKH-3 on all base instances + their event variants.

    Vehicle count strategy:
    1. Solve base instance (no events) first  -> K_base
    2. unknown: base TT,   cap = K_base
    3. known:   event TT,  cap = K_base
    All three share the same vehicle pool -> fair comparison.
    """
    import os as _os

    header = (f"{'Scenario':<22} {'Type':<8} {'Veh':>4} {'OnTime':>7} "
              f"{'TWViol':>7} {'Unserved':>9} {'Dist':>8} {'Late':>8} {'T':>6}")
    sep = "-" * len(header)
    rows: List[str] = [f"# LKH-3 batch  time_limit={time_limit}s  runs={runs}",
                       header, sep]

    def _row(scenario, label, r, n, rain_events=None):
        if rain_events:
            sim      = evaluate_routes_with_rain(r["routes"], _load_inst(scenario), rain_events)
            on_time  = sim["served_count"]
            tw_viol  = sim["late_count"]
            unserved = n - on_time - tw_viol
            dist     = sim["total_distance"]
            late     = sim["total_late"]
        else:
            on_time  = r["served_on_time"]
            tw_viol  = r["served_late"]
            unserved = r["unserved"]
            dist     = r["total_distance"]
            late     = r["total_late"]
        return (f"  {scenario:<22} {label:<8} {r['vehicles_used']:>4} "
                f"{on_time:>4}/{n:<3} {tw_viol:>4}/{n:<3} {unserved:>6}/{n:<3} "
                f"{dist:>8.1f} {late:>8.2f} {r['elapsed']:>5.1f}s")

    def _load_inst(scenario):
        return load_solomon(_os.path.join(solomon_dir, scenario + ".txt"))

    print(f"\n[Batch] {len(base_instances)} base instances  time_limit={time_limit}s\n")
    print(header); print(sep)

    for base in base_instances:
        base_path = _os.path.join(solomon_dir, base + ".txt")
        if not _os.path.exists(base_path):
            rows.append(f"  {base:<22} SKIP (file not found)"); print(rows[-1])
            rows.append(sep); continue

        # Base instance (no events)
        inst_base = load_solomon(base_path)
        tt_base   = build_tt_matrix(inst_base)
        n         = inst_base["n_customers"]
        r_base    = solve_vrptw_lkh(inst_base, tt_base, time_limit_sec=time_limit, runs=runs)
        row = _row(base, "base   ", r_base, n)
        rows.append(row); print(row)

        # Event variants: known only (static rain TT, no vehicle cap)
        for ev in _EVENTS:
            scenario  = f"{base}_{ev}"
            ev_path   = _os.path.join(solomon_dir, scenario + ".txt")
            if not _os.path.exists(ev_path):
                rows.append(f"  {scenario:<22} SKIP (file not found)"); print(rows[-1])
                continue

            inst_ev     = load_solomon(ev_path)
            rain_events = parse_rain_events(ev_path)
            tt_ev_event = build_rain_tt_matrix(inst_ev, rain_events)
            n_ev        = inst_ev["n_customers"]

            r_kno = solve_vrptw_lkh(inst_ev, tt_ev_event, time_limit_sec=time_limit,
                                     runs=runs)
            row_kno = _row(scenario, "known  ", r_kno, n_ev, rain_events)
            rows.append(row_kno); print(row_kno)

        rows.append(sep); print(sep)

    if result_path:
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"\n[Result] saved -> {result_path}")


if __name__ == "__main__":
    import argparse
    import os as _os

    parser = argparse.ArgumentParser(description="LKH-3 VRPTW solver (WSL backend)")
    parser.add_argument("scenario", type=str, nargs="?", default=None,
                        help="Single dataset name (e.g. c101). Omit for --batch mode.")
    parser.add_argument("--batch",      choices=["C1", "C2", "all"],
                        help="Batch mode: run all C1 / C2 / both instance families")
    parser.add_argument("--data-dir",   default=None)
    parser.add_argument("--time-limit", type=int, default=60,
                        help="LKH-3 time limit per run in seconds (default: 60)")
    parser.add_argument("--runs",       type=int, default=1,
                        help="LKH-3 number of independent runs (default: 1)")
    parser.add_argument("--zone-mode",
                        choices=["all", "solution", "global"],
                        default="all")
    parser.add_argument("--result-dir", default="results",
                        help="Directory to save batch results (default: results/)")
    args = parser.parse_args()

    # ----- Batch mode -----
    if args.batch:
        _SOLOMON_DIR = args.data_dir or _os.path.join(_ROOT, "data", "Solomon")
        if args.batch == "C1":
            instances = _C1_BASE
        elif args.batch == "C2":
            instances = _C2_BASE
        else:
            instances = _C1_BASE + _C2_BASE
        result_path = None
        if args.result_dir:
            _os.makedirs(args.result_dir, exist_ok=True)
            result_path = _os.path.join(args.result_dir, f"lkh3_batch_{args.batch}.txt")
        run_batch(instances, _SOLOMON_DIR, args.time_limit, args.runs, result_path)
        sys.exit(0)

    DATA_FILE  = args.scenario
    TIME_LIMIT = args.time_limit
    ZONE_MODE  = args.zone_mode

    _is_event     = "_rain_" in DATA_FILE or "_acc_" in DATA_FILE
    INSTANCE      = DATA_FILE.split("_rain_")[0].split("_acc_")[0] if _is_event else DATA_FILE
    RAIN_SCENARIO = DATA_FILE if _is_event else None

    _BASE_DIR    = _os.path.dirname(_os.path.abspath(__file__))
    _SOLOMON_DIR = args.data_dir if args.data_dir else _os.path.join(_ROOT, "data", "Solomon")
    data_path    = _os.path.join(_SOLOMON_DIR, DATA_FILE + ".txt")
    sol_path     = _os.path.join(_SOLOMON_DIR, INSTANCE + "_sol.txt")
    _PLOT_DIR    = _os.path.join(_BASE_DIR, "plots", INSTANCE)

    inst      = load_solomon(data_path)
    tt_base   = build_tt_matrix(inst)
    sol_zones = parse_solution_zones(sol_path)

    print(f"[Instance]  {inst['name']}  customers={inst['n_customers']}  "
          f"vehicles={inst['vehicle_limit']}  capacity={inst['vehicle_capacity']}")
    print(f"[ZoneMode]  {ZONE_MODE}")
    print(f"[LKH-3]     time_limit={TIME_LIMIT}s  runs={args.runs}")

    rain_events: List[Tuple[float, float, float, List[int]]] = []
    if RAIN_SCENARIO:
        rain_path   = _os.path.join(_SOLOMON_DIR, RAIN_SCENARIO + ".txt")
        rain_events = parse_rain_events(rain_path)
        print(f"[Rain]      {RAIN_SCENARIO}  events={len(rain_events)}")

    def _run_one(label: str, tt: List[List[int]], plot_name: str,
                 n_vehicles_max: Optional[int] = None) -> Tuple[str, Dict]:
        cap_str = f"  cap={n_vehicles_max}" if n_vehicles_max is not None else ""
        print(f"  solving: {label}{cap_str} ...", end=" ", flush=True)
        r = solve_vrptw_lkh(inst, tt, time_limit_sec=TIME_LIMIT,
                             n_vehicles_max=n_vehicles_max, runs=args.runs)
        print(f"{r['vehicles_used']} veh  t={r['elapsed']:.1f}s")
        plot_result(r, inst, f"{label} -{INSTANCE}",
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

        if (run_all or ZONE_MODE == "solution") and sol_zones is not None:
            label = "LKH-3 solution zones"
            res.append(_run_one(label, tt, f"{plot_prefix}F_solution.png", _nveh(label)))
        elif ZONE_MODE == "solution" and sol_zones is None:
            print(f"[Warning] solution file not found: {sol_path}")

        if run_all or ZONE_MODE == "global":
            label = "LKH-3 global"
            res.append(_run_one(label, tt, f"{plot_prefix}G_global.png", _nveh(label)))

        return res

    if RAIN_SCENARIO:
        # ---- Known: plan with static rain TT, evaluate under actual event TT ----
        tt_rain = build_rain_tt_matrix(inst, rain_events)
        known_results = _run_experiments(tt_rain, plot_prefix="known_")
        _print_rain_eval(known_results, inst, rain_events,
                         "KNOWN - planned with static event TT, re-simulated")
    else:
        # ---- Base: no events ----
        base_results = _run_experiments(tt_base, plot_prefix="base_")
        _print_summary(base_results, inst, "BASE - no event")
