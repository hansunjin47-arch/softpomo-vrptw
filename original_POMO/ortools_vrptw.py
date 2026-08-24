"""
ortools_vrptw.py -- OR-Tools VRPTW baseline for Solomon instances.

Two runs per call:
  [1] No events     -- solve with base Euclidean travel times
  [2] Events known  -- solve with increased TT (static full-horizon approximation)
      Rain:     both endpoints in rain zone → multiply edge
      Accident: affected edge pair → multiply

OR-Tools is a static solver with full up-front knowledge of the TT matrix.
It gives an upper bound on what is achievable with perfect prior information.

Usage
-----
  python ortools_vrptw.py c101                          # no events
  python ortools_vrptw.py c101_rain_A                   # rain scenario
  python ortools_vrptw.py c101_acc_A                    # accident scenario
  python ortools_vrptw.py c101 --data-dir data/Solomon --time-limit 60
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_HERE)
DEFAULT_DATA = os.path.join(_ROOT, "data", "Solomon")

_SEVERITY_TO_MULT = {"low": 2.0, "medium": 3.5, "high": 5.0}


# ── Solomon raw loader (un-normalised, for OR-Tools) ───────────────────────────

def load_solomon_raw(path: str) -> dict:
    """Parse Solomon .txt (with optional EVENTS section).
    Returns raw arrays in original units (minutes, units of demand, etc.).
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    name = lines[0]
    vehicle_limit = vehicle_capacity = None

    vi = next(i for i, l in enumerate(lines) if l.upper().startswith("VEHICLE"))
    for off in (1, 2):
        parts = lines[vi + off].split()
        try:
            vehicle_limit = int(parts[0]); vehicle_capacity = float(parts[1]); break
        except (ValueError, IndexError):
            continue

    ci    = next(i for i, l in enumerate(lines) if l.upper().startswith("CUST"))
    ev_i  = next((i for i, l in enumerate(lines) if l.upper() == "EVENTS"), len(lines))
    rows  = []
    for ln in lines[ci + 1 : ev_i]:
        parts = ln.split()
        if len(parts) >= 7:
            try:
                rows.append([float(x) for x in parts[:7]])
            except ValueError:
                continue

    data      = np.array(rows, dtype=np.float64)
    T         = float(data[0, 5])          # depot tw_close = episode horizon
    coords    = [(float(r[1]), float(r[2])) for r in data]
    demands   = [float(r[3]) for r in data]
    tw_open   = [float(r[4]) for r in data]
    tw_close  = [float(r[5]) for r in data]
    service   = [float(r[6]) for r in data]

    # Parse preset events
    preset_rain = []
    preset_acc  = []
    if ev_i < len(lines):
        for ln in lines[ev_i + 1 :]:
            toks = ln.split()
            if not toks or toks[0].startswith('#'):
                continue
            kw = toks[0].upper()
            if kw == "RAIN" and len(toks) >= 6:
                preset_rain.append(dict(
                    trigger_time = float(toks[1]),
                    duration     = float(toks[2]),
                    multiplier   = float(toks[3]),
                    rainfall_mm  = float(toks[4]),
                    nodes        = [int(x) for x in toks[5:]],
                ))
            elif kw == "ACCIDENT" and len(toks) >= 5:
                sev  = toks[3].lower()
                mult = _SEVERITY_TO_MULT.get(sev, float(sev) if sev.replace('.','').isdigit() else 2.0)
                preset_acc.append(dict(
                    trigger_time = float(toks[1]),
                    duration     = float(toks[2]),
                    multiplier   = mult,
                    nodes        = [int(x) for x in toks[4:]],
                ))

    return dict(
        name=name, n_customers=len(data) - 1,
        T=T, vehicle_limit=vehicle_limit, vehicle_capacity=vehicle_capacity,
        coords=coords, demands=demands, tw_open=tw_open, tw_close=tw_close,
        service_time=service,
        preset_rain=preset_rain, preset_acc=preset_acc,
    )


# ── Travel-time matrix builders ────────────────────────────────────────────────

def build_tt(inst: dict) -> List[List[int]]:
    """Plain Euclidean tt matrix (OR-Tools requires integers)."""
    coords = inst["coords"]
    n = len(coords)
    return [[
        int(round(math.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)))
        for j in range(n)] for i in range(n)]


def build_tt_rain(inst: dict) -> List[List[int]]:
    """Rain-adjusted tt matrix for OR-Tools (static solver).

    Multiplier scaled by the fraction of episode the rain is active:
      effective_mult = 1 + (mult - 1) * (duration / T)
    Rain_B (duration=5 out of T=1236) → nearly no effect.
    Rain_A (duration≈T)               → full multiplier applied.
    """
    coords = inst["coords"]
    n      = len(coords)
    T      = max(inst["T"], 1.0)
    tt = []
    for i in range(n):
        row = []
        for j in range(n):
            base = math.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)
            mult = 1.0
            for ev in inst["preset_rain"]:
                if i in ev["nodes"] and j in ev["nodes"]:
                    coverage = min(ev["duration"], T) / T
                    eff_mult = 1.0 + (ev["multiplier"] - 1.0) * coverage
                    mult = max(mult, eff_mult)
            row.append(int(round(base * mult)))
        tt.append(row)
    return tt


def build_tt_accident(inst: dict) -> List[List[int]]:
    """Accident-adjusted tt matrix for OR-Tools (static solver).

    Effective multiplier scales with accident duration relative to episode horizon,
    matching the rain approach so OR-Tools sees a realistic cost for long accidents:
      eff_mult = 1 + (mult - 1) * (duration / T)
    A short accident (duration=60 out of T=1236) → small effect.
    A long accident  (duration≈T)                → full multiplier applied.
    """
    coords = inst["coords"]
    n      = len(coords)
    T      = max(inst["T"], 1.0)
    tt = []
    for i in range(n):
        row = []
        for j in range(n):
            base = math.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)
            mult = 1.0
            for ev in inst["preset_acc"]:
                nodes = ev["nodes"]
                if len(nodes) >= 2 and {i, j} == {nodes[0], nodes[1]}:
                    coverage = min(ev["duration"], T) / T
                    eff_mult = 1.0 + (ev["multiplier"] - 1.0) * coverage
                    mult = max(mult, eff_mult)
            row.append(int(round(base * mult)))
        tt.append(row)
    return tt


# ── OR-Tools solver ────────────────────────────────────────────────────────────

def solve(inst: dict, tt: List[List[int]], time_limit_sec: int = 30) -> dict:
    """Solve VRPTW with given tt matrix. No zone restrictions."""
    n_nodes   = len(inst["coords"])
    n_veh     = inst["vehicle_limit"]
    depot     = 0

    manager = pywrapcp.RoutingIndexManager(n_nodes, n_veh, depot)
    routing = pywrapcp.RoutingModel(manager)

    def travel_cb(fi, ti):
        return tt[manager.IndexToNode(fi)][manager.IndexToNode(ti)]

    def time_cb(fi, ti):
        fn = manager.IndexToNode(fi)
        tn = manager.IndexToNode(ti)
        return tt[fn][tn] + int(inst["service_time"][fn])

    def demand_cb(fi):
        return int(inst["demands"][manager.IndexToNode(fi)])

    tc_idx = routing.RegisterTransitCallback(travel_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tc_idx)

    tcb_idx = routing.RegisterTransitCallback(time_cb)
    horizon = int(inst["tw_close"][0])
    routing.AddDimension(tcb_idx, horizon, horizon, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(n_nodes):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(int(inst["tw_open"][node]), int(inst["tw_close"][node]))

    dc_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        dc_idx, 0, [int(inst["vehicle_capacity"])] * n_veh, True, "Cap")

    penalty = 100_000
    for node in range(1, n_nodes):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy  = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_sec
    params.log_search = False

    t0  = time.time()
    sol = routing.SolveWithParameters(params)
    elapsed = time.time() - t0

    if not sol:
        return dict(status="no_solution", routes=[], served=0, served_on_time=0,
                    served_late=0, unserved=inst["n_customers"],
                    total_distance=0.0, total_late=0.0, n_late_stops=0,
                    vehicles_used=0, elapsed=elapsed)

    routes       = []
    vehicles_used = 0
    for veh in range(n_veh):
        if routing.IsVehicleUsed(sol, veh):
            vehicles_used += 1
        route = []
        idx = routing.Start(veh)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != depot:
                route.append(node)
            idx = sol.Value(routing.NextVar(idx))
        routes.append(route)

    # Re-simulate with base Euclidean tt for accurate distance/lateness reporting
    coords = inst["coords"]
    total_dist  = 0.0
    total_late  = 0.0
    n_late      = 0
    on_time     = 0
    served      = 0

    for route in routes:
        if not route:
            continue
        path     = [depot] + route + [depot]
        cur_time = 0.0
        for k in range(len(path) - 1):
            src, dst = path[k], path[k + 1]
            dist = math.sqrt((coords[src][0]-coords[dst][0])**2 +
                             (coords[src][1]-coords[dst][1])**2)
            total_dist += dist
            cur_time   += dist
            if dst != depot:
                served += 1
                cur_time = max(cur_time, inst["tw_open"][dst])
                late     = max(0.0, cur_time - inst["tw_close"][dst])
                total_late += late
                if late > 0:
                    n_late += 1
                else:
                    on_time += 1
                cur_time += inst["service_time"][dst]

    return dict(
        status        = "optimal" if routing.status() == 1 else "feasible",
        routes        = routes,
        served        = served,
        served_on_time= on_time,
        served_late   = n_late,
        n_late_stops  = n_late,
        unserved      = inst["n_customers"] - served,
        total_distance= total_dist,
        total_late    = total_late,
        vehicles_used = vehicles_used,
        elapsed       = elapsed,
    )


# ── Dynamic re-simulation ─────────────────────────────────────────────────────

def resimulate_with_events(routes: List[List[int]], inst: dict) -> dict:
    """Re-simulate routes with time-dependent event TT.

    Rain:     edge slowed if BOTH endpoints in zone AND rain active at departure time.
    Accident: affected edge pair slowed if accident active at departure time.
    """
    coords = inst["coords"]
    depot  = 0

    def edge_mult(src: int, dst: int, t: float) -> float:
        mult = 1.0
        for ev in inst["preset_rain"]:
            if ev["trigger_time"] <= t < ev["trigger_time"] + ev["duration"]:
                if src in ev["nodes"] and dst in ev["nodes"]:
                    mult = max(mult, ev["multiplier"])
        for ev in inst["preset_acc"]:
            if ev["trigger_time"] <= t < ev["trigger_time"] + ev["duration"]:
                nodes = ev["nodes"]
                if len(nodes) >= 2 and {src, dst} == {nodes[0], nodes[1]}:
                    mult = max(mult, ev["multiplier"])
        return mult

    total_dist = 0.0
    total_late = 0.0
    n_late     = 0
    on_time    = 0

    for route in routes:
        if not route:
            continue
        path     = [depot] + route + [depot]
        cur_time = 0.0
        for k in range(len(path) - 1):
            src, dst = path[k], path[k + 1]
            base = math.sqrt((coords[src][0] - coords[dst][0]) ** 2 +
                             (coords[src][1] - coords[dst][1]) ** 2)
            dist      = base * edge_mult(src, dst, cur_time)
            total_dist += dist
            cur_time   += dist
            if dst != depot:
                cur_time = max(cur_time, inst["tw_open"][dst])
                late     = max(0.0, cur_time - inst["tw_close"][dst])
                total_late += late
                if late > 0:
                    n_late  += 1
                else:
                    on_time += 1
                cur_time += inst["service_time"][dst]

    return dict(total_distance=total_dist, total_late=total_late,
                n_late_stops=n_late, served_on_time=on_time)


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_result(r: dict, inst: dict, label: str = "") -> None:
    N = inst["n_customers"]
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Status          : {r['status']}")
    print(f"  Vehicles used   : {r['vehicles_used']} / {inst['vehicle_limit']}")
    print(f"  Served on-time  : {r['served_on_time']:3d} / {N}")
    print(f"  Unserved (drop) : {r['unserved']:3d} / {N}")
    print(f"  late_stops      : {r['served_late']:3d} / {N}")
    print(f"  total_late      : {r['total_late']:.2f}")
    print(f"  dist            : {r['total_distance']:.2f}")
    print(f"  Solve time      : {r['elapsed']:.1f}s")


def save_solution(r: dict, inst: dict, path: str, label: str = "") -> None:
    """Save routes + per-stop timing to a text file."""
    coords   = inst["coords"]
    demands  = inst["demands"]
    tw_open  = inst["tw_open"]
    tw_close = inst["tw_close"]
    service  = inst["service_time"]
    depot    = 0
    N        = inst["n_customers"]

    lines = []
    lines.append("=" * 75)
    lines.append(f"  SOLUTION: {inst['name']}  [{label}]")
    lines.append("=" * 75)
    lines.append(f"  vehicles={r['vehicles_used']}  served={r['served_on_time']+r['served_late']}/{N}"
                 f"  dist={r['total_distance']:.2f}  late_stops={r['n_late_stops']}"
                 f"  total_late={r['total_late']:.2f}")
    lines.append("")
    lines.append("  VISIT ORDER")
    lines.append("-" * 75)
    for vi, route in enumerate(r["routes"], 1):
        if not route:
            continue
        load  = sum(demands[n] for n in route)
        dist  = sum(math.sqrt((coords[route[k]][0]-coords[route[k-1] if k > 0 else depot][0])**2 +
                              (coords[route[k]][1]-coords[route[k-1] if k > 0 else depot][1])**2)
                   for k in range(len(route)))
        dist += math.sqrt((coords[depot][0]-coords[route[-1]][0])**2 +
                          (coords[depot][1]-coords[route[-1]][1])**2)
        seq = " -> ".join([f"{n}" for n in [depot] + route + [depot]])
        lines.append(f"  V{vi:02d}: {seq}  [dist={dist:.1f}, load={load:.0f}/{inst['vehicle_capacity']:.0f}]")

    lines.append("")
    lines.append("  DETAILED TIMING")
    lines.append("-" * 75)
    for vi, route in enumerate(r["routes"], 1):
        if not route:
            continue
        rpath = [depot] + route + [depot]
        load  = sum(demands[n] for n in route)
        dist  = sum(math.sqrt((coords[rpath[k]][0]-coords[rpath[k+1]][0])**2 +
                              (coords[rpath[k]][1]-coords[rpath[k+1]][1])**2)
                   for k in range(len(rpath)-1))
        lines.append(f"\n  -- Vehicle {vi:2d}  [{len(route):3d} stops | dist={dist:7.1f}"
                     f" | load={load:.0f}/{inst['vehicle_capacity']:.0f}]")
        cur_time = 0.0
        prev = depot
        for n in route:
            d = math.sqrt((coords[prev][0]-coords[n][0])**2 + (coords[prev][1]-coords[n][1])**2)
            arrival = cur_time + d
            wait    = max(0.0, tw_open[n] - arrival)
            svc_start = max(arrival, tw_open[n])
            depart  = svc_start + service[n]
            late_tag = " LATE" if arrival > tw_close[n] + 1e-6 else ""
            lines.append(f"  {prev:5d} -> {n:3d}  ({coords[n][0]:5.1f},{coords[n][1]:5.1f})"
                         f"  arr={arrival:7.1f}  wait={wait:5.1f}  dep={depart:7.1f}"
                         f"  TW=[{tw_open[n]:6.1f},{tw_close[n]:6.1f}]{late_tag}"
                         f"  dem={demands[n]:3.0f}")
            cur_time = depart
            prev = n
        ret_dist = math.sqrt((coords[prev][0]-coords[depot][0])**2 +
                             (coords[prev][1]-coords[depot][1])**2)
        lines.append(f"  {prev:5d} ->   0  depot  dist={ret_dist:.1f}  arr={cur_time+ret_dist:.1f}/{inst['T']:.0f}")

    lines.append("\n" + "=" * 75)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [Saved] {path}")


def plot_routes(r: dict, inst: dict, path: str, title: str = "") -> None:
    """Save a route visualisation to path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not available — skipped")
        return

    coords = inst["coords"]
    depot  = coords[0]
    colors = plt.cm.tab10.colors

    _, ax = plt.subplots(figsize=(12, 10))
    for vi, route in enumerate(r["routes"]):
        if not route:
            continue
        color = colors[vi % len(colors)]
        path_ = [0] + route + [0]
        xs = [coords[n][0] for n in path_]
        ys = [coords[n][1] for n in path_]
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.8)
        for k in range(len(path_) - 1):
            x0, y0 = coords[path_[k]]
            x1, y1 = coords[path_[k+1]]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
        for n in route:
            ax.scatter(coords[n][0], coords[n][1], color=color, s=20, zorder=3)
            ax.text(coords[n][0]+0.3, coords[n][1]+0.3, str(n), fontsize=6, color=color)

    ax.scatter(depot[0], depot[1], marker="*", s=300, color="black", zorder=5, label="Depot")
    ax.set_title(f"{inst['name']} — {title}\nveh={r['vehicles_used']}  "
                 f"dist={r['total_distance']:.1f}  late={r['n_late_stops']}", fontsize=11)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OR-Tools VRPTW baseline")
    parser.add_argument("scenario",
                        help="Instance or scenario name (e.g. c101, c101_rain_A, c101_acc_A)")
    parser.add_argument("--data-dir",   default=DEFAULT_DATA,
                        help="Directory containing Solomon .txt files")
    parser.add_argument("--time-limit", type=int, default=30,
                        help="OR-Tools time limit in seconds (default: 30)")
    parser.add_argument("--no-events",  action="store_true",
                        help="Ignore events even if present in file (base tt only)")
    parser.add_argument("--out-dir",    default=None,
                        help="Output directory for plots/solutions (default: result/ortools/<scenario>)")
    args = parser.parse_args()

    data_path = os.path.join(args.data_dir, args.scenario + ".txt")
    if not os.path.isfile(data_path):
        print(f"[ERROR] File not found: {data_path}")
        return

    inst = load_solomon_raw(data_path)
    N    = inst["n_customers"]
    T    = inst["T"]
    has_rain = bool(inst["preset_rain"]) and not args.no_events
    has_acc  = bool(inst["preset_acc"])  and not args.no_events

    print(f"[Instance]  {inst['name']}  N={N}  T={T:.0f}"
          f"  vehicles={inst['vehicle_limit']}  cap={inst['vehicle_capacity']:.0f}")
    print(f"[Events]    rain={len(inst['preset_rain'])}  accident={len(inst['preset_acc'])}"
          + ("  [ignored: --no-events]" if args.no_events else ""))
    print(f"[TimeLimit] {args.time_limit}s\n")

    out_dir = args.out_dir or os.path.join(_HERE, "result", "ortools", args.scenario)
    os.makedirs(out_dir, exist_ok=True)

    N = inst["n_customers"]

    # ── [1] Solve with base TT → re-simulate with actual event effects ─────────
    # Represents: OR-Tools had no prior event info, routes encounter events mid-run
    tt_base = build_tt(inst)
    r_base  = solve(inst, tt_base, args.time_limit)

    if (has_rain or has_acc) and r_base["routes"]:
        ev1 = resimulate_with_events(r_base["routes"], inst)
        r1  = dict(
            status        = r_base["status"],
            routes        = r_base["routes"],
            vehicles_used = r_base["vehicles_used"],
            served_on_time= ev1["served_on_time"],
            served_late   = ev1["n_late_stops"],
            n_late_stops  = ev1["n_late_stops"],
            unserved      = r_base["unserved"],
            total_distance= ev1["total_distance"],
            total_late    = ev1["total_late"],
            elapsed       = r_base["elapsed"],
        )
        print_result(r1, inst, label="[1] Solved without event info (events hit mid-route)")
    else:
        r1 = r_base
        print_result(r_base, inst, label="[1] No events (base TT)")

    # ── [2] Solve with event TT known upfront (perfect information) ────────────
    r2 = None
    if has_rain:
        tt_ev = build_tt_rain(inst)
        r2    = solve(inst, tt_ev, args.time_limit)
        print_result(r2, inst, label="[2] Rain TT known upfront (perfect info)")
    elif has_acc:
        tt_ev = build_tt_accident(inst)
        r2    = solve(inst, tt_ev, args.time_limit)
        print_result(r2, inst, label="[2] Accident TT known upfront (perfect info)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  Summary  [{args.scenario}]")
    print(f"{'='*75}")
    print(f"  {'Experiment':<38} {'Veh':>4} {'OnTime':>7} {'LateCnt':>8} {'LateMag':>8} {'Dist':>9}")
    print(f"  {'-'*38} {'-'*4} {'-'*7} {'-'*8} {'-'*8} {'-'*9}")

    def _row(label, r):
        print(f"  {label:<38} "
              f"{r['vehicles_used']:>2}/{inst['vehicle_limit']:<2} "
              f"{r['served_on_time']:>4}/{N:<3} "
              f"{r['served_late']:>5}/{N:<3} "
              f"{r['total_late']:>8.2f} "
              f"{r['total_distance']:>9.1f}")

    if has_rain or has_acc:
        _row("No prior event info (re-simulated)", r1)
    else:
        _row("No events", r1)
    if r2 is not None:
        label = "Rain known upfront" if has_rain else "Accident known upfront"
        _row(label, r2)
    print()

    # ── Save plots + solution files ───────────────────────────────────────────
    label1 = "no_event_info" if (has_rain or has_acc) else "base"
    save_solution(r1, inst, os.path.join(out_dir, f"solution_{label1}.txt"),
                  label="No prior event info" if (has_rain or has_acc) else "Base TT")
    plot_routes(r1, inst, os.path.join(out_dir, f"route_{label1}.png"),
                title="No prior event info" if (has_rain or has_acc) else "Base TT")

    if r2 is not None:
        label2 = "rain_known" if has_rain else "acc_known"
        lbl2   = "Rain known upfront" if has_rain else "Accident known upfront"
        save_solution(r2, inst, os.path.join(out_dir, f"solution_{label2}.txt"), label=lbl2)
        plot_routes(r2, inst, os.path.join(out_dir, f"route_{label2}.png"), title=lbl2)

    print(f"\n  Output directory: {out_dir}")


if __name__ == "__main__":
    main()
