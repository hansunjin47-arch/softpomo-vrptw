"""
run_lkh_eval_60s.py
LKH-3 evaluation at 60s time limit: no-info + known for C1/RC1/R1.
Overwrites soft_pomo/results/lkh/lkh_eval_{c101,rc101,r101}.txt
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "clustering_routing"))

from R_lkh import (
    load_solomon, build_tt_matrix, build_rain_tt_matrix,
    solve_vrptw_lkh, parse_rain_events, evaluate_routes_with_rain,
)

SOLOMON_DIR = os.path.join(_ROOT, "data", "Solomon")
OUT_DIR     = os.path.join(_HERE, "results", "lkh")
TIME_LIMIT  = 60
RUNS        = 1
EVENTS      = ["rain_A", "rain_B", "acc_A", "acc_B"]

BENCHMARKS = [
    ("C1",  "c101"),
    ("RC1", "rc101"),
    ("R1",  "r101"),
]


def fmt_row(scenario, plan, veh, on_time, late, n, dist, late_min):
    return (
        f"  {scenario:<26} {plan:<9} {on_time:>4}/{n:<3} {late:>5}"
        f"         0 {veh:>5} {dist:>9.2f} {late_min:>8.1f}"
    )


def run_eval(label, base):
    base_path = os.path.join(SOLOMON_DIR, base + ".txt")
    inst_base = load_solomon(base_path)
    tt_base   = build_tt_matrix(inst_base)
    n         = inst_base["n_customers"]

    header = (
        f"  {'Scenario':<26} {'Plan':<9} {'OnTime':>7} {'Late':>5}"
        f" {'Unserved':>9} {'Veh':>5} {'Dist':>9} {'LateMin':>8}"
    )
    sep   = "  " + "-" * 82
    sep_s = "  " + "." * 82
    rows  = [
        "-" * 90,
        f"  LKH-3 Evaluation  [{label}]  (time_limit={TIME_LIMIT}s/solve)",
        "  " + "-" * 88,
        header,
        sep,
    ]

    print(f"\n=== {label} — solving base ({base}) ===")
    r_base = solve_vrptw_lkh(inst_base, tt_base, time_limit_sec=TIME_LIMIT, runs=RUNS)
    row = fmt_row(
        f"{base} (base)", "no-info",
        r_base["vehicles_used"], r_base["served_on_time"], r_base["served_late"],
        n, r_base["total_distance"], r_base["total_late"],
    )
    rows.append(row); print(row)
    rows.append(sep_s)

    for ev in EVENTS:
        scenario = f"{base}_{ev}"
        ev_path  = os.path.join(SOLOMON_DIR, scenario + ".txt")
        if not os.path.exists(ev_path):
            print(f"  SKIP {scenario} (not found)")
            continue

        inst_ev     = load_solomon(ev_path)
        rain_events = parse_rain_events(ev_path)
        tt_ev       = build_rain_tt_matrix(inst_ev, rain_events)
        n_ev        = inst_ev["n_customers"]

        print(f"  {scenario} — no-info ...", flush=True)
        # no-info: execute base plan under event conditions
        sim_noi = evaluate_routes_with_rain(r_base["routes"], inst_ev, rain_events)
        row_noi = fmt_row(
            scenario, "no-info",
            r_base["vehicles_used"],
            sim_noi["served_count"], sim_noi["late_count"],
            n_ev, sim_noi["total_distance"], sim_noi["total_late"],
        )
        rows.append(row_noi); print(row_noi)

        print(f"  {scenario} — known   ...", flush=True)
        # known: plan with event TT, execute under event conditions
        r_kno   = solve_vrptw_lkh(inst_ev, tt_ev, time_limit_sec=TIME_LIMIT, runs=RUNS)
        sim_kno = evaluate_routes_with_rain(r_kno["routes"], inst_ev, rain_events)
        row_kno = fmt_row(
            scenario, "known",
            r_kno["vehicles_used"],
            sim_kno["served_count"], sim_kno["late_count"],
            n_ev, sim_kno["total_distance"], sim_kno["total_late"],
        )
        rows.append(row_kno); print(row_kno)

    rows.append(sep)

    out_path = os.path.join(OUT_DIR, f"lkh_eval_{base}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for label, base in BENCHMARKS:
        run_eval(label, base)
    print("\n=== All done ===")
