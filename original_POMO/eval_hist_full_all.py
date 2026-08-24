"""
eval_hist_full_all.py

hist_full baseline evaluation across ALL C1 / RC1 / R1 instances.

hist_full: plan using TT averaged from training-set scenarios
           where event multipliers are applied at FULL coverage (duration=T),
           i.e. the planner assumes events could last the entire horizon.

For each benchmark group the historical avg TT is built from its training instances:
  C1  → C102-C109  (same node coords as C101-C109)
  RC1 → RC102-RC108
  R1  → R102-R112

Each test instance (all Cxx/RCxx/Rxx in the group) is then:
  1. Planned once with the group's hist_full TT via OR-Tools
  2. Simulated on Base / Rain-A / Rain-B / Acc-A / Acc-B scenarios

Output: per-instance table  K  D  Lc  Lt  and BKS comparison for Base.
"""
from __future__ import annotations

import math, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from ortools_vrptw import load_solomon_raw, solve, resimulate_with_events, build_tt

DATA_DIR = os.path.join(_ROOT, "data", "Solomon")
AVG_DIR  = os.path.join(DATA_DIR, "historical_avg")

SCENARIOS = ["base", "rain_A", "rain_B", "acc_A", "acc_B"]
SCEN_LABEL = {"base": "Base", "rain_A": "Rain-A", "rain_B": "Rain-B",
              "acc_A": "Acc-A",  "acc_B":  "Acc-B"}

# ── benchmark groups ───────────────────────────────────────────────────────────
GROUPS = {
    "C1": dict(
        train=["c102","c103","c104","c105","c106","c107","c108","c109"],
        test= ["c101","c102","c103","c104","c105","c106","c107","c108","c109"],
    ),
    "RC1": dict(
        train=["rc102","rc103","rc104","rc105","rc106","rc107","rc108"],
        test= ["rc101","rc102","rc103","rc104","rc105","rc106","rc107","rc108"],
    ),
    "R1": dict(
        train=["r102","r103","r104","r105","r106","r107","r108","r109","r110","r111","r112"],
        test= ["r101","r102","r103","r104","r105","r106","r107","r108",
               "r109","r110","r111","r112"],
    ),
}


# ── TT helpers ─────────────────────────────────────────────────────────────────

def build_tt_full(inst: dict) -> np.ndarray:
    """hist_full TT: event multiplier applied at full coverage (whole horizon). Vectorised."""
    coords = np.array(inst["coords"], dtype=np.float64)   # (n, 2)
    diff = coords[:, None, :] - coords[None, :, :]        # (n, n, 2)
    base = np.sqrt((diff ** 2).sum(-1))                    # (n, n)
    mult = np.ones_like(base)

    for ev in inst.get("preset_rain", []):
        nodes = np.array(ev["nodes"], dtype=int)
        mask = np.zeros(base.shape, dtype=bool)
        mask[np.ix_(nodes, nodes)] = True
        mult = np.where(mask, np.maximum(mult, ev["multiplier"]), mult)

    for ev in inst.get("preset_acc", []):
        nodes = ev["nodes"]
        if len(nodes) >= 2:
            i, j = nodes[0], nodes[1]
            mult[i, j] = max(mult[i, j], ev["multiplier"])
            mult[j, i] = max(mult[j, i], ev["multiplier"])

    return base * mult


def compute_hist_full_tt(train_names: list) -> list:
    """Average hist_full TT matrix over all training scenarios."""
    matrices = []
    for name in train_names:
        # base scenario (no events → same as Euclidean)
        base_path = os.path.join(DATA_DIR, f"{name}.txt")
        inst_base = load_solomon_raw(base_path)
        matrices.append(build_tt_full(inst_base))   # base has no events → Euclidean
        # event scenarios
        for scen in ["rain_A", "rain_B", "acc_A", "acc_B"]:
            spath = os.path.join(DATA_DIR, f"{name}_{scen}.txt")
            if os.path.isfile(spath):
                inst_ev = load_solomon_raw(spath)
                matrices.append(build_tt_full(inst_ev))

    avg = np.mean(matrices, axis=0)
    n = avg.shape[0]
    return [[int(round(avg[i, j])) for j in range(n)] for i in range(n)]


def tt_to_list(npy: np.ndarray) -> list:
    n = npy.shape[0]
    return [[int(round(npy[i, j])) for j in range(n)] for i in range(n)]


# ── BKS loader ─────────────────────────────────────────────────────────────────

def load_bks(name: str) -> dict | None:
    sol_path = os.path.join(DATA_DIR, f"{name}_sol.txt")
    if not os.path.isfile(sol_path):
        return None
    with open(sol_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    routes = []
    for ln in lines:
        if ln.startswith("Route"):
            # Format: "Route  1 : 81 78 76 ..."
            colon_idx = ln.find(":")
            customers = []
            for x in (ln[colon_idx+1:] if colon_idx >= 0 else ln).split():
                try:
                    customers.append(int(x))
                except ValueError:
                    pass
            if customers:
                routes.append(customers)

    base_path = os.path.join(DATA_DIR, f"{name}.txt")
    inst = load_solomon_raw(base_path)
    result = resimulate_with_events(routes, inst)
    result["vehicles_used"] = len([r for r in routes if r])
    return result


# ── simulate fixed plan on a scenario ─────────────────────────────────────────

def simulate(routes: list, scen_name: str, inst_name: str) -> dict | None:
    if scen_name == "base":
        spath = os.path.join(DATA_DIR, f"{inst_name}.txt")
    else:
        spath = os.path.join(DATA_DIR, f"{inst_name}_{scen_name}.txt")
    if not os.path.isfile(spath):
        return None
    inst = load_solomon_raw(spath)
    result = resimulate_with_events(routes, inst)
    result["vehicles_used"] = sum(1 for r in routes if r)
    return result


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    for group_name, cfg in GROUPS.items():
        print(f"\n{'='*80}")
        print(f"  GROUP: {group_name}")
        print(f"{'='*80}")

        # ── build hist_full TT ──────────────────────────────────────────────
        # Check if C1 has precomputed M_all_avg we can compare with
        print(f"  Computing hist_full TT from training: {cfg['train']}")
        tt_hist = compute_hist_full_tt(cfg["train"])
        print(f"  Done. Matrix {len(tt_hist)}×{len(tt_hist[0])}")

        # ── header ──────────────────────────────────────────────────────────
        hdr = (f"{'Instance':<10} {'Scenario':<9} "
               f"{'K':>4} {'D':>8} {'Lc':>5} {'Lt':>8}  "
               f"{'BKS_K':>6} {'BKS_D':>8}")
        print(f"\n{hdr}")
        print("-" * 70)

        for inst_name in cfg["test"]:
            # Plan once with hist_full TT
            base_path = os.path.join(DATA_DIR, f"{inst_name}.txt")
            inst_base = load_solomon_raw(base_path)

            plan_result = solve(inst_base, tt_hist, time_limit_sec=60)
            routes = plan_result["routes"]
            print(f"  [{inst_name}] OR-Tools: {plan_result['status']}  "
                  f"K={plan_result['vehicles_used']}  served={plan_result['served']}"
                  f"  ({plan_result['elapsed']:.1f}s)")

            # BKS
            bks = load_bks(inst_name)
            bks_k = bks["vehicles_used"] if bks else "-"
            bks_d = f"{bks['total_distance']:.1f}" if bks else "-"

            # Simulate on all scenarios
            for scen in SCENARIOS:
                m = simulate(routes, scen, inst_name)
                if m is None:
                    continue
                K   = m["vehicles_used"]
                D   = m["total_distance"]
                Lc  = m["n_late_stops"]
                Lt  = m["total_late"]
                lbl = SCEN_LABEL[scen]
                bks_info = f"{bks_k:>6} {bks_d:>8}" if scen == "base" else f"{'':>6} {'':>8}"
                print(f"  {inst_name:<10} {lbl:<9} "
                      f"{K:>4d} {D:>8.1f} {Lc:>5d} {Lt:>8.1f}  {bks_info}")

            print()

    print("\nDone.")


if __name__ == "__main__":
    main()
