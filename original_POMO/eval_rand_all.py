"""
eval_rand_all.py

hist_avg baseline evaluation with randomly generated event scenarios.

Avg TT is built from TRAINING instances (r102-r112, c102-c109, rc102-rc108)
using their base + rand_rain_001..010 files (10 scenarios per instance).
rand_acc is excluded from avg computation — consistent with current design.

Each TEST instance is then:
  1. Planned once with the group's hist_avg TT via OR-Tools (60 s)
  2. Simulated on: base + rand_rain_001..010 + rand_acc_001..010  (21 scenarios)

Output saved to results/rand_eval_all.log
"""
from __future__ import annotations

import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from ortools_vrptw import load_solomon_raw, solve, resimulate_with_events

DATA_DIR    = os.path.join(_ROOT, "data", "Solomon")
RESULTS_DIR = os.path.join(_HERE, "results")
RAND_COUNT  = 10   # rand_rain / rand_acc scenarios per instance

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
        train=["r102","r103","r104","r105","r106","r107","r108",
               "r109","r110","r111","r112"],
        test= ["r101","r102","r103","r104","r105","r106","r107","r108",
               "r109","r110","r111","r112"],
    ),
}

TEST_SCENARIOS = (
    ["base"]
    + [f"rand_rain_{j:03d}" for j in range(1, RAND_COUNT + 1)]
    + [f"rand_acc_{j:03d}"  for j in range(1, RAND_COUNT + 1)]
)   # 21 scenarios per test instance


# ── TT helpers ─────────────────────────────────────────────────────────────────

def build_tt_full(inst: dict) -> np.ndarray:
    """hist_full TT: apply event multipliers at full coverage (vectorised)."""
    coords = np.array(inst["coords"], dtype=np.float64)
    diff   = coords[:, None, :] - coords[None, :, :]
    base   = np.sqrt((diff ** 2).sum(-1))
    mult   = np.ones_like(base)

    for ev in inst.get("preset_rain", []):
        nodes = np.array(ev["nodes"], dtype=int)
        mask  = np.zeros(base.shape, dtype=bool)
        mask[np.ix_(nodes, nodes)] = True
        mult  = np.where(mask, np.maximum(mult, ev["multiplier"]), mult)

    for ev in inst.get("preset_acc", []):
        nodes = ev["nodes"]
        if len(nodes) >= 2:
            i, j = nodes[0], nodes[1]
            mult[i, j] = max(mult[i, j], ev["multiplier"])
            mult[j, i] = max(mult[j, i], ev["multiplier"])

    return base * mult


def compute_hist_rand_tt(train_names: list) -> list:
    """Average TT over base + rand_rain scenarios for training instances.

    Excludes rand_acc: single-edge events contribute negligibly to the avg
    and are reserved as unseen test scenarios — consistent with A/B design.
    """
    matrices = []
    missing  = 0
    for name in train_names:
        # base (no events)
        base_path = os.path.join(DATA_DIR, f"{name}.txt")
        inst_base = load_solomon_raw(base_path)
        matrices.append(build_tt_full(inst_base))
        # rand_rain scenarios
        for j in range(1, RAND_COUNT + 1):
            spath = os.path.join(DATA_DIR, f"{name}_rand_rain_{j:03d}.txt")
            if os.path.isfile(spath):
                inst_ev = load_solomon_raw(spath)
                matrices.append(build_tt_full(inst_ev))
            else:
                missing += 1

    if missing:
        print(f"  [WARN] {missing} rand_rain files not found — skipped from avg")

    avg = np.mean(matrices, axis=0)
    n   = avg.shape[0]
    base0 = matrices[0]
    mask  = base0 > 0
    pct   = float((avg[mask] / base0[mask] - 1).mean() * 100)
    print(f"  avg TT from {len(matrices)} matrices  (+{pct:.2f}% vs base)")
    return [[int(round(avg[i, j])) for j in range(n)] for i in range(n)]


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
            colon_idx = ln.find(":")
            customers = []
            for x in (ln[colon_idx + 1:] if colon_idx >= 0 else ln).split():
                try:
                    customers.append(int(x))
                except ValueError:
                    pass
            if customers:
                routes.append(customers)
    if not routes:
        return None
    inst   = load_solomon_raw(os.path.join(DATA_DIR, f"{name}.txt"))
    result = resimulate_with_events(routes, inst)
    result["vehicles_used"] = len([r for r in routes if r])
    return result


# ── Simulate fixed plan on a scenario ─────────────────────────────────────────

def simulate(routes: list, scen: str, inst_name: str) -> dict | None:
    if scen == "base":
        spath = os.path.join(DATA_DIR, f"{inst_name}.txt")
    else:
        spath = os.path.join(DATA_DIR, f"{inst_name}_{scen}.txt")
    if not os.path.isfile(spath):
        return None
    inst   = load_solomon_raw(spath)
    result = resimulate_with_events(routes, inst)
    result["vehicles_used"] = sum(1 for r in routes if r)
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "rand_eval_all.log")
    lines    = []

    def p(s=""):
        print(s)
        lines.append(s)

    for group_name, cfg in GROUPS.items():
        p(f"\n{'='*80}")
        p(f"  GROUP: {group_name}")
        p(f"{'='*80}")

        # ── build hist_avg TT from training rand_rain ───────────────────────
        p(f"  Building hist_avg TT from {cfg['train']} × (base + rand_rain×{RAND_COUNT})")
        tt_hist = compute_hist_rand_tt(cfg["train"])
        p(f"  Done. {len(tt_hist)}×{len(tt_hist[0])} matrix\n")

        # ── summary accumulators ────────────────────────────────────────────
        rain_lc_list, acc_lc_list = [], []

        # ── header ─────────────────────────────────────────────────────────
        hdr = (f"{'Instance':<10} {'Scenario':<18} "
               f"{'K':>4} {'D':>8} {'Lc':>5} {'Lt':>8}  "
               f"{'BKS_K':>6} {'BKS_D':>8}")
        p(hdr)
        p("-" * 78)

        for inst_name in cfg["test"]:
            base_path = os.path.join(DATA_DIR, f"{inst_name}.txt")
            inst_base = load_solomon_raw(base_path)

            plan = solve(inst_base, tt_hist, time_limit_sec=60)
            routes = plan["routes"]
            p(f"  [{inst_name}] OR-Tools: {plan['status']}  "
              f"K={plan['vehicles_used']}  served={plan['served']}"
              f"  ({plan['elapsed']:.1f}s)")

            bks   = load_bks(inst_name)
            bks_k = bks["vehicles_used"]  if bks else "-"
            bks_d = f"{bks['total_distance']:.1f}" if bks else "-"

            inst_rain_lc, inst_acc_lc = [], []

            for scen in TEST_SCENARIOS:
                m = simulate(routes, scen, inst_name)
                if m is None:
                    continue
                K   = m["vehicles_used"]
                D   = m["total_distance"]
                Lc  = m["n_late_stops"]
                Lt  = m["total_late"]

                if scen == "base":
                    bks_col = f"  {bks_k:>6} {bks_d:>8}"
                else:
                    bks_col = ""

                p(f"  {inst_name:<10} {scen:<18} "
                  f"{K:>4d} {D:>8.1f} {Lc:>5d} {Lt:>8.1f}{bks_col}")

                if scen.startswith("rand_rain"):
                    inst_rain_lc.append(Lc)
                elif scen.startswith("rand_acc"):
                    inst_acc_lc.append(Lc)

            if inst_rain_lc:
                avg_r = sum(inst_rain_lc) / len(inst_rain_lc)
                p(f"  {inst_name:<10} {'>> rand_rain avg':<18} "
                  f"{'':>4} {'':>8} {avg_r:>5.1f}")
                rain_lc_list.append(avg_r)
            if inst_acc_lc:
                avg_a = sum(inst_acc_lc) / len(inst_acc_lc)
                p(f"  {inst_name:<10} {'>> rand_acc avg':<18} "
                  f"{'':>4} {'':>8} {avg_a:>5.1f}")
                acc_lc_list.append(avg_a)

            p()

        if rain_lc_list:
            p(f"  GROUP {group_name} rand_rain avg Lc across instances: "
              f"{sum(rain_lc_list)/len(rain_lc_list):.2f}")
        if acc_lc_list:
            p(f"  GROUP {group_name} rand_acc  avg Lc across instances: "
              f"{sum(acc_lc_list)/len(acc_lc_list):.2f}")

    p("\nDone.")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Saved] {log_path}")


if __name__ == "__main__":
    main()
