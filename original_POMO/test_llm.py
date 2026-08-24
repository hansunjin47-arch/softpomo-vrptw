"""
test_llm.py — Standalone LLM top-K pruning test (no RL training).

Builds the prompt, calls LLM, shows selected top-K, and compares
with optimal solution starting customers.

Usage:
  python test_llm.py                          # c101, top_k=25
  python test_llm.py --instance c101_rain_A   # rain scenario
  python test_llm.py --top-k 10
  python test_llm.py --no-ontology            # LLM+RL ablation
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

from vrptw_env import load_solomon
from VRPTWOntology import VRPTWOntology
from VRPTWLLMModule import (
    get_start_nodes, build_start_nodes_prompt,
    DEFAULT_MODEL,
)


def load_starts_from_sol(sol_path: str) -> list[int] | None:
    """Load first customer of each route from a sol file."""
    if not os.path.isfile(sol_path):
        return None
    starts = []
    with open(sol_path) as f:
        for line in f:
            if line.startswith("Route"):
                stops = list(map(int, line.split(":")[1].strip().split()))
                if stops:
                    starts.append(stops[0])
    return starts if starts else None


def load_optimal_starts(data_dir: str, inst_name: str) -> tuple[list[int] | None, str]:
    """
    Load comparison starts based on instance type:
    - base / rain_B: use base optimal sol file
    - rain_A: use OR-Tools rain-known solution (true TT optimal)
    Returns (starts, label).
    """
    base_name = inst_name.split("_")[0]

    if inst_name.endswith("_rain_A"):
        # OR-Tools solution with rain TT known upfront
        ortools_path = os.path.join(
            _HERE, "result", "ortools", inst_name, "solution_rain_known.txt")
        starts = _parse_ortools_solution(ortools_path)
        if starts:
            return starts, "OR-Tools (rain TT known)"

    # Fallback: base optimal sol file
    sol_path = os.path.join(data_dir, f"{base_name}_sol.txt")
    starts = load_starts_from_sol(sol_path)
    label  = "base optimal" if not inst_name.endswith("_rain_A") else "base optimal (rain sol unavailable)"
    return starts, label


def _parse_ortools_solution(path: str) -> list[int] | None:
    """Parse first stop from each route in OR-Tools solution_rain_known.txt."""
    if not os.path.isfile(path):
        return None
    starts = []
    with open(path) as f:
        for line in f:
            # Format: V01: 0 -> 81 -> 78 -> ...
            if line.strip().startswith("V") and "->" in line:
                parts = line.split("->")
                if len(parts) >= 2:
                    # first node after depot (0)
                    for p in parts[1:]:
                        val = p.strip().split()[0]
                        if val.isdigit() and int(val) != 0:
                            starts.append(int(val))
                            break
    return starts if starts else None


def run_instance(inst_name: str, data_dir: str, top_k: int,
                 model: str, use_ont: bool, show_prompt: bool,
                 use_cot: bool = True) -> dict:
    inst_path = os.path.join(data_dir, inst_name + ".txt")
    inst = load_solomon(inst_path)
    N, T = inst["n_customers"], float(inst["T"])

    rain_evs = [e for e in inst.get("preset_events", []) if e["type"] == "RAIN"]
    acc_evs  = [e for e in inst.get("preset_events", []) if e["type"] == "ACCIDENT"]
    event_tag = ""
    if rain_evs: event_tag += f"  RAIN×{len(rain_evs)}"
    if acc_evs:  event_tag += f"  ACC×{len(acc_evs)}(skip—mid-episode)"

    print(f"\n{'='*65}")
    print(f"[Instance] {inst['name']}  N={N}  T={T:.0f}  ontology={use_ont}{event_tag}", flush=True)

    ont     = VRPTWOntology(inst)
    all_c   = list(range(1, N + 1))
    ont_ctx = ont.get_context(all_c) if use_ont else {}

    if show_prompt:
        prompt = build_start_nodes_prompt(inst, all_c, ont_ctx, top_k,
                                          use_cot=True, use_ontology=use_ont)
        print("\n" + "="*70 + "\nPROMPT:\n" + "="*70)
        print(prompt)
        print("="*70 + "\n")

    if acc_evs and not rain_evs:
        print("  [SKIP] Accident-only scenario — accident triggers mid-episode, cannot test standalone.")
        return {"instance": inst_name, "selected": [], "skipped": True}

    print("Calling LLM...", flush=True)
    selected = get_start_nodes(inst, top_k, model,
                               use_cot=use_cot, use_ontology=use_ont)
    print(f"  Selected: {selected}")

    # Save to disk cache (same format as train_vrptw_llm.py)
    cache_dir  = os.path.join(_HERE, "result_llm", "llm_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{inst['name']}.json")
    cache_data = json.load(open(cache_path)) if os.path.isfile(cache_path) else {}
    cache_data['start_nodes'] = selected
    json.dump(cache_data, open(cache_path, 'w'))
    print(f"  [Cache] saved → {cache_path}", flush=True)

    optimal_starts, opt_label = load_optimal_starts(data_dir, inst_name)
    result = {"instance": inst_name, "selected": selected,
              "optimal_starts": optimal_starts, "coverage": None,
              "opt_label": opt_label}

    if optimal_starts:
        overlap  = sorted(set(selected) & set(optimal_starts))
        coverage = len(overlap) / len(optimal_starts)
        result["coverage"] = coverage
        print(f"  [{opt_label}] starts: {sorted(optimal_starts)}")
        print(f"  Coverage: {len(overlap)}/{len(optimal_starts)} ({coverage*100:.0f}%)")
        print(f"  Matched: {overlap}")
        missing = sorted(set(optimal_starts) - set(selected))
        if missing:
            print(f"  Missing: {missing}")

        # Position analysis (per-stop detail)
        base_name = inst_name.split("_")[0]
        sol_path = os.path.join(data_dir, f"{base_name}_sol.txt")
        if os.path.isfile(sol_path):
            routes = []
            with open(sol_path) as f:
                for line in f:
                    if line.startswith("Route"):
                        routes.append(list(map(int, line.split(":")[1].strip().split())))

            positions = {"START": 0, "mid": 0, "end": 0}
            print(f"\n  [Position in optimal routes]")
            print(f"  {'Stop':>4}  {'Route':>5}  {'Pos':>8}  {'Tag':>5}  "
                  f"{'TW_open':>7}  {'wait':>6}  {'TW_slack':>8}")
            print(f"  {'-'*4}  {'-'*5}  {'-'*8}  {'-'*5}  "
                  f"{'-'*7}  {'-'*6}  {'-'*8}")
            for c in selected:
                tw_o  = float(ont.tw_open[c])
                tw_c  = float(ont.tw_close[c])
                dist  = float(ont.tt[0, c])
                wait  = max(0.0, tw_o - dist)
                slack = tw_c - max(dist, tw_o)
                found = False
                for i, route in enumerate(routes):
                    if c in route:
                        pos = route.index(c) + 1
                        tag = "START" if pos == 1 else ("end" if pos == len(route) else "mid")
                        positions[tag] += 1
                        print(f"  {c:>4}  R{i+1:>4}  {pos:>3}/{len(route):<4}  "
                              f"{tag:>5}  {tw_o:>7.0f}  {wait:>6.0f}  {slack:>8.0f}")
                        found = True
                        break
                if not found:
                    print(f"  {c:>4}  {'?':>5}  {'?':>8}  {'?':>5}  "
                          f"{tw_o:>7.0f}  {wait:>6.0f}  {slack:>8.0f}")

            print(f"\n  Aggregate: START={positions['START']}  "
                  f"mid={positions['mid']}  end={positions['end']}")
            result["positions"] = positions

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance",   default=None,
                        help="Single instance name (e.g. c101)")
    _BASE_ALL = [f"c{i:03d}" for i in range(101, 110)]
    _DEFAULT  = (_BASE_ALL
                 + [f"{n}_rain_A" for n in _BASE_ALL]
                 + [f"{n}_rain_B" for n in _BASE_ALL])
    parser.add_argument("--instances",  nargs="+",
                        default=_DEFAULT,
                        help="Multiple instances (default: c101-c109 + rain_A/B)")
    parser.add_argument("--data-dir",   default=os.path.join(_ROOT, "data", "Solomon"))
    parser.add_argument("--top-k",      type=int, default=25)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--no-ontology",action="store_true")
    parser.add_argument("--show-prompt",action="store_true")
    parser.add_argument("--no-cot",     action="store_true",
                        help="Disable chain-of-thought for faster calls")
    args = parser.parse_args()

    instances = [args.instance] if args.instance else args.instances
    use_ont   = not args.no_ontology
    data_dir  = args.data_dir

    results = []
    for inst_name in instances:
        r = run_instance(inst_name, data_dir, args.top_k,
                         args.model, use_ont, args.show_prompt,
                         use_cot=not args.no_cot)
        results.append(r)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*65}")
        print(f"  SUMMARY  top_k={args.top_k}  ontology={use_ont}")
        print(f"{'='*65}")
        print(f"  {'Instance':<15}  {'Coverage':>10}  START  mid  end")
        print(f"  {'-'*15}  {'-'*10}  {'-'*5}  {'-'*3}  {'-'*3}")
        coverages = []
        for r in results:
            if r.get("skipped"):
                print(f"  {r['instance']:<15}  {'SKIPPED':>10}")
                continue
            cov  = r.get("coverage")
            pos  = r.get("positions", {})
            cov_str = f"{cov*100:.0f}%" if cov is not None else "N/A"
            if cov is not None: coverages.append(cov)
            print(f"  {r['instance']:<15}  {cov_str:>10}"
                  f"  {pos.get('START',0):>5}  {pos.get('mid',0):>3}  {pos.get('end',0):>3}")
        if coverages:
            print(f"\n  Avg coverage: {sum(coverages)/len(coverages)*100:.1f}%")

    # old single-instance path removed — everything goes through run_instance
    inst_path = None  # suppress unused warning


if __name__ == "__main__":
    main()
