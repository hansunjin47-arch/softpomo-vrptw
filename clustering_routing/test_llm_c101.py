"""Quick LLM confidence test on c101 base with CoT=True."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'original_POMO'))

import math
from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from R_utils import TrainConfig, infer_solution_path, parse_solution_routes, build_zone_plan_from_routes
from R_llm_module import query_ollama_instance_confidence
from R_rl_module import BaselineEnv
from R_rl import MAX_SCOPE
from ortools_vrptw import load_solomon_raw, build_tt, solve

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')

# Setup env
inst = load_waste_benchmark_txt(os.path.join(DATA, 'c101.txt'))
tt   = build_travel_time_matrix(inst)
cfg  = TrainConfig(data_path='')
sol  = infer_solution_path(os.path.join(DATA, 'c101.txt'), None)
zp   = build_zone_plan_from_routes(inst, parse_solution_routes(sol), cfg)
env  = BaselineEnv(inst=inst, tt=tt, max_vehicles=zp.n_zones,
    late_count_penalty=cfg.late_count_penalty,
    late_penalty=cfg.late_penalty, unserved_penalty=cfg.unserved_penalty,
    max_scope_size=MAX_SCOPE)
env.set_zone_assignment(zp.customer_to_zone, zp.n_zones, adjacent_zones=zp.adjacent_zones)
env.reset()

# OR-Tools optimal
inst_ort = load_solomon_raw(os.path.join(DATA, 'c101.txt'))
r_ort = solve(inst_ort, build_tt(inst_ort), time_limit_sec=30)
ort_routes = [r for r in r_ort['routes'] if r]
# Map stop → ORT visit rank
ort_rank = {}
for vi, route in enumerate(ort_routes):
    for rank, stop in enumerate(route):
        ort_rank[stop] = (vi+1, rank+1)

print("Calling LLM (use_cot=True)...\n")
import time
t0 = time.time()
zone_scores = query_ollama_instance_confidence(
    env=env, zone_plan=zp, model_name="qwen3:32b", use_cot=True
)
elapsed = time.time() - t0
print(f"\nTotal LLM time: {elapsed:.1f}s\n")

# Compare per zone
sep = "=" * 65
print(sep)
print("  LLM (CoT=True) vs OR-Tools optimal per zone")
print(sep)

import scipy.stats as stats_
for z_idx, z_stops in enumerate(zp.zones):
    z_sc = zone_scores.get(z_idx, {})
    if not z_sc:
        continue

    # Find ORT vehicle/route for this zone
    ort_route = next((r for r in ort_routes if set(r) & set(z_stops)), None)
    if not ort_route:
        continue
    ort_zone_rank = {s: i+1 for i, s in enumerate(ort_route) if s in set(z_stops)}

    llm_ranked = sorted(z_sc.items(), key=lambda x: -x[1])
    llm_rank_map = {s: i+1 for i, (s,_) in enumerate(llm_ranked)}

    # Spearman
    common = [s for s in z_stops if s in llm_rank_map and s in ort_zone_rank]
    if len(common) >= 3:
        llm_r = [llm_rank_map[s] for s in common]
        ort_r = [ort_zone_rank[s] for s in common]
        rho, _ = stats_.spearmanr(llm_r, ort_r)
    else:
        rho = float('nan')

    print(f"\nZone {z_idx} ({len(z_stops)} stops)  Spearman rho={rho:+.2f}")
    print(f"  {'Stop':>4}  {'Score':>6}  {'LLM#':>5}  {'ORT#':>5}")
    for stop, score in llm_ranked:
        lr = llm_rank_map[stop]
        or_ = ort_zone_rank.get(stop, '?')
        print(f"  {stop:>4}  {score:>6.2f}  {lr:>5}  {str(or_):>5}")
