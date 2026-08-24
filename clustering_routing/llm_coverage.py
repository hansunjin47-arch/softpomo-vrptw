"""Compare LLM confidence scores with OR-Tools optimal visit order for c101 base."""
import sys, os, pickle, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'original_POMO'))

from ortools_vrptw import load_solomon_raw, build_tt, solve

DATA   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')
RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# Find latest cache
import glob
caches = sorted(glob.glob(os.path.join(RESULT, '*', 'llm_conf_cache_qwen3_32b.pkl')),
                key=os.path.getmtime)
if not caches:
    print("No LLM cache found")
    sys.exit(1)
cache_path = caches[-1]
print(f"Using cache: {cache_path}")

with open(cache_path, 'rb') as f:
    llm_conf_cache = pickle.load(f)

if 'c101' not in llm_conf_cache:
    print("c101 not in cache")
    sys.exit(1)

# LLM scores: {zone_idx: {stop: score}}
c101_scores = llm_conf_cache['c101']
flat_scores = {c: s for zs in c101_scores.values() for c, s in zs.items()}

# OR-Tools optimal solution
inst = load_solomon_raw(os.path.join(DATA, 'c101.txt'))
tt   = build_tt(inst)
r    = solve(inst, tt, time_limit_sec=30)
ort_routes = [route for route in r['routes'] if route]

print(f"\n{'='*70}")
print(f"  LLM Confidence vs OR-Tools Optimal Visit Order (c101 base)")
print(f"  Zones: {len(c101_scores)}  |  Stops scored: {len(flat_scores)}")
print(f"{'='*70}")

# Per-zone analysis
from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from R_utils import TrainConfig, infer_solution_path, parse_solution_routes, build_zone_plan_from_routes

inst_rl = load_waste_benchmark_txt(os.path.join(DATA, 'c101.txt'))
tt_rl   = build_travel_time_matrix(inst_rl)
cfg     = TrainConfig(data_path='')
sol     = infer_solution_path(os.path.join(DATA, 'c101.txt'), None)
zp      = build_zone_plan_from_routes(inst_rl, parse_solution_routes(sol), cfg)

# Map each stop to its OR-Tools visit rank within its vehicle route
ort_rank = {}  # stop -> (vehicle_idx, rank_in_route)
for vi, route in enumerate(ort_routes):
    for rank, stop in enumerate(route):
        ort_rank[stop] = (vi+1, rank+1)

print(f"\n  {'Zone':>4}  {'Stop':>4}  {'LLM score':>10}  {'LLM rank in zone':>16}  {'ORT vehicle':>11}  {'ORT rank':>8}  {'Correlation?'}")
print(f"  {'-'*80}")

zone_correlations = []
for z_idx, z_stops in enumerate(zp.zones):
    z_scores = c101_scores.get(z_idx, {})
    if not z_scores:
        continue

    # LLM ranking (high score = visit first = rank 1)
    llm_ranked = sorted(z_scores.items(), key=lambda x: -x[1])
    llm_rank_map = {stop: rank+1 for rank, (stop, _) in enumerate(llm_ranked)}

    # Find which OR-Tools vehicle services this zone
    ort_vehicle = None
    for vi, route in enumerate(ort_routes):
        if set(route) & set(z_stops):
            ort_vehicle = vi+1
            ort_route = route
            break

    if ort_vehicle is None:
        continue

    # Correlation: does LLM rank match ORT visit order?
    # ORT rank within the zone's vehicle route
    ort_zone_rank = {stop: i+1 for i, stop in enumerate(ort_route) if stop in set(z_stops)}

    # Compute Spearman rank correlation for this zone
    common = [s for s in z_stops if s in llm_rank_map and s in ort_zone_rank]
    if len(common) >= 2:
        n = len(common)
        # d^2 for Spearman
        d2 = sum((llm_rank_map[s] - ort_zone_rank[s])**2 for s in common)
        spearman = 1 - (6*d2) / (n*(n*n-1))
        zone_correlations.append(spearman)
        corr_str = f"rho={spearman:+.2f}"
    else:
        corr_str = "n/a"

    print(f"\n  Zone {z_idx} ({len(z_stops)} stops, ORT V{ort_vehicle})  {corr_str}")
    print(f"  {'Stop':>4}  {'Score':>6}  {'LLM#':>5}  {'ORT#':>5}  {'Agree':>5}")
    print(f"  {'-'*30}")
    for stop, score in llm_ranked:
        lr = llm_rank_map[stop]
        or_ = ort_zone_rank.get(stop, '?')
        agree = 'OK' if isinstance(or_, int) and abs(lr - or_) <= 2 else ''
        print(f"  {stop:>4}  {score:>6.2f}  {lr:>5}  {str(or_):>5}  {agree}")

if zone_correlations:
    avg_corr = sum(zone_correlations) / len(zone_correlations)
    print(f"\n  {'='*50}")
    print(f"  Average Spearman rank correlation (LLM vs ORT): {avg_corr:+.3f}")
    print(f"  (1.0 = perfect match, 0 = random, -1 = inverse)")
    positive = sum(1 for r in zone_correlations if r > 0.3)
    print(f"  Zones with rho > 0.3: {positive}/{len(zone_correlations)}")
