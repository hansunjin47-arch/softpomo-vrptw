"""Simulate routing if we follow LLM confidence order vs OR-Tools optimal."""
import sys, os, pickle, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'original_POMO'))

from ortools_vrptw import load_solomon_raw, build_tt, solve
from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from R_utils import TrainConfig, infer_solution_path, parse_solution_routes, build_zone_plan_from_routes

DATA   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')
RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

import glob
caches = sorted(glob.glob(os.path.join(RESULT, '*', 'llm_conf_cache_qwen3_32b.pkl')), key=os.path.getmtime)
cache_path = caches[-1]
print(f"Cache: {cache_path}\n")

with open(cache_path, 'rb') as f:
    llm_conf_cache = pickle.load(f)

c101_scores = llm_conf_cache['c101']   # {zone_idx: {stop: score}}

# OR-Tools solution
inst_ort = load_solomon_raw(os.path.join(DATA, 'c101.txt'))
tt_ort   = build_tt(inst_ort)
r_ort    = solve(inst_ort, tt_ort, time_limit_sec=30)
ort_routes = [route for route in r_ort['routes'] if route]

# Zone plan
inst_rl = load_waste_benchmark_txt(os.path.join(DATA, 'c101.txt'))
tt_rl   = build_travel_time_matrix(inst_rl)
cfg     = TrainConfig(data_path='')
sol     = infer_solution_path(os.path.join(DATA, 'c101.txt'), None)
zp      = build_zone_plan_from_routes(inst_rl, parse_solution_routes(sol), cfg)

coords   = inst_ort['coords']
tw_open  = inst_ort['tw_open']
tw_close = inst_ort['tw_close']
service  = inst_ort['service_time']
T        = inst_ort['T']

def simulate(route, label=""):
    """Simulate a route from depot, return (dist, late_cnt, total_late, on_time)."""
    cur, t = 0, 0.0
    dist, total_late, late_cnt, on_time = 0.0, 0.0, 0, 0
    for node in route:
        d = math.sqrt((coords[cur][0]-coords[node][0])**2 + (coords[cur][1]-coords[node][1])**2)
        arr = t + d
        dist += d
        late = max(0.0, arr - tw_close[node])
        if late > 0:
            late_cnt += 1
            total_late += late
        else:
            on_time += 1
        t = max(arr, tw_open[node]) + service[node]
        cur = node
    return dict(dist=dist, late_cnt=late_cnt, total_late=total_late, on_time=on_time)

sep = "=" * 72
print(sep)
print(f"  c101 base: LLM confidence order vs OR-Tools optimal (per zone)")
print(sep)
print(f"  {'Zone':>4}  {'Stops':>5}  {'':35} {'ORT':>6} {'LLM':>6}")
print(f"  {'':46} {'dist':>6} {'dist':>6}  {'ORT late':>8} {'LLM late':>8}")
print(f"  {'-'*70}")

total_ort_dist = total_llm_dist = 0.0
total_ort_late = total_llm_late = 0
total_ort_late_min = total_llm_late_min = 0.0

for z_idx, z_stops in enumerate(zp.zones):
    z_scores = c101_scores.get(z_idx, {})
    if not z_scores:
        continue

    # Find matching OR-Tools route for this zone
    ort_route = None
    for route in ort_routes:
        if set(route) & set(z_stops):
            ort_route = route
            break
    if not ort_route:
        continue

    # LLM order: sort by confidence descending
    llm_route = [s for s, _ in sorted(z_scores.items(), key=lambda x: -x[1])]

    ort_sim = simulate(ort_route)
    llm_sim = simulate(llm_route)

    total_ort_dist    += ort_sim['dist']
    total_llm_dist    += llm_sim['dist']
    total_ort_late    += ort_sim['late_cnt']
    total_llm_late    += llm_sim['late_cnt']
    total_ort_late_min += ort_sim['total_late']
    total_llm_late_min += llm_sim['total_late']

    diff_dist = llm_sim['dist'] - ort_sim['dist']
    diff_late = llm_sim['late_cnt'] - ort_sim['late_cnt']
    flag = " <-- LLM worse" if diff_late > 0 or diff_dist > 5 else ""

    print(f"  Zone {z_idx:>2}  ({len(z_stops):>2} stops)")
    print(f"    ORT order: {ort_route}")
    print(f"    LLM order: {llm_route}")
    print(f"    ORT: dist={ort_sim['dist']:6.1f}  late_cnt={ort_sim['late_cnt']}  late_min={ort_sim['total_late']:.1f}")
    print(f"    LLM: dist={llm_sim['dist']:6.1f}  late_cnt={llm_sim['late_cnt']}  late_min={llm_sim['total_late']:.1f}  (diff dist={diff_dist:+.1f}  late={diff_late:+d}){flag}")

print(f"\n  {sep}")
print(f"  TOTAL SUMMARY")
print(f"  {'-'*50}")
print(f"  {'':20} {'OR-Tools':>10}  {'LLM order':>10}  {'Diff':>8}")
print(f"  {'Total distance':20} {total_ort_dist:>10.1f}  {total_llm_dist:>10.1f}  {total_llm_dist-total_ort_dist:>+8.1f}")
print(f"  {'Total late stops':20} {total_ort_late:>10}  {total_llm_late:>10}  {total_llm_late-total_ort_late:>+8}")
print(f"  {'Total late (min)':20} {total_ort_late_min:>10.1f}  {total_llm_late_min:>10.1f}  {total_llm_late_min-total_ort_late_min:>+8.1f}")
