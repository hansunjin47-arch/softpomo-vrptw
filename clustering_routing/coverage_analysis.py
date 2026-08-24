"""Coverage analysis: compare RL solution with OR-Tools optimal on c101 base."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'original_POMO'))

import numpy as np
import torch
import torch.optim as optim

from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from R_utils import TrainConfig, SHARED_HYPERPARAMS, infer_solution_path, parse_solution_routes, build_zone_plan_from_routes
from R_rl_module import BaselineEnv, AttentionActorCritic, load_checkpoint
from R_rl import _collect_episode, MAX_SCOPE
from ortools_vrptw import load_solomon_raw, build_tt, solve

DATA  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')
CKPT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'rl_pomo', 'multi_pomo_last.pkl')

# ── OR-Tools solution ─────────────────────────────────────────────────────
inst_ort = load_solomon_raw(os.path.join(DATA, 'c101.txt'))
tt_ort   = build_tt(inst_ort)
r_ort    = solve(inst_ort, tt_ort, time_limit_sec=30)
N = inst_ort['n_customers']

# Simulate OR-Tools routes with base TT for per-stop timing
def simulate_route_timing(inst, routes, tt_matrix):
    """Return per-stop timing for each route."""
    depot, results = 0, []
    coords  = inst['coords']
    tw_open = inst['tw_open']
    tw_close= inst['tw_close']
    service = inst['service_time']
    for vi, route in enumerate(routes):
        if not route:
            continue
        stops = []
        cur, t = depot, 0.0
        for node in route:
            d = math.sqrt((coords[cur][0]-coords[node][0])**2 + (coords[cur][1]-coords[node][1])**2)
            arr = t + d
            wait = max(0.0, tw_open[node] - arr)
            svc_start = max(arr, tw_open[node])
            depart = svc_start + service[node]
            late = max(0.0, arr - tw_close[node])
            stops.append(dict(node=node, arrival=arr, wait=wait,
                              svc_start=svc_start, depart=depart,
                              tw_open=tw_open[node], tw_close=tw_close[node],
                              service=service[node], late=late))
            t = depart
            cur = node
        results.append(dict(vehicle=vi+1, route=route, stops=stops))
    return results

ort_routes = [r for r in r_ort['routes'] if r]
ort_timing  = simulate_route_timing(inst_ort, ort_routes, tt_ort)

# ── RL solution ──────────────────────────────────────────────────────────
hp  = dict(SHARED_HYPERPARAMS)
cfg = TrainConfig(data_path='')
path = os.path.join(DATA, 'c101.txt')
inst = load_waste_benchmark_txt(path)
tt   = build_travel_time_matrix(inst)
sol  = infer_solution_path(path, None)
zp   = build_zone_plan_from_routes(inst, parse_solution_routes(sol), cfg)
env  = BaselineEnv(inst=inst, tt=tt, max_vehicles=zp.n_zones,
    late_count_penalty=cfg.late_count_penalty,
    late_penalty=cfg.late_penalty, unserved_penalty=cfg.unserved_penalty,
    max_scope_size=MAX_SCOPE)
env.set_zone_assignment(zp.customer_to_zone, zp.n_zones, adjacent_zones=zp.adjacent_zones)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = AttentionActorCritic(
    stop_feat_dim=env.stop_feat_dim, max_scope_size=MAX_SCOPE,
    vehicle_feat_dim=env.vehicle_feat_dim, global_feat_dim=env.global_feat_dim,
    embed_dim=hp['embed_dim'], n_heads=hp['n_heads'], n_encoder_layers=hp['n_encoder_layers'],
).to(device)
opt = optim.Adam(model.parameters())
load_checkpoint(CKPT, model, opt, ontology_engine=None, device=device)

_, lc, sc, _, _, _ = _collect_episode(env, zp, cfg, model, device, MAX_SCOPE, 10000, greedy=True)

rl_routes  = [v.route_nodes[1:-1] for v in env.vehicles if v.activated and len(v.route_nodes) > 2]
rl_timing  = simulate_route_timing(inst_ort, rl_routes, tt_ort)

# ── Comparison ────────────────────────────────────────────────────────────
sep = "=" * 72

print(f"\n{sep}")
print(f"  COVERAGE ANALYSIS: c101 base  (OR-Tools vs RL)")
print(f"{sep}")
print(f"  {'':32} {'OR-Tools':>10}  {'RL':>10}")
print(f"  {'-'*52}")
print(f"  {'Served (on-time)':32} {r_ort['served_on_time']:>10}  {sc - lc:>10}")
print(f"  {'Served (late)':32} {r_ort['served_late']:>10}  {lc:>10}")
print(f"  {'Unserved':32} {r_ort['unserved']:>10}  {inst.n_customers - sc:>10}")
print(f"  {'Total distance':32} {r_ort['total_distance']:>10.1f}  {env.total_distance:>10.1f}")
print(f"  {'Vehicles used':32} {r_ort['vehicles_used']:>10}  {env.vehicles_used:>10}")
print(f"  {'Total late (min)':32} {r_ort['total_late']:>10.1f}  {env.total_late:>10.1f}")

# Check stop grouping match
ort_set = {frozenset(r) for r in ort_routes}
rl_set  = {frozenset(r) for r in rl_routes}
print(f"\n  Same zone groupings: {ort_set == rl_set}")

# Per-vehicle detail
print(f"\n{sep}")
print(f"  PER-VEHICLE ROUTE COMPARISON")
print(f"{sep}")

# Match RL routes to OR-Tools routes by stop set
rl_matched = {}
for rl_v in rl_timing:
    rl_set_v = frozenset(rl_v['route'])
    for ort_v in ort_timing:
        if frozenset(ort_v['route']) == rl_set_v:
            rl_matched[ort_v['vehicle']] = rl_v
            break

for ort_v in ort_timing:
    vi = ort_v['vehicle']
    rl_v = rl_matched.get(vi)
    ort_dist = sum(
        math.sqrt((inst_ort['coords'][ort_v['route'][k-1] if k>0 else 0][0] -
                   inst_ort['coords'][n][0])**2 +
                  (inst_ort['coords'][ort_v['route'][k-1] if k>0 else 0][1] -
                   inst_ort['coords'][n][1])**2)
        for k, n in enumerate(ort_v['route'])
    )
    rl_dist = sum(
        math.sqrt((inst_ort['coords'][rl_v['route'][k-1] if k>0 else 0][0] -
                   inst_ort['coords'][n][0])**2 +
                  (inst_ort['coords'][rl_v['route'][k-1] if k>0 else 0][1] -
                   inst_ort['coords'][n][1])**2)
        for k, n in enumerate(rl_v['route'])
    ) if rl_v else 0

    print(f"\n  Vehicle {vi:2d}  ({len(ort_v['route'])} stops)")
    print(f"  OR-Tools order: {ort_v['route']}")
    if rl_v:
        print(f"  RL       order: {rl_v['route']}")
        same_order = ort_v['route'] == rl_v['route']
        print(f"  Same order: {same_order}  |  ORT dist={ort_dist:.1f}  RL dist={rl_dist:.1f}")

        # Show TW violations if any
        late_stops = [s for s in ort_v['stops'] if s['late'] > 0.5]
        if late_stops:
            late_info = [(s['node'], f"{s['late']:.0f}min") for s in late_stops]
            print(f"  OR-Tools late: {late_info}")
    else:
        print(f"  RL: (no matching route)")
