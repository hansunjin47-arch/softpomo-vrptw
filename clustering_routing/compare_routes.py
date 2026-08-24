"""Compare RL vs OR-Tools solution for c101 base."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'original_POMO'))

import numpy as np
import torch
import torch.optim as optim

from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from R_utils import TrainConfig, SHARED_HYPERPARAMS, infer_solution_path, parse_solution_routes, build_zone_plan_from_routes
from R_rl_module import BaselineEnv, AttentionActorCritic, load_checkpoint
from R_rl import _collect_episode, MAX_SCOPE, _TRAIN_BASE
from ortools_vrptw import load_solomon_raw, build_tt, solve

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'rl_pomo', 'multi_pomo_last.pkl')

# ── OR-Tools solution ─────────────────────────────────────────────────────
print("=== OR-Tools ===")
inst_ort = load_solomon_raw(os.path.join(DATA, 'c101.txt'))
tt_ort   = build_tt(inst_ort)
r_ort    = solve(inst_ort, tt_ort, time_limit_sec=30)
print(f"  served={r_ort['served_on_time']+r_ort['served_late']}/100  "
      f"late={r_ort['served_late']}  dist={r_ort['total_distance']:.1f}  "
      f"veh={r_ort['vehicles_used']}")
ort_routes = [r for r in r_ort['routes'] if r]
print(f"  routes ({len(ort_routes)} vehicles):")
for i, route in enumerate(ort_routes):
    print(f"    V{i+1}: {route}")

# ── RL solution ──────────────────────────────────────────────────────────
print("\n=== RL ===")
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
print(f"  served={sc}/100  late={lc}  dist={env.total_distance:.1f}  veh={env.vehicles_used}")
rl_routes = [v.route_nodes[1:-1] for v in env.vehicles if v.activated and len(v.route_nodes) > 2]
print(f"  routes ({len(rl_routes)} vehicles):")
for i, route in enumerate(rl_routes):
    print(f"    V{i+1}: {route}")

# ── Compare ───────────────────────────────────────────────────────────────
print("\n=== Comparison ===")
ort_set = {frozenset(r) for r in ort_routes}
rl_set  = {frozenset(r) for r in rl_routes}
same_zones = ort_set == rl_set
print(f"  Same stop groupings: {same_zones}")
print(f"  OR-Tools dist: {r_ort['total_distance']:.1f}")
print(f"  RL dist:       {env.total_distance:.1f}")
print(f"  Diff: {env.total_distance - r_ort['total_distance']:+.1f}")
