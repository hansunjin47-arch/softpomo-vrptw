"""Quick greedy eval on all 5 test instances."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.optim as optim

from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from R_rl import _collect_episode, _TEST_INSTANCES, MAX_SCOPE
from R_rl_module import AttentionActorCritic, load_checkpoint, BaselineEnv
from R_utils import (
    TrainConfig, SHARED_HYPERPARAMS, set_seed,
    infer_solution_path, parse_solution_routes,
    build_zone_plan_from_routes, build_initial_zones,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')
CKPT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'multi_pomo_last.pkl')

set_seed(42)
hp = dict(SHARED_HYPERPARAMS)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_env(name):
    path = os.path.join(DATA_DIR, name + '.txt')
    inst = load_waste_benchmark_txt(path)
    tt   = build_travel_time_matrix(inst)
    cfg  = TrainConfig(
        data_path=path, use_solution_zones=True,
        prefer_on_time_candidates=False,
        use_ontology_pruning=False, use_ontology_reasoning=False,
        late_count_penalty=hp.get('late_count_penalty', 20.0),
        late_penalty=hp.get('late_penalty', 150.0),
        unserved_penalty=hp.get('unserved_penalty', 500.0),
    )
    sol_path = infer_solution_path(path, None)
    if sol_path:
        zp = build_zone_plan_from_routes(inst, parse_solution_routes(sol_path), cfg)
    else:
        zp = build_initial_zones(inst, cfg, tt)
    max_veh = zp.n_zones if zp else inst.vehicle_limit
    env = BaselineEnv(
        inst=inst, tt=tt, max_vehicles=max_veh,
        late_count_penalty=cfg.late_count_penalty,
        late_penalty=cfg.late_penalty,
        unserved_penalty=cfg.unserved_penalty,
        max_scope_size=MAX_SCOPE,
    )
    if zp:
        env.set_zone_assignment(zp.customer_to_zone, zp.n_zones, adjacent_zones=zp.adjacent_zones)
    return inst, tt, zp, cfg, env


_, _, _, _, env0 = make_env('c102')
actor_critic = AttentionActorCritic(
    stop_feat_dim=env0.stop_feat_dim,
    max_scope_size=MAX_SCOPE,
    vehicle_feat_dim=env0.vehicle_feat_dim,
    global_feat_dim=env0.global_feat_dim,
    embed_dim=hp.get('embed_dim', 128),
    n_heads=hp.get('n_heads', 8),
    n_encoder_layers=hp.get('n_encoder_layers', 3),
).to(device)

opt = optim.Adam(actor_critic.parameters())
start_ep, _, _, _, _ = load_checkpoint(CKPT, actor_critic, opt, ontology_engine=None, device=device)
ep = start_ep - 1
print(f'Loaded checkpoint ep={ep}')
print(f'obs_dim={env0.observation_space.shape[0]}  stop_feat_dim={env0.stop_feat_dim}  MAX_SCOPE={MAX_SCOPE}')
print()
print(f"  {'Instance':<22}  served  unserved  veh  dist        late_cnt  late_min")
print('  ' + '-' * 75)

for name in _TEST_INSTANCES:
    _, _, zp, cfg, env = make_env(name)
    rg, lc, sc, reason, _, _ = _collect_episode(
        env, zp, cfg, actor_critic, device, MAX_SCOPE, 10000, greedy=True)
    unserved = env.N - sc
    print(f"  {name:<22}  {sc:6d}  {unserved:8d}  {env.vehicles_used:3d}"
          f"  {env.total_distance:9.1f}  {lc:8d}  {env.total_late:8.1f}")
