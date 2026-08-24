import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from R_utils import TrainConfig, SHARED_HYPERPARAMS, infer_solution_path, parse_solution_routes, build_zone_plan_from_routes
from R_rl_module import BaselineEnv, AttentionActorCritic

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')
inst = load_waste_benchmark_txt(os.path.join(DATA, 'c101.txt'))
tt   = build_travel_time_matrix(inst)
cfg  = TrainConfig(data_path='')
sol  = infer_solution_path(os.path.join(DATA, 'c101.txt'), None)
zp   = build_zone_plan_from_routes(inst, parse_solution_routes(sol), cfg)

env = BaselineEnv(inst=inst, tt=tt, max_vehicles=zp.n_zones,
    late_penalty=cfg.late_penalty, unserved_penalty=cfg.unserved_penalty,
    max_scope_size=16)
env.set_zone_assignment(zp.customer_to_zone, zp.n_zones, adjacent_zones=zp.adjacent_zones)
env.reset()

hp = SHARED_HYPERPARAMS
print(f"SHARED_HYPERPARAMS:")
print(f"  n_encoder_layers = {hp['n_encoder_layers']}  (C+R: 6)")
print(f"  n_heads          = {hp['n_heads']}  (C+R: 8)")
print(f"  entropy_coef     = {hp['entropy_coef']}  (C+R: 0)")
print()
print(f"BaselineEnv:")
print(f"  stop_feat_dim    = {env.stop_feat_dim}  (was 8, now 9 with demand)")
print(f"  vehicle_feat_dim = {env.vehicle_feat_dim}  (was 3, now 4 with remaining_cap)")
print(f"  obs_dim          = {env.observation_space.shape[0]}")

model = AttentionActorCritic(
    stop_feat_dim=env.stop_feat_dim,
    max_scope_size=16,
    vehicle_feat_dim=env.vehicle_feat_dim,
    global_feat_dim=env.global_feat_dim,
    embed_dim=hp['embed_dim'],
    n_heads=hp['n_heads'],
    n_encoder_layers=hp['n_encoder_layers'],
).to('cpu')
total_params = sum(p.numel() for p in model.parameters())
print()
print(f"AttentionActorCritic:")
print(f"  total params     = {total_params:,}")

obs, _ = env.reset()
obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
logits, value = model(obs_t)
print(f"  forward pass     = logits{tuple(logits.shape)} value{tuple(value.shape)}  OK")

# Verify obs contains demand info
print()
print("Spot-check obs[8] (demand) for scope stop 1:")
# stop_feat_dim=9, max_scope=16 -> stop section = 16*9=144 tokens
# stop 1 starts at offset 9
demand_in_obs = float(obs[9 + 8])  # stop index 1, feature 8
cap = float(inst.vehicle_capacity)
scope_custs = [c for c in range(1, inst.n_customers+1)
               if zp.customer_to_zone[c] == 0]
if scope_custs:
    first_cust = scope_custs[0]
    expected = float(inst.demands[first_cust]) / cap
    print(f"  obs demand_norm  = {demand_in_obs:.4f}  expected {expected:.4f}")

# Verify vehicle_feat has remaining_cap
# offset = max_scope*stop_feat + rain_tokens + acc_tokens = 16*9 + 32*4 + 16*4 = 144+128+64 = 336
veh_offset = 16*9 + 32*4 + 16*4
remaining_cap_norm = float(obs[veh_offset + 3])
expected_cap_norm = float(inst.vehicle_capacity) / cap
print(f"  remaining_cap    = {remaining_cap_norm:.4f}  expected {expected_cap_norm:.4f}")
