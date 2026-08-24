import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import R_env, R_utils, R_rl, R_main
from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from R_utils import TrainConfig, infer_solution_path, parse_solution_routes, build_zone_plan_from_routes, SHARED_HYPERPARAMS
from R_rl_module import BaselineEnv

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')
inst = load_waste_benchmark_txt(os.path.join(DATA, 'c101.txt'))
tt   = build_travel_time_matrix(inst)
cfg  = TrainConfig(data_path='')
sol  = infer_solution_path(os.path.join(DATA, 'c101.txt'), None)
zp   = build_zone_plan_from_routes(inst, parse_solution_routes(sol), cfg)
env  = BaselineEnv(inst=inst, tt=tt, max_vehicles=zp.n_zones,
    late_count_penalty=cfg.late_count_penalty,
    late_penalty=cfg.late_penalty, unserved_penalty=cfg.unserved_penalty, max_scope_size=16)
env.set_zone_assignment(zp.customer_to_zone, zp.n_zones, adjacent_zones=zp.adjacent_zones)
env.reset()

T = env._time_scale()
N = env.N

print(f"SHARED late_count_penalty = {SHARED_HYPERPARAMS['late_count_penalty']}")
print(f"Config late_count_penalty = {cfg.late_count_penalty}")
print(f"Env    late_count_penalty = {env.late_count_penalty}")
print()

env.total_distance = 827.3

# Perfect
env.total_late = 0.0; env.late_count = 0
r_perfect = env._episode_reward(0)

# 2 stops late, total 30min late
env.total_late = 30.0; env.late_count = 2
r_late = env._episode_reward(0)

# 1 unserved, same late
r_unserved = env._episode_reward(1)

print(f"Perfect (dist=827.3, 0 late, 0 unserved):  {r_perfect:.4f}")
print(f"2 late (total 30min): {r_late:.4f}")
print(f"  = -(dist/T={827.3/T:.4f} + late_cnt={20*2/N:.4f} + late_mag={150*30/T:.4f})")
print(f"1 unserved (+ 2 late 30min): {r_unserved:.4f}")
print(f"  unserved penalty = {500*1/N:.4f}")
print()
print("All OK")
