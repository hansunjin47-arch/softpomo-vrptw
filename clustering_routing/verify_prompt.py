import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from R_utils import TrainConfig, infer_solution_path, parse_solution_routes, build_zone_plan_from_routes
from R_llm_module import build_confidence_prompt, build_instance_confidence_prompt
from R_rl_module import BaselineEnv

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')
path = os.path.join(DATA, 'c101.txt')
inst = load_waste_benchmark_txt(path)
tt   = build_travel_time_matrix(inst)
cfg  = TrainConfig(data_path=path, use_solution_zones=True)
sol  = infer_solution_path(path, None)
zp   = build_zone_plan_from_routes(inst, parse_solution_routes(sol), cfg)
env  = BaselineEnv(inst=inst, tt=tt, max_vehicles=zp.n_zones,
    late_penalty=cfg.late_penalty, unserved_penalty=cfg.unserved_penalty, max_scope_size=16)
env.set_zone_assignment(zp.customer_to_zone, zp.n_zones, adjacent_zones=zp.adjacent_zones)
env.reset()

CHECKS = [
    '## Role',
    '## Objective',
    'Priority (highest to lowest)',
    '1. Maximise stops',
    '2. Minimise late arrivals',
    '3. Minimise total',
    '## Column definitions',
    'Negative      ',
    'small positive',
    'Large positive',
    'demand=',
    'service_time=',
    'Phase 1',
]

zone0 = list(zp.zones[0])
p1 = build_confidence_prompt(env, zone0, 0, use_cot=True)
lines1 = p1.split('\n')
print(f'build_confidence_prompt: {len(p1)} chars / {len(lines1)} lines')
for kw in CHECKS:
    found = kw in p1
    status = 'OK' if found else 'MISSING'
    print(f'  [{status}] {repr(kw)}')

print()
p2 = build_instance_confidence_prompt(env, zp, use_cot=True)
lines2 = p2.split('\n')
print(f'build_instance_confidence_prompt: {len(p2)} chars / {len(lines2)} lines')
for kw in CHECKS:
    found = kw in p2
    status = 'OK' if found else 'MISSING'
    print(f'  [{status}] {repr(kw)}')

print()
print('--- per-zone prompt first 40 lines ---')
for ln in lines1[:40]:
    print(ln)
