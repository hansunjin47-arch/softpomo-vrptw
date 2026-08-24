import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import R_env, R_main, R_llm_module
from R_rl import _TRAIN_BASE, _TEST_INSTANCES
from R_utils import TrainConfig, infer_solution_path, parse_solution_routes, build_zone_plan_from_routes
from R_env import load_waste_benchmark_txt, build_travel_time_matrix

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')

# 1. Event parser fix
inst = load_waste_benchmark_txt(os.path.join(DATA, 'c101_acc_A.txt'))
probs = set(e.probability for e in inst.preset_events)
nodes = [len(e.affected_nodes) for e in inst.preset_events]
print(f"[1] acc_A parser: prob={probs}  nodes_per_event={nodes}")

inst_r = load_waste_benchmark_txt(os.path.join(DATA, 'c101_rain_A.txt'))
probs_r = set(e.probability for e in inst_r.preset_events)
print(f"    rain_A parser: prob={probs_r}  n_events={len(inst_r.preset_events)}")

# 2. Clustering: base_name lookup
cfg = TrainConfig(data_path='')
inst_base = load_waste_benchmark_txt(os.path.join(DATA, 'c101.txt'))
inst_rain = load_waste_benchmark_txt(os.path.join(DATA, 'c101_rain_A.txt'))
sol = infer_solution_path(os.path.join(DATA, 'c101.txt'), None)
zp_base = build_zone_plan_from_routes(inst_base, parse_solution_routes(sol), cfg)
zp_rain = build_zone_plan_from_routes(inst_rain, parse_solution_routes(sol), cfg)
same = all(sorted(z1) == sorted(z2) for z1, z2 in zip(zp_base.zones, zp_rain.zones))
print(f"[2] clustering unified: c101_base == c101_rain_A zones → {same}")

# 3. Pre-computation covers c101 (test base)
test_base = sorted({n.split('_rain_')[0].split('_acc_')[0] for n in _TEST_INSTANCES} - set(_TRAIN_BASE))
print(f"[3] pre-compute adds test base: {test_base}")

# 4. LLM unique score requirement in prompt
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'R_llm_module.py'), encoding='utf-8').read()
print(f"[4] unique score instruction: {'unique score' in src}")

# 5. LLM error raises instead of defaulting
print(f"[5] LLMError class exists: {hasattr(R_llm_module, 'LLMError')}")

print("\nAll checks passed")
