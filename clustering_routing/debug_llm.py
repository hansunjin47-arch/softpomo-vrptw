"""Debug LLM truncation issue."""
import sys, os, json, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from R_env import load_waste_benchmark_txt, build_travel_time_matrix
from R_utils import TrainConfig, infer_solution_path, parse_solution_routes, build_zone_plan_from_routes
from R_llm_module import build_instance_confidence_prompt, _build_payload, OLLAMA_URL
from R_rl_module import BaselineEnv

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')
inst = load_waste_benchmark_txt(os.path.join(DATA, 'c102.txt'))
tt   = build_travel_time_matrix(inst)
cfg  = TrainConfig(data_path='')
sol  = infer_solution_path(os.path.join(DATA, 'c102.txt'), None)
zp   = build_zone_plan_from_routes(inst, parse_solution_routes(sol), cfg)
env  = BaselineEnv(inst=inst, tt=tt, max_vehicles=zp.n_zones,
    late_count_penalty=cfg.late_count_penalty,
    late_penalty=cfg.late_penalty, unserved_penalty=cfg.unserved_penalty, max_scope_size=16)
env.set_zone_assignment(zp.customer_to_zone, zp.n_zones, adjacent_zones=zp.adjacent_zones)
env.reset()

prompt = build_instance_confidence_prompt(env, zp, event=None, use_cot=False)
payload = _build_payload("qwen3:32b", prompt, use_cot=False, temperature=0.0)

# Inspect what we're sending
p = json.loads(payload.decode())
print(f"think={p['think']}")
print(f"num_predict={p['options']['num_predict']}")
print(f"num_ctx={p['options']['num_ctx']}")
print(f"prompt length: {len(prompt)} chars")
print()

import time
print("Calling Ollama...")
t0 = time.time()
req = urllib.request.Request(OLLAMA_URL, data=payload,
    headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=None) as resp:
    body = json.loads(resp.read().decode("utf-8"))
elapsed = time.time() - t0

print(f"Response in {elapsed:.1f}s")
print(f"done_reason: {body.get('done_reason')}")
print(f"eval_count (tokens generated): {body.get('eval_count')}")
print(f"prompt_eval_count (prompt tokens): {body.get('prompt_eval_count')}")
print()

output = body.get("response", "")
print(f"Response length: {len(output)} chars")
print(f"Response preview (first 500 chars):")
print(output[:500])
print("...")
print(f"Response tail (last 200 chars):")
print(output[-200:])
