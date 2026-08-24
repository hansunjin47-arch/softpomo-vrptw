"""Print current LLM prompts: base and accident scenarios."""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from R_env import load_waste_benchmark_txt, build_travel_time_matrix, EventType
from R_utils import TrainConfig, infer_solution_path, parse_solution_routes, build_zone_plan_from_routes
from R_llm_module import build_instance_confidence_prompt
from R_rl_module import BaselineEnv

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Solomon')

def make_env(name):
    path = os.path.join(DATA, name + '.txt')
    inst = load_waste_benchmark_txt(path)
    tt   = build_travel_time_matrix(inst)
    cfg  = TrainConfig(data_path=path)
    sol  = infer_solution_path(os.path.join(DATA, 'c101.txt'), None)
    zp   = build_zone_plan_from_routes(inst, parse_solution_routes(sol), cfg)
    env  = BaselineEnv(inst=inst, tt=tt, max_vehicles=zp.n_zones,
        late_penalty=cfg.late_penalty, unserved_penalty=cfg.unserved_penalty, max_scope_size=16)
    env.set_zone_assignment(zp.customer_to_zone, zp.n_zones, adjacent_zones=zp.adjacent_zones)
    env.reset()
    return inst, zp, env

SEP = "=" * 70

# ── 1. Base prompt ────────────────────────────────────────────────────────
_, zp, env = make_env('c101')
print(SEP)
print("  [1] BASE INSTANCE PROMPT (c101)")
print(SEP)
p1 = build_instance_confidence_prompt(env, zp, event=None, use_cot=True)
print(p1)
print(f"\n[{len(p1)} chars / {len(p1.splitlines())} lines]")

# ── 2. Accident prompt (mid-routing re-call simulation) ───────────────────
inst_acc, zp_acc, env_acc = make_env('c101_acc_A')
# Simulate mid-routing: run a few steps so some stops are served
import numpy as np
env_acc.reset()
# mark a few stops served to simulate mid-routing state
for c in list(zp_acc.zones[0])[:3]:
    env_acc.served[c] = 1

# Get first active accident event
acc_events = [e for e in (inst_acc.preset_events or []) if e.event_type == EventType.ACCIDENT]
acc_ev = acc_events[0] if acc_events else None

print(f"\n\n{SEP}")
print("  [2] ACCIDENT PROMPT (c101_acc_A, mid-routing re-call)")
print(f"      Event: {acc_ev.event_type.value} nodes={acc_ev.affected_nodes} "
      f"mult={acc_ev.multiplier:.1f} t={acc_ev.trigger_time:.0f}-{acc_ev.end_time:.0f}")
print(f"      Simulated: 3 stops already served in zone 0")
print(SEP)

# prev_scores: simulate some prior scores
prev = {c: round(0.3 + 0.5 * np.random.default_rng(42).random(), 2)
        for z in zp_acc.zones for c in z}
p2 = build_instance_confidence_prompt(env_acc, zp_acc, event=acc_ev, use_cot=True, prev_scores=prev)
print(p2)
print(f"\n[{len(p2)} chars / {len(p2.splitlines())} lines]")
