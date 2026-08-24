"""
LLM-guided RL inference for pure VRPTW.

LLM (Qwen3:32b) is called at episode start and on each vehicle retirement
to maintain a global top-k priority list of unserved customers.
The pre-trained RL policy selects step-by-step from this pruned action space.

Role separation
---------------
  LLM : "Which customers are most urgent/critical to visit soon?" (global)
        → reduces search space, does NOT assign customers to vehicles
  RL  : decides which vehicle, in what order, when to return to depot
        → full routing authority within the pruned space

This preserves RL exploration: LLM acts as a radar highlighting important
targets; RL decides how to engage them.

Design
------
  Training : pure RL (rl_pure_vrptw.py) — no LLM, fast
  Inference : same trained RL checkpoint, two modes
    --no-llm  : pure RL greedy baseline
    (default) : RL + LLM global priority pruning

LLM call triggers (not per-step, not per-dispatch)
---------------------------------------------------
  1. Episode start       — global view of all customers
  2. Each vehicle retire — state has changed significantly; refresh priority

~10 calls/episode × ~20s = ~3 min/episode. Feasible for evaluation.

Ablation
--------
  Pure RL    : python llm_rl_vrptw.py c101 --ckpt best.pt --no-llm
  RL + LLM   : python llm_rl_vrptw.py c101 --ckpt best.pt
  Ontology   : python ontology_pure_vrptw.py c101
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Set

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import numpy as np
import torch

from CR_env import VRPTWEnv, load_solomon, build_tt
from CR_rl_module import VRPTWAttentionModel, build_node_feats, get_vehicle_feat
from CR_llm_module import DEFAULT_MODEL, refresh_priority


# ============================================================
# RL greedy decoder with optional priority mask
# ============================================================

@torch.no_grad()
def _rl_greedy_step(
    env: VRPTWEnv,
    model: VRPTWAttentionModel,
    h: torch.Tensor,
    h_bar: torch.Tensor,
    device: str,
    priority: Optional[Set[int]],
) -> int:
    """
    One greedy step. If `priority` is given, restrict to depot + priority ∩ feasible.
    Falls back to full feasible mask when priority set is exhausted.
    """
    full_mask = env.get_feasible_mask()   # [N+1] bool

    if priority is not None:
        pruned = np.zeros_like(full_mask)
        pruned[0] = True                  # depot always available
        for c in priority:
            if full_mask[c]:
                pruned[c] = True
        # Fallback: if no priority customer is feasible, use full mask
        mask = pruned if pruned[1:].any() else full_mask
    else:
        mask = full_mask

    if not mask.any():
        return 0

    av     = env.vehicles[env.active_idx]
    last_t = torch.tensor([av.cur_node], device=device)
    veh_t  = torch.tensor(
        get_vehicle_feat(env), dtype=torch.float32, device=device
    ).unsqueeze(0)
    mask_t = torch.tensor(mask.astype(np.float32), device=device).unsqueeze(0)

    action, _ = model.act(h, h_bar, last_t, veh_t, mask_t, greedy=True)
    return int(action.item())


# ============================================================
# Episode runner
# ============================================================

def run_episode(
    env: VRPTWEnv,
    model: VRPTWAttentionModel,
    node_feats_t: torch.Tensor,
    device: str,
    max_steps: int,
    llm_model: Optional[str],
    top_k: int,
) -> tuple[float, int]:
    """
    Run one greedy evaluation episode.
    llm_model=None → pure RL (no LLM pruning).
    Returns (total_reward, llm_calls).
    """
    env.reset()
    h, h_bar = model.encode(node_feats_t)

    priority:   Optional[Set[int]] = None
    llm_calls   = 0
    total_reward = 0.0
    prev_active  = env.active_idx
    prev_retired = set(i for i, v in enumerate(env.vehicles) if v.retired)

    def _refresh():
        nonlocal priority, llm_calls
        if llm_model is None:
            return
        priority  = refresh_priority(env, llm_model, top_k)
        llm_calls += 1

    # Initial priority at episode start
    _refresh()

    for _ in range(max_steps):
        if not env.get_feasible_mask().any():
            break

        # Detect vehicle retirement → state changed significantly → refresh
        cur_retired = set(i for i, v in enumerate(env.vehicles) if v.retired)
        if cur_retired != prev_retired:
            prev_retired = cur_retired
            _refresh()

        action = _rl_greedy_step(env, model, h, h_bar, device, priority)

        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    return total_reward, llm_calls


# ============================================================
# Output formatting
# ============================================================

def _fmt(tag: str, ep: int, reward: float, env: VRPTWEnv,
         ep_time: float, llm_calls: int) -> str:
    served   = int(np.sum(env.served[1:]))
    late_cnt = env.late_count
    on_time  = served - late_cnt
    unserved = env.N - served
    late_avg = env.total_late / late_cnt if late_cnt > 0 else 0.0
    llm_str  = f"  llm_calls={llm_calls:2d}" if llm_calls >= 0 else ""
    return (
        f"[{tag} EP {ep:4d}]"
        f" reward={reward:10.2f}"
        f"  served={on_time:3d}"
        f"  tw_late={late_cnt:3d}"
        f"  unserved={unserved:3d}/{env.N}"
        f"  veh={env.vehicles_used:2d}/{env.max_vehicles:2d}"
        f"  dist={env.total_distance:10.2f}"
        f"  late={env.total_late:9.2f}"
        f"  late_avg={late_avg:8.2f}"
        f"{llm_str}"
        f"  ep_time={ep_time:.1f}s"
        f"  done={env.last_end_reason}"
    )


# ============================================================
# Main
# ============================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="LLM-guided RL inference for pure VRPTW",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("instance", nargs="?", default="c101")
    p.add_argument("--ckpt",       required=True,
                   help="Path to trained RL checkpoint (.pt)")
    p.add_argument("--data-dir",   default=os.path.join(_ROOT, "data", "Solomon"))
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--top-k",      type=int, default=20,
                   help="Global priority list size (LLM top-k customers to focus on)")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--max-steps",  type=int, default=600)
    p.add_argument("--no-llm",     action="store_true",
                   help="Disable LLM — pure RL greedy (ablation baseline)")
    # env rewards — must match training config
    p.add_argument("--visit-reward",        type=float, default=2.0)
    p.add_argument("--late-penalty-fixed",  type=float, default=4.0)
    p.add_argument("--late-penalty-scale",  type=float, default=5.0)
    p.add_argument("--unserved-penalty",    type=float, default=8.0)
    p.add_argument("--vehicle-use-penalty", type=float, default=10.0)
    # model arch — must match training config
    p.add_argument("--embed-dim",    type=int, default=128)
    p.add_argument("--n-heads",      type=int, default=8)
    p.add_argument("--n-enc-layers", type=int, default=3)
    return p.parse_args()


def main():
    args   = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inst_arg = args.instance
    if os.path.isfile(inst_arg):
        data_path = inst_arg
    else:
        name = inst_arg if inst_arg.endswith(".txt") else inst_arg + ".txt"
        data_path = os.path.join(args.data_dir, name)

    inst = load_solomon(data_path)
    tt   = build_tt(inst)
    env  = VRPTWEnv(
        inst=inst, tt=tt,
        visit_reward        = args.visit_reward,
        late_penalty_fixed  = args.late_penalty_fixed,
        late_penalty_scale  = args.late_penalty_scale,
        unserved_penalty    = args.unserved_penalty,
        vehicle_use_penalty = args.vehicle_use_penalty,
    )

    n_nodes = inst.n_customers + 1
    model   = VRPTWAttentionModel(
        n_nodes=n_nodes, embed_dim=args.embed_dim,
        n_heads=args.n_heads, n_enc_layers=args.n_enc_layers,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    node_feats_t = torch.tensor(
        build_node_feats(inst), dtype=torch.float32, device=device
    ).unsqueeze(0)

    llm_model = None if args.no_llm else args.model
    tag       = "RL    " if args.no_llm else "RL+LLM"

    print(f"[{tag}] instance={inst.name}  N={inst.n_customers}  vehicles={inst.vehicle_limit}")
    print(f"[{tag}] ckpt={args.ckpt}  device={device}")
    if llm_model:
        print(f"[{tag}] llm={llm_model}  top_k={args.top_k}")
        print(f"[{tag}] LLM triggered at: episode start + each vehicle retirement")
    else:
        print(f"[{tag}] LLM disabled — pure RL greedy")
    print()

    best_reward = -1e9
    best_served = -1
    best_line   = ""
    t0          = time.time()

    for ep in range(1, args.n_episodes + 1):
        ep_t0  = time.time()
        reward, llm_calls = run_episode(
            env, model, node_feats_t, device,
            args.max_steps, llm_model, args.top_k,
        )
        ep_time = time.time() - ep_t0
        line    = _fmt(tag, ep, reward, env, ep_time, llm_calls)

        served  = int(np.sum(env.served[1:]))
        is_best = served > best_served or (served == best_served and reward > best_reward)
        if is_best:
            best_served, best_reward = served, reward
            best_line = line
            print("[Best EP] " + line)

        print(line)

    total = time.time() - t0
    print(f"\n[Time] total={total:.1f}s  best_served={best_served}")
    if best_line:
        print("[Best EP] " + best_line)


if __name__ == "__main__":
    main()
