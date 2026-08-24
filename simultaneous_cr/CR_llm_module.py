"""
LLM interface for simultaneous C+R VRPTW.

Imported by llm_rl_vrptw.py and llm_pure_vrptw.py.

Two usage modes
---------------
  Priority pruning (RL+LLM):
    refresh_priority(env, model, top_k) → Set[int]
    Called at episode start + vehicle retirement.

  Per-step selection (LLM-only):
    select_action(env, mask, model, top_k) → (action, used_llm)
    Called every step — impractical for N=100 but kept for reference.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import List, Optional, Set, Tuple

import numpy as np

from CR_env import VRPTWEnv

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:32b"


# ============================================================
# Shared: Ollama query
# ============================================================

def query_llm(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=None) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return (body.get("response") or "").strip()


# ============================================================
# Priority pruning (RL+LLM mode)
# ============================================================

def build_priority_prompt(env: VRPTWEnv, remaining: List[int], top_k: int) -> str:
    """
    Global priority prompt: which customers should be visited soon (by any vehicle)?
    LLM considers all factors holistically; does NOT assign customers to vehicles.
    """
    T       = env._T
    served  = int(np.sum(env.served[1:]))
    n_alive = sum(1 for v in env.vehicles if not v.retired)

    veh_summary = []
    for i, v in enumerate(env.vehicles):
        if not v.retired:
            veh_summary.append(
                f"  V{i}: node={v.cur_node}  time={v.cur_time:.0f}"
                f"  cap_left={v.remaining_cap:.0f}"
            )

    lines = [
        "You are a VRPTW routing expert advising a fleet of delivery vehicles.",
        "",
        "=== GLOBAL STATE ===",
        f"  Time horizon        : {T:.0f}",
        f"  Customers served    : {served} / {env.N}",
        f"  Active vehicles     : {n_alive} / {env.max_vehicles}",
        f"  Unserved customers  : {len(remaining)}",
        "",
        "=== ACTIVE VEHICLES ===",
    ] + veh_summary + [
        "",
        "=== UNSERVED CUSTOMERS ===",
        f"{'ID':>4}  {'x':>6}  {'y':>6}  {'tw_open':>7}  {'tw_close':>8}"
        f"  {'demand':>6}  {'dist_depot':>10}  {'tw_width':>8}",
        "-" * 68,
    ]

    for c in remaining:
        x    = float(env.inst.coords[c, 0])
        y    = float(env.inst.coords[c, 1])
        tw_o = float(env.inst.tw_open[c])
        tw_c = float(env.inst.tw_close[c])
        dem  = float(env.inst.demands[c])
        dist = float(env.tt[0, c])
        tw_w = tw_c - tw_o
        lines.append(
            f"{c:>4}  {x:>6.1f}  {y:>6.1f}  {tw_o:>7.0f}  {tw_c:>8.0f}"
            f"  {dem:>6.0f}  {dist:>10.1f}  {tw_w:>8.0f}"
        )

    lines += [
        "",
        "=== TASK ===",
        f"Given the current state, select the top-{top_k} customers that should",
        "be visited soon (by any vehicle, in any order).",
        "",
        "Consider ALL of the following factors together:",
        "  - Time window tightness (risk of missing deadline)",
        "  - Geographic position (isolated customers are harder to reach later)",
        "  - Distance from active vehicles (routing efficiency)",
        "  - Demand vs remaining vehicle capacity",
        "  - Interaction effects (visiting A now enables visiting B later)",
        "",
        "Do NOT assign customers to specific vehicles.",
        "",
        f"Output ONLY valid JSON (no extra text):",
        f'{{"priority": [id1, id2, ..., id{top_k}], "reason": "one-line explanation"}}',
    ]

    return "\n".join(lines)


def parse_priority(response: str, remaining: List[int], top_k: int) -> Optional[List[int]]:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return None

    raw = data.get("priority", [])
    if not isinstance(raw, list):
        return None

    valid_set = set(remaining)
    priority = [int(c) for c in raw if isinstance(c, (int, float)) and int(c) in valid_set]
    return priority[:top_k] if priority else None


def refresh_priority(env: VRPTWEnv, model: str, top_k: int) -> Set[int]:
    """
    Call LLM for a fresh global priority set.
    Falls back to all remaining customers on failure.
    """
    remaining = [c for c in range(1, env.N + 1) if not env.served[c]]
    if not remaining:
        return set()

    effective_k = min(top_k, len(remaining))
    prompt      = build_priority_prompt(env, remaining, effective_k)

    try:
        response = query_llm(prompt, model)
        priority = parse_priority(response, remaining, effective_k)
        if priority:
            return set(priority)
        print(f"  [LLM:parse_fail] fallback to all {len(remaining)} remaining")
    except (urllib.error.URLError, Exception) as e:
        print(f"  [LLM:error] {e} — fallback to all remaining")

    return set(remaining)


# ============================================================
# Per-step selection (LLM-only mode)
# ============================================================

def build_step_prompt(env: VRPTWEnv, mask: np.ndarray, top_k: int) -> str:
    """Per-step prompt: select best next action for the active vehicle."""
    av      = env.vehicles[env.active_idx]
    T       = env._T
    cap     = env._cap
    served  = int(np.sum(env.served[1:]))
    n_alive = sum(1 for v in env.vehicles if not v.retired)

    lines = [
        "You are an expert VRPTW routing agent.",
        "Select the best next action for the current delivery vehicle.",
        "",
        "=== VEHICLE STATE ===",
        f"  Position      : node {av.cur_node}",
        f"  Current time  : {av.cur_time:.1f} / {T:.0f}",
        f"  Remaining cap : {av.remaining_cap:.0f} / {cap:.0f}",
        f"  Served so far : {served} / {env.N} customers",
        f"  Active vehicles (incl. this) : {n_alive} / {env.max_vehicles}",
        "",
        "=== FEASIBLE ACTIONS ===",
        f"{'ID':>4}  {'dist':>6}  {'tw_open':>7}  {'tw_close':>8}  "
        f"{'demand':>6}  {'tw_slack':>8}  {'wait':>5}  {'class'}",
        "-" * 68,
    ]

    depot_travel = float(env.tt[av.cur_node, 0])
    lines.append(
        f"{'0':>4}  {depot_travel:>6.1f}  {'—':>7}  {'—':>8}  "
        f"{'—':>6}  {'—':>8}  {'—':>5}  DEPOT_RETURN"
    )

    for c in range(1, env.N + 1):
        if not mask[c]:
            continue
        travel   = float(env.tt[av.cur_node, c])
        arrival  = av.cur_time + travel
        tw_o     = float(env.inst.tw_open[c])
        tw_c     = float(env.inst.tw_close[c])
        demand   = float(env.inst.demands[c])
        wait     = max(0.0, tw_o - arrival)
        tw_slack = tw_c - max(arrival, tw_o)
        cls = "URGENT" if tw_slack <= 30 else ("early" if wait > 50 else "normal")
        lines.append(
            f"{c:>4}  {travel:>6.1f}  {tw_o:>7.0f}  {tw_c:>8.0f}  "
            f"{demand:>6.0f}  {tw_slack:>8.1f}  {wait:>5.1f}  {cls}"
        )

    lines += [
        "",
        "=== TASK ===",
        f"Return the top-{top_k} actions in priority order. Consider:",
        "  1. Serve URGENT customers first (tight time windows).",
        "  2. Prefer nearby customers to keep routes compact (implicit clustering).",
        "  3. Choose DEPOT_RETURN (ID=0) when remaining customers are better",
        "     served by a fresh vehicle or when capacity is nearly exhausted.",
        "",
        f'Output ONLY valid JSON: {{"ranking": [id1, id2, ...top{top_k}], '
        f'"reason": "one-line explanation"}}',
        "No extra text outside the JSON.",
    ]

    return "\n".join(lines)


def parse_ranking(response: str, mask: np.ndarray) -> Optional[List[int]]:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return None

    ranking = data.get("ranking", [])
    if not isinstance(ranking, list):
        return None

    valid = [int(a) for a in ranking if isinstance(a, (int, float))
             and 0 <= int(a) <= len(mask) - 1 and mask[int(a)]]
    return valid if valid else None


def heuristic_fallback(env: VRPTWEnv, mask: np.ndarray) -> int:
    """Nearest urgent customer, else nearest feasible, else depot."""
    av = env.vehicles[env.active_idx]
    best, best_score = 0, -1e9
    for c in range(1, env.N + 1):
        if not mask[c]:
            continue
        travel   = float(env.tt[av.cur_node, c])
        arrival  = av.cur_time + travel
        tw_c     = float(env.inst.tw_close[c])
        tw_slack = tw_c - max(arrival, float(env.inst.tw_open[c]))
        score    = -travel + 5.0 / (tw_slack + 1.0)
        if score > best_score:
            best, best_score = c, score
    return best


def select_action(env: VRPTWEnv, mask: np.ndarray, model: str, top_k: int,
                  verbose: bool = False) -> Tuple[int, bool]:
    """
    Query LLM and return (action, used_llm).
    Falls back to heuristic on parse failure.
    """
    prompt = build_step_prompt(env, mask, top_k)
    try:
        response = query_llm(prompt, model)
        if verbose:
            print(f"  [LLM raw] {response[:200]}")
        ranking = parse_ranking(response, mask)
        if ranking:
            return ranking[0], True
        print("  [LLM:parse_fail] falling back to heuristic")
    except (urllib.error.URLError, Exception) as e:
        print(f"  [LLM:error] {e} — falling back to heuristic")

    return heuristic_fallback(env, mask), False
