from __future__ import annotations

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import os
import random
import time
from typing import Dict, List, Optional, Set, Tuple

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from R_env import (
    WasteFleetEnv,
    EventType,
    build_travel_time_matrix,
    load_waste_benchmark_txt,
)
from R_ontology import OntologyReasoningEngine

from R_utils import (
    TrainConfig,
    ZonePlan,
    SHARED_HYPERPARAMS,
    set_seed,
    infer_solution_path,
    parse_solution_routes,
    validate_solution_routes,
    build_zone_plan_from_routes,
    build_initial_zones,
    estimate_vehicle_count,
    get_feasible_actions_with_zone_control,
    prune_candidates_by_ontology,
    select_fixed_solution_action,
    CandidateFact,
)
from R_llm_module import (
    query_ollama_zone_confidence,
    query_ollama_confidence,
    query_ollama_instance_confidence,
)
from R_rl_module import (
    RolloutBuffer,
    AttentionActorCritic,
    BaselineEnv,
    ppo_update,
    pomo_update,
    kool_update,
    save_checkpoint,
    load_checkpoint,
    _plot_routing,
    _plot_reward_curve,
)


# ============================================================
# get_candidate_actions — integrates ontology + LLM + RL
# ============================================================

def get_candidate_actions(
    env: WasteFleetEnv,
    feasible: List[int],
    cfg: TrainConfig,
    base_zone_idx: Optional[int],
    zone_plan: Optional[ZonePlan],
) -> Tuple[List[int], str, List[CandidateFact], str]:
    """Returns (candidate_actions, prune_source, ontology_facts, llm_reasoning)."""
    if len(feasible) == 0:
        return [], "empty", [], ""

    candidates, ontology_facts = prune_candidates_by_ontology(
        env=env, feasible=feasible,
        zone_plan=zone_plan, active_zone_idx=base_zone_idx, cfg=cfg,
    )
    return candidates if candidates else feasible, "ontology", ontology_facts, ""


# ============================================================
# POMO episode helper
# ============================================================

def _run_pomo_episode(
    env: WasteFleetEnv,
    cfg: TrainConfig,
    actor_critic: AttentionActorCritic,
    zone_plan: Optional[ZonePlan],
    llm_confidence_scores: Dict[int, float],
    max_scope: int,
    forced_first_per_zone: Optional[Dict[int, int]] = None,
    greedy: bool = False,
    llm_model: Optional[str] = None,
    ep_accident_cache: Optional[Dict] = None,
) -> Tuple[RolloutBuffer, List[np.ndarray], float, int, int, str, dict]:
    """Run one episode with optional per-zone first customer forcing.
    Returns (buf, masks, ep_reward, ep_late_cnt, served_count, end_reason, env_snapshot).

    llm_model + ep_accident_cache: if provided, re-calls LLM when accident fires mid-routing.
    ep_accident_cache is shared across POMO rollouts for the same episode (LLM called once,
    result reused by all subsequent rollouts in the same episode).
    """
    obs, _ = env.reset()
    buf = RolloutBuffer()
    masks: List[np.ndarray] = []
    ep_reward = 0.0
    ep_late_cnt = 0
    terminated = False
    truncated = False
    step_in_ep = 0
    end_reason = "unknown"
    zone_first_done: set = set()
    _processed_acc: Set[int] = set()   # trigger_times already handled this rollout

    while not (terminated or truncated):
        step_in_ep += 1

        # ── Accident detection: re-call LLM when each accident fires ─────────
        if llm_model is not None and ep_accident_cache is not None and zone_plan is not None:
            for ev in env.active_events:
                if (ev.event_type == EventType.ACCIDENT
                        and ev.active
                        and int(ev.trigger_time) not in _processed_acc):
                    cache_key = int(ev.trigger_time)
                    _processed_acc.add(cache_key)

                    acc_nodes = set(ev.affected_nodes)
                    unserved = {c for c in range(1, env.N + 1) if env.served[c] == 0}
                    # Accident edge (A, B) is only relevant if BOTH endpoints are still unserved.
                    # If A is already served, the vehicle has passed A and won't traverse A→B again.
                    if len(ev.affected_nodes) < 2 or not acc_nodes.issubset(unserved):
                        continue

                    if cache_key not in ep_accident_cache:
                        # Only re-score zones that contain accident-affected stops
                        affected_zone_idx = [
                            z_idx for z_idx, z_stops in enumerate(zone_plan.zones)
                            if acc_nodes & set(z_stops)
                        ]
                        acc_zone_scores = query_ollama_instance_confidence(
                            env, zone_plan, llm_model, event=ev, use_cot=False,
                            prev_scores=llm_confidence_scores,
                            target_zone_indices=affected_zone_idx,
                            topk_ratio=cfg.llm_confidence_topk_ratio,
                        )
                        ep_accident_cache[cache_key] = {
                            c: s for zs in acc_zone_scores.values() for c, s in zs.items()
                        }
                        print(
                            f"  [LLM:acc] t={ev.trigger_time:.0f}  zones={affected_zone_idx}"
                            f"  affected={sorted(acc_nodes & unserved)}"
                            f"  → {len(ep_accident_cache[cache_key])} scores updated",
                            flush=True,
                        )
                    for c, s in ep_accident_cache[cache_key].items():
                        if env.served[c] == 0:
                            llm_confidence_scores[c] = s

        feasible, base_zone_idx, _ = get_feasible_actions_with_zone_control(env, zone_plan, cfg)
        if not feasible:
            truncated = True
            break

        candidate_actions, _, _, _ = get_candidate_actions(env, feasible, cfg, base_zone_idx, zone_plan)

        scope_custs = env._get_scope_customers()
        abs_to_rel: Dict[int, int] = {0: 0}
        rel_to_abs: Dict[int, int] = {0: 0}
        for _i, _c in enumerate(scope_custs):
            abs_to_rel[_c] = _i + 1
            rel_to_abs[_i + 1] = _c
        ca_pairs = [(abs_to_rel[a], a) for a in candidate_actions if a in abs_to_rel]
        candidate_actions_rel = [p[0] for p in ca_pairs]
        if not candidate_actions_rel:
            truncated = True
            break

        # LLM confidence vector for this step (depot=0; padding positions=0)
        conf_vec_np = np.zeros(max_scope, dtype=np.float32)
        for _rel, _abs in rel_to_abs.items():
            if _abs != 0:
                conf_vec_np[_rel] = llm_confidence_scores.get(_abs, 0.0)
        conf_vec_t = torch.tensor(conf_vec_np, dtype=torch.float32, device=cfg.device).unsqueeze(0)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)

        # Trust-gated LLM confidence bias is applied inside actor_critic.forward()
        with torch.no_grad():
            logits, value_t = actor_critic(obs_t, conf_vec_t)
        logits = logits[0]
        mask_t = torch.full_like(logits, float("-inf"))
        for a_rel in candidate_actions_rel:
            mask_t[a_rel] = 0.0
        masked_logits = logits + mask_t

        if greedy:
            action_rel = int(masked_logits.argmax().item())
            action_abs = rel_to_abs[action_rel]
        else:
            # Per-zone first customer forcing
            active_zone = int(env.active_zone_idx)
            forced_cust = (
                forced_first_per_zone.get(active_zone)
                if forced_first_per_zone and active_zone not in zone_first_done
                else None
            )
            if forced_cust is not None and forced_cust in abs_to_rel and abs_to_rel[forced_cust] in candidate_actions_rel:
                action_rel = abs_to_rel[forced_cust]
                zone_first_done.add(active_zone)
            else:
                if forced_cust is not None:
                    zone_first_done.add(active_zone)
                dist = Categorical(logits=masked_logits)
                action_rel = int(dist.sample().item())

            dist = Categorical(logits=masked_logits)
            log_prob = float(dist.log_prob(torch.tensor(action_rel, device=cfg.device)).item())
            value = float(value_t[0].item())
            action_abs = rel_to_abs[action_rel]

            mask_np = np.full(max_scope, float("-inf"), dtype=np.float32)
            for a_rel in candidate_actions_rel:
                mask_np[a_rel] = 0.0

        next_obs, reward, terminated, truncated, info = env.step(action_abs, external_penalty=0.0)
        ep_reward += reward
        if info.get("late", 0.0) > 0:
            ep_late_cnt += 1

        if not greedy:
            buf.push(obs, action_rel, float(reward), float(terminated or truncated), value, log_prob, conf_vec_np)
            masks.append(mask_np)
        obs = next_obs

        if step_in_ep >= cfg.max_steps_per_episode:
            truncated = True
            end_reason = "max_steps"
        if info.get("reason") == "vehicle_limit_reached":
            truncated = True
            end_reason = "vehicle_limit_reached"
        if info.get("reason") in ("all_served", "all_vehicles_retired"):
            end_reason = info["reason"]

    served_count = int(np.sum(env.served[1:]))
    env_snapshot = {
        "served": env.served.copy(),
        "vehicle_activated": {id(v): v.activated for v in env.vehicles},
        "route_nodes": {id(v): v.route_nodes[:] for v in env.vehicles},
        "total_distance": env.total_distance,
        "total_late": env.total_late,
    }
    return buf, masks, ep_reward, ep_late_cnt, served_count, end_reason, env_snapshot


# ============================================================
# 16) Main train
# ============================================================

def train(cfg: TrainConfig, algorithm: str = "ppo"):
    set_seed(cfg.seed)
    print(f"[Config] num_episodes={cfg.num_episodes}  rollout_steps={cfg.rollout_steps}  "
          f"ppo_epochs={cfg.ppo_epochs}  embed_dim={cfg.embed_dim}  n_heads={cfg.n_heads}")

    _mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("proposed_model")
        _mlflow_run = mlflow.start_run(run_name=f"{algorithm}_{os.path.basename(cfg.data_path).replace('.txt', '')}")
        mlflow.log_params({
            "algorithm":             algorithm,
            "num_episodes":          cfg.num_episodes,
            "lr":                    cfg.lr,
            "seed":                  cfg.seed,
            "data":                  os.path.basename(cfg.data_path),
            "use_llm":               cfg.use_llm_zone_confidence or cfg.use_llm_event_response,
            "use_ontology":          cfg.use_ontology_reasoning,
            "llm_model":             cfg.llm_model,
            "llm_confidence_lambda": cfg.llm_confidence_lambda,
            "use_learned_trust_gate": cfg.use_learned_trust_gate,
            "llm_confidence_topk_ratio": cfg.llm_confidence_topk_ratio,
            **{k: v for k, v in vars(cfg).items()
               if k in ("embed_dim", "n_heads", "n_encoder_layers", "rollout_steps",
                        "ppo_epochs", "entropy_coef", "late_penalty", "unserved_penalty")
               and isinstance(v, (int, float, str, bool))},
        })

    inst = load_waste_benchmark_txt(cfg.data_path)
    tt = build_travel_time_matrix(inst)

    zone_plan: Optional[ZonePlan] = None

    if cfg.use_solution_zones:
        sol_path = infer_solution_path(cfg.data_path, cfg.solution_path)
        if sol_path is not None:
            routes = parse_solution_routes(sol_path)
            validate_solution_routes(inst, routes)
            zone_plan = build_zone_plan_from_routes(inst, routes, cfg)
        else:
            print("[Zone] solution file not found — falling back to clustering")
            zone_plan = build_initial_zones(inst, cfg, tt)
    else:
        zone_plan = build_initial_zones(inst, cfg, tt)

    max_vehicles = inst.vehicle_limit if zone_plan is None else zone_plan.n_zones

    if zone_plan is not None:
        max_scope = max(len(z) for z in zone_plan.zones) + 1
    else:
        max_scope = inst.n_customers

    if zone_plan is not None:
        print("\n[Zone source]")
        if cfg.use_solution_zones:
            _sp = infer_solution_path(cfg.data_path, cfg.solution_path)
            print(f"  using solution routes as zones: {_sp if _sp else '(not found, used clustering)'}")
        else:
            total_demand = float(np.sum(inst.demands[1:]))
            print(
                f"  estimated vehicles by demand = ceil({total_demand:.1f} / {inst.vehicle_capacity:.1f})"
                f" = {estimate_vehicle_count(inst)}"
            )

        print(f"  max_vehicles = {max_vehicles}")
        for zi, zone in enumerate(zone_plan.zones):
            zone_demand = float(np.sum(inst.demands[zone]))
            due_min = float(np.min(inst.tw_close[zone]))
            due_max = float(np.max(inst.tw_close[zone]))
            print(
                f"  zone {zi:2d} | size={len(zone):3d} | "
                f"demand={zone_demand:8.1f}/{zone_plan.zone_capacity:8.1f} | "
                f"due_range=({due_min:.1f}, {due_max:.1f}) | "
                f"customers={zone}"
            )

        print("  hull-based adjacency:")
        for zi in range(zone_plan.n_zones):
            print(f"    zone {zi}: {sorted(zone_plan.adjacent_zones.get(zi, {zi}))}")

    env = BaselineEnv(
        inst=inst,
        tt=tt,
        max_vehicles=max_vehicles,
        late_count_penalty=cfg.late_count_penalty,
        late_penalty=cfg.late_penalty,
        unserved_penalty=cfg.unserved_penalty,
        max_scope_size=max_scope,
    )
    print(f"  obs_dim={env.observation_space.shape[0]} (max_scope_size={max_scope})")

    if zone_plan is not None:
        env.set_zone_assignment(
            zone_plan.customer_to_zone,
            zone_plan.n_zones,
            adjacent_zones=zone_plan.adjacent_zones,
        )

    ontology_engine: Optional[OntologyReasoningEngine] = None
    if cfg.use_ontology_reasoning:
        ontology_engine = OntologyReasoningEngine(
            inst=inst,
            zone_plan=zone_plan,
            base_unserved_penalty=cfg.unserved_penalty,
            penalty_threshold=cfg.ontology_penalty_threshold,
            penalty_scale_factor=cfg.ontology_penalty_scale_factor,
            penalty_max_multiplier=cfg.ontology_penalty_max_multiplier,
            num_episodes=cfg.num_episodes,
            unserved_window_ratio=cfg.ontology_unserved_window_ratio,
            min_warmup_ratio=cfg.ontology_min_warmup_ratio,
        )

    stop_feat_dim = env.stop_feat_dim
    vehicle_feat_dim = env.vehicle_feat_dim
    global_feat_dim = env.global_feat_dim

    actor_critic = AttentionActorCritic(
        stop_feat_dim=stop_feat_dim,
        max_scope_size=max_scope,
        vehicle_feat_dim=vehicle_feat_dim,
        global_feat_dim=global_feat_dim,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        n_encoder_layers=cfg.n_encoder_layers,
        event_feat_dim=getattr(env, "event_feat_dim", 0),
        llm_confidence_lambda_max=cfg.llm_confidence_lambda,
        learn_trust_gate=cfg.use_learned_trust_gate,
    ).to(cfg.device)

    optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.lr)
    rollout = RolloutBuffer()
    candidate_masks_list: List[np.ndarray] = []

    global_step = 0
    start_ep = 1
    ep_reward_history: List[float] = []
    best_ep_log: Optional[str] = None
    best_served: int = -1
    best_ep_reward: float = float('-inf')
    best_ep_routes: Optional[List[List[int]]] = None
    best_dist: float = 0.0
    best_late: float = 0.0
    best_late_cnt: int = 0
    train_start_time = time.time()
    ep_wall_times: List[float] = []

    if cfg.resume_from is not None:
        start_ep, global_step, best_served, best_ep_reward, best_late_cnt = load_checkpoint(
            path=cfg.resume_from,
            actor_critic=actor_critic,
            optimizer=optimizer,
            ontology_engine=ontology_engine,
            device=cfg.device,
        )
        if ontology_engine is not None:
            pass  # ontology adjusts action pruning only; reward is episode-level

    zone_confidence_cache: Dict[int, Dict[int, float]] = {}

    _solution_routes: Optional[List[List[int]]] = None
    if cfg.use_llm_zone_confidence and cfg.solution_path is not None:
        try:
            _sol_path = infer_solution_path(cfg.data_path, cfg.solution_path)
            _solution_routes = parse_solution_routes(_sol_path)
        except Exception:
            _solution_routes = None

    if cfg.use_llm_zone_confidence and zone_plan is not None:
        env.reset()
        print("[LLM] Pre-populating zone confidence cache for all zones...")
        total_match = 0
        total_stops = 0
        for z_idx, z_stops in enumerate(zone_plan.zones):
            t0 = time.time()
            scores, raw_output = query_ollama_zone_confidence(
                env=env,
                zone_stops=z_stops,
                zone_idx=z_idx,
                model_name=cfg.llm_model,
                high_unserved=None,
                use_cot=cfg.llm_use_cot,
                temperature=cfg.llm_temperature,
            )
            elapsed = time.time() - t0
            zone_confidence_cache[z_idx] = scores
            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
            llm_order = [s for s, _ in sorted_scores]
            score_str = " → ".join(f"stop{s}({v:.2f})" for s, v in sorted_scores)
            match_str = ""
            if _solution_routes is not None and z_idx < len(_solution_routes):
                sol_order = _solution_routes[z_idx]
                match = sum(1 for i, s in enumerate(sol_order) if i < len(llm_order) and llm_order[i] == s)
                total_match += match
                total_stops += len(sol_order)
                match_str = f" | SOL match: {match}/{len(sol_order)}"
            print(f"  [LLM:Zone{z_idx}] {elapsed:.1f}s{match_str} | {score_str}")
            print(f"  [LLM:Raw] {raw_output}")
        if total_stops > 0:
            print(f"[LLM] Total match: {total_match}/{total_stops} ({total_match/total_stops:.1%})")
        print("[LLM] Pre-population complete.\n")

    # Build POMO starting points
    pomo_k = cfg.pomo_k
    # LLM-guided starts are built once (confidence scores are stable across episodes)
    pomo_zone_firsts_llm: List[Dict[int, int]] = []
    if algorithm == "pomo" and zone_plan is not None and zone_confidence_cache:
        for k in range(pomo_k):
            forced: Dict[int, int] = {}
            for z_idx, scores in zone_confidence_cache.items():
                ranked = sorted(scores.items(), key=lambda x: -x[1])
                # Cycle through ranked list — guarantees unique assignment up to len(ranked)
                forced[z_idx] = ranked[k % len(ranked)][0]
            pomo_zone_firsts_llm.append(forced)
        print(f"[POMO] {pomo_k} LLM-guided starting points per zone built (fixed across episodes).")
        for k, f in enumerate(pomo_zone_firsts_llm):
            print(f"  rollout {k+1}: " + ", ".join(f"zone{z}→cust{c}" for z, c in sorted(f.items())))

    # Base LLM confidence scores (zone cache flattened, used across POMO rollouts)
    llm_confidence_scores_base: Dict[int, float] = {}
    if cfg.use_llm_zone_confidence and zone_confidence_cache:
        for scores in zone_confidence_cache.values():
            llm_confidence_scores_base.update(scores)

    for ep in range(start_ep, cfg.num_episodes + 1):
        progress = (ep - 1) / max(cfg.num_episodes - 1, 1)
        current_lr = cfg.lr + (cfg.lr_min - cfg.lr) * progress
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        current_entropy_coef = cfg.entropy_coef + (cfg.entropy_coef_min - cfg.entropy_coef) * progress

        ep_start_time = time.time()
        ep_reward = 0.0
        ep_loss = 0.0
        ep_train_steps = 0
        ep_late_cnt = 0
        llm_used_count = 0
        fallback_used_count = 0
        ontology_used_count = 0

        # ── POMO branch ──────────────────────────────────────────────────────
        if algorithm == "pomo":
            if pomo_zone_firsts_llm:
                pomo_zone_firsts = pomo_zone_firsts_llm
            else:
                # Resample each episode; shuffle zone pools then assign cyclically (no duplicate starts)
                zone_pools = {
                    z_idx: [c for c in z_custs if c > 0]
                    for z_idx, z_custs in enumerate(zone_plan.zones) if z_custs
                } if zone_plan is not None else {}
                zone_shuffled = {z_idx: random.sample(pool, len(pool)) for z_idx, pool in zone_pools.items()}
                pomo_zone_firsts = [
                    {z_idx: sh[k % len(sh)] for z_idx, sh in zone_shuffled.items()}
                    for k in range(pomo_k)
                ]
            bufs_k: List[RolloutBuffer] = []
            masks_k: List[List[np.ndarray]] = []
            rewards_k: List[float] = []
            late_k: List[int] = []
            served_k: List[int] = []
            snaps_k: List[dict] = []
            for k in range(pomo_k):
                forced = pomo_zone_firsts[k] if k < len(pomo_zone_firsts) else {}
                buf, msk, r, lc, sc, _, snap = _run_pomo_episode(
                    env, cfg, actor_critic, zone_plan,
                    llm_confidence_scores_base, max_scope, forced,
                )
                bufs_k.append(buf)
                masks_k.append(msk)
                rewards_k.append(r)
                late_k.append(lc)
                served_k.append(sc)
                snaps_k.append(snap)
                global_step += len(buf)
            ep_loss = pomo_update(
                actor_critic, optimizer, bufs_k, masks_k, rewards_k, cfg,
                entropy_coef=current_entropy_coef,
            )
            ep_train_steps = 1
            on_time_k = [served_k[i] - late_k[i] for i in range(len(served_k))]
            best_k = max(range(len(rewards_k)), key=lambda i: (on_time_k[i], rewards_k[i]))
            ep_reward = rewards_k[best_k]
            ep_late_cnt = late_k[best_k]
            served_count = served_k[best_k]
            end_reason = f"pomo_k={pomo_k}"
            # Restore best rollout's env state for logging
            best_snap = snaps_k[best_k]
            env.served = best_snap["served"]
            env.total_distance = best_snap["total_distance"]
            env.total_late = best_snap["total_late"]
            env.total_late_violation_penalty = best_snap["total_late_violation_penalty"]
            for v in env.vehicles:
                v_id = id(v)
                v.activated = best_snap["vehicle_activated"].get(v_id, v.activated)
                v.route_nodes = best_snap["route_nodes"].get(v_id, v.route_nodes)
        elif algorithm == "kool":
            # ── Kool 2019 branch ─────────────────────────────────────────────────
            # Stochastic rollout — collect trajectory for gradient
            buf_s, msk_s, r_s, ep_late_cnt, served_count, end_reason, snap_s = _run_pomo_episode(
                env, cfg, actor_critic, zone_plan,
                llm_confidence_scores_base, max_scope,
                forced_first_per_zone=None, greedy=False,
            )
            # Save stochastic rollout's env state before greedy rollout overwrites it
            stoch_served = snap_s["served"]
            stoch_dist = snap_s["total_distance"]
            stoch_late = snap_s["total_late"]
            stoch_late_pen = snap_s["total_late_violation_penalty"]
            stoch_routes = snap_s["route_nodes"]
            stoch_activated = snap_s["vehicle_activated"]

            # Greedy rollout — argmax baseline, no gradient
            _, _, r_g, _, _, _, _ = _run_pomo_episode(
                env, cfg, actor_critic, zone_plan,
                llm_confidence_scores_base, max_scope,
                forced_first_per_zone=None, greedy=True,
            )

            # Restore stochastic rollout's env state for logging
            env.served = stoch_served
            env.total_distance = stoch_dist
            env.total_late = stoch_late
            env.total_late_violation_penalty = stoch_late_pen
            for v in env.vehicles:
                v_id = id(v)
                v.activated = stoch_activated.get(v_id, v.activated)
                v.route_nodes = stoch_routes.get(v_id, v.route_nodes)

            ep_loss = kool_update(
                actor_critic, optimizer, buf_s, msk_s, r_s, r_g,
                cfg, entropy_coef=current_entropy_coef,
            )
            ep_reward = r_s
            global_step += len(buf_s)
            ep_train_steps = 1
        else:
            # ── PPO branch ───────────────────────────────────────────────────────
            obs, _ = env.reset()

            llm_confidence_scores: Dict[int, float] = {}
            env._llm_processed_events = set()
            llm_scored_zones: Set[int] = set()

            if cfg.use_llm_event_response:
                rain_events = [e for e in env.active_events if e.event_type == EventType.RAIN]
                for rain_ev in rain_events:
                    zone_stops = list(rain_ev.affected_nodes)
                    scores, _ = query_ollama_confidence(
                        env=env,
                        feasible=zone_stops,
                        event=rain_ev,
                        zone_plan=zone_plan,
                        base_zone_idx=None,
                        model_name=cfg.llm_model,
                        use_cot=cfg.llm_use_cot,
                        temperature=cfg.llm_temperature,
                        topk_ratio=cfg.llm_confidence_topk_ratio,
                    )
                    llm_confidence_scores.update(scores)
                    llm_used_count += 1
                    print(f"  [LLM:Rain] zone stops={zone_stops} "
                          f"t={rain_ev.trigger_time:.0f}~{rain_ev.end_time:.0f} "
                          f"rainfall={rain_ev.rainfall_mm:.0f}mm/h")

            terminated = False
            truncated = False
            step_in_ep = 0
            end_reason = "unknown"
            info: dict = {}

            while not (terminated or truncated):
                step_in_ep += 1
                global_step += 1

                feasible, base_zone_idx, _ = get_feasible_actions_with_zone_control(
                    env=env,
                    zone_plan=zone_plan,
                    cfg=cfg,
                )

                if (cfg.use_llm_zone_confidence
                        and base_zone_idx is not None
                        and base_zone_idx not in llm_scored_zones
                        and zone_plan is not None):
                    zone_stops = [c for c in zone_plan.zones[base_zone_idx]
                                  if env.served[c] == 0]
                    if zone_stops:
                        if base_zone_idx in zone_confidence_cache:
                            scores = {c: zone_confidence_cache[base_zone_idx].get(c, 0.0)
                                      for c in zone_stops}
                        else:
                            high_unserved = (
                                ontology_engine.get_high_unserved_stops(base_zone_idx, zone_plan)
                                if ontology_engine is not None and cfg.use_ontology_reasoning
                                else None
                            )
                            prev_scores = zone_confidence_cache.get(base_zone_idx)
                            scores, raw_output = query_ollama_zone_confidence(
                                env=env,
                                zone_stops=zone_plan.zones[base_zone_idx],
                                zone_idx=base_zone_idx,
                                model_name=cfg.llm_model,
                                high_unserved=high_unserved,
                                use_cot=cfg.llm_use_cot,
                                temperature=cfg.llm_temperature,
                                prev_scores=prev_scores,
                                topk_ratio=cfg.llm_confidence_topk_ratio,
                            )
                            zone_confidence_cache[base_zone_idx] = scores
                            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
                            score_str = " → ".join(f"stop{s}({v:.2f})" for s, v in sorted_scores)
                            print(f"  [LLM:Zone{base_zone_idx}] {score_str}")
                            print(f"  [LLM:Raw] {raw_output}")
                            if _solution_routes is not None and base_zone_idx < len(_solution_routes):
                                sol_order = _solution_routes[base_zone_idx]
                                print(f"  [SOL:Zone{base_zone_idx}] " + " → ".join(f"stop{s}" for s in sol_order))
                        llm_used_count += 1
                        llm_confidence_scores.update(scores)
                        llm_scored_zones.add(base_zone_idx)

                candidate_actions, prune_source, ontology_facts, llm_reasoning = get_candidate_actions(
                    env=env,
                    feasible=feasible,
                    cfg=cfg,
                    base_zone_idx=base_zone_idx,
                    zone_plan=zone_plan,
                )

                if prune_source == "llm":
                    llm_used_count += 1
                elif prune_source == "fallback":
                    fallback_used_count += 1
                elif prune_source == "ontology":
                    ontology_used_count += 1

                scope_custs = env._get_scope_customers()
                abs_to_rel = {0: 0}
                rel_to_abs = {0: 0}
                for _i, _c in enumerate(scope_custs):
                    abs_to_rel[_c] = _i + 1
                    rel_to_abs[_i + 1] = _c
                ca_pairs = [(abs_to_rel[a], a) for a in candidate_actions if a in abs_to_rel]
                candidate_actions_rel = [p[0] for p in ca_pairs]

                if cfg.use_llm_event_response:
                    newly_triggered = [
                        e for e in env.active_events
                        if e.event_type == EventType.ACCIDENT and e.active
                        and e not in getattr(env, "_llm_processed_events", set())
                    ]
                    for acc_ev in newly_triggered:
                        acc_owl_ctx = None
                        scores, _ = query_ollama_confidence(
                            env=env,
                            feasible=candidate_actions,
                            event=acc_ev,
                            zone_plan=zone_plan,
                            base_zone_idx=base_zone_idx,
                            model_name=cfg.llm_model,
                            use_cot=cfg.llm_use_cot,
                            temperature=cfg.llm_temperature,
                            owl_context=acc_owl_ctx,
                            topk_ratio=cfg.llm_confidence_topk_ratio,
                        )
                        llm_confidence_scores.update(scores)
                        env._llm_processed_events.add(acc_ev)
                        llm_used_count += 1
                        print(f"  [LLM:Accident] t={env.vehicles[env.active_vehicle_idx].cur_time:.0f} "
                              f"accident detected, confidence scores updated.")

                    for ev in env.active_events:
                        if ev.recovered and ev in getattr(env, "_llm_processed_events", set()):
                            for c in ev.affected_nodes:
                                llm_confidence_scores.pop(c, None)
                            env._llm_processed_events.discard(ev)

                if cfg.use_fixed_solution_policy:
                    action_abs = select_fixed_solution_action(env, zone_plan)
                    if action_abs not in candidate_actions:
                        action_abs = 0 if 0 in candidate_actions else (candidate_actions[0] if candidate_actions else -1)
                    action_rel = abs_to_rel.get(action_abs, 0)
                    log_prob, value = 0.0, 0.0
                else:
                    if len(candidate_actions_rel) == 0:
                        action_abs = -1
                        action_rel = -1
                        log_prob, value = 0.0, 0.0
                    else:
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                        with torch.no_grad():
                            use_conf = (llm_confidence_scores and (
                                cfg.use_llm_event_response or cfg.use_llm_zone_confidence
                            ))
                            if use_conf:
                                logits, value_t = actor_critic.forward(obs_t)
                                logits = logits[0]
                                if ep <= 3 and step_in_ep <= 5:
                                    cand_logits = logits[candidate_actions_rel]
                                    print(f"  [logit] ep={ep} step={step_in_ep} "
                                          f"range=[{cand_logits.min():.3f}, {cand_logits.max():.3f}] "
                                          f"mean={cand_logits.mean():.3f}")
                                mask_t = torch.full_like(logits, float("-inf"))
                                for a_rel in candidate_actions_rel:
                                    mask_t[a_rel] = 0.0
                                masked_logits = logits + mask_t
                                conf_bias = torch.zeros_like(logits)
                                for a_rel, a_abs in ca_pairs:
                                    conf = llm_confidence_scores.get(a_abs, 0.0)
                                    conf_bias[a_rel] = cfg.llm_confidence_lambda * conf
                                final_logits = masked_logits + conf_bias
                                dist = torch.distributions.Categorical(logits=final_logits)
                                action_rel = int(dist.sample().item())
                                log_prob = float(dist.log_prob(
                                    torch.tensor(action_rel, device=obs_t.device)
                                ).item())
                                value = float(value_t[0].item())
                                action_abs = rel_to_abs[action_rel]
                            else:
                                action_rel, log_prob, value = actor_critic.get_action_and_value(
                                    obs_t, candidate_actions_rel
                                )
                                action_abs = rel_to_abs[action_rel]

                if action_abs < 0:
                    truncated = True
                    break

                mask = np.full(max_scope, float("-inf"), dtype=np.float32)
                for a_rel in candidate_actions_rel:
                    mask[a_rel] = 0.0

                next_obs, reward, terminated, truncated, info = env.step(
                    action_abs,
                    external_penalty=0.0,
                )
                done_flag = float(terminated or truncated)
                ep_reward += reward
                if info.get("late", 0.0) > 0:
                    ep_late_cnt += 1

                if not cfg.use_fixed_solution_policy:
                    rollout.push(obs, action_rel, float(reward), done_flag, value, log_prob)
                    candidate_masks_list.append(mask)

                    if len(rollout) >= cfg.rollout_steps:
                        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                        with torch.no_grad():
                            _, last_val = actor_critic(obs_t)
                            last_val = float(last_val[0].item())
                        loss = ppo_update(actor_critic, optimizer, rollout, candidate_masks_list, last_val, cfg, entropy_coef=current_entropy_coef)
                        ep_loss += loss
                        ep_train_steps += 1
                        rollout.clear()
                        candidate_masks_list.clear()

                obs = next_obs

                if step_in_ep >= cfg.max_steps_per_episode:
                    truncated = True
                    end_reason = "max_steps"
                if info.get("reason") == "vehicle_limit_reached":
                    truncated = True
                    end_reason = "vehicle_limit_reached"
                if info.get("reason") in ("all_served", "all_vehicles_retired"):
                    end_reason = info["reason"]

            if not cfg.use_fixed_solution_policy and len(rollout) > 0:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                with torch.no_grad():
                    _, last_val = actor_critic(obs_t)
                    last_val = float(last_val[0].item())
                loss = ppo_update(actor_critic, optimizer, rollout, candidate_masks_list, last_val, cfg, entropy_coef=current_entropy_coef)
                ep_loss += loss
                ep_train_steps += 1
                rollout.clear()
                candidate_masks_list.clear()

            served_count = int(np.sum(env.served[1:]))
        on_time = served_count - ep_late_cnt
        unserved = env.N - served_count
        avg_loss = ep_loss / max(ep_train_steps, 1)
        ep_reward_history.append(ep_reward)
        avg_late = env.total_late / max(served_count, 1)
        mode_name = "FIXED" if cfg.use_fixed_solution_policy else "RL"
        ep_wall = time.time() - ep_start_time
        ep_wall_times.append(ep_wall)

        if ontology_engine is not None:
            chunk_fired = ontology_engine.record_episode(env.served)

            if chunk_fired and cfg.use_llm_zone_confidence and cfg.llm_recall_on_chunk:
                zone_confidence_cache.clear()
                print(f"  [OWL:ChunkFired] zone_confidence_cache cleared → LLM will re-score all zones")

        zone_served_str = ""
        if zone_plan is not None:
            parts = []
            for z_custs in zone_plan.zones:
                z_served = sum(1 for c in z_custs if env.served[c] == 1)
                parts.append(f"{z_served}/{len(z_custs)}")
            zone_served_str = "  zones=[" + " ".join(parts) + "]"

        ep_log = (
            f"[{mode_name} EP {ep:4d}] "
            f"reward={ep_reward:10.2f}  "
            f"served={on_time:3d}  tw_late={ep_late_cnt:3d}  unserved={unserved:3d}/{env.N}"
            f"{zone_served_str}  "
            f"veh={env.vehicles_used:2d}/{max_vehicles:2d}  "
            f"dist={env.total_distance:10.2f}  "
            f"late={env.total_late:9.2f}  "
            f"late_avg={avg_late:8.2f}  "
            f"late_pen={env.total_late_violation_penalty:9.2f}  "
            f"loss={avg_loss:8.4f}  "
            f"ont={ontology_used_count:4d}  "
            f"llm={llm_used_count:4d}  "
            f"fb={fallback_used_count:4d}  "
            f"ep_time={ep_wall:.1f}s  "
            f"done={end_reason}"
        )
        if MLFLOW_AVAILABLE and _mlflow_run is not None:
            mlflow.log_metrics({
                "ep_reward":   ep_reward,
                "served":      on_time,
                "tw_late":     ep_late_cnt,
                "unserved":    unserved,
                "late_avg":    avg_late,
                "loss":        avg_loss,
                "dist":        env.total_distance,
                "llm_calls":   llm_used_count,
                "ont_calls":   ontology_used_count,
            }, step=ep)

        best_on_time_so_far = best_served - best_late_cnt  # -1 initially
        if on_time > best_on_time_so_far or (on_time == best_on_time_so_far and ep_reward > (best_ep_reward if best_ep_log else float('-inf'))):
            best_served = served_count
            best_ep_reward = ep_reward
            best_ep_log = ep_log
            best_ep_routes = [v.route_nodes[:] for v in env.vehicles]
            best_dist = env.total_distance
            best_late = env.total_late
            best_late_cnt = ep_late_cnt
            print(f"[Best EP {ep:4d}] " + ep_log)

        if cfg.checkpoint_path is not None and ep % cfg.checkpoint_interval == 0:
            save_checkpoint(
                path=cfg.checkpoint_path,
                ep=ep,
                global_step=global_step,
                actor_critic=actor_critic,
                optimizer=optimizer,
                ontology_engine=ontology_engine,
                best_served=best_served,
                best_ep_reward=best_ep_reward,
                best_late_cnt=best_late_cnt,
            )

    total_train_time = time.time() - train_start_time
    avg_ep_time = sum(ep_wall_times) / max(len(ep_wall_times), 1)
    print(f"\n[Time] total={total_train_time:.1f}s  "
          f"avg_per_ep={avg_ep_time:.2f}s  "
          f"episodes={len(ep_wall_times)}")

    if best_ep_log is not None:
        label = f"RL+LLM+Ontology ({os.path.splitext(os.path.basename(cfg.data_path))[0]})"
        best_on_time = best_served - best_late_cnt
        best_unserved = env.N - best_served
        print(f"\n{'='*80}")
        print(f"[Best Episode] Summary  episodes={cfg.num_episodes}")
        print(f"  {'Experiment':<35} {'Served':>7} {'TWLate':>7} {'Unserved':>9} {'Dist':>8} {'Late':>8}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*9} {'-'*8} {'-'*8}")
        print(f"  {label:<35} {best_on_time:>4}/{env.N:<3} {best_late_cnt:>4}/{env.N:<3} {best_unserved:>6}/{env.N:<3} {best_dist:>8.1f} {best_late:>8.1f}")
        print(f"{'='*75}")
        print(f"\n[Best Episode Log]")
        print(best_ep_log)

    if best_ep_routes is not None:
        _plot_routing(best_ep_routes, env, cfg)

    _plot_reward_curve(ep_reward_history, cfg)

    if MLFLOW_AVAILABLE and _mlflow_run is not None:
        mlflow.log_metric("best_served", best_served)
        mlflow.log_metric("best_ep_reward", best_ep_reward)
        mlflow.log_metric("total_train_time_s", total_train_time)
        _base = cfg.checkpoint_path.replace("_ckpt.pkl", "") if cfg.checkpoint_path else "."
        routing_path = f"{_base}_routing_best.png"
        reward_path  = f"{_base}_reward_curve.png"
        for p in (routing_path, reward_path):
            if os.path.exists(p):
                mlflow.log_artifact(p)
        mlflow.end_run()

    return actor_critic, env, zone_plan


# ============================================================
# 16) Entry
# ============================================================

def train_multi_llm(
    data_dir: str,
    ckpt_dir: str,
    num_episodes: int = 4000,
    algorithm: str = "pomo",
    no_llm: bool = False,
    no_ontology: bool = False,
    llm_model: str = "qwen3:32b",
    seed: int = 42,
    resume: Optional[str] = None,
    result_dir: Optional[str] = None,
    pomo_start_ratio: float = 1.0,
    train_base: Optional[List[str]] = None,
    test_base: Optional[List[str]] = None,
    llm_lambda: float = 1.0,
    llm_topk_ratio: float = 1.0,
    learn_trust_gate: bool = True,
):
    """Multi-instance proposed model training.

    Training pool: base (c102-c109) + rain/acc event variants, sampled uniformly i.i.d.
    Test: c101 base + rain_A/B + acc_A/B
    """
    from R_rl import _TRAIN_BASE

    set_seed(seed)
    algorithm = "pomo"  # training loop always runs POMO; pin to keep eval/ckpt labels accurate

    base_list = train_base if train_base else _TRAIN_BASE
    test_base_list = test_base if test_base else ["c101"]
    test_list = []
    for _tb in test_base_list:
        test_list += [_tb, f"{_tb}_rain_A", f"{_tb}_rain_B", f"{_tb}_acc_A", f"{_tb}_acc_B"]
    event_list = (
        [f"{n}_rain_A" for n in base_list] +
        [f"{n}_rain_B" for n in base_list] +
        [f"{n}_acc_A"  for n in base_list] +
        [f"{n}_acc_B"  for n in base_list]
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    def _make_cfg(path):
        _sol = infer_solution_path(path, None)
        return TrainConfig(
            data_path=path, use_solution_zones=False,
            solution_path=str(_sol) if _sol else None,
            prefer_on_time_candidates=False,
            use_ontology_pruning=not no_ontology,
            use_ontology_reasoning=not no_ontology,
            use_llm_zone_confidence=not no_llm,
            use_llm_event_response=not no_llm,
            llm_model=llm_model,
            llm_confidence_lambda=llm_lambda,
            use_learned_trust_gate=learn_trust_gate,
            llm_confidence_topk_ratio=llm_topk_ratio,
            num_episodes=num_episodes,
            seed=seed,
            checkpoint_path=os.path.join(ckpt_dir, "proposed_multi.pkl"),
            checkpoint_interval=1000,
            resume_from=resume,
            **SHARED_HYPERPARAMS,
        )

    # Load all instances + build zone plans
    print("[ProposedMulti] Pre-loading instances...", flush=True)
    _cache = {}
    for name in (base_list + event_list + test_list):
        path = os.path.join(data_dir, name + ".txt")
        try:
            inst = load_waste_benchmark_txt(path)
            tt   = build_travel_time_matrix(inst)
            cfg  = _make_cfg(path)
            # Event instances derive zones from base's data (same clustering across variants)
            base_name = name.split("_rain_")[0].split("_acc_")[0]
            base_path = os.path.join(data_dir, base_name + ".txt")
            sol  = infer_solution_path(base_path, None)
            zp   = (build_zone_plan_from_routes(inst, parse_solution_routes(sol), cfg)
                    if (cfg.use_solution_zones and sol) else build_initial_zones(inst, cfg, tt))
            ms   = max(len(z) for z in zp.zones) + 1 if zp else inst.n_customers
            env  = BaselineEnv(
                inst=inst, tt=tt,
                max_vehicles=zp.n_zones if zp else inst.vehicle_limit,
                late_penalty=cfg.late_penalty,
                unserved_penalty=cfg.unserved_penalty,
                max_scope_size=ms,
            )
            if zp:
                env.set_zone_assignment(zp.customer_to_zone, zp.n_zones,
                                        adjacent_zones=zp.adjacent_zones)
            _cache[name] = (inst, tt, zp, cfg, env, ms)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    _MS = max(ms for _, _, _, _, _, ms in _cache.values())
    print(f"[ProposedMulti] max_scope={_MS}  cached={len(_cache)} instances", flush=True)

    # Rebuild envs with unified _MS
    for name, (inst, tt, zp, cfg, _, _) in list(_cache.items()):
        env = BaselineEnv(
            inst=inst, tt=tt,
            max_vehicles=zp.n_zones if zp else inst.vehicle_limit,
            late_penalty=cfg.late_penalty,
            unserved_penalty=cfg.unserved_penalty,
            max_scope_size=_MS,
        )
        if zp:
            env.set_zone_assignment(zp.customer_to_zone, zp.n_zones,
                                    adjacent_zones=zp.adjacent_zones)
        _cache[name] = (inst, tt, zp, cfg, env, _MS)

    # Pre-compute LLM zone confidence — saved to result_dir (per-setting) or ckpt_dir (fallback).
    # Storing in result_dir means each experiment setting gets its own cache; no manual cleanup needed.
    import pickle as _pickle
    _cache_dir = result_dir if result_dir else ckpt_dir
    os.makedirs(_cache_dir, exist_ok=True)
    from R_llm_module import _USE_THINK
    _cot_tag = "cot" if _USE_THINK else "nocot"
    _llm_cache_path = os.path.join(_cache_dir, f"llm_conf_cache_{llm_model.replace(':','_')}_{_cot_tag}.pkl")

    llm_conf_cache: Dict[str, Dict[int, Dict[int, float]]] = {}
    if not no_llm:
        if os.path.isfile(_llm_cache_path):
            with open(_llm_cache_path, "rb") as _f:
                llm_conf_cache = _pickle.load(_f)
            print(f"[ProposedMulti] Loaded LLM cache from {_llm_cache_path} "
                  f"({len(llm_conf_cache)} instances)", flush=True)
        else:
            print("[ProposedMulti] Pre-computing LLM zone confidence...", flush=True)
        # Include test base instances (e.g. c101) so test eval also gets LLM confidence
        _test_base_names = sorted({
            n.split("_rain_")[0].split("_acc_")[0]
            for n in test_list
        } - set(base_list))
        # Skip instances already in loaded cache
        _names_to_compute = [n for n in list(base_list) + _test_base_names
                             if n not in llm_conf_cache]
        if not _names_to_compute:
            print("[ProposedMulti] All instances already in cache — skipping LLM pre-computation",
                  flush=True)

        for name in _names_to_compute:
            inst, tt, zp, cfg_inst, env, _ = _cache[name]
            if zp is None:
                continue
            env.reset()
            base_scores = query_ollama_instance_confidence(
                env=env, zone_plan=zp, model_name=llm_model, use_cot=True,
                topk_ratio=cfg_inst.llm_confidence_topk_ratio,
            )
            llm_conf_cache[name] = base_scores
            print(f"  [LLM] {name} (base): {len(base_scores)} zones", flush=True)

            for suffix in ("_rain_A", "_rain_B", "_acc_A", "_acc_B"):
                ev_name = name + suffix
                if ev_name not in _cache or ev_name in llm_conf_cache:
                    continue

                if suffix in ("_rain_A", "_rain_B"):
                    ev_inst, _, ev_zp, _, ev_env, _ = _cache[ev_name]
                    ev_env.reset()
                    rain_evs = [e for e in (ev_inst.preset_events or [])
                                if e.event_type == EventType.RAIN]
                    if rain_evs:
                        rain_ev = max(rain_evs, key=lambda e: e.duration)
                        rain_scores = query_ollama_instance_confidence(
                            env=ev_env, zone_plan=ev_zp,
                            model_name=llm_model, event=rain_ev, use_cot=False,
                            topk_ratio=cfg_inst.llm_confidence_topk_ratio,
                        )
                        llm_conf_cache[ev_name] = rain_scores
                        print(f"  [LLM] {ev_name} (rain-aware): {len(rain_scores)} zones", flush=True)
                    else:
                        llm_conf_cache[ev_name] = base_scores

                elif suffix in ("_acc_A", "_acc_B"):
                    llm_conf_cache[ev_name] = base_scores

            # Save after each base instance so partial progress is preserved
            with open(_llm_cache_path, "wb") as _f:
                _pickle.dump(llm_conf_cache, _f)
            print(f"  [Cache] saved → {_llm_cache_path}", flush=True)

    # Init model
    _, _, _, _, env0, _ = _cache[base_list[0]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor_critic = AttentionActorCritic(
        stop_feat_dim=env0.stop_feat_dim,
        max_scope_size=_MS,
        vehicle_feat_dim=env0.vehicle_feat_dim,
        global_feat_dim=env0.global_feat_dim,
        embed_dim=SHARED_HYPERPARAMS.get("embed_dim", 128),
        n_heads=SHARED_HYPERPARAMS.get("n_heads", 8),
        n_encoder_layers=SHARED_HYPERPARAMS.get("n_encoder_layers", 3),
        llm_confidence_lambda_max=llm_lambda,
        learn_trust_gate=learn_trust_gate,
    ).to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=SHARED_HYPERPARAMS.get("lr", 1e-4))

    start_ep = 1
    if resume:
        start_ep, _, _, _, _ = load_checkpoint(resume, actor_critic, optimizer, None, device)

    print(f"[ProposedMulti] Training: episodes={num_episodes}  "
          f"algorithm={algorithm}  device={device}", flush=True)
    print(f"  Pool: {base_list} + {len(event_list)} event instances (i.i.d. sampling)", flush=True)
    t0 = time.time()
    instance_counts: Dict[str, int] = {}

    pool = base_list + event_list
    for ep in range(start_ep, num_episodes + 1):
        name = random.choice(pool)
        instance_counts[name] = instance_counts.get(name, 0) + 1
        inst, tt, zp, cfg, env, _ = _cache[name]
        cfg.device = device
        zone_conf = llm_conf_cache.get(name, {})

        # Get per-zone confidence scores
        all_conf: Dict[int, float] = {}
        if zp and zone_conf:
            for z_idx, scores in zone_conf.items():
                all_conf.update(scores)

        # POMO starts: pomo_start_ratio controls fraction of zone stops used
        max_zone = max(len(z) for z in zp.zones) if zp else _MS - 1
        pomo_k = max(1, round(max_zone * pomo_start_ratio))
        rewards_k, bufs_k, masks_k = [], [], []
        # acc instances: share accident-LLM cache across K rollouts (LLM called once per episode)
        # Both acc_A and acc_B get the same treatment — fair comparison:
        # acc_A accidents are on route edges (real impact), acc_B on non-route edges (no impact);
        # the LLM call happens identically either way, difference comes from actual routing outcome.
        is_acc = "_acc_" in name
        ep_accident_cache: Dict = {} if (is_acc and not no_llm) else None
        ep_llm_model = llm_model if (is_acc and not no_llm) else None

        for _ in range(pomo_k):
            ep_conf = dict(all_conf)   # per-rollout copy; accident updates stay local
            buf, masks, ep_reward, _, _, _, _ = _run_pomo_episode(
                env=env, cfg=cfg, actor_critic=actor_critic,
                zone_plan=zp, llm_confidence_scores=ep_conf,
                max_scope=_MS,
                llm_model=ep_llm_model,
                ep_accident_cache=ep_accident_cache,
            )
            rewards_k.append(ep_reward); bufs_k.append(buf); masks_k.append(masks)
            # fresh env for next rollout
            env2 = BaselineEnv(
                inst=inst, tt=tt,
                max_vehicles=zp.n_zones if zp else inst.vehicle_limit,
                late_penalty=cfg.late_penalty,
                unserved_penalty=cfg.unserved_penalty,
                max_scope_size=_MS,
            )
            if zp:
                env2.set_zone_assignment(zp.customer_to_zone, zp.n_zones,
                                         adjacent_zones=zp.adjacent_zones)
            env = env2
            _cache[name] = (inst, tt, zp, cfg, env, _MS)

        pomo_update(actor_critic, optimizer, bufs_k, masks_k, rewards_k, cfg)

        if ep % 50 == 0:
            avg_r = sum(rewards_k) / max(len(rewards_k), 1)
            print(f"ep={ep:4d}/{num_episodes}"
                  f"  avg_r={avg_r:.2f}  elapsed={time.time()-t0:.0f}s", flush=True)

        if ep % max(200, num_episodes // 5) == 0 or ep == num_episodes:
            eval_lines = [
                f"EVALUATION  ep={ep}  [LLM+RL  alg={algorithm}  seed={seed}]",
                "=" * 75,
                f"  {'Scenario':<22}  {'Served':>6}  {'Unserved':>8}  {'Veh':>3}"
                f"  {'Dist':>8}  {'LateCnt':>7}  {'LateMin':>8}",
                "-" * 75,
            ]
            print(f"\n{'='*65}\n  {eval_lines[0]}\n{'='*65}")
            for tname in test_list:
                try:
                    _, _, tzp, tcfg, tenv, _ = _cache[tname]
                    tcfg.device = device
                    base_tname = tname.split("_rain_")[0].split("_acc_")[0]
                    tconf = llm_conf_cache.get(tname, llm_conf_cache.get(base_tname, {}))
                    tall_conf: Dict[int, float] = {}
                    for zs in tconf.values():
                        tall_conf.update(zs)
                    is_test_acc = "_acc_" in tname
                    t_ep_acc_cache: Dict = {} if (is_test_acc and not no_llm) else None
                    t_llm_model = llm_model if (is_test_acc and not no_llm) else None
                    _, _, tr, tlc, tsc, _, _ = _run_pomo_episode(
                        env=tenv, cfg=tcfg, actor_critic=actor_critic,
                        zone_plan=tzp, llm_confidence_scores=tall_conf,
                        max_scope=_MS, greedy=True,
                        llm_model=t_llm_model,
                        ep_accident_cache=t_ep_acc_cache,
                    )
                    unserved = tenv.N - tsc
                    row = (f"  {tname:<22}  {tsc:3d}/{tenv.N}  {unserved:8d}"
                           f"  {tenv.vehicles_used:3d}  {tenv.total_distance:8.1f}"
                           f"  {tlc:7d}  {tenv.total_late:8.1f}")
                    print(row)
                    eval_lines.append(row)
                except Exception as e:
                    msg = f"  {tname}: error {e}"
                    print(msg)
                    eval_lines.append(msg)
            if result_dir:
                fname = f"llm_rl_{algorithm}_eval_ep{ep}.txt"
                with open(os.path.join(result_dir, fname), "w", encoding="utf-8") as f:
                    f.write("\n".join(eval_lines) + "\n")
            save_checkpoint(
                os.path.join(ckpt_dir, f"proposed_multi_ep{ep}.pkl"),
                ep, ep, actor_critic, optimizer, None, -1, -1,
            )

    total_time_min = (time.time() - t0) / 60
    total_eps = sum(instance_counts.values())

    usage_lines = [f"INSTANCE USAGE  (total={total_eps} episodes)", "=" * 50]
    for inst_name in sorted(instance_counts):
        cnt = instance_counts[inst_name]
        usage_lines.append(f"  {inst_name:<30}  {cnt:4d}  ({cnt/total_eps*100:5.1f}%)")
    print(f"\n{'='*65}\n  {usage_lines[0]}\n{'='*65}")
    for line in usage_lines[2:]:
        print(line)

    # Save to result_dir
    if result_dir:
        with open(os.path.join(result_dir, f"llm_rl_{algorithm}_instance_usage.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(usage_lines) + "\n")
        print(f"\n[Result] saved to {result_dir}")

    print(f"\nTotal: {total_time_min:.1f}min")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL+LLM+Ontology (proposed model)")
    parser.add_argument("scenario", type=str, nargs="?", default=None,
                        help="Single instance (e.g. c101). Omit for --multi mode.")
    parser.add_argument("--multi",           action="store_true",
                        help="Multi-instance training (c102-c109 base+events → test c101)")
    parser.add_argument("--data-dir", default=os.path.join(_ROOT, "data", "Solomon"))
    parser.add_argument("--ckpt-dir", default=os.path.join(_ROOT, "checkpoints"))
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (single-instance mode only; "
                             "--multi uses a fixed LR per POMO convention)")
    parser.add_argument("--algorithm", default="pomo", choices=["ppo", "pomo", "kool"])
    parser.add_argument("--no-solution-zones", action="store_true")
    parser.add_argument("--no-ontology", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-model", default="qwen3:32b")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result-dir", default=None,
                        help="Directory to save eval results and usage stats")
    parser.add_argument("--pomo-start-ratio", type=float, default=1.0,
                        help="Fraction of zone stops used as POMO starts (1.0=all, 0.25=quarter)")
    parser.add_argument("--train-base", default=None,
                        help="Comma-separated training instances, e.g. c102 or c102,c103")
    parser.add_argument("--test-base",  default=None,
                        help="Comma-separated test base instance(s) (default: c101). E.g. r101 or c101,rc101,r101")
    parser.add_argument("--llm-lambda", type=float, default=1.0,
                        help="LLM confidence logit bias weight (1.0=strong, 0.1=weak)")
    parser.add_argument("--llm-topk-ratio", type=float, default=1.0,
                        help="Fraction of stops per cluster the LLM selects+scores "
                             "(boost-only). 0=none scored, 1=all stops scored (legacy).")
    parser.add_argument("--fixed-trust-gate", action="store_true",
                        help="Use a fixed LLM-confidence bias weight (=--llm-lambda) "
                             "instead of the learned trust gate.")
    args = parser.parse_args()

    if args.multi or args.scenario is None:
        train_multi_llm(
            data_dir=args.data_dir,
            ckpt_dir=args.ckpt_dir,
            num_episodes=args.episodes,
            algorithm=args.algorithm,
            no_llm=args.no_llm,
            no_ontology=args.no_ontology,
            llm_model=args.llm_model,
            seed=args.seed,
            resume=args.resume,
            result_dir=args.result_dir,
            pomo_start_ratio=args.pomo_start_ratio,
            train_base=[n.strip() for n in args.train_base.split(",")] if args.train_base else None,
            test_base=[n.strip() for n in args.test_base.split(",")] if args.test_base else None,
            llm_lambda=args.llm_lambda,
            llm_topk_ratio=args.llm_topk_ratio,
            learn_trust_gate=not args.fixed_trust_gate,
        )
    else:
        _dataset = args.scenario
        _use_sol = not args.no_solution_zones
        _base      = _dataset.split("_rain_")[0] if "_rain_" in _dataset else _dataset
        _data_path = rf"{args.data_dir}\{_dataset}.txt"
        _sol_path  = rf"{args.data_dir}\{_base}_sol.txt" if _use_sol else None
        _ckpt_path = rf"{args.ckpt_dir}\{_dataset}_ckpt.pkl"

        cfg = TrainConfig(
            data_path=_data_path,
            use_solution_zones=_use_sol,
            solution_path=_sol_path,
            use_fixed_solution_policy=False,
            fixed_policy_print_trace=False,
            num_episodes=args.episodes,
            seed=args.seed,
            lr=args.lr,
            estimated_steps_per_episode=80,
            prefer_on_time_candidates=False,
            use_ontology_pruning=not args.no_ontology,
            ontology_filter_on_return_feasible=True,
            ontology_filter_late_actions=False,
            ontology_keep_depot_action=True,
            use_ontology_reasoning=not args.no_ontology,
            ontology_unserved_window_ratio=0.1,
            ontology_min_warmup_ratio=0.5,
            use_llm_zone_confidence=not args.no_llm,
            use_llm_event_response=not args.no_llm,
            llm_recall_on_chunk=False,
            llm_confidence_lambda=args.llm_lambda,
            use_learned_trust_gate=not args.fixed_trust_gate,
            llm_confidence_topk_ratio=args.llm_topk_ratio,
            llm_model=args.llm_model,
            checkpoint_path=_ckpt_path,
            checkpoint_interval=1000,
            resume_from=args.resume,
            **SHARED_HYPERPARAMS,
        )
        train(cfg, algorithm=args.algorithm)
