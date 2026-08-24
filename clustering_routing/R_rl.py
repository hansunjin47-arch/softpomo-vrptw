"""
Pure RL baseline for rain scenario comparison.

Rain forecast info (강수량, 시간대, 강수확률, 강수지역) is fully embedded
in the state vector as numerical features from t=0, so the agent has the
same information as the LLM-augmented approach — but must learn to use it
purely through reward signal rather than semantic reasoning.

No LLM, no ontology. Plain PPO.
"""
from __future__ import annotations

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import random
from copy import deepcopy
from typing import List, Optional, Tuple

import time


import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

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
    get_feasible_actions_with_zone_control,
)
from R_rl_module import (
    RolloutBuffer,
    AttentionActorCritic,
    ppo_update,
    kool_update,
    pomo_update,
    pomo_ppo_update,
    save_checkpoint,
    load_checkpoint,
    _plot_routing,
    _plot_reward_curve,
    BaselineEnv,
)
from R_env import (
    build_travel_time_matrix,
    load_waste_benchmark_txt,
)


# ============================================================
# 학습 실행 설정 — 여기서만 수정
# ============================================================
RUN_CONFIG = dict(
    data_dir      = os.path.join(_ROOT, "data", "Solomon"),
    ckpt_dir      = os.path.join(_ROOT, "checkpoints"),
    num_episodes  = 5000,
    lr            = 1e-4,
    algorithm     = "pomo",  # "ppo", "kool", "pomo", or "grpo"
    seed          = 42,
    resume        = None,
    use_solution_zones = False,
)


def _collect_episode(
    env: BaselineEnv,
    zone_plan,
    cfg,
    actor_critic: AttentionActorCritic,
    device: str,
    max_scope: int,
    max_steps_per_episode: int,
    forced_first_per_zone: Optional[dict] = None,
    greedy: bool = False,
) -> Tuple[float, int, int, str, RolloutBuffer, List[np.ndarray]]:
    """Run one episode and return (ep_reward, ep_late_cnt, served_count, end_reason, rollout, masks).

    forced_first_per_zone: {zone_idx: customer_abs} — each zone's first customer is forced,
    while still computing log_prob from the policy so gradients flow correctly.
    greedy: if True, select actions via argmax (no sampling, no buf.push) — used for Kool baseline rollout.
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
    zone_first_done: set = set()  # zones whose first customer has already been forced

    while not (terminated or truncated):
        step_in_ep += 1

        feasible, _, _ = get_feasible_actions_with_zone_control(env=env, zone_plan=zone_plan, cfg=cfg)
        if not feasible:
            truncated = True
            break

        scope_custs = env._get_scope_customers()
        abs_to_rel: dict = {0: 0}
        rel_to_abs: dict = {0: 0}
        for _i, _c in enumerate(scope_custs):
            abs_to_rel[_c] = _i + 1
            rel_to_abs[_i + 1] = _c
        feasible_rel = [abs_to_rel[a] for a in feasible if a in abs_to_rel]
        if not feasible_rel:
            truncated = True
            break

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if greedy:
            with torch.no_grad():
                logits, _ = actor_critic(obs_t)
            mask_t = torch.full_like(logits[0], float("-inf"))
            for a_rel in feasible_rel:
                mask_t[a_rel] = 0.0
            action_rel = int((logits[0] + mask_t).argmax().item())
            action_abs = rel_to_abs[action_rel]
        else:
            # Per-zone first customer forcing: check if current zone needs forcing
            active_zone = int(env.active_zone_idx)
            forced_cust = (
                forced_first_per_zone.get(active_zone)
                if forced_first_per_zone is not None and active_zone not in zone_first_done
                else None
            )

            if forced_cust is not None and forced_cust in abs_to_rel:
                action_rel = abs_to_rel[forced_cust]
                with torch.no_grad():
                    logits, value = actor_critic(obs_t)
                mask_t = torch.full_like(logits[0], float("-inf"))
                for a_rel in feasible_rel:
                    mask_t[a_rel] = 0.0
                dist = Categorical(logits=logits[0] + mask_t)
                log_prob = float(dist.log_prob(torch.tensor(action_rel, device=device)).item())
                val = float(value[0].item())
                action_abs = forced_cust
                zone_first_done.add(active_zone)
            else:
                if forced_cust is not None:
                    zone_first_done.add(active_zone)  # forced customer not feasible, skip forcing
                with torch.no_grad():
                    action_rel, log_prob, val = actor_critic.get_action_and_value(obs_t, feasible_rel)
                action_abs = rel_to_abs[action_rel]

            mask_np = np.full(max_scope, float("-inf"), dtype=np.float32)
            for a_rel in feasible_rel:
                mask_np[a_rel] = 0.0

        next_obs, reward, terminated, truncated, info = env.step(action_abs, external_penalty=0.0)
        ep_reward += reward
        if info.get("late", 0.0) > 0:
            ep_late_cnt += 1

        if not greedy:
            buf.push(obs, action_rel, float(reward), float(terminated or truncated), val, log_prob,
                     np.zeros(max_scope, dtype=np.float32))
            masks.append(mask_np)
        obs = next_obs

        if step_in_ep >= max_steps_per_episode:
            truncated = True
            end_reason = "max_steps"
        if info.get("reason") == "vehicle_limit_reached":
            truncated = True
            end_reason = "vehicle_limit_reached"
        if info.get("reason") in ("all_served", "all_vehicles_retired"):
            end_reason = info["reason"]

    served_count = int(np.sum(env.served[1:]))
    return ep_reward, ep_late_cnt, served_count, end_reason, buf, masks


def train_baseline(
    data_path: str,
    num_episodes: int = RUN_CONFIG["num_episodes"],
    lr: float = RUN_CONFIG["lr"],
    algorithm: str = RUN_CONFIG["algorithm"],
    use_solution_zones: bool = True,
    solution_path: Optional[str] = None,
    max_steps_per_episode: int = 10_000,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 1000,
    resume_from: Optional[str] = None,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **hyperparams,
):
    assert algorithm in ("ppo", "kool", "pomo", "grpo", "pomo_ppo"), f"algorithm must be one of ppo/kool/pomo/grpo/pomo_ppo, got {algorithm!r}"
    set_seed(seed)
    print(f"[Baseline] data={data_path}  algorithm={algorithm.upper()}")
    print(f"[Baseline] rain features embedded in state  LLM=OFF  Ontology=OFF")

    _mlflow_run = None
    if MLFLOW_AVAILABLE:
        import os
        mlflow.set_experiment("rl_baseline")
        _mlflow_run = mlflow.start_run(run_name=f"{algorithm}_{os.path.basename(data_path).replace('.txt','')}")
        mlflow.log_params({
            "algorithm":        algorithm,
            "num_episodes":     num_episodes,
            "lr":               lr,
            "seed":             seed,
            "data":             os.path.basename(data_path),
            **{k: v for k, v in {**SHARED_HYPERPARAMS, **hyperparams}.items()
               if isinstance(v, (int, float, str, bool))},
        })

    inst = load_waste_benchmark_txt(data_path)
    tt = build_travel_time_matrix(inst)

    if inst.preset_events:
        print(f"[Events] {len(inst.preset_events)} preset event(s):")
        for e in inst.preset_events:
            print(f"  {e.event_type.value.upper()}  t=[{e.trigger_time:.0f}, {e.end_time:.0f}]  "
                  f"multiplier={e.multiplier:.2f}  "
                  f"rainfall={e.rainfall_mm:.0f}mm/h  prob={e.probability:.0f}%  "
                  f"nodes={e.affected_nodes}")

    hp = {**SHARED_HYPERPARAMS, **hyperparams}
    embed_dim = hp["embed_dim"]
    n_heads = hp["n_heads"]
    n_encoder_layers = hp["n_encoder_layers"]
    lr_min = hp["lr_min"]
    entropy_coef = hp["entropy_coef"]
    entropy_coef_min = hp["entropy_coef_min"]
    pomo_k = hp.get("pomo_k", 4)

    _cfg = TrainConfig(
        data_path=data_path,
        use_solution_zones=use_solution_zones,
        solution_path=solution_path,
        prefer_on_time_candidates=False,
        use_ontology_pruning=False,
        use_ontology_reasoning=False,
        use_llm_zone_confidence=False,
        use_llm_event_response=False,
        num_episodes=num_episodes,
        lr=lr,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=checkpoint_interval,
        device=device,
        **hp,
    )

    # Zone plan
    zone_plan: Optional[ZonePlan] = None
    if use_solution_zones:
        sol_path = infer_solution_path(data_path, solution_path)
        if sol_path is not None:
            routes = parse_solution_routes(sol_path)
            validate_solution_routes(inst, routes)
            zone_plan = build_zone_plan_from_routes(inst, routes, _cfg)
            print(f"[Zone] solution-based: {zone_plan.n_zones} zones")
            for z_i, z_custs in enumerate(zone_plan.zones):
                print(f"  zone {z_i}: {len(z_custs)} customers  {sorted(z_custs)}")
        else:
            print("[Zone] solution file not found — falling back to clustering")
            zone_plan = build_initial_zones(inst, _cfg, tt)
            print(f"[Zone] clustered: {zone_plan.n_zones} zones")
            for z_i, z_custs in enumerate(zone_plan.zones):
                print(f"  zone {z_i}: {len(z_custs)} customers  {sorted(z_custs)}")
    else:
        zone_plan = build_initial_zones(inst, _cfg, tt)
        print(f"[Zone] clustered: {zone_plan.n_zones} zones")
        for z_i, z_custs in enumerate(zone_plan.zones):
            print(f"  zone {z_i}: {len(z_custs)} customers  {sorted(z_custs)}")

        sol_path = infer_solution_path(data_path, solution_path)
        if sol_path is not None:
            try:
                sol_routes = parse_solution_routes(sol_path)
                validate_solution_routes(inst, sol_routes)
                sol_zone_plan = build_zone_plan_from_routes(inst, sol_routes, _cfg)
                print(f"[Zone] solution reference: {sol_zone_plan.n_zones} zones")
                for z_i, z_custs in enumerate(sol_zone_plan.zones):
                    print(f"  zone {z_i}: {len(z_custs)} customers  {sorted(z_custs)}")
                print("[Zone] diff (clustered vs solution):")
                for z_i, z_custs in enumerate(zone_plan.zones):
                    cl_set = set(z_custs)
                    best_match_idx, _ = max(
                        ((s_i, len(cl_set & set(s_custs))) for s_i, s_custs in enumerate(sol_zone_plan.zones)),
                        key=lambda x: x[1],
                    )
                    sol_set = set(sol_zone_plan.zones[best_match_idx])
                    only_cl = sorted(cl_set - sol_set)
                    only_sol = sorted(sol_set - cl_set)
                    print(f"  cluster zone {z_i} ↔ sol zone {best_match_idx}:"
                          f"  overlap={len(cl_set & sol_set)}/{len(cl_set)}"
                          f"  only_in_cluster={only_cl}"
                          f"  only_in_sol={only_sol}")
            except Exception as e:
                print(f"[Zone] could not load solution for comparison: {e}")
        else:
            print("[Zone] no solution file found — skipping comparison")

    max_vehicles = zone_plan.n_zones if zone_plan is not None else inst.vehicle_limit
    max_scope = max(len(z) for z in zone_plan.zones) + 1 if zone_plan is not None else inst.n_customers

    env = BaselineEnv(
        inst=inst,
        tt=tt,
        max_vehicles=max_vehicles,
        late_count_penalty=_cfg.late_count_penalty,
        late_penalty=_cfg.late_penalty,
        unserved_penalty=_cfg.unserved_penalty,
        max_scope_size=max_scope,
    )
    print(f"[Env] obs_dim={env.observation_space.shape[0]}  event_feat_dim={env.event_feat_dim}")

    if zone_plan is not None:
        env.set_zone_assignment(
            zone_plan.customer_to_zone,
            zone_plan.n_zones,
            adjacent_zones=zone_plan.adjacent_zones,
        )
        if "pomo_k" not in hyperparams:
            per_zone_k = [max(1, round(len(z) ** 0.5)) if z else 1 for z in zone_plan.zones]
            pomo_k = max(per_zone_k)
            zone_sizes = [len(z) for z in zone_plan.zones]
            print(f"[Zone] max_scope={max_scope}  zone_sizes={zone_sizes}  per_zone_k={per_zone_k}  pomo_k={pomo_k} (auto: max(round(sqrt(N))))")
        else:
            per_zone_k = [min(pomo_k, max(1, len(z))) if z else 1 for z in zone_plan.zones]
            print(f"[Zone] max_scope={max_scope}  pomo_k={pomo_k} (manual)")

    actor_critic = AttentionActorCritic(
        stop_feat_dim=env.stop_feat_dim,
        max_scope_size=max_scope,
        vehicle_feat_dim=env.vehicle_feat_dim,
        global_feat_dim=env.global_feat_dim,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        event_feat_dim=env.event_feat_dim,
    ).to(device)

    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    global_step = 0
    start_ep = 1
    ep_reward_history: List[float] = []
    best_ep_log: Optional[str] = None
    best_served: int = -1
    best_ep_reward: float = float("-inf")
    best_ep_routes: Optional[List[List[int]]] = None
    train_start_time = time.time()
    ep_wall_times: List[float] = []

    if resume_from is not None:
        start_ep, global_step, best_served, best_ep_reward, _ = load_checkpoint(
            path=resume_from,
            actor_critic=actor_critic,
            optimizer=optimizer,
            ontology_engine=None,
            device=device,
        )

    # Kool frozen baseline — init after resume so it matches loaded weights
    actor_critic_bl = None
    BL_UPDATE_INTERVAL = 100
    if algorithm == "kool":
        actor_critic_bl = deepcopy(actor_critic)
        actor_critic_bl.eval()

    for ep in range(start_ep, num_episodes + 1):
        progress = (ep - 1) / max(num_episodes - 1, 1)
        current_lr = lr + (lr_min - lr) * progress
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        current_entropy_coef = entropy_coef + (entropy_coef_min - entropy_coef) * progress

        ep_start_time = time.time()
        ep_reward = 0.0
        ep_loss = 0.0
        ep_train_steps = 0
        ep_late_cnt = 0
        end_reason = "unknown"

        if algorithm in ("pomo", "grpo", "pomo_ppo"):
            # POMO: K rollouts with unique first customer per zone (shuffle+cycle, no duplicates)
            zone_cust_pools = [
                [c for c in z_custs if c > 0]
                for z_custs in zone_plan.zones
            ] if zone_plan is not None else [[]]
            # Shuffle each zone's pool once, then assign rollout k → pool[k % len(pool)]
            zone_shuffled = [
                random.sample(pool, len(pool)) if pool else []
                for pool in zone_cust_pools
            ]
            bufs_k: List[RolloutBuffer] = []
            masks_k: List[List[np.ndarray]] = []
            rewards_k: List[float] = []
            late_k: List[int] = []
            served_k: List[int] = []
            routes_k: List[List[List[int]]] = []
            served_np_k: List[np.ndarray] = []
            dist_k: List[float] = []
            total_late_k: List[float] = []
            for k in range(pomo_k):
                forced_first_per_zone = {
                    z_idx: zone_shuffled[z_idx][k % per_zone_k[z_idx]]
                    for z_idx in range(len(zone_shuffled))
                    if zone_shuffled[z_idx]
                }
                r, lc, sc, _, buf, msk = _collect_episode(
                    env, zone_plan, _cfg, actor_critic, device,
                    max_scope, max_steps_per_episode,
                    forced_first_per_zone=forced_first_per_zone,
                )
                bufs_k.append(buf)
                masks_k.append(msk)
                rewards_k.append(r)
                late_k.append(lc)
                served_k.append(sc)
                routes_k.append([v.route_nodes[:] for v in env.vehicles])
                served_np_k.append(env.served.copy())
                dist_k.append(env.total_distance)
                total_late_k.append(env.total_late)
                global_step += len(buf)
            if algorithm == "pomo_ppo":
                ep_loss = pomo_ppo_update(
                    actor_critic, optimizer, bufs_k, masks_k, _cfg,
                    entropy_coef=current_entropy_coef,
                )
            else:
                ep_loss = pomo_update(
                    actor_critic, optimizer, bufs_k, masks_k, rewards_k, _cfg,
                    entropy_coef=current_entropy_coef,
                    normalize_std=(algorithm == "grpo"),
                )
            ep_train_steps = 1
            on_time_k = [served_k[i] - late_k[i] for i in range(len(served_k))]
            best_k = max(range(len(rewards_k)), key=lambda i: (on_time_k[i], rewards_k[i]))
            ep_reward = rewards_k[best_k]
            ep_late_cnt = late_k[best_k]
            served_count = served_k[best_k]
            end_reason = f"{algorithm}_k={pomo_k}"
            # Restore best rollout's env state for accurate logging
            env.served = served_np_k[best_k]
            env.total_distance = dist_k[best_k]
            env.total_late = total_late_k[best_k]
            for v, route in zip(env.vehicles, routes_k[best_k]):
                v.route_nodes = route
        elif algorithm == "kool":
            # Stochastic rollout — collect trajectory for gradient
            r_s, lc, sc, end_reason, buf_s, masks_s = _collect_episode(
                env, zone_plan, _cfg, actor_critic, device,
                max_scope, max_steps_per_episode,
                greedy=False,
            )
            # Save stochastic rollout's env state before greedy rollout overwrites it
            stoch_routes = [v.route_nodes[:] for v in env.vehicles]
            stoch_served = env.served.copy()
            stoch_dist = env.total_distance
            stoch_late_total = env.total_late

            # Greedy rollout — frozen baseline model, no gradient
            r_g, _, _, _, _, _ = _collect_episode(
                env, zone_plan, _cfg, actor_critic_bl, device,
                max_scope, max_steps_per_episode,
                greedy=True,
            )

            # Restore stochastic rollout's env state for logging
            env.served = stoch_served
            env.total_distance = stoch_dist
            env.total_late = stoch_late_total
            for v, route in zip(env.vehicles, stoch_routes):
                v.route_nodes = route

            ep_loss = kool_update(
                actor_critic, optimizer, buf_s, masks_s, r_s, r_g, _cfg,
            )
            ep_reward = r_s
            ep_late_cnt = lc
            served_count = sc
            global_step += len(buf_s)
            ep_train_steps = 1

            # Periodic baseline update — single instance: direct greedy comparison
            if ep % BL_UPDATE_INTERVAL == 0:
                r_curr_g, _, _, _, _, _ = _collect_episode(
                    env, zone_plan, _cfg, actor_critic, device,
                    max_scope, max_steps_per_episode,
                    greedy=True,
                )
                r_base_g, _, _, _, _, _ = _collect_episode(
                    env, zone_plan, _cfg, actor_critic_bl, device,
                    max_scope, max_steps_per_episode,
                    greedy=True,
                )
                if r_curr_g > r_base_g:
                    actor_critic_bl.load_state_dict(actor_critic.state_dict())
                    print(f"  [Kool] baseline updated ep={ep}: "
                          f"curr_g={r_curr_g:.2f} > base_g={r_base_g:.2f}")
                else:
                    print(f"  [Kool] baseline held    ep={ep}: "
                          f"curr_g={r_curr_g:.2f} <= base_g={r_base_g:.2f}")
        else:
            ep_reward, ep_late_cnt, served_count, end_reason, buf_ppo, masks_ppo = _collect_episode(
                env, zone_plan, _cfg, actor_critic, device,
                max_scope, max_steps_per_episode,
            )
            global_step += len(buf_ppo)

            if len(buf_ppo) > 0:
                obs_last = np.array(buf_ppo.obs[-1])
                obs_t = torch.tensor(obs_last, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, last_val = actor_critic(obs_t)
                    last_val = float(last_val[0].item())
                ep_loss = ppo_update(actor_critic, optimizer, buf_ppo, masks_ppo,
                                     last_val, _cfg, entropy_coef=current_entropy_coef)
                ep_train_steps = 1

        on_time = served_count - ep_late_cnt
        unserved = env.N - served_count
        avg_loss = ep_loss / max(ep_train_steps, 1)
        ep_reward_history.append(ep_reward)
        avg_late = env.total_late / max(served_count, 1)
        ep_wall = time.time() - ep_start_time
        ep_wall_times.append(ep_wall)

        zone_served_str = ""
        if zone_plan is not None:
            parts = []
            for z_custs in zone_plan.zones:
                z_served = sum(1 for c in z_custs if env.served[c] == 1)
                parts.append(f"{z_served}/{len(z_custs)}")
            zone_served_str = "  zones=[" + " ".join(parts) + "]"

        ep_log = (
            f"[BASELINE EP {ep:4d}] "
            f"reward={ep_reward:10.2f}  "
            f"served={on_time:3d}  tw_late={ep_late_cnt:3d}  unserved={unserved:3d}/{env.N}"
            f"{zone_served_str}  "
            f"veh={env.vehicles_used:2d}/{max_vehicles:2d}  "
            f"dist={env.total_distance:10.2f}  "
            f"late={env.total_late:9.2f}  "
            f"late_avg={avg_late:8.2f}  "
            f"loss={avg_loss:8.4f}  "
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
            }, step=ep)

        if on_time > best_served or (on_time == best_served and ep_reward > best_ep_reward):
            best_served = on_time
            best_ep_reward = ep_reward
            best_ep_log = ep_log
            best_ep_routes = [v.route_nodes[:] for v in env.vehicles]
            print(f"[Best EP {ep:4d}] " + ep_log)

        if checkpoint_path is not None and ep % checkpoint_interval == 0:
            save_checkpoint(
                path=checkpoint_path,
                ep=ep,
                global_step=global_step,
                actor_critic=actor_critic,
                optimizer=optimizer,
                ontology_engine=None,
                best_served=best_served,
                best_ep_reward=best_ep_reward,
            )

    total_train_time = time.time() - train_start_time
    avg_ep_time = sum(ep_wall_times) / max(len(ep_wall_times), 1)
    print(f"\n[Time] total={total_train_time:.1f}s  "
          f"avg_per_ep={avg_ep_time:.2f}s  "
          f"episodes={len(ep_wall_times)}")

    if best_ep_log is not None:
        print("\n[Best Episode]")
        print(best_ep_log)

    if best_ep_routes is not None:
        _plot_routing(best_ep_routes, env, _cfg)

    _plot_reward_curve(ep_reward_history, _cfg)

    if MLFLOW_AVAILABLE and _mlflow_run is not None:
        import os
        mlflow.log_metric("best_served", best_served)
        mlflow.log_metric("best_ep_reward", best_ep_reward)
        mlflow.log_metric("total_train_time_s", total_train_time)
        _base = _cfg.checkpoint_path.replace("_ckpt.pkl", "") if _cfg.checkpoint_path else "."
        routing_path = f"{_base}_routing_best.png"
        reward_path  = f"{_base}_reward_curve.png"
        for p in (routing_path, reward_path):
            if os.path.exists(p):
                mlflow.log_artifact(p)
        mlflow.end_run()

    return actor_critic, env, zone_plan


MAX_SCOPE = 16   # fixed zone size for consistent model input across all C-type instances

_TRAIN_BASE = ["c102","c103","c104","c105","c106","c107","c108","c109"]
_TRAIN_EVENTS = (
    [f"{n}_rain_A" for n in _TRAIN_BASE] +
    [f"{n}_rain_B" for n in _TRAIN_BASE] +
    [f"{n}_acc_A"  for n in _TRAIN_BASE] +
    [f"{n}_acc_B"  for n in _TRAIN_BASE]
)
_TEST_INSTANCES = ["c101", "c101_rain_A", "c101_rain_B", "c101_acc_A", "c101_acc_B"]


def train_multi(
    data_dir: str = RUN_CONFIG["data_dir"],
    ckpt_dir: str = RUN_CONFIG["ckpt_dir"],
    num_episodes: int = 10_000,
    lr: float = RUN_CONFIG["lr"],
    algorithm: str = "pomo",
    seed: int = 42,
    resume_from: Optional[str] = None,
    result_dir: Optional[str] = None,
    pomo_start_ratio: float = 1.0,
    train_base: Optional[List[str]] = None,
    test_base: Optional[List[str]] = None,
    **hyperparams,
):
    """Multi-instance C→R training.

    Training pool: base (c102-c109) + rain/acc event variants, sampled uniformly i.i.d.
    Test: c101 base + rain_A/B + acc_A/B
    """
    set_seed(seed)
    os.makedirs(ckpt_dir, exist_ok=True)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hp = {**SHARED_HYPERPARAMS, **hyperparams}

    # Allow overriding training/test instances
    base_list = train_base if train_base else _TRAIN_BASE
    event_list = (
        [f"{n}_rain_A" for n in base_list] +
        [f"{n}_rain_B" for n in base_list] +
        [f"{n}_acc_A"  for n in base_list] +
        [f"{n}_acc_B"  for n in base_list]
    )
    test_base_list = test_base if test_base else ["c101"]
    test_list = []
    for _tb in test_base_list:
        test_list += [_tb, f"{_tb}_rain_A", f"{_tb}_rain_B", f"{_tb}_acc_A", f"{_tb}_acc_B"]

    print(f"[MultiTrain] algorithm={algorithm}  episodes={num_episodes}  device={device}")
    print(f"  Pool: {base_list} + {len(event_list)} event instances (i.i.d. sampling)")

    def _make_cfg(path):
        return TrainConfig(
            data_path=path, use_solution_zones=False,
            prefer_on_time_candidates=False,
            use_ontology_pruning=False, use_ontology_reasoning=False,
            late_count_penalty=hp.get("late_count_penalty", 20.0),
            late_penalty=hp.get("late_penalty", 150.0),
            unserved_penalty=hp.get("unserved_penalty", 500.0),
        )

    def _load_zone_plan(name):
        """Phase 1: load instance + build zone plan (no env yet).

        Event instances (rain_A/B, acc_A/B) derive zones from the BASE instance's
        data (coords/demands/tw/base tt are unchanged by events), so all variants
        share the same clustering — only routing adapts to events.
        """
        path = os.path.join(data_dir, name + ".txt")
        inst = load_waste_benchmark_txt(path)
        tt   = build_travel_time_matrix(inst)
        cfg  = _make_cfg(path)
        # Derive base name: "c102_rain_A" → "c102", "c103_acc_B" → "c103"
        base_name = name.split("_rain_")[0].split("_acc_")[0]
        base_path = os.path.join(data_dir, base_name + ".txt")
        sol_path  = infer_solution_path(base_path, None)
        if cfg.use_solution_zones and sol_path:
            zone_plan = build_zone_plan_from_routes(inst, parse_solution_routes(sol_path), cfg)
        else:
            zone_plan = build_initial_zones(inst, cfg, tt)
        actual_scope = max(len(z) for z in zone_plan.zones) + 1 if zone_plan else inst.n_customers
        return inst, tt, zone_plan, cfg, actual_scope

    def _build_env(inst, tt, zone_plan, cfg, scope):
        """Phase 2: build env with fixed scope."""
        max_vehicles = zone_plan.n_zones if zone_plan else inst.vehicle_limit
        env = BaselineEnv(
            inst=inst, tt=tt, max_vehicles=max_vehicles,
            late_count_penalty=cfg.late_count_penalty,
            late_penalty=cfg.late_penalty,
            unserved_penalty=cfg.unserved_penalty,
            max_scope_size=scope,
        )
        if zone_plan:
            env.set_zone_assignment(zone_plan.customer_to_zone, zone_plan.n_zones,
                                    adjacent_zones=zone_plan.adjacent_zones)
        return env

    # Pass 1: load zone plans, compute max scope
    print("[MultiTrain] Pre-computing zone plans...", flush=True)
    _zone_data = {}
    for _n in (base_list + event_list + test_list):
        try:
            _zone_data[_n] = _load_zone_plan(_n)
        except Exception as _e:
            print(f"  [SKIP] {_n}: {_e}")

    # Compute max scope from actual zone sizes across training instances
    _MS = max(scope for _, _, _, _, scope in _zone_data.values())
    print(f"[MultiTrain] max_scope={_MS} (from {len(_zone_data)} instances)", flush=True)

    # Pass 2: build envs with unified _MS
    _cache = {}
    for _n, (inst, tt, zp, cfg, _) in _zone_data.items():
        env = _build_env(inst, tt, zp, cfg, _MS)
        _cache[_n] = (inst, tt, zp, cfg, env, _MS)

    def _get(name):
        return _cache[name]

    def _load(name):
        """Fresh env for POMO rollouts (reset state)."""
        inst, tt, zp, cfg, _ = _zone_data[name]
        env = _build_env(inst, tt, zp, cfg, _MS)
        return inst, tt, zp, cfg, env, _MS

    # Init model from a base instance
    _, _, _, cfg0, env0, _ = _get(base_list[0])
    actor_critic = AttentionActorCritic(
        stop_feat_dim=env0.stop_feat_dim,
        max_scope_size=_MS,  # computed from actual zone sizes
        vehicle_feat_dim=env0.vehicle_feat_dim,
        global_feat_dim=env0.global_feat_dim,
        embed_dim=hp.get("embed_dim", 128),
        n_heads=hp.get("n_heads", 8),
        n_encoder_layers=hp.get("n_encoder_layers", 3),
    ).to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
    start_ep = 1
    if resume_from:
        start_ep, _, _, _, _ = load_checkpoint(resume_from, actor_critic, optimizer, None, device)

    reward_history, t0 = [], time.time()
    instance_counts: dict = {}   # track usage per instance

    pool = base_list + event_list
    for ep in range(start_ep, num_episodes + 1):
        name = random.choice(pool)
        instance_counts[name] = instance_counts.get(name, 0) + 1
        inst, tt, zone_plan, cfg, env, _ = _get(name)
        # POMO starts: pomo_start_ratio controls fraction of zone stops used (1.0 = all)
        max_zone = max(len(z) for z in zone_plan.zones) if zone_plan else _MS - 1
        pomo_k = max(1, round(max_zone * pomo_start_ratio))

        if algorithm == "pomo":
            rewards_k, bufs_k, masks_k = [], [], []
            for _ in range(pomo_k):
                r, lc, sc, reason, buf, masks = _collect_episode(
                    env, zone_plan, cfg, actor_critic, device, _MS, 10_000)
                rewards_k.append(r); bufs_k.append(buf); masks_k.append(masks)
                _, _, _, _, env, _ = _load(name)   # fresh env for next rollout (re-load to reset state)
            pomo_update(actor_critic, optimizer, bufs_k, masks_k, rewards_k, cfg)
            ep_reward = max(rewards_k)
        else:
            ep_reward, _, _, _, buf, masks = _collect_episode(
                env, zone_plan, cfg, actor_critic, device, MAX_SCOPE, 10_000)
            if algorithm == "ppo":
                obs_t = torch.tensor(np.array(buf.obs[-1]) if buf.obs else
                                     np.zeros(env0.observation_space.shape[0]),
                                     dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, lv = actor_critic(obs_t)
                ppo_update(actor_critic, optimizer, buf, masks, float(lv[0].item()), cfg)

        reward_history.append(ep_reward)

        if ep % 50 == 0:
            avg_r = sum(reward_history[-50:]) / min(50, len(reward_history))
            print(f"ep={ep:4d}/{num_episodes}"
                  f"  avg_r={avg_r:.2f}  elapsed={time.time()-t0:.0f}s", flush=True)

        if ep % max(200, num_episodes // 5) == 0 or ep == num_episodes:
            # Quick eval on all test instances (base + rain/acc event variants)
            print(f"  [Eval] ep={ep}", flush=True)
            for tname in test_list:
                try:
                    _, _, ztp, cfgt, evt, _ = _get(tname)
                    _, lct, sct, _, _, _ = _collect_episode(
                        evt, ztp, cfgt, actor_critic, device, MAX_SCOPE, 10_000, greedy=True)
                    print(f"    [Eval {tname:<14}] served={sct:3d}/{evt.N}  late_cnt={lct:3d}  "
                          f"late_min={evt.total_late:8.1f}  dist={evt.total_distance:8.1f}", flush=True)
                except Exception as e:
                    print(f"    [Eval {tname}] error: {e}")
            save_checkpoint(
                os.path.join(ckpt_dir, f"multi_{algorithm}_ep{ep}.pkl"),
                ep, ep, actor_critic, optimizer, None, -1, ep_reward)

    total_time_min = (time.time() - t0) / 60

    # Instance usage summary
    total_eps = sum(instance_counts.values())
    usage_lines = [
        f"INSTANCE USAGE  (total={total_eps} episodes)",
        "=" * 50,
    ]
    for inst_name in sorted(instance_counts):
        cnt = instance_counts[inst_name]
        usage_lines.append(f"  {inst_name:<30}  {cnt:4d}  ({cnt/total_eps*100:5.1f}%)")
    usage_text = "\n".join(usage_lines)
    print(f"\n{'='*65}\n  {usage_lines[0]}\n{'='*65}")
    for line in usage_lines[2:]:
        print(line)

    # Final evaluation
    eval_lines = [
        f"FINAL EVALUATION  [RL  ep={num_episodes}  alg={algorithm}  seed={seed}]",
        "=" * 75,
        f"  {'Scenario':<22}  {'Served':>6}  {'Unserved':>8}  {'Veh':>3}"
        f"  {'Dist':>8}  {'LateCnt':>7}  {'LateMin':>8}",
        "-" * 75,
    ]
    print(f"\n{'='*65}\n  FINAL EVALUATION\n{'='*65}")
    for name in test_list:
        try:
            _, _, zp, cfg_t, ev, _ = _get(name)
            rg, lc, sc, _, _, _ = _collect_episode(
                ev, zp, cfg_t, actor_critic, device, MAX_SCOPE, 10_000, greedy=True)
            unserved = ev.N - sc
            row = (f"  {name:<22}  {sc:3d}/{ev.N}  {unserved:8d}  {ev.vehicles_used:3d}"
                   f"  {ev.total_distance:8.1f}  {lc:7d}  {ev.total_late:8.1f}")
            print(row)
            eval_lines.append(row)
        except Exception as e:
            msg = f"  {name:<22}  SKIP: {e}"
            print(msg)
            eval_lines.append(msg)
    eval_lines.append(f"\nTotal training time: {total_time_min:.1f} min")
    eval_text = "\n".join(eval_lines)

    # Save to result_dir
    if result_dir:
        with open(os.path.join(result_dir, f"rl_{algorithm}_final_eval.txt"), "w", encoding="utf-8") as f:
            f.write(eval_text + "\n")
        with open(os.path.join(result_dir, f"rl_{algorithm}_instance_usage.txt"), "w", encoding="utf-8") as f:
            f.write(usage_text + "\n")
        print(f"\n[Result] saved to {result_dir}")

    save_checkpoint(os.path.join(ckpt_dir, f"multi_{algorithm}_last.pkl"),
              num_episodes, num_episodes, actor_critic, optimizer, None, -1, -1)
    print(f"\nTotal: {total_time_min:.1f}min")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Pure RL baseline (no LLM, no Ontology)")
    parser.add_argument("scenario",            type=str, nargs="?", default=None,
                        help="Single instance (e.g. c101). Omit for multi-instance training.")
    parser.add_argument("--multi",             action="store_true",
                        help="Multi-instance training (c102-c109 base+events → test c101)")
    parser.add_argument("--data-dir",          default=RUN_CONFIG["data_dir"])
    parser.add_argument("--ckpt-dir",          default=RUN_CONFIG["ckpt_dir"])
    parser.add_argument("--episodes",          type=int,  default=RUN_CONFIG["num_episodes"])
    parser.add_argument("--lr",                type=float, default=RUN_CONFIG["lr"])
    parser.add_argument("--algorithm",         default=RUN_CONFIG["algorithm"],
                        choices=["ppo", "kool", "pomo", "grpo", "pomo_ppo"])
    parser.add_argument("--seed",              type=int,  default=RUN_CONFIG["seed"])
    parser.add_argument("--resume",            default=RUN_CONFIG["resume"])
    parser.add_argument("--no-solution-zones", action="store_true",
                        default=not RUN_CONFIG["use_solution_zones"])
    parser.add_argument("--result-dir",        default=None,
                        help="Directory to save final eval and usage stats")
    parser.add_argument("--pomo-start-ratio",  type=float, default=1.0,
                        help="Fraction of zone stops used as POMO starts (1.0=all, 0.25=quarter)")
    parser.add_argument("--train-base",        default=None,
                        help="Comma-separated training instances, e.g. c102 or c102,c103")
    parser.add_argument("--test-base",         default=None,
                        help="Comma-separated test base instance(s) (default: c101). E.g. r101 or c101,rc101,r101")
    args = parser.parse_args()

    train_base_override = [n.strip() for n in args.train_base.split(",")] if args.train_base else None
    test_base_override = [n.strip() for n in args.test_base.split(",")] if args.test_base else None

    if args.multi or args.scenario is None:
        train_multi(
            data_dir=args.data_dir,
            ckpt_dir=args.ckpt_dir,
            num_episodes=args.episodes,
            lr=args.lr,
            algorithm=args.algorithm,
            seed=args.seed,
            resume_from=args.resume,
            result_dir=args.result_dir,
            pomo_start_ratio=args.pomo_start_ratio,
            train_base=train_base_override,
            test_base=test_base_override,
        )
    else:
        train_baseline(
            data_path=rf"{args.data_dir}\{args.scenario}.txt",
            use_solution_zones=not args.no_solution_zones,
            num_episodes=args.episodes,
            lr=args.lr,
            algorithm=args.algorithm,
            checkpoint_path=rf"{args.ckpt_dir}\{args.scenario}_{args.algorithm}_baseline_ckpt.pkl",
            checkpoint_interval=1000,
            resume_from=args.resume,
            seed=args.seed,
        )
