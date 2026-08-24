"""
LLM + Ontology baseline for VRPTW routing evaluation (ablation: no RL).

Action selection: greedy argmax of LLM confidence score — no RL policy, no learning.
Ontology provides: action pruning, high-unserved hints in prompts, OWL context for events.

Ablation axis:
  rl_baseline.py      → Pure RL (no LLM, no Ontology)
  llm_baseline.py     → LLM + Ontology greedy (no RL policy; confidence → argmax)
  main.py              → RL + LLM + Ontology (full model)
"""
from __future__ import annotations

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import time
from typing import Dict, List, Optional

from R_ontology import VRPTWOntologyManager, OntologyReasoningEngine

from R_utils import (
    TrainConfig,
    ZonePlan,
    set_seed,
    infer_solution_path,
    parse_solution_routes,
    validate_solution_routes,
    build_zone_plan_from_routes,
    build_initial_zones,
    get_feasible_actions_with_zone_control,
    prune_candidates_by_ontology,
    build_candidate_facts,
    heuristic_topk_fallback,
)
from R_llm_module import (
    query_ollama_zone_confidence,
    query_ollama_confidence,
)
from R_env import (
    WasteFleetEnv,
    EventType,
    build_travel_time_matrix,
    load_waste_benchmark_txt,
)


def _greedy_action(feasible: List[int], confidence: Dict[int, float]) -> int:
    return max(feasible, key=lambda s: confidence.get(s, 0.0))


def eval_llm(
    data_path: str,
    llm_model: str = "qwen3:32b",
    llm_use_cot: bool = True,
    llm_temperature: float = 0.0,
    num_runs: int = 1,
    use_solution_zones: bool = True,
    solution_path: Optional[str] = None,
    use_llm_event_response: bool = True,
    use_ontology: bool = True,
    seed: int = 42,
    max_steps: int = 10_000,
    late_penalty_fixed: float = 2.0,
    late_penalty_scale: float = 2.0,
    visit_reward: float = 1.0,
    zone_completion_bonus: float = 1.0,
    unserved_penalty: float = 4.0,
    llm_confidence_topk_ratio: float = 1.0,
):
    set_seed(seed)
    print(f"[LLM-Baseline] data={data_path}")
    print(f"[LLM-Baseline] model={llm_model}  cot={llm_use_cot}  "
          f"temperature={llm_temperature}  num_runs={num_runs}")
    print(f"[LLM-Baseline] RL=OFF  Ontology={'ON' if use_ontology else 'OFF'}  "
          f"LLM=ON (confidence greedy)\n")

    inst = load_waste_benchmark_txt(data_path)
    tt = build_travel_time_matrix(inst)

    if inst.preset_events:
        print(f"[Events] {len(inst.preset_events)} preset event(s):")
        for e in inst.preset_events:
            print(f"  {e.event_type.value.upper()}  "
                  f"t=[{e.trigger_time:.0f},{e.end_time:.0f}]  "
                  f"multiplier={e.multiplier:.2f}  nodes={e.affected_nodes}")
        print()

    _cfg = TrainConfig(
        data_path=data_path,
        use_solution_zones=use_solution_zones,
        solution_path=solution_path,
        prefer_on_time_candidates=True,
        use_ontology_pruning=use_ontology,
        use_ontology_reasoning=use_ontology,
        ontology_filter_on_capacity=True,
        ontology_filter_on_return_feasible=True,
        ontology_filter_late_actions=False,
        ontology_keep_depot_action=True,
        use_llm_zone_confidence=False,
        use_llm_event_response=False,
        num_episodes=num_runs,
        unserved_penalty=unserved_penalty,
        visit_reward=visit_reward,
        llm_confidence_topk_ratio=llm_confidence_topk_ratio,
    )

    zone_plan: Optional[ZonePlan] = None
    if use_solution_zones:
        sol_path = infer_solution_path(data_path, solution_path)
        if sol_path is not None:
            routes = parse_solution_routes(sol_path)
            validate_solution_routes(inst, routes)
            zone_plan = build_zone_plan_from_routes(inst, routes, _cfg)
            print(f"[Zone] loaded from solution: {zone_plan.n_zones} zones")
        else:
            print("[Zone] solution not found — using clustering")
            zone_plan = build_initial_zones(inst, _cfg, tt)
    else:
        zone_plan = build_initial_zones(inst, _cfg, tt)

    max_vehicles = zone_plan.n_zones if zone_plan else inst.vehicle_limit
    max_scope = max(len(z) for z in zone_plan.zones) + 1 if zone_plan else inst.n_customers

    env = WasteFleetEnv(
        inst=inst,
        tt=tt,
        max_vehicles=max_vehicles,
        late_penalty_fixed=late_penalty_fixed,
        late_penalty_scale=late_penalty_scale,
        visit_reward=visit_reward,
        zone_completion_bonus=zone_completion_bonus,
        unserved_penalty=unserved_penalty,
        max_scope_size=max_scope,
    )
    if zone_plan is not None:
        env.set_zone_assignment(
            zone_plan.customer_to_zone,
            zone_plan.n_zones,
            adjacent_zones=zone_plan.adjacent_zones,
        )

    n_customers = inst.n_customers
    print(f"[Env] customers={n_customers}  zones={zone_plan.n_zones if zone_plan else '?'}\n")

    # Ontology components (shared across runs for cross-episode learning)
    ontology_engine: Optional[OntologyReasoningEngine] = None
    owl_manager: Optional[VRPTWOntologyManager] = None
    if use_ontology:
        ontology_engine = OntologyReasoningEngine(
            inst=inst,
            zone_plan=zone_plan,
            base_unserved_penalty=unserved_penalty,
            penalty_threshold=_cfg.ontology_penalty_threshold,
            penalty_scale_factor=_cfg.ontology_penalty_scale_factor,
            penalty_max_multiplier=_cfg.ontology_penalty_max_multiplier,
            num_episodes=num_runs,
            unserved_window_ratio=_cfg.ontology_unserved_window_ratio,
            min_warmup_ratio=_cfg.ontology_min_warmup_ratio,
        )
        owl_manager = VRPTWOntologyManager(
            inst=inst,
            zone_plan=zone_plan,
            urgent_slack_threshold=_cfg.ontology_urgent_slack_threshold,
            critical_zone_threshold=_cfg.ontology_critical_zone_threshold,
        )
        print(f"[OWL] OntologyReasoningEngine + VRPTWOntologyManager initialized")

    run_rewards: List[float] = []
    run_served: List[int] = []
    run_on_time: List[int] = []
    run_dist: List[float] = []
    run_late: List[float] = []
    run_late_pen: List[float] = []
    run_late_cnt: List[int] = []
    run_unserved: List[int] = []
    run_llm_calls: List[int] = []
    run_times: List[float] = []
    total_start = time.time()

    for run in range(1, num_runs + 1):
        env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False
        step = 0
        llm_calls = 0
        zone_confidence: Dict[int, Dict[int, float]] = {}
        stop_late: Dict[int, float] = {}
        env._llm_processed_events = set()
        t_start = time.time()

        if num_runs > 1:
            print(f"── Run {run}/{num_runs} ──")

        # Episode-start: pre-score rain-affected zones
        if use_llm_event_response and zone_plan is not None:
            rain_events = [e for e in env.active_events if e.event_type == EventType.RAIN]
            for rain_ev in rain_events:
                affected = set(rain_ev.affected_nodes)
                for zi, zone_stops in enumerate(zone_plan.zones):
                    if any(s in affected for s in zone_stops):
                        candidates = [s for s in zone_stops if not env.served[s]]
                        if candidates:
                            rain_owl_ctx = None
                            if owl_manager is not None:
                                llm_facts = build_candidate_facts(env, candidates, zone_plan, zi)
                                owl_manager.update_state(env, ontology_engine, llm_facts)
                                rain_owl_ctx = owl_manager.get_owl_context_for_llm(
                                    feasible_actions=candidates,
                                    active_zone_idx=zi,
                                    base_penalty=unserved_penalty,
                                    env=env,
                                )
                            scores, _ = query_ollama_confidence(
                                env, candidates, rain_ev, zone_plan, zi,
                                llm_model, llm_use_cot, llm_temperature,
                                owl_context=rain_owl_ctx,
                                topk_ratio=_cfg.llm_confidence_topk_ratio,
                            )
                            zone_confidence[zi] = scores
                            llm_calls += 1
                            score_str = " → ".join(
                                f"stop{s}({v:.2f})"
                                for s, v in sorted(scores.items(), key=lambda x: -x[1])
                            )
                            print(f"  [LLM:Rain→Zone{zi}] {score_str}")

        while not (terminated or truncated):
            step += 1
            if step > max_steps:
                break

            feasible, base_zone_idx, _ = get_feasible_actions_with_zone_control(
                env=env, zone_plan=zone_plan, cfg=_cfg,
            )
            if not feasible:
                break

            # Ontology pruning
            if use_ontology:
                pruned, _ = prune_candidates_by_ontology(
                    env=env, feasible=feasible,
                    zone_plan=zone_plan, active_zone_idx=base_zone_idx, cfg=_cfg,
                )
                if pruned:
                    feasible = pruned

            # Zone entry: score all stops in this zone
            if (base_zone_idx is not None
                    and base_zone_idx not in zone_confidence
                    and zone_plan is not None):
                zone_stops = zone_plan.zones[base_zone_idx]
                high_unserved = (
                    ontology_engine.get_high_unserved_stops(base_zone_idx, zone_plan)
                    if ontology_engine is not None else None
                )
                zone_owl_ctx = None
                if owl_manager is not None:
                    unserved_zone_stops = [s for s in zone_stops if not env.served[s]]
                    llm_facts = build_candidate_facts(env, unserved_zone_stops, zone_plan, base_zone_idx)
                    owl_manager.update_state(env, ontology_engine, llm_facts)
                    zone_owl_ctx = owl_manager.get_owl_context_for_llm(
                        feasible_actions=unserved_zone_stops,
                        active_zone_idx=base_zone_idx,
                        base_penalty=unserved_penalty,
                        env=env,
                    )
                if zone_owl_ctx is not None:
                    tw_r = zone_owl_ctx.get("tw_reasoning", {})
                    tw_c = zone_owl_ctx.get("tw_context", {})
                    n_inf_total = sum(len(r["infeasible_downstream"]) for r in tw_r.values())
                    urgent = tw_c.get("urgent_stops", [])
                    high_pri = tw_c.get("high_priority_stops", [])
                    print(f"  [OWL:Zone{base_zone_idx}] propagation_stops={len(tw_r)} "
                          f"total_infeasible={n_inf_total} urgent={urgent} high_pri={high_pri}")
                scores, _ = query_ollama_zone_confidence(
                    env, zone_stops, base_zone_idx, llm_model,
                    high_unserved=high_unserved,
                    use_cot=llm_use_cot, temperature=llm_temperature,
                    owl_context=zone_owl_ctx,
                    topk_ratio=_cfg.llm_confidence_topk_ratio,
                )
                zone_confidence[base_zone_idx] = scores
                llm_calls += 1
                score_str = " → ".join(
                    f"stop{s}({v:.2f})"
                    for s, v in sorted(scores.items(), key=lambda x: -x[1])
                )
                print(f"  [LLM:Zone{base_zone_idx}] {score_str}")

            # Accident detection: re-score with OWL context
            if use_llm_event_response:
                newly_triggered = [
                    e for e in env.active_events
                    if e.event_type == EventType.ACCIDENT and e.active
                    and e not in env._llm_processed_events
                ]
                for acc_ev in newly_triggered:
                    if base_zone_idx is not None:
                        remaining = [s for s in feasible if not env.served[s]]
                        if remaining:
                            acc_owl_ctx = None
                            if owl_manager is not None:
                                llm_facts = build_candidate_facts(env, remaining, zone_plan, base_zone_idx)
                                owl_manager.update_state(env, ontology_engine, llm_facts)
                                acc_owl_ctx = owl_manager.get_owl_context_for_llm(
                                    feasible_actions=remaining,
                                    active_zone_idx=base_zone_idx,
                                    base_penalty=unserved_penalty,
                                    env=env,
                                )
                            scores, _ = query_ollama_confidence(
                                env, remaining, acc_ev, zone_plan, base_zone_idx,
                                llm_model, llm_use_cot, llm_temperature,
                                owl_context=acc_owl_ctx,
                                topk_ratio=_cfg.llm_confidence_topk_ratio,
                            )
                            if base_zone_idx in zone_confidence:
                                zone_confidence[base_zone_idx].update(scores)
                            else:
                                zone_confidence[base_zone_idx] = scores
                            llm_calls += 1
                            score_str = " → ".join(
                                f"stop{s}({v:.2f})"
                                for s, v in sorted(scores.items(), key=lambda x: -x[1])
                            )
                            print(f"  [LLM:Accident→Zone{base_zone_idx}] {score_str}")
                    env._llm_processed_events.add(acc_ev)

            # Greedy action: argmax confidence; fallback to heuristic
            action = None
            if base_zone_idx is not None and base_zone_idx in zone_confidence:
                action = _greedy_action(feasible, zone_confidence[base_zone_idx])
            if action is None:
                fb = heuristic_topk_fallback(env, feasible, 1)
                action = fb[0] if fb else feasible[0]

            _, reward, terminated, truncated, info = env.step(action, external_penalty=0.0)
            total_reward += reward
            if info.get("late", 0.0) > 0:
                stop_late[action] = float(info["late"])

            if info.get("reason") in ("all_served", "all_vehicles_retired", "vehicle_limit_reached"):
                terminated = True

        # Cross-episode ontology learning
        if ontology_engine is not None:
            ontology_engine.record_episode(env.served)

        elapsed = time.time() - t_start
        served_count = int(sum(env.served[1:]))
        avg_late = env.total_late / max(served_count, 1)

        zone_served_str = ""
        if zone_plan is not None:
            parts = [f"{sum(1 for c in z if env.served[c])}/{len(z)}" for z in zone_plan.zones]
            zone_served_str = "  zones=[" + " ".join(parts) + "]"

        if zone_plan is not None:
            print(f"\n  {'Zone':<6} {'served':>6} {'unserved':>8} {'late':>10}")
            print(f"  {'-'*34}")
            for zi, zone in enumerate(zone_plan.zones):
                z_served   = sum(1 for c in zone if env.served[c])
                z_unserved = len(zone) - z_served
                z_late     = sum(stop_late.get(c, 0.0) for c in zone)
                print(f"  Zone{zi:<2}  {z_served:>6}  {z_unserved:>8}  {z_late:>10.2f}")

        late_cnt = len(stop_late)
        run_on_time_val = served_count - late_cnt
        run_unserved_val = n_customers - served_count
        run_log = (
            f"[LLM+ONT run={run:2d}] "
            f"reward={total_reward:10.2f}  "
            f"served={run_on_time_val:3d}  tw_late={late_cnt:3d}  unserved={run_unserved_val:3d}/{n_customers}"
            f"{zone_served_str}  "
            f"veh={env.vehicles_used:2d}/{max_vehicles:2d}  "
            f"dist={env.total_distance:10.2f}  "
            f"late={env.total_late:9.2f}  "
            f"late_avg={avg_late:8.2f}  "
            f"late_pen={env.total_late_violation_penalty:9.2f}  "
            f"llm_calls={llm_calls:3d}  "
            f"run_time={elapsed:.1f}s"
        )
        print(f"\n{run_log}")

        run_rewards.append(total_reward)
        run_served.append(served_count)
        run_on_time.append(run_on_time_val)
        run_dist.append(env.total_distance)
        run_late.append(env.total_late)
        run_late_pen.append(env.total_late_violation_penalty)
        run_late_cnt.append(late_cnt)
        run_unserved.append(run_unserved_val)
        run_llm_calls.append(llm_calls)
        run_times.append(elapsed)

    total_elapsed = time.time() - total_start

    import statistics as _s

    def _mean(vals: List[float]) -> float:
        return _s.mean(vals) if vals else 0.0

    ont_tag = "+Ontology" if use_ontology else ""
    label = f"LLM{ont_tag}-Baseline ({llm_model})"
    dist_v    = _mean(run_dist)
    late_v    = _mean(run_late)
    ont_v     = _mean([float(s) for s in run_on_time])
    lcnt_v    = _mean([float(c) for c in run_late_cnt])
    unsrv_v   = _mean([float(u) for u in run_unserved])

    print(f"\n{'='*80}")
    print(f"[LLM-Baseline] Summary  runs={num_runs}  temperature={llm_temperature}  "
          f"total_time={total_elapsed:.1f}s")
    print(f"  {'Experiment':<35} {'Served':>7} {'TWLate':>7} {'Unserved':>9} {'Dist':>8} {'Late':>8}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*9} {'-'*8} {'-'*8}")
    print(f"  {label:<35} "
          f"{ont_v:>4.1f}/{n_customers:<3} "
          f"{lcnt_v:>4.1f}/{n_customers:<3} "
          f"{unsrv_v:>6.1f}/{n_customers:<3} "
          f"{dist_v:>8.1f} {late_v:>8.1f}")
    if num_runs > 1:
        dist_std  = _s.stdev(run_dist)
        late_std  = _s.stdev(run_late)
        ont_std   = _s.stdev([float(s) for s in run_on_time])
        lcnt_std  = _s.stdev([float(c) for c in run_late_cnt])
        unsrv_std = _s.stdev([float(u) for u in run_unserved])
        print(f"  {'  (± std)':<35} "
              f"{ont_std:>7.1f} "
              f"{lcnt_std:>7.1f} "
              f"{unsrv_std:>9.1f} "
              f"{dist_std:>8.1f} {late_std:>8.1f}")
    print(f"{'='*80}")

    return {
        "rewards": run_rewards,
        "served": run_served,
        "dist": run_dist,
        "late": run_late,
        "late_cnt": run_late_cnt,
        "llm_calls": run_llm_calls,
        "run_times": run_times,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLM+Ontology baseline (no RL)")
    parser.add_argument("scenario", type=str, help="Dataset name, e.g. c101 or c101_rain_C")
    parser.add_argument("--data-dir", default=os.path.join(_ROOT, "data", "Solomon"))
    parser.add_argument("--llm-model", default="qwen3:32b")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0=deterministic; >0 enables stochastic runs")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of evaluation runs (use ≥10 with --temperature >0)")
    parser.add_argument("--no-cot", action="store_true", help="Disable chain-of-thought prompting")
    parser.add_argument("--no-solution-zones", action="store_true")
    parser.add_argument("--no-event-response", action="store_true")
    parser.add_argument("--no-ontology", action="store_true", help="Disable ontology (pure LLM greedy)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-topk-ratio", type=float, default=1.0,
                        help="Fraction of stops per cluster the LLM selects+scores "
                             "(boost-only). 0=none scored, 1=all stops scored (legacy).")
    args = parser.parse_args()

    eval_llm(
        data_path=rf"{args.data_dir}\{args.scenario}.txt",
        llm_model=args.llm_model,
        llm_use_cot=not args.no_cot,
        llm_temperature=args.temperature,
        num_runs=args.runs,
        use_solution_zones=not args.no_solution_zones,
        use_llm_event_response=not args.no_event_response,
        use_ontology=not args.no_ontology,
        seed=args.seed,
        late_penalty_fixed=2.0,
        late_penalty_scale=2.0,
        visit_reward=1.0,
        unserved_penalty=1.0,
        llm_confidence_topk_ratio=args.llm_topk_ratio,
    )
