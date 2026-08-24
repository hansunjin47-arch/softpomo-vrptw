"""
LLM interface: prompt building, response parsing, Ollama query functions.

Imported by main.py and llm_baseline.py.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import random
import re
import sys
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

from R_env import WasteFleetEnv, EventType
from R_utils import ZonePlan

OLLAMA_URL       = "http://localhost:11434/api/generate"
_USE_THINK       = True   # True = CoT (slow, thorough), False = direct (fast)
_NUM_PREDICT_COT  = -1      # think=True: no generation limit
_NUM_PREDICT_FAST = 8192    # think=False: enough for 100-stop JSON (~500 tokens) with large margin
_LLM_MAX_RETRIES = 2


class LLMError(RuntimeError):
    """Raised when LLM call fails or returns unparseable output."""


# Ontology concept vocabulary — included in every CoT prompt so LLM uses
# these exact names in reasoning, making --no-ontology ablation clean.
_ONTOLOGY_CONCEPT_VOCAB = (
    "\n## Ontology Concept Vocabulary\n"
    "Label each stop in your reasoning using these concept names:\n"
    "  NormalStop        - no urgent TW constraint; standard routing priority.\n"
    "  UrgentStop        - deadline_slack <= 30; must be visited soon to avoid TW violation.\n"
    "  OverdueStop       - deadline_slack < 0; TW already passed or unreachable.\n"
    "  HighPriorityStop  - carries extra penalty weight if missed (penalty_multiplier > 1.0).\n"
    "  AccidentZone      - stop is near an accident-affected segment during the event window.\n"
    "  RainZone          - stop is in a rain-affected zone during the event window.\n"
    "  EventAtRiskStop   - UrgentStop AND in AccidentZone/RainZone; highest event-driven risk.\n"
    "If none of the above apply, label the stop NormalStop.\n"
)


def _confidence_output_instruction(
    stops: List[int],
    topk_ratio: float,
    response_instruction: str,
) -> str:
    """Output-format instruction for a confidence prompt.

    topk_ratio >= 1.0: legacy full-scoring - score every stop 0.0-1.0.
    0.0 < topk_ratio < 1.0: top-K pruning + boost-only confidence - select only the
    K stops whose priority should increase, score them in (0.0, 1.0]; any stop
    omitted from the JSON is treated as confidence=0 (no bias change).
    """
    ids_str = ", ".join(f'"{c}": <score>' for c in stops)
    if topk_ratio >= 1.0:
        return (
            f"Assign a confidence score (0.0–1.0) per stop. Higher = visit sooner.\n"
            f"IMPORTANT: include ALL {len(stops)} stops in the JSON.\n"
            f"\n{response_instruction}\n"
            f'{{"confidence": {{{ids_str}}}}}'
        )
    k = max(1, round(topk_ratio * len(stops)))
    entry_word = "entry" if k == 1 else "entries"
    return (
        f"From the {len(stops)} stops above, select the {k} stop(s) whose priority "
        f"should INCREASE THE MOST given the information above. "
        f"Assign each selected stop a confidence score in (0.0, 1.0] "
        f"(higher = stronger increase in priority).\n"
        f"Do NOT list any other stop - every stop you omit is automatically treated "
        f"as unchanged (confidence 0).\n"
        f"\n{response_instruction}\n"
        f'{{"confidence": {{"<stop_id>": <score>, ...}}}}  // exactly {k} {entry_word}'
    )


def build_confidence_prompt(
    env: WasteFleetEnv,
    stops: List[int],
    zone_idx: int,
    event=None,
    high_unserved: Optional[List[Tuple[int, float]]] = None,
    prev_scores: Optional[Dict[int, float]] = None,
    use_cot: bool = True,
    owl_context: Optional[dict] = None,
    topk_ratio: float = 1.0,
) -> str:
    depot_due = float(env.inst.tw_close[0])
    av = env.vehicles[env.active_vehicle_idx]
    cur_node = int(av.cur_node)
    cur_time = float(av.cur_time)

    stops = list(stops)
    random.Random(42).shuffle(stops)

    cap = max(float(env.inst.vehicle_capacity), 1.0)
    stop_lines = []
    for c in stops:
        travel = float(env.tt[av.cur_node, c])
        arrival = cur_time + travel
        tw_open = float(env.inst.tw_open[c])
        tw_close = float(env.inst.tw_close[c])
        service = float(env.inst.service_time[c])
        demand = float(env.inst.demands[c])
        depart = max(arrival, tw_open) + service
        deadline_slack = tw_close - arrival
        wait_time = max(0.0, tw_open - arrival)
        depot_return_slack = depot_due - depart - float(env.tt[c, 0])
        stop_lines.append(
            f"  {c}: tw=[{tw_open:.0f},{tw_close:.0f}], travel={travel:.1f}, "
            f"service_time={service:.0f}, demand={demand:.0f}, "
            f"deadline_slack={deadline_slack:.1f}, "
            f"wait_time={wait_time:.1f}, depot_return_slack={depot_return_slack:.1f}"
        )
    stops_str = "\n".join(stop_lines)

    tw_ctx = (owl_context or {}).get("tw_context", {})
    tw_reasoning = (owl_context or {}).get("tw_reasoning", {})
    owl_tw_section = ""

    if tw_reasoning:
        lines = ["\n[Ontology: TW Constraint Propagation Analysis]",
                 "The following shows cascade effects of visiting each stop next from your current position.",
                 "Use this to choose stops that preserve downstream feasibility:\n"]
        for c in stops:
            r = tw_reasoning.get(c)
            if r is None:
                continue
            n_inf = len(r["infeasible_downstream"])
            tight = r["tight_downstream"]
            slack = r["deadline_slack"]
            wait = r["wait_time"]
            parts = []
            if slack < 0:
                parts.append(f"already overdue (slack={slack:.0f}min)")
            else:
                parts.append(f"own_slack={slack:.0f}min")
            if wait > 0:
                parts.append(f"wait={wait:.0f}min")
            if n_inf > 0:
                inf_list = ", ".join(str(x) for x in r["infeasible_downstream"][:5])
                parts.append(f"causes {n_inf} stops INFEASIBLE [{inf_list}]")
            elif tight:
                tight_desc = ", ".join(f"{b}(slack->{s:.0f})" for b, s in tight[:3])
                parts.append(f"tightens: {tight_desc}")
            else:
                parts.append("no cascade infeasibility")
            lines.append(f"  Stop {c}: {', '.join(parts)}")
        owl_tw_section = "\n".join(lines) + "\n"

    owl_cls_section = ""
    if tw_ctx:
        lines = []
        if tw_ctx.get("urgent_stops"):
            lines.append(f"  UrgentStop (end_slack ≤ 30): {sorted(tw_ctx['urgent_stops'])}")
        if tw_ctx.get("overdue_stops"):
            lines.append(f"  OverdueStop (end_slack < 0): {sorted(tw_ctx['overdue_stops'])}")
        if tw_ctx.get("high_priority_stops"):
            lines.append(f"  HighPriorityStop (penalty_multiplier > 1.0): {sorted(tw_ctx['high_priority_stops'])}")
        if lines:
            owl_cls_section = (
                "\n[OWL TW Classifications]\n"
                + "\n".join(lines) + "\n"
            )
    owl_tw_section = owl_tw_section + owl_cls_section

    unserved_section = ""
    if high_unserved:
        lines = []
        for s, r in high_unserved:
            prev = f", previous confidence={prev_scores[s]:.2f}" if prev_scores and s in prev_scores else ""
            lines.append(f"  {s}: unserved_rate={r:.0%}{prev}")
        unserved_section = (
            "\n[Ontology Alert] The following stops have been frequently unvisited in recent episodes"
            + (" (previous confidence shown for reference)" if prev_scores else "")
            + ". Prioritize them by re-examining deadline_slack and wait_time:\n"
            + "\n".join(lines) + "\n"
        )

    event_section = ""
    owl_event_section = ""
    phase2_steps = ""

    if event is not None:
        ev_ctx = (owl_context or {}).get("event_context", {})

        if event.event_type == EventType.ACCIDENT:
            event_section = (
                f"\n## Event\n"
                f"ACCIDENT: Between Stop {event.affected_nodes[0]} and Stop {event.affected_nodes[1]}, "
                f"travel time on this segment has increased (exact amount unknown). "
                f"Expected recovery at t={event.end_time:.0f} "
                f"(current time: t={cur_time:.0f}, duration: {event.duration:.0f} time units).\n"
            )
            acc_zones = ev_ctx.get("accident_zones", [])
            if acc_zones:
                zone_lines = []
                for az in acc_zones:
                    zone_lines.append(
                        f"  AccidentZone({az['zone_id']}): severity={az['severity']}, "
                        f"active t={az['active_start']:.0f}–{az['active_end']:.0f}, "
                        f"stops={az['stops_in_zone']}"
                    )
                at_risk = ev_ctx.get("event_at_risk_stops", [])
                if at_risk:
                    zone_lines.append(f"  EventAtRiskStop (urgent AND accident-affected): {sorted(at_risk)}")
                owl_event_section = (
                    "\n[OWL Event Classifications - use in Phase 2 only]\n"
                    + "\n".join(zone_lines) + "\n"
                )
            phase2_steps = (
                f"Segment: Stop {event.affected_nodes[0]} ↔ Stop {event.affected_nodes[1]} is blocked.\n"
                f"- Unaffected stop + tight TW → keep high (maintain Phase 1 score)\n"
                f"- Affected stop + delay feasible (can wait for recovery at t={event.end_time:.0f}) → reduce significantly\n"
                f"- EventAtRiskStop (accident-affected AND urgent TW) → moderate; must go despite delay\n"
                f"Output the final adjusted score.\n"
            )

        else:  # RAIN
            rainfall_mm = getattr(event, "rainfall_mm", 0.0)
            probability = getattr(event, "probability", 100.0)
            event_section = (
                f"\n## Event\n"
                f"RAIN FORECAST: stops {event.affected_nodes}, "
                f"rainfall {rainfall_mm:.0f}mm/h, probability {probability:.0f}%. "
                f"Expected from t={event.trigger_time:.0f} to t={event.end_time:.0f} "
                f"(current time: t={cur_time:.0f}). "
                f"Exact travel time increase is unknown - infer severity from rainfall intensity.\n"
            )
            rain_zones = ev_ctx.get("rain_zones", [])
            if rain_zones:
                zone_lines = []
                for rz in rain_zones:
                    zone_lines.append(
                        f"  RainZone({rz['zone_id']}): {rz['rainfall_mm']:.0f}mm/h, "
                        f"prob={rz['probability']:.0f}%, "
                        f"active t={rz['active_start']:.0f}–{rz['active_end']:.0f}, "
                        f"stops={rz['stops_in_zone']}"
                    )
                at_risk = ev_ctx.get("event_at_risk_stops", [])
                if at_risk:
                    zone_lines.append(f"  EventAtRiskStop (urgent AND rain-affected): {sorted(at_risk)}")
                zone_lines.append(
                    "  Both-endpoint rule: an edge (A→B) is slowed only if both A and B "
                    "are in the same RainZone."
                )
                owl_event_section = (
                    "\n[OWL Event Classifications - use in Phase 2 only]\n"
                    + "\n".join(zone_lines) + "\n"
                )
            phase2_steps = (
                f"Rainfall: {rainfall_mm:.0f}mm/h ({probability:.0f}% probability), "
                f"active t={event.trigger_time:.0f} to t={event.end_time:.0f}.\n"
                f"Both-endpoint rule: edge (A→B) is slowed only if both A and B are in the same RainZone.\n"
                f"- EventAtRiskStop (rain-affected AND urgent TW) AND rain likely delays → reduce significantly\n"
                f"- Urgent TW AND can still arrive despite rain → keep high (maintain Phase 1 score)\n"
                f"- Relaxed TW → reduce further; safely defer until rain eases\n"
                f"Output the final adjusted score.\n"
            )

    prefix = "" if use_cot else "/no_think\n"
    if use_cot:
        has_propagation = bool(tw_reasoning)
        propagation_guidance = (
            "\n- cascade effect (from Ontology TW Propagation Analysis above): "
            "if visiting this stop first causes other stops to become infeasible, "
            "consider whether the urgency of this stop justifies the loss of downstream stops. "
            "A stop with moderate deadline_slack that causes 0 infeasible downstream may be preferred "
            "over a stop with tighter slack that causes 3 stops to become infeasible.\n"
        ) if has_propagation else ""
        phase1_guidance = (
            "Phase 1 - Baseline Urgency Scoring:\n"
            "Use deadline_slack as the primary urgency signal (see Column definitions above).\n"
            "Consider wait_time and depot_return_slack together with deadline_slack - "
            "do not treat any column in isolation.\n"
            + propagation_guidance +
            "\nFor EACH stop, cite its Ontology concept(s) by name from the "
            "## Ontology Concept Vocabulary above before stating the score. "
            "Example: 'Stop 42: UrgentStop (deadline_slack=5) → 0.95'. "
            "If no special concept applies, write NormalStop.\n"
            "Assign scores 0.0–1.0 reflecting actual urgency differences.\n"
        )
        phase2_intro = (
            "\nPhase 2 - Event Adjustment:\n"
            "Starting from the Phase 1 score for each stop, adjust as follows:\n"
        ) if phase2_steps else ""
        task_block = phase1_guidance + phase2_intro + (phase2_steps if phase2_steps else "")
        response_instruction = "Show your reasoning first, then end with JSON in this exact format:"
    else:
        task_block = ""
        response_instruction = "Respond with JSON only in this exact format:"

    prompt = f"""{prefix}## Role
You are an expert vehicle routing assistant for a Waste Collection VRPTW.

## Objective
Priority (highest to lowest):
  1. Maximise stops served within their time windows [tw_open, tw_close].
     A stop is served only if the vehicle arrives before tw_close.
  2. Minimise late arrivals (arriving after tw_close is a time window violation).
  3. Minimise total travel distance.

Maximising service count takes absolute precedence - fewer stops with less travel is strictly worse.
Hard constraint: the vehicle must return to depot before t={depot_due:.0f}.

## Column definitions
- tw=[tw_open, tw_close]: must arrive before tw_close. Arriving before tw_open means waiting.
- service_time: time spent at the stop. Vehicle departs at max(arrival, tw_open) + service_time.
  High → vehicle spends longer here, delaying all subsequent stops in this zone.
- demand: cargo consumed from vehicle capacity.
  High → vehicle capacity exhausted faster.
- deadline_slack = tw_close - arrival: time remaining before the arrival deadline.
  Negative      → hard miss; vehicle cannot arrive before the window closes.
  ~0 to small positive → urgent; any further delay will cause a miss.
  Large positive → flexible; stop can be deferred later in the zone.
  This is the most important urgency signal.
- wait_time = max(0, tw_open - arrival): idle time if the window is not yet open.
  Zero          → service starts immediately on arrival.
  Large positive → window opens much later; visiting now forces unnecessary idling.
  Exception: if deadline_slack is also tight despite large wait_time, the vehicle must
             still depart early to avoid missing the closing deadline - always read both together.
- depot_return_slack = depot_due - depart - travel_to_depot: margin to return after this stop.
  Negative      → depot deadline would be violated; stop must not be served.
  Small positive → tight; risk increases if more stops remain in this zone.
  Large positive → safe margin to return.

## Current State
Location: stop {cur_node}, Time: t={cur_time:.0f}, Depot closes: t={depot_due:.0f}

## Zone {zone_idx} stops:
{stops_str}
{owl_tw_section}{unserved_section}{event_section}{owl_event_section}{_ONTOLOGY_CONCEPT_VOCAB if use_cot else ""}## Task
{task_block}{_confidence_output_instruction(stops, topk_ratio, response_instruction)}
""".strip()

    return prompt


def build_zone_confidence_prompt(
    env: WasteFleetEnv,
    zone_stops: List[int],
    zone_idx: int,
    high_unserved: Optional[List[Tuple[int, float]]] = None,
    use_cot: bool = True,
    prev_scores: Optional[Dict[int, float]] = None,
    owl_context: Optional[dict] = None,
    topk_ratio: float = 1.0,
) -> str:
    return build_confidence_prompt(
        env, zone_stops, zone_idx,
        event=None, high_unserved=high_unserved,
        prev_scores=prev_scores, use_cot=use_cot,
        owl_context=owl_context, topk_ratio=topk_ratio,
    )


def build_event_prompt(
    env: WasteFleetEnv,
    feasible: List[int],
    event,
    zone_plan: Optional[ZonePlan],
    base_zone_idx: Optional[int],
    use_cot: bool = True,
    owl_context: Optional[dict] = None,
    topk_ratio: float = 1.0,
) -> str:
    return build_confidence_prompt(
        env, feasible, base_zone_idx if base_zone_idx is not None else 0,
        event=event, use_cot=use_cot, owl_context=owl_context,
        topk_ratio=topk_ratio,
    )


def build_instance_confidence_prompt(
    env: WasteFleetEnv,
    zone_plan: ZonePlan,
    event=None,
    use_cot: bool = True,
    prev_scores: Optional[Dict[int, float]] = None,
    target_zone_indices: Optional[List[int]] = None,
    topk_ratio: float = 1.0,
) -> str:
    """Single-call prompt for all (or selected) zones in one instance.

    target_zone_indices: if given, only include those zones (used for accident re-calls
    to focus on affected zones only — much smaller prompt, faster response).

    Features computed from depot at t=0 for pre-routing calls.
    For mid-routing accident re-calls: only unserved stops are included
    (env.served already reflects routing progress), and prev_scores shows
    previous confidence values for context.
    """
    depot_due = float(env.inst.tw_close[0])
    av = env.vehicles[env.active_vehicle_idx]

    # Use current vehicle position for mid-routing calls; depot for pre-routing
    is_mid_routing = av.cur_node != 0 or av.cur_time > 0
    ref_node  = int(av.cur_node)
    ref_time  = float(av.cur_time)

    zone_sections = []
    all_stops: List[int] = []

    cap = max(float(env.inst.vehicle_capacity), 1.0)
    zones_iter = (
        [(i, zone_plan.zones[i]) for i in target_zone_indices]
        if target_zone_indices is not None
        else list(enumerate(zone_plan.zones))
    )
    for z_idx, z_stops in zones_iter:
        # Only include unserved stops
        unserved = [c for c in sorted(z_stops) if env.served[c] == 0]
        if not unserved:
            continue   # skip fully-served zones
        stop_lines = []
        for c in unserved:
            if is_mid_routing:
                travel  = float(env.tt_base[ref_node, c])
                arrival = ref_time + travel
            else:
                travel  = float(env.tt_base[0, c])
                arrival = travel
            tw_open   = float(env.inst.tw_open[c])
            tw_close  = float(env.inst.tw_close[c])
            service   = float(env.inst.service_time[c])
            demand    = float(env.inst.demands[c])
            depart    = max(arrival, tw_open) + service
            deadline_slack     = tw_close - arrival
            wait_time          = max(0.0, tw_open - arrival)
            depot_return_slack = depot_due - depart - float(env.tt_base[c, 0])
            prev_tag = f", prev={prev_scores[c]:.2f}" if prev_scores and c in prev_scores else ""
            stop_lines.append(
                f"  {c}: tw=[{tw_open:.0f},{tw_close:.0f}], travel={travel:.1f}, "
                f"service_time={service:.0f}, demand={demand:.0f}, "
                f"deadline_slack={deadline_slack:.1f}, "
                f"wait_time={wait_time:.1f}, depot_return_slack={depot_return_slack:.1f}"
                + prev_tag
            )
            all_stops.append(c)
        zone_sections.append(
            f"### Zone {z_idx} ({len(unserved)} unserved stops):\n" + "\n".join(stop_lines)
        )

    zones_str = "\n\n".join(zone_sections)

    # State header differs for pre-routing vs mid-routing
    if is_mid_routing:
        state_header = (
            f"All features relative to current vehicle position: "
            f"stop {ref_node}, t={ref_time:.0f}. "
            f"Depot closes at t={depot_due:.0f}.\n"
            f"Only unserved stops shown."
        )
    else:
        state_header = (
            f"All vehicles depart from depot (node 0) at t=0. "
            f"Scores are computed pre-routing.\n"
            f"Each zone is served by one dedicated vehicle."
        )

    # Event section (rain/accident known at dispatch)
    event_section = ""
    phase2_steps  = ""
    if event is not None:
        if event.event_type == EventType.ACCIDENT:
            event_section = (
                f"\n## Event\n"
                f"ACCIDENT: Between Stop {event.affected_nodes[0]} and Stop {event.affected_nodes[1]}, "
                f"travel time increased. Recovery at t={event.end_time:.0f}.\n"
            )
            phase2_steps = (
                f"Segment Stop {event.affected_nodes[0]} ↔ Stop {event.affected_nodes[1]} is blocked.\n"
                f"- Unaffected stop + tight TW → keep Phase 1 score\n"
                f"- Affected stop + delay feasible → reduce significantly\n"
                f"Output final adjusted score.\n"
            )
        else:  # RAIN
            rainfall_mm = getattr(event, "rainfall_mm", 0.0)
            probability = getattr(event, "probability", 100.0)
            event_section = (
                f"\n## Event\n"
                f"RAIN: stops {event.affected_nodes}, "
                f"{rainfall_mm:.0f}mm/h ({probability:.0f}% prob), "
                f"t={event.trigger_time:.0f}–{event.end_time:.0f}.\n"
                f"Both-endpoint rule: edge slowed only if both endpoints are in the rain zone.\n"
            )
            phase2_steps = (
                f"Rainfall {rainfall_mm:.0f}mm/h, t={event.trigger_time:.0f}–{event.end_time:.0f}.\n"
                f"- Rain-affected + tight TW → reduce significantly\n"
                f"- Tight TW + reachable despite rain → keep Phase 1 score\n"
                f"- Relaxed TW → reduce further\n"
                f"Output final adjusted score.\n"
            )

    prefix = "" if use_cot else "/no_think\n"

    if use_cot:
        phase1_guidance = (
            "Phase 1 - Baseline Urgency Scoring:\n"
            "Use deadline_slack as the primary urgency signal (see Column definitions above).\n"
            "Consider wait_time and depot_return_slack together with deadline_slack - "
            "do not treat any column in isolation.\n"
            "For EACH stop, cite its Ontology concept(s) by name from the "
            "## Ontology Concept Vocabulary above before stating the score. "
            "Example: 'Stop 42: UrgentStop (deadline_slack=5) → 0.95'. "
            "If no special concept applies, write NormalStop.\n"
            "Assign scores 0.0–1.0 reflecting actual urgency differences.\n"
        )
        phase2_intro  = "\nPhase 2 - Event Adjustment:\nAdjust Phase 1 scores:\n" if phase2_steps else ""
        task_block    = phase1_guidance + phase2_intro + (phase2_steps if phase2_steps else "")
        response_instr = "Show your reasoning first, then end with JSON in this exact format:"
    else:
        task_block     = ""
        response_instr = "Respond with JSON only in this exact format:"

    prompt = f"""{prefix}## Role
You are an expert vehicle routing assistant for a Waste Collection VRPTW.

## Objective
Priority (highest to lowest):
  1. Maximise stops served within their time windows [tw_open, tw_close].
     A stop is served only if the vehicle arrives before tw_close.
  2. Minimise late arrivals (arriving after tw_close is a time window violation).
  3. Minimise total travel distance.

Maximising service count takes absolute precedence.
Hard constraint: each vehicle must return to depot before t={depot_due:.0f}.

## Current State
{state_header}

## Column definitions
- service_time: time at stop; vehicle departs at max(arrival, tw_open) + service_time.
  High → delays subsequent stops.
- demand: capacity consumed. High → exhausts capacity faster.
- deadline_slack = tw_close - arrival:
  Negative      → hard miss; cannot be served.
  ~0 to small positive → urgent.
  Large positive → flexible.  [Most important signal]
- wait_time = max(0, tw_open - arrival): idle time.
  Large + tight deadline_slack → must still depart early.
- depot_return_slack = depot_due - depart - travel_to_depot:
  Negative → depot deadline violated.  Small positive → tight.
- prev (shown where available): previous confidence score before this re-assessment.

## Remaining Stops by Zone
{zones_str}
{event_section}{_ONTOLOGY_CONCEPT_VOCAB if use_cot else ""}
## Task
{task_block}{_confidence_output_instruction(all_stops, topk_ratio, response_instr)}
""".strip()

    return prompt


def query_ollama_instance_confidence(
    env: WasteFleetEnv,
    zone_plan: ZonePlan,
    model_name: str,
    event=None,
    use_cot: bool = True,
    temperature: float = 0.0,
    prev_scores: Optional[Dict[int, float]] = None,
    target_zone_indices: Optional[List[int]] = None,
    topk_ratio: float = 1.0,
) -> Dict[int, Dict[int, float]]:
    """LLM confidence call for specified zones (or all zones if target_zone_indices=None).

    target_zone_indices: for accident mid-routing re-calls, pass only the zones
    containing accident-affected stops → smaller prompt, faster response.
    Returns {zone_idx: {stop: score}}.

    topk_ratio <= 0: skip the LLM call entirely - all stops get confidence=0.
    """
    zones_to_use = (
        [(i, zone_plan.zones[i]) for i in target_zone_indices]
        if target_zone_indices is not None
        else list(enumerate(zone_plan.zones))
    )
    # Only include unserved stops in the LLM call
    all_stops: List[int] = [
        c for _, z_stops in zones_to_use
        for c in sorted(z_stops)
        if env.served[c] == 0
    ]
    if not all_stops:
        return {}   # nothing to score

    if topk_ratio <= 0:
        return {z_idx: {c: 0.0 for c in z_stops} for z_idx, z_stops in zones_to_use}

    prompt  = build_instance_confidence_prompt(
        env, zone_plan, event=event, use_cot=use_cot,
        prev_scores=prev_scores, target_zone_indices=target_zone_indices,
        topk_ratio=topk_ratio)
    payload = _build_payload(model_name, prompt, use_cot, temperature)

    output = _call_ollama(payload, "instance")
    flat_scores, _ = parse_llm_confidence(output, all_stops)

    if all(v == 0.0 for v in flat_scores.values()):
        preview = output[:200].replace("\n", " ") if output else "(empty)"
        print(f"  [LLM:warn] instance prompt returned all-zero scores — skipping bias for this call.\n"
              f"  Output preview: {preview}")

    # Split flat scores back into per-zone dicts (only targeted zones)
    zone_scores: Dict[int, Dict[int, float]] = {}
    for z_idx, z_stops in zones_to_use:
        zone_scores[z_idx] = {c: flat_scores.get(c, 0.0) for c in z_stops}
        sorted_z = sorted(zone_scores[z_idx].items(), key=lambda x: -x[1])
        score_str = " → ".join(f"stop{s}({v:.2f})" for s, v in sorted_z)
        print(f"  [LLM:Zone{z_idx}] {score_str}")

    return zone_scores


def parse_llm_confidence(
    text: str,
    feasible: List[int],
) -> Tuple[Dict[int, float], str]:
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    reasoning = ""
    # 0.0 = "not selected by LLM / no change" (boost-only design: only LLM-selected
    # stops get a positive score, everything else stays at the no-bias baseline).
    default = {c: 0.0 for c in feasible}

    try:
        # Find the outermost { that contains "confidence"
        conf_idx = text.rfind('"confidence"')
        if conf_idx != -1:
            # Find the { that opens the object containing "confidence"
            idx = text.rfind('{', 0, conf_idx)
        else:
            idx = text.rfind('{')
        if idx == -1:
            return default, reasoning
        match = re.search(r'\{.*\}', text[idx:], flags=re.DOTALL)
        if not match:
            return default, reasoning
        obj = json.loads(match.group(0))
        reasoning = obj.get("reasoning", "")
        raw = obj.get("confidence", {})

        raw = {k.replace("stop_", "").replace("stop", "").strip(): v for k, v in raw.items()}
        result: Dict[int, float] = {}
        for c in feasible:
            key = str(c)
            val = raw.get(key, 0.0)
            result[c] = float(max(0.0, min(1.0, val)))
        return result, reasoning
    except Exception:
        return default, ""


def _build_payload(model_name: str, prompt: str, use_cot: bool, temperature: float) -> bytes:
    think = _USE_THINK and use_cot
    num_predict = _NUM_PREDICT_COT if think else _NUM_PREDICT_FAST
    # Estimate prompt tokens (~4 chars/token); pick ctx that fits prompt + output
    prompt_tokens_est = len(prompt) // 4
    if prompt_tokens_est > 3000:      # instance-level (100 stops): needs big ctx
        num_ctx = 32768
    elif prompt_tokens_est > 2000:    # per-zone or mid-routing
        num_ctx = 16384
    else:
        num_ctx = 8192
    return json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }).encode("utf-8")


def _call_ollama(payload: bytes, label: str) -> str:
    import time
    for attempt in range(1, _LLM_MAX_RETRIES + 1):
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=None) as resp:  # no timeout
                body = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            print(f"  [LLM:{label}] response in {elapsed:.1f}s", flush=True)
            output = body.get("response") or body.get("message", {}).get("content", "")
            if not output:
                print(f"  [LLM:debug] body keys: {list(body.keys())}", flush=True)
                print(f"  [LLM:debug] done_reason: {body.get('done_reason')}", flush=True)
            return output.strip()
        except urllib.error.URLError as e:
            elapsed = time.time() - t0
            is_timeout = isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError)
            if is_timeout:
                print(f"  [LLM:{label}] timeout after {elapsed:.1f}s - terminating.", flush=True)
                sys.exit(1)
            print(f"  [LLM:{label}] attempt {attempt} failed after {elapsed:.1f}s: {e}", flush=True)
            if attempt < _LLM_MAX_RETRIES:
                print(f"  [LLM:{label}] retrying...", flush=True)
            else:
                raise
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [LLM:{label}] error after {elapsed:.1f}s: {e}", flush=True)
            raise


def query_ollama_zone_confidence(
    env: WasteFleetEnv,
    zone_stops: List[int],
    zone_idx: int,
    model_name: str,
    high_unserved: Optional[List[Tuple[int, float]]] = None,
    use_cot: bool = True,
    temperature: float = 0.0,
    prev_scores: Optional[Dict[int, float]] = None,
    owl_context: Optional[dict] = None,
    topk_ratio: float = 1.0,
) -> Tuple[Dict[int, float], str]:
    if topk_ratio <= 0:
        return {c: 0.0 for c in zone_stops}, ""

    prompt  = build_zone_confidence_prompt(
        env, zone_stops, zone_idx, high_unserved, use_cot, prev_scores, owl_context,
        topk_ratio=topk_ratio)
    payload = _build_payload(model_name, prompt, use_cot, temperature)
    output  = _call_ollama(payload, f"zone{zone_idx}")
    scores, _ = parse_llm_confidence(output, zone_stops)

    missing = set(zone_stops) - set(scores.keys())
    if missing:
        raise LLMError(f"[LLM:missing] zone={zone_idx} missing stops={sorted(missing)}")
    if all(v == 0.0 for v in scores.values()):
        raise LLMError(
            f"[LLM:parse_fail] zone={zone_idx} returned all defaults.\n"
            f"  Output preview: {output[:400].replace(chr(10), ' ')}"
        )
    return scores, output


def query_ollama_confidence(
    env: WasteFleetEnv,
    feasible: List[int],
    event,
    zone_plan: Optional[ZonePlan],
    base_zone_idx: Optional[int],
    model_name: str,
    use_cot: bool = True,
    temperature: float = 0.0,
    owl_context: Optional[dict] = None,
    topk_ratio: float = 1.0,
) -> Tuple[Dict[int, float], str]:
    if topk_ratio <= 0:
        return {c: 0.0 for c in feasible}, ""

    prompt  = build_event_prompt(
        env, feasible, event, zone_plan, base_zone_idx, use_cot, owl_context,
        topk_ratio=topk_ratio)
    payload = _build_payload(model_name, prompt, use_cot, temperature)
    output  = _call_ollama(payload, "event")
    scores, reasoning = parse_llm_confidence(output, feasible)

    if all(v == 0.0 for v in scores.values()):
        raise LLMError(
            f"[LLM:parse_fail] event confidence returned all defaults.\n"
            f"  Output preview: {output[:400].replace(chr(10), ' ')}"
        )
    return scores, reasoning
