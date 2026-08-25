"""
LLM module for POMO VRPTW — C+R simultaneous routing (no zones).
Adapted from R_llm_module.py + CR_llm_module.py.

Reference
---------
NCO-LLM: "Large Language Models powered Neural Solvers for
Generalized Vehicle Routing Problems", Tran et al., ICLR 2025 Workshop.

Two call points
---------------
1. Episode start — top-K start nodes for POMO (replaces zone-entry LLM)
   Phase 1 : baseline urgency priority (no event info)
   Phase 2 : weather-adjusted priority (rain forecast known at dispatch)

2. Accident event — mid-routing priority update (same event across all rollouts)
   All POMO rollouts share the same accident → one LLM call, shared mask

Structural difference from C→R
-------------------------------
  C→R : LLM gives confidence scores per stop within a zone
        → RL routes within zone following confidence order
  C+R : LLM ranks top-K customers as dispatch targets
        → bias applied ONLY at depot dispatch moments (current_node == 0)
        → within each vehicle trip RL selects freely (no LLM interference)
        → top-K = n_min_vehicles = ceil(total_demand / capacity)

Output
------
  get_start_nodes()      → List[int]   (ranked dispatch-target customer indices)
  get_accident_priority()→ Set[int]    (updated priority set after accident)
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from VRPTWOntology import VRPTWOntology

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "deepseek-r1:32b"
_USE_THINK    = False   # True = CoT on (slow), False = direct response (fast)
_NUM_PREDICT  = 6000    # deepseek-r1: always thinks → needs more tokens


# ── Event dataclasses ─────────────────────────────────────────────────────────

@dataclass
class WeatherEvent:
    """Rain forecast known at episode start (Phase 2 input)."""
    affected_nodes: List[int]          # customer indices in rain area
    start_time:     float              # raw time when rain begins
    end_time:       float              # raw time when rain ends
    rainfall_mm:    float = 10.0       # mm/h intensity


@dataclass
class AccidentEvent:
    """Road accident occurring during routing (shared across all rollouts)."""
    node_a:         int                  # affected segment (bidirectional)
    node_b:         int
    severity:       str  = "medium"      # low / medium / high (multiplier stays in env)
    affected_nodes: List[int] = field(default_factory=list)  # nodes in accident zone


# ── Ollama query ──────────────────────────────────────────────────────────────

_LLM_TIMEOUT_SEC = 1200
_LLM_MAX_RETRIES      = 2  # retries on network error
_LLM_MAX_LENGTH_RETRY = 3  # retries on done_reason == 'length' (thinking overflow)


def query_llm(prompt: str, model: str,
              think: bool = _USE_THINK,
              num_ctx: int = 65536) -> str:
    import time

    net_attempt    = 0
    length_attempt = 0

    while True:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": think,
            "options": {
                "temperature": 0.0,
                "num_ctx": num_ctx,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT_SEC) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            eval_tok   = body.get('eval_count', '?')
            prompt_tok = body.get('prompt_eval_count', '?')
            think_tok  = body.get('thinking_token_count', body.get('thinking_eval_count', '?'))
            done       = body.get('done_reason', 'stop')
            print(f"  [LLM] response in {elapsed:.1f}s  prompt_tokens={prompt_tok}"
                  f"  eval_tokens={eval_tok}  think_tokens={think_tok}  done={done}", flush=True)
            resp_text = body.get("response") or body.get("message", {}).get("content", "")
            if not resp_text:
                print(f"  [LLM:debug] body keys: {list(body.keys())}", flush=True)
            # thinking overflow: response empty despite successful HTTP call
            if done == 'length' and not resp_text:
                if length_attempt < _LLM_MAX_LENGTH_RETRY:
                    length_attempt += 1
                    print(f"  [LLM] done=length (thinking overflow), retry {length_attempt}/{_LLM_MAX_LENGTH_RETRY}...", flush=True)
                    continue
                else:
                    print(f"  [LLM] done=length exhausted — giving up", flush=True)
            return resp_text.strip()
        except urllib.error.URLError as e:
            elapsed = time.time() - t0
            net_attempt += 1
            print(f"  [LLM] network attempt {net_attempt} failed after {elapsed:.1f}s: {e}", flush=True)
            if net_attempt < _LLM_MAX_RETRIES:
                print(f"  [LLM] retrying...", flush=True)
            else:
                raise
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [LLM] error after {elapsed:.1f}s: {e}", flush=True)
            raise


def parse_priority(response: str, unvisited: List[int], top_k: int) -> Optional[List[int]]:
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
    valid_set = set(unvisited)
    parsed = [int(c) for c in raw if isinstance(c, (int, float)) and int(c) in valid_set]
    return parsed[:top_k] if parsed else None


# ── Prompt builders ───────────────────────────────────────────────────────────

# N-car collision descriptions shown to LLM (multiplier hidden).
_NCAR_DISPLAY: dict[str, str] = {
    "3-car-collision": "3-car collision",
    "4-car-pile-up":   "4-car pile-up",
    "5-car-pile-up":   "5-car pile-up",
    # legacy fallbacks
    "low":    "minor traffic disruption",
    "medium": "moderate traffic disruption",
    "high":   "major traffic disruption",
}

_NCAR_MULT: dict[str, float] = {
    "3-car-collision": 5.0,
    "4-car-pile-up":   8.5,
    "5-car-pile-up":  13.0,
    "low": 1.5, "medium": 2.0, "high": 3.0,
}


def _sev_to_mult(sev) -> float:
    """Internal multiplier — NOT shown to LLM."""
    if sev is None:
        return 5.0
    return _NCAR_MULT.get(str(sev).lower(), 5.0)


def _sev_display(sev) -> str:
    """Human-readable accident description shown to LLM."""
    if sev is None:
        return "traffic accident"
    return _NCAR_DISPLAY.get(str(sev).lower(), str(sev))


def _customer_table(
    ont: VRPTWOntology,
    customers: List[int],
    ont_ctx: dict,
    use_ontology: bool = True,
    show_demand: bool = True,
) -> str:
    concepts      = ont_ctx.get("concepts", {})
    stops_by_rain = ont_ctx.get("stops_by_rain", {})
    stops_by_acc  = ont_ctx.get("stops_by_acc",  {})
    rain_evs      = ont_ctx.get("rain_events", [])
    acc_evs       = ont_ctx.get("accident_events", [])

    # For ACC-affected stops, recompute TW_slack via the accident edge:
    # TW_close - max(tt[depot→partner] + tt[partner→c]*mult, TW_open)
    customer_set = set(customers)
    acc_slack_override: Dict[int, float] = {}
    for ev in acc_evs:
        affected = ev.get('affected_nodes', [])
        if len(affected) < 2:
            continue
        mult = _sev_to_mult(ev.get('severity'))
        for c, partner in [(affected[0], affected[1]), (affected[1], affected[0])]:
            if c in customer_set and c not in acc_slack_override:
                t_via = float(ont.tt[0, partner]) + float(ont.tt[partner, c]) * mult
                acc_slack_override[c] = float(ont.tw_close[c]) - max(t_via, float(ont.tw_open[c]))

    hdr = (f"{'ID':>4}  {'x':>6}  {'y':>6}  {'TW_open':>8}  {'TW_close':>9}"
           f"  {'svc':>5}")
    if show_demand:
        hdr += f"  {'demand':>7}"
    hdr += f"  {'dist_depot':>10}  {'TW_slack':>9}  {'wait':>6}"
    if use_ontology:
        hdr += f"  {'concept':<18}  events"
    sep = "-" * (len(hdr) + 2)
    rows = [hdr, sep]

    for c in customers:
        x    = float(ont.coords[c, 0])
        y    = float(ont.coords[c, 1])
        tw_o = float(ont.tw_open[c])
        tw_c = float(ont.tw_close[c])
        svc  = float(ont.service[c])
        dem  = float(ont.demands[c])
        dist = float(ont.tt[0, c])
        slack = acc_slack_override.get(c, tw_c - max(dist, tw_o))
        wait  = max(0.0, tw_o - dist)
        row  = (f"{c:>4}  {x:>6.1f}  {y:>6.1f}  {tw_o:>8.0f}  {tw_c:>9.0f}"
                f"  {svc:>5.0f}")
        if show_demand:
            row += f"  {dem:>7.0f}"
        row += f"  {dist:>10.1f}  {slack:>9.0f}  {wait:>6.0f}"
        if use_ontology:
            lbls = concepts.get(c, {"NormalStop"})
            routing_lbls = {l for l in lbls
                            if not l.startswith("Rain_") and not l.startswith("Acc_")}
            concept_str = ",".join(sorted(routing_lbls - {"NormalStop"}) or {"NormalStop"})
            ev_tags = []
            for i, ev in enumerate(rain_evs):
                if c in stops_by_rain.get(i, []):
                    ev_tags.append(f"Rain_{i}({ev['rainfall_mm']:.0f}mm,t={ev['t_start']:.0f}-{ev['t_end']:.0f})")
            for i, ev in enumerate(acc_evs):
                if c in stops_by_acc.get(i, []):
                    ev_tags.append(f"Acc_{i}({_sev_display(ev.get('severity','?'))},t={ev['t_start']:.0f}-{ev['t_end']:.0f})")
            row += f"  {concept_str:<22}  {', '.join(ev_tags) if ev_tags else '-'}"
        rows.append(row)
    return "\n".join(rows)


def build_start_nodes_prompt(
    inst: dict,
    unvisited: List[int],
    ont_ctx: dict,
    top_k: int,
    use_cot: bool = True,
    use_ontology: bool = True,
) -> str:
    """
    Single-call prompt: baseline priority → weather adjustment (if events exist).
    LLM reasons through both steps internally via CoT.
    """
    ont = VRPTWOntology(inst)
    T   = float(inst['T'])
    N   = int(inst['n_customers'])

    # ── Ontology section ──────────────────────────────────────────────────────
    ont_section = ""
    if use_ontology:
        if ont_ctx.get("overdue"):
            ont_section += f"\n[Ontology] OverdueStop: {ont_ctx['overdue']}"
        chronic = {c: r for c, r in ont_ctx.get("cross_ep_stats", {}).items()
                   if c in unvisited}
        if chronic:
            unserved = sorted(c for c, r in chronic.items() if r['unserved_rate'] >= 0.3)
            lated    = sorted(c for c, r in chronic.items() if r['late_rate']     >= 0.3)
            if unserved:
                ont_section += f"\n[Ontology] UnservedStop: {unserved}"
            if lated:
                ont_section += f"\n[Ontology] LatedStop: {lated}"
        # DepotViolationStop: computed after ont_guidance block (dep_violation list)
        # Injected into ont_section after dep_violation is computed below.

    # ── Weather event section (read from inst directly, independent of ontology) ─
    raw_rain_evs  = [e for e in inst.get('preset_events', []) if e['type'] == 'RAIN']
    stops_by_rain = ont_ctx.get("stops_by_rain", {})   # ontology-structured per-stop mapping
    rain_evs_list = ont_ctx.get("rain_events", raw_rain_evs)  # fallback to raw if no ontology
    unvisited_set = set(unvisited)

    if raw_rain_evs:
        rain_lines = []
        for i, ev in enumerate(raw_rain_evs):
            # per-stop mapping available only with ontology; otherwise list all affected nodes
            if stops_by_rain:
                stops = [c for c in stops_by_rain.get(i, []) if c in unvisited_set]
            else:
                stops = [c for c in ev.get('nodes', []) if c in unvisited_set]
            t_s = ev.get('trigger_time', ev.get('t_start', 0))
            t_e = t_s + ev.get('duration', ev.get('t_end', 0) - t_s)
            mm  = ev.get('rainfall_mm', 0)
            rain_lines.append(
                f"  Rain_{i}: {mm:.0f}mm/h  t={t_s:.0f}–{t_e:.0f}  stops={stops}"
            )
        if use_ontology:
            rain_affected_all = {n for ev in raw_rain_evs for n in ev.get('nodes', [])}
            at_risk = sorted(set(ont_ctx.get("overdue", [])) & rain_affected_all & unvisited_set)
            if at_risk:
                rain_lines.append(f"  EventAtRisk (OverdueStop AND rain-affected): {at_risk}")
        weather_section = (
            "## Weather Events (known at dispatch)\n"
            + "\n".join(rain_lines) + "\n"
            "Both-endpoint rule: travel slowed only if BOTH endpoints are in the same rain zone.\n"
        )
        step2_instruction = (
            "Rain increases travel time between two stops only when both are within the same rain zone "
            "(both-endpoint rule). Stops outside the rain zone are unaffected.\n"
            "The rainfall intensity, active period, and affected stops are listed above.\n"
            "Adjust the Step 1 ranking based on how weather conditions interact with each customer's "
            "time window and geographic position.\n"
        )
    else:
        weather_section   = "## Weather Events\nNone.\n"
        step2_instruction = ""

    # ── Ontology guidance ─────────────────────────────────────────────────────
    ont_guidance = (
        "Ontology concepts in the table: "
        "OverdueStop = TW deadline missed even if dispatched from depot right now (highest urgency); "
        "UnservedStop(r%) = left unserved in r% of past routing episodes; "
        "LatedStop(r%) = arrived late in r% of past episodes; "
        "EventAtRiskStop = OverdueStop AND in a weather/accident event zone (critical priority).\n"
    ) if use_ontology else ""

    # ── DepotViolationStop: feasibility warning (separate from mask-only filtering) ─
    dep_violation = []
    if use_ontology:
        dep_violation = sorted(
            c for c in unvisited
            if "DepotViolationStop" in ont.get_concepts(c, cur_node=0, cur_time=0)
        )

    # Inject DepotViolationStop into ont_section now that dep_violation is computed
    if use_ontology and dep_violation:
        ont_section += (
            f"\n[Ontology] DepotViolationStop (vehicle cannot return to depot in time "
            f"even serving only this customer first — env will mask these, but avoid ranking high): "
            f"{dep_violation}"
        )

    # EventAtRiskStop: OverdueStop AND rain-affected (already in weather_section rain_lines,
    # also surfaced here in ont_section for prominence)
    if use_ontology and raw_rain_evs:
        rain_affected_all = {n for ev in raw_rain_evs for n in ev.get('nodes', [])}
        at_risk_rain = sorted(set(ont_ctx.get("overdue", [])) & rain_affected_all & unvisited_set)
        if at_risk_rain:
            ont_section += f"\n[Ontology] EventAtRiskStop (OverdueStop AND rain-affected): {at_risk_rain}"

    cap = float(inst['vehicle_capacity'])
    table  = _customer_table(ont, unvisited, ont_ctx, use_ontology)

    return f"""## Role
You are an expert logistics planner specialising in Vehicle Routing with Time Windows (VRPTW).

## Objective
Rank the {top_k} customers that should be targeted as the **first stop when each vehicle is dispatched from the depot**.
At each dispatch moment, a vehicle leaves the depot and must commit to its first customer.
An RL policy then handles the remaining routing within that vehicle's trip.

Your ranking determines which customer each vehicle will target first:
  - Rank 1 → targeted by the 1st vehicle dispatch
  - Rank 2 → targeted by the 2nd vehicle dispatch
  - ...
  - Rank {top_k} → targeted by the {top_k}th vehicle dispatch

The {top_k} value equals the estimated minimum number of vehicles needed (ceil(total_demand / capacity)).
Each ranked customer seeds one vehicle's route cluster; RL completes the route from that seed.

Optimisation priorities (highest to lowest):
  1. Maximise number of customers served
  2. Minimise time window violations (no late deliveries)
  3. Minimise number of vehicles used
  4. Minimise total travel distance

## Problem
- Customers: {N}  |  Vehicle dispatches to plan: {top_k}  |  Time horizon: {T:.0f} min  |  Vehicle capacity: {cap:.0f} units
- Each vehicle departs from and returns to the depot (node 0)

Column definitions (high vs low meaning):
- **TW_open**: time when the service window opens.
  High → window opens late; vehicle arriving early must wait.
  Low → window opens early; vehicle can serve immediately on arrival.
- **TW_close**: deadline by which the vehicle must ARRIVE.
  Arriving after TW_close is a time window violation.
  Low → tight deadline, must be committed soon.
  High → flexible deadline, can be deferred.
- **svc (service_time)**: time spent at the stop after arrival. Vehicle departs at
  max(arrival, TW_open) + service_time.
  High → vehicle spends longer here, delaying subsequent stops.
- **demand**: cargo consumed from vehicle capacity.
  High → uses up capacity quickly, may force an early depot return.
- **dist_depot**: Euclidean travel time from depot (speed = 1 unit/min).
  High → far from depot; more time spent just reaching the customer.
  Low → close to depot; efficient to include in many routes.
- **TW_slack** = TW_close − max(dist_depot, TW_open): time remaining before the deadline
  assuming direct service from depot.
  Negative → deadline cannot be met even if dispatched immediately; already overdue.
  Small positive → urgent; must be committed early or the window will be missed.
  Large positive → flexible; can be deferred to later in the routing.
  This is the most direct signal of urgency for start-node selection.
- **wait** = max(0, TW_open − dist_depot): estimated idle time if chosen as a vehicle's first stop,
  assuming departure from depot at t=0 (episode start).
  For later vehicle dispatches the actual departure time is higher, so effective wait will be smaller
  or zero — treat this as a conservative upper bound on idle time.
  Large wait means the window opens much later — choosing this customer as a start
  wastes time the vehicle could spend on other customers.
  However, if TW_slack is also tight despite large wait, the vehicle may still need
  to depart early to avoid missing the closing deadline — read both values together.

## Customer Data
{'(concept: ontology classification | events: associated weather/accident events)' if use_ontology else ''}
{table}
{ont_section}

{weather_section}
## Task — Step-by-Step Reasoning

**Step 1 — Rank dispatch targets (without events):**
Determine the ranked order of the {top_k} customers that each vehicle should target first upon depot dispatch.
Consider ALL of the following factors together:
- **TW_slack**: urgency signal — customers with small or negative slack must be committed early
- **wait**: idle time if chosen as first stop — large wait wastes the vehicle's early departure
- **dist_depot**: reachability — a distant first stop consumes more travel time before serving begins
- **Geographic position (x, y)**: each first stop seeds one vehicle's cluster; diverse selections ensure full geographic coverage
- **Interaction effects**: a well-chosen first stop enables efficient routing to geographically and temporally compatible nearby customers
{ont_guidance}
**Step 2 — {'Incorporate weather events:' if rain_evs_list else 'No weather events — proceed with Step 1 result.'}**
{step2_instruction}
Output the **final** ranked {top_k} customer IDs after completing both steps.
Output ONLY valid JSON (no extra text):
{{"priority": [id1, id2, ..., id{top_k}], "reason": "one-line explanation"}}
""".strip()


def build_accident_prompt(
    inst: dict,
    unvisited: List[int],
    ont_ctx: dict,
    accident: AccidentEvent,
    top_k: int,
    use_cot: bool = True,
    use_ontology: bool = True,
) -> str:
    ont = VRPTWOntology(inst)

    overdue_and_acc = []
    if use_ontology:
        overdue_and_acc = sorted(set(ont_ctx.get("overdue", [])) & set(accident.affected_nodes)
                                 & set(unvisited))

    # Per-event acc stop lists
    stops_by_acc = ont_ctx.get("stops_by_acc", {})
    acc_evs_list = ont_ctx.get("accident_events", [])
    acc_detail = ""
    for i, ev in enumerate(acc_evs_list):
        stops = [c for c in stops_by_acc.get(i, []) if c in set(unvisited)]
        sev = ev.get('severity', 'high')
        acc_detail += (
            f"  Acc_{i}: {_sev_display(sev)}  "
            f"t={ev['t_start']:.0f}–{ev['t_end']:.0f}  stops={stops}\n"
        )

    guidance = (
        "Active accident events:\n" + acc_detail +
        "\nAccidents increase travel time on segments between affected nodes. "
        "Stops listed under each accident share affected road segments.\n"
        + (f"[Ontology] EventAtRisk (OverdueStop AND accident-affected): {overdue_and_acc}\n"
           if (use_ontology and overdue_and_acc) else "")
    )

    ont_line = f"[Ontology] OverdueStop: {ont_ctx.get('overdue', [])}" \
               if (use_ontology and ont_ctx.get("overdue")) else ""
    table = _customer_table(ont, unvisited, ont_ctx, use_ontology)

    return f"""## Role
You are a real-time logistics advisor managing an ongoing VRPTW delivery operation.

## Objective
An accident has occurred mid-route. Determine which of the remaining unvisited customers
should be prioritised given the current disruption.
The selected customers will be given routing priority by the RL policy.

## Remaining Customers
(TW_slack reflects pre-accident schedule; accident zones increase travel time on affected segments)
{table}
{ont_line}

## Accident Event
{guidance}
Select top-{top_k} unvisited customers to prioritise for remaining routes.

Output ONLY valid JSON:
{{"priority": [id1, id2, ..., id{top_k}], "reason": "one-line explanation"}}
""".strip()


# ── High-level API ────────────────────────────────────────────────────────────

def build_rain_adjustment_prompt(
    inst: dict,
    base_starts: List[int],
    unvisited: List[int],
    ont_ctx: dict,
    top_k: int,
    use_cot: bool = True,
    use_ontology: bool = True,
) -> str:
    """Phase 2 only: adjust base starts for rain, given base result as Phase 1."""
    ont = VRPTWOntology(inst)
    T   = float(inst['T'])

    raw_rain_evs  = [e for e in inst.get('preset_events', []) if e['type'] == 'RAIN']
    stops_by_rain = ont_ctx.get("stops_by_rain", {})
    unvisited_set = set(unvisited)

    rain_lines = []
    for i, ev in enumerate(raw_rain_evs):
        stops_in_rain = stops_by_rain.get(i, [ev.get('nodes', [])]) if stops_by_rain else [n for n in ev.get('nodes', []) if n in unvisited_set]
        if isinstance(stops_in_rain[0], list):
            stops_in_rain = stops_in_rain[0]
        stops_in_rain = [c for c in stops_in_rain if c in unvisited_set]
        t_s = ev.get('trigger_time', 0)
        t_e = t_s + ev.get('duration', 0)
        mm  = ev.get('rainfall_mm', 0)
        rain_lines.append(
            f"  Rain_{i}: {mm:.0f}mm/h  t={t_s:.0f}–{t_e:.0f}  stops={stops_in_rain}"
        )

    if use_ontology:
        rain_all = {n for ev in raw_rain_evs for n in ev.get('nodes', [])}
        at_risk  = sorted(set(ont_ctx.get("overdue", [])) & rain_all & unvisited_set)
        if at_risk:
            rain_lines.append(f"  EventAtRisk (OverdueStop AND rain-affected): {at_risk}")

    cap    = float(inst['vehicle_capacity'])
    N      = inst['n_customers']
    table  = _customer_table(ont, unvisited, ont_ctx, use_ontology)  # all customers

    ont_section = ""
    if use_ontology:
        if ont_ctx.get("overdue"):
            ont_section += f"\n[Ontology] OverdueStop: {ont_ctx['overdue']}"

    return f"""## Role
You are an expert logistics planner specialising in VRPTW.

## Objective
Adjust the baseline dispatch-target ranking to account for incoming weather.
The baseline ranking was computed without weather information.
Each ranked customer represents the intended first stop for one vehicle dispatch from the depot.

## Problem
- Customers: {N}  |  Vehicle dispatches to plan: {top_k}  |  Time horizon: {T:.0f} min  |  Vehicle capacity: {cap:.0f} units
- Each vehicle departs from and returns to the depot (node 0)

Column definitions:
- **TW_open / TW_close**: arrival interval. Before TW_open → wait; after TW_close → violation.
- **svc**: service time at stop. Vehicle departs at max(arrival, TW_open) + service_time.
- **dist_depot**: travel time from depot (speed = 1 unit/min)
- **TW_slack** = TW_close − max(dist_depot, TW_open): urgency from depot. Negative → deadline impossible.
- **wait** = max(0, TW_open − dist_depot): idle time if chosen as first stop.
  Large wait + tight TW_slack → still need early departure.

## Phase 1 Result (baseline, already computed)
The following {len(base_starts)} customers were selected as optimal starts
for baseline routing (no weather events):
  {base_starts}

## All Customers
(full data to allow substitutions if needed)
{table}
{ont_section}

## Weather Events (now known)
{chr(10).join(rain_lines)}
Both-endpoint rule: travel slowed only if BOTH endpoints are in the same rain zone.

## Phase 2 Task
Starting from the Phase 1 ranking, adjust for rain.
Rain increases travel time between stops in the same rain zone.
You may keep, reorder, swap, or substitute customers from the full table above.
Consider how rain timing and affected stops interact with each customer's TW_slack, wait, and dispatch priority.

Output the final ranked {top_k} dispatch-target customer IDs after adjustment.
Output ONLY valid JSON:
{{"priority": [id1, id2, ..., id{top_k}], "reason": "one-line explanation"}}
""".strip()


def get_start_nodes(
    inst: dict,
    top_k: int,
    model: str = DEFAULT_MODEL,
    weather: Optional[WeatherEvent] = None,
    use_cot: bool = True,
    use_ontology: bool = True,
    unvisited: Optional[List[int]] = None,
) -> List[int]:
    """Single LLM call: baseline priority → weather adjustment (if events in inst)."""
    ont   = VRPTWOntology(inst)
    N     = int(inst['n_customers'])
    all_c = unvisited if unvisited is not None else list(range(1, N + 1))
    eff_k = min(top_k, len(all_c))
    ont_ctx = ont.get_context(all_c) if use_ontology else {}

    prompt = build_start_nodes_prompt(inst, all_c, ont_ctx, eff_k, use_cot, use_ontology)
    resp   = query_llm(prompt, model)
    nodes  = parse_priority(resp, all_c, eff_k)
    if not nodes:
        nodes = sorted(all_c,
                       key=lambda c: float(ont.tw_close[c]) - float(ont.tt[0, c]))[:eff_k]
        print(f"  [LLM:start] parse failed → fallback TW_slack order")
        print(f"  [LLM:debug] response preview: {repr(resp[:300])}", flush=True)
    else:
        print(f"  [LLM:start] top-{eff_k}: {nodes[:10]}{'...' if eff_k > 10 else ''}")
    return nodes


def get_accident_priority(
    inst: dict,
    unvisited: List[int],
    accident: AccidentEvent,
    top_k: int,
    model: str = DEFAULT_MODEL,
    use_cot: bool = True,
    use_ontology: bool = True,
) -> Set[int]:
    """
    Mid-routing LLM call triggered by accident event.
    use_ontology=False: LLM+RL ablation (no ontology context in prompt).
    """
    if not unvisited:
        return set()

    ont     = VRPTWOntology(inst)
    eff_k   = min(top_k, len(unvisited))
    ont_ctx = ont.get_context(unvisited) if use_ontology else {}

    prompt   = build_accident_prompt(inst, unvisited, ont_ctx, accident, eff_k, use_cot, use_ontology)
    resp     = query_llm(prompt, model)
    priority = parse_priority(resp, unvisited, eff_k)
    if not priority:
        raise ValueError(f"[LLM:accident] parse failed. response: {resp[:200]}")
    print(f"  [LLM:accident] priority update: {priority[:10]}{'...' if eff_k > 10 else ''}")
    return set(priority)
