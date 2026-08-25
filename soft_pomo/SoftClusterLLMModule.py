"""
SoftClusterLLMModule.py — LLM cluster-confidence functions for Soft-Clustering POMO.

Each vehicle is assigned a soft cluster (K-means geographic grouping).
LLM assigns confidence scores (0.0-1.0) to customers WITHIN each cluster considering
TW urgency and events.  Scores are used directly as per-step logit bias throughout
that vehicle's entire route.

Key difference from original_POMO (VRPTWLLMModule):
  original_POMO: LLM → top-K global start nodes → bias ONLY at depot dispatch moments
  soft_pomo    : LLM → per-cluster confidence   → bias at EVERY routing step per vehicle
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from typing import Dict, List
from VRPTWLLMModule import (
    query_llm, _customer_table,
    DEFAULT_MODEL,
)
from SoftClusterOntology import SoftClusterOntology


# ── Reward-config → LLM objective mapping ────────────────────────────────────
# Aligned with RL reward configs in train_vrptw.py.
# K is excluded (LLM does not control cluster count).
# Metric definitions are always appended below the priority line.
_CONFIG_OBJECTIVES: dict[str, str] = {
    'A': "Minimise in order of priority: (1) Total travel distance D, (2) Total lateness Lt.",
    'B': "Minimise in order of priority: (1) Total travel distance D, (2) Total lateness Lt.",
    'C': "Minimise in order of priority: (1) Total travel distance D, (2) Total lateness Lt.",
    'D': "Minimise in order of priority: (1) Total travel distance D, (2) Total lateness Lt.",
    'E': "Minimise in order of priority: (1) Total lateness Lt, (2) Total travel distance D.",
    'F': "Minimise in order of priority: (1) Total lateness Lt, (2) Total travel distance D.",
    'G': "Minimise in order of priority: (1) Number of late customers Lc, (2) Total lateness Lt, (3) Total travel distance D.",
}
_METRIC_DEFS = (
    "- Lt: sum of arrival-time overflows beyond TW_close across all customers (minutes)\n"
    "- D: total Euclidean travel distance across all routes\n"
    "- Lc: number of customers whose arrival time exceeds their TW_close"
)
_DEFAULT_OBJECTIVE = _CONFIG_OBJECTIVES['F']


def _objective_str(reward_config: str | None) -> str:
    obj = _CONFIG_OBJECTIVES.get(reward_config or 'F', _DEFAULT_OBJECTIVE)
    return f"{obj}\n{_METRIC_DEFS}"


# ── Severity helpers ──────────────────────────────────────────────────────────

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


# ── BKS few-shot examples ─────────────────────────────────────────────────────

# One instance per Solomon family — fixed regardless of benchmark.
# C-type: clustered customers, long TW windows.
# RC-type: semi-clustered, medium TW windows.
# R-type: random layout, tight TW windows.
_BKS_REFERENCE_INSTANCES = ['c102', 'rc102', 'r102']


def _parse_sol_file(path: str) -> list[list[int]]:
    """Parse a Solomon BKS solution file → list of routes (1-indexed node IDs)."""
    routes = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith('route'):
                colon = line.find(':')
                if colon >= 0:
                    nodes = [int(x) for x in line[colon + 1:].split()]
                    if nodes:
                        routes.append(nodes)
    return routes


def build_bks_fewshot_block(data_dir: str, benchmark: str = '') -> str:
    """Build BKS few-shot examples (one per Solomon family) as proper input→output pairs.

    Always uses c102, rc102, r102 regardless of the current benchmark.
    Returns an empty string if data files are missing or loading fails.
    """
    from vrptw_env import load_solomon  # already on sys.path

    _META = {
        'c102':  ('C-type  (clustered customers, long TW windows)',
                  'Tight periodic TW pattern — ranked by chronological deadline; '
                  'final node has very large TW_slack so is least urgent'),
        'rc102': ('RC-type (semi-clustered, medium TW windows)',
                  'First node visited for geometric route entry despite large slack; '
                  'remainder ranked by TW deadline order'),
        'r102':  ('R-type  (random layout, tight TW windows)',
                  'First node is geographically efficient depot-adjacent entry; '
                  'tight-TW nodes (slack≈10) then ordered by TW_close'),
    }

    blocks: list[str] = []
    for idx, name in enumerate(_BKS_REFERENCE_INSTANCES, 1):
        data_path = os.path.join(data_dir, f'{name.upper()}.txt')
        sol_path  = os.path.join(data_dir, f'{name}_sol.txt')
        if not (os.path.isfile(data_path) and os.path.isfile(sol_path)):
            continue
        try:
            inst   = load_solomon(data_path)
            routes = _parse_sol_file(sol_path)
        except Exception as e:
            print(f'[BKS fewshot] failed to load {name}: {e}')
            continue

        T            = float(inst['T'])
        tw_open_raw  = inst['node_tw_open']
        tw_close_raw = inst['node_tw_close']
        tt_raw       = inst['tt']

        chosen: list[int] | None = None
        for r in routes:
            if 4 <= len(r) <= 8:
                chosen = r
                break
        if chosen is None:
            continue

        slacks = {
            c: round(float(tw_close_raw[c - 1]) * T
                     - max(float(tt_raw[0, c]) * T, float(tw_open_raw[c - 1]) * T), 1)
            for c in chosen
        }

        hdr  = f"  {'ID':>4}  {'TW_open':>8}  {'TW_close':>9}  {'TW_slack':>9}  {'dist_depot':>10}"
        sep  = "  " + "-" * (len(hdr) - 2)
        rows = [hdr, sep]
        for c in sorted(chosen, key=lambda x: float(tw_close_raw[x - 1]) * T):
            tw_o = float(tw_open_raw[c - 1]) * T
            tw_c = float(tw_close_raw[c - 1]) * T
            dist = float(tt_raw[0, c])        * T
            rows.append(f"  {c:>4}  {tw_o:>8.1f}  {tw_c:>9.1f}  {slacks[c]:>9.1f}  {dist:>10.1f}")

        label, reason = _META.get(name, (name.upper(), ''))
        output_json = (
            '{"ranking": {"1": ' + str(chosen) + '}, '
            '"reason": "' + reason + '"}'
        )

        blocks.append(
            f"### Example {idx} — {label}\n"
            f"Input (one vehicle's assigned customers):\n" + "\n".join(rows) + "\n"
            f"Output:\n{output_json}"
        )

    if not blocks:
        return ""

    return (
        "## Examples\n"
        "(BKS-optimal rankings for similar instances — study before ranking your instance below.)\n"
        "Ranking reflects ALL factors: TW urgency, geographic position, travel time.\n"
        "\n"
        + "\n\n".join(blocks)
    )


def build_rl_fewshot_block(cache_dir: str, data_dir: str,
                           instance_name: str = 'r106') -> str:
    """Build RL few-shot example from RL-derived cluster for one R-type instance.

    Reads {instance_name}_rl_cluster.json from cache_dir and formats the RL
    visit order as a ranking example (analogous to BKS few-shot, but derived
    from the trained RL policy rather than BKS solutions).
    """
    import json
    from vrptw_env import load_solomon

    rl_path   = os.path.join(cache_dir, f'{instance_name.upper()}_rl_cluster.json')
    data_path = os.path.join(data_dir,  f'{instance_name.upper()}.txt')

    if not (os.path.isfile(rl_path) and os.path.isfile(data_path)):
        return ""

    try:
        with open(rl_path, encoding='utf-8') as f:
            cache = json.load(f)
        inst = load_solomon(data_path)
    except Exception as e:
        print(f'[RL fewshot] failed to load {instance_name}: {e}')
        return ""

    T            = float(inst['T'])
    tw_open_raw  = inst['node_tw_open']
    tw_close_raw = inst['node_tw_close']
    tt_raw       = inst['tt']

    # Pick a cluster with 4-8 nodes; preserve RL visit order (dict insertion order)
    chosen: list[int] | None = None
    for cluster_key in sorted(cache.keys(), key=int):
        nodes = [int(n) for n in cache[cluster_key].keys()]
        if 4 <= len(nodes) <= 8:
            chosen = nodes
            break
    if chosen is None:
        return ""

    slacks = {
        c: round(float(tw_close_raw[c - 1]) * T
                 - max(float(tt_raw[0, c]) * T, float(tw_open_raw[c - 1]) * T), 1)
        for c in chosen
    }

    hdr  = f"  {'ID':>4}  {'TW_open':>8}  {'TW_close':>9}  {'TW_slack':>9}  {'dist_depot':>10}"
    sep  = "  " + "-" * (len(hdr) - 2)
    rows = [hdr, sep]
    for c in sorted(chosen, key=lambda x: float(tw_close_raw[x - 1]) * T):
        tw_o = float(tw_open_raw[c - 1]) * T
        tw_c = float(tw_close_raw[c - 1]) * T
        dist = float(tt_raw[0, c])        * T
        rows.append(f"  {c:>4}  {tw_o:>8.1f}  {tw_c:>9.1f}  {slacks[c]:>9.1f}  {dist:>10.1f}")

    reason = ("RL-optimized sequence: tight-TW nodes served first, "
              "then remaining customers ordered by TW deadline and proximity")
    output_json = (
        '{"ranking": {"1": ' + str(chosen) + '}, '
        '"reason": "' + reason + '"}'
    )

    block = (
        f"### Example 1 — R-type (random layout, tight TW windows) — RL-derived\n"
        f"Input (one vehicle's assigned customers):\n" + "\n".join(rows) + "\n"
        f"Output:\n{output_json}"
    )

    return (
        "## Examples\n"
        "(RL-optimized ranking from policy trained on similar R-type instances — "
        "study before ranking your instance below.)\n"
        "Ranking reflects ALL factors: TW urgency, geographic position, travel time.\n"
        "\n"
        + block
    )


def build_cluster_confidence_prompt(
    inst: dict,
    cluster_nodes: List[int],
    cluster_idx: int,
    total_clusters: int,
    ont_ctx: dict,
    use_cot: bool = True,
    use_ontology: bool = True,
) -> str:
    """Build prompt for ranking customers within vehicle cluster_idx's soft cluster."""
    ont   = SoftClusterOntology(inst)
    T     = float(inst['T'])
    cap   = float(inst['vehicle_capacity'])
    top_k = len(cluster_nodes)

    table = _customer_table(ont, cluster_nodes, ont_ctx, use_ontology, show_demand=False)

    # Disruption section — RAIN and ACCIDENT events affecting this cluster's nodes
    raw_events    = inst.get('preset_events', [])
    stops_by_rain = ont_ctx.get("stops_by_rain", {})
    stops_by_acc  = ont_ctx.get("stops_by_acc",  {})
    acc_events    = ont_ctx.get("accident_events", [])
    cluster_set   = set(cluster_nodes)

    rain_lines = []
    for i, ev in enumerate(e for e in raw_events if e['type'] == 'RAIN'):
        in_cluster = [c for c in stops_by_rain.get(i, []) if c in cluster_set]
        if in_cluster:
            t_s = ev.get('trigger_time', 0)
            t_e = t_s + ev.get('duration', 0)
            mm  = ev.get('rainfall_mm', 10)
            rain_lines.append(
                f"  Rain_{i}: {mm:.0f}mm/h  t={t_s:.0f}-{t_e:.0f}  "
                f"cluster stops affected={in_cluster}"
            )

    acc_lines = []
    for i, ev in enumerate(acc_events):
        in_cluster = [c for c in stops_by_acc.get(i, []) if c in cluster_set]
        if in_cluster:
            sev  = ev.get('severity', 'high')
            t_s  = ev.get('t_start', 0)
            t_e  = ev.get('t_end',   0)
            acc_lines.append(
                f"  Acc_{i}: {_sev_display(sev)}  "
                f"t={t_s:.0f}-{t_e:.0f}  cluster stops on accident edge={in_cluster}"
            )

    disruption_section = ""
    parts = []
    if rain_lines:
        parts.append(
            "### Rain\n" + "\n".join(rain_lines) +
            "\nRain increases travel time on affected road segments. "
            "Visit rain-affected stops before rain starts or after it ends if TW allows."
        )
    if acc_lines:
        parts.append(
            "### Accident\n" + "\n".join(acc_lines) +
            "\nTravel time between the two accident-edge nodes is multiplied during the event window. "
            "Note: TW_slack shown is based on normal (pre-event) travel times — "
            "actual slack for stops adjacent to the accident edge may be smaller. "
            "Strategy: visit both accident-edge nodes before the accident starts, "
            "or after it ends, or separate them so only one falls inside the event window."
        )
    if parts:
        disruption_section = "\n## Disruption Events (affecting this vehicle's cluster)\n" + "\n\n".join(parts) + "\n"

    ont_section = ""
    if use_ontology:
        # TW feasibility: only flag nodes where deadline is already missed from depot
        tw_closed = [c for c in ont_ctx.get("tw_closed", []) if c in cluster_set]
        if tw_closed:
            ont_section += f"\n[Ontology] TW_ClosedStop (deadline already missed from depot): {tw_closed}"

        overdue_in_cluster = [c for c in ont_ctx.get("overdue", []) if c in cluster_set]
        if overdue_in_cluster:
            ont_section += f"\n[Ontology] OverdueStop in this cluster: {overdue_in_cluster}"

        # Cross-episode violation rates (raw, no threshold)
        hist = {c: r for c, r in ont_ctx.get("cross_ep_stats", {}).items()
                if c in cluster_set and r.get('total', 0) > 0}
        if hist:
            lines = []
            for c, r in sorted(hist.items()):
                late_r    = r.get('late_rate', 0.0)
                unserved_r = r.get('unserved_rate', 0.0)
                if late_r > 0 or unserved_r > 0:
                    lines.append(f"  node {c}: late={late_r:.0%}  unserved={unserved_r:.0%}"
                                 f"  (n={r['total']})")
            if lines:
                ont_section += "\n[History] Per-customer violation rates from training:\n" + "\n".join(lines)

    prefix = "" if use_cot else "/no_think\n"

    has_events = bool(rain_lines or acc_lines)
    ids_example = ", ".join(str(c) for c in cluster_nodes[:3])
    if use_cot:
        phase1 = (
            "## Task\n"
            f"Rank the {top_k} customers assigned to vehicle {cluster_idx + 1} by visit priority (highest priority first).\n\n"
            "**Phase 1 — Priority Analysis:**\n"
            "Reason about each stop's visit order considering TW_slack, wait time, geographic position, and service duration.\n"
            "Small or negative TW_slack → rank earlier. Large wait with tight TW_slack → also rank earlier.\n"
            "Keep reasoning brief — one short line per stop.\n"
        )
        if has_events:
            phase2 = (
                "\n**Phase 2 — Event Adjustment:**\n"
                "Re-examine the ranking in light of disruption events:\n"
            )
            if rain_lines:
                phase2 += "- Rain-affected stops: consider whether the rain window overlaps with the likely visit time.\n"
            if acc_lines:
                phase2 += "- Accident-edge stops: consider whether traversal during the event window makes the deadline infeasible.\n"
            phase2 += "Adjust the ranking if needed and output the final order.\n"
        else:
            phase2 = ""
        task_section = (
            phase1 + phase2 +
            f"\nReason through each stop, then output JSON with ALL {top_k} customers ranked:\n"
            f'{{"ranking": [{ids_example}, ...], "reason": "one-line summary"}}'
        )
    else:
        task_section = (
            "## Task\n"
            f"Rank the {top_k} customers assigned to vehicle {cluster_idx + 1} by visit priority (highest priority first).\n"
            "Consider TW_slack, wait time, geographic position, and any disruption events.\n"
            f"Include ALL {top_k} customers. Output ONLY valid JSON:\n"
            f'{{"ranking": [{ids_example}, ...], "reason": "one-line explanation"}}'
        )

    objective = _objective_str(reward_config)
    return f"""{prefix}## Role
You are an expert logistics planner managing a VRPTW delivery operation.

## Problem
- Time horizon: {T:.0f} min
- Travel time between any two points = Euclidean distance (speed = 1 unit/min)
- Vehicle departs from and returns to the depot (node 0)
- Vehicle {cluster_idx + 1} of {total_clusters} is assigned the {top_k} customers below

## Column Definitions
- **dist_depot**: Euclidean travel time from depot. Same formula applies between any two customers.
- **TW_open**: time when the service window opens. Arriving early means waiting until TW_open before service begins.
- **TW_close**: deadline by which the vehicle must ARRIVE. Arriving after TW_close is a time window violation.
- **svc**: service duration (min). Visit sequence: arrive → wait if early → serve svc min → depart.
  Departure time = max(arrival, TW_open) + svc. High svc delays all subsequent stops.
- **TW_slack** = TW_close − max(dist_depot, TW_open) under normal conditions. For ACC-affected stops, recomputed as TW_close − max(tt[depot→partner] + tt[partner→stop]×mult, TW_open) — reflecting worst-case slack when the vehicle must traverse the accident edge.
  Negative → deadline impossible (TW_ClosedStop).  Small → urgent.  Large → flexible.
- **wait** = max(0, TW_open − dist_depot): idle time if visited directly from depot at t=0.
  Large wait means the window opens much later — but if TW_slack is also tight, early commitment is still needed.
- **concept**: ontology label — e.g. OverdueStop (TW missed en route), PrimaryClusterStop (assigned to this vehicle)

## Customers Assigned to Vehicle {cluster_idx + 1}
{table}
{ont_section}
{disruption_section}
## Objective
{objective}

{task_section}
""".strip()


def parse_cluster_confidence(
    response: str,
    cluster_nodes: List[int],
) -> Dict[int, float]:
    """Parse per-cluster LLM ranking response → {node: score ∈ [0,1]}.

    Expects {"ranking": [id1, id2, ...], ...}.  Position 0 → 0.90, last → 0.10
    (uniform linear).  Falls back to legacy {"confidence": {id: score}} format.
    """
    import re, json
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return {}

    valid_set = set(cluster_nodes)
    scores: Dict[int, float] = {}

    ranked = data.get("ranking", [])
    if isinstance(ranked, list) and ranked:
        valid_ranked = []
        for x in ranked:
            try:
                node = int(x)
                if node in valid_set:
                    valid_ranked.append(node)
            except (ValueError, TypeError):
                continue
        n = len(valid_ranked)
        for i, node in enumerate(valid_ranked):
            scores[node] = round(0.9 - i * 0.8 / max(n - 1, 1), 4) if n > 1 else 0.9
        return scores

    # Legacy fallback: {"confidence": {id: score}}
    raw = data.get("confidence", {})
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                node = int(k)
                if node in valid_set:
                    scores[node] = max(0.0, min(1.0, float(v)))
            except (ValueError, TypeError):
                continue
    return scores


def get_cluster_confidence(
    inst: dict,
    cluster_nodes: List[int],
    cluster_idx: int,
    total_clusters: int,
    model: str = DEFAULT_MODEL,
    use_cot: bool = True,
    use_ontology: bool = True,
    cur_time: float = 0.0,
) -> Dict[int, float]:
    """Score customers within a cluster via LLM; return confidence ∈ [0, 1] per node."""
    if not cluster_nodes:
        return {}

    ont     = SoftClusterOntology(inst)
    ont_ctx = ont.get_cluster_context(cluster_nodes, cluster_idx, cur_time=cur_time) if use_ontology else {}

    prompt = build_cluster_confidence_prompt(
        inst, cluster_nodes, cluster_idx, total_clusters, ont_ctx, use_cot, use_ontology
    )
    resp   = query_llm(prompt, model, think=use_cot)
    scores = parse_cluster_confidence(resp, cluster_nodes)

    if not scores or max(scores.values()) == 0.0:
        print(f"  [LLM:cluster{cluster_idx}] FAILED -- empty/zero response (no fallback)")
        print(f"  [LLM:debug] response: {repr(resp[:300])}", flush=True)
        raise RuntimeError(f"LLM confidence scoring failed for cluster {cluster_idx}")

    top = sorted(scores.items(), key=lambda x: -x[1])
    print(f"  [LLM:cluster{cluster_idx}] "
          f"conf={[(n, f'{s:.2f}') for n,s in top[:8]]}"
          f"{'...' if len(top) > 8 else ''}")
    for node in cluster_nodes:
        scores.setdefault(node, 0.0)
    return scores


def _customer_table_clustered(
    ont: SoftClusterOntology,
    clusters: List[List[int]],
    ont_ctx_list: List[dict],
    use_ontology: bool = True,
) -> str:
    """Customer table for all clusters, sorted by vehicle then TW_slack within each."""
    # Aggregate concepts and event info across all clusters
    all_concepts: dict = {}
    stops_by_rain_all: dict = {}
    stops_by_acc_all:  dict = {}
    rain_evs: list = []
    acc_evs:  list = []
    for k, (cluster_nodes, ont_ctx) in enumerate(zip(clusters, ont_ctx_list)):
        all_concepts.update(ont_ctx.get("concepts", {}))
        for i, stops in ont_ctx.get("stops_by_rain", {}).items():
            stops_by_rain_all.setdefault(i, set()).update(stops)
        for i, stops in ont_ctx.get("stops_by_acc", {}).items():
            stops_by_acc_all.setdefault(i, set()).update(stops)
        if not rain_evs:
            rain_evs = ont_ctx.get("rain_events", [])
        if not acc_evs:
            acc_evs  = ont_ctx.get("accident_events", [])

    # For ACC-affected stops, recompute TW_slack via the accident edge
    all_node_set = {c for nodes in clusters for c in nodes}
    acc_slack_override: dict = {}
    for ev in acc_evs:
        affected = ev.get('affected_nodes', [])
        if len(affected) < 2:
            continue
        mult = _sev_to_mult(ev.get('severity'))
        for c, partner in [(affected[0], affected[1]), (affected[1], affected[0])]:
            if c in all_node_set and c not in acc_slack_override:
                t_via = float(ont.tt[0, partner]) + float(ont.tt[partner, c]) * mult
                acc_slack_override[c] = float(ont.tw_close[c]) - max(t_via, float(ont.tw_open[c]))

    hdr = (f"{'veh':>4}  {'ID':>4}  {'x':>6}  {'y':>6}  {'TW_open':>8}  {'TW_close':>9}"
           f"  {'svc':>5}  {'dist_depot':>10}  {'TW_slack':>9}  {'wait':>6}")
    if use_ontology:
        hdr += f"  {'concept':<22}  events"
    rows = [hdr, "-" * (len(hdr) + 2)]

    for k, cluster_nodes in enumerate(clusters):
        for c in sorted(cluster_nodes):
            x     = float(ont.coords[c, 0])
            y     = float(ont.coords[c, 1])
            tw_o  = float(ont.tw_open[c])
            tw_c  = float(ont.tw_close[c])
            svc   = float(ont.service[c])
            dist  = float(ont.tt[0, c])
            slack = acc_slack_override.get(c, tw_c - max(dist, tw_o))
            wait  = max(0.0, tw_o - dist)
            row   = (f"{k+1:>4}  {c:>4}  {x:>6.1f}  {y:>6.1f}  {tw_o:>8.0f}  {tw_c:>9.0f}"
                     f"  {svc:>5.0f}  {dist:>10.1f}  {slack:>9.0f}  {wait:>6.0f}")
            if use_ontology:
                lbls = all_concepts.get(c, {"NormalStop"})
                routing_lbls = {l for l in lbls
                                if not l.startswith("Rain_") and not l.startswith("Acc_")}
                concept_str = ",".join(sorted(routing_lbls - {"NormalStop"}) or {"NormalStop"})
                ev_tags = []
                for i, ev in enumerate(rain_evs):
                    if c in stops_by_rain_all.get(i, set()):
                        ev_tags.append(f"Rain_{i}({ev.get('rainfall_mm',0):.0f}mm,"
                                       f"t={ev.get('t_start',0):.0f}-{ev.get('t_end',0):.0f})")
                for i, ev in enumerate(acc_evs):
                    if c in stops_by_acc_all.get(i, set()):
                        ev_tags.append(f"Acc_{i}({ev.get('severity','')},t="
                                       f"{ev.get('t_start',0):.0f}-{ev.get('t_end',0):.0f})")
                row += f"  {concept_str:<22}  {', '.join(ev_tags) if ev_tags else '-'}"
            rows.append(row)

    return "\n".join(rows)



def build_experience_examples(
    inst: dict,
    clusters: List[List[int]],
    cluster_conf_tensor,       # torch.Tensor (K, N+1) or None (unused, kept for API compat)
    scenario: str,
    base_scenario: str = None,
) -> str:
    """Build few-shot experience section from concrete best/worst episode examples.

    Shows top-BEST_K and bottom-WORST_K actual rollouts so the LLM can reason
    about what orderings and TW decisions led to good vs bad outcomes.
    """
    from SoftClusterOntology import episode_tracker

    best  = episode_tracker.best_episodes(scenario)
    worst = episode_tracker.worst_episodes(scenario)

    if not best and not worst:
        return ""

    # node set for this instance (used to filter route display)
    all_cluster_nodes: set[int] = set()
    for nodes in clusters:
        all_cluster_nodes.update(nodes)

    def fmt_route(route: list[int]) -> str:
        return "→".join(str(n) for n in route if n != 0)

    def fmt_episode(ep: dict, label: str) -> str:
        lines = [
            f"[{label} — ep#{ep['ep']}, reward={ep['reward']:.3f},"
            f" Lc={ep['Lc']}, Lt={ep['Lt']:.1f}]"
        ]
        for v_idx, route in enumerate(ep['routes'], 1):
            route_nodes = [n for n in route if n != 0]
            if not route_nodes:
                continue
            route_str = fmt_route(route)
            late_in_route = {n: ep['late_times'][n] for n in route_nodes
                             if n in ep['late_times']}
            if late_in_route:
                late_str = "  late: " + ", ".join(
                    f"{n}+{t:.1f}min" for n, t in sorted(late_in_route.items())
                )
            else:
                late_str = ""
            lines.append(f"  V{v_idx}: {route_str}{late_str}")
        return "\n".join(lines)

    sections = ["## Experience from Past Episodes\n"
                "(Concrete rollout examples — best = lowest penalty, worst = highest penalty)\n"]

    for ep in best:
        sections.append(fmt_episode(ep, "Best"))

    for ep in worst:
        sections.append(fmt_episode(ep, "Worst"))

    if base_scenario:
        base_best = episode_tracker.best_episodes(base_scenario)
        if base_best:
            base_ep = base_best[0]
            sections.append(
                f"[Base scenario best — ep#{base_ep['ep']},"
                f" reward={base_ep['reward']:.3f}, Lc={base_ep['Lc']}, Lt={base_ep['Lt']:.1f}]"
                f"  (event-free reference)"
            )

    return "\n\n".join(sections)


def build_all_clusters_prompt(
    inst: dict,
    clusters: List[List[int]],
    ont_ctx_list: List[dict],
    use_cot: bool = True,
    use_ontology: bool = True,
    experience_section: str = "",
    reward_config: str = 'F',
    bks_fewshot: str = "",
) -> str:
    """Single prompt covering all K clusters; LLM returns per-cluster rankings."""
    ont = SoftClusterOntology(inst)
    T   = float(inst['T'])
    K   = len(clusters)
    prefix = "" if use_cot else "/no_think\n"

    table = _customer_table_clustered(ont, clusters, ont_ctx_list, use_ontology)

    # Aggregate ontology section
    ont_section = ""
    if use_ontology:
        all_tw_closed: list = []
        all_overdue:   list = []
        hist_lines:    list = []
        for k, (cluster_nodes, ont_ctx) in enumerate(zip(clusters, ont_ctx_list)):
            cluster_set = set(cluster_nodes)
            all_tw_closed += [c for c in ont_ctx.get("tw_closed", []) if c in cluster_set]
            all_overdue   += [c for c in ont_ctx.get("overdue",   []) if c in cluster_set]
            for c, r in sorted(ont_ctx.get("cross_ep_stats", {}).items()):
                if c not in cluster_set or r.get('total', 0) == 0:
                    continue
                late_r     = r.get('late_rate', 0.0)
                unserved_r = r.get('unserved_rate', 0.0)
                if late_r > 0 or unserved_r > 0:
                    hist_lines.append(
                        f"  node {c} (veh {k+1}): late={late_r:.0%}  "
                        f"unserved={unserved_r:.0%}  (n={r['total']})")
        if all_tw_closed:
            ont_section += f"\n[Ontology] TW_ClosedStop (deadline already missed from depot): {sorted(all_tw_closed)}"
        if all_overdue:
            ont_section += f"\n[Ontology] OverdueStop: {sorted(all_overdue)}"
        # hist_lines skipped when experience_section is present — per-node stats are there instead
        if hist_lines and not experience_section:
            ont_section += ("\n[History] Per-customer violation rates from training:\n"
                            + "\n".join(hist_lines))

    # Disruption section — RAIN and ACCIDENT
    raw_events = inst.get('preset_events', [])

    # Aggregate stops_by_rain and stops_by_acc across all clusters
    stops_by_rain_all: dict = {}
    stops_by_acc_all:  dict = {}
    acc_events_global: list = []
    for ont_ctx in ont_ctx_list:
        for i, stops in ont_ctx.get("stops_by_rain", {}).items():
            stops_by_rain_all.setdefault(i, set()).update(stops)
        for i, stops in ont_ctx.get("stops_by_acc", {}).items():
            stops_by_acc_all.setdefault(i, set()).update(stops)
        if not acc_events_global:
            acc_events_global = ont_ctx.get("accident_events", [])

    rain_lines = []
    for i, ev in enumerate(e for e in raw_events if e['type'] == 'RAIN'):
        affected = sorted(stops_by_rain_all.get(i, set()))
        if affected:
            t_s = ev.get('trigger_time', 0)
            t_e = t_s + ev.get('duration', 0)
            rain_lines.append(
                f"  Rain_{i}: {ev.get('rainfall_mm', 10):.0f}mm/h  "
                f"t={t_s:.0f}-{t_e:.0f}  affected stops={affected}")

    acc_lines = []
    for i, ev in enumerate(acc_events_global):
        affected = sorted(stops_by_acc_all.get(i, set()))
        if affected:
            sev  = ev.get('severity', 'high')
            t_s  = ev.get('t_start', 0)
            t_e  = ev.get('t_end',   0)
            acc_lines.append(
                f"  Acc_{i}: {_sev_display(sev)}  "
                f"t={t_s:.0f}-{t_e:.0f}  stops on accident edge={affected}")

    disruption_section = ""
    d_parts = []
    if rain_lines:
        d_parts.append(
            "### Rain\n" + "\n".join(rain_lines) +
            "\nRain increases travel time on affected road segments. "
            "Visit rain-affected stops before rain starts or after it ends if TW allows."
        )
    if acc_lines:
        d_parts.append(
            "### Accident\n" + "\n".join(acc_lines) +
            "\nTravel time between the two accident-edge nodes is multiplied during the event window. "
            "Note: TW_slack shown is based on normal (pre-event) travel times -- "
            "actual slack for stops adjacent to the accident edge may be smaller. "
            "Strategy: visit both accident-edge nodes before the accident starts, "
            "or after it ends, or separate them so only one falls inside the event window."
        )
    if d_parts:
        disruption_section = "\n## Disruption Events\n" + "\n\n".join(d_parts) + "\n"

    example_parts = []
    for k in range(min(K, 2)):
        ids = ", ".join(str(n) for n in clusters[k][:3])
        example_parts.append(f'"{k+1}": [{ids}, ...]')
    if K > 2:
        example_parts.append(f'..., "{K}": [...]')
    example_ranking = ", ".join(example_parts)

    has_events = bool(rain_lines or acc_lines)
    has_experience = bool(experience_section)
    if use_cot:
        exp_guidance = ""
        if has_experience:
            exp_guidance = (
                "Use the Experience section to inform ranking order:\n"
                "  - 'score' = prior LLM confidence; 'late' = empirical late rate from RL training.\n"
                "  - If 'late' is high: rank this customer earlier (higher priority).\n"
                "  - If 'late' is low: current priority is working — keep or rank lower.\n"
            )
            if has_events:
                exp_guidance += (
                    "  - 'base_late(Δ)' shows the base-scenario late rate and the event-induced change.\n"
                    "    A large positive Δ means this event specifically hurts this node — rank it higher.\n"
                    "    A near-zero Δ means the event does not affect this node — keep the base ranking.\n"
                )
        phase1 = (
            "**Phase 1 — Priority Analysis (per vehicle):**\n"
            "Rank each vehicle's customers by visit priority (highest priority first).\n"
            "A customer ranked earlier gets a stronger bias toward being visited sooner by the RL agent.\n"
            "Base ranking on TW urgency: small or negative TW_slack → rank earlier; large TW_slack with large wait → rank later.\n"
            "Also factor in geographic position and service duration.\n"
            + exp_guidance +
            "Keep reasoning brief — one short line per stop.\n"
        )
        if has_events:
            phase2 = (
                "\n**Phase 2 — Event Adjustment:**\n"
                "Re-examine rankings in light of disruption events:\n"
            )
            if rain_lines:
                phase2 += "- Rain-affected stops: consider whether the rain window overlaps with the likely visit time.\n"
            if acc_lines:
                phase2 += "- Accident-edge stops: consider whether traversal during the event makes the deadline infeasible.\n"
            phase2 += "Adjust rankings if needed and output the final order.\n"
        else:
            phase2 = ""
        task_section = (
            "## Task\n"
            f"For each vehicle (1-{K}), rank its assigned customers by visit priority (highest priority first).\n\n"
            + phase1 + phase2 +
            f"\nReason through each vehicle's stops, then output JSON (include ALL vehicles, ALL customers):\n"
            f'{{"ranking": {{{example_ranking}}}, "reason": "one-line summary"}}'
        )
    else:
        exp_hint = (
            " Use 'late' rates from the Experience section to inform ranking: high late rate → rank earlier."
            if has_experience else ""
        )
        task_section = (
            "## Task\n"
            f"For each vehicle (1-{K}), rank its assigned customers by visit priority (highest priority first).\n"
            "Base ranking on TW_slack urgency, wait time, and geographic position."
            + exp_hint + "\n"
            f"Include ALL customers for ALL vehicles. Output ONLY valid JSON:\n"
            f'{{"ranking": {{{example_ranking}}}, "reason": "one-line explanation"}}'
        )

    objective = _objective_str(reward_config)
    fewshot_block = f"\n{bks_fewshot}\n" if bks_fewshot else ""
    return f"""{prefix}## Role
You are an expert logistics planner managing a VRPTW delivery operation.

## Problem
- Time horizon: {T:.0f} min
- Travel time between any two points = Euclidean distance (speed = 1 unit/min)
- Vehicle departs from and returns to the depot (node 0)
- {K} vehicles are deployed; each is pre-assigned a cluster of customers (see "veh" column)

## Column Definitions
- **veh**: vehicle index (1-{K}) assigned to serve this customer
- **dist_depot**: Euclidean travel time from depot. Same formula applies between any two customers.
- **TW_open**: time when the service window opens. Arriving early means waiting until TW_open before service begins.
- **TW_close**: deadline by which the vehicle must ARRIVE. Arriving after TW_close is a time window violation.
- **svc**: service duration (min). Visit sequence: arrive -> wait if early -> serve svc min -> depart.
  Departure time = max(arrival, TW_open) + svc. High svc delays all subsequent stops.
- **TW_slack** = TW_close - max(dist_depot, TW_open) under normal conditions. For ACC-affected stops, recomputed as TW_close - max(tt[depot->partner] + tt[partner->stop]*mult, TW_open) — worst-case slack when the vehicle must traverse the accident edge.
  Negative -> deadline impossible (TW_ClosedStop).  Small -> urgent.  Large -> flexible.
- **wait** = max(0, TW_open - dist_depot): idle time if visited directly from depot at t=0.
  Large wait means the window opens much later -- but if TW_slack is also tight, early commitment is still needed.
- **concept**: ontology label -- e.g. OverdueStop (TW missed en route), PrimaryClusterStop (assigned to this vehicle)
{fewshot_block}
## Instance to Solve
{table}
{ont_section}
{disruption_section}
## Objective
{objective}

{experience_section}
{task_section}
""".strip()


def parse_all_clusters_confidence(
    response: str,
    clusters: List[List[int]],
) -> Dict[int, Dict[int, float]]:
    """Parse combined LLM ranking response → {cluster_idx (0-based): {node: score}}.

    Expects {"ranking": {"1": [id1, id2, ...], "2": [...]}, ...}.
    Position 0 → 0.90, last → 0.10 (uniform linear conversion).
    Falls back to legacy {"confidence": {"1": {id: score}}} format.
    """
    import re, json
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        return {}
    json_str = re.sub(r'//[^\n]*', '', m.group())
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {}

    result: Dict[int, Dict[int, float]] = {}

    rank_raw = data.get("ranking", {})
    if isinstance(rank_raw, dict) and rank_raw:
        for k, cluster_nodes in enumerate(clusters):
            valid_set = set(cluster_nodes)
            ranked = rank_raw.get(str(k + 1), [])
            if not isinstance(ranked, list):
                continue
            valid_ranked: List[int] = []
            for x in ranked:
                try:
                    node = int(x)
                    if node in valid_set:
                        valid_ranked.append(node)
                except (ValueError, TypeError):
                    continue
            if not valid_ranked:
                continue
            n = len(valid_ranked)
            result[k] = {
                node: round(0.9 - i * 0.8 / max(n - 1, 1), 4) if n > 1 else 0.9
                for i, node in enumerate(valid_ranked)
            }
        return result

    # Legacy fallback: {"confidence": {"1": {id: score}}}
    conf_raw = data.get("confidence", {})
    if not isinstance(conf_raw, dict):
        return result
    for k, cluster_nodes in enumerate(clusters):
        valid_set = set(cluster_nodes)
        cluster_conf = conf_raw.get(str(k + 1), {})
        if not isinstance(cluster_conf, dict):
            continue
        scores: Dict[int, float] = {}
        for node_str, score in cluster_conf.items():
            try:
                node = int(node_str)
                if node in valid_set:
                    scores[node] = max(0.0, min(1.0, float(score)))
            except (ValueError, TypeError):
                continue
        if scores:
            result[k] = scores
    return result


def get_all_clusters_confidence(
    inst: dict,
    clusters: List[List[int]],
    model: str = DEFAULT_MODEL,
    use_cot: bool = True,
    use_ontology: bool = True,
    cur_time: float = 0.0,
    prompt_save_path: str = None,
    experience_section: str = "",
    reward_config: str = 'F',
    visited: Optional[Set[int]] = None,
    bks_fewshot: str = "",
) -> Dict[int, float]:
    """Single LLM call for all clusters; returns flat {node: score} dict.

    If visited is provided, only remaining (unvisited) nodes are included in
    the prompt and scored. TW slack is recalculated relative to cur_time.
    """
    if not clusters:
        return {}

    visited = visited or set()
    # When visited is non-empty (accident re-scoring at test time), filter each
    # cluster to remaining nodes only so the LLM reasons about the actual state.
    if visited:
        prompt_clusters = [[n for n in c if n not in visited] for c in clusters]
        prompt_clusters = [c for c in prompt_clusters if c]  # drop fully-visited clusters
        if not prompt_clusters:
            return {}
    else:
        prompt_clusters = clusters

    ont = SoftClusterOntology(inst)
    ont_ctx_list = [
        ont.get_cluster_context(cluster_nodes, k, cur_time=cur_time, visited=visited) if use_ontology else {}
        for k, cluster_nodes in enumerate(prompt_clusters)
    ]

    prompt = build_all_clusters_prompt(inst, prompt_clusters, ont_ctx_list, use_cot, use_ontology,
                                       experience_section=experience_section,
                                       reward_config=reward_config,
                                       bks_fewshot=bks_fewshot)
    if prompt_save_path:
        with open(prompt_save_path, 'w', encoding='utf-8') as _f:
            _f.write(prompt)
    resp   = query_llm(prompt, model, think=use_cot)
    conf_by_cluster = parse_all_clusters_confidence(resp, prompt_clusters)

    flat_scores: Dict[int, float] = {}
    failed_clusters: List[int] = []
    for k, cluster_nodes in enumerate(prompt_clusters):
        cluster_conf = conf_by_cluster.get(k)
        if not cluster_conf:
            failed_clusters.append(k)
        else:
            top = sorted(cluster_conf.items(), key=lambda x: -x[1])
            print(f"  [LLM:cluster{k}] conf={[(n, f'{s:.2f}') for n,s in top[:8]]}"
                  f"{'...' if len(top) > 8 else ''}")
            flat_scores.update(cluster_conf)
            for node in cluster_nodes:
                flat_scores.setdefault(node, 0.0)

    if failed_clusters:
        print(f"  [LLM] combined parse FAILED for {len(failed_clusters)}/{len(prompt_clusters)} clusters")
        print(f"  [LLM:debug] response preview: {repr(resp[:500])}", flush=True)
        for k in failed_clusters:
            print(f"  [LLM:cluster{k}] FAILED -- no confidence assigned")
        raise RuntimeError(
            f"LLM combined call failed for {len(failed_clusters)}/{len(prompt_clusters)} clusters"
        )

    return flat_scores


def refresh_cluster_confidence(
    inst: dict,
    cluster_nodes: List[int],
    cluster_idx: int,
    total_clusters: int,
    model: str = DEFAULT_MODEL,
    use_cot: bool = True,
    use_ontology: bool = True,
    cur_time: float = 0.0,
) -> Dict[int, float]:
    """Re-score ALL nodes in cluster after accident; caller applies visited mask."""
    return get_cluster_confidence(
        inst, cluster_nodes, cluster_idx, total_clusters,
        model=model, use_cot=use_cot, use_ontology=use_ontology,
        cur_time=cur_time,
    )
