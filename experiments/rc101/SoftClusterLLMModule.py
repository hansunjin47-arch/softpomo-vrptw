"""
SoftClusterLLMModule.py — LLM cluster-confidence functions for Soft-Clustering POMO.

Each vehicle is assigned a soft cluster (from RL-derived routes, or K-means
geographic grouping in the older kim2006 pipeline). The LLM makes two SEQUENTIAL,
non-contradicting judgments per vehicle (see build_all_clusters_prompt /
build_cluster_confidence_prompt for the exact prompt):
  Step 1 (fit, binary):     does this customer actually belong in this vehicle's
                            group at all? Flagged misfits get NO priority score --
                            fit is a reliable yes/no call, so it gates step 2
                            rather than living on the same continuous scale as it
                            (a single scalar meaning "priority" when positive and
                            "wrong cluster" when negative was tried and rejected --
                            it made zero the ambiguous boundary between "belongs but
                            low urgency" and "borderline misfit", and let the model
                            emit a priority score for a customer it simultaneously
                            flagged as not belonging).
  Step 2 (priority, [0,1]): for customers that passed step 1, visit-priority score
                            used as per-step logit bias throughout that vehicle's
                            route (higher -> visited sooner).
A flagged misfit is assigned the fixed MISFIT_BIAS constant below, not an
LLM-reported magnitude -- the binary fit call is reliable across calls, a
self-reported severity number is not, so calibration stays a system/code
responsibility rather than the model's.

Key difference from original_POMO (VRPTWLLMModule):
  original_POMO: LLM → top-K global start nodes → bias ONLY at depot dispatch moments
  soft_pomo    : LLM → per-cluster confidence   → bias at EVERY routing step per vehicle
"""
from __future__ import annotations

# Fixed logit-bias value assigned to any customer the LLM flags as a cluster
# misfit (see module docstring). Symmetric with the +1.0 ceiling of the
# priority scale, applied via the SAME cluster_conf tensor: -1.0 is the
# strongest "avoid this pick" signal in the same units bias_strength scales,
# no separate hyperparameter needed. Not derived from LLM output.
MISFIT_BIAS = -1.0

import sys
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from typing import Dict, List, Tuple
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
    # E's actual reward has dist_penalty_weight=0 -- D is not penalized at all, only TT
    # (total_travel_time, event-multiplier-affected) via time_penalty; see train_vrptw.py's
    # _REWARD_CONFIGS['E'] comment ("D dropped (TT already includes it, dist_penalty_weight=0
    # avoids double-charging)"). Do not collapse this back to D like F/G below -- that would
    # tell the LLM to optimise a metric the RL agent isn't actually scored on.
    'E': "Minimise in order of priority: (0) leave NO customer unserved -- this is penalized far "
         "more heavily than anything below and dominates every other consideration, (1) Total "
         "lateness Lt, (2) Number of vehicles used K, (3) Total travel time TT (actual time "
         "driven, including any rain/accident slowdown -- this config does not penalize raw "
         "distance at all, so judge cost by TT/travel_time (the per-stop travel_time column in the "
         "table), not raw distance).",
    'F': "Minimise in order of priority: (1) Total lateness Lt, (2) Total travel distance D.",
    'G': "Minimise in order of priority: (1) Number of late customers Lc, (2) Total lateness Lt, (3) Total travel distance D.",
}
_METRIC_DEFS = (
    "- Lt: sum of arrival-time overflows beyond TW_close across all customers (minutes)\n"
    "- D: total Euclidean travel distance across all routes (static, event-independent)\n"
    "- TT: total actual travel time across all routes, INCLUDING rain/accident multiplier "
    "slowdowns (equals D exactly whenever no disruption event is active on the traversed edge)\n"
    "- Lc: number of customers whose arrival time exceeds their TW_close"
)
_DEFAULT_OBJECTIVE = _CONFIG_OBJECTIVES['F']


def _objective_str(reward_config: str | None) -> str:
    obj = _CONFIG_OBJECTIVES.get(reward_config or 'F', _DEFAULT_OBJECTIVE)
    return f"{obj}\n{_METRIC_DEFS}"


# ── Severity helpers ──────────────────────────────────────────────────────────

# vehicles_involved -> English news-bulletin phrase. Recorded at generation time
# in the scenario file (never reverse-mapped from the multiplier), so this can't
# silently desync if the multiplier tiers are retuned later.
_VEHICLES_TEXT: dict[int, str] = {2: "2-car collision", 3: "3-car pile-up", 5: "5-car chain collision"}


def _vehicles_display(vehicles) -> str:
    if vehicles is None:
        return "traffic accident"
    return _VEHICLES_TEXT.get(int(vehicles), f"{int(vehicles)}-car collision")


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
                f"  [Weather Advisory] Rainfall {mm:.0f}mm/h expected t={t_s:.0f}-{t_e:.0f}, "
                f"affecting cluster stops {in_cluster}"
            )

    acc_lines = []
    for i, ev in enumerate(acc_events):
        in_cluster = [c for c in stops_by_acc.get(i, []) if c in cluster_set]
        if in_cluster:
            t_s  = ev.get('t_start', 0)
            t_e  = ev.get('t_end',   0)
            acc_lines.append(
                f"  [Traffic Alert] {_vehicles_display(ev.get('vehicles_involved'))} reported "
                f"t={t_s:.0f}-{t_e:.0f}, affecting cluster stops {in_cluster}"
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
            "\nTravel time on road segments near the listed stops is multiplied during the event window. "
            "Note: TW_slack shown is based on normal (pre-event) travel times — "
            "actual slack for stops near the accident may be smaller. "
            "Strategy: visit affected stops before the accident starts or after it ends if TW allows, "
            "otherwise expect longer travel time when passing through during the window."
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
    fit_gate = (
        "\n## Step 1 — Cluster Fit Check (do this FIRST)\n"
        "Check whether any of this vehicle's assigned customers do NOT actually belong in this "
        "group. Flag a customer as a misfit ONLY for a SPECIFIC, concrete reason: geographically "
        "far from the rest of this group, a time-window conflict with the group's schedule, or a "
        "disruption event making the stop unreachable on this vehicle's plausible path.\n"
        "Most customers should NOT be flagged -- do not invent a misfit just because a customer is "
        "the least urgent among an otherwise fine group. A misfit customer gets NO priority score "
        "in Step 2 below (excluded from it entirely).\n"
    )
    if use_cot:
        phase1 = (
            "## Task\n"
            f"For vehicle {cluster_idx + 1}: (1) flag any misfit customers, (2) score the rest for visit priority.\n\n"
            + fit_gate +
            "\n## Step 2 — Priority Analysis (misfit customers excluded)\n"
            "For every customer NOT flagged as a misfit, score visit priority (0.0-1.0) considering "
            "TW_slack, wait time, geographic position, and service duration.\n"
            "Small or negative TW_slack → higher priority. Large wait with tight TW_slack → also higher priority.\n"
            "Do NOT force a spread across the group -- a well-formed group can legitimately have "
            "every member score high.\n"
            "Keep reasoning brief — one short line per stop.\n"
        )
        if has_events:
            phase2 = (
                "\n## Step 3 — Event Adjustment\n"
                "Re-examine Step 1/2 in light of disruption events:\n"
            )
            if rain_lines:
                phase2 += "- Rain-affected stops: consider whether the rain window overlaps with the likely visit time.\n"
            if acc_lines:
                phase2 += "- Accident-edge stops: consider whether traversal during the event window makes the deadline infeasible.\n"
            phase2 += ("Adjust misfit flags or scores if needed.\n" if use_misfit
                       else "Adjust scores if needed.\n")
        else:
            phase2 = ""
        task_section = (
            phase1 + phase2 +
            f"\nReason through each stop, then output JSON (confidence covers every customer "
            f"EXCEPT those in misfit):\n"
            f'{{"misfit": [id, id, ...], "confidence": {{"{cluster_nodes[0]}": <score>, ...}}, "reason": "one-line summary"}}'
        )
    else:
        task_section = (
            "## Task\n"
            f"For vehicle {cluster_idx + 1}: (1) flag any misfit customers, (2) score the rest for visit priority.\n"
            + fit_gate +
            "For every customer NOT flagged as misfit, score visit priority (0.0-1.0) based on "
            "TW_slack, wait time, geographic position, and any disruption events.\n"
            f"Output ONLY valid JSON (confidence covers every customer EXCEPT those in misfit):\n"
            f'{{"misfit": [id, id, ...], "confidence": {{"{cluster_nodes[0]}": <score>, ...}}, "reason": "one-line explanation"}}'
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
    """Parse per-cluster LLM response → {node: score}.

    Primary format: {"misfit": [id, id, ...], "confidence": {id: score}, ...}.
    `misfit` lists customers judged not to belong in this vehicle's group --
    these get the fixed MISFIT_BIAS, not an LLM-reported magnitude (see this
    module's docstring for why). `confidence` gives an absolute per-customer
    priority score ∈ [0,1] for every OTHER customer, independently judged (not
    a forced relative spread) -- see build_cluster_confidence_prompt's Step 1/
    Step 2 sections.

    Falls back to the older {"ranking": [id1, id2, ...]} format (position 0 ->
    0.90, last -> 0.10, uniform linear) for responses from before this switch
    -- that format has no fit/misfit concept at all.
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

    raw = data.get("confidence", {})
    if isinstance(raw, dict) and raw:
        for k, v in raw.items():
            try:
                node = int(k)
                if node in valid_set:
                    scores[node] = max(0.0, min(1.0, float(v)))
            except (ValueError, TypeError):
                continue

        misfit_raw = data.get("misfit", [])
        if isinstance(misfit_raw, list):
            for x in misfit_raw:
                try:
                    node = int(x)
                    if node in valid_set:
                        scores[node] = MISFIT_BIAS
                except (ValueError, TypeError):
                    continue

        if scores:
            return scores

    # Fallback: {"ranking": [id1, id2, ...]} (older ordinal format)
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
    """Score customers within a cluster via LLM; return confidence ∈ [-1, 1] per node
    (negative = does not belong in this cluster, see build_cluster_confidence_prompt)."""
    if not cluster_nodes:
        return {}

    ont     = SoftClusterOntology(inst)
    ont_ctx = ont.get_cluster_context(cluster_nodes, cluster_idx, cur_time=cur_time) if use_ontology else {}

    prompt = build_cluster_confidence_prompt(
        inst, cluster_nodes, cluster_idx, total_clusters, ont_ctx, use_cot, use_ontology
    )
    resp   = query_llm(prompt, model, think=use_cot)
    scores = parse_cluster_confidence(resp, cluster_nodes)

    if not scores:
        print(f"  [LLM:cluster{cluster_idx}] FAILED -- empty/unparseable response (no fallback)")
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
    stats: Optional[Dict[int, dict]] = None,
    events_visible_until: Optional[float] = None,
    stats_is_estimate: bool = False,
    cluster_source: str = 'rl',
) -> str:
    """Customer table sorted within each vehicle by TW_close, TW_open, then ID.

    stats (optional): {cluster_idx: {late_count, total_late, travel_time,
    node_detail}} -- each vehicle's actual realized performance, rendered as
    an extra line per vehicle alongside its RL visit order (see
    extract_rl_routes.py). travel_time is event-multiplier-affected (what the
    RL reward is actually penalized on under Config E's dist_penalty_weight=0),
    not static geometric distance. node_detail ({node: {slack, travel_time}})
    drives the per-row 'slack' and 'travel_time' columns: slack = TW_close -
    actual arrival under the route as given (positive -> buffer, negative ->
    exactly how late -- this alone tells the LLM which specific customer(s)
    caused lateness, no guessing from a vehicle-level aggregate needed), and
    this row's travel_time is the actual travel time from the preceding stop
    in the given route (per-STOP, not to be confused with the per-vehicle
    travel_time in the '[Actual performance]' line above). Both are ground
    columns, which were computed as if this customer were the vehicle's ONLY
    stop reached directly from the depot and so did not reflect the actual
    multi-stop schedule this customer sits in."""
    # Aggregate event info across all clusters (concept/ontology labels no
    # longer rendered here, see the per-row comment below for why)
    stops_by_rain_all: dict = {}
    stops_by_acc_all:  dict = {}
    rain_evs: list = []
    acc_evs:  list = []
    for k, (cluster_nodes, ont_ctx) in enumerate(zip(clusters, ont_ctx_list)):
        for i, stops in ont_ctx.get("stops_by_rain", {}).items():
            stops_by_rain_all.setdefault(i, set()).update(stops)
        for i, stops in ont_ctx.get("stops_by_acc", {}).items():
            stops_by_acc_all.setdefault(i, set()).update(stops)
        if not rain_evs:
            rain_evs = ont_ctx.get("rain_events", [])
        if not acc_evs:
            acc_evs  = ont_ctx.get("accident_events", [])

    if cluster_source not in ('rl', 'kim'):
        raise ValueError(f'Unknown cluster_source: {cluster_source!r}')
    if cluster_source == 'kim' and (stats is not None or stats_is_estimate):
        raise ValueError('Kim initial prompts cannot include route statistics')
    hdr = (f"{'veh':>4}  {'ID':>4}")
    if cluster_source == 'rl':
        hdr += f"  {'order':>5}"
    hdr += (f"  {'x':>6}  {'y':>6}  {'TW_open':>8}  {'TW_close':>9}"
           f"  {'svc':>5}")
    if cluster_source == 'rl':
        hdr += f"  {'slack':>7}  {'travel_time':>11}"
    if use_ontology:
        hdr += "  events"
    rows = [
        "[Table ordering] We sorted customers within each vehicle by TW_close "
        "ascending for readability (ties: TW_open ascending, then customer ID). "
        "This display order is NOT a recommended visit order or priority ranking. "
        "Do not assign confidence merely by row position or deadline order. "
        "The 'order' column gives each stop's 1-indexed position in THIS vehicle's "
        "[RL visit order] below (so you don't have to cross-reference the arrow "
        "chain against this differently-sorted table by hand) -- it is a label for "
        "where RL currently placed this stop, not a target or a priority ranking "
        "either; a low 'order' number is not automatically deserving of a high "
        "score. Use the separate [RL visit order] as the reference route and "
        "consider time windows, route slack, travel time, service duration and "
        "any visible events together. The slack and travel_time columns still "
        "describe the reference route (or its stated estimate), NOT a replay in "
        "table order.",
        hdr, "-" * (len(hdr) + 2),
    ]
    if cluster_source == 'kim':
        rows[0] = (
            '[Table ordering] Customers are sorted by TW_close ascending '
            '(ties: TW_open ascending, then customer ID) for readability only. '
            'Neither table order nor cluster member order is a recommended visit order '
            'or priority ranking. No executed route or route timing is supplied.')

    for k, cluster_nodes in enumerate(clusters):
        # cluster_nodes is already in the RL-derived visit order (preserved through
        # the {node: score} dict's insertion order at extraction time, through JSON,
        # through _parse_cluster_cache) -- surface it explicitly before the
        # TW-sorted stat rows below, since the sort discards that sequence. A
        # customer that breaks an otherwise-coherent geographic sequence is a
        # stronger misfit signal than the static per-row stats alone.
        if cluster_source == 'kim':
            rows.append(f"  [Cluster members, vehicle {k+1}] "
                        + ", ".join(str(c) for c in cluster_nodes if c != 0))
        else:
            rows.append(f"  [RL visit order, vehicle {k+1}] "
                        + " -> ".join(str(c) for c in cluster_nodes if c != 0))
        vstat = (stats or {}).get(k)
        if vstat is not None:
            # total_late/exact late customers dropped here -- fully derivable from
            # counting negative values in the per-row 'slack' column below, no
            # need to state it twice. late_stops is kept as a fast "is there a
            # problem here at all" flag.
            perf_label = "Estimated performance" if stats_is_estimate else "Actual performance"
            rows.append(
                f"  [{perf_label}, vehicle {k+1}] "
                f"late_stops={vstat.get('late_count', 0)}  "
                f"travel_time={vstat.get('travel_time', 0.0):.1f}"
            )
        node_detail = (vstat or {}).get('node_detail', {})
        # 1-indexed position within THIS vehicle's own RL visit order (depot
        # excluded) -- lets each row carry its route position directly instead
        # of forcing a manual lookup against the separate arrow-chain line above.
        visit_pos = {n: i + 1 for i, n in enumerate(c for c in cluster_nodes if c != 0)}
        for c in sorted(cluster_nodes, key=lambda n: (
                float(ont.tw_close[n]), float(ont.tw_open[n]), n)):
            x     = float(ont.coords[c, 0])
            y     = float(ont.coords[c, 1])
            tw_o  = float(ont.tw_open[c])
            tw_c  = float(ont.tw_close[c])
            svc   = float(ont.service[c])
            # No data and "actually 0.0" must render differently -- a customer
            # missing from node_detail (no stats block, or this vehicle simply
            # has none for it) previously showed as slack=0.0/travel_time=0.0,
            # indistinguishable from "arrived exactly on time, zero travel".
            nd = node_detail.get(str(c))
            if nd is None:
                slack_str, tt_str = f"{'N/A':>7}", f"{'N/A':>11}"
            else:
                slack_str = f"{float(nd.get('slack', 0.0)):>7.1f}"
                tt_str    = f"{float(nd.get('travel_time', 0.0)):>11.1f}"
            row   = f"{k+1:>4}  {c:>4}"
            if cluster_source == 'rl':
                row += f"  {visit_pos.get(c, 0):>5}"
            row  += f"  {x:>6.1f}  {y:>6.1f}  {tw_o:>8.0f}  {tw_c:>9.0f}  {svc:>5.0f}"
            if cluster_source == 'rl':
                row += f"  {slack_str}  {tt_str}"
            if use_ontology:
                # concept (OverdueStop etc.) removed: it's a static rule-based
                # estimate of lateness risk, computed independently of and
                # sometimes inconsistent with the ACTUAL simulated performance
                # now shown above (see [Actual performance]) -- keeping both
                # risked the model treating a heuristic guess as equal evidence
                # to ground truth. Event tags (which stops fall in an active
                # rain/accident zone) are kept -- that's not duplicated elsewhere.
                # rain_evs/acc_evs' t_start is RAW (ontology's own event lists,
                # not normalized by T) -- compare against events_visible_until * T.
                # "now" is events_visible_until*T (same framing as the aggregate
                # Disruption Events section) -- an event whose window has already
                # fully closed by then is stale news and dropped here too, not
                # just future ones, since a per-row tag showing an ended window
                # is just as misleading with no "current time" to compare it to.
                ev_tags = []
                for i, ev in enumerate(rain_evs):
                    if events_visible_until is not None and (
                        ev.get('t_start', 0) > events_visible_until * ont.T
                        or ev.get('t_end', 0) <= events_visible_until * ont.T
                    ):
                        continue  # not yet revealed, or already over, at this point in the timeline
                    if c in stops_by_rain_all.get(i, set()):
                        ev_tags.append(f"Rain_{i}({ev.get('rainfall_mm',0):.0f}mm,"
                                       f"t={ev.get('t_start',0):.0f}-{ev.get('t_end',0):.0f})")
                for i, ev in enumerate(acc_evs):
                    if events_visible_until is not None and (
                        ev.get('t_start', 0) > events_visible_until * ont.T
                        or ev.get('t_end', 0) <= events_visible_until * ont.T
                    ):
                        continue  # not yet revealed, or already over, at this point in the timeline
                    if c in stops_by_acc_all.get(i, set()):
                        ev_tags.append(f"Acc_{i}({_vehicles_display(ev.get('vehicles_involved'))},t="
                                       f"{ev.get('t_start',0):.0f}-{ev.get('t_end',0):.0f})")
                row += f"  {', '.join(ev_tags) if ev_tags else '-'}"
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


def _kim_initial_prompt(T, K, table, disruption_section, objective,
                        fewshot, experience, example_ranking, use_cot, use_misfit,
                        confidence_only=False):
    """Initial grouping context, without assuming an observed route or timing."""
    prefix = '' if use_cot else '/no_think\n'
    fit = (
        'Flag misfits only for concrete structural incompatibility that reordering '
        'cannot resolve. Do not equate low urgency with misfit. Provide misfit_reasons '
        'keyed by customer ID. Exclude flagged customers from confidence.\n'
        if use_misfit else
        'Misfit detection is OFF. Score every assigned customer; do not omit customers.\n')
    schema = ('{"misfit": [], "misfit_reasons": {}, ' if use_misfit else '{')
    schema += f'"confidence": {{{example_ranking}}}, "reason": "brief explanation"}}'
    if confidence_only:
        if use_misfit:
            raise ValueError('confidence_only requires misfit OFF')
        schema = ('Return ONLY a JSON object with sole key "confidence", mapping each '
                  'group ID to its customer/depot score object. No reason or explanation. '
                  'Every required key must occur exactly once. Required keys by group: '
                  + example_ranking)
    examples = f'\n## Few-shot Examples\n{fewshot}\n' if fewshot else ''
    history = (f'\n## Historical Context (not an execution of the current grouping)\n'
               f'{experience}\n' if experience else '')
    return f'''{prefix}## Role
You are an expert logistics planner managing a VRPTW delivery operation.

## Problem
- Time horizon: {T:.0f} min
- Travel time between points = Euclidean distance (speed = 1 unit/min), before event effects.
- Vehicle departs from and returns to depot (node 0).
- {K} customer groups were generated by the Kim clustering algorithm.

## Input Provenance [kim-initial-v1]
These are algorithm-generated groups, NOT verified routing solutions.
Cluster membership lists have no visit-order meaning. No executed RL route,
arrival times, route slack, or measured route performance is supplied for this query.
Do not invent those observations or assume a group is feasible because it was clustered.
Examples and historical context belong to their stated settings, not this execution.

## Column Definitions
- veh: assigned group index (1-{K}).
- TW_open: arriving early requires waiting until this time before service.
- TW_close: arrival deadline; arriving after it incurs lateness.
- svc: service duration. Departure = max(arrival, TW_open) + svc.
- x,y: coordinates used to assess travel between customers.
{examples}
## Instance to Solve
{table}
{disruption_section}
## Objective
{objective}
{history}
## Task
{fit}For each group, assign visit-priority confidence in [0,1]. Higher scores provide
stronger soft preference toward earlier selection; they are not membership probabilities
or guaranteed visit orders. Consider time windows, geography, waiting, service duration
and visible events together. Do not score mechanically by ID, row position or deadline.
Do not force a spread across the group.
Include a separate "0" depot score in [0,1] for every group. Keep it exactly 0.0
unless concrete structural overload that reordering cannot resolve supports a cutoff.
If a cutoff is justified, place it above customers intended for another vehicle and
below those to retain. Do not claim measured overload when no execution data exists.
{'Output ONLY valid JSON; do not print analysis.' if confidence_only else ('Keep the explanation brief, then output valid JSON.' if use_cot else 'Output ONLY valid JSON.')}
Include all {K} groups. Confidence includes every non-misfit customer and each depot.
{schema}
'''.strip()


def build_all_clusters_prompt(
    inst: dict,
    clusters: List[List[int]],
    ont_ctx_list: List[dict],
    use_cot: bool = True,
    use_ontology: bool = True,
    experience_section: str = "",
    reward_config: str = 'F',
    bks_fewshot: str = "",
    stats: Optional[Dict[int, dict]] = None,
    use_misfit: bool = False,
    events_visible_until: Optional[float] = None,
    stats_is_estimate: bool = False,
    cluster_source: str = 'rl',
    confidence_only: bool = False,
) -> str:
    """Single prompt covering all K clusters; LLM returns per-cluster fit+priority.

    cluster_source: 'rl' preserves the route-based prompt; 'kim' describes an
    initial algorithmic grouping and rejects route statistics. Callers must
    explicitly choose the source; missing stats never implies Kim provenance.

    stats (optional): {cluster_idx: {late_count, total_late, travel_time}} --
    each vehicle's actual realized performance, rendered alongside its RL visit
    order in the table (see _customer_table_clustered / get_all_clusters_confidence).

    events_visible_until (optional): if given (normalized, fraction of T -- same
    scale as an accident's fp['t_start']), RAIN/ACCIDENT events whose own start
    time is AFTER this point are omitted from the Disruption Events section --
    used by the accident-wave refresh call so an earlier wave's prompt can't see
    a later, not-yet-triggered accident. None (default) shows every event, which
    is correct for the initial full-instance/full-episode scoring call (there is
    no single "current time" for a whole completed route to be revealed up to).

    stats_is_estimate: when True, `stats` is NOT a replayed route (see
    _apply_accident_to_stats) -- every "[Actual performance]"/"ground truth"/
    "ACTUAL" reference in the table label and column definitions is swapped for
    estimate-appropriate wording instead, so the prompt doesn't simultaneously
    call the same numbers both an estimate and a verified replay.
    """
    ont = SoftClusterOntology(inst)
    T   = float(inst['T'])
    K   = len(clusters)
    perf_label = "Estimated performance" if stats_is_estimate else "Actual performance"
    # Column-definition prose swaps with stats_is_estimate too, instead of just
    # appending a caveat note elsewhere -- the definitions themselves otherwise
    # keep asserting "ground truth"/"ACTUAL" no matter what note precedes them.
    slack_basis = (
        "an ESTIMATE for the route above (BASE-route timing with directly "
        "affected stops' own leg adjusted for this accident, NOT a re-simulated replay)"
        if stats_is_estimate else
        "the CURRENT route above (ground truth from replaying it, not a recomputation)"
    )
    tt_basis = (
        "an ESTIMATE of the travel time from the PRECEDING stop"
        if stats_is_estimate else
        "the ACTUAL travel time from the PRECEDING stop"
    )
    no_data_basis = "estimate" if stats_is_estimate else "ground-truth"
    slack_history_note = (
        "Only directly affected stops' OWN slack is adjusted for this accident -- "
        "stops after one on this vehicle's route are NOT recomputed, so their slack "
        "may still be optimistic"
        if stats_is_estimate else
        "Already reflects every stop visited before this one, cumulative waiting, "
        "and any event slowdown so far"
    )
    prefix = "" if use_cot else "/no_think\n"

    table = _customer_table_clustered(ont, clusters, ont_ctx_list, use_ontology, stats=stats,
                                       events_visible_until=events_visible_until,
                                       stats_is_estimate=stats_is_estimate,
                                       cluster_source=cluster_source)

    # Aggregate ontology section -- intentionally empty now. TW_ClosedStop /
    # OverdueStop (rule-based lateness estimates from depot-direct arrival)
    # and cross_ep_stats (the old "[History]" block, violation rates
    # accumulated across the WHOLE training run epoch 1..500) were both
    # dropped: they're static/stale estimates that can disagree with the
    # single current-best-episode "[Actual performance]" line now shown per
    # vehicle (real simulated ground truth), and showing both risked the
    # model treating a heuristic guess as equally strong evidence. use_ontology
    # still gates the per-stop event tags in _customer_table_clustered.
    ont_section = ""

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

    # "now" for this call IS events_visible_until*T -- modeling each accident-
    # refresh call as news arriving exactly when its own wave's accident
    # starts (see the matching node_mult fix in train_soft_cluster.py). An
    # event whose OWN window has already fully closed by "now" is stale news:
    # showing it (with no "current time" reference for the LLM to compare
    # against) previously left the LLM unable to tell an ended disruption
    # from an ongoing one, since the prompt never states what time "now" is.
    now_t = events_visible_until * T if events_visible_until is not None else None

    rain_lines = []
    for i, ev in enumerate(e for e in raw_events if e['type'] == 'RAIN'):
        t_s = ev.get('trigger_time', 0)
        t_e = t_s + ev.get('duration', 0)
        if now_t is not None and (t_s > now_t or t_e <= now_t):
            continue  # not yet revealed, or already over, at this point in the timeline
        affected = sorted(stops_by_rain_all.get(i, set()))
        if affected:
            rain_lines.append(
                f"  [Weather Advisory] Rainfall {ev.get('rainfall_mm', 10):.0f}mm/h expected "
                f"t={t_s:.0f}-{t_e:.0f}, affecting stops {affected}")

    acc_lines = []
    for i, ev in enumerate(acc_events_global):
        # accident_events' t_start is RAW (matches trigger_time), same as rain's
        # trigger_time above -- both compared against now_t.
        t_s  = ev.get('t_start', 0)
        t_e  = ev.get('t_end',   0)
        if now_t is not None and (t_s > now_t or t_e <= now_t):
            continue  # not yet revealed, or already over, at this point in the timeline
        affected = sorted(stops_by_acc_all.get(i, set()))
        if affected:
            acc_lines.append(
                f"  [Traffic Alert] {_vehicles_display(ev.get('vehicles_involved'))} reported "
                f"t={t_s:.0f}-{t_e:.0f}, affecting stops {affected}")

    disruption_section = ""
    d_parts = []
    now_line = f"Current time: t={now_t:.0f} min.\n" if now_t is not None else ""
    if rain_lines:
        d_parts.append(
            "### Rain\n" + "\n".join(rain_lines) +
            "\nRain increases travel time on affected road segments. The exact multiplier is NOT "
            "given to you -- only rainfall intensity (mm/h) -- so you must ESTIMATE the actual "
            "slowdown (heavier rainfall implies a larger one), not treat any specific number as known. "
            "Visit rain-affected stops before rain starts or after it ends if TW allows."
        )
    if acc_lines:
        d_parts.append(
            "### Accident\n" + "\n".join(acc_lines) +
            "\nTravel time on road segments near the listed stops is multiplied during the event window. "
            "The exact multiplier is NOT given to you -- only the collision's rough size (e.g. "
            "2-car vs 5-car) -- so you must ESTIMATE the actual slowdown (a bigger collision implies "
            "a larger one), not treat any specific number as known. "
            + (
                "Note: this table's (per-stop) travel_time is an ESTIMATE that assumes every "
                "directly affected stop's leg IS slowed by this accident -- it does NOT check whether "
                "this route's timing would actually place that stop inside the accident's active "
                "window, so treat it as a worst-case, not a verified crossing. "
                if stats_is_estimate else
                'No executed route timing is available; crossing this window depends on the chosen order. '
                if cluster_source == 'kim' else
                "Note: the table's (per-stop) travel_time already reflects whether THIS route's actual "
                "timing happened to cross the accident window on that leg -- but a stop's arrival/"
                "travel_time being fine in the CURRENT order doesn't mean every order would be; if "
                "you're considering whether reordering helps, remember visiting an affected stop at a "
                "different time could still cross the window. "
            ) +
            "Strategy: visit affected stops before the accident starts or after it ends if TW allows, "
            "otherwise expect longer travel time when passing through during the window."
        )
    if d_parts:
        disruption_section = "\n## Disruption Events\n" + now_line + "\n\n".join(d_parts) + "\n"

    example_parts = []
    for k in range(min(K, 2)):
        ids = ", ".join(f'"{n}": <score>' for n in clusters[k][:3])
        example_parts.append(f'"{k+1}": {{"0": <score>, {ids}, ...}}')
    if K > 2:
        example_parts.append(f'..., "{K}": {{...}}')
    example_ranking = ", ".join(example_parts)
    if cluster_source == 'kim':
        if confidence_only:
            example_ranking = json.dumps({str(k + 1): ['0'] + [str(n) for n in ns]
                                         for k, ns in enumerate(clusters)})
        return _kim_initial_prompt(T, K, table, disruption_section,
                                   _objective_str(reward_config), bks_fewshot,
                                   experience_section, example_ranking, use_cot, use_misfit,
                                   confidence_only=confidence_only)

    has_events = bool(rain_lines or acc_lines)
    fit_gate = "" if not use_misfit else (
        "\n## Step 1 — Cluster Fit Check (do this FIRST, for every vehicle)\n"
        "The question for EVERY customer is ONE thing: if this vehicle's stops were freely "
        "reordered (Step 2 below), would that fix whatever problem this customer causes or "
        "experiences -- or does the problem persist no matter what order this vehicle visits "
        "its assigned customers in? Fixable by reordering -> NOT a misfit, handle it via "
        "priority in Step 2. Persists regardless of order -> misfit: this customer structurally "
        "belongs in a DIFFERENT vehicle's group, not this one. late_stops, geographic distance, "
        "and disruption events (below) are not three separate misfit categories -- they are just "
        "the evidence you use to answer this one question.\n"
        "Each vehicle's table block starts with two reference lines:\n"
        "  '[RL visit order, vehicle k]' -- the actual sequence the routing agent visits this "
        "vehicle's customers in.\n"
        "  '[" + perf_label + ", vehicle k]' -- "
        + ("an ESTIMATE of what this vehicle's plan produces: " if stats_is_estimate else
           "what this vehicle's plan REALLY produced: ") +
        "late_stops (count arriving after TW_close) and travel_time (total time actually spent "
        "driving this route, INCLUDING any rain/accident slowdown -- this is what the RL agent's "
        "reward is actually penalized on, not raw geometric distance, so treat it as the real cost "
        "when judging whether a customer is 'too costly' for this group) -- a quick flag, "
        "not the detail; the table's per-row 'slack' column (negative = late) gives the exact "
        "per-customer picture. " +
        ("This is an ESTIMATE, not a verified replay -- weigh it, but don't treat it as certain.\n"
         if stats_is_estimate else
         "This is ground truth, not a prediction -- weigh it heavily.\n") +
        "  - late_stops=0 -> there is nothing reordering needs to fix either; do NOT flag anyone in "
        "this vehicle without a very specific, undeniable reason (e.g. a customer so geographically "
        "remote that ANY order through this group would need a long detour to reach them, even "
        "though today's particular order happens to still make every deadline).\n"
        "  - late_stops>0 -> the table's per-row 'slack' column tells you EXACTLY which customer(s) "
        "actually arrived late (slack<0). For each one, use 'slack' and 'travel_time' (per-stop) on "
        "the stops around it: could visiting it EARLIER in this same vehicle's sequence (still the "
        "same group, just reordered) plausibly have avoided the lateness -- e.g. is there positive "
        "slack on an earlier stop that could absorb this one's travel_time instead? If yes, that's a "
        "Step 2 priority fix, NOT a misfit. If no -- e.g. this stop's travel_time is large from every "
        "plausible neighbor "
        "so any order still costs a long detour, their time window conflicts with the group's "
        "schedule no matter when they're visited, or a disruption event blocks every plausible path "
        "to them from this group -- THEN it's a misfit.\n"
        "Do NOT flag a different, on-time (slack>=0) customer instead of the actual late (slack<0) "
        "one unless that on-time customer is clearly what's BLOCKING the late one (e.g. an earlier "
        "stop forcing a huge detour) -- in that case the blocking stop is the misfit, not the one "
        "arriving late as a result.\n"
        "Most customers should NOT be flagged -- reordering solves most lateness; only flag when the "
        "problem is structural (which group this customer is in), not sequential (the order within "
        "it). A misfit customer gets NO priority score in Step 2 below (they are excluded entirely).\n"
        "For every customer you flag, give a one-line reason keyed by their id in 'misfit_reasons' "
        "stating specifically why reordering could NOT fix it -- this is for later audit, not the "
        "model's own use, so keep each line short.\n"
    )
    # Depot-cutoff overload must clear the SAME bar as a Step-1 misfit (or, with
    # misfit off, the same "reordering can't fix it" test) -- otherwise a
    # customer Step 1 keeps (fixable by reordering) could still get excluded via
    # the depot cutoff for the same symptom, undermining what Step 1 decided.
    depot_reorder_test = (
        "the SAME test as Step 1's misfit call above"
        if use_misfit else
        "reordering could not fix it"
    )
    if use_cot:
        # NOTE: no generic "how to read the Experience section" guidance here.
        # For the 'rl' pipeline, experience_section is _build_base_prior_ref's
        # own plain-English sentence (which customers were misfits under the
        # base scenario) and already states its own usage inline ("if they
        # appear here too, that's corroborating evidence... fine to disagree
        # if the event resolves it") -- a prior version of this block described
        # a 'score'/'late'/'base_late(Δ)' per-customer table that no longer
        # exists anywhere in this codebase, telling the model to read fields
        # that were never actually in its input. Keep this empty unless
        # experience_section's actual content changes to something needing it.
        phase1_head = (
            "\n## Step 2 — Priority Analysis (per vehicle, misfit customers excluded)\n"
            "For every customer NOT flagged as a misfit in Step 1, score visit priority (0.0-1.0).\n"
            if use_misfit else
            "\n## Priority Analysis (per vehicle)\n"
            "For every customer, score visit priority (0.0-1.0).\n"
        )
        phase1 = (
            phase1_head +
            "A customer with a higher score gets a stronger bias toward being visited sooner by the RL agent.\n"
            "The '[RL visit order]' line above is simply the CURRENT solution for this vehicle -- not a "
            "sensible default worth preserving, and not a blank slate either. Your job is to give this "
            "current solution guidance, grounded in the real evidence in this table, toward eliminating "
            "whatever lateness it produces. The 'order' column is only a label for where that current "
            "solution placed each stop, not something to reward.\n"
            "The 'slack' column (real remaining buffer under the CURRENT route; negative = this specific "
            "stop actually arrived late under it) is that evidence, and it is decisive: ANY stop with "
            "slack < 0 must receive the highest priority in its vehicle, no exception, regardless of its "
            "'order' position. If more than one stop in the same vehicle is negative, rank the more "
            "negative one higher between them. A stop with large positive slack can safely absorb being "
            "deprioritized to make room for a late one. Route position, geographic position, and service "
            "duration are only tie-breakers among stops that are NOT late -- never a reason to outrank "
            "one that is.\n"
            "When raising a negative-slack stop's priority, check the slack of the OTHER stops in this "
            "vehicle it would likely be visited ahead of now (those you are scoring lower than it). Only "
            "push it as high as needed -- if fixing its lateness would leave one of those other stops "
            "with little or negative slack of its own, moderate the boost instead of maximizing it. The "
            "goal is zero late stops in this vehicle overall, not moving the problem from one stop to "
            "another.\n"
            "Do NOT force a spread across the group -- a well-formed group can legitimately have every "
            "member score high.\n"
            "Also score \"0\" (depot -- end this vehicle's route here) on the same 0.0-1.0 scale, "
            "alongside these customers. This one is not just another priority -- it's a cutoff: any "
            "customer you score BELOW \"0\" is treated exactly like a misfit for THIS vehicle -- pushed "
            "away from it so another vehicle serves them instead. The cutoff only works if \"0\"'s "
            "NUMBER actually lands below/above the right customers -- your stated intent alone does "
            "nothing. Base it on the same '[" + perf_label + "]' line: performance looks fine -> score "
            "\"0\" EXACTLY 0.0, not some vague low-but-nonzero number -- 0.0 means no cutoff at all; "
            "performance already shows signs of overload (late_stops>0, or travel_time clearly more "
            "than this group's own geography/TW should require) AND that overload would persist no "
            "matter how this vehicle's stops are reordered (" + depot_reorder_test + ""
            " -- if reordering COULD fix it, that's a priority fix instead, not a depot cutoff) -> "
            "score \"0\" HIGHER than every customer you mean to push away from this vehicle "
            "(check it against their actual numbers, not just their rank) and still lower than the "
            "customers it should keep. Note: every OTHER vehicle's customers are invisible to THIS "
            "vehicle -- they always get zero pull here. Scoring \"0\" at 0.0 keeps this vehicle's depot "
            "choice on equal footing with those other vehicles' customers, letting the base routing "
            "decide freely between them.\n"
            "Keep reasoning brief — one short line per stop.\n"
        )
        if has_events:
            phase2 = (
                "\n## Step 3 — Event Adjustment\n"
                "Re-examine Step 1/2 in light of disruption events. Remember the multiplier itself "
                "is never given to you -- any judgment about how much an event actually slows a "
                "given stop down is your own estimate from the qualitative severity info above, not "
                "a known number:\n"
            )
            if rain_lines:
                phase2 += "- Rain-affected stops: consider whether the rain window overlaps with the likely visit time.\n"
            if acc_lines:
                phase2 += "- Accident-edge stops: consider whether traversal during the event makes the deadline infeasible.\n"
            phase2 += ("Adjust misfit flags or scores if needed.\n" if use_misfit
                       else "Adjust scores if needed.\n")
        else:
            phase2 = ""
        task_desc = (f"For each vehicle (1-{K}): (1) flag any misfit customers, (2) score the rest for visit priority.\n"
                     if use_misfit else
                     f"For each vehicle (1-{K}), score every customer for visit priority.\n")
        json_schema = (
            f'{{"misfit": [id, id, ...], "misfit_reasons": {{"id": "one-line reason", ...}}, '
            f'"confidence": {{{example_ranking}}}, "reason": "one-line overall summary"}}'
            if use_misfit else
            f'{{"confidence": {{{example_ranking}}}, "reason": "one-line overall summary"}}'
        )
        schema_note = (
            "confidence covers every customer EXCEPT those listed in misfit, plus a \"0\" depot "
            "score per vehicle; misfit_reasons has one entry per flagged id, omit it entirely if "
            "misfit is empty"
            if use_misfit else
            "confidence covers every customer in every vehicle, plus a \"0\" depot score per vehicle"
        )
        task_section = (
            "## Task\n" + task_desc
            + fit_gate + phase1 + phase2 +
            f"\nReason through each vehicle's stops, then output JSON (include ALL vehicles; "
            f"{schema_note}):\n" + json_schema
        )
    else:
        task_desc = (f"For each vehicle (1-{K}): (1) flag any misfit customers, (2) score the rest for visit priority.\n"
                     if use_misfit else
                     f"For each vehicle (1-{K}), score every customer for visit priority.\n")
        json_schema = (
            f'{{"misfit": [id, id, ...], "misfit_reasons": {{"id": "one-line reason", ...}}, '
            f'"confidence": {{{example_ranking}}}, "reason": "one-line overall explanation"}}'
            if use_misfit else
            f'{{"confidence": {{{example_ranking}}}, "reason": "one-line overall explanation"}}'
        )
        priority_desc = (
            "For every customer NOT flagged as misfit, score visit priority (0.0-1.0). Use the "
            if use_misfit else
            "For every customer, score visit priority (0.0-1.0). Use the "
        )
        schema_note = (
            "confidence covers every customer EXCEPT those in misfit, plus a \"0\" depot score per "
            "vehicle; misfit_reasons has one entry per flagged id, omit it entirely if misfit is empty"
            if use_misfit else
            "confidence covers every customer in every vehicle, plus a \"0\" depot score per vehicle"
        )
        task_section = (
            "## Task\n" + task_desc
            + fit_gate + priority_desc +
            "'[RL visit order]' line as the CURRENT solution -- not a default to preserve. Give it "
            "guidance from the 'slack' column (real remaining buffer under the given route): ANY stop "
            "with slack < 0 must get the highest priority in its vehicle, no exception, regardless of "
            "its position in that order (rank multiple negative stops by how negative). A stop with "
            "large positive slack can safely be deprioritized to make room. Position and geography are "
            "only tie-breakers among stops that are NOT late. Only push a boost as high as needed -- if "
            "fixing one stop's lateness would leave a stop you're deprioritizing with little or negative "
            "slack instead, moderate it; the goal is zero late stops overall, not moving the problem.\n"
            "Also score \"0\" (depot -- end this vehicle's route here) on the same 0.0-1.0 scale: it's "
            "a cutoff, not just another priority -- any customer scored BELOW \"0\" is treated exactly "
            "like a misfit for THIS vehicle, pushed away so another vehicle serves them instead. The "
            "cutoff only works if \"0\"'s NUMBER actually lands below/above the right customers -- "
            "stating intent alone does nothing. Base it on the '[" + perf_label + "]' line: fine "
            "performance -> score \"0\" EXACTLY 0.0 (no vague low-but-nonzero number -- 0.0 means no "
            "cutoff); signs of overload that would persist no matter how this vehicle's stops are "
            "reordered (" + depot_reorder_test + " -- if reordering COULD fix it, that's a priority fix "
            "instead, not a depot cutoff) -> score \"0\" HIGHER than every customer you mean to push "
            "away (check it against their actual numbers) and still lower than the ones it should keep. "
            "Note: every OTHER vehicle's customers are invisible to THIS vehicle -- they always get "
            "zero pull here. Scoring \"0\" at 0.0 keeps this vehicle's depot choice on equal footing "
            "with those other vehicles' customers, letting the base routing decide freely between them.\n"
            f"Output ONLY valid JSON ({schema_note}):\n" + json_schema
        )

    if confidence_only:
        if use_misfit or not use_cot:
            raise ValueError('confidence_only requires misfit OFF and the full decision guidelines')
        phase1 = ''.join(
            line for line in phase1.splitlines(keepends=True)
            if not line.startswith('Keep reasoning brief')
        )
        task_section = (
            '## Task\n' + task_desc + fit_gate + phase1 + phase2
            + '\n## Output contract\n'
            'Apply the guidelines above without printing your analysis. Return ONLY one valid JSON object. '
            'Its sole top-level key must be "confidence". Include every vehicle and every assigned customer, '
            'plus depot "0" for each vehicle. Scores must be finite numbers in [0,1]. '
            'Do not include reason, explanations, route restatements, markdown fences, or any other keys.\n'
            'Required keys by vehicle (this is a key inventory, not scores):\n'
            + json.dumps({str(k + 1): ['0'] + [str(n) for n in ns]
                          for k, ns in enumerate(clusters)})
            + '\nReturn a confidence object mapping every listed vehicle to a score object '
            'with exactly its listed keys. Do not output this inventory as arrays. '
            'Do not omit low-priority customers; assign their scores too. '
            'Before returning, check every required key is present exactly once.\n'
        )

    objective = _objective_str(reward_config)
    fewshot_block = f"\n{bks_fewshot}\n" if bks_fewshot else ""
    travel_time_note = (
        (
            " Under an active rain/accident event, some legs may run slower than their raw Euclidean "
            "distance (x,y) would suggest -- this is why the RL agent's reward is penalized on summed "
            "travel_time, not raw distance. Do NOT back out an exact slowdown factor by "
            "comparing this row's travel_time to (x,y) distance -- use the qualitative severity info in "
            "Disruption Events below (rainfall mm/h, collision size) for how bad the event itself is, "
            + (
                "and treat travel_time only as an ESTIMATE for reordering decisions -- it adjusts each "
                "directly affected stop's OWN leg but does not re-derive what a reordered route's actual "
                "timing would be."
                if stats_is_estimate else
                "and treat travel_time as this route's actual ground truth for reordering decisions."
            )
        )
        if has_events else
        " This instance has no active rain/accident event, so every row's travel_time here equals its "
        "raw Euclidean (x,y) distance exactly -- no disruption multiplier is in play."
    )
    return f"""{prefix}## Role
You are an expert logistics planner managing a VRPTW delivery operation.

## Problem
- Time horizon: {T:.0f} min
- Travel time between any two points = Euclidean distance (speed = 1 unit/min)
- Vehicle departs from and returns to the depot (node 0)
- {K} vehicles are deployed; each is pre-assigned a cluster of customers (see "veh" column)

## Column Definitions
- **veh**: vehicle index (1-{K}) assigned to serve this customer
- **TW_open**: time when the service window opens. Arriving early means waiting until TW_open before service begins.
- **TW_close**: deadline by which the vehicle must ARRIVE. Arriving after TW_close is a time window violation.
- **svc**: service duration (min). Visit sequence: arrive -> wait if early -> serve svc min -> depart.
  Departure time = max(arrival, TW_open) + svc. High svc delays all subsequent stops.
- **slack** = TW_close - actual arrival, under {slack_basis}. Positive -> buffer left (this customer was
  fine, by that many minutes). Negative -> exactly how late this SPECIFIC customer arrived -- slack<0
  identifies the EXACT stop(s) responsible for the vehicle's lateness, not just which vehicle; use it
  directly instead of guessing. {slack_history_note} -- do not recompute it from TW_open/TW_close alone,
  those don't account for the actual multi-stop schedule. "N/A" means no {no_data_basis} data exists for this stop (not that
  it's 0 / on time) -- fall back to TW_open/TW_close/x,y for it.
- **travel_time** (per-row, not to be confused with the per-vehicle travel_time in "[{perf_label}]"
  above): {tt_basis} in the current route to this one (first stop's
  travel_time is measured from the depot). This is what reordering actually trades off -- moving a customer
  earlier or later changes which travel_times apply, so use this (not x,y guesswork) to judge whether a
  different order in this same vehicle could plausibly help.{travel_time_note}
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


def _log_misfit_reasons(response: str) -> None:
    """Print the LLM's own 'misfit_reasons' (per flagged id) and overall 'reason'
    fields, purely for human audit -- neither is consumed by parse_all_clusters_
    confidence or used in training. Best-effort: silently does nothing if the
    response doesn't parse (parse_all_clusters_confidence already surfaces that
    failure on its own)."""
    import re, json
    stripped = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not m:
        return
    try:
        data = json.loads(re.sub(r'//[^\n]*', '', m.group()))
    except json.JSONDecodeError:
        return
    reasons = data.get("misfit_reasons")
    if isinstance(reasons, dict) and reasons:
        print(f"  [LLM:misfit_reasons] {reasons}")
    overall = data.get("reason")
    if isinstance(overall, str) and overall.strip():
        print(f"  [LLM:reason] {overall.strip()}")


def parse_all_clusters_confidence(
    response: str,
    clusters: List[List[int]],
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, float]]:
    """Parse combined LLM response → ({cluster_idx (0-based): {node: score}}, {cluster_idx: depot_score}).

    Primary format: {"misfit": [id, id, ...], "confidence": {"1": {id: score,
    "0": depot_score}, "2": {...}}, ...}. `misfit` is a flat list (each node
    already belongs to exactly one cluster) of customers the LLM judged do NOT
    belong in their assigned vehicle's group -- these get the fixed MISFIT_BIAS,
    not an LLM-reported magnitude. `confidence` gives an absolute per-customer
    priority score ∈ [0,1] for every OTHER (non-misfit) customer, independently
    judged (not a forced relative spread) -- see build_all_clusters_prompt's
    Step 1/Step 2 sections and this module's docstring for the fit-then-priority
    rationale. Each cluster's own "0" entry (if present) is that vehicle's
    depot/return-to-depot score -- kept OUT of the per-node `scores` dict
    (0 is never a real customer id) and returned separately in `depot_scores`,
    keyed by cluster_idx, since -- unlike every real customer -- depot needs a
    DIFFERENT value per vehicle (see build_all_clusters_prompt's depot-scoring
    instructions: it's a cutoff, not just another priority).

    Falls back to the older {"ranking": {"1": [id1, id2, ...], ...}} format
    (position 0 -> 0.90, last -> 0.10, uniform linear) for responses from
    before this switch -- that format has no fit/misfit concept at all, and
    never carries a depot score (depot_scores comes back empty in that case).
    """
    import re, json
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", response, re.DOTALL)
    data = None
    if m:
        json_str = re.sub(r'//[^\n]*', '', m.group())
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            pass

    if data is None:
        # Observed failure mode (deterministic for some prompts even at
        # temperature=0.0, e.g. RC103's 21-cluster combined prompt): the
        # model writes complete, correct "confidence"/"reason" content but
        # never emits the final closing brace(s) for the outermost object --
        # so `m` above either finds no {...} at all (no closing brace exists
        # anywhere after truncation) or matches a string that's short by
        # exactly the outer brace(s). This is a pure SYNTAX gap, not missing
        # DATA -- take everything from the first "{" onward and try
        # appending 1-3 closing braces; the completeness check below (every
        # cluster's every customer + depot present) still applies exactly as
        # for a normal response, so a genuinely incomplete/truncated-mid-
        # cluster response still fails there, same as before this repair.
        start = response.find('{')
        if start != -1:
            tail = re.sub(r'//[^\n]*', '', response[start:])
            for _n_close in range(1, 4):
                try:
                    data = json.loads(tail + '}' * _n_close)
                    print(f"  [LLM] response was missing {_n_close} closing brace(s) "
                          f"-- auto-repaired (syntax only, no data invented)", flush=True)
                    break
                except json.JSONDecodeError:
                    continue

    if data is None:
        return {}, {}

    result: Dict[int, Dict[int, float]] = {}
    depot_scores: Dict[int, float] = {}

    # misfit is a flat list of node ids (each node belongs to exactly one
    # cluster already, from the input `clusters`, so no per-vehicle keying
    # is needed here) -- assign the fixed MISFIT_BIAS, not an LLM-supplied
    # magnitude (see parse_cluster_confidence's docstring for why: the
    # binary fit/no-fit call is reliable, a self-reported severity number
    # is not, so the system -- not the LLM -- owns that constant).
    misfit_raw = data.get("misfit", [])
    misfit_set: Set[int] = set()
    if isinstance(misfit_raw, list):
        for x in misfit_raw:
            try:
                misfit_set.add(int(x))
            except (ValueError, TypeError):
                continue

    # Primary format: {"confidence": {"1": {node: score, "0": depot_score}, ...}}
    conf_raw = data.get("confidence", {})
    if isinstance(conf_raw, dict) and conf_raw:
        for k, cluster_nodes in enumerate(clusters):
            valid_set = set(cluster_nodes)
            cluster_conf = conf_raw.get(str(k + 1), {})
            scores: Dict[int, float] = {}
            if isinstance(cluster_conf, dict):
                for node_str, score in cluster_conf.items():
                    try:
                        node = int(node_str)
                    except (ValueError, TypeError):
                        continue
                    if node == 0:
                        depot_scores[k] = max(0.0, min(1.0, float(score)))
                    elif node in valid_set:
                        scores[node] = max(0.0, min(1.0, float(score)))
            for node in valid_set & misfit_set:
                scores[node] = MISFIT_BIAS
            if scores:
                result[k] = scores
        if result:
            return result, depot_scores

    # Fallback: {"ranking": {"1": [id, id, "0", id, ...], "2": [...]}} -- a
    # per-vehicle ORDER (customer ids plus "0" for depot, placed wherever the
    # cutoff belongs), not a hand-picked 0-1 magnitude, kept for responses
    # from a brief experiment with order-as-primary. Position 0 (first) ->
    # 0.95, last -> 0.10, uniform linear. "0" placed after some customers
    # gives those customers a lower derived score than depot's own (both from
    # the same linear scale), so the existing below-depot -> MISFIT_BIAS
    # conversion in _build_cluster_conf_tensor applies to them automatically.
    rank_raw = data.get("ranking", {})
    if isinstance(rank_raw, dict) and rank_raw:
        for k, cluster_nodes in enumerate(clusters):
            valid_set = set(cluster_nodes)
            raw_list = rank_raw.get(str(k + 1), [])
            if not isinstance(raw_list, list):
                continue
            seq: List[int] = []
            for x in raw_list:
                try:
                    node = int(x)
                except (ValueError, TypeError):
                    continue
                if node == 0 or node in valid_set:
                    seq.append(node)
            if not seq:
                continue
            n = len(seq)
            scores: Dict[int, float] = {}
            depot_score = None
            for i, node in enumerate(seq):
                score = round(0.95 - i * 0.85 / max(n - 1, 1), 4) if n > 1 else 0.95
                if node == 0:
                    depot_score = score
                else:
                    scores[node] = score
            for node in valid_set & misfit_set:
                scores[node] = MISFIT_BIAS
            if depot_score is not None:
                depot_scores[k] = depot_score
            if scores:
                result[k] = scores
    return result, depot_scores


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
    stats: Optional[Dict[int, dict]] = None,
    use_misfit: bool = False,
    depot_out: Optional[Dict[int, float]] = None,
    events_visible_until: Optional[float] = None,
    stats_is_estimate: bool = False,
    cluster_source: str = 'rl',
) -> Dict[int, float]:
    """Single LLM call for all clusters; returns flat {node: score} dict.

    cluster_source selects prompt provenance ('rl' by default, or 'kim').
    Kim mode uses the same parser and output schema, not a separate backend.

    If visited is provided, only remaining (unvisited) nodes are included in
    the prompt and scored. TW slack is recalculated relative to cur_time.

    stats (optional): {cluster_idx: {late_count, total_late, travel_time}} --
    each vehicle's ACTUAL realized performance (see extract_rl_routes.py /
    _parse_cluster_stats), shown alongside the RL visit order so the fit
    check is grounded in real outcomes, not just static TW/geometry. None
    when unavailable (e.g. mid-episode accident re-scoring, older caches).

    depot_out (optional): if a dict is passed, it is updated IN PLACE with
    {cluster_idx: depot_score} -- unlike every real customer, depot needs a
    DIFFERENT value per vehicle, so it can't ride along in the flat {node:
    score} return value (node 0 isn't a real customer, and one flat dict
    entry can't hold K different vehicle-specific values). Callers that don't
    care about depot scores just omit this and get the old behavior.
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
        stats = None  # vehicle-level stats no longer apply once nodes are removed
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
                                       bks_fewshot=bks_fewshot,
                                       stats=stats,
                                       use_misfit=use_misfit,
                                       events_visible_until=events_visible_until,
                                       stats_is_estimate=stats_is_estimate,
                                       cluster_source=cluster_source)
    if prompt_save_path:
        with open(prompt_save_path, 'w', encoding='utf-8') as _f:
            _f.write(prompt)
    resp   = query_llm(prompt, model, think=use_cot)
    conf_by_cluster, depot_by_cluster = parse_all_clusters_confidence(resp, prompt_clusters)
    _log_misfit_reasons(resp)

    flat_scores: Dict[int, float] = {}
    failed_clusters: List[int] = []
    for k, cluster_nodes in enumerate(prompt_clusters):
        cluster_conf  = conf_by_cluster.get(k)
        # A cluster missing SOME of its customers (or its depot score) previously
        # looked identical to "the LLM scored it near zero" -- the missing ones
        # silently defaulted to 0.0 below. Both are now treated as a failure of
        # THIS cluster so the caller's retry loop actually re-asks, instead of
        # fabricating a "lowest priority" judgment the LLM never made.
        missing_nodes = set(cluster_nodes) - (cluster_conf or {}).keys()
        missing_depot = k not in depot_by_cluster
        if not cluster_conf or missing_nodes or missing_depot:
            failed_clusters.append(k)
        else:
            top = sorted(cluster_conf.items(), key=lambda x: -x[1])
            print(f"  [LLM:cluster{k}] conf={[(n, f'{s:.2f}') for n,s in top[:8]]}"
                  f"{'...' if len(top) > 8 else ''}")
            flat_scores.update(cluster_conf)

    if failed_clusters:
        print(f"  [LLM] combined parse FAILED for {len(failed_clusters)}/{len(prompt_clusters)} clusters")
        print(f"  [LLM:debug] response preview: {repr(resp[:500])}", flush=True)
        for k in failed_clusters:
            print(f"  [LLM:cluster{k}] FAILED -- no confidence assigned")
        raise RuntimeError(
            f"LLM combined call failed for {len(failed_clusters)}/{len(prompt_clusters)} clusters"
        )

    # depot_out is only ever updated once the response is FULLY complete (past
    # the raise above) -- updating it as soon as parsing succeeded (before this
    # completeness check could still reject the response) let a caller's retry
    # loop see a stray partial depot score alongside customer scores it never
    # actually accepted, from an attempt that otherwise failed.
    if depot_by_cluster:
        print(f"  [LLM:depot] {{{', '.join(f'{k}: {v:.2f}' for k, v in sorted(depot_by_cluster.items()))}}}")
        if depot_out is not None:
            depot_out.update(depot_by_cluster)

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


# ── Incremental (delta-from-base) event confidence ───────────────────────────
# Rationale: re-ranking ALL customers per event forces the LLM to re-derive a
# score for every customer even when nothing about them changed, conflates
# "noise from re-ranking" with "genuine event judgment", and cannot express
# "no change needed" (see design doc: C1 rain flips 0/15 LKH-3 oracle plans --
# the correct answer there IS "adjust nothing"). Asking only "what changes from
# base" sidesteps both problems and is largely insensitive to how many other
# customers are in the prompt (unlike a fresh full re-ranking, whose derived
# score for a customer depends on how many peers it's being ranked against).

def _flat_customer_table(inst: dict, node_ids: List[int], base_conf: Dict[int, float]) -> str:
    """Simple flat table (no per-cluster grouping, no concept labels) for the
    incremental-adjustment prompts: just enough for the LLM to judge whether an
    event changes THIS customer's priority relative to its base score."""
    ont = SoftClusterOntology(inst)
    hdr = f"{'ID':>4}  {'TW_open':>8}  {'TW_close':>9}  {'svc':>5}  {'dist_depot':>10}  {'confidence':>10}"
    rows = [hdr, "-" * len(hdr)]
    for c in sorted(node_ids):
        tw_o = float(ont.tw_open[c])
        tw_c = float(ont.tw_close[c])
        svc  = float(ont.service[c])
        dist = float(ont.tt[0, c])
        conf = base_conf.get(c, 0.0)
        rows.append(f"{c:>4}  {tw_o:>8.0f}  {tw_c:>9.0f}  {svc:>5.0f}  {dist:>10.1f}  {conf:>10.2f}")
    return "\n".join(rows)


def _time_structure_line(inst: dict, node_ids: List[int]) -> str:
    """Service-vs-travel-time ratio -- lets the LLM judge whether a travel-time
    slowdown can plausibly move the schedule at all (design doc §2.2: C1's 90min
    service / 92min travel means a 19% speed loss moves <2% of the schedule)."""
    ont = SoftClusterOntology(inst)
    svc_total = sum(float(ont.service[c]) for c in node_ids)
    tt_total  = sum(float(ont.tt[0, c]) for c in node_ids)
    denom = svc_total + tt_total
    tt_frac = (tt_total / denom * 100) if denom > 0 else 0.0
    return (f"Per-vehicle service time ~{svc_total / max(len(node_ids), 1):.0f}min/stop, "
            f"depot-to-stop travel time ~{tt_total / max(len(node_ids), 1):.0f}min/stop "
            f"(travel is {tt_frac:.0f}% of a stop's time budget).")


def _extract_json_block(response: str) -> dict | None:
    import re, json
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        return None
    json_str = re.sub(r'//[^\n]*', '', m.group())
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def build_rain_incremental_prompt(
    inst: dict,
    node_ids: List[int],
    base_conf: Dict[int, float],
    rain_events: List[dict],
) -> str:
    """RAIN is known before dispatch (rho_e=0) -> a single incremental adjustment
    covers the whole episode; no runtime swap needed (see get_rain_incremental_confidence
    caller, which just replaces the base cache with the merged result)."""
    table = _flat_customer_table(inst, node_ids, base_conf)
    node_set = set(node_ids)
    rain_lines = []
    for ev in rain_events:
        affected = sorted(n for n in ev.get('affected_nodes', ev.get('nodes', [])) if n in node_set)
        if not affected:
            continue
        t_s = ev.get('trigger_time', ev.get('t_start', 0))
        t_e = t_s + ev.get('duration', 0)
        rain_lines.append(
            f"  Rainfall {ev.get('rainfall_mm', 10):.0f}mm/h expected t={t_s:.0f}-{t_e:.0f}, "
            f"affecting stops {affected}"
        )
    rain_section = "\n".join(rain_lines) if rain_lines else "  (no stops in this cluster are in the forecast zone)"

    return f"""## Existing priority (assigned under no-event conditions)
{table}

## Weather forecast (issued before dispatch)
{rain_section}

## Time structure
{_time_structure_line(inst, node_ids)}

## Task
Considering the forecast, list ONLY the customers whose priority (confidence) needs
adjusting relative to the existing table above. Rain increases travel time on affected
road segments -- if that plausibly threatens a stop's time window given the time
structure above, raise its confidence; if the schedule has enough slack to absorb it,
leave it unchanged.
If NO customer needs adjusting, return an empty object for "adjust" -- this is a
valid and often-correct answer, do not force a change.

Output ONLY valid JSON with the customer's NEW confidence value (0.0-1.0):
{{"adjust": {{"<id>": <new_confidence>, ...}}, "reason": "one-line summary"}}"""


def parse_incremental_adjust(response: str, valid_nodes: set) -> Dict[int, float]:
    """Parse {"adjust": {"<id>": val, ...}} -> {node: new_confidence}. Empty/missing
    -> {} (means "no change", NOT a parse failure -- caller keeps base values)."""
    data = _extract_json_block(response)
    if data is None:
        raise RuntimeError(f"incremental-adjust parse failed; response preview: {response[:300]!r}")
    adjust_raw = data.get("adjust", {})
    if not isinstance(adjust_raw, dict):
        return {}
    out: Dict[int, float] = {}
    for k, v in adjust_raw.items():
        try:
            node = int(k)
            val  = float(v)
        except (ValueError, TypeError):
            continue
        if node in valid_nodes:
            out[node] = max(0.0, min(1.0, val))
    return out


def get_rain_incremental_confidence(
    inst: dict,
    node_ids: List[int],
    base_conf: Dict[int, float],
    rain_events: List[dict],
    model: str = DEFAULT_MODEL,
    use_cot: bool = True,
) -> Dict[int, float]:
    """Returns the FULL merged {node: confidence} for all node_ids (base values
    for untouched nodes, adjusted values for the rest) -- ready to drop straight
    into the existing flat-confidence cache format, no downstream merge needed."""
    if not rain_events:
        return dict(base_conf)
    prompt = build_rain_incremental_prompt(inst, node_ids, base_conf, rain_events)
    adjust = None
    for attempt in range(3):
        resp = query_llm(prompt, model, think=use_cot)
        try:
            adjust = parse_incremental_adjust(resp, set(node_ids))
            break
        except RuntimeError as e:
            print(f"  [LLM:rain-incremental] parse failed (attempt {attempt+1}/3): {e}")
    if adjust is None:
        print(f"  [LLM:rain-incremental] all attempts failed -- falling back to 'no adjustment' "
              f"(base confidence kept as-is; a valid answer under this design).")
        adjust = {}
    print(f"  [LLM:rain-incremental] {len(adjust)}/{len(node_ids)} customers adjusted: "
          f"{[(n, f'{base_conf.get(n,0):.2f}->{v:.2f}') for n, v in list(adjust.items())[:8]]}"
          f"{'...' if len(adjust) > 8 else ''}")
    merged = dict(base_conf)
    merged.update(adjust)
    return merged


def build_acc_staged_prompt(
    inst: dict,
    node_ids: List[int],
    base_conf: Dict[int, float],
    accident_events: List[dict],
) -> str:
    """ACC is revealed in real time (rho_e=sigma_e), and a wave can contain 2-3
    accidents at different trigger times -- ask for all stages in ONE call
    (cost stays 1 call/instance; see design doc §3.2/§4)."""
    table = _flat_customer_table(inst, node_ids, base_conf)
    node_set = set(node_ids)
    sorted_evs = sorted(accident_events, key=lambda e: e.get('trigger_time', e.get('t_start', 0)))
    bulletin_lines = []
    for ev in sorted_evs:
        affected = sorted(n for n in ev.get('affected_nodes', ev.get('nodes', [])) if n in node_set)
        t_s = ev.get('trigger_time', ev.get('t_start', 0))
        veh = ev.get('vehicles_involved')
        veh_str = f"{veh}-vehicle collision" if veh else "traffic incident"
        bulletin_lines.append(f"  [Bulletin t={t_s:.1f}] {veh_str} -- affecting stops {affected}")
    bulletins = "\n".join(bulletin_lines)
    stage_keys = ", ".join(f'"{ev.get("trigger_time", ev.get("t_start", 0)):.1f}"' for ev in sorted_evs)

    return f"""## Existing priority (assigned under no-event conditions)
{table}

## Traffic bulletins -- arriving sequentially during execution
{bulletins}

## Time structure
{_time_structure_line(inst, node_ids)}

## Task
These bulletins arrive one at a time, in the order shown, as the route is already
underway. For EACH bulletin, list ONLY the customers whose priority needs adjusting
AT THAT POINT -- each bulletin's adjustment is relative to the state AFTER all
earlier bulletins' adjustments have already been applied (cumulative), not relative
to the original base table.
If a bulletin needs no adjustment, give it an empty object -- this is a valid answer.

Output ONLY valid JSON, one key per bulletin time ({stage_keys}), each mapping to
the customers' NEW confidence values (0.0-1.0):
{{{stage_keys.split(',')[0]}: {{"<id>": <new_confidence>, ...}}, ..., "reason": "one-line summary"}}"""


def parse_acc_staged_response(response: str, valid_nodes: set) -> List[tuple]:
    """Parse {"<t>": {"<id>": val, ...}, ..., "reason": ...} -> [(t, {node: val}), ...]
    sorted by t ascending. Non-numeric keys (e.g. "reason") are skipped."""
    data = _extract_json_block(response)
    if data is None:
        raise RuntimeError(f"acc-staged parse failed; response preview: {response[:300]!r}")
    stages = []
    for k, v in data.items():
        try:
            t = float(k)
        except (ValueError, TypeError):
            continue  # "reason" or other non-time key
        if not isinstance(v, dict):
            continue
        adj: Dict[int, float] = {}
        for nk, nv in v.items():
            try:
                node = int(nk)
                val  = float(nv)
            except (ValueError, TypeError):
                continue
            if node in valid_nodes:
                adj[node] = max(0.0, min(1.0, val))
        stages.append((t, adj))
    stages.sort(key=lambda x: x[0])
    return stages


def get_acc_staged_confidence(
    inst: dict,
    node_ids: List[int],
    base_conf: Dict[int, float],
    accident_events: List[dict],
    model: str = DEFAULT_MODEL,
    use_cot: bool = True,
) -> Dict:
    """Returns {"base": {node: conf}, "stages": [{"t": t, "conf": {node: conf}}, ...]}
    with each stage's "conf" already fully resolved (base + all adjustments up to
    and including that stage merged in) -- so the training loop just swaps in the
    stage's dict directly, no delta-merging needed at load time."""
    if not accident_events:
        return {"base": dict(base_conf), "stages": []}
    prompt = build_acc_staged_prompt(inst, node_ids, base_conf, accident_events)
    raw_stages = None
    for attempt in range(3):
        resp = query_llm(prompt, model, think=use_cot)
        try:
            raw_stages = parse_acc_staged_response(resp, set(node_ids))
            break
        except RuntimeError as e:
            print(f"  [LLM:acc-staged] parse failed (attempt {attempt+1}/3): {e}")
    if raw_stages is None:
        print(f"  [LLM:acc-staged] all attempts failed -- falling back to 'no adjustment' "
              f"for all stages (base confidence kept as-is; a valid answer under this design).")
        raw_stages = []

    sorted_evs = sorted(accident_events, key=lambda e: e.get('trigger_time', e.get('t_start', 0)))
    expected_ts = [ev.get('trigger_time', ev.get('t_start', 0)) for ev in sorted_evs]

    # Match parsed stage keys to expected trigger times by closest value (LLM may
    # round/reformat the time slightly); missing stages default to "no change".
    result_stages = []
    running = dict(base_conf)
    for t_expected in expected_ts:
        best = min(raw_stages, key=lambda s: abs(s[0] - t_expected), default=None) if raw_stages else None
        adj = best[1] if (best is not None and abs(best[0] - t_expected) < 1.0) else {}
        running = dict(running)
        running.update(adj)
        print(f"  [LLM:acc-staged] t={t_expected:.1f}: {len(adj)}/{len(node_ids)} customers adjusted")
        result_stages.append({"t": t_expected, "conf": running})

    return {"base": dict(base_conf), "stages": result_stages}
