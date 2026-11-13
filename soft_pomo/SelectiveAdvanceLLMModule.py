"""Sparse positive guidance within the existing active-route cluster scope."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from SoftClusterLLMModule import _customer_table_clustered
from SoftClusterOntology import SoftClusterOntology
from VRPTWLLMModule import query_llm

SCHEMA_VERSION = "selective_advance_cluster_v2"
APPLICATION_SCOPE = "active_route_cluster"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(text):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject)


def parse_advance_response(response: str, customer_ids) -> dict[int, float]:
    data = strict_json(response)
    if not isinstance(data, dict) or set(data) != {"advance"}:
        raise ValueError("Expected exactly one top-level key: advance")
    if not isinstance(data["advance"], dict):
        raise ValueError("advance must be an object")
    allowed = {str(n) for n in customer_ids if n != 0}
    result = {}
    for node, score in data["advance"].items():
        if node not in allowed:
            raise ValueError(f"Unknown/noncanonical customer ID: {node}")
        if (type(score) not in (int, float) or not math.isfinite(score)
                or not 0 < score <= 1):
            raise ValueError(f"Customer {node}: strength must be finite and in (0, 1]")
        result[int(node)] = float(score)
    return result


def build_advance_prompt(inst, clusters, *, stats, objective: str,
                         events_visible_until=None, stats_is_estimate=False):
    """Use caller-supplied scenario-specific reference routes, never re-sort them."""
    nodes = [n for cluster in clusters for n in cluster]
    expected = set(range(1, inst['n_customers'] + 1))
    if len(nodes) != len(set(nodes)) or set(nodes) != expected:
        raise ValueError("Reference routes must cover every customer exactly once, without depot")
    if not objective.strip():
        raise ValueError("Explicit experiment objective/reward description is required")
    if set(stats) != set(range(len(clusters))):
        raise ValueError("Reference statistics must cover every route")
    for k, cluster in enumerate(clusters):
        detail = stats[k].get('node_detail', {})
        for node in cluster:
            row = detail.get(str(node), detail.get(node))
            if not isinstance(row, dict) or not {'slack', 'travel_time'} <= set(row):
                raise ValueError(f"Missing reference timing for customer {node}")
            if any(type(row[key]) not in (int, float) or not math.isfinite(row[key])
                   for key in ('slack', 'travel_time')):
                raise ValueError(f"Invalid reference timing for customer {node}")
    if 'ACC' in inst['name'].upper() and events_visible_until is None:
        raise ValueError("ACC prompts require an explicit event visibility cutoff")
    ont = SoftClusterOntology(inst)
    contexts = [ont.get_cluster_context(c, k, cur_time=events_visible_until or 0)
                for k, c in enumerate(clusters)]
    table = _customer_table_clustered(
        ont, clusters, contexts, stats=stats,
        events_visible_until=events_visible_until,
        stats_is_estimate=stats_is_estimate, cluster_source='rl')
    return f"""Select sparse positive interventions for a VRPTW RL policy.
Experiment objective: {objective}

Reference routes define the existing fixed reference clusters, not optimal
solutions or hard vehicle assignments. Keep their membership unchanged.
Select only customers for which earlier service within their cluster is plausibly useful
under the objective. Leave other decisions to RL. No change is a valid decision.
Compare the benefit with extra travel, waiting, capacity and delays to OTHER
customers. Arrival strictly after TW_close is late. Positive slack, whether
small or large, is not inherently good or bad. Negative slack alone does not
justify intervention: advancing that customer may cause greater downstream harm.
Do not copy ID order, TW order, slack order, or reference order into scores.

Each selected customer's score is an intervention strength in (0,1], NOT a
probability of correctness, absolute urgency, target rank, or number of positions
to move. A larger value requests a stronger positive policy-logit bonus.
Preserve the existing cluster-based application rule: the first customer visited
since the most recent depot visit anchors the active cluster. That anchor stays
fixed until depot return, even if RL visits a customer in another cluster.
The consumer adds lambda * strength ONLY to selected, feasible, unvisited
customers belonging to that active cluster. Before its first customer is chosen,
at the depot, or after all members of the active cluster have been visited,
this guidance is off. Customers in other clusters receive zero bonus, even if
selected for their own cluster. Cross-cluster visits remain legal under the
existing RL feasibility masks; do not create hard cluster constraints.
The existing POMO starts and cluster anchoring are not changed. RL still chooses the
actual insertion position; an earlier visit is not guaranteed. Unselected nodes
and depot receive exactly zero bonus. Do not request negative bonuses or delays.
Multiple selections are allowed; assess their combined effects. Use comparable
strengths for similarly supported interventions, and avoid unnecessarily strong
or dense interventions. Do not invent a numeric optimal calibration.

Reference statistics are {'estimates, NOT actual post-event rollout outcomes' if stats_is_estimate else 'supplied scenario-specific reference outcomes'}.
They describe the reference route, not every future training episode.
{'ACC timing uses BASE routes with visible-event own-leg adjustments only. Downstream delays are not propagated; no current episode position or visited set is known. Do not treat this as an actual post-accident execution.' if stats_is_estimate else ''}

{table}

Allowed customer IDs: {', '.join(map(str, sorted(nodes)))}
Return ONLY a JSON object with exactly the key "advance", mapping selected
customer ID strings to numeric strengths. Omit unselected customers entirely.
The deliberate no-intervention response is {{"advance": {{}}}}.
No depot, prose, markdown fences, additional fields or ellipses.
"""


def generate_advance(inst, clusters, *, stats, objective, model, output_dir,
                     events_visible_until=None, stats_is_estimate=False,
                     think=False, num_ctx=131072):
    """One logical request in a fresh directory; preserve failures, never fill scores."""
    prompt = build_advance_prompt(
        inst, clusters, stats=stats, objective=objective,
        events_visible_until=events_visible_until, stats_is_estimate=stats_is_estimate)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'prompt.txt').write_text(prompt, encoding='utf-8')
    reference = json.dumps({'clusters': clusters, 'stats': stats}, sort_keys=True)
    metadata = dict(schema_version=SCHEMA_VERSION, instance=inst['name'],
                    application_scope=APPLICATION_SCOPE, clusters=clusters,
                    model=model, think=think, num_ctx=num_ctx,
                    events_visible_until=events_visible_until,
                    stats_is_estimate=stats_is_estimate,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    reference_sha256=hashlib.sha256(reference.encode()).hexdigest())
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    response_format = {'type': 'json_schema', 'json_schema': {
        'name': 'selective_advance', 'strict': True,
        'schema': {'type': 'object', 'properties': {
            'advance': {'type': 'array', 'items': {'type': 'object', 'properties': {
                'customer_id': {'type': 'integer', 'enum': sorted(n for c in clusters for n in c)},
                'strength': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 1}},
                'required': ['customer_id', 'strength'], 'additionalProperties': False}}},
            'required': ['advance'], 'additionalProperties': False}}}
    if '/' in model:
        prompt += ('\nAPI wire format override: return {"advance":['
                   '{"customer_id":42,"strength":0.5}]} with only selected customers. '
                   '42 and 0.5 illustrate syntax only, not a recommendation. '
                   'No intervention is {"advance":[]}. Do not repeat customer IDs. '
                   'The application losslessly converts this list to the ID-score cache map.\n')
        (out / 'prompt.txt').write_text(prompt, encoding='utf-8')
        metadata['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
    metadata['response_format'] = response_format if '/' in model else None
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    response = query_llm(prompt, model, think=think, num_ctx=num_ctx,
                         response_format=metadata['response_format'])
    (out / 'response.txt').write_text(response, encoding='utf-8')
    parsed_response = response
    if '/' in model:
        wire = strict_json(response)
        if not isinstance(wire, dict) or set(wire) != {'advance'} or not isinstance(wire['advance'], list):
            raise ValueError('Expected structured advance array')
        mapping = {}
        for item in wire['advance']:
            if not isinstance(item, dict) or set(item) != {'customer_id', 'strength'} or type(item['customer_id']) is not int:
                raise ValueError('Invalid structured intervention')
            key = str(item['customer_id'])
            if key in mapping:
                raise ValueError('Duplicate selected customer')
            mapping[key] = item['strength']
        parsed_response = json.dumps({'advance': mapping})
    scores = parse_advance_response(parsed_response, [n for c in clusters for n in c])
    cache = dict(metadata, advance={str(n): s for n, s in scores.items()})
    temporary = out / 'cache.pending.json'
    temporary.write_text(json.dumps(cache, indent=2, allow_nan=False), encoding='utf-8')
    temporary.rename(out / 'cache.json')
    return cache


def load_advance_cache(path, *, instance, customer_ids):
    data = strict_json(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != SCHEMA_VERSION or data.get('instance') != instance:
        raise ValueError("Selective advance cache schema/instance mismatch")
    if data.get('application_scope') != APPLICATION_SCOPE:
        raise ValueError("Cache must use active-route cluster scope")
    _validate_clusters(data.get('clusters', []), set(customer_ids))
    return parse_advance_response(json.dumps({'advance': data['advance']}), customer_ids)


def _validate_clusters(clusters, expected):
    nodes = [n for cluster in clusters for n in cluster]
    if (any(type(n) is not int for n in nodes) or len(nodes) != len(set(nodes))
            or set(nodes) != expected or 0 in nodes):
        raise ValueError('Clusters must partition the customer IDs exactly once')


def ensure_trainer_cache(trainer, inst, *, force_refresh=False):
    """Cache-only adapter; reuse existing anchor and per-rollout ACC stage selection."""
    from train_vrptw_llm import _get_accidents
    if force_refresh:
        raise ValueError('Selective advance caches are immutable; live refresh prohibited')
    name = inst['name']
    if name in trainer._cluster_conf_cache:
        return
    root = Path(trainer._llm_cache_dir)
    n = inst['n_customers']
    is_acc = 'ACC' in name.upper()
    base_name = name.split('_')[0] if is_acc else name

    def read(filename, expected_name):
        path = root / filename
        scores = load_advance_cache(path, instance=expected_name, customer_ids=range(1, n+1))
        data = strict_json(path.read_text(encoding='utf-8'))
        tensor = advance_bias(scores, n, clusters=data['clusters'], device=trainer.device)
        return data, tensor

    base, tensor = read(f'{base_name}_advance.json', base_name)
    if base.get('events_visible_until') is not None:
        raise ValueError('Pre-event cache cannot contain accident guidance')
    stages = []
    if is_acc:
        for t in sorted({float(f['t_start']) for _, f in _get_accidents(inst)}):
            wave, wave_tensor = read(f'{name}_wave_{t:.6f}_advance.json', name)
            if wave['clusters'] != base['clusters'] or wave.get('events_visible_until') != t:
                raise ValueError('ACC wave membership/time differs from reference')
            stages.append({'t_start': t, 'tensor': wave_tensor})
    trainer._cluster_assign_cache[name] = base['clusters']
    trainer._cluster_conf_cache[name] = tensor
    trainer._original_cluster_conf[name] = tensor.clone()
    trainer._cluster_stats_cache[name] = {}
    trainer._acc_stage_events[name] = stages
    trainer._acc_precompute_done.add(name)
    print(f'[Selective cache] {name}: clusters={len(base["clusters"])} selected={len(base["advance"])} waves={len(stages)}', flush=True)


def advance_bias(scores, n_customers, *, clusters, strength=1.0, device=None):
    """K x (N+1) rows for existing cluster selection, never a global bonus.

    Consumer must select the anchored row, gate it with the existing enabled
    mask, and retain decoder feasibility masks. This helper does not alter starts.
    """
    import torch
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("Bias strength must be finite and nonnegative")
    validated = parse_advance_response(
        json.dumps({'advance': {str(n): s for n, s in scores.items()}}),
        range(1, n_customers + 1))
    _validate_clusters(clusters, set(range(1, n_customers + 1)))
    result = torch.zeros((len(clusters), n_customers + 1), device=device)
    for k, nodes in enumerate(clusters):
        for node in nodes:
            result[k, node] = strength * validated.get(node, 0.0)
    return result
