"""Versioned offline train feedback. Does not alter legacy or test prompts."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

VERSION = 'concurrent_train_execution_feedback_v2'
INSTRUCTIONS = '''
This is OFFLINE TRAINING guidance, not a live routing decision.
The reference routes and execution outcomes come from a separate frozen Pure RL
policy evaluated on training data. These are reference-policy outcomes, not measured
outcomes of the recipient policy. Guidance must not assume a particular initialization.
Use the supplied objective, routes and measured outcomes to inform customer scores.
The reference is neither an optimum nor a target to imitate exactly. A late visit
does not prove that increasing its score or changing its order improves the objective.
Do not claim a measured improvement without an evaluated alternative.
Execution feedback is historical training evidence, NOT a current fleet snapshot.
Initial event-free feedback contains no future accident execution. Post-disclosure
feedback is privileged offline evidence and its scores activate only after disclosure.
Do not infer hidden event multipliers or invent counterfactual execution outcomes.
Customer arrival/service times are minutes; destination 0 denotes depot return.
Reference vehicle IDs are not the recipient policy's future vehicle assignments.
The reference routes can differ from the fixed guidance groups. Preserve the supplied
output groups and schema; do not change groups or output routes.
'''
METRICS = ('reward', 'vehicles', 'travel_time', 'total_late', 'late_count',
           'served', 'unserved', 'cost_vehicle', 'cost_travel', 'cost_late',
           'cost_unserved', 'cost_distance', 'cost_late_count')


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Feedback requires finite numeric values')
    return value


def build_request(original, reference, *, reference_id, event_free=False):
    from experiments.concurrent_fleet_v9_guided.prompts import COMMON
    request = copy.deepcopy(original)
    context = json.loads(request['messages'][1]['content'])
    stage = context.get('stage')
    if stage not in ('offline_train_initial', 'offline_train_post_disclosure'):
        raise ValueError('Execution feedback is train-only')
    routes = [[int(n) for n in route if n] for route in reference['routes']]
    if routes != context['reference_routes']:
        raise ValueError('Reference routes do not match the original request')
    metrics = {key: number(reference['metrics'][key]) for key in METRICS}
    cost = sum(value for key, value in metrics.items() if key.startswith('cost_'))
    if not math.isclose(metrics['reward'], -cost, abs_tol=2e-5, rel_tol=1e-5):
        raise ValueError('Reward does not match cost components')
    fixed = context['fixed_instance']
    columns = fixed['customer_columns']
    closes = {row[columns.index('id')]: row[columns.index('close_min')]
              for row in fixed['customers']}
    steps = []
    traced = {}
    for step in reference['steps']:
        vehicle, dest = step['vehicle'], step['destination']
        if type(vehicle) is not int or vehicle < 0 or type(dest) is not int or dest not in closes:
            raise ValueError('Invalid reference vehicle or destination')
        arrival = number(step['arrival_min'])
        end = number(step['service_end_min'])
        if arrival < 0 or end < arrival:
            raise ValueError('Invalid reference timing')
        traced.setdefault(vehicle, []).append(dest)
        steps.append(dict(vehicle=vehicle, destination=dest, arrival_min=arrival,
                          service_end_min=end, lateness_min=max(0., arrival-closes[dest])))
    if list(traced.values()) != reference['routes']:
        raise ValueError('Step trace and routes disagree')
    if not math.isclose(sum(s['lateness_min'] for s in steps), metrics['total_late'], abs_tol=.01):
        raise ValueError('Visit lateness does not match total lateness')
    context['protocol'] = VERSION
    context['train_execution_feedback'] = dict(
        reference_id=reference_id, recipient_initialization='experimental_arm_dependent',
        information_scope='offline_train_only', event_free=event_free,
        metrics=metrics, visits=steps)
    request['messages'] = [dict(role='system', content=COMMON+INSTRUCTIONS),
                           dict(role='user', content=json.dumps(context, separators=(',', ':'), allow_nan=False))]
    return request


def prepare(source, destination):
    """Build requests from audited saved train references; no GPU or API calls."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('Use a new destination; existing caches are immutable')
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))
    split = read(source/'split.json')
    pending = {}
    hashes = {}
    for row in split['train']:
        name = row['name']
        folder = source/'requests'/name
        future = (folder/'post_disclosure.json').exists()
        for stage in ['initial'] + (['post_disclosure'] if future else []):
            refpath = (source/'event_free_train_reference.json' if future and stage == 'initial'
                       else source/'train_references'/(name+'.json'))
            requestpath = folder/(stage+'.json')
            pending[Path('requests')/name/(stage+'.json')] = build_request(
                read(requestpath), read(refpath), reference_id=str(refpath),
                event_free=future and stage == 'initial')
            for path in (refpath, requestpath):
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('groups.json', 'config.json', 'split.json', 'request_template.json'):
        pending[Path(name)] = read(source/name)
    pending[Path('protocol.json')] = dict(version=VERSION, source=str(source),
        source_hashes=hashes, requests=len(pending)-4, paid_calls=0,
        recipient_initialization='experimental_arm_dependent', training_started=False,
        test_policy='unchanged causal test builder; no execution feedback')
    for path, value in pending.items():
        target = destination/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Prepared {len(pending)-5} train requests; no API calls or training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.destination)
