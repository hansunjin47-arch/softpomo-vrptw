"""Typed relation audit of fixed Pure500 routes; no LLM calls or route changes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats

EPS = 1e-7


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exposure(origin, destination, departure, observation):
    spatial = origin in observation['affected_nodes'] or destination in observation['affected_nodes']
    temporal = observation['start_min'] <= departure < observation['end_min']
    return dict(spatial=spatial, temporal=temporal, exposed=spatial and temporal)


def timeline(vehicle, stats):
    rows, departure = [], 0.0
    for stop in vehicle['stop_details']:
        rows.append(dict(origin=stop['prev'], destination=stop['node'], departure=departure,
                         arrival=stop['arrival'], wait=stop['wait'], next_departure=stop['depart'],
                         travel=stop['travel_time'], late=stop['late']))
        departure = stop['depart']
    rows.append(dict(origin=vehicle['prev_last'], destination=0, departure=departure,
                     arrival=vehicle['arr_depot'], wait=0.0, next_departure=vehicle['arr_depot'],
                     travel=vehicle['arr_depot'] - departure, late=vehicle['ret_late']))
    return rows


def analyze(inst, routes, observation):
    actual = _solution_stats(inst, routes)
    base = _solution_stats(dict(inst, preset_events=[]), routes)
    if len(inst['preset_events']) != 1:
        raise ValueError('Only single-event scenarios supported')
    event = inst['preset_events'][0]
    assert set(event['nodes']) == set(observation['affected_nodes'])
    assert abs(event['trigger_time'] - observation['start_min']) < EPS
    assert abs(event['trigger_time'] + event['duration'] - observation['end_min']) < EPS
    nodes, relations, legs = [], [], []
    def node(identifier, kind, **properties):
        nodes.append(dict(id=identifier, type=kind, **properties))
    def edge(subject, predicate, obj, **properties):
        relations.append(dict(subject=subject, predicate=predicate, object=obj, **properties))
    node('event', 'RainEvent' if observation['event_type'] == 'RAIN' else 'AccidentEvent',
         public_observation=observation, source='public_sidecar')
    node('active', 'TimeInterval', start=observation['start_min'], end=observation['end_min'],
         boundary='start inclusive, end exclusive', unit='minutes after dispatch')
    edge('event', 'activeDuring', 'active')
    # Region membership comes from fixed coordinates, not the affected-node list.
    import math
    region_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    for region in region_names:
        node(region, 'Region')
        if region in observation['regions']:
            edge('event', 'affectsRegion', region)
    for customer in range(inst['n_customers'] + 1):
        node(f'c{customer}', 'Depot' if customer == 0 else 'Customer')
        if customer:
            x, y = actual['coords'][customer] - actual['coords'][0]
            sector = int(((math.pi / 2 - math.atan2(y, x) + math.pi / 8) % (2 * math.pi)) // (math.pi / 4))
            region = region_names[sector]
            edge(f'c{customer}', 'locatedIn', region)
            assert (region in observation['regions']) == (customer in observation['affected_nodes'])
    for k, (av, bv) in enumerate(zip(actual['per_vehicle'], base['per_vehicle']), 1):
        route_id = f'r{k}'
        node(route_id, 'ReferenceRoute', source='saved_Pure500_scenario_selected_route')
        previous = None
        for index, (a, b) in enumerate(zip(timeline(av, actual), timeline(bv, base))):
            identifier = f'{route_id}:leg{index}'
            actual_exposure = exposure(a['origin'], a['destination'], a['departure'], observation)
            base_exposure = exposure(b['origin'], b['destination'], b['departure'], observation)
            increment = a['travel'] - b['travel']
            arrival_delay = a['arrival'] - b['arrival']
            incoming_delay = a['departure'] - b['departure']
            absorbed = b['wait'] - a['wait']
            outgoing = a['next_departure'] - b['next_departure']
            late_increment = a['late'] - b['late']
            assert abs(arrival_delay - incoming_delay - increment) < EPS
            assert abs(outgoing - arrival_delay + absorbed) < EPS
            assert min(increment, arrival_delay, absorbed, late_increment) > -EPS
            if b['travel'] > EPS:
                assert actual_exposure['exposed'] == (increment > EPS)
            row = dict(id=identifier, route=route_id, origin=a['origin'], destination=a['destination'],
                baseline=b, reference=a, baseline_exposure=base_exposure, reference_exposure=actual_exposure,
                incoming_delay=incoming_delay, added_travel=increment, arrival_delay=arrival_delay,
                wait_absorption=absorbed, outgoing_delay=outgoing, added_lateness=late_increment,
                new_late_visit=b['late'] <= EPS and a['late'] > EPS,
                indirect_only=arrival_delay > EPS and increment <= EPS)
            legs.append(row)
            node(identifier, 'ScheduledLeg', route_order=index, origin=a['origin'], destination=a['destination'],
                 reference_departure=a['departure'], baseline_departure=b['departure'],
                 timing_source='simulator_reference_and_event_removed_counterfactual')
            edge(identifier, 'partOf', route_id)
            edge(identifier, 'from', f'c{a["origin"]}')
            edge(identifier, 'to', f'c{a["destination"]}')
            if previous:
                edge(previous, 'precedes', identifier)
                if incoming_delay > EPS:
                    edge(previous, 'transmitsDelayTo', identifier, minutes=incoming_delay, source='simulator_audit')
            if actual_exposure['exposed']:
                edge(identifier, 'exposedTo', 'event', schedule='realised_reference', source='public_rule_plus_simulated_timing')
            if absorbed > EPS:
                edge(identifier, 'delayAbsorbedAt', f'c{a["destination"]}', minutes=absorbed, source='simulator_audit')
            if late_increment > EPS:
                edge(identifier, 'increasesLatenessAt', f'c{a["destination"]}', minutes=late_increment, source='simulator_audit')
            previous = identifier
    summary = dict(legs=len(legs), spatial_legs=sum(x['reference_exposure']['spatial'] for x in legs),
        exposed_legs=sum(x['reference_exposure']['exposed'] for x in legs),
        baseline_exposed_legs=sum(x['baseline_exposure']['exposed'] for x in legs),
        exposure_changed=sum(x['reference_exposure']['exposed'] != x['baseline_exposure']['exposed'] for x in legs),
        indirectly_delayed_visits=sum(x['indirect_only'] for x in legs),
        wait_absorption=sum(x['wait_absorption'] for x in legs),
        added_travel=actual['total_travel_time'] - base['total_travel_time'],
        baseline_lateness=base['total_late'], reference_lateness=actual['total_late'],
        added_lateness=actual['total_late'] - base['total_late'],
        new_late_visits=sum(x['new_late_visit'] for x in legs),
        added_customer_lateness=sum(x['added_lateness'] for x in legs if x['destination']),
        added_depot_lateness=sum(x['added_lateness'] for x in legs if not x['destination']))
    assert abs(summary['added_travel'] - sum(x['added_travel'] for x in legs)) < EPS
    assert abs(summary['added_travel'] - summary['wait_absorption'] -
               sum(x['outgoing_delay'] for x in legs if x['destination'] == 0)) < EPS
    return dict(summary=summary, nodes=nodes, relations=relations, legs=legs), actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'analysis/pure500_rain_relation_audit')
    args = ap.parse_args()
    data = ROOT / 'data/rc101_train80_test80_20260929'
    source = ROOT / 'result/rc101_pure_train80_fresh1000_20260929/eval_500'
    split = read(data / 'split.json')
    entries = {e['name']: e for e in split['test']}
    saved = read(source / 'results.json')
    assert [r['name'] for r in saved] == [e['name'] for e in split['test']]
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    rows, hashes = [], {}
    for row in saved:
        if row['kind'] != 'RAIN':
            continue
        name = row['name']
        path = data / (name + '.txt')
        assert sha(path) == entries[name]['sha256']
        obs_path = data / (name + '.observations.json')
        inst = load_solomon(str(path))
        result, actual = analyze(inst, row['routes'], read(obs_path))
        assert abs(actual['total_travel_time'] - row['travel_time']) < EPS
        assert abs(actual['total_late'] - row['total_late']) < EPS
        assert actual['n_served'] == 100
        hashes[name] = dict(instance=sha(path), observation=sha(obs_path))
        (out / (name + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
        rows.append(dict(name=name, **result['summary']))
    summary = dict(scenarios=len(rows), totals={key: sum(r[key] for r in rows) for key in rows[0] if key != 'name'},
        scenarios_with_exposure=sum(r['exposed_legs'] > 0 for r in rows),
        exposed_without_added_lateness=sum(r['exposed_legs'] > 0 and r['added_lateness'] <= EPS for r in rows),
        scenarios_with_added_lateness=sum(r['added_lateness'] > EPS for r in rows),
        scenarios_with_changed_exposure=sum(r['exposure_changed'] > 0 for r in rows),
        worst=sorted(rows, key=lambda r: r['added_lateness'], reverse=True)[:5])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'scenarios.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (out / 'provenance.json').write_text(json.dumps(dict(reference=read(source / 'summary.json'),
        results_sha256=sha(source / 'results.json'), script_sha256=sha(Path(__file__)), inputs=hashes,
        scope='RAIN test40 exploratory audit; fixed routes, no retraining or LLM calls',
        implementation='Typed JSON relation graph, not an OWL reasoner',
        information='Exact delay and lateness differences use simulator effects; not public-only estimates',
        counterfactual='Same scenario-selected routes replayed without events; not an independently optimized baseline'), indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
