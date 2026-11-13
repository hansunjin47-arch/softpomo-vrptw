"""Train-only pilot: append reference relations to original Pure500 confidence prompts."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from audit_event_relations import ROOT, analyze, load_solomon, read, sha
from run_rc101_kim_conversation import decode
from run_fixed_kim_sonnet import backend, MODEL, save

RULES = '''
REFERENCE RELATION SUPPLEMENT v1 (not a live rollout state):
Direct exposure requires a zone endpoint AND departure in [event start,event end).
An unexposed leg can receive delay from preceding legs. Waiting can absorb delay.
Exposure alone does not imply urgent service. Consider downstream customers and depot returns.
The following quantities are simulator-derived reference feedback, NOT inferred from public news.
Counterfactual quantities replay the SAME selected routes without the event; they are not a new plan.
Do not treat reference relations as guaranteed after the policy changes the route.
Use these relations as evidence when assigning the existing within-group confidence scores.
Do not output route edits or change group membership. No reassignment recommendation is encoded here.
Confidence is a soft additive preference, not a hard rejection threshold or a probability of feasibility.
For accidents this cached feedback is offline reference information; guidance remains disabled before disclosure.
Keep the existing confidence-only JSON schema. This supplement supplies no optimized repair or cost oracle.
'''


def supplement(audit):
    return dict(provenance='simulator_reference_vs_same_route_event_removed',
        routes=[dict(route=node['id']) for node in audit['nodes'] if node['type'] == 'ReferenceRoute'],
        legs=[dict(id=x['id'], route=x['route'], origin=x['origin'], destination=x['destination'],
            reference_departure_min=round(x['reference']['departure'], 4),
            direct_exposure=x['reference_exposure']['exposed'],
            inherited_delay_min=round(x['incoming_delay'], 4),
            added_travel_min=round(x['added_travel'], 4),
            wait_absorption_min=round(x['wait_absorption'], 4),
            forwarded_delay_min=round(x['outgoing_delay'], 4),
            reference_lateness_min=round(x['reference']['late'], 4),
            added_lateness_min=round(x['added_lateness'], 4)) for x in audit['legs']
              if x['reference_exposure']['exposed'] or x['arrival_delay'] > 1e-7],
        relations=[r for r in audit['relations'] if r['predicate'] in
                   ('transmitsDelayTo', 'delayAbsorbedAt', 'increasesLatenessAt')])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--call', action='store_true')
    ap.add_argument('--out', type=Path, default=ROOT / 'result/rc101_relation_train20_cache_v1')
    args = ap.parse_args()
    source = ROOT / 'result/rc101_pure500_fullconfidence_phase2'
    prior = read(source / 'protocol.json')
    data = Path(prior['source_config']['trainer']['data_dir'])
    split_path = Path(prior['checkpoint']).parents[2] / 'provenance/split.json'
    split = read(split_path)
    chosen = split['train']
    assert len(chosen) == 20
    assert not {x['name'] for x in chosen} & {x['name'] for x in split['test']}
    assert [x['name'] for x in chosen] == prior['train_names']
    assert sum(x['kind'] == 'RAIN' for x in chosen) == 10
    assert sum(x['kind'] == 'ACCIDENT' for x in chosen) == 10
    assert sha(Path(prior['checkpoint'])) == prior['checkpoint_sha256']
    protocol = dict(version='reference_relations_v1', model=MODEL, train_names=[x['name'] for x in chosen],
        selection='Original frozen train20: 10 RAIN and 10 ACCIDENT; original train20 Pure500 checkpoint',
        reference_checkpoint_sha256=prior['checkpoint_sha256'], original_reference_run=str(source),
        test_calls=False, training_started=False, original_prompt_plus_relations=True,
        information='Additional simulator-derived no-event counterfactual feedback; not information-matched ontology ablation',
        prompt_script_sha256=sha(Path(__file__)), relation_script_sha256=sha(ROOT / 'analysis/audit_event_relations.py'))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        assert read(out / 'protocol.json') == protocol
    save(out / 'protocol.json', protocol)
    (out / 'cache').mkdir(exist_ok=True)
    provenance = out / 'provenance'
    provenance.mkdir(exist_ok=True)
    for path in [Path(__file__), ROOT / 'analysis/audit_event_relations.py', split_path]:
        shutil.copy2(path, provenance / path.name)
    comparison, prepared = [], []
    for index, entry in enumerate(chosen):
        name = entry['name']
        path = data / (name + '.txt')
        assert sha(path) == entry['sha256']
        record = out / 'requests' / name
        record.mkdir(parents=True, exist_ok=True)
        reference = read(source / 'requests' / name / 'reference.json')
        assert reference['checkpoint_sha256'] == prior['checkpoint_sha256']
        clusters = reference['clusters']
        assert sorted(n for c in clusters for n in c) == list(range(1, 101))
        routes = [[0] + c + [0] for c in clusters]
        audit, _ = analyze(load_solomon(str(path)), routes, read(data / (name + '.observations.json')))
        request = copy.deepcopy(read(source / 'requests' / name / 'request.json'))
        assert request['model'] == MODEL and len(request['messages']) == 1
        request['messages'][0]['content'] += RULES + '\n' + json.dumps(supplement(audit), separators=(',', ':'))
        save(record / 'reference.json', reference)
        save(record / 'relations.json', audit)
        if (record / 'request.json').exists():
            assert read(record / 'request.json') == request
        save(record / 'request.json', request)
        prepared.append(dict(name=name, instance_sha256=sha(path),
            observation_sha256=sha(data / (name + '.observations.json')),
            reference_sha256=sha(source / 'requests' / name / 'reference.json'),
            original_request_sha256=sha(source / 'requests' / name / 'request.json'),
            request_sha256=sha(record / 'request.json')))
        if not args.call:
            continue
        save(out / 'pipeline.json', dict(stage='calling', completed=index, total=20, name=name))
        response_path = record / 'response.txt'
        if response_path.exists():
            response = response_path.read_text(encoding='utf-8')
        else:
            print(f'CALL {index+1}/20 {name}', flush=True)
            response = backend.query_llm(request['messages'][0]['content'], MODEL,
                                         response_format=request['response_format'])
            response_path.write_text(response, encoding='utf-8')
        _, cache = decode(response, clusters)
        save(out / 'cache' / (name + '_rl_cluster.json'), cache)
        old = read(source / 'cache' / (name + '_rl_cluster.json'))
        differences = [abs(cache[str(k)][str(n)] - old[str(k)][str(n)])
                       for k, c in enumerate(clusters) for n in c]
        comparison.append(dict(name=name, customer_mae=sum(differences)/100,
            changed_customers=sum(d > 1e-8 for d in differences),
            nonzero_depots=sum(v > 0 for v in cache['_depot'].values())))
        print(f'VALIDATED {index+1}/20 {name}', flush=True)
    save(out / 'input_hashes.json', prepared)
    if args.call:
        save(out / 'comparison.json', dict(scenarios=comparison,
            customer_mae=sum(x['customer_mae'] for x in comparison)/20,
            changed_customers=sum(x['changed_customers'] for x in comparison),
            nonzero_depots=sum(x['nonzero_depots'] for x in comparison)))
    save(out / 'pipeline.json', dict(stage='cache_complete' if args.call else 'prepared',
        completed=20, training_started=False, test_calls=False))
    print('DONE: ' + ('20 validated caches' if args.call else '20 prompts prepared; no calls'))


if __name__ == '__main__':
    main()
