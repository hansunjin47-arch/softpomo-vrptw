"""Independent audit of saved v4 LLM packets; no API calls or prompt rewrites."""
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from audit_event_relations import ROOT, read, sha, load_solomon
from routing_ontology import validate


def records(facts, name, columns):
    header = facts[columns]
    if len(header) != len(set(header)):
        raise ValueError('Duplicate column')
    rows = facts[name]
    if any(len(r) != len(header) for r in rows):
        raise ValueError('Column width mismatch')
    return [dict(zip(header, r)) for r in rows]


def main():
    root = ROOT/'result/rc101_ontology_train20_v4'
    legacy = ROOT/'result/rc101_relation_train20_cache_v1'
    protocol = read(root/'protocol.json')
    original = read(ROOT/'result/rc101_pure500_fullconfidence_phase2/protocol.json')
    counts, counterexamples = Counter(), Counter()
    cases, gaps = [], []
    # The check below operates on what the LLM actually receives, not the graph builder.
    for name in protocol['train_names']:
        packet_path = root/'requests'/name/'context.json'
        packet = read(packet_path)
        request = read(root/'requests'/name/'request.json')
        content = request['messages'][0]['content']
        assert content.endswith(json.dumps(packet,separators=(',',':'),allow_nan=False))
        assert sha(packet_path)==protocol['output_sha256'][str(packet_path.relative_to(root))]
        facts = packet['facts']
        legs = records(facts,'legs','leg_columns')
        visits = records(facts,'visits','visit_columns')
        diagnoses = records(facts,'diagnoses','diagnosis_columns')
        by_leg = {r['id']:r for r in legs}
        by_visit = {r['id']:r for r in visits}
        assert len(by_leg)==len(legs)==len(by_visit)==len(visits)==len(diagnoses)
        groups = packet['instance']['groups']
        assert sorted(c for g in groups for c in g['reference_order'])==list(range(1,101))
        expected_visits = []
        route_legs = {}
        for group in groups:
            route = group['route']
            selected = [l for l in legs if l['route']==route]
            route_legs[route] = selected
            route_nodes = [0]+group['reference_order']+[0]
            assert [(l['origin'],l['destination']) for l in selected]==list(zip(route_nodes,route_nodes[1:]))
            assert abs(selected[0]['reference_departure'])<1e-6
            assert abs(selected[0]['base_departure'])<1e-6
            expected_visits.extend(l['id']+':visit' for l in selected)
        assert set(by_visit)==set(expected_visits)
        types = {'event':'RainEvent' if facts['event']['event_type']=='RAIN' else 'AccidentEvent',
                 'active':'TimeInterval', 'c0':'Depot'}
        types.update({f'c{i}':'Customer' for i in range(1,101)})
        types.update({r:'ReferenceRoute' for r in route_legs})
        types.update({r:'Region' for r in ('N','NE','E','SE','S','SW','W','NW')})
        types.update({r:'ScheduledLeg' for r in by_leg})
        types.update({r:'ServiceVisit' for r in by_visit})
        for subject,predicate,obj in facts['relations']:
            signature = packet['ontology']['relations'][predicate]
            assert types[subject] in signature['domain'] and types[obj] in signature['range']
        for d in diagnoses:
            v = by_visit[d['visit']]
            l = by_leg[v['incoming_leg']]
            assert v['location']==f'c{l["destination"]}' and v['route']==l['route']
            event = facts['event']
            exposed = (l['origin'] in event['affected_nodes'] or l['destination'] in event['affected_nodes']) and event['start_min']<=l['reference_departure']<event['end_min']
            # Display-rounding tolerance is separate from the unrounded builder tolerance.
            arrival_delay = l['reference_arrival']-l['base_arrival']
            assert abs(arrival_delay-l['incoming_delay']-l['added_travel'])<3e-6
            assert abs(l['outgoing_delay']-arrival_delay+l['wait_absorption'])<3e-6
            expected = dict(direct_exposure=exposed,
                indirect_only=arrival_delay>1e-6 and not exposed,
                fully_absorbed_here=arrival_delay>1e-6 and abs(l['outgoing_delay'])<=1e-6,
                preexisting_lateness=l['base_late']>1e-6,
                event_added_lateness=l['reference_late']-l['base_late']>1e-6,
                new_late_visit=l['base_late']<=1e-6 and l['reference_late']>1e-6)
            assert all(d[k]==value for k,value in expected.items()), (name,v['id'],d,expected)
            counts.update(k for k,v in expected.items() if v)
            counterexamples['exposed_but_not_late'] += exposed and l['reference_late']<=1e-6
            counterexamples['unexposed_but_event_added_lateness'] += not exposed and expected['event_added_lateness']
            counterexamples['late_without_event_increment'] += expected['preexisting_lateness'] and not expected['event_added_lateness']
        scope = packet['reference_scope']
        assert scope['scenario']==name and scope['checkpoint_sha256']==protocol['reference_checkpoint_sha256']
        counts['visits'] += len(visits)
        counts['routes'] += len(groups)
        counts['scenarios'] += 1
        cases.append(dict(name=name,visits=len(visits),routes=len(groups)))

    # Adversarial checks of the standalone validator, separate from source replay.
    data = Path(original['source_config']['trainer']['data_dir'])
    name = next(n for n in protocol['train_names'] if '_ACC_' in n)
    graph = read(legacy/'requests'/name/'relations.json')
    inst = load_solomon(str(data/(name+'.txt')))
    coords = np.vstack([inst['depot_xy'].numpy(),inst['node_xy'].numpy()])*inst['max_coord']
    def probe(label, mutate):
        candidate = copy.deepcopy(graph)
        mutate(candidate)
        try:
            validate(candidate,coords)
        except (ValueError,KeyError,TypeError):
            accepted = False
        else:
            accepted = True
        gaps.append(dict(probe=label,standalone_validator_accepts=accepted,
            production_builder='Saved-source replay equality check rejects this mutation; not observed in prepared v4 packets.'))
    def invalid_event(g):
        g['nodes'][0]['public_observation']['event_type']='UNKNOWN_EVENT'
    def invalid_travel(g):
        first = g['legs'][0]
        shift = max(first['reference']['travel'],first['baseline']['travel'])+1
        for schedule in ('reference','baseline'):
            first[schedule]['travel'] -= shift
    probe('unknown event type treated as AccidentEvent',invalid_event)
    probe('negative travel times preserving travel differences',invalid_travel)
    result = dict(scope='Independent saved-prompt semantic audit; not OWL consistency proof or LLM efficacy evaluation',
        status='All saved train20 packets passed; standalone validator hardening gaps remain',
        packet_checks=cases, counts=dict(counts), counterexamples=dict(counterexamples),
        standalone_probes=gaps,
        conceptual_gaps=[
            'ReferenceGroup is a class but membership and reference-route associations are JSON fields, not declared ontology predicates.',
            'ServiceVisit and ReferenceRoute IDs are local to the enclosing scenario; merging packets requires scenario-qualified IDs.',
            'Competency questions are diagnostic. They do not establish a ranking-to-cost implication.',
            'Python computations and source checks, not ontology entailment alone, establish numerical facts.',
            'Only the stated one-event fixed-network domain is validated; generalisation is not established.'],
        input_protocol_sha256=sha(root/'protocol.json'))
    output = ROOT/'analysis/ontology_research_validation_v4.json'
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='packet_checks'},indent=2))


if __name__=='__main__':
    main()
