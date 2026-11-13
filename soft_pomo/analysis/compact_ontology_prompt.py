"""Versioned prompt-only compaction; no API calls or active-cache mutation."""
import copy
import hashlib
import json
from pathlib import Path

VERSION = 'ontology_v5_compact_v1'
INSTRUCTIONS = '''Assign conservative static customer/group and depot confidence from the reference evidence.
Minimize J=0.04*K+T/5760+0.0075*Lt: K=vehicles, T=driving minutes including returns,
Lt=total customer and depot lateness minutes. This is a weighted sum, not lateness-first.
Capacity is hard; closing times are soft. Serve all customers; unserved demand has a large penalty.
Confidence in [0,1] is a soft additive decoder bias with beta=5, not a probability or hard cutoff.
After the first customer, its reference group remains active until depot return. Before that start no bias applies.
Outsiders have zero bias but remain selectable. Depot competes softly; scores do not encode reassignment.
Each vehicle starts at time zero; accident bias activates only after its own disclosure time.
Relative action odds are multiplied by exp(5*(c_i-c_j)). Equal positive group scores favor members over outsiders;
0.5 is not neutral. All-zero scores disable additive guidance; equal scores cancel only against equal-biased actions.
Python supplies checked reference facts, you supply static preferences, and RL selects time/state-dependent actions.
Use direct exposure, inherited delay, waiting absorption and pre-existing versus added lateness jointly.
Neither exposure nor lateness mandates a higher score. Slack does not certify a safe move; changing order changes
downstream timing and exposure. Full absorption stops inherited delay here, not subsequent exposure or existing lateness.
Prefer modest scores when evidence does not support a robust direction. No alternative-route search, guaranteed repair,
or measured edit improvement is supplied. Do not invent one.
Schedules are rounded exact offline simulator feedback, not public-only estimates or the live RL state.
Event/base travel ratios can reveal hidden multipliers. Relations apply only to the identified fixed schedule pair.
Return ONLY the supplied confidence JSON schema: one {cluster,customer,score} for every group member and depot 0
per group, with one-based cluster IDs. No extra pairs, explanations, group changes or route edits.
'''


def dumps(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def compact(payload):
    p = copy.deepcopy(payload)
    sidecar = {}
    f = p['facts']
    cols = f['leg_columns']
    legs = [dict(zip(cols, row)) for row in f['legs']]
    expected = [[r['id'] + ':visit', r['route'], r['id'], 'c' + str(r['destination'])] for r in legs]
    if f['visits'] != expected:
        raise ValueError('Visit table is not redundant with leg table')
    del f['visits']
    diagnoses = f.pop('diagnoses')
    if [r[0] for r in diagnoses] != [r[0] for r in expected]:
        raise ValueError('Diagnosis/leg order mismatch')
    if any(type(v) is not bool for row in diagnoses for v in row[1:]):
        raise ValueError('Sparse encoding requires explicit Boolean values')
    f['positive_diagnoses'] = {name: [legs[i]['id'] for i, row in enumerate(diagnoses) if row[k+1]]
                               for k, name in enumerate(f['diagnosis_columns'][1:])}
    f['compact_visit_rule'] = 'Each leg has exactly one visit: id=leg.id+":visit", route=leg.route, incoming_leg=leg.id, location="c"+leg.destination. Diagnosis lists identify legs reaching positive visits. All unlisted leg/diagnosis pairs are false, not unknown. All legs remain present.'
    grouped = {}
    for subject, predicate, obj in f.pop('relations'):
        grouped.setdefault(predicate, []).append([subject, obj])
    f['relations_by_predicate'] = grouped
    f['compact_relation_rule'] = 'For each predicate, every [subject,object] pair is an explicit relation with the ontology-declared meaning and provenance.'
    for group in p['instance']['groups']:
        if group['required_output_customers'] != [0] + group['reference_order']:
            raise ValueError('Output membership is not redundant')
        del group['required_output_customers']
    p['instance']['compact_output_rule'] = 'Each group requires output for depot 0 and every customer in reference_order, exactly once.'
    # Validation questions and cryptographic identifiers are retained in local provenance.
    sidecar['competency_questions'] = p['ontology'].pop('competency_questions')
    sidecar['task'] = p.pop('task')
    sidecar['reference_hashes'] = {k:p['reference_scope'].pop(k) for k in list(p['reference_scope']) if k.endswith('_sha256')}
    p['prompt_version'] = VERSION
    return p, sidecar


def expand(packed, sidecar):
    p = copy.deepcopy(packed)
    del p['prompt_version']
    f = p['facts']
    legs = [dict(zip(f['leg_columns'], row)) for row in f['legs']]
    ids = {r['id'] for r in legs}
    positives = f.pop('positive_diagnoses')
    if set(positives) != set(f['diagnosis_columns'][1:]):
        raise ValueError('Diagnosis columns changed')
    if any(len(v) != len(set(v)) or not set(v) <= ids for v in positives.values()):
        raise ValueError('Unknown or duplicate positive diagnosis leg')
    f['visits'] = [[r['id']+':visit',r['route'],r['id'],'c'+str(r['destination'])] for r in legs]
    f['diagnoses'] = [[r['id']+':visit'] + [r['id'] in positives[k] for k in f['diagnosis_columns'][1:]] for r in legs]
    f['relations'] = [[subject, pred, obj] for pred, pairs in f.pop('relations_by_predicate').items() for subject, obj in pairs]
    del f['compact_visit_rule'], f['compact_relation_rule']
    for group in p['instance']['groups']:
        group['required_output_customers'] = [0] + group['reference_order']
    del p['instance']['compact_output_rule']
    p['ontology']['competency_questions'] = copy.deepcopy(sidecar['competency_questions'])
    p['task'] = sidecar['task']
    p['reference_scope'].update(sidecar['reference_hashes'])
    return p


def canonical(p):
    p = copy.deepcopy(p)
    p['facts']['relations'].sort()
    return p


def build_request(request, payload):
    packed, sidecar = compact(payload)
    if canonical(expand(packed, sidecar)) != canonical(payload):
        raise ValueError('Compaction changed reference facts')
    result = copy.deepcopy(request)
    result['messages'] = [dict(role='user', content=INSTRUCTIONS + '\n' + dumps(packed))]
    return result, packed, sidecar


def main():
    import argparse
    import statistics
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'result/rc101_ontology_v5_train20_b128_seed43')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'result/rc101_ontology_v5_compact_train20_b128_seed43')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Never overwrite prepared experiments')
    prepared, audit = [], []
    for path in sorted((args.source / 'requests').glob('*/request.json')):
        context_path = path.with_name('context.json')
        original = json.loads(path.read_text(encoding='utf-8'))
        context = json.loads(context_path.read_text(encoding='utf-8'))
        result, packed, sidecar = build_request(original, context)
        assert result['model'] == original['model'] and result['response_format'] == original['response_format']
        audit.append(dict(scenario=path.parent.name, original_characters=len(original['messages'][0]['content']),
                          compact_characters=len(result['messages'][0]['content']),
                          source_request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          source_context_sha256=hashlib.sha256(context_path.read_bytes()).hexdigest(),
                          request_sha256=hashlib.sha256(dumps(result).encode()).hexdigest()))
        prepared.append((path.parent.name, result, packed, sidecar))
    if len(prepared) != 20:
        raise ValueError('Expected complete train20 inputs')
    before = statistics.mean(r['original_characters'] for r in audit)
    after = statistics.mean(r['compact_characters'] for r in audit)
    for name, request, context, sidecar in prepared:
        folder = args.out / 'requests' / name
        folder.mkdir(parents=True)
        for filename, value in [('request.json',request),('context.json',context),('provenance.json',sidecar)]:
            (folder / filename).write_text(dumps(value), encoding='utf-8')
    report = dict(version=VERSION, status='prepared_only', api_calls=0, training_started=False,
                  original_mean_characters=before, compact_mean_characters=after, reduction_percent=100*(before-after)/before,
                  exact_facts_roundtrip=True, relation_order_significant=False,
                  caveat='Facts and diagnostic labels preserved; instructions consolidated. Character reduction is not token/cost measurement or evidence of equivalent LLM behavior.',
                  scenarios=audit)
    (args.out / 'protocol.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='scenarios'}, indent=2))


if __name__ == '__main__':
    main()
