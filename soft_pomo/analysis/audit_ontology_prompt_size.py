"""Offline reversible serialization trial; never modifies active API inputs."""
import copy
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'result/rc101_ontology_v5_train20_b128_seed43/requests'
OUT = ROOT / 'analysis/ontology_prompt_compaction_trial'


def dump(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=True)


def compress(payload):
    p = copy.deepcopy(payload)
    f = p['facts']
    expected = [[r[0] + ':visit', r[1], r[0], 'c' + str(r[3])] for r in f['legs']]
    assert f['visits'] == expected
    del f['visits']
    original = f.pop('diagnoses')
    assert [r[0] for r in original] == [r[0] for r in expected]
    f['positive_diagnoses'] = {label: [i for i, row in enumerate(original) if row[k + 1]]
                               for k, label in enumerate(f['diagnosis_columns'][1:])}
    # Keep original order for lossless verification; row indices replace repeated IDs.
    f['visit_encoding'] = 'One visit per leg row, same order: [leg.id+":visit",leg.route,leg.id,"c"+leg.destination]. Positive diagnosis lists contain zero-based leg-row indices; every other diagnosis is false, not unknown.'
    for g in p['instance']['groups']:
        assert g['required_output_customers'] == [0] + g['reference_order']
        del g['required_output_customers']
    p['instance']['output_membership_encoding'] = 'required_output_customers = [0] + reference_order for every group; include each pair exactly once.'
    return p


def decompress(payload):
    p = copy.deepcopy(payload)
    f = p['facts']
    f['visits'] = [[r[0] + ':visit', r[1], r[0], 'c' + str(r[3])] for r in f['legs']]
    positives = f.pop('positive_diagnoses')
    f['diagnoses'] = [[r[0]] + [i in positives[k] for k in f['diagnosis_columns'][1:]]
                      for i, r in enumerate(f['visits'])]
    del f['visit_encoding']
    for g in p['instance']['groups']:
        g['required_output_customers'] = [0] + g['reference_order']
    del p['instance']['output_membership_encoding']
    return p


def main():
    rows = []
    OUT.mkdir(exist_ok=True)
    for path in sorted(SOURCE.glob('*/context.json')):
        p = json.loads(path.read_text(encoding='utf-8'))
        request = json.loads((path.parent / 'request.json').read_text(encoding='utf-8'))
        compact = compress(p)
        assert decompress(compact) == p
        facts = p['facts']
        sections = {'ontology': len(dump(p['ontology'])), 'legs': len(dump(facts['legs'])),
                    'relations': len(dump(facts['relations'])), 'visits': len(dump(facts['visits'])),
                    'diagnoses': len(dump(facts['diagnoses'])), 'instance': len(dump(p['instance']))}
        before = len(request['messages'][0]['content'])
        after = before - len(dump(p)) + len(dump(compact))
        rows.append(dict(name=path.parent.name, original_characters=before, compact_characters=after,
                         sections=sections, exact_roundtrip=True))
        (OUT / (path.parent.name + '.json')).write_text(dump(compact), encoding='utf-8')
    assert len(rows) == 20
    before = statistics.mean(r['original_characters'] for r in rows)
    after = statistics.mean(r['compact_characters'] for r in rows)
    report = dict(api_calls=0, active_requests_modified=False, scenarios=len(rows),
                  original_mean_characters=before, compact_mean_characters=after,
                  reduction_percent=100*(before-after)/before,
                  mean_sections={k:statistics.mean(r['sections'][k] for r in rows) for k in rows[0]['sections']},
                  limitation='Character lengths, not actual provider token counts. Exact data recovery does not establish equal LLM comprehension.',
                  scenarios_detail=rows)
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='scenarios_detail'}, indent=2))


if __name__ == '__main__':
    main()
