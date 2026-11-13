"""Join the actual API payload, cached scores and completed route outcomes."""
from collections import Counter
import json
import statistics as st
from audit_target_guidance import ROOT, SOURCE, read


def main():
    audit = read(ROOT/'analysis/all_customer_guidance_audit.json')
    inputs = {}
    for name in sorted({r['name'] for r in audit['visits']['10']}):
        request = read(SOURCE/'requests'/name/'request.json')
        content = request['messages'][0]['content']
        payload = json.loads(content[content.index('{"ontology":'):])
        facts = payload['facts']
        legs = [dict(zip(facts['leg_columns'], row)) for row in facts['legs']]
        inputs[name] = {int(x['destination']): x for x in legs if x['destination']}
        assert len(inputs[name]) == 100
    enriched, summaries = {}, {}
    for scale, visits in audit['visits'].items():
        enriched[scale] = []
        for r in visits:
            f = inputs[r['name']][r['node']]
            assert abs(f['reference_late']-r['reference_late'])<2e-6
            ref_late, actual_late = r['reference_late']>1e-7, r['late']>1e-7
            category = ('persistent' if actual_late else 'resolved') if ref_late else (
                'new' if actual_late else 'neither')
            cache = read(SOURCE/'cache'/(r['name']+'_rl_cluster.json'))
            scores = list(cache[r['reference_group']].values())
            enriched[scale].append(dict(r, category=category,
                input_base_late=f['base_late'], input_added_lateness=f['added_lateness'],
                input_exposure=f['direct_exposure'],
                group_mean_score=st.mean(scores),
                group_max_score=max(scores),
                top_score=r['raw_score']==max(scores),
                distinct_group_scores=len(set(scores)),
                below_own_depot=r['raw_score']<r['own_depot_score']))
        summary = {}
        for cat in ('persistent','resolved','new','neither'):
            rr = [r for r in enriched[scale] if r['category']==cat]
            summary[cat] = dict(n=len(rr),reference_late=sum(r['reference_late'] for r in rr),
                actual_late=sum(r['late'] for r in rr),
                mean_score=st.mean(r['raw_score'] for r in rr) if rr else None,
                top_score=sum(r['top_score'] for r in rr),
                below_own_depot=sum(r['below_own_depot'] for r in rr),
                prefix_match=sum(r['prefix_match'] for r in rr),
                status=dict(Counter(r['status'] for r in rr)))
        summaries[scale] = summary
    reference = [r for r in enriched['10'] if r['reference_late']>1e-7]
    overview = dict(reference_late_visits=len(reference),
        mean_score=st.mean(r['raw_score'] for r in reference),
        top_score=sum(r['top_score'] for r in reference),
        tied_top=sum(r['top_score'] and r['distinct_group_scores']==1 for r in reference),
        below_depot=sum(r['below_own_depot'] for r in reference),
        existing_without_event=sum(r['input_base_late']>1e-7 for r in reference))
    report = dict(scope='Descriptive audit, all 16000 selected customer visits. '
        'API payload reference lateness checked against replay. No API calls or training.',
        reference=overview,summaries=summaries,
        late_visits={k:[r for r in v if r['category']!='neither'] for k,v in enriched.items()})
    (ROOT/'analysis/reference_score_outcomes.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(reference=overview,summaries=summaries),indent=2))


if __name__=='__main__':
    main()
