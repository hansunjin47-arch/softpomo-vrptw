"""Descriptive cache comparison; reference groups differ across versions."""
import itertools
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = dict(original='rc101_pure500_fullconfidence_phase2',
            v1='rc101_relation_train20_cache_v1', v5='rc101_ontology_v5_train20_b128_seed43')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    names = read(ROOT/'result'/RUNS['v5']/'protocol.json')['train_names']
    versions, report = {}, {}
    for version, run in RUNS.items():
        records = {}
        for name in names:
            c = read(ROOT/'result'/run/'cache'/(name+'_rl_cluster.json'))
            groups = {k:v for k,v in c.items() if not k.startswith('_')}
            scores = {int(n):v for group in groups.values() for n,v in group.items()}
            membership = {int(n):k for k,group in groups.items() for n in group}
            assert sorted(scores)==list(range(1,101))
            records[name] = dict(groups=groups, scores=scores, membership=membership, depot=c['_depot'])
        versions[version] = records
        allscores = [v for r in records.values() for v in r['scores'].values()]
        depots = [v for r in records.values() for v in r['depot'].values()]
        ranges = [max(g.values())-min(g.values()) for r in records.values() for g in r['groups'].values()]
        margins = [s-r['depot'][r['membership'][n]] for r in records.values() for n,s in r['scores'].items()]
        report[version] = dict(customers=len(allscores), groups=len(ranges), mean=st.mean(allscores),
            sd=st.stdev(allscores), minimum=min(allscores), maximum=max(allscores),
            distinct_scores=len(set(allscores)), uniform_groups=sum(x<1e-9 for x in ranges),
            mean_group_range=st.mean(ranges), depot_mean=st.mean(depots), depot_min=min(depots), depot_max=max(depots),
            nonzero_depots=sum(x>0 for x in depots), mean_customer_minus_depot=st.mean(margins),
            customers_above_depot=sum(x>1e-9 for x in margins), customers_below_depot=sum(x < -1e-9 for x in margins))
    changes = {}
    for old in ('original','v1'):
        diffs, examples = [], []
        intersection=union=reversals=ties=preserved=0
        for name in names:
            a,b=versions[old][name],versions['v5'][name]
            for n in range(1,101):
                d=b['scores'][n]-a['scores'][n]
                diffs.append(d)
                examples.append(dict(scenario=name,customer=n,old=a['scores'][n],new=b['scores'][n],delta=d))
            for i,j in itertools.combinations(range(1,101),2):
                samea=a['membership'][i]==a['membership'][j]
                sameb=b['membership'][i]==b['membership'][j]
                union += samea or sameb
                intersection += samea and sameb
                if samea and sameb:
                    da=a['scores'][i]-a['scores'][j]; db=b['scores'][i]-b['scores'][j]
                    if abs(da)<1e-9 or abs(db)<1e-9: ties+=1
                    elif da*db<0: reversals+=1
                    else: preserved+=1
        changes[old+'_to_v5']=dict(customer_mae=st.mean(abs(x) for x in diffs),
            increased=sum(x>1e-9 for x in diffs), decreased=sum(x < -1e-9 for x in diffs), unchanged=sum(abs(x)<=1e-9 for x in diffs),
            coassigned_pair_jaccard=intersection/union, pairs_together_in_both=intersection,
            strict_rank_preserved=preserved, strict_rank_reversed=reversals, tied_in_either=ties,
            largest_changes=sorted(examples,key=lambda x:abs(x['delta']),reverse=True)[:8])
    diagnoses={}
    for name in names:
        p=read(ROOT/'result'/RUNS['v5']/'requests'/name/'context.json')
        visits={r[0]:r[3] for r in p['facts']['visits']}
        cols=p['facts']['diagnosis_columns']
        for row in p['facts']['diagnoses']:
            n=int(visits[row[0]][1:])
            if n==0: continue
            labels=[k for k,v in zip(cols[1:],row[1:]) if v] or ['no_positive_diagnosis']
            r=versions['v5'][name]
            for label in labels:
                diagnoses.setdefault(label,[]).append((r['scores'][n],r['scores'][n]-r['depot'][r['membership'][n]]))
    diag={k:dict(count=len(v),mean_score=st.mean(x[0] for x in v),mean_minus_depot=st.mean(x[1] for x in v)) for k,v in diagnoses.items()}
    out=dict(versions=report,comparisons=changes,v5_diagnosis=diag,
             caveat='2000 customer-scenario observations; not independent training samples. v5 uses different Pure500 routes, seed and batch. Overlapping diagnosis categories. Score differences are not causal ontology effects or action-probability differences.')
    path=ROOT/'analysis/ontology_cache_comparison_b128_v5.json'
    path.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
