"""Paired final-test decomposition; no training or paid calls."""
import collections
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
PURE=ROOT/'result/pair300_epsilon005_floor001_std_20261026/late10'
KIM=ROOT/'result/pair300_epsilon005_floor001_std_kim_20261026'
OUT=ROOT/'result/pair_kim_final_diagnosis_20261028'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def mean(x): return st.mean(x) if x else None
def scores(payload):
    values=payload['confidence']
    return {int(k):v for k,v in values.items()} if isinstance(values,dict) else {r['customer']:r['score'] for r in values}
def starts(history):
    return sorted(int(row[0][2]) for row in history if row[0][3] and row[0][4])

def main():
    a={r['name']:r for r in read(PURE/'results.json')}
    b={r['name']:r for r in read(KIM/'training/late10/results.json')}
    rows=[]
    for n in a:
        row=dict(name=n, reward_gain=b[n]['reward']-a[n]['reward'])
        for k in ('vehicles','travel_time','total_late','cost_vehicle','cost_travel','cost_late','cost_unserved'):
            row[k+'_delta']=b[n][k]-a[n][k]
        row['pure_starts']=starts(read(PURE/'test_history'/(n+'.json')))
        row['kim_starts']=starts(read(KIM/'training/late10/test_history'/(n+'.json')))
        calls=read(KIM/'late10/online_test'/n/'calls.json')
        row['calls']=len(calls)
        row['initial_mean_score']=mean(list(scores(calls[0]['response']).values()))
        if len(calls)>1:
            before,after=scores(calls[0]['response']),scores(calls[1]['response'])
            row['remaining_at_refresh']=len(after)
            row['changed_at_refresh']=sum(abs(v-before[k])>1e-8 for k,v in after.items())
            row['refresh_mean_absolute_change']=mean([abs(v-before[k]) for k,v in after.items()])
        rows.append(row)
    summary={}
    for label in ('ALL','RAIN','ACC'):
        subset=[r for r in rows if label=='ALL' or '_'+label+'_' in r['name']]
        summary[label]=dict(count=len(subset), wins=sum(r['reward_gain']>1e-6 for r in subset),
            mean_reward_gain=mean([r['reward_gain'] for r in subset]),
            median_reward_gain=st.median(r['reward_gain'] for r in subset),
            mean_deltas={k:mean([r[k+'_delta'] for r in subset]) for k in ('vehicles','travel_time','total_late','cost_vehicle','cost_travel','cost_late')},
            pure_unique_starts=len({tuple(r['pure_starts']) for r in subset}),
            kim_unique_starts=len({tuple(r['kim_starts']) for r in subset}),
            same_starts=sum(r['pure_starts']==r['kim_starts'] for r in subset),
            mean_initial_confidence=mean([r['initial_mean_score'] for r in subset]))
    train={}
    for event in ('RAIN','ACC'):
        cs=[read(p) for p in (KIM/'artifacts/train_cache').glob('*_'+event+'_*.json')]
        train[event]=dict(initial_mean=mean([mean(list(scores(c['initial']).values())) for c in cs]),
            post_mean=mean([mean(list(scores(c['post_disclosure']).values())) for c in cs if c.get('post_disclosure')]))
    report=dict(summary=summary, train_confidence=train,
                worst=sorted(rows,key=lambda r:r['reward_gain'])[:5], best=sorted(rows,key=lambda r:r['reward_gain'])[-5:],rows=rows)
    OUT.mkdir(exist_ok=True)
    (OUT/'paired_test.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__=='__main__': main()
