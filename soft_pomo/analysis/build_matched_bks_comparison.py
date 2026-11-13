"""Match every BKS result to its own instance, train pool and event-history baseline."""
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon

HISTORY = {
    ('rc101',20): 'rc101_pilot_history_20_80_20260923_104517',
    ('rc101',40): 'rc101_history_train40_20260928',
    ('rc101',80): 'rc101_history_train80_20260929',
    ('c101',20): 'c101_history_strength200_20261013_141106',
    ('r101',20): 'r101_history_strength050_20260925_233437',
}
BKS = {'rc101':(14,1696.95),'c101':(10,828.94),'r101':(19,1650.80)}


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def same_dynamics(a,b):
    x,y=load_solomon(str(a)),load_solomon(str(b))
    for key in ('vehicle_limit','vehicle_capacity','T','depot_xy','node_xy','node_demand',
                'node_tw_open','node_tw_close','node_service'):
        np.testing.assert_equal(np.asarray(x[key]),np.asarray(y[key]))
    def events(v):
        return [{k:e[k] for k in ('type','nodes','trigger_time','duration','multiplier')} for e in v['preset_events']]
    assert events(x)==events(y)


def main():
    patterns = [
        'result/bks_normalized*/*/eval_1000_test80_gpu/summary.json',
        'result/rc101_*bks*/eval_1000_test80_gpu/summary.json',
        'server_export/rc101_bks_verified_20261014_111512/*/*/eval_1000_test80_gpu/summary.json',
        'server_export/rc101_bks_late*_received_*/*/eval_1000_test80_gpu/summary.json',
        'server_export/bks_completed_received_20261015_220737/*/*/eval_1000_test80_gpu/summary.json',
        'server_export/bks_train80_received_20261016_095923/*/*/eval_1000_test80_gpu/summary.json',
        'server_export/cr_completed_received_20261016_174210/*/*/eval_1000_test80_gpu/summary.json',
    ]
    rows=[]
    checked=set()
    for path in sorted({p for pattern in patterns for p in ROOT.glob(pattern)}):
        run=path.parent.parent
        bundle=run/'bundle' if (run/'bundle').exists() else run
        config_path=next((bundle/'configs').glob('*.json'))
        cfg=read(config_path)
        train=cfg['trainer']['train_instances']
        inst=train[0].split('_')[0].lower()
        size=len(train)
        weight=cfg['environment']['late_penalty']
        k0,t0=BKS[inst]
        assert abs(cfg['environment']['vehicle_penalty']-100/k0)<1e-8
        split=read(bundle/'split.json')
        assert set(train)=={r['name'] for r in split['train']}
        history=ROOT/'analysis'/HISTORY[(inst,size)]
        hs=read(history/'split.json')
        for group in ('train','test'):
            assert {r['name'] for r in hs[group]}=={r['name'] for r in split[group]},(path,group)
        for r in split['train']+split['test']:
            a=bundle/'data'/(r['name']+'.txt')
            b=Path(hs['data'])/(r['name']+'.txt')
            key=(str(a),str(b))
            if key not in checked:
                same_dynamics(a,b)
                checked.add(key)
        actual=read(path.parent/'results.json')
        hr=read(history/'results.json')
        assert len(actual)==len(hr)==80
        assert {r['name'] for r in actual}=={r['name'] for r in hr}
        baseline={k:statistics.mean(r[k] for r in hr) for k in ('vehicles','travel_time','total_late')}
        baseline['late_count']=statistics.mean(r.get('late_count',r.get('late_stops')) for r in hr)
        baseline['reward']=-(baseline['vehicles']/k0+(baseline['travel_time']+weight*baseline['total_late'])/t0)
        result=read(path)['ALL']['mean']
        rows.append(dict(instance=inst,train_size=size,lambda_late=weight,run=str(run.relative_to(ROOT)),
                         history=str(history.relative_to(ROOT)),history_metrics=baseline,metrics=result,
                         matched_train_test_and_event_dynamics=True,
                         cost_reduction_vs_history_pct=100*(1-result['cost']/(-baseline['reward']))))
    out=ROOT/'analysis/bks_matched_history_comparison.json'
    out.write_text(json.dumps(dict(results=rows,missing_history=[dict(instance=i,train_size=s)
        for i in ('c101','r101') for s in (40,80)],
        note='History routes are rescored under each lambda, not optimized for the weighted soft-TW objective.'),indent=2),encoding='utf-8')
    print('VERIFIED',len(rows),'evaluated runs;',len(checked),'data pairs')
    for r in rows:
        if r['instance']=='rc101' and r['train_size'] in (40,80):
            print(r['train_size'],r['lambda_late'],r['history_metrics'],'improvement_pct',r['cost_reduction_vs_history_pct'])


if __name__=='__main__':
    main()
