"""Validate and summarize completed sequential multi-seed test evaluations."""
import json
from pathlib import Path
import statistics as st
import argparse

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'server_export/seq_completed_received_20261024_112331'
HISTORY={'RC101':'rc101_pilot_history_20_80_20260923_104517',
         'C101':'c101_history_strength200_20261013_141106',
         'R101':'r101_history_strength050_20260925_233437'}


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=IN)
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/seq_received_summary_20261024.json')
    args=parser.parse_args()
    groups={}
    received=[]
    seen=set()
    for path in sorted(args.input.rglob('eval_1000_test80_gpu/results.json')):
        bundle=path.parent.parent
        cfg=read(bundle/'configs/pure.json')
        env=cfg['environment']; seed=cfg['trainer']['seed']; late=env['late_penalty']
        done=read(path.parent/'status.json'); train=read(bundle/'runs/pure/status.json')
        assert done['stage']=='complete' and done['epoch']==1000 and done['completed']==80
        assert train['status']=='complete' and train['epoch']==1000
        rows=read(path); split=read(bundle/'split.json')
        assert [r['name'] for r in rows]==[r['name'] for r in split['test']]
        assert len(rows)==80 and all(r['served']==100 for r in rows)
        case=rows[0]['name'].split('_')[0]
        identity=(case,late,seed)
        if identity in seen:
            raise ValueError(f'Duplicate experiment: {identity}')
        seen.add(identity)
        horizon={'RC101':240,'C101':1236,'R101':230}[case]
        def reward(r):
            return -(env['vehicle_penalty']*r['vehicles']/100+
                     r['travel_time']/(horizon*env['TT_max'])+
                     late*r['total_late']/(horizon*env['Lt_max']))
        assert max(abs(reward(r)-r['reward']) for r in rows)<1e-4
        history=read(ROOT/'analysis'/HISTORY[case]/'results.json')
        assert {r['name'] for r in history}=={r['name'] for r in rows}
        baseline=st.mean(reward(r) for r in history)
        metrics={k:st.mean(r[k] for r in rows) for k in ['reward','vehicles','travel_time','total_late','late_count']}
        groups.setdefault((case,late),[]).append(dict(seed=seed,lkh_reward=baseline,**metrics))
        received.append(bundle.relative_to(args.input).as_posix())
    report=[]
    for (case,late),seeds in sorted(groups.items()):
        rewards=[s['reward'] for s in seeds]; b=seeds[0]['lkh_reward']; m=st.mean(rewards)
        row=dict(instance=case,late=late,count=len(seeds),seeds=seeds,lkh_reward=b,
                 mean_reward=m,std_reward=st.stdev(rewards) if len(rewards)>1 else None,
                 cost_reduction_percent=(m-b)/abs(b)*100,winning_seeds=sum(r>b for r in rewards),
                 metrics={k:st.mean(s[k] for s in seeds) for k in ['vehicles','travel_time','total_late','late_count']})
        report.append(row)
        print(json.dumps(row),flush=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    manifest=ROOT/'server_export/seq_received_runs.json'
    previous=read(manifest) if manifest.exists() else []
    manifest.write_text(json.dumps(sorted(set(previous+received)),indent=2),encoding='utf-8')


if __name__=='__main__':main()
