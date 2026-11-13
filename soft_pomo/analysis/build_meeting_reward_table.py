"""Compare recorded RL rewards and common-objective historical rewards."""
import json
import statistics as st
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    entries=[]
    def add(pool,method,seed,epoch,data,source):
        assert len(data)==80 and len({r['name'] for r in data})==80
        assert all(r['served']==100 for r in data)
        if entries:
            assert {r['name'] for r in data}==set(entries[0]['names'])
        reconstructed=[-(.04*r['vehicles']+r['travel_time']/5760+.0075*r['total_late']) for r in data]
        saved=method!='LKH-3 Full History'
        rewards=[r['reward'] for r in data] if saved else reconstructed
        error=max(abs(a-b) for a,b in zip(rewards,reconstructed))
        assert error<1e-4,(source,error)
        metrics={k:dict(mean=st.mean(r.get(k,r.get('late_stops')) for r in data),
                           std=st.stdev(r.get(k,r.get('late_stops')) for r in data))
                 for k in ('vehicles','travel_time','late_count','total_late')}
        metrics['reward']=dict(mean=st.mean(rewards),std=st.stdev(rewards))
        entries.append(dict(pool=pool,method=method,seed=seed,epoch=epoch,source=source,
            names=[r['name'] for r in data],reward_source='saved' if saved else 'reconstructed',
            max_reward_error=error,metrics=metrics))
    for pool in (20,80,200):
        h={20:'rc101_pilot_history_20_80_20260923_104517',80:'rc101_history_train80_20260929',
           200:'rc101_history_train200_20261002'}[pool]
        path=ROOT/'analysis'/h/'results.json'
        add(pool,'LKH-3 Full History','--','--',json.loads(path.read_bytes()),str(path))
        seed=43 if pool==20 else 42
        for epoch in (1000,2000,3000):
            if pool==80:
                path=ROOT/f'server_export/server_pure128_results/eval_{epoch}_test80/results.json'
                data=json.loads(path.read_bytes());source=str(path)
            else:
                path=ROOT/f'server_export/train{pool}_seed{seed}_results.tar.gz'
                member=('train20/' if pool==20 else '')+f'runs/pure128_4000_seed{seed}/eval_{epoch}_test80/results.json'
                with tarfile.open(path) as tf:
                    data=json.load(tf.extractfile(member))
                source=str(path)+':'+member
            add(pool,'Pure RL',seed,epoch,data,source)
        if pool!=200:
            for method,directory in [('Kim + Confidence', 'rc101_kim_independent1000' if pool==20 else 'rc101_kim_train80_fresh1000_20260929'),
                ('Pure RL500 + Phase2','rc101_relation_train20_cache_v1' if pool==20 else 'rc101_pure500_train80_phase2_20260930')]:
                path=ROOT/'result'/directory/'eval_1000_test80/results.json'
                add(pool,method,42,1000,json.loads(path.read_bytes()),str(path))
    lines=[r'% Requires \usepackage{booktabs,graphicx,multirow}',r'\begin{table}[htbp]',r'\centering',
        r'\caption{Test80 results: mean $\pm$ sample standard deviation. Pure RL batch128; guided methods batch64.}',
        r'\label{tab:routing_reward}',r'\small',r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{@{}rlrrccccc@{}}',r'\toprule',
        r'Train size & Method & Seed & Epoch & Vehicles & Travel time & Late stops & Total lateness & Reward $\uparrow$ \\',r'\midrule']
    for pool in (20,80,200):
        block=[e for e in entries if e['pool']==pool]
        best={k:(max if k=='reward' else min)(e['metrics'][k]['mean'] for e in block) for k in block[0]['metrics']}
        if pool!=20: lines.append(r'\midrule')
        for i,e in enumerate(block):
            method=e['method']
            if i==1: method=r'\multirow{3}{*}{Pure RL}'
            elif i in (2,3): method=''
            cells=[r'\multirow{%d}{*}{%d}'%(len(block),pool) if i==0 else '',method,str(e['seed']),str(e['epoch'])]
            for k,v in e['metrics'].items():
                digits=4 if k=='reward' else 2
                cell=f"${v['mean']:.{digits}f} \\pm {v['std']:.{digits}f}$"
                if abs(v['mean']-best[k])<1e-12: cell=r'{\boldmath '+cell+'}'
                cells.append(cell)
            lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}%','}',r'\par\smallskip',r'\begin{minipage}{\linewidth}',r'\footnotesize',
        r'Reward is $R=-[0.04K+T/5760+0.0075L_t]$; higher (less negative) is better. RL rewards are stored values checked against replay metrics; LKH-3 rewards are reconstructed under the same objective, not its solver objective. All solutions serve 100 customers. Lateness includes depot returns. Standard deviations describe 80 scenarios, not training seeds. Phase2 representatives are selected by lowest mean test cost at epoch1000: relation-feedback v1 (train20), original Phase2 (train80). Settings differ; this is not an isolated ontology or training-size ablation.',
        r'\end{minipage}',r'\end{table}']
    (ROOT/'analysis/meeting_reward_audit.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    (ROOT/'analysis/pure_rl_meeting_table_with_reward.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for e in entries:
        print(e['pool'],e['method'],e['epoch'],e['metrics']['reward'], 'max error',e['max_reward_error'])


if __name__=='__main__':
    main()
