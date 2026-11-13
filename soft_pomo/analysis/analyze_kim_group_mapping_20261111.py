"""CPU-only active-anchor intervention exposure and paired-result audit."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/kim_group_mapping_20261111'


def read(path):
    return json.loads(path.read_text())


def main():
    assert read(OUT/'status.json')['completed']==240
    groups={'normal_four':['seed42','seed43','seed45','seed46'],'failed44':['seed44'],
        'recovered44':['seed44_epoch1010'],'original_five':[f'seed{s}' for s in range(42,47)]}
    paired=read(OUT/'paired.json')
    report=dict(groups={},limitations=[
        'Pooled control preserves unweighted mean across nonempty group means, not the distribution of bonuses at visited anchors.',
        'Permuted control preserves each state group-mean multiset, but visited anchor bonus magnitude can change; not a fully scale-matched causal comparison.',
        'Only one preregistered deterministic shuffle map per available group set; no permutation replication or significance claim.',
        'Active groups are feasible for current selected vehicle, not all vehicles jointly. Frozen R101 Train20 only, no training/RC generalization.'])
    for control in ('pooled_group_mean','permuted_group_mean'):
        report['groups'][control]={}
        for group,labels in groups.items():
            steps=[]
            for label in labels:
                for path in (OUT/control/label/control).glob('*.json'):
                    steps.extend(t for t in read(path)['trace'] if not t['dispatch_open'])
            active=[t for t in steps if t['valid_anchor_customers']>0]
            subset=[r for r in paired['cases'] if r['control']==control and r['checkpoint'] in labels]
            report['groups'][control][group]=dict(routing_states=len(steps),active_anchor_states=len(active),
                assigned_differs_states=sum(abs(t['assigned_anchor_mean']-t['original_anchor_mean'])>1e-12 for t in active),
                singleton_available_states=sum(t['available_groups']==1 for t in active),
                active_mean={k:statistics.mean(t[k] for t in active) for k in ('original_anchor_mean','assigned_anchor_mean','mean_bonus_delta','available_groups')},
                active_variance={k:statistics.pvariance(t[k] for t in active) for k in ('original_anchor_mean','assigned_anchor_mean','mean_bonus_delta','available_groups')},
                paired_reward_win_tie_loss={key:sum((r['delta_vs_mean']['reward']>1e-5 if key=='win' else
                    r['delta_vs_mean']['reward'] < -1e-5 if key=='loss' else abs(r['delta_vs_mean']['reward'])<=1e-5) for r in subset) for key in ('win','tie','loss')},
                worst_case=min(subset,key=lambda r:r['delta_vs_mean']['reward']))
    protocol=read(OUT/'inputs.json')
    protocol['limitations']=report['limitations']
    (OUT/'inputs.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (OUT/'exposure_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
