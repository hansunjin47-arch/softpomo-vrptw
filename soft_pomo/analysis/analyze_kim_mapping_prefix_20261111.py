"""Saved-trace first-divergence audit; no environment replay or model calls."""
from collections import Counter
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/kim_group_mapping_20261111'


def read(path):
    return json.loads(path.read_text())


def main():
    group_path=ROOT/'server_export/r101_pair100_l5_mask_kim_s42_46_20261106/artifacts/groups.json'
    groups=read(group_path)
    membership={node:g for g,row in enumerate(groups) for node in row}
    assert sorted(membership)==list(range(1,101))
    def kind(action,anchor):
        if action<100:
            return dict(kind='NEW',destination=action+1)
        dest=(action-100)%101
        return dict(kind='DEPOT' if dest==0 else 'ingroup' if membership[dest]==anchor else 'outgroup',destination=dest)
    paired=read(OUT/'paired.json')
    rows=[]
    for row in paired['cases']:
        if row['checkpoint']!='seed44':
            continue
        control=row['control'];name=row['name']
        new=read(OUT/control/'seed44'/control/(name+'.json'))
        old=read(ROOT/'result/kim_group_mean_20261111/seed44/group_mean'/(name+'.json'))
        diffs=[i for i,(a,b) in enumerate(zip(old['trace'],new['trace'])) if a['action']!=b['action']]
        first=diffs[0] if diffs else None
        value=dict(control=control,name=name,first_divergence=first,
            delta_reward=row['delta_vs_mean']['reward'],delta_travel_time=row['delta_vs_mean']['travel_time'],
            delta_lateness=row['delta_vs_mean']['total_late'])
        if first is not None:
            assert all(a['action']==b['action'] and a['dispatch_open']==b['dispatch_open']
                for a,b in zip(old['trace'][:first],new['trace'][:first]))
            a=old['trace'][first];b=new['trace'][first]
            assert a['step']==b['step'] and not a['dispatch_open'] and not b['dispatch_open']
            anchor=b['anchor_group']
            value.update(prefix_actions_identical=True,anchor_group=anchor,
                original_anchor_mean=b['original_anchor_mean'],assigned_anchor_mean=b['assigned_anchor_mean'],
                available_groups=b['available_groups'],valid_anchor_customers=b['valid_anchor_customers'],
                bonus_delta=b['mean_bonus_delta'],clock=None,clock_reason='Clock was not saved in these traces; no replay permitted',
                baseline_action=kind(a['action'],anchor),modified_action=kind(b['action'],anchor),
                bonus_direction='lowered' if b['mean_bonus_delta']<0 else 'raised' if b['mean_bonus_delta']>0 else 'same')
        else:
            assert len(old['trace'])==len(new['trace'])
            value['prefix_actions_identical']=True
        rows.append(value)
    assert len(rows)==40
    summary={}
    for control in ('pooled_group_mean','permuted_group_mean'):
        selected=[r for r in rows if r['control']==control]
        changed=[r for r in selected if r['first_divergence'] is not None]
        summary[control]=dict(cases=len(selected),changed=len(changed),
            bonus_directions=dict(Counter(r['bonus_direction'] for r in changed)),
            action_transitions=dict(Counter(r['baseline_action']['kind']+'->'+r['modified_action']['kind'] for r in changed)),
            joint=dict(Counter(r['bonus_direction']+'|'+r['baseline_action']['kind']+'->'+r['modified_action']['kind']+'|'+
                ('worse' if r['delta_reward'] < -1e-5 else 'better' if r['delta_reward']>1e-5 else 'tie') for r in changed)))
    result=dict(cases=rows,summary=summary,method='Saved matched-action prefixes; same deterministic environment/config/cache implies matching pre-divergence state, not independently replayed state tensors',
        action_decode='Existing pair destination=(action-100)%101, as frozen pair env.decode_action; membership read from saved groups',
        limitations=['Clock unavailable in saved traces; no new replay performed',
            'First change uses identical history, avoiding later visitation confounding for this one decision only',
            'Association of first bonus change with final reward is not its isolated causal effect; future control decisions also differ',
            'Single deterministic permutation, frozen R101 Train20 failed44 only; not training/RC generalization'])
    (OUT/'first_divergence_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
