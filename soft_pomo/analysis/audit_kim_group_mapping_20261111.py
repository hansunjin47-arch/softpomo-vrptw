"""Independent saved-output and pure permutation checks; no model/environment run."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/kim_group_mapping_audit_20261111'
SOURCE=ROOT/'result/kim_group_mapping_20261111'
MEAN=ROOT/'result/kim_group_mean_20261111'
FULL=ROOT/'result/kim_fusion_stable_20261110'


def read(p):return json.loads(p.read_text())


def main():
    assert read(SOURCE/'status.json')['stage']=='complete'
    inp=read(SOURCE/'inputs.json')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in inp['sha256'].items())
    tree=ast.parse((ROOT/'analysis/diagnose_kim_group_mapping_20261111.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='donor_ids')
    ns=dict(hashlib=hashlib,random=random,PERMUTATIONS={})
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<donor_ids audit>','exec'),ns)
    rng=random.Random(817)
    for _ in range(100):
        ids=sorted(rng.sample(range(25),rng.randrange(1,26)))
        first=ns['donor_ids'](ids)
        ns['PERMUTATIONS'].clear()
        assert first==ns['donor_ids'](ids) and sorted(first)==ids
    results=[]
    fields=('reward','vehicles','travel_time','total_late','unserved')
    expected=read(SOURCE/'paired.json')
    lookup={(r['control'],r['checkpoint'],r['name']):r for r in expected['cases']}
    labels=[f'seed{s}' for s in range(42,47)]+['seed44_epoch1010']
    for control in ('pooled_group_mean','permuted_group_mean'):
        protocol=read(SOURCE/control/'protocol.json')
        assert protocol['checkpoints']==read(MEAN/'protocol.json')['checkpoints']==read(FULL/'protocol.json')['checkpoints']
        assert protocol['decode']=='greedy AMP' and protocol['setting']==read(MEAN/'protocol.json')['setting']
        for label in labels:
            files=list((SOURCE/control/label/control).glob('*.json'))
            assert len(files)==20
            for path in files:
                cur=read(path);mean=read(MEAN/label/'group_mean'/path.name);full=read(FULL/label/'alpha025'/path.name)
                assert cur['alpha']==mean['alpha']==full['alpha']==.25 and cur['beta']==mean['beta']==full['beta']==5
                delta={k:cur['metrics'][k]-mean['metrics'][k] for k in fields}
                df={k:cur['metrics'][k]-full['metrics'][k] for k in fields}
                paired=lookup[(control,label,cur['name'])]
                assert paired['delta_vs_mean']==delta and paired['delta_vs_full']==df
                assert delta['vehicles']==df['vehicles']==0 and cur['metrics']['unserved']==0
                dispatch=lambda r:[t['action'] for t in r['trace'] if t['dispatch_open']]
                assert dispatch(cur)==dispatch(mean)==dispatch(full)
                active=[t for t in cur['trace'] if t['valid_anchor_customers']>0]
                if control=='pooled_group_mean':
                    assert all(abs(t['assigned_anchor_mean']-t['group_mean_average'])<1e-12 for t in active)
                for t in cur['trace']:
                    assert abs(t['mean_bonus_delta']-5*(t['assigned_anchor_mean']-t['original_anchor_mean']))<1e-12
                results.append(dict(control=control,checkpoint=label,name=cur['name'],delta_vs_mean=delta,
                    changed_path=paired['changed_path'],active_decisions=len(active),
                    changed_bonus_decisions=sum(abs(t['mean_bonus_delta'])>1e-9 for t in active)))
    assert len(results)==240
    groups={}
    for control in ('pooled_group_mean','permuted_group_mean'):
        groups[control]={}
        for group,members in dict(normal_four=['seed42','seed43','seed45','seed46'],failed44=['seed44'],recovered44=['seed44_epoch1010']).items():
            rs=[r for r in results if r['control']==control and r['checkpoint'] in members]
            groups[control][group]=dict(count=len(rs),wins=sum(r['delta_vs_mean']['reward']>1e-6 for r in rs),
                losses=sum(r['delta_vs_mean']['reward'] < -1e-6 for r in rs),ties=sum(abs(r['delta_vs_mean']['reward'])<=1e-6 for r in rs),
                changed_paths=sum(r['changed_path'] for r in rs),delta={k:statistics.mean(r['delta_vs_mean'][k] for r in rs) for k in fields},
                active_decisions=sum(r['active_decisions'] for r in rs),changed_bonus_decisions=sum(r['changed_bonus_decisions'] for r in rs))
    report=dict(status='passed',cases=240,groups=groups,permutation_reproducibility_cases=100,input_hashes_unchanged=True,
        mask_validation='Code asserts exact zero mask and dispatch/beta0 argmax; saved action sequences dispatch identical; no new rollout.',
        pooling='Unweighted mean across nonempty groups feasible for current selected vehicle. Every saved active-state assigned mean matches saved group mean average.',
        permutation='Pure donor function independently tested after cache clear on 100 sorted ID sets; exact multiset preservation per inference is asserted by experiment code, not rederived from incomplete saved score vectors.',
        limitation='Neither control matches the applied anchor bonus per state. Group-wide mean/multiset preservation does not preserve actual selected-vehicle bias dose. Dynamic available-group permutation is not one fixed whole-episode mapping.',
        scope='Read-only CPU JSON/hash/pure permutation audit; no API/Test/model/environment execution.')
    OUT.mkdir(exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    (OUT/'cases.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
