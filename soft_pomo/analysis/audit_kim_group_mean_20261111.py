"""Independent saved-record audit; no environment/model execution."""
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/kim_group_mean_audit_20261111'
MEAN=ROOT/'result/kim_group_mean_20261111'
FULL=ROOT/'result/kim_fusion_stable_20261110'


def read(p):return json.loads(p.read_text())


def main():
    assert read(MEAN/'status.json')['stage']=='complete'
    protocol=read(MEAN/'protocol.json');original=read(FULL/'protocol.json')
    assert protocol['checkpoints']==original['checkpoints']
    assert protocol['setting']==original['setting']
    assert protocol['decode']==original['decode']=='greedy AMP'
    hashes=read(MEAN/'inputs.json')['sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    labels=[f'seed{s}' for s in range(42,47)]+['seed44_epoch1010']
    cases=[]
    fields=('reward','vehicles','travel_time','total_late','unserved')
    for label in labels:
        files=list((MEAN/label/'group_mean').glob('*.json'))
        assert len(files)==20
        for path in files:
            a,b=read(path),read(FULL/label/'alpha025'/path.name)
            assert a['name']==b['name'] and a['alpha']==b['alpha']==.25 and a['beta']==b['beta']==5
            aa,bb=a['trace'],b['trace']
            assert [t['action'] for t in aa if t['dispatch_open']]==[t['action'] for t in bb if t['dispatch_open']]
            assert a['metrics']['vehicles']==b['metrics']['vehicles']
            first=next(((x,y) for x,y in zip(bb,aa) if x['action']!=y['action']),None)
            def describe(t):
                assert not t['dispatch_open']
                assert 100<=t['action']<=200
                return dict(step=t['step'],dest=t['action']-100,kind='depot' if t['action']==100 else 'customer')
            delta={k:a['metrics'][k]-b['metrics'][k] for k in fields}
            cases.append(dict(checkpoint=label,name=a['name'],mean_metrics=a['metrics'],full_metrics=b['metrics'],delta=delta,
                exact_path=[t['action'] for t in aa]==[t['action'] for t in bb],
                first_divergence=None if first is None else dict(full=describe(first[0]),mean=describe(first[1]))))
    memberships=dict(original_five=labels[:5],normal_four=['seed42','seed43','seed45','seed46'],failed44=['seed44'],recovered44=['seed44_epoch1010'])
    groups={}
    for group,members in memberships.items():
        rs=[r for r in cases if r['checkpoint'] in members]
        transitions=collections.Counter(r['first_divergence']['full']['kind']+'->'+r['first_divergence']['mean']['kind'] for r in rs if r['first_divergence'])
        groups[group]=dict(count=len(rs),full={k:statistics.mean(r['full_metrics'][k] for r in rs) for k in fields},
            mean={k:statistics.mean(r['mean_metrics'][k] for r in rs) for k in fields},
            delta={k:statistics.mean(r['delta'][k] for r in rs) for k in fields},
            paths_identical=sum(r['exact_path'] for r in rs),first_transitions=dict(transitions),
            wins=sum(r['delta']['reward']>1e-6 for r in rs),losses=sum(r['delta']['reward'] < -1e-6 for r in rs),
            ties=sum(abs(r['delta']['reward'])<=1e-6 for r in rs))
    own=read(MEAN/'paired.json')['groups']
    for key,value in groups.items():
        assert value['mean']==own[key]['mean_only']
        assert value['delta']==own[key]['delta_vs_full']
    report=dict(status='passed',role='Independent read-only CPU saved-record audit; no new rollout/API/Test/training',
        validated_cases=len(cases),checkpoint_hashes_same_as_comparator=True,config_cache_hashes_unchanged=True,
        code_review='Valid anchor membership AND current mask mean; NEW/depot/dispatch zero; full=bonus+residual with empty/singleton checks; original AMP q and FP64 fusion.',
        groups=groups,
        limitation='Saved actions establish depot/customer type only; no claimed anchor group at first divergence without environment replay.')
    OUT.mkdir(exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    (OUT/'cases.json').write_text(json.dumps(cases,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
