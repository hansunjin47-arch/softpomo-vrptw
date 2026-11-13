"""CPU-only independent deadline reassignment and saved-pair audit."""
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'result/kim_deadline_rank_20261111'
OUT=ROOT/'result/kim_deadline_rank_audit_20261111'
BASE=ROOT/'result/kim_group_mean_20261111'
FULL=ROOT/'result/kim_fusion_stable_20261110'


def read(p):return json.loads(p.read_text())


def main(percentile=False):
    global SOURCE,OUT
    if percentile:
        SOURCE=ROOT/'result/kim_deadline_percentile_20261111'
        OUT=ROOT/'result/kim_deadline_percentile_audit_20261111'
    variant='deadline_percentile_llm_free' if percentile else 'deadline_rank_matched'
    assert read(SOURCE/'status.json')['stage']=='complete'
    inputs=read(SOURCE/'inputs.json')['sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in inputs.items())
    cachepaths={Path(p).stem:Path(p) for p in inputs if Path(p).parent.name=='train_cache'}
    fixed=read(ROOT/'result/confidence_features_20261111/report.json')['datasets']['R101_weight100']['customers']
    close={c['id']:round(c['close_min'],4) for c in fixed}
    stagecount=0
    for name,path in cachepaths.items():
        cache=read(path)
        phases=[cache['initial']]+([cache['post_disclosure']] if cache['post_disclosure'] else [])
        facts=list((SOURCE/'fixed_facts'/name).glob('phase*.json'))
        assert len(facts)==len(phases)
        for i,payload in enumerate(phases):
            record=read(SOURCE/'fixed_facts'/name/f'phase{i}.json')
            original={int(x['customer']):float(x['score']) for x in payload['confidence']}
            new={int(k):v for k,v in record['reassigned_scores'].items()}
            deadlines={int(k):v for k,v in record['deadline_minutes_rounded4'].items()}
            assert sorted(new)==list(range(1,101))
            if not percentile:
                old={int(k):v for k,v in record['original_scores'].items()}
                assert old==original and sorted(old)==list(range(1,101))
                assert sorted(old.values())==sorted(new.values())
            assert deadlines==close
            order=sorted(close,key=lambda n:(close[n],n))
            assert record['order']==order
            if percentile:
                assert new=={n:(100-i)/100 for i,n in enumerate(order)}
                assert record['input_score_values_unused'] and record['arbitrary_input_values_invariance']
                if i:
                    assert record['reassigned_scores']==read(SOURCE/'fixed_facts'/name/'phase0.json')['reassigned_scores']
            else:
                assert [new[n] for n in order]==sorted(old.values(),reverse=True)
            stagecount+=1
    proto=read(SOURCE/'protocol.json')
    assert proto['checkpoints']==read(BASE/'protocol.json')['checkpoints']==read(FULL/'protocol.json')['checkpoints']
    assert proto['decode']=='greedy AMP' and proto['setting']==read(BASE/'protocol.json')['setting']
    fields=('reward','vehicles','travel_time','total_late','unserved')
    cases=[]
    for label in [f'seed{s}' for s in range(42,47)]+['seed44_epoch1010']:
        paths=list((SOURCE/label/variant).glob('*.json'));assert len(paths)==20
        for path in paths:
            row=read(path);base=read(BASE/label/'group_mean'/path.name);full=read(FULL/label/'alpha025'/path.name)
            assert row['alpha']==base['alpha']==full['alpha']==.25 and row['beta']==base['beta']==full['beta']==5
            dispatch=lambda r:[t['action'] for t in r['trace'] if t['dispatch_open']]
            assert dispatch(row)==dispatch(base)==dispatch(full)
            assert row['metrics']['vehicles']==base['metrics']['vehicles']==full['metrics']['vehicles']
            assert row['metrics']['unserved']==0
            cases.append(dict(checkpoint=label,name=row['name'],delta={k:row['metrics'][k]-base['metrics'][k] for k in fields},
                same_path=[t['action'] for t in row['trace']]==[t['action'] for t in base['trace']]))
    expected=read(SOURCE/'paired.json')['groups'];groups={}
    for key,labels in dict(normal_four=['seed42','seed43','seed45','seed46'],failed44=['seed44'],recovered44=['seed44_epoch1010']).items():
        rs=[r for r in cases if r['checkpoint'] in labels]
        groups[key]=dict(count=len(rs),same_paths=sum(r['same_path'] for r in rs),delta={k:st.mean(r['delta'][k] for r in rs) for k in fields},
            wins=sum(r['delta']['reward']>1e-5 for r in rs),ties=sum(abs(r['delta']['reward'])<=1e-5 for r in rs),losses=sum(r['delta']['reward'] < -1e-5 for r in rs))
        assert groups[key]['delta']==expected[key]['delta_vs_mean']
    report=dict(status='passed',cases=len(cases),fixed_fact_stages=stagecount,groups=groups,input_hashes_unchanged=True,
        checks=('100 unique keys; fixed percentiles1..0.01; initial/post exact; no original score values' if percentile else 'Original cache stage exact; score multiset exact')+'; round4 TW from independent fixed facts exact; ID tie sorting exact; paired config/AMP/alpha/beta/checkpoints/dispatch/vehicles; all unserved0',
        gate_review='DeadlineEnv delegates the original revealed_at/effective_time arguments unchanged to GuidedEnv; no online rollout in audit.',
        limitation=('Guidance is LLM-free; frozen Kim backbone is not. Score scale distribution differs from LLM, so this does not isolate correspondence alone.' if percentile else 'Distribution still comes from LLM and differs by event stage; this is not an LLM-free comparator.'),
        scope='CPU saved JSON/hash audit, no API/GPU/Test/training/environment or model execution.')
    OUT.mkdir(exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main('--percentile' in sys.argv)
