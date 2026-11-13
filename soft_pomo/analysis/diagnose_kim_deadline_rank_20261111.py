"""Match confidence distribution but replace customer correspondence by deadline rank."""
import hashlib
import statistics

from analysis import diagnose_kim_fusion_20261110 as experiment
from analysis.diagnose_kim_group_mean_20261111 import MeanOnlyPolicy,configure_scale
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv

OUT=experiment.suite.ROOT/'result/kim_deadline_rank_20261111'
VARIANT='deadline_rank_matched'
INTERVENTION='Stage-specific 100-score multiset sorted descending and assigned by rounded4 deadline ascending/customerID tie; current feasible anchor mean-only alpha.25 beta5'
LIMITATIONS='Not LLM-free: uses LLM score distributions. Event reveal gate unchanged but event-customer correspondence removed. Frozen R101 Train20 only; no API/Test/training/server changes.'


class DeadlineEnv(GuidedEnv):
    def __init__(self,inst,count,params,device='cuda',*,groups,scores,beta=5.):
        self.deadlines={i+1:round(float(x)*float(inst['T']),4) for i,x in enumerate(inst['node_tw_close'].reshape(-1).tolist())}
        assert sorted(self.deadlines)==list(range(1,101))
        self.deadline_order=sorted(self.deadlines,key=lambda n:(self.deadlines[n],n))
        self.deadline_name=inst['name']
        self.score_phase=0
        super().__init__(inst,count,params,device,groups=groups,scores=self.remap(scores),beta=beta)

    def remap(self,scores):
        original={int(n):float(v) for row in scores.values() for n,v in row.items()}
        assert sorted(original)==list(range(1,101))
        ordered=sorted(original.values(),reverse=True)
        mapping=dict(zip(self.deadline_order,ordered))
        assert sorted(mapping.values())==sorted(original.values())
        assert all(mapping[a]>=mapping[b] for a,b in zip(self.deadline_order,self.deadline_order[1:]))
        transformed={g:{n:mapping[int(n)] for n in row} for g,row in scores.items()}
        experiment.write(OUT/'fixed_facts'/self.deadline_name/f'phase{self.score_phase}.json',dict(
            deadline_minutes_rounded4=self.deadlines,order=self.deadline_order,original_scores=original,
            reassigned_scores=mapping,stage_multiset_preserved=True,monotonic=True,
            rule='earliest close gets highest score; equal rounded close ties by customer ID'))
        self.score_phase+=1
        return transformed

    def update_event_confidence(self,scores,*,revealed_at_min,effective_time_min):
        super().update_event_confidence(self.remap(scores),revealed_at_min=revealed_at_min,effective_time_min=effective_time_min)


def main():
    OUT.mkdir(exist_ok=True)
    files=[experiment.suite.RECEIVED/f'runs/seed{s}/protocol.json' for s in range(42,47)]
    files += [experiment.suite.RECEIVED/'runs/seed44/split.json',experiment.suite.BUNDLE/'artifacts/groups.json']
    files += sorted((experiment.suite.BUNDLE/'artifacts/train_cache').glob('*.json'))
    split=experiment.suite.read(experiment.suite.RECEIVED/'runs/seed44/split.json')
    files += [experiment.suite.BUNDLE/'dataset/data'/(r['name']+'.txt') for r in split['train']]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    experiment.write(OUT/'inputs.json',dict(sha256=hashes,budget=120,
        intervention=INTERVENTION,limitations=LIMITATIONS))
    experiment.OUT=OUT
    experiment.IMPLEMENTATION=INTERVENTION
    experiment.configure_scale=configure_scale
    experiment.suite.GuidedPolicy=MeanOnlyPolicy
    experiment.suite.GuidedEnv=DeadlineEnv
    experiment.VARIANTS=[(VARIANT,.25,'original',5.)]
    experiment.BASELINE_ROOT=experiment.suite.ROOT/'result/kim_group_mean_20261111'
    experiment.BASELINE_GUIDED_VARIANT='group_mean'
    experiment.main()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files)
    protocol=experiment.suite.read(OUT/'protocol.json')
    protocol['off_control']='Per-step beta0/dispatch argmax invariant, no additional off rollout'
    protocol['controls']='Stage score multiset, deadline monotonicity/ID ties, unique100 customers, valid0/1, exact masks, paired vehicles/dispatch, input hashes'
    experiment.write(OUT/'protocol.json',protocol)
    cases=[]
    fields=('reward','vehicles','travel_time','total_late','unserved')
    for path in OUT.glob(f'seed*/{VARIANT}/*.json'):
        r=experiment.suite.read(path);label=r['checkpoint']
        old=experiment.suite.read(experiment.BASELINE_ROOT/label/'group_mean'/path.name)
        full=experiment.suite.read(experiment.suite.ROOT/'result/kim_fusion_stable_20261110'/label/'alpha025'/path.name)
        dispatch=lambda v:[t['action'] for t in v['trace'] if t['dispatch_open']]
        assert dispatch(r)==dispatch(old)==dispatch(full)
        assert r['metrics']['vehicles']==old['metrics']['vehicles']==full['metrics']['vehicles']
        cases.append(dict(checkpoint=label,name=r['name'],metrics={k:r['metrics'][k] for k in fields},
            delta_vs_mean={k:r['metrics'][k]-old['metrics'][k] for k in fields},
            delta_vs_full={k:r['metrics'][k]-full['metrics'][k] for k in fields},
            changed_path=[t['action'] for t in r['trace']]!=[t['action'] for t in old['trace']]))
    assert len(cases)==120
    groups={}
    for group,labels in {'normal_four':['seed42','seed43','seed45','seed46'],
        'failed44':['seed44'],'recovered44':['seed44_epoch1010'],'original_five':[f'seed{s}' for s in range(42,47)]}.items():
        selected=[r for r in cases if r['checkpoint'] in labels]
        groups[group]=dict(count=len(selected),changed_paths=sum(r['changed_path'] for r in selected),
            **{f:{k:statistics.mean(r[f][k] for r in selected) for k in fields} for f in ('metrics','delta_vs_mean','delta_vs_full')},
            win_tie_loss={v:sum(r['delta_vs_mean']['reward']>1e-5 if v=='win' else r['delta_vs_mean']['reward'] < -1e-5 if v=='loss' else abs(r['delta_vs_mean']['reward'])<=1e-5 for r in selected) for v in ('win','tie','loss')})
    experiment.write(OUT/'paired.json',dict(cases=cases,groups=groups))
    experiment.write(OUT/'status.json',dict(stage='complete',completed=120,target=120,input_hashes_unchanged=True,
        checkpoints_unchanged=True,vehicles_dispatch_unchanged=True))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        experiment.write(OUT/'status.json',dict(stage='failed',error=repr(error)))
        raise
