"""Destroy group-mean correspondence while keeping matched state-level scale."""
import hashlib
import json
import random
import statistics

import torch

from analysis import diagnose_kim_fusion_20261110 as experiment
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy

OUT=experiment.suite.ROOT/'result/kim_group_mapping_20261111'
CONTROLS=('pooled_group_mean','permuted_group_mean')
PERMUTATIONS={}


def donor_ids(ids):
    ids=tuple(ids)
    if ids not in PERMUTATIONS:
        payload='20261111:'+','.join(map(str,ids))
        rng=random.Random(int.from_bytes(hashlib.sha256(payload.encode('ascii')).digest(),'big'))
        donors=list(ids)
        rng.shuffle(donors)
        PERMUTATIONS[ids]=tuple(donors)
    return PERMUTATIONS[ids]


class MappingPolicy(GuidedPolicy):
    fusion_alpha=.25
    control=None

    def probabilities(self,frame,starts,encodings,env):
        assert not self.training and env.count==1 and self.decoder.logit_clipping==10.
        raw=FleetPolicy.probabilities(self,frame,starts,encodings,env)
        logits=raw.double().log()*self.fusion_alpha
        bias=torch.zeros_like(logits)
        g=env.guidance
        if not bool(env.dispatch_open[0]) and not bool(env.done[0]):
            assert frame['vehicles'].shape[1]==1
            anchor=int(g.vehicle_group.gather(1,frame['vehicles'])[0,0])
            scores=g.initial
            for effective,target,updated in g.updates:
                if bool(target[0] & (env.time[0]>=effective[0])):
                    scores=updated
            valid=frame['mask'][0,env.n:].clone()
            valid[0]=False
            membership=torch.arange(len(g.groups),device=env.device)[:,None]==g.membership[None,:]
            feasible=membership & valid[None,:]
            counts=feasible.sum(1)
            means=(scores.double()*feasible).sum(1)/counts.clamp_min(1)
            ids=counts.nonzero().flatten().tolist()
            assigned=means.clone()
            if ids:
                if self.control=='pooled_group_mean':
                    assigned[ids]=means[ids].mean()
                    assert torch.allclose(assigned[ids].mean(),means[ids].mean(),atol=1e-14,rtol=0)
                elif self.control=='permuted_group_mean':
                    donors=donor_ids(ids)
                    assigned[ids]=means[list(donors)]
                    assert torch.equal(assigned[ids].sort().values,means[ids].sort().values)
                else:
                    raise ValueError(self.control)
            selected=(g.membership==anchor)&valid if anchor>=0 else torch.zeros_like(valid)
            value=assigned[anchor] if anchor>=0 and bool(counts[anchor]) else means.new_tensor(0.)
            base=means[anchor] if anchor>=0 and bool(counts[anchor]) else means.new_tensor(0.)
            bias[0,env.n:]=g.beta*value*selected
            assert bias[0,env.n]==0 and (bias[0,:env.n]==0).all()
            if len(ids)<=1 and ids:
                assert torch.equal(assigned[ids],means[ids])
            if g.beta:
                self.last_diagnostics=dict(anchor_group=anchor,available_groups=len(ids),
                    valid_anchor_customers=int(selected.sum()),original_anchor_mean=float(base),
                    assigned_anchor_mean=float(value),mean_bonus_delta=float(g.beta*(value-base)),
                    group_mean_average=float(means[ids].mean()) if ids else 0.)
        elif g.beta:
            self.last_diagnostics=dict(anchor_group=-1,available_groups=0,valid_anchor_customers=0,
                original_anchor_mean=0.,assigned_anchor_mean=0.,mean_bonus_delta=0.,group_mean_average=0.)
        p=(logits+bias).softmax(1)
        assert torch.isfinite(p).all() and torch.equal(p==0,raw==0)
        if g.beta==0 or bool(env.dispatch_open[0]):
            assert torch.equal(p.argmax(1),raw.argmax(1))
        return p


def configure_scale(model,alpha):
    model.fusion_alpha=alpha


def main():
    assert donor_ids([3])==(3,)
    assert sorted(donor_ids([0,2,5]))==[0,2,5]
    assert donor_ids([0,2,5])==donor_ids([0,2,5])
    OUT.mkdir(exist_ok=True)
    sources=[experiment.suite.RECEIVED/f'runs/seed{s}/protocol.json' for s in range(42,47)]
    sources += [experiment.suite.RECEIVED/'runs/seed44/split.json',experiment.suite.BUNDLE/'artifacts/groups.json']
    sources += sorted((experiment.suite.BUNDLE/'artifacts/train_cache').glob('*.json'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    experiment.write(OUT/'inputs.json',dict(sha256=hashes,budget=240,controls=CONTROLS,
        hypothesis='Does the correspondence of mean bonus to anchor group matter beyond matched common/group-distribution scale?',
        feasible_definition='Feasible customers for the currently selected vehicle, not availability across all vehicles',
        pooling='Unweighted mean of nonempty group feasible-customer means, recomputed each state; not episode-fixed scalar',
        permutation='sha256 ASCII 20261111:comma-separated sorted available group IDs seeds Python Random shuffle; cached solely by available IDs',
        restrictions='Frozen R101 Train20 only; no API/Test/training/server changes'))
    experiment.configure_scale=configure_scale
    experiment.suite.GuidedPolicy=MappingPolicy
    experiment.BASELINE_ROOT=experiment.suite.ROOT/'result/kim_group_mean_20261111'
    experiment.BASELINE_GUIDED_VARIANT='group_mean'
    for index,control in enumerate(CONTROLS):
        experiment.write(OUT/'status.json',dict(stage='running',control=control,completed=index*120,target=240))
        MappingPolicy.control=control
        experiment.OUT=OUT/control
        experiment.IMPLEMENTATION=f'Original AMP q, float64 alpha.25 beta5 {control}; current selected vehicle feasible groups; group correspondence control'
        experiment.VARIANTS=[(control,.25,'original',5.)]
        experiment.main()
        protocol=experiment.suite.read(experiment.OUT/'protocol.json')
        protocol['off_control']='Per-step beta0 argmax and dispatch controls; no extra rollout'
        protocol['controls']='Exact masks, nonempty group multiset/mean preservation, empty anchor0, singleton identity, input/checkpoint hashes'
        experiment.write(experiment.OUT/'protocol.json',protocol)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in sources)
    cases=[]
    for control in CONTROLS:
        for path in (OUT/control).glob(f'seed*/{control}/*.json'):
            r=experiment.suite.read(path)
            label=r['checkpoint']
            base=experiment.suite.read(experiment.BASELINE_ROOT/label/'group_mean'/path.name)
            full=experiment.suite.read(experiment.suite.ROOT/'result/kim_fusion_stable_20261110'/label/'alpha025'/path.name)
            fields=('reward','vehicles','travel_time','total_late','unserved')
            dispatch=lambda v:[t['action'] for t in v['trace'] if t['dispatch_open']]
            assert dispatch(r)==dispatch(base)==dispatch(full)
            assert r['metrics']['vehicles']==base['metrics']['vehicles']==full['metrics']['vehicles']
            steps=[t for t in r['trace'] if not t['dispatch_open']]
            cases.append(dict(control=control,checkpoint=label,name=r['name'],
                metrics={k:r['metrics'][k] for k in fields},
                delta_vs_mean={k:r['metrics'][k]-base['metrics'][k] for k in fields},
                delta_vs_full={k:r['metrics'][k]-full['metrics'][k] for k in fields},
                changed_path=[t['action'] for t in r['trace']]!=[t['action'] for t in base['trace']],
                bonus_stats={k:statistics.mean(t[k] for t in steps) for k in
                    ('original_anchor_mean','assigned_anchor_mean','mean_bonus_delta','available_groups','valid_anchor_customers')}))
    assert len(cases)==240
    groups={'normal_four':['seed42','seed43','seed45','seed46'],'failed44':['seed44'],
        'recovered44':['seed44_epoch1010'],'original_five':[f'seed{s}' for s in range(42,47)]}
    aggregated={}
    for control in CONTROLS:
        aggregated[control]={}
        for group,labels in groups.items():
            subset=[r for r in cases if r['control']==control and r['checkpoint'] in labels]
            aggregated[control][group]=dict(count=len(subset),changed_paths=sum(r['changed_path'] for r in subset),
                **{f:{k:statistics.mean(r[f][k] for r in subset) for k in fields} for f in ('metrics','delta_vs_mean','delta_vs_full')},
                bonus_stats={k:statistics.mean(r['bonus_stats'][k] for r in subset) for k in subset[0]['bonus_stats']})
    experiment.write(OUT/'paired.json',dict(cases=cases,groups=aggregated))
    experiment.write(OUT/'status.json',dict(stage='complete',completed=240,target=240,input_hashes_unchanged=True,
        checkpoints_unchanged=True,paired_dispatch_vehicles_unchanged=True))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        experiment.write(OUT/'status.json',dict(stage='failed',error=repr(error)))
        raise
