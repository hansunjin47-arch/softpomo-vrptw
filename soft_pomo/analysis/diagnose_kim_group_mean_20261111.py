"""One approved 120-rollout valid-group mean-only control on frozen Train20."""
import hashlib
import statistics

import torch

from analysis import diagnose_kim_fusion_20261110 as experiment
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy

OUT = experiment.suite.ROOT/'result/kim_group_mean_20261111'


class MeanOnlyPolicy(GuidedPolicy):
    fusion_alpha = .25

    def probabilities(self, frame, starts, encodings, env):
        assert not self.training and self.decoder.logit_clipping == 10.
        raw = FleetPolicy.probabilities(self, frame, starts, encodings, env)
        guidance = env.guidance
        group = guidance.vehicle_group.gather(1, frame['vehicles'])
        confidence = guidance.initial[group.clamp(min=0)]
        for effective, target, scores in guidance.updates:
            eligible = target & (env.time >= effective)
            confidence = torch.where(eligible[:, None, None], scores[group.clamp(min=0)], confidence)
        active = (group >= 0) & ~env.dispatch_open[:, None] & ~env.done[:, None]
        membership = guidance.membership[None, None, :] == group[:, :, None]
        membership[:, :, 0] = False
        valid = frame['mask'][:, env.n:].reshape_as(confidence)
        selected = membership & active[:, :, None] & valid
        count = selected.sum(-1, keepdim=True)
        mean = (confidence.double()*selected).sum(-1, keepdim=True)/count.clamp_min(1)
        full = guidance.beta*confidence.double()*selected
        bonus = guidance.beta*mean*selected
        residual = (full-bonus).masked_fill(~selected, 0.)
        assert torch.isfinite(bonus).all()
        assert (bonus.masked_select((count==0).expand_as(bonus))==0).all()
        assert (residual.masked_select((count<=1).expand_as(residual))==0).all()
        assert torch.allclose(full,bonus+residual,rtol=0,atol=1e-14)
        logits = raw.double().log()*self.fusion_alpha
        bias = torch.cat([bonus.new_zeros(env.count,env.n),bonus.flatten(1)],1)
        p = (logits+bias).softmax(1)
        assert torch.isfinite(p).all()
        assert torch.equal(p==0,raw==0)
        if guidance.beta==0. or bool(env.dispatch_open.all()):
            assert torch.equal(p.argmax(1),raw.argmax(1))
        return p


def configure_scale(model,alpha):
    model.fusion_alpha=alpha


def paired_summary():
    records=[]
    labels=[f'seed{s}' for s in range(42,47)]+['seed44_epoch1010']
    for label in labels:
        for path in (OUT/label/'group_mean').glob('*.json'):
            value=experiment.suite.read(path)
            full=experiment.suite.read(experiment.suite.ROOT/'result/kim_fusion_stable_20261110'/label/'alpha025'/path.name)
            residual=experiment.suite.read(experiment.suite.ROOT/'result/kim_group_residual_20261111'/label/'group_residual'/path.name)
            assert value['metrics']['vehicles']==full['metrics']['vehicles']==residual['metrics']['vehicles']
            dispatch=lambda row:[t['action'] for t in row['trace'] if t['dispatch_open']]
            assert dispatch(value)==dispatch(full)==dispatch(residual)
            fields=('reward','vehicles','travel_time','total_late','unserved')
            records.append(dict(checkpoint=label,name=value['name'],mean_only={k:value['metrics'][k] for k in fields},
                delta_vs_full={k:value['metrics'][k]-full['metrics'][k] for k in fields},
                delta_vs_residual={k:value['metrics'][k]-residual['metrics'][k] for k in fields},
                full_path_equal=[t['action'] for t in value['trace']]==[t['action'] for t in full['trace']]))
    assert len(records)==120
    groups={'original_five':labels[:5],'normal_four':['seed42','seed43','seed45','seed46'],
        'failed44':['seed44'],'recovered44':['seed44_epoch1010']}
    aggregate={}
    for name, members in groups.items():
        subset=[r for r in records if r['checkpoint'] in members]
        aggregate[name]=dict(count=len(subset),**{field:{k:statistics.mean(r[field][k] for r in subset)
            for k in fields} for field in ('mean_only','delta_vs_full','delta_vs_residual')},
            identical_full_paths=sum(r['full_path_equal'] for r in subset))
    experiment.write(OUT/'paired.json',dict(cases=records,groups=aggregate))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    files=[experiment.suite.RECEIVED/f'runs/seed{s}/protocol.json' for s in range(42,47)]
    files += [experiment.suite.RECEIVED/'runs/seed44/split.json',experiment.suite.BUNDLE/'artifacts/groups.json']
    files += sorted((experiment.suite.BUNDLE/'artifacts/train_cache').glob('*.json'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    experiment.write(OUT/'inputs.json',dict(sha256=hashes,budget=120,
        comparator='Stored stable alpha.25 beta5 full and valid-group residual trajectories',
        intervention='Current valid anchor-group mean only; invalid/group-external/depot/NEW bias0; no server/API/Test/training'))
    experiment.OUT=OUT
    experiment.IMPLEMENTATION='Original AMP q, float64 alpha.25 beta5 valid anchor-group mean-only bias; paired against saved full and residual'
    experiment.BASELINE_ROOT=experiment.suite.ROOT/'result/kim_fusion_stable_20261110'
    experiment.BASELINE_GUIDED_VARIANT='alpha025'
    experiment.configure_scale=configure_scale
    experiment.VARIANTS=[('group_mean',.25,'original',5.)]
    experiment.suite.GuidedPolicy=MeanOnlyPolicy
    try:
        experiment.main()
        protocol=experiment.suite.read(OUT/'protocol.json')
        protocol['off_control']='No extra off rollout; per-step off argmax invariant and previous stable controls reused'
        protocol['controls']='valid0/1, full=mean+residual, exact masks, dispatch argmax, paired vehicle/dispatch trace, input/checkpoint hashes'
        experiment.write(OUT/'protocol.json',protocol)
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files)
        paired_summary()
        experiment.write(OUT/'status.json',dict(stage='complete',completed=120,target=120,
            checkpoints_unchanged=True,configs_caches_unchanged=True,paired_validation=True))
    except Exception as error:
        experiment.write(OUT/'status.json',dict(stage='failed',error=repr(error)))
        raise
