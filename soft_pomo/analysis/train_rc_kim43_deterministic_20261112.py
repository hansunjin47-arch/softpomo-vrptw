"""Deterministic NewKim43 matched replication; original failure remains separate."""
import copy
import math
import os
import random
from pathlib import Path
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.prepare_rc_kim_curriculum_20261112 import CurriculumGuidedPolicy, BUNDLE, RECEIVED
from analysis.train_rc_curriculum_confirmatory_20261112 import means, compare
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy, GuidedEnv
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_v3 import run as data_module

OUT=pilot.ROOT/'result/rc_kim43_curriculum_deterministic_20261112'
SOURCE=RECEIVED/'runs/vehicle100/seed43'
ART=BUNDLE/'artifacts/vehicle100'


def main():
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    assert not OUT.exists()
    cfg=pilot.read(SOURCE/'protocol.json')['configs']['5']
    rows=pilot.read(SOURCE/'split.json')['train']
    assert len(rows)==20 and cfg['trainer']['seed']==43
    assert cfg['environment']['late_penalty']==5 and cfg['environment']['protect_last_service_vehicle']
    assert cfg['environment']['normalize_advantage_std']
    assert abs(cfg['environment']['vehicle_penalty']/100-100/1696.95)<1e-12
    groups=pilot.read(ART/'groups.json')
    data_module.SOURCE=BUNDLE/'dataset'
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            cache=pilot.read(ART/'train_cache'/(inst['name']+'.json'))
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(cache['initial'],groups),beta=5.)
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min']
                self.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    pilot.trainer.TensorFleetEnv=CachedEnv
    pilot.trainer.load=load
    checkpoint=SOURCE/'late5/checkpoint-1000.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['epoch']==1000
    state=saved['python_rng'],saved['cpu_rng'],saved['cuda_rng']
    order_rng=random.Random();order_rng.setstate(saved['python_rng'])
    order=[order_rng.choice(cfg['trainer']['train_instances']) for _ in range(800)]
    lookup={r['name']:r for r in rows}
    assert set(lookup)==set(cfg['trainer']['train_instances'])
    files=[checkpoint,SOURCE/'split.json',SOURCE/'protocol.json',Path(__file__),
        pilot.ROOT/'analysis/prepare_rc_kim_curriculum_20261112.py',
        pilot.ROOT/'analysis/train_rc_k17_curriculum_20261112.py',
        pilot.ROOT/'analysis/train_rc_curriculum_confirmatory_20261112.py']
    files+=list((ART/'train_cache').glob('*.json'))+[ART/'groups.json']
    files+=list((BUNDLE/'dataset/data').glob('*'))
    files+=[pilot.ROOT/p for p in pilot.read(BUNDLE/'manifest.json')['sha256'] if p.startswith('experiments/') and p.endswith('.py')]
    hashes={str(p.relative_to(pilot.ROOT)):pilot.digest(p) for p in files if p.is_file()}
    OUT.mkdir()
    pilot.write(OUT/'preregistration.json',dict(seed=43,config=cfg,hashes=hashes,data_order=order,
        rule='max(1,floor(initial Train20 mean K)-1)',schedule='free40 versus constrained20/free20,final40 fixed',
        success='Train20 reward improves versus matched40 and U0; original1000 comparison also reported',
        budget='80epochs,1600updates,approximately55-65GPUminutes',stop='Exception,nonfinite,20 consecutive AMP skips,hash mismatch',
        scope='Deterministic matched replication,Train-only/API0; original nondeterministic failure preserved',
        numerical_settings=dict(deterministic=True,cublas_workspace=':4096:8',amp=True),
        target_fixed='initialTrain rule,not Test-informed'))
    initial=None
    for cls in (GuidedPolicy,CurriculumGuidedPolicy):
        model=cls(**cfg['model']).cuda().eval()
        model.target=None
        model.load_state_dict(saved['model_state_dict'],strict=True)
        pilot.restore(state)
        values=pilot.evaluate(model,rows,cfg,None)
        if initial is None:initial=values
        else:assert initial==values
        del model
    target=max(1,math.floor(means(initial)['vehicles'])-1)
    pilot.write(OUT/'initial_train20.json',initial)
    pilot.write(OUT/'protocol.json',dict(config=cfg,target=target,initial=means(initial),original_policy_train20_exact=True,hashes=hashes))
    print('TARGET_FIXED',target,means(initial),flush=True)
    for arm in ('control','curriculum'):
        dest=OUT/arm;dest.mkdir()
        model=CurriculumGuidedPolicy(**cfg['model']).cuda().train()
        model.load_state_dict(saved['model_state_dict'],strict=True)
        opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer'])
        opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
        scaler=torch.amp.GradScaler('cuda');scaler.load_state_dict(copy.deepcopy(saved['scaler']))
        pilot.restore(state)
        assert pilot.evaluate(model,rows,cfg,None)==initial
        skips=0
        for epoch in range(1,41):
            assert all(pilot.digest(pilot.ROOT/p)==h for p,h in hashes.items())
            model.target=target if arm=='curriculum' and epoch<=20 else None
            model.new_customer_epsilon=.01
            for batch_index in range(20):
                update=(epoch-1)*20+batch_index;name=order[update]
                inst,public=load(lookup[name]);model.fallback_count=0
                pilot.write(OUT/'status.json',dict(stage='training',pid=os.getpid(),arm=arm,epoch=epoch,update=update,target=model.target))
                stats,_=pilot.trainer.train_batch(model,opt,scaler,inst,public,cfg['environment'],min(128,2500-batch_index*128),100,True)
                assert math.isfinite(stats['loss']) and all(torch.isfinite(p).all() for p in model.parameters())
                skips=skips+1 if stats['optimizer_step_skipped'] else 0
                assert skips<20
                pilot.trainer.append(dest/'training.jsonl',dict(epoch=epoch,update=update+1,name=name,target=model.target,fallback_count=model.fallback_count,**stats))
            if epoch%10==0:
                values=pilot.evaluate(model,rows,cfg,None)
                pilot.write(dest/f'free_epoch{epoch:03d}.json',values)
                torch.save(dict(epoch=epoch,model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),scaler=scaler.state_dict(),rng=pilot.rng()),dest/f'checkpoint-{epoch}.pt')
                print('MILESTONE',arm,epoch,means(values),flush=True)
        del model,opt,scaler
        torch.cuda.empty_cache()
    arms={a:pilot.read(OUT/a/'free_epoch040.json') for a in ('control','curriculum')}
    pilot.write(OUT/'train_comparison.json',compare(arms))
    pilot.write(OUT/'original_comparison.json',{a:compare(dict(control=initial,curriculum=r)) for a,r in arms.items()})
    pilot.write(OUT/'status.json',dict(stage='complete',api_calls=0,total_epochs=80,validation_pending_approval=True,test_accessed=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        pilot.write(OUT/'failure.json',dict(error=repr(error)))
        raise
