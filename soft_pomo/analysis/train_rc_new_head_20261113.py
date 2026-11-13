"""Fresh 1000-epoch RC Kim run with NEW group-head bias; no API/Test calls."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv
from experiments.concurrent_fleet_pair_kim.start_head_guidance import StartHeadGuidedPolicy
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.support import load, read
from experiments.concurrent_fleet_pair.io import write

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'server_export/rc101_pair100_200_l5_mask_kim_s42_46_20261111'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'result/rc_kim_newhead_v100_l5_s42_20261113')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.out.resolve()
    art=BUNDLE/'artifacts/vehicle100'
    cfg=read(art/'config.json');split=read(BUNDLE/'dataset/split.json');groups=read(art/'groups.json')
    assert cfg['trainer']['epochs']==1000
    assert cfg['environment']['late_penalty']==5 and cfg['environment']['protect_last_service_vehicle']
    assert split==read(art/'split.json') and len(split['train'])==20 and len(split['test'])==80
    caches={r['name']:read(art/'train_cache'/(r['name']+'.json')) for r in split['train']}
    for cache in caches.values():
        parse_response(cache['initial'],groups)
        if cache.get('post_disclosure') is not None:parse_response(cache['post_disclosure'],groups)
    # Use the exact server trainer, whose only evaluation change is to stop
    # after training_complete. This prevents accidental Test/API evaluation.
    trainer_file=BUNDLE/'experiments/concurrent_fleet_pair/train.py'
    spec=importlib.util.spec_from_file_location('rc_newhead_frozen_trainer',trainer_file)
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            c=caches[inst['name']]
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(c['initial'],groups),beta=5.)
            if c.get('post_disclosure') is not None:
                t=c['revealed_at_min']
                self.update_event_confidence(parse_response(c['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    trainer.TensorFleetEnv=CachedEnv;trainer.FleetPolicy=StartHeadGuidedPolicy;trainer.load=load
    files=[Path(__file__),trainer_file,art/'config.json',art/'groups.json',BUNDLE/'dataset/split.json']
    files+=list((art/'train_cache').glob('*.json'))
    files+=[ROOT/p for p in ('experiments/concurrent_fleet_pair_kim/start_head_guidance.py',
        'experiments/concurrent_fleet_pair_kim/run.py','experiments/concurrent_fleet_guided/guidance.py')]
    protocol=dict(seed=args.seed,epochs=1000,vehicle_cost=100,late=5,mask=True,
        new_head_beta=5.,routing_confidence_beta=5.,heads=[g[0] for g in groups],
        initialization='fresh model; no checkpoint transfer',
        treatment='Bias only NEW(group[0]) while dispatch_open; afterward unchanged legacy Kim confidence',
        cache='Frozen original Kim Train20 caches; only bias application changed; no new API',
        evaluation='Training only; causal Test evaluation remains pending',
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if args.resume:
        assert read(out/'experiment.json')==protocol,'Resume experiment mismatch'
    else:
        if (out/'experiment.json').exists() or (out/'run').exists():raise ValueError('Refusing duplicate run')
        write(out/'experiment.json',protocol)
        for p in files:
            dest=out/'provenance'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    write(out/'process.json',dict(pid=os.getpid(),stage='starting',seed=args.seed))
    train_args=argparse.Namespace(out=out/'run',seed=args.seed,vehicle_cost=100.,resume=args.resume,
        smoke_updates=0,config=art/'config.json',source=BUNDLE/'dataset',weights=[5],
        reward_denominator=1696.95,return_mask=True,no_advantage_std=False,
        new_customer_epsilon=.05,epsilon_final=.01,epsilon_zero_epoch=1000,resume_from=None)
    print(json.dumps(dict(stage='starting',pid=os.getpid(),seed=args.seed,epochs=1000,heads=protocol['heads'])),flush=True)
    trainer.run(train_args)
    write(out/'process.json',dict(pid=os.getpid(),stage='training_complete',evaluation_pending=True))

if __name__=='__main__':main()
