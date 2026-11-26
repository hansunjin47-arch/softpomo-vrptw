"""Local matched fleet-context Proposed training; cached Train20, no API calls."""
import argparse, hashlib, importlib.util, json, os, shutil
from pathlib import Path
from experiments.concurrent_fleet_pair_kim.fleet_context import FleetContextKimPolicy
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.support import load, read
from experiments.concurrent_fleet_pair.io import write

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'server_export/rc_pure_fleetctx_v100_l5_10_s46_20261124'

def setup(late, *, portable=False):
    art=ROOT/('cache/rc_rl_lkh_l5_20261122' if late==5 else 'cache/rc_g5_l10_promptfix_20261122')
    source=BUNDLE/'dataset'; config=BUNDLE/f'config_l{late}.json'
    if portable:
        assert late == 5
        art = ROOT / 'experiment_inputs/fleet_context/b_guided_l5'
        source = ROOT / 'experiment_inputs/new_head/rc101/dataset'
        config = ROOT / 'experiment_inputs/fleet_context/config_l5.json'
    cfg=read(config);split=read(source/'split.json');groups=read(art/'groups.json')
    assert split==read(art/'split.json')
    assert len(split['train'])==20 and len(split['test'])==80
    assert cfg['environment']['late_penalty']==late and cfg['trainer']['epochs']==1000
    for flag in ('require_dispatch_capacity','protect_last_service_vehicle','normalize_advantage_std'):
        assert cfg['environment'][flag]
    caches={r['name']:read(art/'train_cache'/(r['name']+'.json')) for r in split['train']}
    for c in caches.values():
        parse_response(c['initial'],groups)
        if c.get('post_disclosure') is not None:parse_response(c['post_disclosure'],groups)
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            c=caches[inst['name']]
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(c['initial'],groups),beta=5.)
            if c.get('post_disclosure') is not None:
                t=c['revealed_at_min']
                self.update_event_confidence(parse_response(c['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    return art,source,config,cfg,CachedEnv

def main():
    p=argparse.ArgumentParser();p.add_argument('--late',type=int,choices=(5,10),required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    art,source,config,cfg,env=setup(a.late)
    out=ROOT/f'result/rc_kim_fleetctx_v100_l{a.late}_s46_20261124'
    trainer_file=BUNDLE/'analysis/maxconf_stability_trainer_20261116.py'
    spec=importlib.util.spec_from_file_location('matched_fleet_trainer',trainer_file)
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    trainer.FleetPolicy=FleetContextKimPolicy;trainer.TensorFleetEnv=env;trainer.load=load
    original=trainer.train_batch;calls=0
    def batch(model,opt,*args,**kw):
        nonlocal calls
        epoch=calls//20+1;factor=1. if epoch<=500 else max(.1,1.-.9*(epoch-500)/500.)
        for pg in opt.param_groups:pg['lr']=1e-4*factor
        summary,detail=original(model,opt,*args,**kw)
        summary.update(learning_rate=opt.param_groups[0]['lr'],new_head_beta=5.,routing_beta=5.)
        calls+=1;return summary,detail
    trainer.train_batch=batch
    files=[Path(__file__),trainer_file,config,source/'split.json',art/'groups.json']+list((art/'train_cache').glob('*.json'))
    files += [ROOT/f for f in ('experiments/concurrent_fleet_pair/fleet_context.py','experiments/concurrent_fleet_pair_kim/fleet_context.py','experiments/concurrent_fleet_pair_kim/confidence_head_guidance.py','experiments/concurrent_fleet_pair_kim/run.py','experiments/concurrent_fleet_pair/policy.py','experiments/concurrent_fleet_pair/env.py')]
    protocol=dict(method='Proposed + fleet context',seed=46,vehicle_weight=100,travel_weight=1,late=a.late,epochs=1000,new_head_beta=5,routing_beta=5,initialization='fresh; zero-initialized fleet residual',learning_rate='1e-4 through500; linear to1e-5 at1000',cache=str(art),reference='RL+LKH' if a.late==5 else 'RL only',evaluation='Training only; causal Test80 pending, no API calls',hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    if a.validate_only:print(json.dumps(protocol));return
    assert not out.exists(),'Refusing duplicate output'
    write(out/'experiment.json',protocol)
    for f in files:
        dest=out/'provenance'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    write(out/'process.json',dict(pid=os.getpid(),stage='training'))
    print(json.dumps(dict(stage='starting',late=a.late,seed=46,pid=os.getpid())),flush=True)
    trainer.run(argparse.Namespace(out=out/'run',seed=46,vehicle_cost=100.,resume=False,smoke_updates=0,config=config,source=source,weights=[a.late],reward_denominator=1696.95,return_mask=True,no_advantage_std=False,new_customer_epsilon=.05,epsilon_final=.01,epsilon_zero_epoch=1000,resume_from=None))
    assert read(out/f'run/late{a.late}/training_complete.json')['epoch']==1000
    write(out/'process.json',dict(pid=os.getpid(),stage='training_complete',causal_test_pending=True))

if __name__=='__main__':main()
