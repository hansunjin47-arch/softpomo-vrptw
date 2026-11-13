"""Exact-state free31-40 replay for failure localization, no interventions."""
import copy
import json
import math
import os
from pathlib import Path
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.train_rc_kim_curriculum_20261112 import OUT as SOURCE, SOURCE as BASE, ART, BUNDLE
from analysis.prepare_rc_kim_curriculum_20261112 import CurriculumGuidedPolicy
from analysis.train_rc_curriculum_confirmatory_20261112 import compare,means
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data_module

OUT=pilot.ROOT/'result/rc_kim_free_tail_replay_20261112'
METRICS=('reward','vehicles','total_late','gradient_norm','optimizer_step_skipped')


def main():
    torch.set_num_threads(1)
    assert not OUT.exists()
    protocol=pilot.read(SOURCE/'preregistration.json')
    cfg=protocol['config'];order=protocol['data_order']
    hashes=dict(protocol['hashes'])
    checkpoint=SOURCE/'curriculum/checkpoint-30.pt'
    for path in (checkpoint,Path(__file__)):
        hashes[str(path.relative_to(pilot.ROOT))]=pilot.digest(path)
    assert all(pilot.digest(pilot.ROOT/p)==h for p,h in hashes.items())
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['epoch']==30 and len(saved['rng'])==3
    groups=pilot.read(ART/'groups.json');data_module.SOURCE=BUNDLE/'dataset'
    rows=pilot.read(BASE/'split.json')['train'];lookup={r['name']:r for r in rows}
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            cache=pilot.read(ART/'train_cache'/(inst['name']+'.json'))
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(cache['initial'],groups),beta=5.)
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min']
                self.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    pilot.trainer.TensorFleetEnv=CachedEnv;pilot.trainer.load=load
    model=CurriculumGuidedPolicy(**cfg['model']).cuda().train()
    model.load_state_dict(saved['model_state_dict'],strict=True)
    opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer'])
    opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
    scaler=torch.amp.GradScaler('cuda');scaler.load_state_dict(copy.deepcopy(saved['scaler']))
    pilot.restore(saved['rng'])
    now=pilot.rng()
    assert now[0]==saved['rng'][0] and torch.equal(now[1],saved['rng'][1]) and torch.equal(now[2],saved['rng'][2])
    OUT.mkdir()
    pilot.write(OUT/'protocol.json',dict(hashes=hashes,config=cfg,data_order=order,
        schedule='Replay updates601-800 from savedcheckpoint30; targetNone epsilon.01 originalLR',
        snapshots='After update779..800; each snapshot supplies prior/after state of780..800',
        no_extra_probes='No optional Train2 diagnostics during replay, to minimize kernelpath changes',
        scope='Reproduction diagnostic,not performance selection; API0/Test0;10epochs',
        restored_rng_exact=True,budget='~6GPUminutes,22snapshots~300MB'))
    old={r['update']:r for r in [json.loads(s) for s in (SOURCE/'curriculum/training.jsonl').read_text().splitlines()]}
    differences=[];first=None
    model.target=None;model.new_customer_epsilon=.01
    for epoch in range(31,41):
        assert all(pilot.digest(pilot.ROOT/p)==h for p,h in hashes.items())
        for batch_index in range(20):
            update=(epoch-1)*20+batch_index+1;name=order[update-1]
            assert old[update]['name']==name
            inst,public=load(lookup[name]);model.fallback_count=0
            pilot.write(OUT/'status.json',dict(stage='replaying',pid=os.getpid(),epoch=epoch,update=update))
            stats,_=pilot.trainer.train_batch(model,opt,scaler,inst,public,cfg['environment'],min(128,2500-batch_index*128),100,True)
            assert math.isfinite(stats['loss'])
            deltas={k: None if stats[k] is None or old[update][k] is None else float(stats[k])-float(old[update][k]) for k in METRICS}
            equal=all(stats[k]==old[update][k] for k in METRICS)
            if not equal and first is None:
                first=dict(update=update,name=name,delta=deltas)
                print('FIRST_DIFFERENCE',first,flush=True)
            differences.append(dict(update=update,name=name,exact=equal,delta=deltas))
            pilot.trainer.append(OUT/'training.jsonl',dict(epoch=epoch,update=update,name=name,**stats))
            if update>=779:
                torch.save(dict(update=update,epoch=epoch,model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),scaler=scaler.state_dict(),rng=pilot.rng()),OUT/f'update-{update}.pt')
        print('REPLAY_EPOCH',epoch,stats['reward'],flush=True)
    final=pilot.evaluate(model,rows,cfg,None)
    pilot.write(OUT/'free_epoch040.json',final)
    report=compare(dict(control=pilot.read(SOURCE/'curriculum/free_epoch040.json'),curriculum=final))
    pilot.write(OUT/'report.json',dict(first_difference=first,updates=differences,exact_updates=sum(r['exact'] for r in differences),final_comparison=report,
        caveat='Saved-state replay; checkpoint IO added but no probe calls; GPU nondeterminism may prevent exact equality.'))
    pilot.write(OUT/'status.json',dict(stage='complete',epochs=10,api_calls=0,test_accessed=False))
    print('FINAL',means(final),'exactupdates',sum(r['exact'] for r in differences),flush=True)


if __name__=='__main__':main()
