"""Fixed100 persistence study, no API/Test or favorable checkpoint selection."""
import argparse
import copy
import math
import os
import time
from pathlib import Path
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.prepare_rc_kim_persistence_20261112 import OUT as DESIGN,SOURCES
from analysis.prepare_rc_kim_curriculum_20261112 import CurriculumGuidedPolicy,RECEIVED,BUNDLE
from analysis.train_rc_curriculum_confirmatory_20261112 import compare,means
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data_module

OUT=pilot.ROOT/'result/rc_kim_persistence_20261112'


def prepare():
    assert not OUT.exists()
    draft=pilot.read(DESIGN/'protocol_draft.json')
    hashes={}
    for seed,folder in SOURCES.items():
        assert pilot.read(folder/'status.json')['stage']=='complete'
        for arm in ('control','curriculum'):
            path=folder/arm/'checkpoint-40.pt'
            hashes[str(path.relative_to(pilot.ROOT))]=pilot.digest(path)
            state=torch.load(path,map_location='cpu',weights_only=False)
            assert state['epoch']==40 and len(state['rng'])==3
        for path,value in pilot.read(folder/'preregistration.json')['hashes'].items():
            assert path not in hashes or hashes[path]==value, f'Conflicting provenance hash: {path}'
            hashes[path]=value
    for path in (Path(__file__),DESIGN/'protocol_draft.json'):
        hashes[str(path.relative_to(pilot.ROOT))]=pilot.digest(path)
    draft['hashes']=hashes
    draft['success']='At fixed100 bothseeds curriculum-control reward>0 andU0; report40->100currchange andcontrolcatchup; no2seedgeneralization.'
    draft['preflight']='Each restored40 freeTrain20 must exactly equal saved40 results before any updates'
    OUT.mkdir();pilot.write(OUT/'protocol.json',draft)
    pilot.write(OUT/'status.json',dict(stage='prepared',api_calls=0))


def comparison(a,b):
    result=compare(dict(control=a,curriculum=b))
    d=result['groups']['all']['delta']
    result['rawcost_delta']=dict(vehicle=100*d['vehicles'],travel=d['travel_time'],late=5*d['total_late'])
    return result


def run():
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    p=pilot.read(OUT/'protocol.json')
    assert pilot.read(OUT/'status.json')['stage']=='prepared','No duplicate or implicit restart'
    assert all(pilot.digest(pilot.ROOT/f)==h for f,h in p['hashes'].items())
    data_module.SOURCE=BUNDLE/'dataset';art=BUNDLE/'artifacts/vehicle100';groups=pilot.read(art/'groups.json')
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            cache=pilot.read(art/'train_cache'/(inst['name']+'.json'))
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(cache['initial'],groups),beta=5.)
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min'];self.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    pilot.trainer.TensorFleetEnv=CachedEnv;pilot.trainer.load=load
    summaries={};started=time.monotonic()
    for seed,source in SOURCES.items():
        cfg=p['seeds'][str(seed)]['config'];order=p['seeds'][str(seed)]['continuation_order']
        rows=pilot.read(RECEIVED/f'runs/vehicle100/seed{seed}/split.json')['train'];lookup={r['name']:r for r in rows}
        folder=OUT/f'seed{seed}';folder.mkdir()
        for arm in ('control','curriculum'):
            dest=folder/arm;dest.mkdir();arm_start=time.monotonic()
            saved=torch.load(source/arm/'checkpoint-40.pt',map_location='cpu',weights_only=False)
            model=CurriculumGuidedPolicy(**cfg['model']).cuda().train();model.load_state_dict(saved['model_state_dict'],strict=True)
            opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer']);opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
            scaler=torch.amp.GradScaler('cuda');scaler.load_state_dict(copy.deepcopy(saved['scaler']))
            pilot.restore(saved['rng']);model.target=None
            actual=pilot.evaluate(model,rows,cfg,None);expected=pilot.read(source/arm/'free_epoch040.json')
            assert actual==expected,'Restored40 preflight mismatch'
            pilot.write(dest/'preflight40.json',dict(exact=True,means=means(actual)))
            skips=0
            for epoch in range(41,101):
                assert all(pilot.digest(pilot.ROOT/f)==h for f,h in p['hashes'].items())
                model.target=None;model.new_customer_epsilon=.01
                for batch_index in range(20):
                    index=(epoch-41)*20+batch_index;name=order[index]
                    inst,public=load(lookup[name]);model.fallback_count=0
                    pilot.write(OUT/'status.json',dict(stage='training',pid=os.getpid(),seed=seed,arm=arm,epoch=epoch,update=index+801))
                    stats,_=pilot.trainer.train_batch(model,opt,scaler,inst,public,cfg['environment'],min(128,2500-batch_index*128),100,True)
                    assert math.isfinite(stats['loss']) and all(torch.isfinite(t).all() for t in model.parameters())
                    skips=skips+1 if stats['optimizer_step_skipped'] else 0
                    assert skips<20
                    pilot.trainer.append(dest/'training.jsonl',dict(epoch=epoch,update=index+801,name=name,**stats))
                if epoch in (60,80,100):
                    values=pilot.evaluate(model,rows,cfg,None)
                    pilot.write(dest/f'free_epoch{epoch:03d}.json',values)
                    torch.save(dict(epoch=epoch,model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),scaler=scaler.state_dict(),rng=pilot.rng()),dest/f'checkpoint-{epoch}.pt')
                    print('MILESTONE',seed,arm,epoch,means(values),flush=True)
            pilot.write(dest/'complete.json',dict(seconds=time.monotonic()-arm_start))
            del model,opt,scaler;torch.cuda.empty_cache()
        result={}
        for epoch in (60,80,100):
            result[str(epoch)]=comparison(pilot.read(folder/'control'/f'free_epoch{epoch:03d}.json'),pilot.read(folder/'curriculum'/f'free_epoch{epoch:03d}.json'))
        result['change40to100']={arm:comparison(pilot.read(source/arm/'free_epoch040.json'),pilot.read(folder/arm/'free_epoch100.json')) for arm in ('control','curriculum')}
        result['versus_original']={arm:comparison(pilot.read(source/'initial_train20.json'),pilot.read(folder/arm/'free_epoch100.json')) for arm in ('control','curriculum')}
        pilot.write(folder/'comparison.json',result);summaries[str(seed)]=result
    pilot.write(OUT/'report.json',dict(seeds=summaries,both_seeds_pass=all(v['100']['passed'] for v in summaries.values()),caveat='Two seedsTrain only; no generalization claim',seconds=time.monotonic()-started))
    pilot.write(OUT/'status.json',dict(stage='complete',api_calls=0,test_accessed=False,additional_epochs=240))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    run() if args.run else prepare()
