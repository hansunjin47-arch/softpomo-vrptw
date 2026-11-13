"""Frozen Train-only late free-phase and numerical-sensitivity audit."""
import json
from statistics import mean
from pathlib import Path
import torch
from analysis import train_rc_k17_curriculum_20261112 as pilot
from analysis.train_rc_curriculum_confirmatory_20261112 import compare
from analysis.train_rc_kim_curriculum_20261112 import OUT as SOURCE, SOURCE as BASE, ART, BUNDLE
from analysis.prepare_rc_kim_curriculum_20261112 import CurriculumGuidedPolicy
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data_module

OUT=pilot.ROOT/'result/rc_kim_free_failure_20261112'
KEYS=('reward','vehicles','travel_time','total_late','unserved','best_rollout_reward','gradient_norm')


def aggregate(rows):
    weights=[68 if r['update']%20==0 else 128 for r in rows]
    return dict(**{k:sum(r[k]*w for r,w in zip(rows,weights) if r[k] is not None)/sum(w for r,w in zip(rows,weights) if r[k] is not None) for k in KEYS},
        amp_skips=sum(r['optimizer_step_skipped'] for r in rows),null_grad_count=sum(r['gradient_norm'] is None for r in rows),max_grad=max(r['gradient_norm'] for r in rows if r['gradient_norm'] is not None))


def main():
    assert pilot.read(SOURCE/'status.json')['stage']=='complete'
    assert not OUT.exists() or not list(OUT.iterdir())
    OUT.mkdir(exist_ok=True)
    logs=[json.loads(s) for s in (SOURCE/'curriculum/training.jsonl').read_text().splitlines()]
    byepoch={str(e):aggregate([r for r in logs if r['epoch']==e]) for e in range(21,41)}
    windows={}
    for lo,hi in ((21,30),(31,40),(26,30),(36,40)):
        selected=[r for r in logs if lo<=r['epoch']<=hi]
        percase={n:aggregate([r for r in selected if r['name']==n]) for n in sorted({r['name'] for r in selected})}
        windows[f'{lo}-{hi}']=dict(sampled=aggregate(selected),percase=percase,
            equally_weighted_cases={k:mean(r[k] for r in percase.values()) for k in KEYS})
    jumps=[]
    for before,after in zip(logs,logs[1:]):
        if 30<=after['epoch']<=40:
            jumps.append(dict(update=after['update'],epoch=after['epoch'],name=after['name'],
                previous_name=before['name'],reward_delta=after['reward']-before['reward'],
                reward=after['reward'],gradient_norm=after['gradient_norm'],amp_skip=after['optimizer_step_skipped']))
    pilot.write(OUT/'sampling_audit.json',dict(epochs=byepoch,windows=windows,
        largest_update_drops=sorted(jumps,key=lambda r:r['reward_delta'])[:15],
        caveat='Adjacent updates may use different instances; cannot attribute jumps solely to optimizer changes. best100 separate.'))
    cfg=pilot.read(SOURCE/'protocol.json')['config'];rows=pilot.read(BASE/'split.json')['train']
    groups=pilot.read(ART/'groups.json');data_module.SOURCE=BUNDLE/'dataset'
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            cache=pilot.read(ART/'train_cache'/(inst['name']+'.json'))
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(cache['initial'],groups),beta=5.)
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min']
                self.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    pilot.trainer.TensorFleetEnv=CachedEnv;pilot.trainer.load=load
    hashes=dict(pilot.read(SOURCE/'protocol.json')['hashes'])
    for path in (Path(__file__),*[SOURCE/a/f'checkpoint-{e}.pt' for a,e in (('control',40),('curriculum',30),('curriculum',40))]):
        hashes[str(path.relative_to(pilot.ROOT))]=pilot.digest(path)
    pilot.write(OUT/'protocol.json',dict(hashes=hashes,scope='Train20 AMPFalse60 inference,original beta5 caches and checkpoints; no official result replacement/API/Test/training'))
    torch.set_num_threads(1)
    report={}
    for arm,epoch in (('control',40),('curriculum',30),('curriculum',40)):
        assert all(pilot.digest(pilot.ROOT/p)==h for p,h in hashes.items())
        model=CurriculumGuidedPolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(torch.load(SOURCE/arm/f'checkpoint-{epoch}.pt',map_location='cpu',weights_only=False)['model_state_dict'])
        result=[]
        with torch.no_grad():
            for row in rows:
                inst,public=load(row)
                env,_,_=pilot.trainer.collect(model,inst,public,cfg['environment'],1,False,greedy=True)
                result.append(dict(name=row['name'],**{k:float(v[0]) for k,v in env.metrics().items()}))
        pilot.write(OUT/f'{arm}_{epoch}_fp32.json',result)
        report[f'{arm}_{epoch}']=compare(dict(control=pilot.read(SOURCE/arm/f'free_epoch{epoch:03d}.json'),curriculum=result))
        print(arm,epoch,report[f'{arm}_{epoch}']['means'],flush=True)
        del model
    pilot.write(OUT/'fp32_comparison.json',report)
    pilot.write(OUT/'status.json',dict(stage='complete',api_calls=0,test_accessed=False))


if __name__=='__main__':main()
