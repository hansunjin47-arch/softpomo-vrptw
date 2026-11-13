"""Paired train-only continuation: preserve Adam vs reset first moment once."""
import copy
import hashlib
import random
import statistics
import torch
from analysis.evaluate_r101_mask_kim_20261109 import BUNDLE, RECEIVED
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_pair import train
from experiments.concurrent_fleet_v3 import run as data_module
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_pair.io import write

OUT=ROOT/'result/r101_kim44_short_control_20261109'


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    data_module.SOURCE=BUNDLE/'dataset'
    run=RECEIVED/'runs/seed44'
    checkpoint=run/'late5/checkpoint-1000.pt'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=read(run/'protocol.json')['configs']['5']
    rows=read(run/'split.json')['train']
    groups=read(BUNDLE/'artifacts/groups.json')
    class CachedEnv(GuidedEnv):
        def __init__(self,inst,count,params,device='cuda'):
            cache=read(BUNDLE/'artifacts/train_cache'/(inst['name']+'.json'))
            super().__init__(inst,count,params,device,groups=groups,scores=parse_response(cache['initial'],groups))
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min'];self.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
    train.TensorFleetEnv=CachedEnv
    data={r['name']:load(r) for r in rows}
    order=[r['name'] for r in rows]
    random.Random(20261109).shuffle(order)
    batch=cfg['trainer']['train_batch_size'];rollouts=cfg['environment']['pomo_size']
    write(OUT/'protocol.json',dict(checkpoint_sha256=digest,config=cfg,order=order,
        updates_per_arm=20,batch=batch,rollouts=rollouts,amp=True,epsilon=.01,
        reset='exp_avg only, once before first update; preserve exp_avg_sq and step',
        rng='same RNG seed before corresponding updates; trajectories can diverge',
        evaluation='Train20 greedy only; no Test, no API; no trained checkpoint saved'))
    report={}
    def evaluate(model,label):
        model.eval();results=[]
        for name in [r['name'] for r in rows]:
            inst,public=data[name]
            env,_,_=train.collect(model,inst,public,cfg['environment'],1,amp=True,greedy=True)
            results.append(dict(name=name,**{k:float(v[0]) for k,v in env.metrics().items()}))
        value=dict(results=results,mean={k:statistics.mean(r[k] for r in results) for k in ('reward','vehicles','travel_time','total_late','unserved')})
        write(OUT/(label+'.json'),value)
        print('EVAL',label,value['mean'],flush=True)
        return value
    for arm in ('existing_adam','reset_first_moment'):
        model=GuidedPolicy(**cfg['model']).cuda()
        model.load_state_dict(saved['model_state_dict'])
        opt=torch.optim.Adam(model.parameters(),**cfg['optimizer']['optimizer'])
        opt.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
        scaler=torch.amp.GradScaler('cuda',enabled=True)
        scaler.load_state_dict(copy.deepcopy(saved['scaler']))
        if arm=='reset_first_moment':
            for state in opt.state.values():state['exp_avg'].zero_()
        if 'baseline' not in report:report['baseline']=evaluate(model,'baseline')
        logs=[]
        for i,name in enumerate(order,1):
            write(OUT/'status.json',dict(stage='training_diagnostic',arm=arm,update=i,total=20,name=name))
            torch.manual_seed(20261109+i)
            model.train();model.new_customer_epsilon=.01
            inst,public=data[name]
            summary,_=train.train_batch(model,opt,scaler,inst,public,cfg['environment'],batch,rollouts,amp=True,
                max_steps=cfg['trainer'].get('max_steps',600))
            logs.append(dict(update=i,name=name,**summary))
            write(OUT/(arm+'_updates.json'),logs)
            print('UPDATE',arm,i,name,summary['reward'],summary['optimizer_step_skipped'],flush=True)
            if i in (5,20):
                write(OUT/'status.json',dict(stage='evaluating_train20',arm=arm,update=i))
                report[arm+'_step'+str(i)]=evaluate(model,arm+'_step'+str(i))
                write(OUT/'report.json',report)
        del model,opt,scaler
        torch.cuda.empty_cache()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    write(OUT/'status.json',dict(stage='complete',original_checkpoint_unchanged=True))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'status.json',dict(stage='failed',error=repr(error)))
        raise
