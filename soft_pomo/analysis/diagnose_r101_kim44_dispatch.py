"""Inspect dispatch mass and separate dispatch/routing sampling, no learning/API."""
import torch
from analysis.evaluate_r101_mask_kim_20261109 import BUNDLE, RECEIVED
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_v3 import run as data_module
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_pair.io import write


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    data_module.SOURCE=BUNDLE/'dataset'
    run=RECEIVED/'runs/seed44'
    cfg=read(run/'protocol.json')['configs']['5']
    saved=torch.load(run/'late5/checkpoint-1000.pt',map_location='cpu',weights_only=False)
    groups=read(BUNDLE/'artifacts/groups.json')
    split=read(run/'split.json')['train']
    results=[]
    for row in [split[0],split[10]]:
        inst,public=load(row)
        cache=read(BUNDLE/'artifacts/train_cache'/(row['name']+'.json'))
        for mode,count in [('greedy',1),('sample_all',500),('sample_dispatch',500),('sample_routing',500),('force_one_new',1)]:
            torch.manual_seed(20261109)
            model=GuidedPolicy(**cfg['model']).cuda().eval()
            model.load_state_dict(saved['model_state_dict'])
            env=GuidedEnv(inst,count,cfg['environment'],'cuda',groups=groups,scores=parse_response(cache['initial'],groups))
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min'];env.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
            starts,encoded=model.encode_events(inst,public)
            closure=[];forced=False
            for step in range(600):
                if bool(env.done.all()):break
                with torch.autocast('cuda',enabled=True):probs=model.probabilities(env.frame(),starts,encoded,env)
                greedy=probs.argmax(1)
                action=greedy.clone()
                if mode.startswith('sample'):
                    sampled=probs.multinomial(1).squeeze(1)
                    choose=(torch.ones_like(env.dispatch_open) if mode=='sample_all' else env.dispatch_open if mode=='sample_dispatch' else ~env.dispatch_open)
                    action=torch.where(choose,sampled,greedy)
                new,_,_=env.decode_action(action)
                if count==1 and bool(env.dispatch_open[0]) and not bool(new[0]):
                    closure.append(dict(step=step,vehicles=float(env.k[0]),new_mass=float(probs[0,:env.n].sum()),
                        existing_mass=float(probs[0,env.n:].sum()),max_new=float(probs[0,:env.n].max()),
                        selected_probability=float(probs[0,action[0]]),selected_slot=int(action[0]),
                        new_positive_count=int((probs[0,:env.n]>0).sum())))
                    if mode=='force_one_new' and not forced:
                        assert float(probs[0,:env.n].sum())>0
                        action=probs[:,:env.n].argmax(1);forced=True
                env.step(action,validate=True)
            else:raise RuntimeError('Action bound')
            metrics=env.metrics()
            keys=['reward','vehicles','travel_time','total_late','unserved']
            by_vehicle=[]
            for k in sorted(set(metrics['vehicles'].cpu().tolist())):
                mask=metrics['vehicles']==k
                by_vehicle.append(dict(vehicles=k,count=int(mask.sum()),**{key:float(metrics[key][mask].float().mean()) for key in keys if key!='vehicles'}))
            result=dict(name=row['name'],mode=mode,count=count,closure=closure,
                metrics={key:float(metrics[key].float().mean()) for key in keys},by_vehicle=by_vehicle)
            results.append(result)
            write(ROOT/'analysis/r101_kim44_dispatch_20261109.json',results)
            print(result,flush=True)
            del model,env


if __name__=='__main__':main()
