"""Fixed-checkpoint decoding/precision controls without training or API calls."""
import collections
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
    settings=[('greedy',True,False,0.,1,1),('greedy',False,False,0.,1,1),
              ('sample',True,False,0.,100,1),('sample',False,False,0.,100,1),
              ('sample',True,True,.01,100,1),('greedy',True,True,.01,1,1),
              ('greedy',True,False,0.,100,1),('greedy',True,False,0.,100,100),
              ('greedy_beta0',True,False,0.,1,1)]
    for row in [split[0],split[10]]:
        inst,public=load(row)
        cache=read(BUNDLE/'artifacts/train_cache'/(row['name']+'.json'))
        for decode,amp,training,epsilon,count,encbatch in settings:
            torch.manual_seed(20261109)
            model=GuidedPolicy(**cfg['model']).cuda()
            model.load_state_dict(saved['model_state_dict']);model.train(training)
            model.new_customer_epsilon=epsilon
            env=GuidedEnv(inst,count,cfg['environment'],'cuda',groups=groups,scores=parse_response(cache['initial'],groups))
            if decode=='greedy_beta0':env.guidance.beta=0.
            if cache['post_disclosure'] is not None:
                t=cache['revealed_at_min'];env.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
            starts,encoded=model.encode_events(inst,public,batch_size=encbatch)
            for step in range(600):
                if bool(env.done.all()):break
                with torch.autocast('cuda',enabled=amp):probs=model.probabilities(env.frame(),starts,encoded,env)
                action=probs.argmax(1) if decode.startswith('greedy') else probs.multinomial(1).squeeze(1)
                env.step(action,validate=True)
            else:raise RuntimeError('Action bound')
            metrics=env.metrics()
            result=dict(name=row['name'],decode=decode,amp=amp,training=training,epsilon=epsilon,count=count,encoder_batch=encbatch,
                metrics={k:float(v.float().mean()) for k,v in metrics.items()},
                vehicle_counts=dict(collections.Counter(metrics['vehicles'].cpu().tolist())))
            results.append(result)
            write(ROOT/'analysis/r101_kim44_matrix_20261109.json',results)
            print(row['name'],decode,amp,training,count,encbatch,{k:round(result['metrics'][k],4) for k in ['reward','vehicles','total_late']},flush=True)
            del model,env


if __name__=='__main__':main()
