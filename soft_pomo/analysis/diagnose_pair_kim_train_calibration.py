"""Training-only fixed-checkpoint sensitivity grid; no test tuning or API."""
import statistics as st
import torch
from analysis.diagnose_pair_kim_example import RUN, read, save
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v9_guided.prompts import parse_response


class CalibratedPolicy(GuidedPolicy):
    temperature=1.
    def probabilities(self, frame, starts, encodings, env):
        beta=env.guidance.beta
        # (p * exp(T*beta*c))**(1/T) is softmax(log(p)/T + beta*c).
        env.guidance.beta=beta*self.temperature
        try:
            p=super().probabilities(frame,starts,encodings,env)
        finally:
            env.guidance.beta=beta
        calibrated=p.pow(1/self.temperature)
        calibrated=calibrated/calibrated.sum(1,keepdim=True)
        return torch.where(env.dispatch_open[:,None],p,calibrated)


def main():
    torch.set_num_threads(2)
    cfg=read(RUN/'training/protocol.json')['configs']['10']
    model=CalibratedPolicy(**cfg['model']).cpu().eval()
    model.load_state_dict(torch.load(RUN/'training/late10/checkpoint-1000.pt',map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    groups=read(RUN/'artifacts/groups.json')
    results=[]
    for row in read(RUN/'artifacts/split.json')['train']:
        inst,public=load(row)
        cache=read(RUN/'artifacts/train_cache'/(row['name']+'.json'))
        with torch.no_grad():
            starts,encoded=model.encode_events(inst,public)
            for beta,temperature in [(0.,1.),(5.,1.),(10.,1.),(5.,2.),(5.,4.)]:
                model.temperature=temperature
                env=GuidedEnv(inst,1,cfg['environment'],'cpu',groups=groups,scores=parse_response(cache['initial'],groups),beta=beta)
                if cache.get('post_disclosure'):
                    t=cache['revealed_at_min']
                    env.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
                for _ in range(600):
                    if bool(env.done.all()): break
                    action=model.probabilities(env.frame(),starts,encoded,env).argmax(1)
                    env.step(action,validate=True)
                assert bool(env.done.all())
                results.append(dict(name=row['name'],beta=beta,temperature=temperature,
                    metrics={k:float(v[0]) for k,v in env.metrics().items()}))
        print('TRAIN',row['name'],flush=True)
    save('train_calibration.json',results)
    for beta,temp in [(0.,1.),(5.,1.),(10.,1.),(5.,2.),(5.,4.)]:
        rows=[r['metrics'] for r in results if r['beta']==beta and r['temperature']==temp]
        print(beta,temp,{k:st.mean(r[k] for r in rows) for k in ['reward','vehicles','travel_time','total_late']})


if __name__=='__main__': main()
