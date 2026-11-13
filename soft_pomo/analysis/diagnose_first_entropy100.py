"""Initial FP32 gradient probe, no optimizer updates or paid calls."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_v3.run import SOURCE,load
from experiments.concurrent_fleet_exploration.objective import first_customer_entropy,dispatch_entropy
from experiments.concurrent_fleet_pair.io import write


def main():
    torch.set_num_threads(2)
    # Check that STOP mass cannot affect the conditional distribution.
    z=torch.tensor([[.3,.8,1.]],requires_grad=True)
    h,_=first_customer_entropy(z.softmax(1),torch.ones_like(z,dtype=torch.bool),torch.tensor([True]),torch.tensor([True]),2)
    grad=torch.autograd.grad(h.sum(),z)[0]
    assert abs(float(grad[0,-1]))<1e-6 and torch.isfinite(grad).all()
    p=ROOT/'server_export/be100_received_20261022_100545/runs/vehicle100/protocol.json'
    cfg=json.loads(p.read_text())['configs']['10']
    saved=torch.load(ROOT/'result/v9_stopbias_entropy1_700_late10_20261021/late10/checkpoint-0.pt',map_location='cpu',weights_only=False)
    rows=json.loads((SOURCE/'split.json').read_text())['train'];results=[]
    for index in [0,3,10]:
        model=FleetPolicy(**cfg['model']).train();model.load_state_dict(saved['model_state_dict'])
        torch.manual_seed(20261022+index)
        inst,public=load(rows[index]);env=TensorFleetEnv(inst,100,cfg['environment'],device='cpu')
        starts,encoded=model.encode_events(inst,public)
        logs=[];hs=[torch.zeros(100),torch.zeros(100)];cs=[torch.zeros(100),torch.zeros(100)]
        for step in range(600):
            if bool(env.done.all()):break
            frame=env.frame();live=~env.done;p=model.probabilities(frame,starts,encoded,env)
            for i,fn in enumerate([dispatch_entropy,first_customer_entropy]):
                h,v=fn(p,frame['mask'],env.dispatch_open,live,env.n);hs[i]=hs[i]+h;cs[i]=cs[i]+v.float()
            with torch.no_grad():a=p.multinomial(1).squeeze(1)
            logs.append(torch.where(live,p[env.ids,a],1.).log());env.step(a)
        assert bool(env.done.all())
        r=env.metrics()['reward'].float();adv=r-r[0];adv=adv-adv.mean()
        losses=[-(adv*torch.stack(logs).sum(0)).mean()]+[-(h/c.clamp_min(1)).mean() for h,c in zip(hs,cs)]
        params=list(model.parameters());norms=[]
        for loss in losses:
            g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            norms.append(sum(float(x.double().square().sum()) for x in g if x is not None)**.5)
        result=dict(name=rows[index]['name'],policy_norm=norms[0],stop_entropy_beta1_norm=norms[1],first_entropy_beta1_norm=norms[2],first_ratio=norms[2]/norms[0])
        results.append(result);print(result,flush=True)
    ratios=sorted(x['first_ratio'] for x in results)
    candidates=[.1,.3,1.,3.,10.]
    chosen=min(candidates,key=lambda b:abs(b*ratios[1]-.005))
    write(ROOT/'analysis/first_entropy100_gradient/report.json',dict(results=results,beta=chosen,target_ratio=.005,
        source='local pure700 checkpoint0, evaluated with vehicle100 reward; identical policy architecture',
        scope='CPU FP32 batch1 rollout100, pre-optimizer; no updates; not actual AMP batch128',
        schedule='constant first entropy; STOP entropy unchanged 1 to zero at500'))
    print('SELECTED',chosen,flush=True)


if __name__=='__main__':main()
