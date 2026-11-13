"""CPU/FP32 gradient probes on separate models; never update parameters."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.concurrent_fleet_v3.run import load, SOURCE
from experiments.concurrent_fleet_v9.policy import FleetPolicy
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_exploration.objective import dispatch_entropy
from experiments.concurrent_fleet_pair.io import write
from train_vrptw import _set_seed

RUN=ROOT/'result/v9_exploration700_late10_20261021'


class Probe(FleetPolicy):
    def _phase_scores(self,encoded,frame,env,dispatch):
        scores=super()._phase_scores(encoded,frame,env,dispatch)
        if dispatch: self.scores=scores.detach()
        return scores


def main():
    torch.set_num_threads(2)
    cfg=json.loads((RUN/'protocol.json').read_text())['configs']['10']
    split=json.loads((SOURCE/'split.json').read_text())['train']
    output=[]
    for label in ['initial_seed42','checkpoint500']:
        for index in [0,3,10]:
            _set_seed(cfg['trainer']['seed'])
            model=Probe(**cfg['model']).cpu().train()
            if label=='checkpoint500':
                saved=torch.load(RUN/'late10/checkpoint-500.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(saved['model_state_dict']);del saved
            torch.manual_seed(20261022+index)
            inst,public=load(split[index])
            env=TensorFleetEnv(inst,100,cfg['environment'],device='cpu')
            starts,enc=model.encode_events(inst,public)
            logs=[]; hs=torch.zeros(100);counts=torch.zeros(100);stats={}
            for step in range(600):
                if bool(env.done.all()):break
                frame=env.frame();live=~env.done
                p=model.probabilities(frame,starts,enc,env)
                h,valid=dispatch_entropy(p,frame['mask'],env.dispatch_open,live,env.n)
                hs=hs+h;counts=counts+valid.float()
                if bool(valid.any()):
                    score=model.scores[:,-1];clip=model.decoder.logit_clipping
                    for k in env.k[valid].unique().tolist():
                        ids=valid&(env.k==k)
                        st=stats.setdefault(k,[])
                        for ps,sc in zip(p.detach()[ids,-1].tolist(),score[ids].tolist()):
                            st.append((ps,sc,max(0.,1-(sc/clip)**2)))
                with torch.no_grad():a=p.multinomial(1).squeeze(1)
                logs.append(torch.where(live,p[env.ids,a],1.).log())
                env.step(a)
            else:raise RuntimeError('Step bound')
            r=env.metrics()['reward'].float()[None]
            adv=r-r[:,:1];adv=adv-adv.mean(1,keepdim=True)
            loss=-(adv.flatten()*torch.stack(logs).sum(0)).mean()
            entropy=(hs/counts.clamp(min=1)).mean()
            named=list(model.named_parameters());params=[p for _,p in named]
            gp=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            ge=torch.autograd.grad(-.01*entropy,params,allow_unused=True)
            norms={}
            for group in ['all','dispatch_head']:
                ids=[i for i,(name,_) in enumerate(named) if group=='all' or name.startswith(('dispatch_query.','stop_embedding'))]
                pn=sum(float(gp[i].double().square().sum()) for i in ids if gp[i] is not None)**.5
                en=sum(float(ge[i].double().square().sum()) for i in ids if ge[i] is not None)**.5
                dot=sum(float((gp[i].double()*ge[i].double()).sum()) for i in ids if gp[i] is not None and ge[i] is not None)
                norms[group]=dict(policy_norm=pn,entropy_norm_beta001=en,ratio=en/pn if pn else None,cosine=dot/(pn*en) if pn*en else None)
            summary={k:dict(n=len(v),pstop=sum(x[0] for x in v)/len(v),stop_score=sum(x[1] for x in v)/len(v),tanh_derivative=sum(x[2] for x in v)/len(v)) for k,v in stats.items()}
            row=dict(model=label,name=split[index]['name'],device='CPU FP32',rollouts=100,
                     actual_beta_at_checkpoint=0 if label=='checkpoint500' else .01,
                     entropy=float(entropy.detach()),policy_loss=float(loss.detach()),norms=norms,
                     vehicles=torch.bincount(env.k,minlength=26).tolist(),dispatch=summary)
            output.append(row);write(ROOT/'analysis/dispatch_gradient_diagnostics/report.json',output)
            print(json.dumps({k:v for k,v in row.items() if k not in ['dispatch','vehicles']}),flush=True)
            del model,env,enc,logs,loss,entropy,gp,ge,params,named,hs,p,h


if __name__=='__main__':main()
