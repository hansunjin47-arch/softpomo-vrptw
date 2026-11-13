"""Frozen Kim1000 Train20 ablation. No training, API or Test selection."""
import io
import json
import tarfile
import statistics as st
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import torch
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_pair_kim.dispatch_guidance import DispatchGuidedPolicy
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'server_export/rc101_pair100_200_l5_mask_kim_s42_46_20261111'
OUT=ROOT/'result/rc_dispatch_guidance_pilot_20261113'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2),encoding='utf-8')

def checks():
    m=DispatchGuidedPolicy.__new__(DispatchGuidedPolicy);torch.nn.Module.__init__(m)
    m.new_customer_epsilon=0.;m.eval()
    env=S(count=1,n=2,time=torch.zeros(1),dispatch_open=torch.tensor([True]),done=torch.tensor([False]),
          guidance=S(vehicle_group=torch.tensor([[0]]),initial=torch.tensor([[0.,.2,.9]]),updates=[],beta=5.))
    frame={'vehicles':torch.tensor([[0]]),'mask':torch.tensor([[True,True,True,True,False]])}
    q=torch.tensor([[.2,.3,.1,.4,0.]],requires_grad=True)
    with patch.object(FleetPolicy,'probabilities',lambda *args:q):
        direct=m.probabilities(frame,None,None,env)
        assert direct[0,:2].sum()<q[0,:2].sum()
        assert direct[0,-1]==0 and torch.allclose(direct.sum(),torch.tensor(1.))
        m.dispatch_mode='preserve_family';p=m.probabilities(frame,None,None,env)
        assert torch.allclose(p[:,:2],q[:,:2]) and torch.allclose(p[:,2:].sum(),q[:,2:].sum())
        env.dispatch_open[:]=False
        a=m.probabilities(frame,None,None,env)
        b=GuidedPolicy.probabilities(m,frame,None,None,env)
        assert torch.allclose(a,b)
        (-a[0,3].log()).backward();assert torch.isfinite(q.grad).all()
    return dict(direct_lowers_new_mass=True,preserved_family_mass=True,mask_exact=True,
                post_dispatch_matches_legacy=True,finite_gradient=True)

@torch.no_grad()
def main():
    torch.set_num_threads(1);data.SOURCE=B/'dataset'
    rows=read(data.SOURCE/'split.json')['train'];groups=read(B/'artifacts/vehicle100/groups.json')
    membership={n:k for k,g in enumerate(groups) for n in g}
    receipt=max((ROOT/'server_export').glob('rc_kim_stop_receipt_*.json'),key=lambda p:p.stat().st_mtime)
    results=[];summary=[]
    with tarfile.open(read(receipt)['backup_after']) as archive:
        prefix=archive.getnames()[0].rstrip('/')
        for seed in range(42,47):
            base=prefix+f'/runs/vehicle100/seed{seed}'
            cfg=json.load(archive.extractfile(base+'/protocol.json'))['configs']['5']
            saved=torch.load(io.BytesIO(archive.extractfile(base+'/late5/checkpoint-1000.pt').read()),map_location='cpu',weights_only=False)
            for arm in ('legacy','direct','preserve_family'):
                model=(GuidedPolicy if arm=='legacy' else DispatchGuidedPolicy)(**cfg['model']).cuda().eval()
                model.load_state_dict(saved['model_state_dict']);model.dispatch_mode=arm
                records=[]
                for row in rows:
                    inst,public=load(row);cache=read(B/'artifacts/vehicle100/train_cache'/(row['name']+'.json'))
                    env=GuidedEnv(inst,1,cfg['environment'],groups=groups,scores=parse_response(cache['initial'],groups),beta=5.)
                    if cache.get('post_disclosure') is not None:
                        t=cache['revealed_at_min']
                        env.update_event_confidence(parse_response(cache['post_disclosure'],groups),revealed_at_min=t,effective_time_min=t)
                    starts,encoded=model.encode_events(inst,public);routes={}
                    for _ in range(600):
                        if bool(env.done.all()):break
                        frame=env.frame()
                        with torch.autocast('cuda',enabled=True):p=model.probabilities(frame,starts,encoded,env)
                        action=p.argmax(1);new,v,n=env.decode_action(action)
                        if int(n[0])>0:routes.setdefault(int(v[0]),[]).append(int(n[0]))
                        env.step(action,validate=True)
                    else:raise RuntimeError('route bound')
                    rec=dict(seed=seed,arm=arm,name=row['name'],**{k:float(v[0]) for k,v in env.metrics().items()})
                    flags=[membership[n]==membership[r[0]] for r in routes.values() for n in r[1:]]
                    rec['anchor_follow_rate']=st.mean(flags) if flags else 0.
                    records.append(rec)
                sm=dict(seed=seed,arm=arm,count=20,**{k:st.mean(r[k] for r in records) for k in ('reward','vehicles','travel_time','total_late','unserved','anchor_follow_rate')})
                results.extend(records);summary.append(sm)
                write(OUT/'report.json',dict(scope='Frozen epoch1000 Kim checkpoints; Train20 only; w100 late5 maskON beta5; inference ablation, NOT retrained models. Existing caches unchanged despite expanded dispatch semantics.',summary=summary,instances=results))
                write(OUT/'status.json',dict(stage='running',completed=len(summary),target=15))
                print(json.dumps(sm),flush=True)
                del model;torch.cuda.empty_cache()
    write(OUT/'status.json',dict(stage='complete',completed=15,target=15))

if __name__=='__main__':
    write(OUT/'checks.json',checks())
    main()
