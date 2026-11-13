"""Frozen final-checkpoint dispatch audit; Train20 only, no API/training."""
import io
import json
import statistics
import tarfile
from pathlib import Path
import torch
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy, GuidedEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'server_export/rc101_pair100_200_l5_mask_kim_s42_46_20261111'
OUT = ROOT/'result/rc_kim_dispatch_comparison_20261113'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def ent(p):
    p=p[p>0].double()
    return float(-(p*p.log()).sum())

@torch.no_grad()
def main():
    torch.set_num_threads(1)
    data.SOURCE=BUNDLE/'dataset'
    rows=read(data.SOURCE/'split.json')['train']
    groups=read(BUNDLE/'artifacts/vehicle100/groups.json')
    receipt=max((ROOT/'server_export').glob('rc_kim_stop_receipt_*.json'),key=lambda p:p.stat().st_mtime)
    results=[]
    with tarfile.open(read(receipt)['backup_after']) as archive:
        prefix=archive.getnames()[0].rstrip('/')
        for method in ('Pure','Kim'):
            for seed in range(42,47):
                if method=='Pure':
                    folder=next((ROOT/'result').glob(f'rc_pair_returnmask_v100_l5_s{seed}_*'))
                    cfg=read(folder/'protocol.json')['configs']['5']
                    assert read(folder/'split.json')['train']==rows
                    ckpt=folder/'late5/checkpoint-1000.pt'
                    saved=torch.load(ckpt,map_location='cpu',weights_only=False)
                    source=str(ckpt)
                else:
                    base=prefix+f'/runs/vehicle100/seed{seed}'
                    cfg=json.load(archive.extractfile(base+'/protocol.json'))['configs']['5']
                    saved=torch.load(io.BytesIO(archive.extractfile(base+'/late5/checkpoint-1000.pt').read()),map_location='cpu',weights_only=False)
                    source=base+'/late5/checkpoint-1000.pt'
                assert saved['epoch']==1000
                model=(FleetPolicy if method=='Pure' else GuidedPolicy)(**cfg['model']).cuda().eval()
                model.load_state_dict(saved['model_state_dict'])
                del saved
                traces=[]
                off_max_difference=0.
                off_argmax_disagreements=0
                for row in rows:
                    inst,public=load(row)
                    if method=='Kim':
                        cache=read(BUNDLE/'artifacts/vehicle100/train_cache'/(row['name']+'.json'))
                        env=GuidedEnv(inst,1,cfg['environment'],groups=groups,scores=parse_response(cache['initial'],groups),beta=5.)
                    else:
                        env=TensorFleetEnv(inst,1,cfg['environment'])
                    starts,encoded=model.encode_events(inst,public)
                    steps=[]
                    for _ in range(env.v+2):
                        if not bool(env.dispatch_open[0]): break
                        frame=env.frame()
                        with torch.autocast('cuda',enabled=True):
                            p=model.probabilities(frame,starts,encoded,env)
                            if method=='Kim':
                                off=FleetPolicy.probabilities(model,frame,starts,encoded,env)
                                off_max_difference=max(off_max_difference,float((p-off).abs().max()))
                                off_argmax_disagreements+=int((p.argmax(1)!=off.argmax(1)).sum())
                        action=p.argmax(1)
                        n,o=p[0,:env.n],p[0,env.n:]
                        nm,om=float(n.sum()),float(o.sum())
                        steps.append(dict(k=int(env.k[0]),new_mass=nm,existing_mass=om,
                            max_new=float(n.max()),max_existing=float(o.max()),
                            family_entropy=ent(torch.tensor([nm,om])),entropy=ent(p[0]),
                            select_new=bool(action[0]<env.n),action=int(action[0])))
                        env.step(action,validate=True)
                    else: raise RuntimeError('dispatch bound')
                    traces.append(dict(name=row['name'],vehicles=int(env.k[0]),steps=steps))
                close=[t['steps'][-1] for t in traces]
                pre=[t['steps'][-2] for t in traces]
                summary=dict(method=method,seed=seed,epoch=1000,source=source,count=len(traces),
                    vehicles=statistics.mean(t['vehicles'] for t in traces),
                    off_max_difference=off_max_difference,off_argmax_disagreements=off_argmax_disagreements,
                    closing={k:statistics.mean(s[k] for s in close) for k in ('new_mass','existing_mass','max_new','max_existing','family_entropy','entropy')},
                    previous={k:statistics.mean(s[k] for s in pre) for k in ('new_mass','family_entropy')},
                    closes_with_new_mass_majority=sum(s['new_mass']>.5 for s in close))
                write(OUT/f'{method}_s{seed}.json',dict(summary=summary,traces=traces))
                results.append(summary)
                print(json.dumps(summary),flush=True)
                write(OUT/'status.json',dict(stage='running',completed=len(results),target=10))
                del model
                torch.cuda.empty_cache()
    write(OUT/'summary.json',dict(scope='RC101 w100 late5 maskON final1000 Train20 greedy dispatch only; no Test, API or training. Each model own trajectory; not matched-state causal comparison.',results=results))
    write(OUT/'status.json',dict(stage='complete',completed=10,target=10))

if __name__=='__main__': main()
