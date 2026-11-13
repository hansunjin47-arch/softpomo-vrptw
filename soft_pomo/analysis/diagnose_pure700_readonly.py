"""Inference-only checkpoint diagnostics. No optimizer or backward calls."""
import hashlib
import json
from pathlib import Path
import sys
import statistics as st
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.concurrent_fleet_v3.run import load, SOURCE
from experiments.concurrent_fleet_v9.policy import FleetPolicy
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.io import write
RUN=ROOT/'result/concurrent_fleet_v9_train20_late10_vehicle700_20261019'
DEST=ROOT/'analysis/pure700_readonly_diagnostics'

@torch.no_grad()
def main():
    torch.set_num_threads(1)
    cfg=json.loads((RUN/'protocol.json').read_text())['configs']['10']
    split=json.loads((SOURCE/'split.json').read_text())
    output={}
    for epoch in (500,1000):
        path=RUN/f'late10/checkpoint-{epoch}.pt'
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        saved=torch.load(path,weights_only=False)
        model=FleetPolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(saved['model_state_dict'])
        rows=[]
        for index,row in enumerate(split['train']):
            torch.manual_seed(42000+index)
            inst,public=load(row)
            env=TensorFleetEnv(inst,100,cfg['environment'])
            starts,encoded=model.encode_events(inst,public)
            dispatch=[]; first=None
            for step in range(600):
                if bool(env.done.all()): break
                with torch.autocast('cuda'):
                    p=model.probabilities(env.frame(),starts,encoded,env)
                action=p.multinomial(1).squeeze(1)
                if first is None: first=action.cpu().tolist()
                active=env.dispatch_open & ~env.done
                if bool(active.any()):
                    entropy=-(p.float()*p.float().clamp_min(1e-30).log()).sum(1)
                    for k in env.k[active].unique().tolist():
                        ids=active & (env.k==k)
                        dispatch.append(dict(k=k,n=int(ids.sum()),pstop=float(p[ids,-1].mean()),
                            stop_allowed=float(env.mask()[ids,-1].float().mean()),
                            entropy=float(entropy[ids].mean())))
                env.step(action)
            else: raise RuntimeError('Exceeded action bound')
            reward=env.metrics()['reward'].float()
            rows.append(dict(name=row['name'],reward_mean=float(reward.mean()),reward_std=float(reward.std()),
                unique_rewards=int(reward.unique().numel()),first_customer_unique=len(set(first)),
                vehicle_histogram=torch.bincount(env.k,minlength=26).cpu().tolist(),dispatch=dispatch))
            print(epoch,row['name'],flush=True)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
        output[str(epoch)]=dict(checkpoint_sha256=before,rows=rows,
            mean_reward_std=st.mean(r['reward_std'] for r in rows),
            zero_std_instances=sum(r['reward_std']==0 for r in rows),
            identical_reward_instances=sum(r['unique_rewards']==1 for r in rows),
            mean_first_customer_unique=st.mean(r['first_customer_unique'] for r in rows))
    logs=[json.loads(l) for l in (RUN/'late10/training.jsonl').read_text().splitlines()]
    windows={}
    for lo,hi in ((501,600),(601,700),(701,800),(901,1000)):
        rows=[r for r in logs if lo<=r['epoch']<=hi]
        windows[f'{lo}-{hi}']=dict(batches=len(rows),
            zero_gradient_fraction=st.mean(r['gradient_norm']==0 for r in rows),
            clipped_fraction=st.mean(r['gradient_norm'] is not None and r['gradient_norm']>1 for r in rows),
            skipped=sum(r['optimizer_step_skipped'] for r in rows),
            reward=st.mean(r['reward'] for r in rows))
    output['training_log_windows']=windows
    write(DEST/'report.json',output)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='rows'} for k,v in output.items()}))

if __name__=='__main__': main()
