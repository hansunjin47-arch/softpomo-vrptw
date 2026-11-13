"""Matched test diagnostics; no LLM calls or optimization."""
import json
import sys
from pathlib import Path
import statistics as st
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.concurrent_fleet_v9_phase2.support import OUT, PURE, load, read
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v9_guided.policy import GuidedFleetPolicy
from experiments.concurrent_fleet_v9_guided.env import GuidedFleetEnv
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_v9_guided.prompts import parse_response

DEST=ROOT/'analysis/v9_phase2_diagnosis_20261020'
TEST=ROOT/'result/v9p2_test_remaining_v1'

@torch.no_grad()
def evaluate(model, inst, public, ep, groups, scores, beta):
    env = GuidedFleetEnv(inst,1,ep,groups=groups,scores=scores,beta=beta)
    starts, enc=model.encode_events(inst,public)
    for _ in range(600):
        if bool(env.done.all()): break
        with torch.autocast('cuda'):
            p=model.probabilities(env.frame(),starts,enc,env)
        env.step(p.argmax(1),validate=True)
    else: raise RuntimeError('Action limit')
    return {k:float(v[0]) for k,v in env.metrics().items()}

def main():
    torch.set_num_threads(1)
    cfg=read(OUT/'training/protocol.json')['configs']['10']
    model=GuidedFleetPolicy(**cfg['model']).cuda().eval()
    saved=torch.load(OUT/'training/late10/checkpoint-1000.pt',weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    groups=read(OUT/'groups.json')
    results={mode:[] for mode in ('phase2_no_bias','phase2_initial_only')}
    for row in read(OUT/'split.json')['test']:
        inst,public=load(row)
        calls=read(TEST/row['name']/'calls.json')
        scores=parse_response(calls[0]['response'],groups)
        for mode,beta in [('phase2_no_bias',0.),('phase2_initial_only',5.)]:
            metrics=evaluate(model,inst,public,cfg['environment'],groups,scores,beta)
            results[mode].append(dict(name=row['name'],**metrics))
        write(DEST/'ablation.json',results)
        print(row['name'],flush=True)
    results['pure1000']=read(PURE/'late10/results.json')
    results['phase2_full']=read(TEST/'results.json')
    summary={}
    for mode, rows in results.items():
        summary[mode]={}
        for kind in ('ALL','RAIN','ACC'):
            chosen=[r for r in rows if kind=='ALL' or f'_{kind}_' in r['name']]
            summary[mode][kind]={k:st.mean(r[k] for r in chosen) for k in
                ('reward','vehicles','travel_time','distance','total_late','late_count','unserved')}
    pure={r['name']:r for r in results['pure1000']}
    paired=[dict(name=r['name'],reward_delta=r['reward']-pure[r['name']]['reward'],
                 late_delta=r['total_late']-pure[r['name']]['total_late']) for r in results['phase2_full']]
    write(DEST/'summary.json',summary)
    write(DEST/'paired.json',sorted(paired,key=lambda r:r['reward_delta']))
    print(json.dumps(summary))

if __name__=='__main__': main()
