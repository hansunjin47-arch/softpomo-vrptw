"""Inspect first NEW choices and their final routes in the saved diversity probe."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_v3.run import SOURCE,load,write


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    run=ROOT/'result/v9_stopbias_entropy1_700_late10_20261021'
    cfg=json.loads((run/'protocol.json').read_text())['configs']['10']
    model=FleetPolicy(**cfg['model']).cpu().eval()
    ckpt=torch.load(run/'late10/checkpoint-500.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    original=json.loads((ROOT/'analysis/v9_route_diversity/report.json').read_text())
    split={r['name']:r for r in json.loads((SOURCE/'split.json').read_text())['train']}
    results=[]
    for row in original['results']:
        inst,public=load(split[row['name']])
        env=TensorFleetEnv(inst,1,cfg['environment'],device='cpu')
        starts,encoded=model.encode_events(inst,public)
        p=model.probabilities(env.frame(),starts,encoded,env)[0,:env.n]
        p=p/p.sum()
        first=[fleet[0][0] for fleet in row['routes']]
        signatures=[tuple(sorted(tuple(r) for r in fleet if r)) for fleet in row['routes']]
        hist=Counter(first)
        groups=defaultdict(list)
        for i,customer in enumerate(first):groups[customer].append(i)
        counts=Counter(signatures)
        common=counts.most_common(1)[0][0]
        top=p.topk(5)
        result=dict(name=row['name'],rollouts=len(first),first_customer_histogram=dict(hist),
            unique_first_customers=len(hist),initial_entropy=float(-(p*p.clamp_min(1e-30).log()).sum()),
            top5_probabilities=[dict(customer=int(i)+1,probability=float(v)) for v,i in zip(top.values,top.indices)],
            distinct_final_route_sets=len(counts),most_common_route_count=counts[common],
            first_customers_leading_to_most_common=dict(Counter(first[i] for i,s in enumerate(signatures) if s==common)),
            by_first_customer={c:dict(count=len(ids),unique_final_route_sets=len({signatures[i] for i in ids}),
                mean_reward=sum(row['metrics']['reward'][i] for i in ids)/len(ids)) for c,ids in groups.items()})
        results.append(result)
        print(json.dumps(result),flush=True)
    write(ROOT/'analysis/v9_first_customer_diagnosis/report.json',dict(checkpoint=500,device='CPU FP32',
        source='analysis/v9_route_diversity/report.json',note='First vehicle first customer is the first NEW action; original sampled trajectories reused; no training.',results=results))


if __name__=='__main__':main()
