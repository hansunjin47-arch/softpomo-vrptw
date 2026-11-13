"""CPU fixed-checkpoint RAIN bias ablation; no API and no active-run edits."""
import json
import statistics
import torch
from analysis.diagnose_pair_kim_final import KIM, OUT, read
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v9_guided.prompts import parse_test_response


def main():
    torch.set_num_threads(2)
    config=read(KIM/'training/protocol.json')['configs']['10']
    checkpoint=KIM/'training/late10/checkpoint-1000.pt'
    model=GuidedPolicy(**config['model']).cpu().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    groups=read(KIM/'artifacts/groups.json')
    rows=[r for r in read(KIM/'artifacts/split.json')['test'] if '_RAIN_' in r['name']]
    results=[]
    for row in rows:
        inst,public=load(row)
        payload=read(KIM/'late10/online_test'/row['name']/'calls.json')[0]['response']
        scores=parse_test_response(payload,groups,list(range(1,101)))
        result={'name':row['name']}
        with torch.no_grad():
            starts,encoded=model.encode_events(inst,public)
            for beta in (0.,5.):
                env=GuidedEnv(inst,1,config['environment'],'cpu',groups=groups,scores=scores,beta=beta)
                for _ in range(600):
                    if bool(env.done.all()): break
                    action=model.probabilities(env.frame(),starts,encoded,env).argmax(1)
                    env.step(action,validate=True)
                else: raise RuntimeError('Action bound exceeded')
                result[str(beta)]={k:float(v[0]) for k,v in env.metrics().items()}
        results.append(result)
        OUT.mkdir(exist_ok=True)
        (OUT/'rain_bias_ablation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(len(results),row['name'],result['0.0']['reward'],result['5.0']['reward'],flush=True)
    print('MEANS', {b:statistics.mean(r[b]['reward'] for r in results) for b in ('0.0','5.0')},flush=True)

if __name__=='__main__': main()
