"""Replay archived sequential solutions through v9, without a policy or training."""
import json
import sys
import hashlib
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.concurrent_fleet_v3.run import SOURCE, load, write
from experiments.concurrent_fleet_v9.env import TensorFleetEnv


def main():
    torch.set_num_threads(1)
    source = ROOT / 'result/bks_normalized_late10_train20_20261013/rc101'
    records = json.loads((source / 'eval_1000_test80_gpu/results.json').read_text())
    split = json.loads((SOURCE / 'split.json').read_text())['test']
    by_name = {r['name']: r for r in records}
    assert set(by_name) == {r['name'] for r in split}
    cfg = json.loads((ROOT / 'result/v9_stopbias_entropy1_700_late10_20261021/protocol.json').read_text())['configs']['10']
    results = []
    for item in split:
        name = item['name']
        assert hashlib.sha256((source/'data'/f'{name}.txt').read_bytes()).hexdigest() == item['sha256']
        inst, _ = load(item)
        saved = by_name[name]
        routes = [[int(j) for j in route if j != 0] for route in saved['routes']]
        assert all(routes)
        assert sorted(j for route in routes for j in route) == list(range(1, inst['n_customers']+1))
        env = TensorFleetEnv(inst, 1, cfg['environment'], device='cpu')
        assert len(routes) <= env.v
        positions = [0]*len(routes)
        history = []
        def step(action, vehicle, dest):
            if not bool(env.mask()[0, action]):
                raise ValueError(dict(name=name, time=float(env.time[0]), vehicle=vehicle, dest=dest, action=action))
            history.append(dict(time=float(env.time[0]),vehicle=vehicle,destination=dest,action=action))
            env.step(torch.tensor([action]), validate=True)
        try:
            for v, route in enumerate(routes):
                assert float(env.time[0]) == 0
                step(route[0]-1,v,route[0])
                positions[v]=1
            step(2*env.n+1,None,None)
            for _ in range(600):
                if bool(env.done[0]): break
                v, ready, _ = env.next_vehicle()
                v=int(v[0]);assert bool(ready[0])
                dest=routes[v][positions[v]] if positions[v]<len(routes[v]) else 0
                step(env.n+dest,v,dest)
                positions[v]+=1
            else: raise RuntimeError('Step bound')
            metrics={k:float(v[0]) for k,v in env.metrics().items()}
            assert metrics['unserved']==0
            delta={k:metrics[k]-saved[k] for k in ['vehicles','travel_time','total_late','late_count']}
            results.append(dict(name=name,valid=True,metrics=metrics,archived={k:saved[k] for k in delta},delta=delta,history=history))
        except (ValueError,AssertionError,RuntimeError) as exc:
            results.append(dict(name=name,valid=False,error=str(exc),history=history))
    valid=[r for r in results if r['valid']]
    summary=dict(count=len(results),valid=len(valid),failed=len(results)-len(valid))
    if valid:
        summary['mean']={k:sum(r['metrics'][k] for r in valid)/len(valid) for k in ['reward','vehicles','travel_time','total_late','late_count','unserved']}
        summary['max_abs_delta']={k:max(abs(r['delta'][k]) for r in valid) for k in valid[0]['delta']}
    write(ROOT/'analysis/sequential_v9_replay/report.json',dict(summary=summary,results=results,device='CPU FP32',policy_used=False,training=False))
    print(json.dumps(summary,indent=2))
    for r in results:
        if not r['valid']:print(r['name'],r['error'])


if __name__=='__main__': main()
