"""Replay saved server actions exactly, without sampling or paid API calls."""
import hashlib
from pathlib import Path
import torch
from experiments.concurrent_fleet_bias_guided.support import read, load, ROOT
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.io import write

SOURCE = ROOT/'server_export/pair_diagnostics_received_20261031_164445/pair_pure_v100_200_l5_s42_46_20261029'


def main():
    torch.set_num_threads(1)
    output = {}
    for seed in (44,45):
        run = SOURCE/f'vehicle200/seed{seed}/runs/pure'
        protocol = read(run/'protocol.json')
        for name, expected in protocol['source_hashes'].items():
            if name.startswith('experiments/'):
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name
        params = protocol['configs']['5']['environment']
        rows = {r['name']:r for r in read(run/'split.json')['test']}
        for result in read(run/'late5/results.json'):
            if not result['unserved']:
                continue
            inst, _ = load(rows[result['name']])
            env = TensorFleetEnv(inst,1,params,'cpu')
            trace=[]
            for step, item in enumerate(read(run/'late5/test_history'/(result['name']+'.json'))):
                clock, vehicle, destination, new, live = item[0]
                assert abs(float(env.time[0])-clock)<.002
                frame=env.frame(); v=int(vehicle); d=int(destination)
                if new:
                    action=d-1
                else:
                    slots=(frame['vehicles'][0]==v).nonzero().flatten()
                    action=next(env.n+int(s)*(env.n+1)+d for s in slots
                                if frame['mask'][0,env.n+int(s)*(env.n+1)+d])
                if d==0:
                    remaining=(env.customer[0,1:]==0).nonzero().flatten().add(1).tolist()
                    active=[i for i in range(env.v) if int(env.status[0,i]) in (1,2,3)
                            and not (int(env.status[0,i])==1 and int(env.target[0,i])==0)]
                    eligible=[n for n in remaining if env.demand[n]<=env.load[0,v]+1e-5]
                    lost=[n for n in remaining if n in eligible and not any(
                          env.demand[n]<=env.load[0,i]+1e-5 for i in active if i!=v)]
                    trace.append(dict(step=step,time=clock,vehicle=v,remaining=remaining,
                        eligible=eligible,load=float(env.load[0,v]),
                        total_remaining_demand=sum(float(env.demand[n]) for n in remaining),
                        active_vehicles=active,lost_individual_serviceability=lost))
                env.step(torch.tensor([action]),validate=True)
            assert bool(env.done.all())
            metrics={k:float(v[0]) for k,v in env.metrics().items()}
            assert abs(metrics['reward']-result['reward'])<1e-3
            key=f'{seed}/{result["name"]}'
            output[key]=dict(metrics=metrics,remaining=(env.customer[0,1:]==0).nonzero().flatten().add(1).tolist(),depot_trace=trace)
            print(key,'METRICS',metrics['reward'],'LAST',trace[-3:],'LOST',[t for t in trace if t['lost_individual_serviceability']],flush=True)
    write(ROOT/'analysis/pair_server_failure_replay_20261031.json',output)


if __name__=='__main__':
    main()
