"""CPU frozen-policy audit with all 100 sampled trajectories per train instance."""
import json
from collections import Counter
from pathlib import Path
import statistics as st
import torch
from experiments.concurrent_fleet_pair.policy import FleetPolicy as Pair
from experiments.concurrent_fleet_pair.env import TensorFleetEnv as PairEnv
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy as Stop
from experiments.concurrent_fleet_v9.env import TensorFleetEnv as StopEnv
from experiments.concurrent_fleet_v3.run import SOURCE, load
from analysis.compare_pair_diversity import RUNS


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    report = {'precision': 'CPU FP32', 'count_per_instance': 100, 'runs': {}}
    for mode, root in RUNS.items():
        cfg = json.loads((root/'protocol.json').read_text())['configs']['10']
        path = root/('late10/checkpoint-last.pt' if mode == 'pair' else 'late10/checkpoint-500.pt')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model = (Pair if mode == 'pair' else Stop)(**cfg['model']).eval()
        model.load_state_dict(saved['model_state_dict'])
        entry = dict(epoch=saved.get('epoch'), state=saved.get('state'), cases=[])
        report['runs'][mode] = entry
        rows = {r['name']: r for r in json.loads((SOURCE/'split.json').read_text())['train']}
        for index, name in enumerate(cfg['trainer']['train_instances']):
            torch.manual_seed(20261025+index)
            inst, public = load(rows[name])
            env = (PairEnv if mode == 'pair' else StopEnv)(inst,100,cfg['environment'],device='cpu')
            starts, encoded = model.encode_events(inst,public)
            first = [[] for _ in range(100)]
            routes = [{} for _ in range(100)]
            for step in range(600):
                if bool(env.done.all()):
                    break
                frame = env.frame()
                p = model.probabilities(frame,starts,encoded,env)
                action = p.multinomial(1).squeeze(1)
                live = ~env.done
                if mode == 'pair':
                    new, vehicle, dest = env.decode_action(action, frame)
                else:
                    new = action < env.n
                    vehicle = torch.where(new,env.k,env.next_vehicle()[0])
                    dest = torch.where(new,action+1,action-env.n)
                    live &= action != 2*env.n+1
                for i in range(100):
                    if not live[i]:
                        continue
                    v, d = int(vehicle[i]), int(dest[i])
                    routes[i].setdefault(v,[]).append(d)
                    if new[i]:
                        first[i].append(d)
                env.step(action,validate=True)
            else:
                raise RuntimeError('Action bound exceeded')
            signatures = [tuple(sorted(tuple(v) for v in r.values())) for r in routes]
            sets = [tuple(sorted(f)) for f in first]
            metrics = {k:v.tolist() for k,v in env.metrics().items()}
            record = dict(name=name, first_order_unique=len(set(map(tuple,first))),
                          first_set_unique=len(set(sets)), route_unique=len(set(signatures)),
                          modal_route_fraction=Counter(signatures).most_common(1)[0][1]/100,
                          reward_std=st.pstdev(metrics['reward']),
                          vehicle_histogram=dict(Counter(metrics['vehicles'])),
                          first_customers=first, routes=routes, metrics=metrics)
            entry['cases'].append(record)
            print(mode,name,record['first_set_unique'],record['route_unique'],flush=True)
            Path('analysis/full_pair_diversity_20261025.json').write_text(json.dumps(report))
    print('COMPLETE',flush=True)


if __name__ == '__main__':
    main()
