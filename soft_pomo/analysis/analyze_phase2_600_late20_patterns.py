"""Replay saved actions on CPU; no policy inference, training, or API calls."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.concurrent_fleet_v3.run import load, SOURCE
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.io import write

PURE = ROOT / 'server_export/late20_received_20261020_182356/runs/vehicle600'
PHASE = ROOT / 'result/v9_p2_600_late20_gpt_20261021'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def replay(inst, params, actions):
    env = TensorFleetEnv(inst, 1, params, device='cpu')
    routes = [[] for _ in range(env.v)]
    arcs = []
    for a in actions:
        if a == 2*env.n+1:
            env.step(torch.tensor([a]), validate=True)
            continue
        v = int(env.k[0]) if a < env.n else int(env.next_vehicle()[0][0])
        dest = a+1 if a < env.n else a-env.n
        origin = int(env.node[0, v])
        departure = float(env.time[0])
        env.step(torch.tensor([a]), validate=True)
        arrival = float(env.arrival[0, v])
        routes[v].append(dest)
        arcs.append(dict(vehicle=v, first=routes[v][0], origin=origin, dest=dest,
                         departure=departure, arrival=arrival,
                         late=max(0., arrival-float(env.close[dest]))))
    assert bool(env.done.all())
    return dict(metrics={k:float(v[0]) for k,v in env.metrics().items()},
                routes=[r for r in routes if r], late_arcs=[a for a in arcs if a['late']>1e-5])


def main():
    torch.set_num_threads(1)
    cfg = read(PURE/'protocol.json')['configs']['20']['environment']
    split = read(SOURCE/'split.json')['test']
    pure = {r['name']:r for r in read(PURE/'late20/results.json')}
    phase = {r['name']:r for r in read(PHASE/'training/late20/results.json')}
    output = []
    for row in split:
        name = row['name']
        inst, public = load(row)
        p = read(ROOT/'result/v9p2_600l20_test'/name/'result.json')
        b = read(PURE/'late20/test_history'/f'{name}.json')
        rp = replay(inst, cfg, [x['action_slot'] for x in p['history']])
        rb = replay(inst, cfg, [int(x[0][2]) for x in b])
        for actual, expected in [(rp['metrics'], phase[name]), (rb['metrics'], pure[name])]:
            for key in ['vehicles','travel_time','total_late','late_count','unserved']:
                assert abs(actual[key]-expected[key]) < .02, (name,key,actual[key],expected[key])
        output.append(dict(name=name, events=public, calls=p['calls'],
                           delta={k:phase[name][k]-pure[name][k] for k in ['reward','travel_time','total_late','late_count']},
                           pure=rb, phase2=rp))
    write(ROOT/'analysis/phase2_600_late20_patterns/report.json',output)
    for r in sorted(output,key=lambda r:r['delta']['reward'])[:5]:
        print(r['name'], 'events', r['events'], flush=True)
        for mode in ['pure','phase2']:
            print(mode, sorted(r[mode]['late_arcs'],key=lambda a:-a['late'])[:5],flush=True)
    print('Verified replay of all 160 saved trajectories')


if __name__ == '__main__':
    main()
