"""Measure initial GPU gradients without optimizer updates; select a beta candidate."""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_v3.run import load, SOURCE
from experiments.concurrent_fleet_exploration.objective import dispatch_entropy
from experiments.concurrent_fleet_pair.io import write


def main():
    torch.set_num_threads(1)
    run = ROOT / 'result/v9_nostd_stopinit_local_20261021/stop_bias'
    cfg = json.loads((run / 'protocol.json').read_text())['configs']['10']
    saved = torch.load(run / 'late10/checkpoint-0.pt', map_location='cpu', weights_only=False)
    split = json.loads((SOURCE / 'split.json').read_text())['train']
    rows = []
    for index in [0, 3, 10]:
        torch.manual_seed(20261022 + index)
        model = FleetPolicy(**cfg['model']).cuda().train()
        model.load_state_dict(saved['model_state_dict'])
        inst, public = load(split[index])
        env = TensorFleetEnv(inst, 100, cfg['environment'])
        starts, encoded = model.encode_events(inst, public)
        logs = []
        hs = torch.zeros(100, device='cuda')
        counts = torch.zeros_like(hs)
        for step in range(600):
            if bool(env.done.all()): break
            frame = env.frame()
            live = ~env.done
            p = model.probabilities(frame, starts, encoded, env)
            h, valid = dispatch_entropy(p, frame['mask'], env.dispatch_open, live, env.n)
            hs = hs + h
            counts = counts + valid.float()
            with torch.no_grad(): action = p.multinomial(1).squeeze(1)
            logs.append(torch.where(live, p[env.ids, action], 1.).log())
            env.step(action)
        else: raise RuntimeError('Step bound')
        r = env.metrics()['reward'].float()
        adv = r - r[0]
        adv = adv - adv.mean()
        loss = -(adv * torch.stack(logs).sum(0)).mean()
        entropy = (hs / counts.clamp(min=1)).mean()
        params = list(model.parameters())
        gp = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        ge = torch.autograd.grad(-entropy, params, allow_unused=True)
        norm = lambda grads: sum(float(g.double().square().sum()) for g in grads if g is not None) ** .5
        pn, en = norm(gp), norm(ge)
        row = dict(name=split[index]['name'], policy_norm=pn, entropy_norm_beta1=en,
                   ratio_beta1=en/pn, entropy=float(entropy.detach()))
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, env, encoded, logs, hs, counts, p, h, loss, entropy, gp, ge, params
    # Target roughly 1% at initialization, not equality with the policy gradient.
    candidates = [0.1, 0.3, 1., 3., 10., 30., 100.]
    median = sorted(x['ratio_beta1'] for x in rows)[1]
    import math
    beta = min(candidates, key=lambda b: abs(math.log(b * median / .01)))
    report = dict(device='CUDA FP32', batch=1, rollouts=100, optimizer_updates=0,
                  reduction='per-rollout valid-step mean then batch mean', rows=rows,
                  candidates={str(b):[b*x['ratio_beta1'] for x in rows] for b in candidates},
                  selected_beta=beta, target_initial_norm_ratio=.01,
                  limitation='Not full batch128 AMP; norm ratio is not Adam update ratio')
    write(ROOT / 'analysis/stop_bias_entropy_calibration/report.json', report)
    print('SELECTED_BETA', beta, flush=True)


if __name__ == '__main__': main()
