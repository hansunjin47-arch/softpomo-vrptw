"""Read-only sampled-policy diagnostics; no optimizer, backward, or API calls."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.concurrent_fleet_v3.run import load, SOURCE
from experiments.concurrent_fleet_v9.policy import FleetPolicy
from experiments.concurrent_fleet_v9.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.io import write

RUN = ROOT / 'result/concurrent_fleet_v9_train20_late10_vehicle700_20261019'
DEST = ROOT / 'analysis/v9_exploration_diagnostics/report.json'


class ProbePolicy(FleetPolicy):
    def _phase_scores(self, encoded, frame, env, dispatch):
        scores = super()._phase_scores(encoded, frame, env, dispatch)
        if dispatch:
            self.dispatch_scores = scores.detach()
        return scores


def diversity(items):
    counts = Counter(items)
    return dict(unique=len(counts), dominant_fraction=max(counts.values()) / len(items))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    cfg = json.loads((RUN / 'protocol.json').read_text())['configs']['10']
    split = json.loads((SOURCE / 'split.json').read_text())
    output = dict(protocol=dict(groups_per_scenario=4, rollouts_per_group=100,
                                mode='eval, stochastic sampling, CUDA autocast',
                                historical_training_batch_measurement=False))
    for epoch in (500, 1000):
        path = RUN / f'late10/checkpoint-{epoch}.pt'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model = ProbePolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(saved['model_state_dict'])
        del saved
        rows = []
        for index, row in enumerate(split['train']):
            torch.manual_seed(20261021 + index)
            inst, public = load(row)
            env = TensorFleetEnv(inst, 400, cfg['environment'])
            starts, encoded = model.encode_events(inst, public, batch_size=4)
            routes = [[[] for _ in range(env.v)] for _ in range(env.count)]
            firsts = [[] for _ in range(env.count)]
            stops = defaultdict(list)
            for step in range(600):
                if bool(env.done.all()):
                    break
                frame = env.frame()
                with torch.autocast('cuda'):
                    p = model.probabilities(frame, starts, encoded, env)
                action = p.multinomial(1).squeeze(1)
                dispatch = env.dispatch_open & ~env.done
                if bool(dispatch.any()):
                    score = model.dispatch_scores.float()
                    best_new = score[:, 1:env.n+1].masked_fill(~frame['mask'][:, :env.n], -torch.inf).max(1).values
                    stop_score = score[:, -1]
                    ps = p[:, -1]
                    binary_entropy = -(ps * ps.clamp_min(1e-30).log() + (1-ps)*(1-ps).clamp_min(1e-30).log())
                    values = torch.stack([ps, stop_score, best_new, stop_score-best_new,
                                          binary_entropy, frame['mask'][:, -1].float()], 1)
                    for k, vals in zip(env.k[dispatch].cpu().tolist(), values[dispatch].cpu().tolist()):
                        stops[k].append(vals)
                vehicles = env.next_vehicle()[0].cpu().tolist()
                ks = env.k.cpu().tolist()
                live = (~env.done).cpu().tolist()
                for r, a in enumerate(action.cpu().tolist()):
                    if not live[r]:
                        continue
                    if a < env.n:
                        routes[r][ks[r]].append(a+1)
                        firsts[r].append(a+1)
                    elif a <= 2*env.n:
                        routes[r][vehicles[r]].append(a-env.n)
                env.step(action)
            else:
                raise RuntimeError('Action bound exceeded')
            rewards = env.metrics()['reward'].float().cpu().reshape(4, 100)
            # Match training's CUDA reduction, including its numerical behavior.
            device_rewards = env.metrics()['reward'].float().reshape(4, 100)
            centered = device_rewards - device_rewards.mean(1, keepdim=True)
            advantages = centered / centered.std(1, keepdim=True).clamp(min=1e-6)
            canonical = [tuple(sorted(tuple(route) for route in fleet if route)) for fleet in routes]
            sets = [tuple(sorted(first)) for first in firsts]
            groups = []
            for g in range(4):
                a, b = g*100, (g+1)*100
                groups.append(dict(reward_std=float(rewards[g].std()),
                                   centered_std=float(centered[g].std()),
                                   advantage_mean=float(advantages[g].mean()),
                                   advantage_abs_max=float(advantages[g].abs().max()),
                                   identical_reward=bool((rewards[g] == rewards[g, 0]).all()),
                                   reward_range=float(rewards[g].max()-rewards[g].min()),
                                   dispatch_order=diversity([tuple(x) for x in firsts[a:b]]),
                                   first_customer_set=diversity(sets[a:b]),
                                   canonical_routes=diversity(canonical[a:b])))
            stop_summary = {}
            for k, vals in stops.items():
                t = torch.tensor(vals)
                stop_summary[k] = dict(n=len(vals), pstop=float(t[:, 0].mean()),
                    pstop_min=float(t[:, 0].min()), pstop_max=float(t[:, 0].max()),
                    stop_score=float(t[:, 1].mean()), best_new_score=float(t[:, 2].mean()),
                    margin=float(t[:, 3].mean()), binary_entropy=float(t[:, 4].mean()),
                    stop_allowed_fraction=float(t[:, 5].mean()),
                    stop_clip_fraction=float((t[:, 1].abs() >= .99*model.decoder.logit_clipping).float().mean()))
            rows.append(dict(name=row['name'], groups=groups, stops=stop_summary,
                             vehicles=dict(Counter(env.k.cpu().tolist()))))
            print(epoch, row['name'], 'identical groups', sum(g['identical_reward'] for g in groups), flush=True)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        output[str(epoch)] = dict(checkpoint_sha256=digest, rows=rows)
        write(DEST, output)
        del model, env, encoded
        torch.cuda.empty_cache()
    print('Saved', DEST, flush=True)


if __name__ == '__main__':
    main()
