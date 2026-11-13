"""Frozen-checkpoint greedy train20 guidance ablations; no API calls or updates."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.concurrent_fleet_bias_guided.support import GuidedBiasPolicy, cached_env, load, read
from experiments.concurrent_fleet_bias_guided.kim300 import OUT as BASE, PURE
from experiments.concurrent_fleet_stop_init.train import collect
from experiments.concurrent_fleet_v3.run import SOURCE, write

OUT = ROOT / 'analysis/guidance_toggle300_20261023'
KEYS = ['reward', 'vehicles', 'travel_time', 'total_late', 'late_count', 'unserved']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    cfg = read(PURE / 'protocol.json')['configs']['10']
    split = read(SOURCE / 'split.json')
    rows = split['train']
    assert len(rows) == 20
    data = {r['name']: load(r) for r in rows}
    cases = [
        ('pure500_phase2', PURE / 'late10/checkpoint-500.pt', 'phase2'),
        ('pure500_kim', PURE / 'late10/checkpoint-500.pt', 'kim'),
        ('phase2_1000', BASE / 'phase2/training/late10/checkpoint-1000.pt', 'phase2'),
        ('kim500', BASE / 'kim/training/late10/checkpoint-500.pt', 'kim'),
        ('kim1000', BASE / 'kim/training/late10/checkpoint-1000.pt', 'kim'),
    ]
    report = dict(protocol='Fixed TRAIN20, greedy one-shot, GPU AMP, beta ON=5 OFF=0; no updates/API calls', cases={})
    for label, checkpoint, method in cases:
        artifact = BASE / method / 'artifacts'
        assert read(artifact / 'config.json')['environment'] == cfg['environment']
        checkpoint_hash = digest(checkpoint)
        model = GuidedBiasPolicy(**cfg['model']).cuda().eval()
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'], strict=True)
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        env_class = cached_env(artifact)
        result = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                      cache_sha256={p.relative_to(artifact).as_posix(): digest(p)
                                    for p in [artifact/'groups.json', *sorted((artifact/'train_cache').glob('*.json'))]},
                      rows=[])
        for row in rows:
            pair = {'name': row['name'], 'kind': row['kind']}
            for mode, beta in [('off', 0.), ('on', 5.)]:
                class ToggleEnv(env_class):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.guidance.beta = beta
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                inst, public = data[row['name']]
                env, _, history = collect(model, inst, public, cfg['environment'], 1,
                                           amp=True, greedy=True, env_type=ToggleEnv)
                metrics = {k: float(v[0]) for k, v in env.metrics().items()}
                trace = torch.stack(history).cpu().tolist()
                pair[mode] = dict(metrics=metrics, history=trace,
                                  first_customers=[int(h[0][2])+1 for h in trace if h[0][3] and h[0][2]<100])
            assert pair['on']['first_customers'] == pair['off']['first_customers']
            result['rows'].append(pair)
            print(label, row['name'], 'OFF', pair['off']['metrics']['reward'], 'ON', pair['on']['metrics']['reward'], flush=True)
        result['means'] = {mode: {k: statistics.mean(r[mode]['metrics'][k] for r in result['rows']) for k in KEYS}
                           for mode in ['off', 'on']}
        delta = [r['on']['metrics']['reward']-r['off']['metrics']['reward'] for r in result['rows']]
        result['on_vs_off'] = dict(mean_reward_delta=statistics.mean(delta), wins=sum(d>1e-5 for d in delta),
                                  ties=sum(abs(d)<=1e-5 for d in delta), losses=sum(d< -1e-5 for d in delta))
        assert all(torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items())
        assert checkpoint_hash == digest(checkpoint)
        report['cases'][label] = result
        write(OUT / 'report.json', report)
        del model, state, before
    write(OUT / 'complete.json', dict(cases=len(cases), paired_instances=100, rollouts=200, paid_calls=0, optimizer_updates=0))


if __name__ == '__main__':
    main()
