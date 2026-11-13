"""Train-only frozen-policy coverage ablations; cached confidence is not retrained."""
import hashlib
import random
import statistics
import torch
from analysis.evaluate_r101_mask_kim_20261109 import BUNDLE, RECEIVED, validate
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_v3 import run as data_module
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_pair.io import write

OUT = ROOT/'result/kim_coverage_20261110'
BASE = ROOT/'result/kim_guidance_suite_20261110'
MODES = ['original', 'global', 'global_uniform', 'global_shuffle']


class CoveragePolicy(GuidedPolicy):
    mode = 'original'

    def probabilities(self, frame, starts, encodings, env):
        assert not self.training
        p = FleetPolicy.probabilities(self, frame, starts, encodings, env)
        g = env.guidance
        group = g.vehicle_group.gather(1, frame['vehicles'])

        def select(tensor):
            if self.mode == 'original':
                return tensor[group.clamp(min=0)]
            # Each customer has exactly one nonzero group entry. Reusing it for
            # every vehicle deliberately changes the original prompt semantics.
            values = tensor.sum(0)
            if self.mode == 'global_uniform':
                values = values.clone()
                values[1:] = .5
            elif self.mode == 'global_shuffle':
                order = list(range(1, env.n + 1))
                random.Random(20261110).shuffle(order)
                values = torch.cat([values[:1], values[order]])
            return values[None, None, :].expand(env.count, group.shape[1], -1)

        confidence = select(g.initial)
        for effective, target, scores in g.updates:
            confidence = torch.where((target & (env.time >= effective))[:, None, None],
                                     select(scores), confidence)
        active = (group >= 0) & ~env.dispatch_open[:, None] & ~env.done[:, None]
        bias = g.beta * confidence * active[:, :, None]
        assert bool((bias[:, :, 0] == 0).all())
        bias = torch.cat([bias.new_zeros(env.count, env.n), bias.flatten(1)], 1)
        weighted = p * torch.exp(bias)
        result = weighted / weighted.sum(1, keepdim=True)
        assert bool((result[p == 0] == 0).all())
        return result


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    validate()
    OUT.mkdir(exist_ok=True)
    data_module.SOURCE = BUNDLE/'dataset'
    rows = read(RECEIVED/'runs/seed44/split.json')['train']
    groups = read(BUNDLE/'artifacts/groups.json')
    data = {r['name']: load(r) for r in rows}
    checkpoints = [(f'seed{s}', s, RECEIVED/f'runs/seed{s}/late5/checkpoint-1000.pt') for s in range(42, 47)]
    checkpoints.append(('seed44_epoch1010', 44, ROOT/'result/r101_kim44_continuation10_20261109/checkpoint-1010.pt'))
    hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, _, path in checkpoints}
    write(OUT/'protocol.json', dict(checkpoints=hashes, variants=MODES, setting='R101 vehicle100 lambda5 maskON beta5',
        decode='Train20 greedy AMP', controls='uniform .5; globally shuffled fixed customer permutation',
        caveat='Global coverage reinterprets group-specific cached scores, a structural diagnostic with prompt mismatch.',
        invariants='No API, Test or training; NEW/depot/dispatch unbiased, event time gate and original mask retained.'))
    results = []
    for label, seed, path in checkpoints:
        cfg = read(RECEIVED/f'runs/seed{seed}/protocol.json')['configs']['5']
        model = CoveragePolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state_dict'])
        for mode in MODES:
            model.mode = mode
            for row in rows:
                name = row['name']
                dest = OUT/label/mode/(name+'.json')
                if dest.exists():
                    results.append(read(dest))
                    continue
                write(OUT/'status.json', dict(stage='running', checkpoint=label, variant=mode,
                    name=name, completed=len(results), target=480))
                cache = read(BUNDLE/'artifacts/train_cache'/(name+'.json'))
                inst, public = data[name]
                env = GuidedEnv(inst, 1, cfg['environment'], 'cuda', groups=groups,
                    scores=parse_response(cache['initial'], groups), beta=5.)
                if cache['post_disclosure'] is not None:
                    t = cache['revealed_at_min']
                    env.update_event_confidence(parse_response(cache['post_disclosure'], groups),
                        revealed_at_min=t, effective_time_min=t)
                starts, enc = model.encode_events(inst, public)
                trace = []
                for step in range(600):
                    if bool(env.done.all()):
                        break
                    frame = env.frame()
                    with torch.autocast('cuda', enabled=True):
                        p = model.probabilities(frame, starts, enc, env)
                        if mode == 'original':
                            reference = GuidedPolicy.probabilities(model, frame, starts, enc, env)
                            assert torch.equal(p, reference), 'Original probability reconstruction mismatch'
                    action = p.argmax(1)
                    trace.append(dict(step=step, time=float(env.time[0]), dispatch_open=bool(env.dispatch_open[0]), action=int(action[0])))
                    env.step(action, validate=True)
                else:
                    raise RuntimeError('Action bound exceeded')
                baseline = read(BASE/label/'original'/(name+'.json'))
                metrics = {k: float(v[0]) for k, v in env.metrics().items()}
                same = [t['action'] for t in trace] == [t['action'] for t in baseline['trace']]
                if mode == 'original':
                    assert same and metrics == baseline['metrics'], 'Saved baseline mismatch'
                value = dict(checkpoint=label, seed=seed, variant=mode, name=name, metrics=metrics, trace=trace,
                    changed_path=not same, delta={k: metrics[k]-baseline['metrics'][k] for k in metrics})
                write(dest, value)
                results.append(value)
            subset = [r for r in results if r['checkpoint'] == label and r['variant'] == mode]
            print(label, mode, statistics.mean(r['metrics']['reward'] for r in subset), flush=True)
        del model
    summary = []
    for label, _, path in checkpoints:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[label]
        for mode in MODES:
            subset = [r for r in results if r['checkpoint'] == label and r['variant'] == mode]
            keys = ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved')
            summary.append(dict(checkpoint=label, variant=mode, count=len(subset),
                mean={k: statistics.mean(r['metrics'][k] for r in subset) for k in keys},
                delta={k: statistics.mean(r['delta'][k] for r in subset) for k in keys},
                changed_paths=sum(r['changed_path'] for r in subset)))
    write(OUT/'summary.json', summary)
    write(OUT/'status.json', dict(stage='complete', completed=len(results), checkpoints_unchanged=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(OUT/'status.json', dict(stage='failed', error=repr(error)))
        raise
