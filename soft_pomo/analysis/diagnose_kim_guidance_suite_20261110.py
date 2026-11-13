"""Frozen-checkpoint Train20 ablations. No training, Test tuning, or API calls."""
import hashlib
import random
import statistics
import torch
from analysis.evaluate_r101_mask_kim_20261109 import BUNDLE, RECEIVED, validate
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_v3 import run as data_module
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_pair.io import write

OUT = ROOT/'result/kim_guidance_suite_20261110'
VARIANTS = [('off', 0.), ('beta1', 1.), ('beta2_5', 2.5), ('original', 5.),
            ('beta10', 10.), ('uniform', 5.), ('shuffle', 5.), ('no_post', 5.)]


def transform(scores, mode):
    result = {}
    rng = random.Random(20261110)
    for group, row in scores.items():
        keys = sorted(row, key=int)
        values = [row[k] for k in keys]
        if mode == 'uniform':
            values = [statistics.mean(values)]*len(values)
        elif mode == 'shuffle':
            rng.shuffle(values)
        result[group] = dict(zip(keys, values))
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
    caches = {r['name']: read(BUNDLE/'artifacts/train_cache'/(r['name']+'.json')) for r in rows}
    checkpoints = [(f'seed{s}', s, RECEIVED/f'runs/seed{s}/late5/checkpoint-1000.pt') for s in range(42,47)]
    checkpoints.append(('seed44_epoch1010', 44, ROOT/'result/r101_kim44_continuation10_20261109/checkpoint-1010.pt'))
    hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, _, path in checkpoints}
    write(OUT/'protocol.json', dict(checkpoints=hashes, variants=VARIANTS, instances=20,
        decode='greedy AMP', setting='R101 vehicle100 lambda5 maskON',
        role='TRAIN diagnostic only; inference ablations do not isolate training effects',
        controls='uniform preserves per-group mean; shuffle preserves per-group score multiset',
        coverage='five original Kim seeds plus recovered44; no retraining, API, or Test ablations'))
    audit = []
    for name, cache in caches.items():
        for phase in ('initial', 'post_disclosure'):
            if cache[phase] is None:
                continue
            scores = parse_response(cache[phase], groups)
            values = [float(v) for row in scores.values() for v in row.values()]
            audit.append(dict(name=name, phase=phase, minimum=min(values), maximum=max(values),
                mean=statistics.mean(values), std=statistics.pstdev(values),
                within_group_std=statistics.mean(statistics.pstdev(list(row.values())) for row in scores.values()),
                revealed_at_min=cache.get('revealed_at_min')))
    write(OUT/'cache_audit.json', audit)
    all_results = []
    for label, seed, path in checkpoints:
        cfg = read(RECEIVED/f'runs/seed{seed}/protocol.json')['configs']['5']
        model = GuidedPolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state_dict'])
        for mode, beta in VARIANTS:
            for row in rows:
                name = row['name']
                dest = OUT/label/mode/(name+'.json')
                if dest.exists():
                    all_results.append(read(dest)); continue
                write(OUT/'status.json', dict(stage='running', checkpoint=label, variant=mode,
                    name=name, completed=len(all_results), target=len(checkpoints)*len(VARIANTS)*len(rows)))
                inst, public = data[name]
                cache = caches[name]
                scores = transform(parse_response(cache['initial'], groups), mode)
                env = GuidedEnv(inst, 1, cfg['environment'], 'cuda', groups=groups, scores=scores, beta=beta)
                if cache['post_disclosure'] is not None and mode != 'no_post':
                    t = cache['revealed_at_min']
                    env.update_event_confidence(transform(parse_response(cache['post_disclosure'], groups), mode),
                        revealed_at_min=t, effective_time_min=t)
                starts, enc = model.encode_events(inst, public)
                trace = []
                for step in range(600):
                    if bool(env.done.all()): break
                    frame = env.frame()
                    with torch.autocast('cuda', enabled=True):
                        p = model.probabilities(frame, starts, enc, env)
                        if mode == 'original':
                            env.guidance.beta = 0.
                            q = model.probabilities(frame, starts, enc, env)
                            env.guidance.beta = beta
                    action = p.argmax(1)
                    entry = dict(step=step, time=float(env.time[0]), dispatch_open=bool(env.dispatch_open[0]), action=int(action[0]))
                    if mode == 'original':
                        a0 = q.argmax(1)
                        new, vehicle, customer = env.decode_action(action)
                        n0, v0, c0 = env.decode_action(a0)
                        entry.update(unguided_action=int(a0[0]), changed=bool(action[0] != a0[0]),
                            selected_customer=int(customer[0]), unguided_customer=int(c0[0]),
                            selected_vehicle=int(vehicle[0]), unguided_vehicle=int(v0[0]),
                            kl=float((p.float()*(p.float().clamp_min(1e-30).log()-q.float().clamp_min(1e-30).log())).sum()),
                            total_variation=float((p-q).abs().sum()/2))
                        if entry['dispatch_open']:
                            assert torch.allclose(p, q), 'Unexpected guidance during dispatch'
                    trace.append(entry)
                    env.step(action, validate=True)
                else: raise RuntimeError('Action bound exceeded')
                value = dict(checkpoint=label, seed=seed, variant=mode, beta=beta, name=name,
                    metrics={k: float(v[0]) for k, v in env.metrics().items()}, trace=trace)
                write(dest, value)
                all_results.append(value)
            subset = [r for r in all_results if r['checkpoint'] == label and r['variant'] == mode]
            print(label, mode, statistics.mean(r['metrics']['reward'] for r in subset), flush=True)
        del model
    summaries = []
    for label, _, path in checkpoints:
        for mode, beta in VARIANTS:
            subset = [r for r in all_results if r['checkpoint'] == label and r['variant'] == mode]
            traces = [t for r in subset for t in r['trace'] if not t['dispatch_open']]
            summaries.append(dict(checkpoint=label, variant=mode, count=len(subset),
                mean={k: statistics.mean(r['metrics'][k] for r in subset)
                      for k in ('reward', 'vehicles', 'travel_time', 'total_late', 'unserved')},
                action_change_rate=statistics.mean(t['changed'] for t in traces) if mode == 'original' else None))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[label]
    write(OUT/'summary.json', summaries)
    write(OUT/'status.json', dict(stage='complete', completed=len(all_results), checkpoints_unchanged=True))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        write(OUT/'status.json', dict(stage='failed', error=repr(error)))
        raise
