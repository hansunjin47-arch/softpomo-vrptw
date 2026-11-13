"""Frozen Kim logit-scale/score-content controls on Train20 only."""
import hashlib
import statistics

import torch

from analysis import diagnose_kim_guidance_suite_20261110 as suite
from experiments.concurrent_fleet_pair.io import write

OUT = suite.ROOT / 'result/kim_fusion_20261110'
IMPLEMENTATION = 'decoder.logit_clipping = 10 * alpha; all masks and existing anchor guidance unchanged'
BASELINE_ROOT = suite.ROOT / 'result/kim_guidance_suite_20261110'
BASELINE_GUIDED_VARIANT = 'original'
VARIANTS = [('alpha025', .25, 'original', 5.),
            ('alpha010', .1, 'original', 5.),
            ('alpha010_uniform', .1, 'uniform', 5.),
            ('alpha010_off_control', .1, 'off', 0.)]


def configure_scale(model, alpha):
    model.decoder.logit_clipping = 10. * alpha


def diagnose_scaling(model, inst, public, cfg, groups, cache):
    comparisons = {}
    for amp in (True, False):
        for alpha in (1., .1):
            configure_scale(model, alpha)
            env = suite.GuidedEnv(inst, 1, cfg['environment'], 'cuda', groups=groups,
                scores=suite.parse_response(cache['initial'], groups), beta=0.)
            starts, enc = model.encode_events(inst, public)
            trace = []
            for step in range(600):
                if bool(env.done.all()):
                    break
                frame = env.frame()
                with torch.autocast('cuda', enabled=amp):
                    p = model.probabilities(frame, starts, enc, env)
                action = p.argmax(1)
                top = p.topk(2, dim=1)
                trace.append(dict(step=step, action=int(action[0]), dispatch_open=bool(env.dispatch_open[0]),
                    top_indices=top.indices[0].tolist(), top_probabilities=top.values[0].tolist()))
                env.step(action, validate=True)
            comparisons[f'amp{amp}_alpha{alpha}'] = dict(trace=trace, metrics={k:float(v[0]) for k,v in env.metrics().items()})
    first = {}
    for amp in (True, False):
        a = comparisons[f'amp{amp}_alpha1.0']['trace']
        b = comparisons[f'amp{amp}_alpha0.1']['trace']
        first[str(amp)] = next((dict(unscaled=x, scaled=y) for x,y in zip(a,b) if x['action']!=y['action']), None)
    configure_scale(model, .1)
    return dict(comparisons=comparisons, first_divergence=first)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    suite.validate()
    OUT.mkdir(exist_ok=True)
    suite.data_module.SOURCE = suite.BUNDLE / 'dataset'
    rows = suite.read(suite.RECEIVED / 'runs/seed44/split.json')['train']
    groups = suite.read(suite.BUNDLE / 'artifacts/groups.json')
    data = {r['name']: suite.load(r) for r in rows}
    caches = {r['name']: suite.read(suite.BUNDLE / 'artifacts/train_cache' / (r['name'] + '.json')) for r in rows}
    checkpoints = [(f'seed{s}', s, suite.RECEIVED / f'runs/seed{s}/late5/checkpoint-1000.pt') for s in range(42, 47)]
    checkpoints.append(('seed44_epoch1010', 44, suite.ROOT / 'result/r101_kim44_continuation10_20261109/checkpoint-1010.pt'))
    hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, _, path in checkpoints}
    write(OUT / 'protocol.json', dict(setting='R101 vehicle100 lambda5 maskON',
        checkpoints=hashes, variants=[dict(name=n, alpha=a, confidence=m, beta=b) for n,a,m,b in VARIANTS],
        role='Train diagnostic only; no API, training, or Test', decode='greedy AMP',
        implementation=IMPLEMENTATION,
        off_control='first RAIN and first ACC per checkpoint', original_parameters_unchanged=True))
    results = []
    target = len(checkpoints) * sum(1 if n=='alpha100_reconstruction' else len(rows) if b else 2 for n,a,m,b in VARIANTS)
    for label, seed, path in checkpoints:
        cfg = suite.read(suite.RECEIVED / f'runs/seed{seed}/protocol.json')['configs']['5']
        model = suite.GuidedPolicy(**cfg['model']).cuda().eval()
        assert model.decoder.logit_clipping == 10.
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state_dict'])
        for variant, alpha, mode, beta in VARIANTS:
            configure_scale(model, alpha)
            selected = rows if beta else [next(r for r in rows if '_RAIN_' in r['name']), next(r for r in rows if '_ACC_' in r['name'])]
            if variant == 'alpha100_reconstruction':
                selected = rows[:1]
            for row in selected:
                name = row['name']
                dest = OUT / label / variant / (name + '.json')
                if dest.exists():
                    results.append(suite.read(dest))
                    continue
                write(OUT/'status.json', dict(stage='running', checkpoint=label, variant=variant, name=name, completed=len(results), target=target))
                inst, public = data[name]
                cache = caches[name]
                scores = suite.transform(suite.parse_response(cache['initial'], groups), mode)
                env = suite.GuidedEnv(inst, 1, cfg['environment'], 'cuda', groups=groups, scores=scores, beta=beta)
                if cache['post_disclosure'] is not None:
                    t = cache['revealed_at_min']
                    env.update_event_confidence(suite.transform(suite.parse_response(cache['post_disclosure'], groups), mode), revealed_at_min=t, effective_time_min=t)
                starts, enc = model.encode_events(inst, public)
                trace = []
                for step in range(600):
                    if bool(env.done.all()):
                        break
                    frame = env.frame()
                    with torch.autocast('cuda', enabled=True):
                        p = model.probabilities(frame, starts, enc, env)
                        env.guidance.beta = 0.
                        q = model.probabilities(frame, starts, enc, env)
                        env.guidance.beta = beta
                    action = p.argmax(1)
                    opened = bool(env.dispatch_open[0])
                    if opened:
                        assert torch.allclose(p, q), 'Guidance unexpectedly changed dispatch'
                    trace.append(dict(step=step, dispatch_open=opened, action=int(action[0]), unguided_action=int(q.argmax(1)[0]), changed=bool(action[0] != q.argmax(1)[0])))
                    if hasattr(model, 'last_diagnostics'):
                        trace[-1].update(model.last_diagnostics)
                    env.step(action, validate=True)
                else:
                    raise RuntimeError('Action bound exceeded')
                baseline_mode = BASELINE_GUIDED_VARIANT if beta else 'off'
                baseline = suite.read(BASELINE_ROOT/label/baseline_mode/(name+'.json'))
                changed = [t['action'] for t in trace] != [t['action'] for t in baseline['trace']]
                if variant == 'alpha100_reconstruction':
                    assert not changed, 'Float64 reconstruction changed original guided trajectory'
                if not beta:
                    if changed:
                        diagnostic = diagnose_scaling(model, inst, public, cfg, groups, cache)
                        write(OUT/label/'control_deviation'/(name+'.json'), diagnostic)
                        assert not changed, 'Off-guidance positive scaling changed greedy trajectory; AMP/FP32 diagnostic saved'
                value = dict(checkpoint=label, seed=seed, variant=variant, alpha=alpha, beta=beta,
                    confidence_transform=mode, name=name, trace=trace,
                    metrics={k:float(v[0]) for k,v in env.metrics().items()},
                    changed_from_baseline=changed,
                    reward_delta_from_baseline=float(env.metrics()['reward'][0])-baseline['metrics']['reward'])
                write(dest, value)
                results.append(value)
            print(label, variant, statistics.mean(r['metrics']['reward'] for r in results if r['checkpoint']==label and r['variant']==variant), flush=True)
        del model
        torch.cuda.empty_cache()
    summary = []
    for label, _, path in checkpoints:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[label]
        for variant, alpha, mode, beta in VARIANTS:
            rs = [r for r in results if r['checkpoint']==label and r['variant']==variant]
            steps = [t for r in rs for t in r['trace'] if not t['dispatch_open']]
            summary.append(dict(checkpoint=label, variant=variant, alpha=alpha, beta=beta, count=len(rs),
                mean={k:statistics.mean(r['metrics'][k] for r in rs) for k in ('reward','vehicles','travel_time','total_late','unserved')},
                changed_cases=sum(r['changed_from_baseline'] for r in rs),
                reward_delta=statistics.mean(r['reward_delta_from_baseline'] for r in rs),
                changed_decisions=sum(t['changed'] for t in steps), decisions=len(steps)))
    write(OUT/'summary.json', summary)
    write(OUT/'status.json', dict(stage='complete', completed=len(results), target=target, checkpoints_unchanged=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(OUT/'status.json', dict(stage='failed', error=repr(error)))
        raise
