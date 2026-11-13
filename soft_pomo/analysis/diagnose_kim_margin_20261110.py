"""Same-state guidance coverage and decision-boundary audit on frozen Train20."""
import math
import statistics
import torch
from analysis.evaluate_r101_mask_kim_20261109 import BUNDLE, RECEIVED, validate
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_v3 import run as data_module
from experiments.concurrent_fleet_bias_guided.support import ROOT, read, load
from experiments.concurrent_fleet_v9_guided.prompts import parse_response
from experiments.concurrent_fleet_pair.io import write

OUT = ROOT/'result/kim_margin_20261110'


def aggregate(rows):
    choices = [r for r in rows if r['valid_count'] > 1]
    eligible = [r for r in choices if r['positive_bias_count']]
    thresholds = [r['beta_to_flip'] for r in choices if r['beta_to_flip'] is not None]
    return dict(decisions=len(rows), genuine_choices=len(choices),
        choices_with_positive_bias=len(eligible), changed=sum(r['changed'] for r in choices),
        mean_valid_count=statistics.mean(r['valid_count'] for r in choices) if choices else None,
        mean_coverage=statistics.mean(r['coverage'] for r in choices) if choices else None,
        median_top2_logit_gap=statistics.median(r['top2_gap'] for r in choices) if choices else None,
        mean_top_probability=statistics.mean(r['top_probability'] for r in choices) if choices else None,
        median_bias_span=statistics.median(r['bias_span'] for r in choices) if choices else None,
        beta5_can_flip=sum(x < 5 for x in thresholds), beta10_can_flip=sum(x < 10 for x in thresholds),
        finite_flip_thresholds=len(thresholds),
        median_finite_beta_to_flip=statistics.median(thresholds) if thresholds else None,
        # If the RL winner also maximizes confidence, scaling nonnegative beta cannot defeat it.
        no_higher_confidence_competitor=len(choices)-len(thresholds))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    validate()
    OUT.mkdir(exist_ok=True)
    data_module.SOURCE = BUNDLE/'dataset'
    rows = read(RECEIVED/'runs/seed44/split.json')['train']
    groups = read(BUNDLE/'artifacts/groups.json')
    checkpoints = [(f'seed{s}', s, RECEIVED/f'runs/seed{s}/late5/checkpoint-1000.pt') for s in range(42,47)]
    checkpoints.append(('seed44_epoch1010', 44, ROOT/'result/r101_kim44_continuation10_20261109/checkpoint-1010.pt'))
    write(OUT/'protocol.json', dict(role='same-state inference diagnosis, no optimization or paid calls',
        setting='R101 vehicle100 lambda5 maskON Kim beta5',
        beta_to_flip='Minimum positive beta where a higher-confidence candidate overtakes unguided winner; ties excluded',
        logit_gap='Recovered from log probabilities before guidance; additive normalization cancels',
        checkpoints=[str(p) for _, _, p in checkpoints]))
    summaries = []
    for label, seed, checkpoint in checkpoints:
        cfg = read(RECEIVED/f'runs/seed{seed}/protocol.json')['configs']['5']
        model = GuidedPolicy(**cfg['model']).cuda().eval()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict'])
        decisions = []
        for index, row in enumerate(rows):
            path = OUT/label/(row['name']+'.json')
            if path.exists():
                decisions.extend(read(path)['decisions']); continue
            write(OUT/'status.json', dict(stage='running', checkpoint=label, completed_cases=index, total_per_checkpoint=20))
            inst, public = load(row)
            cache = read(BUNDLE/'artifacts/train_cache'/(row['name']+'.json'))
            env = GuidedEnv(inst, 1, cfg['environment'], 'cuda', groups=groups,
                scores=parse_response(cache['initial'], groups), beta=5.)
            if cache['post_disclosure'] is not None:
                t = cache['revealed_at_min']
                env.update_event_confidence(parse_response(cache['post_disclosure'], groups),
                    revealed_at_min=t, effective_time_min=t)
            starts, enc = model.encode_events(inst, public)
            trace = []
            for step in range(600):
                if bool(env.done.all()): break
                frame = env.frame()
                with torch.autocast('cuda'):
                    p = model.probabilities(frame, starts, enc, env)
                    q = FleetPolicy.probabilities(model, frame, starts, enc, env)
                group = env.guidance.vehicle_group.gather(1, frame['vehicles'])
                confidence = env.guidance.initial[group.clamp(min=0)]
                for effective, target, scores in env.guidance.updates:
                    eligible = target & (env.time >= effective)
                    confidence = torch.where(eligible[:, None, None], scores[group.clamp(min=0)], confidence)
                active = (group >= 0) & ~env.dispatch_open[:, None] & ~env.done[:, None]
                score = confidence*active[:, :, None]
                score = torch.cat([score.new_zeros(1, env.n), score.flatten(1)], 1)
                weighted = q*torch.exp(5*score)
                expected = weighted/weighted.sum(1, keepdim=True)
                assert torch.allclose(p, expected, atol=1e-6, rtol=1e-5), 'Bias reconstruction mismatch'
                valid = frame['mask'][0]
                values, ids = q[0, valid].sort(descending=True)
                indices = valid.nonzero().flatten()[ids]
                if not len(indices): raise RuntimeError('No valid decision')
                winner = int(indices[0]); chosen = int(p.argmax(1)[0])
                conf = score[0, indices]
                logp = values.clamp_min(1e-30).log()
                higher = conf > conf[0]+1e-7
                threshold = float(((logp[0]-logp[higher])/(conf[higher]-conf[0])).min()) if bool(higher.any()) else None
                phase = 'dispatch' if bool(env.dispatch_open[0]) else ('post_acc' if cache['post_disclosure'] is not None and float(env.time[0]) >= cache['revealed_at_min'] else 'routing')
                record = dict(name=row['name'], step=step, phase=phase, time=float(env.time[0]),
                    valid_count=len(indices), positive_bias_count=int((conf > 0).sum()),
                    coverage=float((conf > 0).float().mean()), top_probability=float(values[0]),
                    top2_gap=float(logp[0]-logp[1]) if len(indices)>1 else None,
                    bias_span=float(5*(conf.max()-conf.min())), winner_confidence=float(conf[0]),
                    beta_to_flip=threshold, changed=chosen != winner,
                    unguided_action=winner, selected_action=chosen)
                trace.append(record)
                env.step(p.argmax(1), validate=True)
            else: raise RuntimeError('Action bound')
            metrics = {k:float(v[0]) for k,v in env.metrics().items()}
            prior = read(ROOT/'result/kim_guidance_suite_20261110'/label/'original'/(row['name']+'.json'))
            assert abs(metrics['reward']-prior['metrics']['reward']) < 1e-5
            write(path, dict(metrics=metrics, decisions=trace))
            decisions.extend(trace)
        summary = dict(checkpoint=label, phases={phase:aggregate([r for r in decisions if r['phase']==phase])
            for phase in ('dispatch','routing','post_acc') if any(r['phase']==phase for r in decisions)},
            all_routing=aggregate([r for r in decisions if r['phase']!='dispatch']))
        summaries.append(summary)
        write(OUT/'summary.json', summaries)
        print(label, summary['all_routing'], flush=True)
        del model
    write(OUT/'status.json', dict(stage='complete', checkpoints=6, cases=120))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        write(OUT/'status.json', dict(stage='failed', error=repr(error)))
        raise
