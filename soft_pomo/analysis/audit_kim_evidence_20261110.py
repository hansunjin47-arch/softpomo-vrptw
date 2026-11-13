"""Audit saved causal calls, fixed reference, and training logs without API use."""
import json
import statistics
from analysis.evaluate_r101_mask_kim_20261109 import BUNDLE, RECEIVED, OUT as TEST
from experiments.concurrent_fleet_bias_guided.support import ROOT, read
from experiments.concurrent_fleet_v9_guided.prompts import parse_test_response, parse_response
from experiments.concurrent_fleet_pair.io import write


def main():
    groups = read(BUNDLE/'artifacts/groups.json')
    base = parse_response(read(BUNDLE/'artifacts/base_confidence.json'), groups)
    reference = read(BUNDLE/'artifacts/reference.json')
    split = read(BUNDLE/'dataset/split.json')
    violations, calls, initials = [], [], []
    folders = [TEST/f'seed{s}' for s in range(42,47)]
    folders.append(ROOT/'result/r101_kim44_epoch1010_test_20261109')
    for folder in folders:
        for path in (folder/'instances').glob('*/calls.json'):
            for call in read(path):
                context = json.loads(call['request']['messages'][1]['content'])
                clock = call['clock_min']
                scores = parse_test_response(call['response'], groups, context['selectable_customers'])
                bad = [e for e in context['public_events'] if e['revealed_at_min'] > clock+1e-6]
                if bad: violations.append(dict(path=str(path), issue='undisclosed event', events=bad))
                if context['rl_reference_example'] != reference:
                    violations.append(dict(path=str(path), issue='reference mismatch'))
                if context.get('prior_base_confidence') != read(BUNDLE/'artifacts/base_confidence.json'):
                    violations.append(dict(path=str(path), issue='base mismatch'))
                selected = context['selectable_customers']
                values = [v for row in scores.values() for n, v in row.items() if n in selected]
                if call['stage'] == 'refresh':
                    if not context.get('fleet') or not context.get('previous_confidence'):
                        violations.append(dict(path=str(path), issue='missing refresh context'))
                    if set(context['fleet']['unassigned_customers']) != set(selected):
                        violations.append(dict(path=str(path), issue='remaining customer mismatch'))
                else:
                    initials.append(dict(name=path.parent.name, payload=call['response']))
                calls.append(dict(run=folder.name, name=path.parent.name, stage=call['stage'], clock=clock,
                    remaining=len(selected), score_mean=statistics.mean(values) if values else None,
                    has_live_fleet='fleet' in context))
    cache_checks = []
    for row in split['train']:
        cache = read(BUNDLE/'artifacts/train_cache'/(row['name']+'.json'))
        initial = parse_response(cache['initial'], groups)
        accident = '_ACC_' in row['name']
        cache_checks.append(dict(name=row['name'], accident=accident, initial_equals_base=initial == base,
            has_post=cache['post_disclosure'] is not None, activation=cache.get('revealed_at_min')))
        if accident and (initial != base or cache['post_disclosure'] is None):
            violations.append(dict(name=row['name'], issue='ACC cache stage mismatch'))
    training = []
    fairness = []
    pure_root = ROOT/'server_export/pair_mask_completed_received_20261106_123111/pair_returnmask_cr_v100_200_l5_s42_46_20261031/runs/r101/vehicle100'
    for seed in range(42,47):
        pure = read(pure_root/f'seed{seed}/protocol.json')
        kim = read(RECEIVED/f'runs/seed{seed}/protocol.json')
        fairness.append(dict(seed=seed, configs_equal=pure['configs'] == kim['configs'],
            split_equal=read(pure_root/f'seed{seed}/split.json') == read(RECEIVED/f'runs/seed{seed}/split.json'),
            exploration_equal=pure['new_customer_exploration'] == kim['new_customer_exploration']))
        path = RECEIVED/f'runs/seed{seed}/late5/training.jsonl'
        logs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        tail = logs[-200:]
        grads = [r['gradient_norm'] for r in logs if r.get('gradient_norm') is not None]
        training.append(dict(seed=seed, updates=len(logs), amp_skips=sum(bool(r.get('optimizer_step_skipped')) for r in logs),
            max_gradient=max(grads), tail_amp_skips=sum(bool(r.get('optimizer_step_skipped')) for r in tail),
            tail_sampling_reward=statistics.mean(r['reward'] for r in tail),
            tail_late=statistics.mean(r['total_late'] for r in tail),
            worst_tail_batch=min(tail, key=lambda r:r['reward'])))
    assert reference['scenario'] not in {r['name'] for r in split['train']+split['test']}
    report = dict(violations=violations, audited_calls=len(calls), calls=calls, train_cache=cache_checks,
        training=training, fairness=fairness, reference_scenario=reference['scenario'],
        limits=['Reference-name separation checked; scenario uniqueness also requires event-content audit.',
                'Causal checks concern serialized LLM inputs; they are not full simulator-information-flow proof.',
                'Tail sampling means are batch-unweighted, not full Train20 greedy.'])
    write(ROOT/'result/kim_guidance_suite_20261110/evidence_audit.json', report)
    print('AUDIT', len(calls), 'calls; violations:', violations, flush=True)
    print('TRAINING', [{k:v for k,v in r.items() if k != 'worst_tail_batch'} for r in training], flush=True)


if __name__ == '__main__':
    main()
