"""CPU-only audit of saved guidance content and disclosure snapshots; no API calls."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'server_export/r101_pair100_l5_mask_kim_s42_46_20261106'
OUT = ROOT / 'result/kim_confidence_content_20261110'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def scores(payload):
    rows = payload['confidence']
    if isinstance(rows, dict):
        return {int(k): float(v) for k, v in rows.items()}
    return {int(r['customer']): float(r['score']) for r in rows}


def mean(values):
    return statistics.mean(values) if values else None


def main():
    groups = read(BUNDLE / 'artifacts/groups.json')
    membership = {n: i + 1 for i, group in enumerate(groups) for n in group}
    base = scores(read(BUNDLE / 'artifacts/base_confidence.json'))
    train = []
    for path in (BUNDLE / 'artifacts/train_cache').glob('*.json'):
        cache = read(path)
        event = read(BUNDLE / 'dataset/data' / (path.stem + '.observations.json'))
        affected = set(event['affected_nodes'])
        final = scores(cache['post_disclosure'] or cache['initial'])
        changed = {n for n in base if abs(final[n] - base[n]) > 1e-8}
        train.append(dict(name=path.stem, affected=len(affected), changed=len(changed),
            changed_affected=len(changed & affected), changed_other=len(changed - affected),
            mean_abs_delta_affected=mean([abs(final[n]-base[n]) for n in affected]),
            mean_abs_delta_other=mean([abs(final[n]-base[n]) for n in base if n not in affected])))
    refresh = []
    mismatches = []
    pair_wording_calls = 0
    call_count = 0
    reference = read(BUNDLE / 'artifacts/reference.json')
    folders = [ROOT / f'result/r101_pair100_l5_mask_kim_test_20261109/seed{s}' for s in range(42,47)]
    folders += [ROOT / 'result/r101_kim44_epoch1010_test_20261109']
    for folder in folders:
        for path in (folder / 'instances').glob('*/calls.json'):
            for call in read(path):
                call_count += 1
                pair_wording_calls += 'RL selects an eligible existing-vehicle/destination pair after initial dispatch.' in call['request']['messages'][0]['content']
                ctx = json.loads(call['request']['messages'][1]['content'])
                if ctx.get('rl_reference_example') != reference:
                    mismatches.append(str(path))
                if call['stage'] != 'refresh':
                    continue
                remaining = set(ctx['selectable_customers'])
                vehicles = ctx['fleet']['vehicles']
                anchors = {v['anchor_cluster'] for v in vehicles if v['status'] != 'retired'}
                reachable = {n for n in remaining if membership[n] in anchors}
                fixed = ctx['fixed_instance']
                columns = fixed['customer_columns']
                demand = {r[columns.index('id')]: r[columns.index('demand')]/fixed['capacity'] for r in fixed['customers']}
                capable = {n for n in remaining if any(
                    v['status'] != 'retired'
                    and not (v['status'] == 'moving' and v['committed_destination'] == 0)
                    and v['anchor_cluster'] == membership[n]
                    and v['remaining_capacity_fraction'] + 1e-5 >= demand[n]
                    for v in vehicles)}
                previous, after = scores(ctx['previous_confidence']), scores(call['response'])
                changed = {n for n in remaining if abs(after[n]-previous[n]) > 1e-8}
                affected = set().union(*(set(e['affected_nodes']) for e in ctx['public_events']))
                refresh.append(dict(run=folder.name, name=path.parent.name, clock=call['clock_min'],
                    remaining=len(remaining), changed=len(changed),
                    anchor_covered=len(reachable), changed_anchor_covered=len(changed & reachable),
                    capable_anchor_covered=len(capable), changed_capable_anchor_covered=len(changed & capable),
                    changed_unreachable=len(changed - reachable), affected_remaining=len(affected & remaining),
                    affected_changed=len(changed & affected),
                    mean_abs_delta=mean([abs(after[n]-previous[n]) for n in remaining]),
                    statuses=sorted({v['status'] for v in vehicles})))
    totals = {key: sum(r[key] for r in refresh) for key in ['remaining', 'changed', 'anchor_covered', 'changed_anchor_covered', 'changed_unreachable', 'affected_remaining', 'affected_changed', 'capable_anchor_covered', 'changed_capable_anchor_covered']}
    breakdown = {}
    for label, rows in [('original_five_seeds', [r for r in refresh if r['run'].startswith('seed')]),
                        ('recovered_seed44_epoch1010', [r for r in refresh if not r['run'].startswith('seed')])]:
        breakdown[label] = dict(calls=len(rows), totals={k: sum(r[k] for r in rows) for k in totals})
    train_requests = []
    for path in (ROOT/'result/r101_pair100_l5_mask_kim_cache_20261106/calls').rglob('request.json'):
        request = read(path)
        context = json.loads(request['messages'][1]['content'])
        train_requests.append(dict(path=str(path.relative_to(ROOT)), stage=context.get('stage'),
            misleading_pair_wording='RL selects an eligible existing-vehicle/destination pair after initial dispatch.' in request['messages'][0]['content']))
    summary = dict(refresh_calls=len(refresh), totals=totals,
        remaining_mean=mean([r['remaining'] for r in refresh]),
        remaining_range=[min(r['remaining'] for r in refresh), max(r['remaining'] for r in refresh)],
        calls_no_changed_covered=sum(r['changed_anchor_covered']==0 for r in refresh),
        mean_abs_delta_per_call=mean([r['mean_abs_delta'] for r in refresh if r['mean_abs_delta'] is not None]),
        anchor_coverage_upper_bound=totals['anchor_covered']/totals['remaining'],
        changed_anchor_coverage_upper_bound=totals['changed_anchor_covered']/totals['changed'],
        reference_mismatch=mismatches,
        total_calls=call_count, potentially_misleading_pair_selection_wording_calls=pair_wording_calls,
        checkpoint500_files=[str(p.relative_to(ROOT)) for p in (ROOT/'server_export/r101_pair100_l5_mask_kim_s42_46_20261106_received').rglob('checkpoint-500.pt')])
    report = dict(summary=summary, refresh_breakdown=breakdown, train_requests=train_requests,
        proposed_correct_wording='Vehicles operate concurrently on a common clock. During initial dispatch, RL chooses NEW(customer) or an existing-vehicle/destination pair; choosing an existing pair closes dispatch. After dispatch closes, the environment designates one ready vehicle and RL chooses only its next destination. RL cannot select a different vehicle at that decision.',
        train=train, refresh=refresh, limits=[
        'Anchor coverage is an upper bound without capacity check. Capable coverage adds current capacity and excludes depot-bound vehicles, but not future readiness, retirement or future action masks.',
        'Affected means geographically reported affected_nodes; it is not measured marginal delay or lateness.',
        'Calls are correlated across seeds and include epoch1010; counts are descriptive, not independent statistical samples.',
        'Only the relevant downloaded experiment tree was searched for checkpoint500.'])
    OUT.mkdir(exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
