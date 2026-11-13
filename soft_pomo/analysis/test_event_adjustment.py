"""Simulator-assisted, relation-targeted route repair diagnostic (not LLM guidance)."""
import argparse
import copy
import json
from pathlib import Path
import time

from audit_event_relations import ROOT, EPS, analyze, load_solomon, read, sha, _solution_stats


def metrics(stats):
    k = len(stats['per_vehicle'])
    t, late = stats['total_travel_time'], stats['total_late']
    return dict(vehicles=k, travel_time=t, total_late=late, late_count=stats['late_count'],
                cost=.04 * k + t / 5760 + .0075 * late, served=stats['n_served'])


def valid(routes, stats, inst):
    customers = [n for r in routes for n in r[1:-1]]
    return (sorted(customers) == list(range(1, inst['n_customers'] + 1))
            and all(r[0] == r[-1] == 0 and len(r) >= 3 for r in routes)
            and len(routes) <= inst['vehicle_limit']
            and all(v['load'] <= inst['vehicle_capacity'] + 1e-4 for v in stats['per_vehicle']))


def relocate(routes, source, position, target, insertion):
    candidate = copy.deepcopy(routes)
    customer = candidate[source].pop(position)
    # Insertion refers to the destination sequence after removal for same-route moves.
    candidate[target].insert(insertion, customer)
    return [r for r in candidate if len(r) > 2]


def targets(audit, routes):
    selected = {}
    for k, route in enumerate(routes, 1):
        legs = [leg for leg in audit['legs'] if leg['route'] == f'r{k}']
        late_positions = [j for j, leg in enumerate(legs) if leg['added_lateness'] > EPS]
        if not late_positions:
            continue
        # Include late customers and upstream exposed endpoints that can initiate the delay.
        eligible = set()
        for j, leg in enumerate(legs):
            if leg['added_lateness'] > EPS and leg['destination']:
                eligible.add(leg['destination'])
            if j <= max(late_positions) and leg['reference_exposure']['exposed']:
                eligible.update(n for n in (leg['origin'], leg['destination']) if n)
        selected[k - 1] = eligible
    return selected


def repair(inst, routes, observation, rounds=3, receivers=3):
    current = copy.deepcopy(routes)
    history, evaluated, rejected_late = [], 0, 0
    initial = _solution_stats(inst, current)
    assert valid(current, initial, inst)
    for iteration in range(rounds):
        audit, stats = analyze(inst, current, observation)
        before = metrics(stats)
        best, seen = None, set()
        selected = targets(audit, current)
        for source, customers in selected.items():
            for position, customer in enumerate(current[source][1:-1], 1):
                if customer not in customers:
                    continue
                nearby = sorted((j for j in range(len(current)) if j != source),
                    key=lambda j: (min(float(stats['tt'][customer, n]) for n in current[j][1:-1]), j))[:receivers]
                for target in [source] + nearby:
                    if source != target and stats['per_vehicle'][target]['load'] + stats['demands'][customer] > inst['vehicle_capacity'] + 1e-4:
                        continue
                    length = len(current[target]) - (source == target)
                    for insertion in range(1, length):
                        candidate = relocate(current, source, position, target, insertion)
                        signature = tuple(tuple(r) for r in candidate)
                        if signature in seen or candidate == current:
                            continue
                        seen.add(signature)
                        outcome = _solution_stats(inst, candidate)
                        assert valid(candidate, outcome, inst)
                        evaluated += 1
                        after = metrics(outcome)
                        if after['total_late'] > before['total_late'] + EPS:
                            rejected_late += 1
                            continue
                        if after['cost'] >= before['cost'] - EPS:
                            continue
                        key = (after['cost'], after['total_late'], after['travel_time'])
                        if best is None or key < best[0]:
                            best = (key, candidate, dict(iteration=iteration + 1, customer=customer,
                                source_route=source + 1, target_route=target + 1,
                                insertion_index=insertion, kind='reorder' if source == target else 'reassign',
                                before=before, after=after, source_relation_ids=[
                                    leg['id'] for leg in audit['legs'] if leg['route'] == f'r{source+1}'
                                    and (leg['added_lateness'] > EPS or leg['reference_exposure']['exposed'])]))
        if best is None:
            break
        _, current, move = best
        history.append(move)
    final = _solution_stats(inst, current)
    assert valid(current, final, inst)
    assert final['total_late'] <= initial['total_late'] + EPS
    assert metrics(final)['cost'] <= metrics(initial)['cost'] + EPS
    # Count newly harmed visits even when total lateness decreases.
    def lateness_by_customer(stats):
        return {s['node']: s['late'] for v in stats['per_vehicle'] for s in v['stop_details']}
    old_late, new_late = lateness_by_customer(initial), lateness_by_customer(final)
    return dict(before=metrics(initial), after=metrics(final), routes=current, moves=history,
                evaluated_candidates=evaluated, rejected_for_total_lateness=rejected_late,
                customers_with_increased_lateness=[n for n in old_late if new_late[n] > old_late[n] + EPS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'analysis/pure500_rain_adjustment_audit')
    ap.add_argument('--rounds', type=int, default=3)
    args = ap.parse_args()
    assert args.rounds > 0
    data = ROOT / 'data/rc101_train80_test80_20260929'
    source = ROOT / 'result/rc101_pure_train80_fresh1000_20260929/eval_500/results.json'
    split = read(data / 'split.json')
    hashes = {r['name']: r['sha256'] for r in split['test']}
    args.out.mkdir(parents=True, exist_ok=False)
    started, results = time.monotonic(), []
    for row in read(source):
        if row['kind'] != 'RAIN':
            continue
        path = data / (row['name'] + '.txt')
        assert sha(path) == hashes[row['name']]
        result = repair(load_solomon(str(path)), row['routes'],
                        read(data / (row['name'] + '.observations.json')), args.rounds)
        assert abs(result['before']['total_late'] - row['total_late']) < EPS
        assert abs(result['before']['travel_time'] - row['travel_time']) < EPS
        result['name'] = row['name']
        results.append(result)
        (args.out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f'{len(results)}/40 {row["name"]}: moves={len(result["moves"])} '
              f'Lt={result["before"]["total_late"]:.2f}->{result["after"]["total_late"]:.2f}', flush=True)
    summary = dict(count=len(results), seconds=time.monotonic() - started,
        improved=sum(r['after']['cost'] < r['before']['cost'] - EPS for r in results),
        candidates=sum(r['evaluated_candidates'] for r in results),
        moves=sum(len(r['moves']) for r in results),
        scenarios_with_harmed_customers=sum(bool(r['customers_with_increased_lateness']) for r in results))
    for stage in ('before', 'after'):
        summary[stage] = {key: sum(r[stage][key] for r in results) / len(results) for key in results[0][stage]}
        summary[stage]['zero_late'] = sum(r[stage]['late_count'] == 0 for r in results)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (args.out / 'protocol.json').write_text(json.dumps(dict(
        method='Relation-targeted simulator-assisted greedy relocation; not LLM confidence or OWL reasoning',
        source=str(source), source_sha256=sha(source), script_sha256=sha(Path(__file__)),
        relation_script_sha256=sha(ROOT / 'analysis/audit_event_relations.py'),
        rounds=args.rounds, receivers=3, acceptance='Strict Config E cost improvement and nonincreasing total lateness',
        scope='Exploratory rain test40; exact hidden simulator effects used to select moves',
        limitations='Not online; no objective/information-matched search control; no ontology causal claim; per-customer lateness may increase'
    ), indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
