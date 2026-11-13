"""Matched confidence and within-group ordering changes, without policy training."""
import json
from collections import defaultdict

from audit_event_relations import ROOT, read


def main():
    run = ROOT / 'result/rc101_relation_train20_cache_v1'
    protocol = read(run / 'protocol.json')
    source = ROOT / 'result/rc101_pure500_fullconfidence_phase2'
    samples, depots, scenarios = [], [], []
    ranks = defaultdict(int)
    for name in protocol['train_names']:
        original = read(source / 'cache' / (name + '_rl_cluster.json'))
        updated = read(run / 'cache' / (name + '_rl_cluster.json'))
        reference = read(run / 'requests' / name / 'reference.json')
        audit = read(run / 'requests' / name / 'relations.json')
        by_customer = {x['destination']: x for x in audit['legs'] if x['destination']}
        kind = 'RAIN' if '_RAIN_' in name else 'ACCIDENT'
        local = []
        for index, cluster in enumerate(reference['clusters']):
            old, new = original[str(index)], updated[str(index)]
            depot_old, depot_new = original['_depot'][str(index)], updated['_depot'][str(index)]
            depots.append(dict(name=name, group=index + 1, old=depot_old, new=depot_new,
                               delta=depot_new-depot_old))
            for customer in cluster:
                leg = by_customer[customer]
                row = dict(name=name, kind=kind, group=index+1, customer=customer,
                    old=old[str(customer)], new=new[str(customer)],
                    delta=new[str(customer)]-old[str(customer)],
                    direct=leg['reference_exposure']['exposed'],
                    inherited=leg['incoming_delay'] > 1e-7,
                    absorbed=leg['wait_absorption'] > 1e-7,
                    late=leg['reference']['late'] > 1e-7)
                row['depot_order_changed'] = ((row['old'] > depot_old) != (row['new'] > depot_new))
                local.append(row)
            for i, a in enumerate(cluster):
                for b in cluster[i+1:]:
                    before, after = old[str(a)]-old[str(b)], new[str(a)]-new[str(b)]
                    ranks['pairs'] += 1
                    if before * after < -1e-12:
                        ranks['strict_reversals'] += 1
                    if abs(before) < 1e-8 and abs(after) >= 1e-8:
                        ranks['ties_broken'] += 1
                    if abs(before) >= 1e-8 and abs(after) < 1e-8:
                        ranks['ties_created'] += 1
        samples.extend(local)
        scenarios.append(dict(name=name, mae=sum(abs(x['delta']) for x in local)/len(local)))
    def aggregate(rows):
        return dict(count=len(rows), mean_old=sum(x['old'] for x in rows)/len(rows) if rows else None,
            mean_new=sum(x['new'] for x in rows)/len(rows) if rows else None,
            mae=sum(abs(x['delta']) for x in rows)/len(rows) if rows else None,
            increased=sum(x['delta'] > 1e-8 for x in rows),
            decreased=sum(x['delta'] < -1e-8 for x in rows),
            unchanged=sum(abs(x['delta']) <= 1e-8 for x in rows))
    report = dict(customer=aggregate(samples), depot=aggregate(depots), ordering=dict(ranks),
        by_event={kind: aggregate([x for x in samples if x['kind'] == kind]) for kind in ('RAIN','ACCIDENT')},
        overlapping_subgroups={key: aggregate([x for x in samples if x[key]])
                              for key in ('direct','inherited','absorbed','late')},
        largest_changes=sorted(samples, key=lambda x: abs(x['delta']), reverse=True)[:20],
        scenarios=scenarios,
        customer_depot_order_changes=sum(x['depot_order_changed'] for x in samples),
        caveat='One cached response per prompt. Added counterfactual information and soft-semantics instructions confound an isolated ontology effect; no repeated-call control or routing-quality inference.')
    (run / 'confidence_analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('scenarios', 'largest_changes')}, indent=2))


if __name__ == '__main__':
    main()
