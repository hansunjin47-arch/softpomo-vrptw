"""Locate paired confidence changes; distinguish associations from causal effects."""
import json
from statistics import mean

from audit_event_relations import ROOT, read, sha


def main():
    old = ROOT / 'result/rc101_pure500_fullconfidence_phase2'
    new = ROOT / 'result/rc101_relation_train20_cache_v1'
    names = read(new / 'protocol.json')['train_names']
    rows, hashes = [], {}

    def tracked(path):
        hashes[str(path.relative_to(ROOT))] = sha(path)
        return read(path)

    for name in names:
        before = tracked(old / 'cache' / (name + '_rl_cluster.json'))
        after = tracked(new / 'cache' / (name + '_rl_cluster.json'))
        ref = tracked(new / 'requests' / name / 'reference.json')
        graph = tracked(new / 'requests' / name / 'relations.json')
        legs = {x['destination']: x for x in graph['legs'] if x['destination']}
        for g, customers in enumerate(ref['clusters']):
            for c in customers:
                leg = legs[c]
                direct = leg['reference_exposure']['exposed']
                indirect = leg['arrival_delay'] > 1e-7 and not direct
                category = 'direct' if direct else 'indirect_only' if indirect else 'no_arrival_impact'
                rows.append(dict(name=name, group=g+1, customer=c, category=category,
                    old=before[str(g)][str(c)], new=after[str(g)][str(c)],
                    delta=after[str(g)][str(c)]-before[str(g)][str(c)],
                    incoming=leg['incoming_delay'], added_travel=leg['added_travel'],
                    absorbed=leg['wait_absorption'], forwarded=leg['outgoing_delay'],
                    lateness=leg['reference']['late']))

    def stats(xs):
        return dict(count=len(xs), changed=sum(abs(x['delta']) > 1e-8 for x in xs),
            mean_delta=mean(x['delta'] for x in xs), mae=mean(abs(x['delta']) for x in xs))

    a = tracked(old / 'eval_1000_test80/results.json')
    b = tracked(new / 'eval_1000_test80/results.json')
    assert [x['name'] for x in a] == [x['name'] for x in b]
    assert all(x['served'] == 100 for x in a + b)
    def cost(x):
        return .04*x['vehicles'] + x['travel_time']/5760 + .0075*x['total_late']
    paired = []
    for x, y in zip(a, b):
        paired.append(dict(name=x['name'], kind=x['kind'],
            **{k:y[k]-x[k] for k in ('vehicles', 'travel_time', 'late_count', 'total_late')},
            cost=cost(y)-cost(x)))
    report = dict(scope='Train20 cache changes and downstream frozen test80 outcomes; not causal attribution',
        confidence_categories={k:stats([x for x in rows if x['category']==k])
                               for k in ('direct','indirect_only','no_arrival_impact')},
        examples={k:sorted([x for x in rows if x['category']==k], key=lambda x:abs(x['delta']), reverse=True)[:5]
                  for k in ('direct','indirect_only','no_arrival_impact')},
        test80=dict(old_cost=mean(map(cost,a)), new_cost=mean(map(cost,b)),
            paired={k:dict(improved=sum(x[k]<-1e-7 for x in paired),
                           unchanged=sum(abs(x[k])<=1e-7 for x in paired),
                           worsened=sum(x[k]>1e-7 for x in paired))
                    for k in ('vehicles','travel_time','total_late','cost')}),
        customer_rows=rows, test_pairs=paired, input_sha256=hashes)
    target = new / 'change_localization.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('customer_rows','test_pairs','input_sha256')}, indent=2))


if __name__ == '__main__':
    main()
