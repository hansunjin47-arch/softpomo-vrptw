"""Matched confidence and pairwise-order diagnostics; not routing evaluation."""
import json
from itertools import combinations
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT/'result/rc101_phase2_train80_news_only_20261001'
OLD = ROOT/'result/rc101_pure500_train80_phase2_20260930'


def main():
    rows = []
    for path in sorted((NEW/'cache').glob('*_rl_cluster.json')):
        a = json.loads((OLD/'cache'/path.name).read_text())
        b = json.loads(path.read_text())
        assert a.keys() == b.keys()
        differences, changed, pairs, flips, rank_changes = [], 0, 0, 0, 0
        for group in a:
            assert a[group].keys() == b[group].keys()
            if group.startswith('_'):
                continue
            for customer in a[group]:
                delta = abs(a[group][customer] - b[group][customer])
                differences.append(delta)
                changed += delta > 1e-9
            for i, j in combinations(a[group], 2):
                x, y = a[group][i]-a[group][j], b[group][i]-b[group][j]
                sx, sy = (x > 1e-9)-(x < -1e-9), (y > 1e-9)-(y < -1e-9)
                pairs += 1
                flips += sx*sy < 0
                rank_changes += sx != sy
        rows.append(dict(name=path.name.removesuffix('_rl_cluster.json'),
                         customer_mae=mean(differences), changed_customers=changed,
                         pairs=pairs, strict_reversals=flips, order_or_tie_changes=rank_changes,
                         old_nonzero_depots=sum(v != 0 for v in a['_depot'].values()),
                         new_nonzero_depots=sum(v != 0 for v in b['_depot'].values())))
    assert rows
    report = dict(completed=len(rows), expected=80, routing_evaluation=False,
                  mean_customer_mae=mean(r['customer_mae'] for r in rows),
                  mean_changed_customers=mean(r['changed_customers'] for r in rows),
                  strict_reversal_rate=sum(r['strict_reversals'] for r in rows)/sum(r['pairs'] for r in rows),
                  order_or_tie_change_rate=sum(r['order_or_tie_changes'] for r in rows)/sum(r['pairs'] for r in rows),
                  old_nonzero_depots=sum(r['old_nonzero_depots'] for r in rows),
                  new_nonzero_depots=sum(r['new_nonzero_depots'] for r in rows), rows=rows)
    (NEW/'cache_comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
