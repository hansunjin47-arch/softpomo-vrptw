"""Compare generated RC train guidance with the submitted Kim cache."""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'result/kim_rl_example_rc_train_20261027'
SERVER = ROOT / 'server_export/kim_l5_rc_r_c_s42_46_20261026/rc101'
BASE = ROOT / 'result/kim_rl_example_base_pilot_rc_20261027/cache/BASE_cluster.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def flatten(cache):
    return {int(n): v for k, group in cache.items() if not k.startswith('_') for n, v in group.items()}


def main():
    base = flatten(read(BASE))
    names = read(NEW / 'protocol.json')['names']
    rows, arrays = [], {}
    for name in names:
        filename = name + '_cluster.json'
        old = read(SERVER / 'seed42/cache/train' / filename)
        new = read(NEW / 'cache' / filename)
        for seed in range(43, 47):
            assert read(SERVER / f'seed{seed}/cache/train' / filename) == old
        a, b = flatten(old), flatten(new)
        assert set(a) == set(b) == set(range(1, 101))
        av, bv = np.array([a[n] for n in sorted(a)]), np.array([b[n] for n in sorted(a)])
        arrays[name] = (av, bv)
        ranks, top_changed, flips, comparable = [], 0, 0, 0
        spreads = [[], []]
        for k in old:
            if k.startswith('_'):
                continue
            assert set(old[k]) == set(new[k])
            nodes = list(old[k])
            x, y = [old[k][n] for n in nodes], [new[k][n] for n in nodes]
            if len(set(x)) > 1 and len(set(y)) > 1:
                ranks.append(float(spearmanr(x, y).statistic))
            spreads[0].append(float(np.std(x)))
            spreads[1].append(float(np.std(y)))
            top_changed += {n for n in nodes if old[k][n] == max(x)} != {n for n in nodes if new[k][n] == max(y)}
            for i, j in itertools.combinations(nodes, 2):
                dx, dy = old[k][i] - old[k][j], new[k][i] - new[k][j]
                if dx and dy:
                    comparable += 1
                    flips += dx * dy < 0
        changed = [n for n in a if abs(b[n] - base[n]) > 1e-8]
        affected = set(read(NEW / 'requests' / name / 'context.json')['public_event']['affected_nodes'])
        rows.append(dict(name=name, old_mean=float(av.mean()), new_mean=float(bv.mean()),
                         mae=float(np.abs(av-bv).mean()), group_rank_correlation=float(np.nanmean(ranks)),
                         old_group_std=float(np.mean(spreads[0])), new_group_std=float(np.mean(spreads[1])),
                         top_set_changed=int(top_changed), strict_pair_flips=int(flips), comparable_pairs=comparable,
                         changed_from_base=len(changed), changed_affected=len(set(changed) & affected),
                         affected_count=len(affected), changed_outside=sorted(set(changed)-affected)))
    summary = {}
    for event in ('RAIN', 'ACC'):
        selected = [r for r in rows if f'_{event}_' in r['name']]
        old, new = (np.stack([arrays[r['name']][i] for r in selected]) for i in (0, 1))
        summary[event] = dict(
            old_mean=float(old.mean()), new_mean=float(new.mean()),
            old_within_instance_std=float(old.std(axis=1).mean()), new_within_instance_std=float(new.std(axis=1).mean()),
            mae=float(np.abs(old-new).mean()),
            mean_group_rank_correlation=float(np.mean([r['group_rank_correlation'] for r in selected])),
            old_group_std=float(np.mean([r['old_group_std'] for r in selected])),
            new_group_std=float(np.mean([r['new_group_std'] for r in selected])),
            top_set_changed=sum(r['top_set_changed'] for r in selected), group_count=160,
            strict_pair_flip_rate=sum(r['strict_pair_flips'] for r in selected)/sum(r['comparable_pairs'] for r in selected),
            old_mean_customer_across_instances_std=float(old.std(axis=0).mean()),
            new_mean_customer_across_instances_std=float(new.std(axis=0).mean()),
            mean_changed_from_base=float(np.mean([r['changed_from_base'] for r in selected])))
    result = dict(old_path=str(SERVER), new_path=str(NEW), all_five_seed_caches_identical=True,
                  summary=summary, instances=rows)
    path = NEW / 'comparison_with_server.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
