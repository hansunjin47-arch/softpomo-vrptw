"""Merge validated receipt summaries, rejecting inconsistent repeated seeds."""
import json
from pathlib import Path
import statistics as st


def main():
    groups = {}
    for path in sorted(Path('analysis').glob('seq_received_summary_*.json')):
        for group in json.loads(path.read_text()):
            key = (group['instance'], group['late'])
            seeds = groups.setdefault(key, {})
            for row in group['seeds']:
                seed = row['seed']
                if seed in seeds and seeds[seed] != row:
                    raise ValueError(f'Conflicting duplicate: {key}, {seed}')
                seeds[seed] = row
    assert len(groups) == 12
    report = []
    for (case, late), seeds in sorted(groups.items()):
        assert set(seeds) == set(range(42,47)), (case,late,seeds.keys())
        rows = list(seeds.values())
        baseline = rows[0]['lkh_reward']
        assert all(abs(r['lkh_reward']-baseline)<1e-10 for r in rows)
        mean = st.mean(r['reward'] for r in rows)
        entry = dict(instance=case, late=late, count=5,
                     reward_mean=mean,reward_std=st.stdev(r['reward'] for r in rows),
                     lkh_reward=baseline,cost_reduction_percent=100*(mean-baseline)/abs(baseline),
                     wins=sum(r['reward']>baseline for r in rows),
                     metrics={k:st.mean(r[k] for r in rows) for k in
                              ('vehicles','travel_time','total_late','late_count')},seeds=rows)
        report.append(entry)
        print(case,late,'R',round(mean,6),'SD',round(entry['reward_std'],6),
              'LKH',round(baseline,6),'gain%',round(entry['cost_reduction_percent'],3),
              'wins',entry['wins'],'metrics',entry['metrics'])
    Path('analysis/seq_all_60_summary_20261026.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
