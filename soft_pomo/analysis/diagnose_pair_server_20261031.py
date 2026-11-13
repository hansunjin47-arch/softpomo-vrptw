"""Inspect server seed stability from downloaded logs; no inference or API calls."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'server_export/pair_completed_received_20261031_162306/pair_pure_v100_200_l5_s42_46_20261029'


def summarize(rows):
    n = sum(r['batch_size'] for r in rows)
    result = {k:sum(r[k]*r['batch_size'] for r in rows)/n for k in
              ('reward','best_rollout_reward','vehicles','travel_time','total_late','unserved')}
    result['skips'] = sum(r['optimizer_step_skipped'] for r in rows)
    result['max_gradient'] = max(r['gradient_norm'] or 0 for r in rows)
    return result


def main():
    report = {}
    for seed in range(42,46):
        folder = SOURCE/f'vehicle200/seed{seed}/runs/pure/late5'
        updates = [json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        assert len(updates) == 20000 and len({r['update'] for r in updates}) == 20000
        epochs = {ep:summarize([r for r in updates if r['epoch']==ep]) for ep in range(1,1001)}
        blocks = {f'{a}-{b}':summarize([r for r in updates if a<=r['epoch']<=b])
                  for a,b in ((101,200),(401,500),(701,800),(901,1000),(981,1000))}
        jumps = sorted(range(101,1001), key=lambda ep:epochs[ep]['reward']-epochs[ep-1]['reward'])[:5]
        spikes = sorted([r for r in updates if r['epoch']>=100], key=lambda r:r['gradient_norm'] or 0, reverse=True)[:5]
        tests = json.loads((folder/'results.json').read_text())
        failures = [r for r in tests if r['unserved']>0]
        groups = {kind:{k:mean(r[k] for r in tests if kind in r['name']) for k in
                       ('reward','vehicles','travel_time','total_late','unserved')} for kind in ('RAIN','ACC')}
        report[seed] = dict(blocks=blocks, groups=groups, failures=failures,
            transition_epochs={ep:epochs[ep] for ep in range(665,686)},
            jumps=[dict(epoch=ep,before=epochs[ep-1],after=epochs[ep]) for ep in jumps],
            spikes=[{k:r[k] for k in ('epoch','update','name','gradient_norm','reward','unserved')} for r in spikes],
            last_epochs={ep:epochs[ep] for ep in range(991,1001)},
            test_without_unserved_penalty=mean(r['reward']+5*r['unserved'] for r in tests),
            test_success_only=mean(r['reward'] for r in tests if r['unserved']==0),
            highest_late=sorted(tests,key=lambda r:r['total_late'],reverse=True)[:5])
    output = ROOT/'analysis/pair_server_diagnosis_20261031.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    for seed,r in report.items():
        print('SEED',seed,'BLOCKS',json.dumps(r['blocks']))
        print('GROUPS',json.dumps(r['groups']))
        print('TOP_SPIKES',json.dumps(r['spikes'][:2]))
        print('JUMP',json.dumps(r['jumps'][:1]))
        print('TEST_NO_UNSERVED_PENALTY',r['test_without_unserved_penalty'])
        print('HIGH_LATE',[(v['name'],round(v['total_late'],2),v['unserved']) for v in r['highest_late']])


if __name__ == '__main__':
    main()
