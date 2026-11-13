"""Per-event static full-information LKH reference, evaluated dynamically."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import datetime
import json
from eval_fixed_v2_lkh import ROOT, solve, sha, write, lkh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split-group', choices=['train', 'test'], default='test')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'manifest.json').exists()
    run = ROOT / 'result/rc101_pilot_discrete_pure1000_20260923_115506'
    split = json.loads((run / 'provenance/split.json').read_text())
    data = Path(split['data'])
    entries = split[args.split_group]
    workers = min(max(1, args.workers), len(entries))
    for entry in entries:
        assert sha(data / (entry['name'] + '.txt')) == entry['sha256']
    manifest = dict(started=datetime.datetime.now().isoformat(), workers=workers,
        time_limit=180, runs=1, max_trials=10000, expected=len(entries), completed=0,
        split_group=args.split_group,
        split=split, results=[],
        planning='Per-instance full event multiplier applied uniformly across the horizon; OR-zone; np.rint matrix.',
        interpretation='Static full-information reference, NOT a time-dependent optimal oracle. No-route is not proof of infeasibility.',
        replay='Current float time-dependent simulator; depot lateness included.',
        code_hashes={str(p):sha(p) for p in [Path(__file__),Path(lkh.__file__),ROOT/'analysis/eval_fixed_v2_lkh.py',ROOT/'vrptw_env.py']})
    write(out / 'manifest.json', manifest)
    baselines = {epoch: {r['name']:r for r in json.loads((run / folder / 'results.json').read_text())}
                 for epoch,folder in [('500','eval_500'),('1000','eval_1000_saved_routes')]}
    print(f'START: {len(entries)} {args.split_group} events; {workers} workers; 180 seconds per solve; no fallback', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(solve, entry['kind'], data/(entry['name']+'.txt'), out, 180):entry for entry in entries}
        for future in as_completed(jobs):
            row = future.result()
            row['kind'] = jobs[future]['kind']
            row['rl'] = {epoch:{k:r[row['name']][k] for k in ['vehicles','travel_time','late_count','total_late']}
                         for epoch,r in baselines.items() if row['name'] in r}
            manifest['results'].append(row)
            manifest['completed'] += 1
            write(out / 'manifest.json', manifest)
            print(f'PROGRESS {manifest["completed"]}/{len(entries)} {row["name"]}: {row["status"]}', flush=True)
    manifest['finished'] = datetime.datetime.now().isoformat()
    write(out / 'manifest.json', manifest)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
