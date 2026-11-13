"""Replay RC training cases using exactly the archived held-out RC101 history matrix."""
import json
import runpy
import argparse
import numpy as np
from eval_fixed_v2_lkh import ROOT, lkh, load_solomon, _solution_stats, sha, write

parser = argparse.ArgumentParser()
parser.add_argument('--rc105-retry', action='store_true')
args = parser.parse_args()
limit = 180 if args.rc105_retry else 60
source = ROOT/'analysis/solomon_history_heldout101_20260921_150450/RC101_history_matrix.json'
saved = json.loads(source.read_text())
data = ROOT.parent/'data/Solomon'
out = ROOT/'analysis/solomon_rc_history_train_same_20260922'
if args.rc105_retry:
    out = ROOT/'analysis/solomon_rc105_history_retry180_20260922'
out.mkdir(exist_ok=False)
assert len(saved['sources']) == 21
for name, digest in saved['sources'].items():
    assert sha(data/name) == digest, f'Input changed: {name}'
reference = lkh.load_solomon(str(data/'rc101.txt'))
write(out/'manifest.json', dict(matrix_source=str(source), matrix_sha256=sha(source),
    sources=saved['sources'], time_limit=limit, runs=1, max_trials=10000,
    policy='Same fixed training-only matrix as RC101; one route replayed under all three scenarios.'))
rows = []
for number in ([105] if args.rc105_retry else range(102,109)):
    name = f'RC{number}'
    raw = lkh.load_solomon(str(data/(name.lower()+'.txt')))
    assert np.array_equal(raw['coords'], reference['coords'])
    print('SOLVE', name, flush=True)
    result = lkh._solve_once(raw, saved['matrix'], limit, None, None, 1, 10000)
    routes = [[int(n) for n in r if n] for r in result.get('routes', [])]
    routes = [r for r in routes if r]
    row = dict(name=name, solver=result, routes=routes, scenarios=[], status='no_route')
    if routes:
        assert sorted(n for r in routes for n in r) == list(range(1,101))
        assert len(routes) <= raw['vehicle_limit']
        assert all(sum(raw['demands'][n] for n in r) <= raw['vehicle_capacity'] for r in routes)
        row['status'] = 'evaluated'
        for suffix in ['', '_RAIN2', '_ACC2']:
            path = data/(name.lower()+suffix.lower()+'.txt')
            st = _solution_stats(load_solomon(str(path)), [[0,*r,0] for r in routes])
            metric = dict(name=name+suffix, vehicles=len(routes), travel_time=st['total_travel_time'],
                          late_stops=st['late_count'],total_late=st['total_late'],served=st['n_served'])
            row['scenarios'].append(metric)
            write(out/(name+suffix+'_replay.json'),st)
            print('RESULT',metric,flush=True)
    write(out/(name+'.json'),row)
    rows.append(row)
    write(out/'results.json',rows)
runpy.run_path(str(ROOT/'analysis/build_solomon_rc_mean_table.py'),run_name='__main__')
print('COMPLETE',flush=True)
