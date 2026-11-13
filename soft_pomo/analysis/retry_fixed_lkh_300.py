"""Wait for both 60s audits, then retry only missing-route failures at 300s."""
import datetime
import json
from pathlib import Path
import sys
import time

import eval_fixed_v2_lkh as event
import eval_fixed_v2_lkh_history as history

ROOT = event.ROOT
OUT = Path(sys.argv[1])
SOURCES = [('event', ROOT/'analysis/fixed_v2_lkh_20260920_234559'),
           ('history', ROOT/'analysis/fixed_v2_lkh_history_20260921_001233')]
status = dict(state='waiting_for_60s_runs', time_limit=300, results=[])
event.write(OUT/'manifest.json', status)
deadline = time.monotonic() + 86400
while True:
    manifests = [json.loads((p/'manifest.json').read_text()) for _,p in SOURCES]
    if all(m.get('finished') for m in manifests):
        break
    if time.monotonic() > deadline:
        raise RuntimeError('60s source runs did not finish within one day')
    time.sleep(30)

# Keep the original caller's objective, seed and trials; only change time budget.
original_solve = event.lkh._solve_once
def extended(inst, matrix, seconds, *args, **kwargs):
    return original_solve(inst, matrix, 300, *args, **kwargs)
event.lkh._solve_once = extended
status['state'] = 'running'
status['sources'] = [str(p) for _,p in SOURCES]
status['started'] = datetime.datetime.now().isoformat()
status['expected'] = sum(sum(r['status']=='failed' and 'LKH returned no route' in (r.get('error') or '') for r in m['results']) for m in manifests)
event.write(OUT/'manifest.json', status)
for (kind, source), manifest in zip(SOURCES, manifests):
    target = OUT/kind
    target.mkdir(exist_ok=False)
    for row in manifest['results']:
        if row['status'] != 'failed' or 'LKH returned no route' not in (row.get('error') or ''):
            continue
        family, name = row['family'], row['name']
        path = event.DATA[family]/(name+'.txt')
        prior = json.loads((source/(family+'_'+name+'.json')).read_text())
        assert event.sha(path) == prior['input_hash']
        if kind == 'event':
            result = event.solve(family, path, target)
            saved = json.loads((target/(family+'_'+name+'.json')).read_text())
            assert saved['planning_matrix'] == prior['planning_matrix']
        else:
            matrix = json.loads((source/(family+'_history_matrix.json')).read_text())
            for filename, digest in matrix['sources'].items():
                assert event.sha(event.DATA[family]/filename) == digest
            event.write(target/(family+'_history_matrix.json'), matrix)
            result = history.solve_family_instance(family, path, matrix['matrix'], target)
        status['results'].append(dict(kind=kind, source_result=str(source/(family+'_'+name+'.json')), result=result))
        event.write(OUT/'manifest.json', status)
status.update(state='finished', finished=datetime.datetime.now().isoformat())
event.write(OUT/'manifest.json', status)
