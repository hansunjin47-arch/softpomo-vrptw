"""Replay the automatically saved final routes and compare with epoch 500."""
import json
import argparse
from pathlib import Path
import re
import sys
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, default=ROOT / 'result/rc101_pilot_discrete_pure1000_20260923_115506')
run = parser.parse_args().run.resolve()
split = json.loads((run / 'provenance/split.json').read_text())
rows = []
for entry in split['test']:
    name = entry['name']
    path = run / 'model/RC101/plots' / (name + '_solution.txt')
    content = path.read_text(encoding='utf-8')
    routes = [[int(n.strip()) for n in route.split('->')]
              for route in re.findall(r'^\s+V\d+:\s*(.*?)\s*\[dist=', content, re.M)]
    assert sorted(n for r in routes for n in r if n) == list(range(1, 101))
    stats = _solution_stats(load_solomon(str(Path(split['data']) / (name + '.txt'))), routes)
    rows.append(dict(name=name, kind=entry['kind'], vehicles=len(routes),
        travel_time=stats['total_travel_time'], late_count=stats['late_count'],
        total_late=stats['total_late'], served=stats['n_served'], routes=routes))
summary = {}
for group in ['RAIN', 'ACCIDENT', 'ALL']:
    rs = [r for r in rows if group == 'ALL' or r['kind'] == group]
    summary[group] = dict(count=len(rs), **{k: statistics.mean(r[k] for r in rs)
        for k in ['vehicles', 'travel_time', 'late_count', 'total_late', 'served']},
        zero_late=sum(r['late_count'] == 0 for r in rs))
out = run / 'eval_1000_saved_routes'
out.mkdir(exist_ok=True)
(out / 'results.json').write_text(json.dumps(rows, indent=2))
(out / 'summary.json').write_text(json.dumps(dict(source='Automatically saved final routes; no new policy inference',
    late_count_includes_depot=True, summary=summary), indent=2))
print(json.dumps(summary, indent=2))
if not (run / 'eval_500/summary.json').exists():
    sys.exit(0)
old = json.loads((run / 'eval_500/summary.json').read_text())['summary']['ALL']
print('Change vs epoch 500:')
for k in ['vehicles', 'travel_time', 'late_count', 'total_late']:
    print(k, summary['ALL'][k] - old[k], 'percent', (summary['ALL'][k] / old[k] - 1) * 100)
