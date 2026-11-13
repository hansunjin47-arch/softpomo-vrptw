"""Report event-aware travel time for saved fixed-generator Pure RL routes."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_env import load_solomon, _solution_stats

source = ROOT/'analysis/fixed_500_1000_eval_20260921_103246/results.json'
rows = []
for row in json.loads(source.read_text())['results']:
    path = ROOT.parent/'data/Solomon'/(row['name'].lower()+'.txt')
    inst = load_solomon(str(path))
    stats = _solution_stats(inst, row['routes'])
    rows.append(dict(name=row['name'], epoch=row['epoch'], vehicles=len(row['routes']),
        travel_time=float(stats['total_travel_time']), distance=float(stats['total_dist']),
        late_stops=int(stats['late_count']), total_late=float(stats['total_late']),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint=row['checkpoint']))
out = source.parent
(out/'travel_time.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
lines = ['# Pure RL: Event-Aware Travel Time', '',
    'Saved routes replayed; no training or new policy evaluation. Travel time includes return to depot, excludes waiting and service.',
    'Historical LKH values are not revalidated here; original routes are required for identical replay.', '',
    '| Instance | Epoch | Vehicles | Travel time | Late stops | Total late |',
    '|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['name']} | {r['epoch']} | {r['vehicles']} | {r['travel_time']:.2f} | {r['late_stops']} | {r['total_late']:.2f} |")
(out/'travel_time.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines))
