"""Recover an archived tool-output table, without rerunning or inventing results."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/Users/hansu/.claude/projects/C--Users-hansu/64ac47f7-b503-4fe8-aa63-b19c1d93af0e.jsonl')
TOOL_ID = 'toolu_01En6REGsqAYZkekkeiYuBzu'
OUT = ROOT / 'analysis/rc_full_confidence_train_eval_20260920_215555'
matches = []
for line in SOURCE.read_text(encoding='utf-8').splitlines():
    obj = json.loads(line)
    blocks = obj.get('message', {}).get('content', [])
    if isinstance(blocks, list):
        for block in blocks:
            if block.get('type') == 'tool_result' and block.get('tool_use_id') == TOOL_ID:
                matches.append((obj['timestamp'], block['content']))
assert len(matches) == 1
timestamp, table = matches[0]
table = table.replace('\r', '')
metric = r'K=(\d+) D=([\d.]+) Lc=(\d+) Lt=([\d.]+)'
pattern = re.compile(r'\| (\w+) \| (base|acc2|rain2) \| ' + metric + r' \| ' + metric + r' \|')
rows = []
for m in pattern.finditer(table):
    name, scenario, *values = m.groups()
    for plan, v in [('M_base', values[:4]), ('M_true_hist', values[4:])]:
        rows.append(dict(name=name.upper(), scenario=scenario.upper(), model=plan,
                         vehicles=int(v[0]), distance=float(v[1]), late_stops=int(v[2]), total_late=float(v[3])))
assert len(rows) == 29 * 3 * 2
rc = [r for r in rows if r['name'] in {f'RC{i}' for i in range(102, 109)}]
assert len(rc) == 42
sources = {'Pure1000': 'rc_pure1000_train_eval_20260920_195014',
           'Full confidence': 'rc_full_confidence_train_eval_20260920_215555',
           'Selective confidence': 'rc_selective_train_eval_20260920_194238'}
models = {}
for label, directory in sources.items():
    models[label] = json.loads((ROOT / 'analysis' / directory / 'results.json').read_text())['results']
models.update({plan: [r for r in rc if r['model'] == plan] for plan in ['M_base', 'M_true_hist']})
summary = []
for scenario in ['BASE', 'RAIN2', 'ACC2']:
    for label, data in models.items():
        subset = [r for r in data if r['scenario'].lstrip('_') == scenario]
        assert len(subset) == 7
        summary.append(dict(scenario=scenario, model=label, **{
            key: sum(r[key] for r in subset) for key in ['vehicles', 'distance', 'late_stops', 'total_late']}))
payload = dict(source=str(SOURCE), source_tool_id=TOOL_ID, source_timestamp=timestamp,
               recovered_table_sha256=hashlib.sha256(table.encode()).hexdigest(),
               precision='Archived D and Lt rounded to one decimal per instance.',
               rows=rows, rc_summary=summary)
(OUT / 'lkh_v2_recovered.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
(OUT / 'lkh_v2_recovered_table.md').write_text(table, encoding='utf-8')
print(json.dumps(summary, indent=2))
