"""Paired frozen training-set metrics; no evaluation or generation calls."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
pure_path=Path(sys.argv[1])
soft_path=ROOT/'analysis/rc_selective_train_eval_20260920_194238/results.json'
pure=json.loads(pure_path.read_text())
soft=json.loads(soft_path.read_text())
assert len(pure['results'])==len(soft['results'])==21
for key in ['seed','pomo','aug']:
    assert pure[key]==soft[key]
p={r['name']:r for r in pure['results']}
s={r['name']:r for r in soft['results']}
assert p.keys()==s.keys()
lines=['# RC Training Set: Pure1000 vs Selective Phase2','',
       'Same evaluator, POMO100, aug8, seed42 per instance. Pure checkpoint with bias0 (asserted every step); selective checkpoint with bias5. Event handling kept active for both. Late counts include depot returns. All customers served.', '',
       '| Instance | Vehicles P/S | Distance P/S | Late stops P/S | Late amount P/S | Reward P/S |',
       '|---|---:|---:|---:|---:|---:|']
for name,a in p.items():
    b=s[name]
    lines.append(f"| {name} | {a['vehicles']}/{b['vehicles']} | {a['distance']:.2f}/{b['distance']:.2f} | {a['late_stops']}/{b['late_stops']} | {a['total_late']:.2f}/{b['total_late']:.2f} | {a['reward']:.4f}/{b['reward']:.4f} |")
summary=[]
for scenario in ['BASE','_RAIN2','_ACC2']:
    ns=[n for n in p if p[n]['scenario']==scenario]
    row={'scenario':scenario}
    for label,data in [('pure',p),('selective',s)]:
        row[label]={k:sum(data[n][k] for n in ns) for k in ['vehicles','distance','late_stops','total_late','customer_late','depot_late']}
        row[label]['reward_mean']=sum(data[n]['reward'] for n in ns)/len(ns)
    summary.append(row)
wins={k:sum(s[n][k]<p[n][k] for n in p) for k in ['vehicles','distance','late_stops','total_late']}
wins['reward']=sum(s[n]['reward']>p[n]['reward'] for n in p)
lines+=['','## Scenario Totals','', '```json',json.dumps(summary,indent=2),'```', '',f'Selective strictly better counts: {wins}', '',f'Pure source: {pure_path}', f'Selective source: {soft_path}', '',
        'Interpretation: training-set fit is descriptive, not held-out generalization or proof of a causal LLM gain. Fixed cache familiarity, additional policy learning and immediate bias effects are combined.']
out=ROOT/'analysis/rc_train_pure_vs_selective.md'
out.write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(summary=summary,strict_wins=wins,report=str(out)),indent=2))
