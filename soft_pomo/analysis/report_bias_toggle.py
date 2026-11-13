"""Recompute geometric distance from saved routes and write paired results."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vrptw_env import load_solomon,_solution_stats

p=Path(sys.argv[1])
d=json.loads((p/'results.json').read_text())
lines=['# Final Checkpoint Bias ON/OFF Diagnostic','',
       'Same epoch1000 weights, seed42, random starts covering100 customers, POMO100, aug8. LLM remains enabled for event handling; only additive beta changes5 ->0. No training or API calls. Original checkpoints and caches verified unchanged.',
       '','Historical start mode was not stored in the checkpoint. These are newly paired evaluations, NOT a reproduction of the original final-evaluation log; ON results differ from historical saved best routes. Do not combine historical ON with current OFF.',
       '','| Instance | Beta | Vehicles | Distance | Late stops | Late amount | Reward |','|---|---:|---:|---:|---:|---:|---:|']
for r in d['results']:
    inst=load_solomon(str(ROOT.parent/'data/Solomon'/(r['name'].lower()+'.txt')))
    st=_solution_stats(inst,r['routes'])
    r['distance']=sum(float(v['rdist']) for v in st['per_vehicle'])
    lines.append(f"| {r['name']} | {r['beta']:.0f} | {r['vehicles']} | {r['distance']:.2f} | {r['late_stops']} | {r['total_late']:.2f} | {r['reward']:.4f} |")
lines+=['','OFF is not Pure RL: these weights were trained with guidance. Better OFF performance identifies a local inference-time cost under this protocol, not a benefit of guided training. Worse OFF may reflect useful guidance or reliance on it. Use validation instances for selecting modes/beta; no best-per-test-scenario selection.']
(p/'results.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
(p/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
