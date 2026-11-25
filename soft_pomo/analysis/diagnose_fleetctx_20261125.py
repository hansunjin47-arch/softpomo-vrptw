"""Saved-action paired fleet-context diagnosis, CPU only, no API."""
import json,statistics as st
from collections import Counter,defaultdict
from pathlib import Path
import torch
from analysis.diagnose_capacity_lateness_20261121 import Replay
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data
R=Path(__file__).resolve().parents[1];O=R/'result/fleetctx_diagnosis_20261125';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
def main():
 torch.set_num_threads(1)
 data.SOURCE=R/'server_export/rc_kim_rl_lkh_cap_decay_v100_l5_s44_46_20261122/dataset'
 cfg=read(R/'result/rc_kim_fleetctx_v100_l5_s46_20261124/run/protocol.json')['configs']['5']['environment']
 roots={'old':R/'result/rl_lkh_l5_s46_test_20261123/seed46','fleet':R/'result/fleetctx_l5_s46_test_20261125/seed46'}
 results={m:{r['name']:r for r in read(p/'results.json')} for m,p in roots.items()};details=[];paired=[]
 for row in read(data.SOURCE/'split.json')['test']:
  name=row['name'];inst,_=load(row);pair={'name':name}
  for m,p in roots.items():
   saved=read(p/'instances'/name/'result.json');env=Replay(inst,1,cfg,'cpu')
   for a in saved['history']:
    assert abs(float(env.time[0])-a['clock_min'])<.01
    env.step(torch.tensor([a['action_slot']]),validate=True)
   assert bool(env.done.all())
   assert abs(sum(l['late'] for l in env.legs)-saved['metrics']['total_late'])<.03
   routes=defaultdict(list)
   for l in env.legs:routes[l['vehicle']].append(l)
   heads=[ls[0]['customer'] for ls in routes.values()]
   pair[m]=dict(metrics=saved['metrics'],heads=heads)
   details.append(dict(model=m,name=name,legs=env.legs,events=inst.get('preset_events')))
  pair['missing_heads']=sorted(set(pair['old']['heads'])-set(pair['fleet']['heads']))
  pair['added_heads']=sorted(set(pair['fleet']['heads'])-set(pair['old']['heads']))
  paired.append(pair)
 summary={}
 for m in roots:
  rr=list(results[m].values());legs=[l for d in details if d['model']==m for l in d['legs']];cs=Counter()
  for l in legs:cs[l['customer']]+=l['late']
  total=sum(l['late'] for l in legs)
  summary[m]=dict(median_late=st.median(r['total_late'] for r in rr),top5_late_share=sum(sorted([r['total_late'] for r in rr],reverse=True)[:5])/total,vehicles=dict(Counter(r['vehicles'] for r in rr)),first_late=sum(l['late'] for l in legs if l['position']==1)/80,depot_late=sum(l['late'] for l in legs if l['customer']==0)/80,late_without_current_leg_delay=sum(l['late_even_without_current_event'] for l in legs)/80,top_customers=cs.most_common(10))
 summary['paired']=dict(reward_wins=sum(p['fleet']['metrics']['reward']>p['old']['metrics']['reward'] for p in paired),late_worse=sum(p['fleet']['metrics']['total_late']>p['old']['metrics']['total_late']+.001 for p in paired),missing_heads=Counter(n for p in paired for n in p['missing_heads']),added_heads=Counter(n for p in paired for n in p['added_heads']))
 (O/'replay.json').write_text(json.dumps(details,indent=2));(O/'paired.json').write_text(json.dumps(paired,indent=2));(O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
