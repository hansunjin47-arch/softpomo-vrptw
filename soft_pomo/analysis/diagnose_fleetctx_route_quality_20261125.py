"""Posthoc route-quality diagnostic. Oracle, not an online policy evaluation."""
import json,time,statistics as st
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import torch,numpy as np
from analysis.train20_fixed_assignment_2opt_20261124 import evaluate
from analysis.diagnose_capacity_lateness_20261121 import Replay
from experiments.concurrent_fleet_bias_guided.support import load
from experiments.concurrent_fleet_v3 import run as data
R=Path(__file__).resolve().parents[1];O=R/'result/fleetctx_route_quality_20261125';read=lambda p:json.loads(p.read_text())
def score(route,e):
 x=evaluate(route,e);x['objective']=x['travel_time']+5*x['total_late'];return x

def main():
 torch.set_num_threads(1);O.mkdir(exist_ok=True);data.SOURCE=R/'server_export/rc_kim_rl_lkh_cap_decay_v100_l5_s44_46_20261122/dataset'
 cfg=read(R/'result/rc_kim_fleetctx_v100_l5_s46_20261124/run/protocol.json')['configs']['5']['environment'];mapping={r['name']:r for r in read(data.SOURCE/'split.json')['test']}
 records=[]
 for rec in read(R/'result/fleetctx_diagnosis_20261125/replay.json'):
  if rec['model']!='fleet':continue
  inst,_=load(mapping[rec['name']]);env=Replay(inst,1,cfg,'cpu');e=SimpleNamespace(**{k:getattr(env,k).numpy() for k in ['base','close','open','service']},true_events=[(float(a),float(b),float(c),d.numpy()) for a,b,c,d in env.true_events])
  routes=defaultdict(list)
  for leg in rec['legs']:
   if leg['customer']:routes[leg['vehicle']].append(leg['customer'])
  routes=list(routes.values());base=[score(r,e) for r in routes];chosen=[]
  expected=read(R/'result/fleetctx_l5_s46_test_20261125/seed46/instances'/rec['name']/'result.json')['metrics']
  for k in ['travel_time','total_late']:assert abs(sum(x[k] for x in base)-expected[k])<.03
  for route,b in zip(routes,base):
   candidates=[b]
   for i in range(1,len(route)-1):
    for j in range(i+1,len(route)):
     alt=score(route[:i]+route[i:j+1][::-1]+route[j+1:],e)
     if alt['travel_time']<=b['travel_time']+1e-5 and alt['total_late']<=b['total_late']+1e-5:candidates.append(alt)
   chosen.append(min(candidates,key=lambda x:x['objective']))
  def agg(xs):return {k:sum(x[k] for x in xs) for k in ['travel_time','total_late','objective']}
  records.append(dict(name=rec['name'],vehicles=len(routes),baseline=agg(base),order_pareto=agg(chosen),routes_before=routes,routes_after=[x['route'] for x in chosen]))
  (O/'records.json').write_text(json.dumps(records,indent=2))
  print(rec['name'],len(routes),round(records[-1]['order_pareto']['total_late']-expected['total_late'],2),flush=True)
 summary={'scope':'Offline full-scenario oracle, not causal Test. K, assignment and first customers fixed. One reversal per route; both travel and lateness cannot worsen. Includes original route.', 'groups':{}}
 for k in [15,16,17]:
  rr=[r for r in records if r['vehicles']==k];summary['groups'][k]=dict(n=len(rr),improved=sum(r['order_pareto']['objective']<r['baseline']['objective']-.001 for r in rr),baseline={m:st.mean(r['baseline'][m] for r in rr) for m in ['travel_time','total_late']},after={m:st.mean(r['order_pareto'][m] for r in rr) for m in ['travel_time','total_late']})
 (O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
