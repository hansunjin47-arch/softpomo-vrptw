"""Fixed-K single customer relocation oracle; preserves first customers and capacities."""
import json,statistics as st
from types import SimpleNamespace
import torch
from analysis.diagnose_fleetctx_route_quality_20261125 import R,read,score,Replay,load,data
O=R/'result/fleetctx_relocation_20261125'
def main():
 torch.set_num_threads(1);O.mkdir(exist_ok=True);data.SOURCE=R/'server_export/rc_kim_rl_lkh_cap_decay_v100_l5_s44_46_20261122/dataset'
 mapping={r['name']:r for r in read(data.SOURCE/'split.json')['test']};cfg=read(R/'result/rc_kim_fleetctx_v100_l5_s46_20261124/run/protocol.json')['configs']['5']['environment'];out=[]
 for rec in read(R/'result/fleetctx_route_quality_20261125/records.json'):
  inst,_=load(mapping[rec['name']]);env=Replay(inst,1,cfg,'cpu');e=SimpleNamespace(**{k:getattr(env,k).numpy() for k in ['base','close','open','service']},true_events=[(float(a),float(b),float(c),d.numpy()) for a,b,c,d in env.true_events]);demand=env.demand.numpy()
  routes=rec['routes_before'];loads=[sum(float(demand[n]) for n in r) for r in routes];cache={}
  def cost(r):
   key=tuple(r)
   if key not in cache:cache[key]=score(r,e)
   return cache[key]
  base=[cost(r) for r in routes];best_delta=0.;best=None;candidates=0
  for i,route in enumerate(routes):
   for pos in range(1,len(route)):
    n=route[pos];src=route[:pos]+route[pos+1:];sa=cost(src)
    for j,dst in enumerate(routes):
     if i==j or loads[j]+demand[n]>1+1e-5:continue
     before={k:base[i][k]+base[j][k] for k in ['travel_time','total_late','objective']}
     for q in range(1,len(dst)+1):
      dr=dst[:q]+[n]+dst[q:];da=cost(dr);candidates+=1
      delta={k:sa[k]+da[k]-before[k] for k in before}
      if delta['travel_time']>1e-5 or delta['total_late']>1e-5:continue
      if delta['objective']<best_delta-1e-5:
       best_delta=delta['objective'];best=dict(customer=n,source=i,target=j,source_before=route,source_after=src,target_before=dst,target_after=dr,delta=delta,source_legs_before=base[i]['legs'],source_legs_after=sa['legs'],target_legs_before=base[j]['legs'],target_legs_after=da['legs'])
  after=dict(rec['baseline'])
  if best:
   for k,v in best['delta'].items():after[k]+=v
  out.append(dict(name=rec['name'],vehicles=len(routes),baseline=rec['baseline'],after=after,best=best,candidates=candidates))
  (O/'records.json').write_text(json.dumps(out,indent=2));print(len(out),rec['name'],round(best_delta,2),flush=True)
 summary={'scope':'Full-event hindsight oracle; not deployable online evaluation. One relocation from ORIGINAL routes, first customers/K fixed, capacity checked, travel and lateness jointly non-increasing. Separate arm from order reversals.','groups':{}}
 for k in [15,16,17]:
  rr=[r for r in out if r['vehicles']==k];summary['groups'][k]=dict(n=len(rr),improved=sum(r['best'] is not None for r in rr),baseline={m:st.mean(r['baseline'][m] for r in rr) for m in ['travel_time','total_late']},after={m:st.mean(r['after'][m] for r in rr) for m in ['travel_time','total_late']})
 (O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
