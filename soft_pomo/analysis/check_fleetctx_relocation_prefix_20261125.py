"""Check earliest divergence legality only; does not certify causal route selection."""
import json
from collections import Counter
import torch
from analysis.diagnose_fleetctx_route_quality_20261125 import R,read,Replay,load,data
O=R/'result/fleetctx_relocation_20261125'
def main():
 torch.set_num_threads(1);data.SOURCE=R/'server_export/rc_kim_rl_lkh_cap_decay_v100_l5_s44_46_20261122/dataset';mapping={r['name']:r for r in read(data.SOURCE/'split.json')['test']};cfg=read(R/'result/rc_kim_fleetctx_v100_l5_s46_20261124/run/protocol.json')['configs']['5']['environment'];out=[]
 for r in read(O/'records.json'):
  b=r['best']
  if not b:continue
  inst,public=load(mapping[r['name']]);env=Replay(inst,1,cfg,'cpu');changed={b['source']:b['source_after']+[0],b['target']:b['target_after']+[0]};counts=Counter()
  for h in read(R/'result/fleetctx_l5_s46_test_20261125/seed46/instances'/r['name']/'result.json')['history']:
   frame=env.frame();a=torch.tensor([h['action_slot']]);new,v,d=env.decode_action(a,frame);v=int(v[0]);d=int(d[0])
   if v in changed:
    wanted=changed[v][counts[v]]
    if wanted!=d:
     slot=int((frame['vehicles'][0]==v).nonzero()[0]);idx=env.n+slot*(env.n+1)+wanted
     out.append(dict(name=r['name'],clock=float(env.time[0]),vehicle=v,old_customer=d,alternative_customer=wanted,alternative_legal=bool(frame['mask'][0,idx]),customer_state=int(env.customer[0,wanted]),public_disclosures=[p['revealed_at_min'] for p in public],scope='Earliest action change legality only; chosen move still uses hindsight'))
     break
   counts[v]+=1;env.step(a,validate=True)
 (O/'first_divergence.json').write_text(json.dumps(out,indent=2));print('n',len(out),'legal',sum(r['alternative_legal'] for r in out))
 for r in out:
  if r['name'] in ['RC101_ACC_P027','RC101_ACC_P012','RC101_RAIN_P029']:print(r)
if __name__=='__main__':main()
