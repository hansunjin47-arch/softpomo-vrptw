"""Saved-prefix policy probes. CPU FP32; no new LLM calls or changed-prefix cache reuse."""
import json,math
import torch
from analysis.probe_rl_lkh_p001_beta_20261124 import R,B,read,OnlineEnv,parse_test_response,load,data
from experiments.concurrent_fleet_pair_kim.fleet_context import FleetContextKimPolicy
O=R/'result/fleetctx_relocation_20261125'
def main():
 torch.set_num_threads(1);data.SOURCE=B/'dataset';groups=read(B/'artifact/groups.json');mapping={r['name']:r for r in read(B/'dataset/split.json')['test']};run=R/'result/rc_kim_fleetctx_v100_l5_s46_20261124/run';cfg=read(run/'protocol.json')['configs']['5'];model=FleetContextKimPolicy(**cfg['model']).eval();model.load_state_dict(torch.load(run/'late5/checkpoint-1000.pt',map_location='cpu',weights_only=False)['model_state_dict']);out=[]
 for target in read(O/'first_divergence.json'):
  if 'ACC' in target['name'] and any(t>target['clock'] for t in target['public_disclosures']):continue
  inst,reports=load(mapping[target['name']]);folder=R/'result/fleetctx_l5_s46_test_20261125/seed46/instances'/target['name'];calls=read(folder/'calls.json');history=read(folder/'result.json')['history']
  def scores(c):return parse_test_response(c['response'],groups,json.loads(c['request']['messages'][1]['content'])['selectable_customers'])
  def refresh(env,disclosure):
   c=next(c for c in calls[1:] if abs(c['clock_min']-float(env.time[0]))<.001);env.update_event_confidence(scores(c),revealed_at_min=disclosure,effective_time_min=float(env.time[0]))
  env=OnlineEnv(inst,1,cfg['environment'],'cpu',groups=groups,scores=scores(calls[0]),beta=5.,reports=reports,on_disclosure=refresh)
  for h in history:
   a=torch.tensor([h['action_slot']]);_,v,d=env.decode_action(a)
   if abs(h['clock_min']-target['clock'])<.001 and int(v[0])==target['vehicle'] and int(d[0])==target['old_customer']:break
   env.step(a,validate=True)
  frame=env.frame();slot=int((frame['vehicles'][0]==target['vehicle']).nonzero()[0]);old=h['action_slot'];alt=env.n+slot*(env.n+1)+target['alternative_customer'];record=dict(**target,probes=[])
  with torch.inference_mode():
   starts,enc=model.encode_events(inst,reports)
   for beta in [0.,5.]:
    env.guidance.beta=beta;p=model.probabilities(frame,starts,enc,env)[0];chosen=int(p.argmax());_,_,dest=env.decode_action(torch.tensor([chosen]),frame)
    record['probes'].append(dict(beta=beta,chosen=int(dest[0]),matches_saved=chosen==old,old_probability=float(p[old]),alternative_probability=float(p[alt]),alternative_rank=1+int((p>p[alt]).sum()),old_vs_alternative_log_odds=math.log(max(float(p[old]),1e-38)/max(float(p[alt]),1e-38))))
  record['guide_log_odds_addition']=record['probes'][1]['old_vs_alternative_log_odds']-record['probes'][0]['old_vs_alternative_log_odds'];out.append(record)
  (O/'policy_probes.json').write_text(json.dumps(out,indent=2))
 print('cases',len(out),'FP32 matches',sum(r['probes'][1]['matches_saved'] for r in out),'guide favors old',sum(r['guide_log_odds_addition']>.001 for r in out),'guide favors alternative',sum(r['guide_log_odds_addition']<-.001 for r in out))
 for r in out:
  if r['name'] in ['RC101_ACC_P027','RC101_RAIN_P029']:print(json.dumps(r))
if __name__=='__main__':main()
