"""Single legal action intervention after identical prefix/disclosures; no later API."""
import json,torch
from analysis.probe_rl_lkh_p001_beta_20261124 import R,B,read,OnlineEnv,parse_test_response,load,data
from experiments.concurrent_fleet_pair_kim.fleet_context import FleetContextKimPolicy
O=R/'result/fleetctx_relocation_20261125'
def main():
 torch.set_num_threads(1);data.SOURCE=B/'dataset';groups=read(B/'artifact/groups.json');mapping={r['name']:r for r in read(B/'dataset/split.json')['test']};run=R/'result/rc_kim_fleetctx_v100_l5_s46_20261124/run';cfg=read(run/'protocol.json')['configs']['5'];model=FleetContextKimPolicy(**cfg['model']).eval();model.load_state_dict(torch.load(run/'late5/checkpoint-1000.pt',map_location='cpu',weights_only=False)['model_state_dict']);out=[]
 for target in read(O/'policy_probes.json'):
  if target['name'] not in ['RC101_ACC_P027','RC101_RAIN_P029']:continue
  inst,reports=load(mapping[target['name']]);folder=R/'result/fleetctx_l5_s46_test_20261125/seed46/instances'/target['name'];calls=read(folder/'calls.json');saved=read(folder/'result.json');history=saved['history']
  def scores(c):return parse_test_response(c['response'],groups,json.loads(c['request']['messages'][1]['content'])['selectable_customers'])
  for forced in [False,True]:
   changed=False
   def refresh(env,disclosure):
    assert not changed,'Cannot reuse LLM response after altered prefix'
    c=next(c for c in calls[1:] if abs(c['clock_min']-float(env.time[0]))<.001);env.update_event_confidence(scores(c),revealed_at_min=disclosure,effective_time_min=float(env.time[0]))
   env=OnlineEnv(inst,1,cfg['environment'],'cpu',groups=groups,scores=scores(calls[0]),beta=5.,reports=reports,on_disclosure=refresh)
   for h in history:
    a=torch.tensor([h['action_slot']]);_,v,d=env.decode_action(a)
    if abs(h['clock_min']-target['clock'])<.001 and int(v[0])==target['vehicle'] and int(d[0])==target['old_customer']:break
    env.step(a,validate=True)
   with torch.inference_mode():
    starts,enc=model.encode_events(inst,reports)
    if forced:
     frame=env.frame();slot=int((frame['vehicles'][0]==target['vehicle']).nonzero()[0]);changed=True;env.step(torch.tensor([env.n+slot*(env.n+1)+target['alternative_customer']]),validate=True)
    for _ in range(600):
     if bool(env.done.all()):break
     p=model.probabilities(env.frame(),starts,enc,env);env.step(p.argmax(1),validate=True)
    assert bool(env.done.all())
   metrics={k:float(v[0]) for k,v in env.metrics().items()}
   out.append(dict(name=target['name'],forced=forced,metrics=metrics,official_metrics=saved['metrics'],scope='CPU FP32 one-action posthoc intervention, unchanged cached disclosure prefix; not deployable action-selection rule'))
 (O/'one_action_continuations.json').write_text(json.dumps(out,indent=2))
 for r in out:print(r['name'],r['forced'],{k:round(r['metrics'][k],4) for k in ['reward','vehicles','travel_time','total_late','unserved']},'baseline reward',r['official_metrics']['reward'])
if __name__=='__main__':main()
