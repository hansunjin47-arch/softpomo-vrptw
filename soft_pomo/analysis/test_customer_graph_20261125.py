"""CPU regression: baseline equivalence, public-state sensitivity, full-graph backward."""
import copy,json
from pathlib import Path
import torch
from experiments.concurrent_fleet_pair.customer_graph import CustomerGraphPolicy
from experiments.concurrent_fleet_pair.fleet_context import FleetContextPolicy as FleetPolicy
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_v3 import run as data

def main():
 torch.set_num_threads(1);torch.manual_seed(46)
 root=Path(__file__).resolve().parents[1]
 bundle=root/'server_export/rc_pure_lr_decay_capacity_v100_l10_s44_46_20261122'
 # In the frozen server bundle tests use its own root.
 if (root/'dataset/split.json').exists():bundle=root
 cfg_path=bundle/'config.json';data.SOURCE=bundle/'dataset'
 if not cfg_path.exists():
  cfg_path=root/'experiment_inputs/fleet_context/config_l10.json'
  data.SOURCE=root/'experiment_inputs/new_head/rc101/dataset'
 cfg=json.loads(cfg_path.read_text())
 row=json.loads((data.SOURCE/'split.json').read_text())['train'][0];inst,public=data.load(row)
 base=FleetPolicy(**cfg['model']).eval();model=CustomerGraphPolicy(**cfg['model']).eval()
 missing=model.load_state_dict(base.state_dict(),strict=False);assert all(k.startswith('graph_') for k in missing.missing_keys)
 env=TensorFleetEnv(inst,1,cfg['environment'],'cpu')
 with torch.no_grad():
  s,e=base.encode_events(inst,public);s2,e2=model.encode_events(inst,public)
  for customer in (1,2,3):env.step(torch.tensor([customer-1]),validate=True)
  f=env.frame();a=base.probabilities(f,s,e,env);b=model.probabilities(f,s2,e2,env)
  assert torch.allclose(a,b,atol=1e-7,rtol=1e-6),'Zero residual changed baseline'
  torch.nn.init.normal_(model.graph_out.weight,std=.1)
  before=model.probabilities(f,s2,e2,env)
  old_available=env.available.clone();old_arrival=env.arrival.clone()
  env.available.fill_(1e9);env.arrival.fill_(2e9)
  hidden=model.probabilities(f,s2,e2,env)
  assert torch.equal(before,hidden),'Hidden future timing leak'
  env.available.copy_(old_available);env.arrival.copy_(old_arrival)
  saved=env.load.clone();env.load[0,1]=.01
  changed=model.probabilities(f,s2,e2,env)
  sensitivity=float((changed-before).abs().max());assert sensitivity>1e-8,'Fleet state not connected'
  env.load.copy_(saved)
  # Hold queries/masks fixed and permute fleet tokens only.
  fields=[env.node,env.target,env.pending,env.status,env.load]
  copies=[x.clone() for x in fields]
  for x in fields:x[:,:3]=x[:,[2,0,1]].clone()
  permuted=model.probabilities(f,s2,e2,env)
  assert torch.allclose(before,permuted,atol=1e-7,rtol=1e-5)
  for x,y in zip(fields,copies):x.copy_(y)
 # Full trajectory graph: compare checkpoint recomputation with direct autograd.
 model.train();model.new_customer_epsilon=.05
 model.graph_chunk_size=2
 direct=copy.deepcopy(model);direct.checkpoint_scores=False;direct.graph_chunk_size=3
 def run(m):
  torch.manual_seed(127);ev=TensorFleetEnv(inst,4,cfg['environment'],'cpu');st,en=m.encode_events(inst,public,batch_size=2);logs=[]
  for _ in range(600):
   if bool(ev.done.all()):break
   frame=ev.frame();prob=m.probabilities(frame,st,en,ev)
   assert torch.isfinite(prob).all() and torch.allclose(prob.sum(1),torch.ones(4),atol=1e-5)
   action=prob.detach().multinomial(1).squeeze(1)
   logs.append(prob[ev.ids,action].clamp_min(1e-30).log());ev.step(action,validate=True)
  assert bool(ev.done.all())
  loss=-torch.stack(logs).sum(0).mean();loss.backward()
  assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
  assert m.graph_out.weight.grad.abs().sum()>0
  assert m.graph_edge_value.weight.grad.abs().sum()>0
  return float(loss.detach()),len(logs)
 l1,n1=run(model);l2,n2=run(direct)
 assert abs(l1-l2)<1e-4 and n1==n2
 for (name,p),(_,q) in zip(model.named_parameters(),direct.named_parameters()):
  if p.grad is not None:assert torch.allclose(p.grad,q.grad,atol=3e-4,rtol=3e-4),name
 report=dict(passed=True,baseline_zero_residual_equivalent=True,hidden_future_time_independent=True,fleet_input_probability_sensitivity=sensitivity,full_trajectory_backward=True,checkpoint_gradients_match_direct=True,steps=n1,device='cpu',rollouts=4)
 dest=root/'result/customer_graph_verification_20261125.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
