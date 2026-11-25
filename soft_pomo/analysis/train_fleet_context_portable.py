"""Fresh RC Pure RL fleet-context experiment; train1000 then automatic Test80."""
import argparse,gc,hashlib,importlib.util,json,math,statistics
from pathlib import Path
import torch
from experiments.concurrent_fleet_pair.fleet_context import FleetContextPolicy
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_pair.io import write
from experiments.concurrent_fleet_v3 import run as data
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def evaluate(out,late,seed,source):
 data.SOURCE=source;run=out/'run';cfg=read(run/'protocol.json')['configs'][str(late)]
 ck=run/f'late{late}/checkpoint-1000.pt';dest=out/'test80'
 protocol=dict(seed=seed,late=late,checkpoint_sha256=sha(ck),split_sha256=sha(source/'split.json'),decode='single greedy CUDA AMP, no augmentation or LLM',model=FleetContextPolicy.__name__,epoch=1000)
 if (dest/'protocol.json').exists():assert read(dest/'protocol.json')==protocol
 else:write(dest/'protocol.json',protocol)
 saved=torch.load(ck,map_location='cpu',weights_only=False)
 model=FleetContextPolicy(**cfg['model']).cuda().eval();model.load_state_dict(saved['model_state_dict'],strict=True);del saved
 results=[]
 with torch.inference_mode():
  for row in read(source/'split.json')['test']:
   path=dest/'instances'/(row['name']+'.json')
   if path.exists():record=read(path)
   else:
    inst,reports=data.load(row);env=TensorFleetEnv(inst,1,cfg['environment'],'cuda');starts,enc=model.encode_events(inst,reports);history=[]
    for _ in range(600):
     if bool(env.done.all()):break
     with torch.autocast('cuda'):action=model.probabilities(env.frame(),starts,enc,env).argmax(1)
     history.append(dict(clock_min=float(env.time[0]),action_slot=int(action[0])))
     env.step(action,validate=True)
    assert bool(env.done.all()),row['name']
    record=dict(name=row['name'],**{k:float(v[0]) for k,v in env.metrics().items()},history=history)
    write(path,record)
   results.append({k:v for k,v in record.items() if k!='history'})
   write(dest/'results.json',results);print('TEST',late,len(results),record['reward'],flush=True)
 assert len(results)==80
 write(dest/'summary.json',{k:dict(mean=statistics.mean(r[k] for r in results),ci95=1.96*statistics.stdev(r[k] for r in results)/math.sqrt(80)) for k in ('reward','vehicles','travel_time','total_late','unserved')})
 write(dest/'complete.json',dict(test_count=80,epoch=1000))

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=('fleet','customer_graph'),default='fleet');p.add_argument('--late',type=int,choices=(5,10),required=True);p.add_argument('--seed',type=int,default=46,choices=(46,));p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
 global FleetContextPolicy
 if a.model=='customer_graph':
  from experiments.concurrent_fleet_pair.customer_graph import CustomerGraphPolicy
  FleetContextPolicy=CustomerGraphPolicy
 a.out=a.out.resolve();source=ROOT/'experiment_inputs/new_head/rc101/dataset';config=ROOT/f'experiment_inputs/fleet_context/config_l{a.late}.json';cfg=read(config);split=read(source/'split.json')
 assert len(split['train'])==20 and len(split['test'])==80 and cfg['trainer']['epochs']==1000
 ep=cfg['environment'];assert ep['late_penalty']==a.late and ep['require_dispatch_capacity'] and ep['protect_last_service_vehicle'] and ep['normalize_advantage_std']
 assert abs(ep['vehicle_penalty']-10000/1696.95)<1e-8
 for row in split['train']+split['test']:assert sha(source/'data'/(row['name']+'.txt'))==row['sha256']
 trainer_file=ROOT/'analysis/fleet_context_trainer.py'
 spec=importlib.util.spec_from_file_location('fleet_context_trainer',trainer_file);trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
 trainer.FleetPolicy=FleetContextPolicy
 module=ROOT/'experiments/concurrent_fleet_pair/fleet_context.py'
 protocol=dict(method='Pure RL + '+a.model,seed=a.seed,late=a.late,vehicle_weight=100,travel_weight=1,epochs=1000,
  initialization='fresh; fleet residual zero initialized; customer graph residual also zero initialized when selected',learning_rate='1e-4 through500, linear to1e-5 at1000',
  observation='Attention over active fleet last-observed node, committed destination, prebooked destination, status and remaining capacity; current clock in query. No arrival/available/true event read.',
  scope='Fleet includes current vehicle; unchanged environment selects routing vehicle. No new action, LLM, reward, mask or epsilon schedule.',
  graph='One vehicle-to-customer message layer into dynamic customer KV, nominal travel/capacity/commitment edges' if a.model=='customer_graph' else 'No customer graph layer',
  evaluation='automatic final Test80 single greedy CUDA AMP; mean +/-1.96 sampleSD/sqrt80',
  hashes={str(f.relative_to(ROOT)):sha(f) for f in (Path(__file__),module,ROOT/'experiments/concurrent_fleet_pair/customer_graph.py',trainer_file,config,source/'split.json')})
 if a.validate_only:print(json.dumps(protocol));return
 if a.resume:assert read(a.out/'experiment.json')==protocol
 else:
  assert not a.out.exists(),'Refusing duplicate output'
  write(a.out/'experiment.json',protocol)
 calls=0
 if a.resume:
  saved=torch.load(a.out/f'run/late{a.late}/checkpoint-last.pt',map_location='cpu',weights_only=False);calls=saved['state']['update'];del saved
 original=trainer.train_batch
 def batch(model,opt,*args,**kw):
  nonlocal calls
  epoch=calls//20+1;factor=1. if epoch<=500 else max(.1,1.-.9*(epoch-500)/500.)
  for pg in opt.param_groups:pg['lr']=1e-4*factor
  summary,detail=original(model,opt,*args,**kw);summary['learning_rate']=opt.param_groups[0]['lr'];calls+=1
  return summary,detail
 trainer.train_batch=batch
 complete=a.out/f'run/late{a.late}/training_complete.json'
 if not complete.exists():
  trainer.run(argparse.Namespace(out=a.out/'run',seed=a.seed,vehicle_cost=100.,resume=a.resume,smoke_updates=0,config=config,source=source,weights=[a.late],reward_denominator=1696.95,return_mask=True,no_advantage_std=False,new_customer_epsilon=.05,epsilon_final=.01,epsilon_zero_epoch=1000,resume_from=None))
 gc.collect();torch.cuda.empty_cache();evaluate(a.out,a.late,a.seed,source)
if __name__=='__main__':main()
