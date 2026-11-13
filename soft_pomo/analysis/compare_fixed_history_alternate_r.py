"""Fixed-gen Pure1000 versus fixed-gen full history on alternate Solomon R."""
import copy
import argparse
import datetime
import json
import numpy as np
import torch
from eval_fixed_v2_lkh import ROOT, lkh, load_solomon, _solution_stats, sha, write
import train_vrptw as train

parser=argparse.ArgumentParser()
parser.add_argument('--time-limit',type=int,default=180)
parser.add_argument('--family',choices=['r','rc'],default='r')
args=parser.parse_args()
assert args.time_limit>0

out=ROOT/'analysis'/('fixed_history_vs_pure_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir()
print('OUTPUT',out,flush=True)
history_source=ROOT/f'analysis/history_101_current_20260921_144216/{args.family.upper()}101_history_matrix.json'
history=json.loads(history_source.read_text())
matrix=history['matrix']
write(out/'history_matrix.json',history)
report=dict(history_source=str(history_source),history_hash=sha(history_source),attempts=[],results=[],
            r_time_limit=args.time_limit,rc_time_limit=60,
            selection='First route-returning instance in R102..R112; R101 failed previously. Not an unbiased family average.')
# Reuse matched RC101 evaluations; no extra model evaluation or solver call.
rc=json.loads((history_source.parent/'RC101.json').read_text())
report['results'] += [dict(r,method='full_history') for r in rc['scenarios']]
pure=json.loads((ROOT/'analysis/fixed_500_1000_eval_20260921_103246/travel_time.json').read_text())
report['results'] += [dict(r,method='pure1000') for r in pure if r['epoch']==1000 and r['name'].startswith('RC101')]
if args.family=='rc':
    report['results']=[]
    report['rc_time_limit']=args.time_limit
    report['selection']='All RC101..RC108, including failed solver attempts.'
for i in (range(101,109) if args.family=='rc' else range(102,113)):
    name=f'{args.family.upper()}{i}'
    path=ROOT.parent/'data/Solomon'/f'{name.lower()}.txt'
    raw=lkh.load_solomon(str(path))
    # Shared coordinates are required when reusing a learned travel-time matrix.
    ref=lkh.load_solomon(str(path.with_name(args.family+'101.txt')))
    assert np.array_equal(raw['coords'],ref['coords'])
    print('SOLVE',name,flush=True)
    solved=lkh._solve_once(raw,matrix,args.time_limit,None,None,1,10000)
    write(out/f'{name}_solver.json',solved)
    routes=[[int(n) for n in r if n] for r in solved.get('routes',[])]
    routes=[r for r in routes if r]
    report['attempts'].append(dict(name=name,status=solved['status']))
    write(out/'results.json',report)
    if not routes:
        continue
    assert sorted(n for r in routes for n in r)==list(range(1,101))
    assert len(routes)<=raw['vehicle_limit']
    assert all(sum(raw['demands'][n] for n in r)<=raw['vehicle_capacity'] for r in routes)
    checkpoint=ROOT/'result/config_E_r_fixed_v2_pure_b64_aug8_20260920_230825/RH01-RH15/checkpoint-1000.pt'
    if args.family=='rc':
        checkpoint=ROOT/'result/config_E_rc_fixed_v2_pure_b64_aug8_20260920_230417/RCH01-RCH15/checkpoint-1000.pt'
    tp=copy.deepcopy(train.trainer_params)
    tp.update(use_cuda=False,train_instances=[name.lower()],test_instances=[],result_dir=str(out/'pure_eval'),
              model_load=dict(enable=False),use_augmentation=True,n_mc_samples=1,eval_top_k=None,pomo_auto=False)
    torch.set_num_threads(4)
    train._set_seed(42)
    reward_config=train._REWARD_CONFIGS_RC1 if args.family=='rc' else train._REWARD_CONFIGS_R1
    trainer=train.VRPTWTrainer(dict(train.env_params,**reward_config['E'],pomo_size=100),
        copy.deepcopy(train.model_params),copy.deepcopy(train.optimizer_params),tp)
    ckpt=torch.load(checkpoint,map_location=trainer.device,weights_only=False)
    assert ckpt['epoch']==1000
    trainer.model.load_state_dict(ckpt['model_state_dict'],strict=True)
    trainer.model.eval()
    report['checkpoint_sha256']=sha(checkpoint)
    report['pure_evaluation']=dict(seed=42,pomo=100,aug=8,mc=1,device='CPU',note='RC reused GPU evaluation; numeric device differences possible')
    for suffix in ['', '_RAIN2','_ACC2']:
        src=path.with_name(name.lower()+suffix.lower()+'.txt')
        inst=load_solomon(str(src))
        train._set_seed(42)
        with torch.no_grad():
            pr,reward=trainer._best_solution(inst)
        pr=[[0]+[int(n) for n in r if n]+[0] for r in pr if any(r)]
        for method,rs in [('full_history',[[0,*r,0] for r in routes]),('pure1000',pr)]:
            customers=[n for r in rs for n in r if n]
            assert len(customers)==len(set(customers)) and set(customers)<=set(range(1,101))
            s=_solution_stats(inst,rs)
            row=dict(name=name+suffix,method=method,vehicles=len(rs),travel_time=s['total_travel_time'],
                     distance=s['total_dist'],late_stops=s['late_count'],total_late=s['total_late'],routes=rs,input_hash=sha(src),
                     served=len(customers),unserved=sorted(set(range(1,101))-set(customers)))
            report['results'].append(row)
            print('RESULT',name+suffix,method,row['vehicles'],row['travel_time'],row['late_stops'],row['total_late'],flush=True)
            write(out/'results.json',report)
    if args.family=='r':
        break
lines=['# Fixed-gen Pure RL1000 vs Full History','','R selection: first successful plan in R102..R112; retain failures. Both trained/estimated from fixed gen, not Solomon.', '',
       '| Instance | Method | Served | Vehicles | Travel time | Distance | Late count | Total late |','|---|---|---:|---:|---:|---:|---:|---:|']
lines[2]=report['selection']
for r in sorted(report['results'],key=lambda r:(r['name'],r['method'])):
    lines.append(f"| {r['name']} | {r['method']} | {r.get('served',100)} | {r['vehicles']} | {r['travel_time']:.2f} | {r['distance']:.2f} | {r['late_stops']} | {r['total_late']:.2f} |")
(out/'review.md').write_text('\n'.join(lines),encoding='utf-8')
print('DONE',out,flush=True)
