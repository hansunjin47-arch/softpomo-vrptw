"""Evaluate trusted Pure RL bundle checkpoints using their travel-time objective."""
import argparse
import copy
from pathlib import Path
import statistics
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--epoch',type=int,required=True)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--one-shot',action='store_true')
    args=parser.parse_args()
    root=args.bundle.resolve()
    sys.path.insert(0,str(root))
    sys.path.insert(0,str(root/'server'))
    import train_cached as io
    import torch
    import train_vrptw as pure
    from vrptw_env import load_solomon, _solution_stats
    digest=io.verify_bundle(root)
    saved=io.read(root/'runs/pure/settings.json')
    assert saved['bundle_sha256']==digest
    assert saved['config']==io.read(root/'configs/pure.json')
    config=copy.deepcopy(saved['config'])
    env=config['environment']
    assert env['dist_penalty_weight']==0 and env['time_penalty']==1
    checkpoint=root/'runs/pure'/f'checkpoint-{args.epoch}.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['epoch']==args.epoch and state['server_contract']==saved
    split=io.read(root/'split.json')
    names=[r['name'] for r in split['test']]
    assert len(set(names))==80 and not set(names)&set(config['trainer']['train_instances'])
    for entry in split['test']:
        assert io.sha(root/'data'/(entry['name']+'.txt'))==entry['sha256']
    if args.check_only:
        print('PASS: checkpoint, objective, bundle and test80 verified',flush=True)
        return
    out=root/(f'eval_{args.epoch}_test80_oneshot' if args.one_shot else f'eval_{args.epoch}_test80_gpu')
    config['trainer'].update(use_augmentation=not args.one_shot,n_mc_samples=1)
    config['model']['eval_type']='argmax'
    assert config['environment']['pomo_size']==100
    if args.one_shot:
        assert config['model'].get('learn_first_customer') is True
        config['environment']['pomo_size']=1
        config['trainer'].update(pomo_auto=False,eval_top_k=None)
    contract=dict(checkpoint_sha256=io.sha(checkpoint),bundle_sha256=digest,
        evaluator_sha256=io.sha(Path(__file__)),config=config,device='cuda',test_names=names)
    if (out/'evaluation_contract.json').exists():
        assert io.read(out/'evaluation_contract.json')==contract
    else:
        io.write(out/'evaluation_contract.json',contract)
    trainer=io.build_trainer(config,root,out/'runtime')
    trainer.model.load_state_dict(state['model_state_dict'],strict=True)
    trainer.model.eval()
    results=io.read(out/'results.json') if (out/'results.json').exists() else []
    assert [r['name'] for r in results]==names[:len(results)]
    for index,entry in enumerate(split['test']):
        if index<len(results):
            continue
        pure._set_seed(42+index)
        inst=load_solomon(str(root/'data'/(entry['name']+'.txt')))
        with torch.no_grad():
            routes,reward=trainer._best_solution(inst)
        if args.one_shot:
            assert trainer.env.pomo_size==1
        routes=[[int(n) for n in route] for route in routes]
        assert sorted(n for route in routes for n in route if n)==list(range(1,101))
        stats=_solution_stats(inst,routes)
        row=dict(name=entry['name'],kind=entry['kind'],routes=routes,
            vehicles=len(stats['per_vehicle']),travel_time=float(stats['total_travel_time']),
            distance=float(stats['total_dist']),late_count=int(stats['late_count']),
            total_late=float(stats['total_late']),served=int(stats['n_served']),reward=float(reward))
        row['cost']=env['vehicle_penalty']*row['vehicles']/100+row['travel_time']/(float(inst['T'])*env['TT_max'])+env['late_penalty']*row['total_late']/(float(inst['T'])*env['Lt_max'])
        assert abs(row['cost']+row['reward'])<1e-4
        results.append(row)
        io.write(out/'results.json',results)
        io.write(out/'status.json',dict(stage='evaluating',completed=len(results),total=80))
        print('EVALUATED',root.name,args.epoch,len(results),row['cost'],flush=True)
    keys=('vehicles','travel_time','late_count','total_late','reward','cost','served','distance')
    summary={}
    for kind in ('ALL','RAIN','ACCIDENT'):
        rows=[r for r in results if kind=='ALL' or r['kind']==kind]
        summary[kind]=dict(count=len(rows),mean={k:statistics.mean(r[k] for r in rows) for k in keys},
            std={k:statistics.stdev(r[k] for r in rows) for k in keys})
    io.write(out/'summary.json',summary)
    io.write(out/'status.json',dict(stage='complete',completed=80,epoch=args.epoch))


if __name__=='__main__':
    main()
