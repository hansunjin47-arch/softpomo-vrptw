"""Frozen final-policy evaluation on all 21 training scenarios; no API calls."""
import copy
import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import train_soft_cluster as train
import train_vrptw
import VRPTWLLMModule
import SoftClusterLLMModule
import SelectiveAdvanceLLMModule
from vrptw_env import load_solomon,_solution_stats

def reject(*a,**kw):
    raise AssertionError('No LLM calls allowed during frozen evaluation')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pure',action='store_true',help='Evaluate Pure RL1000 with zero bias and identical event handling')
    parser.add_argument('--full-confidence',action='store_true',help='Evaluate full-confidence Sonnet Phase2 with its own caches')
    args=parser.parse_args()
    if args.pure and args.full_confidence:
        parser.error('Select only one evaluation model')
    run=ROOT/'result_soft_rc1_selective_cluster_v2_b32_20260920_130657'
    label='rc_pure1000_train_eval_' if args.pure else 'rc_selective_train_eval_'
    if args.full_confidence:
        label='rc_full_confidence_train_eval_'
    out=ROOT/'analysis'/(label+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    checkpoint=run/'training/rc102-rc108/checkpoint-1000.pt'
    if args.pure:
        checkpoint=ROOT/'result/config_E_rc1_final_weightedsum_b32_aug8/rc102-rc108/checkpoint-1000.pt'
    cache_dir=run/'runtime_cache'
    cache_hashes={}
    if args.full_confidence:
        checkpoint=ROOT/'result_soft_rc1_conf_sonnet46/rc102-rc108/checkpoint-1000.pt'
        cache_dir=out/'runtime_cache'
        cache_dir.mkdir()
        for p in (ROOT/'result_soft/llm_cache_rc1_conf_sonnet46').glob('*.json'):
            cache_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            shutil.copyfile(p,cache_dir/p.name)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for module in [VRPTWLLMModule,SoftClusterLLMModule,SelectiveAdvanceLLMModule]:
        module.query_llm=reject
    train.get_all_clusters_confidence=reject
    tp=copy.deepcopy(train.trainer_params)
    tp.update(train_instances=['rc102'],test_instances=['rc101'],result_dir=str(out/'runtime'),
        llm_cache_dir=str(cache_dir),cluster_mode='rl',guidance_mode='confidence' if args.full_confidence else 'selective_advance',
        start_mode='random',pomo_size=100,model_load=dict(enable=False),test_only=True,
        experience_refresh_epoch=-1,benchmark='rc1')
    beta=0. if args.pure else 5.
    # Keep event-token updates active; zero bias removes guidance without disabling ACC handling.
    lp=dict(train.llm_params,enabled=True,bias_strength=beta,force_refresh=False)
    if args.full_confidence:
        lp.update(model='anthropic/claude-sonnet-4.6',reward_config='E',use_misfit=False,use_fewshot_cache=False)
    train._set_seed(42)
    trainer=train.SoftClusterTrainer(dict(train.env_params,**train_vrptw._REWARD_CONFIGS_RC1['E']),
        copy.deepcopy(train.model_params),copy.deepcopy(train.optimizer_params),tp,lp)
    ckpt=torch.load(checkpoint,map_location=trainer.device,weights_only=False)
    assert ckpt['epoch']==1000
    trainer.model.load_state_dict(ckpt['model_state_dict'],strict=True)
    trainer.model.eval()
    if args.pure:
        original=trainer._acc_conf_now
        def checked_bias(*a,**kw):
            value=original(*a,**kw)
            assert torch.count_nonzero(value).item()==0
            return value
        trainer._acc_conf_now=checked_bias
    model_name='Pure RL1000' if args.pure else 'Full confidence Sonnet Phase2' if args.full_confidence else 'Selective Phase2'
    manifest=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,seed=42,pomo=100,aug=8,
        bias_strength=beta,model=model_name,cache_source_hashes=cache_hashes,
        split='training',llm_calls=False,event_handling='shared evaluator; all POMO100 starts; pure bias asserted zero',results=[])
    for suffix in ['', '_RAIN2','_ACC2']:
        for n in range(102,109):
            name=f'RC{n}{suffix}'
            inst=load_solomon(str(ROOT.parent/'data/Solomon'/f'{name.lower()}.txt'))
            train._set_seed(42)
            trainer._ensure_soft_cluster_cache(inst)
            if suffix=='_ACC2':
                assert len(trainer._acc_stage_events[inst['name']])==len({float(f['t_start']) for _,f in train._get_accidents(inst)})
            with torch.no_grad():
                routes,reward=trainer._best_solution_aug(inst)
            routes=[[0]+[int(n) for n in r if n]+[0] for r in routes if any(r)]
            served=[n for r in routes for n in r if n]
            assert sorted(served)==list(range(1,101))
            stats=_solution_stats(inst,routes)
            customers=[s for v in stats['per_vehicle'] for s in v['stop_details']]
            row=dict(name=name,scenario=suffix or 'BASE',vehicles=len(routes),served=len(served),
                distance=float(sum(v['rdist'] for v in stats['per_vehicle'])),
                late_stops=int(stats['late_count']),total_late=float(stats['total_late']),
                customer_late=int(sum(s['late']>0 for s in customers)),
                depot_late=int(sum(v['ret_late']>0 for v in stats['per_vehicle'])),
                depot_late_amount=float(sum(v['ret_late'] for v in stats['per_vehicle'])),
                reward=float(reward),routes=routes)
            manifest['results'].append(row)
            (out/'results.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in row.items() if k!='routes'}),flush=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cache_hashes.items())
    lines=['# RC Training-Set Evaluation','',
        f'Frozen {manifest["model"]}, bias{beta}, POMO100, augmentation8, seed42 reset per instance. All serve100 customers. Late stops include depot returns. This is training-set fit, not generalization.', '',
        '| Instance | Vehicles | Distance | Late customers | Late returns | Total late | Reward |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in manifest['results']:
        lines.append(f"| {r['name']} | {r['vehicles']} | {r['distance']:.2f} | {r['customer_late']} | {r['depot_late']} | {r['total_late']:.2f} | {r['reward']:.4f} |")
    (out/'review.md').write_text('\n'.join(lines),encoding='utf-8')
    print('DONE',out,flush=True)

if __name__=='__main__':main()
