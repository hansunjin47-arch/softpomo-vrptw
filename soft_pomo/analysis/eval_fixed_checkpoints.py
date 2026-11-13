"""Matched 500/1000 held-out evaluation using the Pure RL solution evaluator."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_vrptw as train
from vrptw_env import load_solomon, _solution_stats

out = ROOT/'analysis'/('fixed_500_1000_eval_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir(exist_ok=False)
print('OUTPUT',out,flush=True)
sources = {'RC': ('config_E_rc_fixed_v2_pure_b64_aug8_20260920_230417/RCH01-RCH15', train._REWARD_CONFIGS_RC1),
           'R': ('config_E_r_fixed_v2_pure_b64_aug8_20260920_230825/RH01-RH15', train._REWARD_CONFIGS_R1)}
report = dict(seed=42,pomo=100,augmentation=8,mc=1,evaluator='VRPTWTrainer._best_solution',results=[])
for family,(folder,configs) in sources.items():
    for epoch in [500,1000]:
        checkpoint=ROOT/'result'/folder/f'checkpoint-{epoch}.pt'
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        train._set_seed(42)
        tp=copy.deepcopy(train.trainer_params)
        tp.update(train_instances=[family.lower()+'102'], test_instances=[],
                  result_dir=str(out/f'{family}_{epoch}'), model_load=dict(enable=False),
                  use_augmentation=True,n_mc_samples=1,eval_top_k=None,pomo_auto=False)
        trainer=train.VRPTWTrainer(dict(train.env_params,**configs['E'],pomo_size=100),
                 copy.deepcopy(train.model_params),copy.deepcopy(train.optimizer_params),tp)
        ckpt=torch.load(checkpoint,map_location=trainer.device,weights_only=False)
        assert ckpt['epoch']==epoch
        trainer.model.load_state_dict(ckpt['model_state_dict'],strict=True)
        trainer.model.eval()
        for suffix in ['', '_RAIN2','_ACC2']:
            name=family+'101'+suffix
            train._set_seed(42)
            inst=load_solomon(str(ROOT.parent/'data/Solomon'/(name+'.txt')))
            routes,reward=trainer._best_solution(inst)
            routes=[[0]+[int(n) for n in r if n]+[0] for r in routes if any(r)]
            assert sorted(n for r in routes for n in r if n)==list(range(1,101))
            stats=_solution_stats(inst,routes)
            row=dict(family=family,epoch=epoch,name=name,checkpoint=str(checkpoint),sha256=digest,
                     vehicles=len(routes),distance=float(stats['total_dist']),
                     late_stops=int(stats['late_count']),total_late=float(stats['total_late']),
                     reward=float(reward),routes=routes)
            report['results'].append(row)
            (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in row.items() if k not in ['routes','checkpoint','sha256']}),flush=True)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
        del trainer,ckpt
        torch.cuda.empty_cache()
lines=['# Fixed Gen: 500 vs 1000', '', 'Matched fresh evaluation, seed42 per case, POMO100, aug8, MC1. Late stops include depot returns. All 100 customers served.', '',
       '| Instance | Epoch | Vehicles | Distance | Late stops | Total late |','|---|---:|---:|---:|---:|---:|']
for r in sorted(report['results'],key=lambda r:(r['name'],r['epoch'])):
    lines.append(f"| {r['name']} | {r['epoch']} | {r['vehicles']} | {r['distance']:.2f} | {r['late_stops']} | {r['total_late']:.2f} |")
(out/'review.md').write_text('\n'.join(lines),encoding='utf-8')
print('DONE',out,flush=True)
