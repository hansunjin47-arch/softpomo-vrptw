"""Frozen, bias-free comparison on existing RCH BASE instances. No LLM imports."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_vrptw as train
from vrptw_env import load_solomon, _solution_stats


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['pure500', 'pure1000'])
    args = parser.parse_args()
    sources = {
        'pure500': 'result/config_E_rc1_final_weightedsum_b32_aug8/rc102-rc108/checkpoint-500.pt',
        'pure1000': 'result/config_E_rc1_final_weightedsum_b32_aug8/rc102-rc108/checkpoint-1000.pt',
        'phase2': 'result_soft_rc1_selective_cluster_v2_b32_20260920_130657/training/rc102-rc108/checkpoint-1000.pt',
    }
    assert all(k in sources for k in args.models)
    out = ROOT / 'analysis' / ('rc_fixed_gen_eval_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    tp = copy.deepcopy(train.trainer_params)
    tp.update(train_instances=['rc102'], test_instances=[], model_load={'enable': False},
              result_dir=str(out / 'runtime'), use_augmentation=True, n_mc_samples=1,
              eval_top_k=None, pomo_auto=False, use_amp=False, train_dataset=None)
    ep = dict(train.env_params, **train._REWARD_CONFIGS_RC1['E'])
    ep['pomo_size'] = 100
    mp = copy.deepcopy(train.model_params)
    mp['eval_type'] = 'argmax'
    trainer = train.VRPTWTrainer(ep, mp, copy.deepcopy(train.optimizer_params), tp)
    manifest = dict(models={}, results=[], bias=0, llm_calls=False, seed=42,
                    pomo=100, augmentation=8, mc=1, amp=False, env_params=ep,
                    scope='RCH01-RCH15 BASE; Phase2 is policy-only, not guidance-on',
                    source_hashes={f: digest(ROOT / f) for f in ['train_vrptw.py', 'vrptw_env.py']})
    for label in args.models:
        path = ROOT / sources[label]
        sha = digest(path)
        checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        trainer.model.eval()
        manifest['models'][label] = dict(path=str(path), sha256=sha, epoch=checkpoint['epoch'])
        for i in range(1, 16):
            name = f'RCH{i:02d}'
            data = ROOT.parent / 'data/Solomon' / (name + '.txt')
            inst = load_solomon(str(data))
            assert not inst['preset_events'], 'This evaluator is intentionally BASE-only'
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            with torch.no_grad():
                routes, reward = trainer._best_solution(inst)
            routes = [[0] + [int(n) for n in r if n] + [0] for r in routes if any(r)]
            visited = [n for r in routes for n in r if n]
            assert len(visited) == len(set(visited)), 'Duplicate customer'
            stats = _solution_stats(inst, routes)
            customer_late = sum(s['late'] > 0 for v in stats['per_vehicle'] for s in v['stop_details'])
            depot_late = sum(v['ret_late'] > 0 for v in stats['per_vehicle'])
            row = dict(model=label, instance=name, vehicles=len(routes), served=len(visited),
                       missing=sorted(set(range(1, 101)) - set(visited)),
                       distance=float(sum(v['rdist'] for v in stats['per_vehicle'])),
                       late_stops=int(stats['late_count']), customer_late=int(customer_late),
                       depot_late=int(depot_late), total_late=float(stats['total_late']),
                       reward=float(reward), routes=routes, input_sha256=digest(data))
            manifest['results'].append(row)
            (out / 'results.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print(json.dumps({k: v for k, v in row.items() if k not in ('routes', 'input_sha256')}), flush=True)
        assert digest(path) == sha
    lines = ['# RC Fixed Gen: Frozen Policy Evaluation', '',
             'All models: bias OFF, BASE only, FP32, POMO100, aug8, seed42. Phase2 policy-only.', '',
             '| Model | Instance | Vehicles | Served | Distance | Late customers | Late returns | Total late |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in manifest['results']:
        lines.append(f"| {r['model']} | {r['instance']} | {r['vehicles']} | {r['served']} | {r['distance']:.2f} | {r['customer_late']} | {r['depot_late']} | {r['total_late']:.2f} |")
    (out / 'review.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
