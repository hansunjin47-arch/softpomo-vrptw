"""
eval_soft_plug.py — LLM soft-clustering plug-in inference for any trained RL checkpoint.

Training: pure RL only (Config A-F from original_POMO).
Inference: LLM soft-clustering confidence bias applied at every routing step.

No retraining. The LLM guides the RL policy without modifying weights.

Run:
  python eval_soft_plug.py --benchmark r1  --config F \\
      --resume ../original_POMO/result/config_F_r1_pomo100/checkpoint-last.pt

  python eval_soft_plug.py --benchmark rc1 --config F \\
      --resume ../original_POMO/result/config_F_rc1_pomo100/checkpoint-last.pt

  python eval_soft_plug.py --benchmark c1  --config F \\
      --resume ../original_POMO/result/config_F_pomo100/checkpoint-last.pt

  python eval_soft_plug.py --benchmark r1  --config F --no-llm \\
      --resume ../original_POMO/result/config_F_r1_pomo100/checkpoint-last.pt
"""
from __future__ import annotations

import os
import sys
import argparse

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from train_soft_cluster import (
    SoftClusterTrainer,
    env_params,
    model_params,
    optimizer_params,
    trainer_params,
    llm_params,
    _set_seed,
)
from vrptw_env import _print_solution, _save_solution, plot_routes
import train_vrptw as _base_config


_BENCHMARK_TEST = {
    'c1':    ["c101",  "c101_rain_A",  "c101_rain_B",  "c101_acc_A",  "c101_acc_B"],
    'r1':    ["r101",  "r101_rain_A",  "r101_rain_B",  "r101_acc_A",  "r101_acc_B"],
    'rc1':   ["rc101", "rc101_rain_A", "rc101_rain_B", "rc101_acc_A", "rc101_acc_B"],
    'mixed': ["c101",  "rc101",  "r101",
              "c101_rain_A", "rc101_rain_A", "r101_rain_A",
              "c101_acc_A",  "rc101_acc_A",  "r101_acc_A"],
}

_BENCHMARK_TRAIN = {
    'c1':  [f"c{i:03d}" for i in range(102, 110)],
    'r1':  [f"r{i:03d}" for i in range(102, 113)],
    'rc1': [f"rc{i:03d}" for i in range(102, 109)],
}


def main():
    parser = argparse.ArgumentParser(description='LLM soft-clustering plug-in for trained RL checkpoint')
    parser.add_argument('--resume',      required=True, help='Path to trained RL checkpoint (.pt)')
    parser.add_argument('--benchmark',   default='r1',  choices=['c1', 'rc1', 'r1', 'mixed'])
    parser.add_argument('--config',      default='F',   choices=list(_base_config._REWARD_CONFIGS))
    parser.add_argument('--no-llm',      action='store_true', help='Disable LLM bias (pure RL inference)')
    parser.add_argument('--no-ontology', action='store_true')
    parser.add_argument('--no-cot',      action='store_true')
    parser.add_argument('--bias-strength', type=float, default=llm_params['bias_strength'])
    parser.add_argument('--model',       default=llm_params['model'], help='LLM model name')
    parser.add_argument('--result-dir',  default=None,
                        help='Base result dir (default: result_soft — shares llm_cache)')
    parser.add_argument('--tag',         default=None,
                        help='Extra tag appended to output subdir: plug_{config}_{benchmark}_{tag}')
    parser.add_argument('--aug',         type=int, default=1, metavar='N',
                        help='Number of coordinate augmentations (1=none, 8=full D4)')
    parser.add_argument('--mc',          type=int, default=1, metavar='N',
                        help='MC sampling runs per instance (1=greedy, N=stochastic × N)')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--pomo',        type=int, required=True,
                        help='POMO rollout count, forced identical for LLM-on and --no-llm.')
    args = parser.parse_args()

    _set_seed(args.seed)

    if not os.path.isfile(args.resume):
        print(f'[Error] Checkpoint not found: {args.resume}')
        sys.exit(1)

    # Reward config
    cfg = _base_config._REWARD_CONFIGS[args.config]
    env_params.update(cfg)
    print(f'[Config {args.config}] {cfg}')

    # LLM params
    llm_p = dict(llm_params)
    llm_p['enabled']      = not args.no_llm
    llm_p['use_ontology'] = not args.no_ontology
    llm_p['use_cot']      = not args.no_cot
    llm_p['bias_strength'] = args.bias_strength
    llm_p['model']        = args.model

    # Trainer params — minimal: only test instances needed
    base_train = _BENCHMARK_TRAIN.get(args.benchmark, _BENCHMARK_TRAIN['c1'])
    tp = dict(trainer_params)
    tp['train_instances'] = base_train          # needed for trainer init (pool build)
    tp['test_instances']  = _BENCHMARK_TEST[args.benchmark]
    tp['pomo_size']       = args.pomo
    tp['model_load']      = dict(enable=False, path=None, epoch=None)

    # tp['result_dir'] = result_soft 고정 → trainer가 {result_dir}/llm_cache 로 캐시 접근
    # result_dir (output용) = result_soft/plug_{config}_{benchmark} → config별 결과 분리
    base_dir   = args.result_dir or os.path.join(_HERE, 'result_soft')
    tp['result_dir'] = base_dir
    subdir_name = f'plug_{args.config}_{args.benchmark}' + (f'_{args.tag}' if args.tag else '')
    result_dir = os.path.join(base_dir, subdir_name)
    os.makedirs(result_dir, exist_ok=True)

    tag = (f'LLM-plug [{args.benchmark.upper()} / Config {args.config}]'
           + (' + ontology' if llm_p['use_ontology'] else '')
           + (' [no-LLM baseline]' if not llm_p['enabled'] else ''))
    print(f'[{tag}]')
    print(f'  checkpoint : {args.resume}')
    print(f'  output_dir : {result_dir}')

    trainer = SoftClusterTrainer(env_params, model_params, optimizer_params, tp, llm_p)

    # Load checkpoint
    ckpt = torch.load(args.resume, map_location=trainer.device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    own_keys   = set(trainer.model.state_dict().keys())
    filtered   = {k: v for k, v in state_dict.items() if k in own_keys}
    missing    = own_keys - set(filtered)
    extra      = set(state_dict) - own_keys
    trainer.model.load_state_dict(filtered, strict=False)
    trainer.model.eval()

    if missing:
        print(f'  [warn] missing keys (random init): {sorted(missing)}')
    if extra:
        print(f'  [info] extra keys skipped: {sorted(extra)}')
    print(f'  [OK] Loaded {len(filtered)}/{len(state_dict)} keys from checkpoint.')

    # Inference
    plots_dir = os.path.join(result_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    results = []
    for inst in trainer.test_pool:
        if args.aug > 1 and args.mc > 1:
            # aug × mc: best across aug augmentations, each sampled mc times
            best_routes, best_reward = None, -float('inf')
            for aug_idx in range(args.aug):
                aug_inst = trainer._aug_instance(inst, aug_idx) if aug_idx > 0 else inst
                routes, reward = trainer._best_solution_mc(aug_inst, n_samples=args.mc)
                if reward > best_reward:
                    best_reward, best_routes = reward, routes
            routes, reward = best_routes, best_reward
        elif args.aug > 1:
            routes, reward = trainer._best_solution_aug(inst, n_aug=args.aug)
        elif args.mc > 1:
            routes, reward = trainer._best_solution_mc(inst, n_samples=args.mc)
        else:
            routes, reward = trainer._best_solution(inst)
        print(f'\n[Plug] {inst["name"]}  reward={reward:.4f}')
        _print_solution(inst, routes, reward)
        sol_path = os.path.join(plots_dir, f'{inst["name"]}_solution.txt')
        _save_solution(inst, routes, reward, sol_path)
        plot_routes(inst, routes, reward,
                    os.path.join(plots_dir, f'{inst["name"]}_routes.png'))
        results.append((inst['name'], reward))

    print('\n' + '=' * 60)
    print(f'  Summary: {args.benchmark.upper()} / Config {args.config} / {"LLM" if llm_p["enabled"] else "no-LLM"}')
    print(f'  {"Instance":<25}  {"Reward":>10}')
    print('  ' + '-' * 38)
    for name, r in results:
        print(f'  {name:<25}  {r:>10.4f}')
    print('  ' + '-' * 38)
    avg = sum(r for _, r in results) / len(results) if results else 0
    print(f'  {"Average":<25}  {avg:>10.4f}')
    print('=' * 60)

    # Save summary
    summary_path = os.path.join(result_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'benchmark={args.benchmark}  config={args.config}  llm={llm_p["enabled"]}\n')
        f.write(f'checkpoint={args.resume}\n\n')
        f.write(f'{"Instance":<25}  {"Reward":>10}\n')
        f.write('-' * 38 + '\n')
        for name, r in results:
            f.write(f'{name:<25}  {r:>10.4f}\n')
        f.write('-' * 38 + '\n')
        f.write(f'{"Average":<25}  {avg:>10.4f}\n')
    print(f'[Saved] {summary_path}')


if __name__ == '__main__':
    main()
