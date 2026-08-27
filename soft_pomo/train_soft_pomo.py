"""
train_soft_pomo.py — Pure RL baseline for Soft-Clustering POMO.

Identical architecture to original_POMO/train_vrptw.py (same model, same env).
Used as the ablation baseline against train_soft_cluster.py (LLM-guided).

Difference from original_POMO/train_vrptw.py:
  - result_dir → soft_pomo/result  (keeps results co-located with proposed model)
  - MLflow experiment name: "SoftPOMO_PureRL"

Run:
  python train_soft_pomo.py --benchmark c1
  python train_soft_pomo.py --benchmark rc1
  python train_soft_pomo.py --benchmark r1
  python train_soft_pomo.py --test-only
"""
from __future__ import annotations

import os
import sys
import argparse

import random
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from train_vrptw import (
    VRPTWTrainer,
    env_params      as _base_env_params,
    model_params,
    optimizer_params,
    trainer_params  as _base_trainer_params,
    _REWARD_CONFIGS,
    _set_seed,
    _load_model_flexible,
)
from vrptw_env import _load_sol, plot_routes, _print_solution, _save_solution, AverageMeter

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

_ROOT    = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_ROOT, 'data', 'Solomon')

# Default dataset — C1 benchmark (same as original_POMO)
_BASE_C1 = ["c102", "c103", "c104", "c105", "c106", "c107", "c108", "c109"]
TRAIN_INSTANCES = (
    _BASE_C1
    + [f"{n}_rain_A" for n in _BASE_C1]
    + [f"{n}_rain_B" for n in _BASE_C1]
)
TEST_INSTANCES = ["c101", "c101_rain_A", "c101_rain_B", "c101_acc_A", "c101_acc_B"]

RESULT_DIR = os.path.join(_HERE, 'result')


# ── MLflow-aware subclass ─────────────────────────────────────────────────────
class SoftPOMOTrainer(VRPTWTrainer):
    """Pure-RL baseline: MLflow experiment name + type_ratio weighted sampling."""

    def __init__(self, env_p, model_p, opt_p, trainer_p):
        super().__init__(env_p, model_p, opt_p, trainer_p)
        # Build per-instance sampling weights from type_ratio
        type_ratio = trainer_p.get('type_ratio', None)
        if type_ratio:
            n_base = sum(1 for i in self.instance_pool
                         if '_rain_' not in i['name'] and '_acc_' not in i['name'])
            n_rain = sum(1 for i in self.instance_pool if '_rain_' in i['name'])
            n_acc  = sum(1 for i in self.instance_pool if '_acc_'  in i['name'])
            def _w(inst):
                n = inst['name']
                if '_acc_'  in n: return type_ratio.get('acc',  0.1) / max(n_acc,  1)
                if '_rain_' in n: return type_ratio.get('rain', 0.4) / max(n_rain, 1)
                return                   type_ratio.get('base', 0.5) / max(n_base, 1)
            self._pool_weights = [_w(i) for i in self.instance_pool]
            print(f'  [TypeRatio] base={type_ratio.get("base")} ({n_base}) '
                  f'rain={type_ratio.get("rain")} ({n_rain}) acc={type_ratio.get("acc")} ({n_acc})')
        else:
            self._pool_weights = None

    def _train_one_epoch(self, epoch: int = 0):
        reward_AM = AverageMeter()
        loss_AM   = AverageMeter()
        n_ep = self.trainer_params['train_episodes']

        use_weights = (self._pool_weights is not None
                       and not (self.curriculum_epoch > 0
                                and epoch <= self.curriculum_epoch
                                and self.base_pool))
        pool = (self.base_pool if self.curriculum_epoch > 0
                               and epoch <= self.curriculum_epoch
                               and self.base_pool
                else self.instance_pool)

        ep = 0
        while ep < n_ep:
            bs   = min(self.trainer_params['train_batch_size'], n_ep - ep)
            inst = (random.choices(pool, weights=self._pool_weights, k=1)[0]
                    if use_weights else random.choice(pool))
            r, l = self._train_one_batch(inst, bs)
            reward_AM.update(r, bs)
            loss_AM.update(l,  bs)
            ep += bs
        return reward_AM.avg, loss_AM.avg

    def run(self):
        if _MLFLOW:
            mlflow.set_experiment("SoftPOMO_PureRL")
        super().run()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import copy
    env_params     = copy.deepcopy(_base_env_params)
    trainer_params = copy.deepcopy(_base_trainer_params)

    # Redirect data / result dirs to this folder's context
    trainer_params['data_dir']       = DATA_DIR
    trainer_params['train_instances'] = TRAIN_INSTANCES
    trainer_params['test_instances']  = TEST_INSTANCES
    trainer_params['result_dir']      = RESULT_DIR

    parser = argparse.ArgumentParser(description='Soft-POMO pure RL baseline')
    parser.add_argument('--test-only',  action='store_true')
    parser.add_argument('--resume',     default=None)
    parser.add_argument('--epochs',     type=int,   default=trainer_params['epochs'])
    parser.add_argument('--n-mc',       type=int,   default=trainer_params['n_mc_samples'],
                        help='MC sampling passes for best_solution')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--config',     default='E', choices=list(_REWARD_CONFIGS),
                        help='Reward shaping config: A B C D E(default)')
    parser.add_argument('--pomo',       type=int,   default=None,
                        help='Override POMO_SIZE')
    parser.add_argument('--pomo-auto',  action='store_true',
                        help='Set pomo_size = ceil(total_demand/capacity) per instance')
    parser.add_argument('--aug',        action='store_true',
                        help='8-fold geometric augmentation at test time')
    parser.add_argument('--top-k',      type=int,   default=None,
                        help='Use K random starts at test time (fair comparison mode)')
    parser.add_argument('--benchmark',  default='c1', choices=['c1', 'rc1', 'r1', 'mixed'],
                        help='Instance benchmark set (default: c1)')
    parser.add_argument('--base-only',  action='store_true',
                        help='Train on base instances only (no event instances)')
    parser.add_argument('--acc-ratio',  type=float, default=0.1,
                        help='ACC instance training ratio (default 0.1 → base:0.5 rain:0.4 acc:0.1)')
    args = parser.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)
        print(f'[Seed] {args.seed}')

    trainer_params['epochs']           = args.epochs
    trainer_params['n_mc_samples']     = args.n_mc
    trainer_params['use_augmentation'] = args.aug

    # Benchmark switching
    if args.benchmark == 'rc1':
        _B = [f"rc{i:03d}" for i in range(102, 109)]
        _E = [f"{n}_rain_A" for n in _B] + [f"{n}_rain_B" for n in _B]
        trainer_params['train_instances'] = _B + _E
        trainer_params['test_instances']  = ["rc101", "rc101_rain_A", "rc101_rain_B",
                                              "rc101_acc_A", "rc101_acc_B"]
        print('[benchmark=rc1] Train: rc102-rc108 + rain  |  Test: rc101')
    elif args.benchmark == 'r1':
        _B = [f"r{i:03d}" for i in range(102, 113)]
        _E = [f"{n}_rain_A" for n in _B] + [f"{n}_rain_B" for n in _B]
        trainer_params['train_instances'] = _B + _E
        trainer_params['test_instances']  = ["r101", "r101_rain_A", "r101_rain_B",
                                              "r101_acc_A", "r101_acc_B"]
        print('[benchmark=r1] Train: r102-r112 + rain  |  Test: r101')
    elif args.benchmark == 'mixed':
        _C  = [f"c{i:03d}"  for i in range(102, 110)]
        _RC = [f"rc{i:03d}" for i in range(102, 109)]
        _R  = [f"r{i:03d}"  for i in range(102, 113)]
        _ALL = _C + _RC + _R
        _E  = [f"{n}_rain_A" for n in _ALL] + [f"{n}_rain_B" for n in _ALL]
        trainer_params['train_instances'] = _ALL + _E
        trainer_params['test_instances']  = ["c101", "rc101", "r101",
                                              "c101_rain_A", "rc101_rain_A", "r101_rain_A",
                                              "c101_acc_A",  "rc101_acc_A",  "r101_acc_A"]
        print('[benchmark=mixed] Train: C1+RC1+R1  |  Test: c101/rc101/r101')

    if args.base_only:
        _B = [n for n in trainer_params['train_instances'] if '_' not in n]
        trainer_params['train_instances']  = _B
        trainer_params['test_instances']   = [trainer_params['test_instances'][0]]
        trainer_params['curriculum_epoch'] = 0
        print(f'[base-only] Train: {_B[0]}-{_B[-1]}  |  Test: {trainer_params["test_instances"]}')

    # ACC instances in training (default ratio 0.1)
    if args.acc_ratio > 0:
        _cur        = trainer_params['train_instances']
        _base_names = [n for n in _cur if '_rain_' not in n and '_acc_' not in n]
        _acc_names  = ([f"{n}_acc_A" for n in _base_names]
                       + [f"{n}_acc_B" for n in _base_names])
        trainer_params['train_instances'] = _cur + _acc_names
        rain_ratio = 1.0 - 0.5 - args.acc_ratio
        trainer_params['type_ratio'] = {'base': 0.5, 'rain': rain_ratio, 'acc': args.acc_ratio}
        print(f'[acc-ratio={args.acc_ratio}] base=0.5 rain={rain_ratio:.1f} acc={args.acc_ratio}')

    if args.top_k is not None:
        trainer_params['eval_top_k'] = args.top_k
        print(f'[Eval] Using {args.top_k} random starts (fair comparison mode)')
    if args.resume:
        trainer_params['model_load']['enable'] = True
        trainer_params['model_load']['path']   = args.resume

    cfg = _REWARD_CONFIGS[args.config]
    env_params.update(cfg)
    if args.pomo is not None:
        env_params['pomo_size'] = args.pomo
        print(f'[POMO] pomo_size overridden to {args.pomo}')
    if args.pomo_auto:
        trainer_params['pomo_auto'] = True
        print('[POMO] pomo_auto: pomo_size = ceil(total_demand/capacity) per instance')

    trainer_params['result_dir'] = os.path.join(RESULT_DIR, f'config_{args.config}')
    print(f'[Config {args.config}] late_count={cfg["late_count_penalty"]}  '
          f'late={cfg["late_penalty"]}  vehicle={cfg["vehicle_penalty"]}')

    trainer = SoftPOMOTrainer(env_params, model_params, optimizer_params, trainer_params)

    if args.test_only:
        ckpt_path = os.path.join(trainer.result_dir, 'checkpoint-last.pt')
        if not os.path.isfile(ckpt_path):
            print(f'[Test] No checkpoint at {ckpt_path}')
            return
        ckpt = torch.load(ckpt_path, map_location=trainer.device)
        _load_model_flexible(trainer.model, ckpt['model_state_dict'])
        trainer.model.eval()
        plots_dir = os.path.join(trainer.result_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        for inst, data_path in zip(trainer.test_pool, trainer.test_paths):
            routes, best_reward = trainer._best_solution(inst)
            sol_routes = _load_sol(data_path, inst['n_customers'])
            _print_solution(inst, routes, best_reward, sol_routes)
            plot_routes(inst, routes, best_reward,
                        os.path.join(plots_dir, f"{inst['name']}_best_route.png"))
            _save_solution(inst, routes, best_reward,
                           os.path.join(plots_dir, f"{inst['name']}_solution.txt"),
                           sol_routes)
    else:
        trainer.run()


if __name__ == '__main__':
    main()
