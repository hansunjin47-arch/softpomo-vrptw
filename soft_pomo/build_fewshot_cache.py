"""
build_fewshot_cache.py

Rebuild cluster JSONs using episode_tracker saved during zero-shot training
(by _refresh_experience_caches at experience_refresh_epoch=500).

Flow:
  1. Trainer init: loads existing cluster JSONs (→ cluster assignments + original conf)
  2. Load episode_tracker from disk (episode_tracker.json saved at epoch 500)
  3. For ALL non-ACC instances:
       a. Call build_experience_examples → generate experience text from tracker data
       b. Save {name}_experience.txt
       c. Force-rebuild cluster JSON via LLM with experience text in prompt

Usage:
  python build_fewshot_cache.py --benchmark c1
  python build_fewshot_cache.py --benchmark rc1
  python build_fewshot_cache.py --benchmark r1
"""

from __future__ import annotations

import os
import sys
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from train_vrptw_llm import (
    env_params as _base_env_params,
    model_params,
    optimizer_params,
    trainer_params as _base_trainer_params,
    llm_params as _base_llm_params,
    _set_seed,
)
import train_vrptw as _base_config
from train_soft_cluster import SoftClusterTrainer
from SoftClusterOntology import episode_tracker
from SoftClusterLLMModule import build_experience_examples

env_params     = dict(_base_env_params)
trainer_params = dict(_base_trainer_params)
trainer_params['result_dir'] = os.path.join(_HERE, 'result_soft')
llm_params     = dict(_base_llm_params)
llm_params['soft_clustering'] = True


def _rebuild(trainer: SoftClusterTrainer, inst: dict):
    """Call LLM (with experience if available) and save result to {name}_fewshot_cluster.json."""
    import json
    from SoftClusterLLMModule import get_all_clusters_confidence
    from train_soft_cluster import _dump_cluster_cache

    name      = inst['name']
    cache_dir = trainer._llm_cache_dir
    lp        = trainer.llm_params
    N         = inst['n_customers']

    clusters = trainer._cluster_assign_cache.get(name)
    if clusters is None:
        print(f'  [{name}] no cluster assignment — skipping')
        return

    exp_path = os.path.join(cache_dir, f'{name}_experience.txt')
    experience_section = ""
    if os.path.isfile(exp_path):
        with open(exp_path, encoding='utf-8') as f:
            experience_section = f.read()

    fewshot_path = os.path.join(cache_dir, f'{name}_fewshot_cluster.json')
    if os.path.isfile(fewshot_path):
        print(f'  [{name}] already exists — skipping', flush=True)
        return

    tag = '+exp' if experience_section else 'zero-shot'
    try:
        flat_conf = get_all_clusters_confidence(
            inst, clusters,
            model=lp['model'], use_cot=lp['use_cot'],
            use_ontology=lp['use_ontology'],
            experience_section=experience_section,
            reward_config=lp.get('reward_config', 'F'),
        )
    except Exception as e:
        print(f'  [{name}] LLM failed ({e}) — will retry later', flush=True)
        return False

    with open(fewshot_path, 'w', encoding='utf-8') as f:
        json.dump(_dump_cluster_cache(clusters, flat_conf), f)
    print(f'  [{name}] saved → {name}_fewshot_cluster.json ({tag})', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description='Build few-shot cluster caches')
    parser.add_argument('--benchmark', default='c1', choices=['c1', 'rc1', 'r1'])
    parser.add_argument('--config',    default='F',  choices=list(_base_config._REWARD_CONFIGS))
    parser.add_argument('--model',     default=llm_params['model'])
    parser.add_argument('--no-cot',    action='store_true')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)

    cfg = _base_config._REWARD_CONFIGS[args.config]
    env_params.update(cfg)
    print(f'[Config {args.config}] {cfg}')

    llm_params['model']         = args.model
    llm_params['reward_config'] = args.config
    if args.no_cot:
        llm_params['use_cot'] = False

    if args.benchmark == 'rc1':
        _B = [f"rc{i:03d}" for i in range(102, 109)]
        test_inst = ["rc101", "rc101_rain_A", "rc101_rain_B", "rc101_acc_A", "rc101_acc_B"]
    elif args.benchmark == 'r1':
        _B = [f"r{i:03d}" for i in range(102, 113)]
        test_inst = ["r101", "r101_rain_A", "r101_rain_B", "r101_acc_A", "r101_acc_B"]
    else:
        _B = [f"c{i:03d}" for i in range(102, 110)]
        test_inst = ["c101", "c101_rain_A", "c101_rain_B", "c101_acc_A", "c101_acc_B"]

    _E   = [f"{n}_rain_A" for n in _B] + [f"{n}_rain_B" for n in _B]
    _acc = [f"{n}_acc_A" for n in _B] + [f"{n}_acc_B" for n in _B]

    trainer_params['train_instances']  = _B + _E + _acc
    trainer_params['test_instances']   = test_inst
    trainer_params['type_ratio']       = {'base': 1/3, 'rain': 1/3, 'acc': 1/3}
    trainer_params['curriculum_epoch'] = 0

    print(f'\n[build_fewshot_cache] benchmark={args.benchmark}  config={args.config}')
    print('[Initialising trainer (loads existing cluster JSONs)...]')

    trainer = SoftClusterTrainer(
        env_params, model_params, optimizer_params, trainer_params, llm_params,
    )
    cache_dir = trainer._llm_cache_dir

    # Load episode_tracker saved at epoch 500 of zero-shot training
    tracker_path = os.path.join(cache_dir, 'episode_tracker.json')
    if os.path.isfile(tracker_path):
        episode_tracker.load_from_disk(tracker_path)
        print(f'[Tracker] loaded from {tracker_path}')
    else:
        print(f'[Tracker] WARNING: {tracker_path} not found — experience.txt will not be generated')

    all_insts = list(trainer.instance_pool) + list(trainer.test_pool)

    import re as _re

    def _base_name(name: str) -> str:
        """Strip event suffix to get base instance name, or None if already base."""
        m = _re.match(r'^(.+?)_(rain|acc)_[AB]$', name, _re.IGNORECASE)
        return m.group(1) if m else None

    print(f'\n[Experience] Generating experience.txt for {len(all_insts)} instances...')
    exp_count = 0
    for inst in all_insts:
        name     = inst['name']
        clusters = trainer._cluster_assign_cache.get(name)
        conf_t   = trainer._original_cluster_conf.get(name)
        if clusters is None or conf_t is None:
            print(f'  [{name}] no cluster cache — skipping experience')
            continue
        base = _base_name(name)   # None for base instances, e.g. "C102" for "C102_RAIN_A"
        exp_text = build_experience_examples(inst, clusters, conf_t, name,
                                             base_scenario=base)
        if exp_text:
            exp_path = os.path.join(cache_dir, f'{name}_experience.txt')
            with open(exp_path, 'w', encoding='utf-8') as f:
                f.write(exp_text)
            tag = f'(vs base={base})' if base else ''
            exp_count += 1
            print(f'  [{name}] experience saved {tag}')
        else:
            print(f'  [{name}] no episode data in tracker')
    print(f'[Experience] {exp_count}/{len(all_insts)} experience.txt saved')

    print(f'\n[Rebuild] Rebuilding {len(all_insts)} cluster JSON(s) with LLM...')
    t0 = time.time()
    failed = []
    for inst in all_insts:
        ok = _rebuild(trainer, inst)
        if ok is False:
            failed.append(inst)

    max_passes = 5
    pass_num   = 1
    while failed and pass_num <= max_passes:
        print(f'\n[Retry pass {pass_num}/{max_passes}] {len(failed)} failed — retrying in 120s...', flush=True)
        time.sleep(120)
        still_failed = []
        for inst in failed:
            ok = _rebuild(trainer, inst)
            if ok is False:
                still_failed.append(inst)
        failed = still_failed
        pass_num += 1

    if failed:
        print(f'\n[WARNING] {len(failed)} instance(s) still failed after all retries:')
        for inst in failed:
            print(f'  {inst["name"]}')

    elapsed = time.time() - t0
    print(f'\n[Done] rebuilt in {elapsed/60:.1f} min  ({len(all_insts)-len(failed)}/{len(all_insts)} succeeded)')
    print(f'Cache dir: {cache_dir}')

    if args.benchmark == 'rc1':
        ckpt = 'result_soft/rc102-rc108/checkpoint-1000.pt'
    elif args.benchmark == 'r1':
        ckpt = 'result_soft/r102-r112/checkpoint-1000.pt'
    else:
        ckpt = 'result_soft/c102-c109/checkpoint-1000.pt'
    print(f'\nNext: few-shot training')
    print(f'  python train_soft_cluster.py --benchmark {args.benchmark} '
          f'--config {args.config} --resume {ckpt} '
          f'--epochs 1500 --experience-refresh-epoch 9999')


if __name__ == '__main__':
    main()
