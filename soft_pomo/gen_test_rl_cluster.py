"""
gen_test_rl_cluster.py
Phase1 체크포인트로 inference 후 RL cluster 생성 → {name}_rl_cluster.json 저장.

- Base instance: 그대로 inference
- RAIN instance: rain 이벤트를 inst['tt']에 적용 후 inference
- ACC instance:  사고 이벤트를 inst['tt']에 적용 후 inference

Usage:
  python gen_test_rl_cluster.py --benchmark r1 \
      --resume result_soft_rl_deepseek/r102-r112/checkpoint-500.pt \
      --result-dir result_soft_rl_deepseek \
      --model deepseek-r1:32b --aug 8

  # 특정 instance만 처리
  python gen_test_rl_cluster.py --benchmark r1 \
      --resume result_soft_rl_deepseek/r102-r112/checkpoint-500.pt \
      --instances r104 r102_rain_a r107_acc_b
"""
import argparse, copy, json, os, sys, time, torch

_RL_FEWSHOT_INSTANCE = 'r106'  # R-type instance used as RL few-shot example

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from train_soft_cluster import (
    SoftClusterTrainer, env_params, model_params,
    optimizer_params, trainer_params, llm_params,
    _set_seed, _dump_cluster_cache, _load_model_flexible,
)
from SoftClusterLLMModule import get_all_clusters_confidence, build_rl_fewshot_block
from train_vrptw_llm import _get_rain_nodes_mult_evs, _get_accidents
import train_vrptw as _base_config

_TEST_INSTANCES = {
    'r1':  ['r101'],
    'rc1': ['rc101'],
    'c1':  ['c101'],
}
def _make_train_instances(prefix, ids):
    base = [f'{prefix}{i:03d}' for i in ids]
    rain = [f'{b}_rain_A' for b in base] + [f'{b}_rain_B' for b in base]
    acc  = [f'{b}_acc_A'  for b in base] + [f'{b}_acc_B'  for b in base]
    return base + rain + acc

_TRAIN_INSTANCES = {
    'r1':  _make_train_instances('r',  range(102, 113)),
    'rc1': _make_train_instances('rc', range(102, 109)),
    'c1':  _make_train_instances('c',  range(102, 110)),
}


def _apply_rain_to_inst(inst: dict) -> dict:
    """Return a deepcopy of inst with rain travel-time multiplier applied."""
    rain_nodes, mult, _ = _get_rain_nodes_mult_evs(inst)
    if not rain_nodes or mult <= 1.0:
        return inst
    modified = copy.deepcopy(inst)
    idx = torch.tensor(rain_nodes, dtype=torch.long)
    idx = idx[(idx >= 0) & (idx <= inst['n_customers'])]
    r = idx.unsqueeze(1).expand(-1, idx.numel())
    c = idx.unsqueeze(0).expand(idx.numel(), -1)
    mask = (r != c)
    modified['tt'][r[mask], c[mask]] *= mult
    return modified


def _apply_acc_to_inst(inst: dict) -> tuple[dict, float]:
    """Return (deepcopy with accident tt applied, cur_t).
    cur_t is the earliest accident trigger time (normalised).
    """
    accident_list = _get_accidents(inst)
    if not accident_list:
        return inst, 0.0
    modified = copy.deepcopy(inst)
    cur_t = min(fp['t_start'] for _, fp in accident_list)
    for acc_ev, feat_params in accident_list:
        na, nb = acc_ev.node_a, acc_ev.node_b
        mult   = feat_params['multiplier']
        modified['tt'][na, nb] = modified['tt'][na, nb] * mult
        modified['tt'][nb, na] = modified['tt'][nb, na] * mult
    return modified, cur_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark',  required=True, choices=['r1', 'rc1', 'c1'])
    parser.add_argument('--resume',     required=True)
    parser.add_argument('--result-dir', default='result_soft_rl_deepseek')
    parser.add_argument('--config',     default='F')
    parser.add_argument('--model',      default=llm_params['model'])
    parser.add_argument('--aug',        type=int, default=8)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--force',      action='store_true',
                        help='Overwrite existing rl_cluster files')
    parser.add_argument('--instances',  nargs='+', default=None,
                        help='Override instance list (searches test+train pool). e.g. --instances r104')
    parser.add_argument('--rl-fewshot', action='store_true',
                        help=f'Use {_RL_FEWSHOT_INSTANCE} RL cluster as few-shot example for LLM calls')
    parser.add_argument('--pomo',       type=int, required=True,
                        help='POMO rollout count, forced identical for LLM-on and --no-llm.')
    args = parser.parse_args()

    _set_seed(args.seed)

    cfg = _base_config._REWARD_CONFIGS[args.config]
    env_params.update(cfg)
    print(f'[Config {args.config}] {cfg}')

    tp = dict(trainer_params)
    tp['train_instances'] = _TRAIN_INSTANCES[args.benchmark]
    tp['test_instances']  = _TEST_INSTANCES[args.benchmark]
    tp['model_load']      = dict(enable=False, path=None, epoch=None)
    tp['result_dir'] = os.path.join(_HERE, args.result_dir)
    tp['pomo_size']  = args.pomo

    lp = dict(llm_params)
    lp['model'] = args.model

    trainer = SoftClusterTrainer(env_params, model_params, optimizer_params, tp, lp)

    # Load checkpoint
    ckpt_path = args.resume if os.path.isabs(args.resume) \
        else os.path.join(_HERE, args.resume)
    ckpt = torch.load(ckpt_path, map_location=trainer.device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    _load_model_flexible(trainer.model, state_dict)
    trainer.model.eval()
    print(f'[Loaded] {ckpt_path}  (epoch={ckpt.get("epoch","?")})')

    cache_dir = trainer._llm_cache_dir

    # DATA_DIR: same resolution as train_soft_cluster.py
    _data_dir = os.path.join(_HERE, '..', 'data', 'Solomon')

    # Build target pool
    if args.instances:
        all_pool = {i['name'].lower(): i for i in trainer.test_pool + trainer.instance_pool}
        target_pool = [all_pool[n.lower()] for n in args.instances if n.lower() in all_pool]
        missing = [n for n in args.instances if n.lower() not in all_pool]
        if missing:
            print(f'[WARN] instances not found in pool: {missing}')
    else:
        target_pool = trainer.test_pool + trainer.instance_pool

    # ── RL few-shot setup ─────────────────────────────────────────────────────
    # If --rl-fewshot: ensure _RL_FEWSHOT_INSTANCE is processed first so its
    # rl_cluster.json can be used as a few-shot example for all other instances.
    rl_fewshot_block = ""
    if args.rl_fewshot:
        _fs_path = os.path.join(cache_dir, f'{_RL_FEWSHOT_INSTANCE.upper()}_rl_cluster.json')
        if os.path.isfile(_fs_path) and not args.force:
            rl_fewshot_block = build_rl_fewshot_block(cache_dir, _data_dir, _RL_FEWSHOT_INSTANCE)
            print(f'[RL fewshot] Loaded pre-existing {_RL_FEWSHOT_INSTANCE} example')
        else:
            # Move _RL_FEWSHOT_INSTANCE to front of pool (base instance, no suffix)
            fs_key = _RL_FEWSHOT_INSTANCE.lower()
            fs_insts  = [i for i in target_pool if i['name'].lower() == fs_key]
            rest_pool = [i for i in target_pool if i['name'].lower() != fs_key]
            target_pool = fs_insts + rest_pool
            print(f'[RL fewshot] Will process {_RL_FEWSHOT_INSTANCE} first, then build few-shot block')

    for inst in target_pool:
        name = inst['name'].upper()

        rl_path = os.path.join(cache_dir, f'{name}_rl_cluster.json')
        if os.path.isfile(rl_path) and not args.force:
            print(f'[{name}] already exists — skip (use --force to overwrite)')
            continue

        # Detect instance type and apply event to tt
        name_up = name.upper()
        is_rain = '_RAIN_' in name_up
        is_acc  = '_ACC_'  in name_up
        cur_t   = None

        if is_rain:
            inf_inst = _apply_rain_to_inst(inst)
            print(f'\n[{name}] RAIN applied — aug×{args.aug} inference...')
        elif is_acc:
            inf_inst, cur_t = _apply_acc_to_inst(inst)
            print(f'\n[{name}] ACC applied (cur_t={cur_t:.3f}) — aug×{args.aug} inference...')
        else:
            inf_inst = inst
            print(f'\n[{name}] aug×{args.aug} inference...')

        # aug×N inference → best routes
        best_routes, best_reward = None, -float('inf')
        with torch.no_grad():
            for aug_idx in range(args.aug):
                aug_inst = trainer._aug_instance(inf_inst, aug_idx) if aug_idx > 0 else inf_inst
                routes, reward = trainer._best_solution(aug_inst)
                if reward > best_reward:
                    best_reward, best_routes = reward, routes

        if not best_routes:
            print(f'  [{name}] No routes — skipping')
            continue

        new_clusters = [
            [n for n in route if n != 0]
            for route in best_routes
            if any(n != 0 for n in route)
        ]
        K = len(new_clusters)
        print(f'  [{name}] {K} clusters from inference (reward={best_reward:.3f})')

        # LLM confidence
        flat_conf = None
        for attempt in range(2):
            try:
                kwargs = dict(
                    model=lp['model'], use_cot=lp['use_cot'],
                    use_ontology=lp['use_ontology'],
                    reward_config=lp.get('reward_config', 'F'),
                    bks_fewshot=rl_fewshot_block,
                )
                if cur_t is not None:
                    kwargs['cur_time'] = cur_t
                flat_conf = get_all_clusters_confidence(inst, new_clusters, **kwargs)
                break
            except Exception as e:
                print(f'  [{name}] LLM attempt {attempt+1}/2 failed ({e})'
                      + (', retrying...' if attempt == 0 else ', skipping.'))
                if attempt == 0:
                    time.sleep(10)

        if flat_conf is None:
            print(f'  [{name}] LLM failed — rl_cluster not saved')
            continue

        with open(rl_path, 'w', encoding='utf-8') as f:
            json.dump(_dump_cluster_cache(new_clusters, flat_conf), f)
        print(f'  [{name}] saved → {rl_path}')

        # Remove old Kim-based cache if present
        kim_path = os.path.join(cache_dir, f'{name}_cluster.json')
        if os.path.isfile(kim_path):
            os.remove(kim_path)
            print(f'  [{name}] removed old cache ({os.path.basename(kim_path)})')

        # Build RL few-shot block after _RL_FEWSHOT_INSTANCE is first processed
        if args.rl_fewshot and not rl_fewshot_block and name.lower() == _RL_FEWSHOT_INSTANCE.lower():
            rl_fewshot_block = build_rl_fewshot_block(cache_dir, _data_dir, _RL_FEWSHOT_INSTANCE)
            if rl_fewshot_block:
                print(f'[RL fewshot] Built few-shot block from {_RL_FEWSHOT_INSTANCE} — applying to remaining instances')


if __name__ == '__main__':
    main()
