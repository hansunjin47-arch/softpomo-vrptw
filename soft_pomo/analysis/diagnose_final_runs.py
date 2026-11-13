"""Read-only checkpoint/cache ablations; no training or LLM calls."""
import json
import pathlib
import sys
import types

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_vrptw as pure
import train_vrptw_llm as soft
from train_soft_cluster import SoftClusterTrainer, _parse_cluster_cache, _parse_cluster_depot
from vrptw_env import VRPTWEnv, load_solomon, _solution_stats


def trainer(family, pool, checkpoint, native):
    device = torch.device('cpu')
    obj = SoftClusterTrainer.__new__(SoftClusterTrainer)
    obj.device = device
    obj.trainer_params = dict(pomo_size=100, max_steps=600, start_mode='random')
    obj.llm_params = dict(enabled=True, bias_strength=0.0)
    cfg = {'r1': pure._REWARD_CONFIGS_R1, 'rc1': pure._REWARD_CONFIGS_RC1,
           'c1': pure._REWARD_CONFIGS}[family]['E']
    obj.env_params = dict(pure.env_params, **cfg)
    obj.env_params['pomo_size'] = 100
    obj.env = VRPTWEnv(**obj.env_params)
    cls = pure.VRPTWModel if native else soft.VRPTWModel
    obj.model = cls(**pure.model_params).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    ignored = sorted(set(state) - set(obj.model.state_dict()))
    obj.model.load_state_dict({k: v for k, v in state.items() if k not in ignored}, strict=True)
    if native:
        forward = obj.model.forward
        obj.model.forward = lambda state, llm_bias=None: forward(state)
    obj._cluster_assign_cache = {}
    obj._cluster_conf_cache = {}
    obj._acc_stage_events = {}
    obj._ensure_soft_cluster_cache = types.MethodType(lambda self, inst: None, obj)
    cache = ROOT / 'result_soft/llm_cache_final_nomisfit'
    prefix = family.replace('1', '101').upper()
    for suffix in ['', '_RAIN2', '_ACC2']:
        name = prefix + suffix
        key = prefix if suffix == '_ACC2' else name
        data = json.loads((cache / f'{key}_rl_cluster.json').read_text())
        clusters, scores = _parse_cluster_cache(data)
        obj._cluster_assign_cache[name] = clusters
        obj._cluster_conf_cache[name] = obj._build_cluster_conf_tensor(
            clusters, scores, len(clusters), 100, _parse_cluster_depot(data))
        if suffix == '_ACC2':
            stages = []
            for path in cache.glob(f'{name}_acc_refresh_*.json'):
                data = json.loads(path.read_text())
                _, values = _parse_cluster_cache(data)
                tensor = obj._build_cluster_conf_tensor(
                    clusters, values, len(clusters), 100, _parse_cluster_depot(data))
                stages.append(dict(t_start=float(path.stem.split('_')[-1]), tensor=tensor))
            obj._acc_stage_events[name] = sorted(stages, key=lambda x: x['t_start'])
    return obj, ignored


def evaluate(obj, name, beta, mode):
    soft._set_seed(42)
    obj.llm_params['bias_strength'] = beta
    obj.trainer_params['start_mode'] = mode
    inst = load_solomon(str(ROOT.parent / 'data/Solomon' / f'{name.lower()}.txt'))
    with torch.inference_mode():
        routes, reward = obj._best_solution(inst)
    stats = _solution_stats(inst, routes)
    best = int(obj.env.total_late.shape[1])
    return dict(instance=name, beta=beta, start_mode=mode, reward=round(reward, 5),
                vehicles=len(routes), served=stats['n_served'],
                distance=round(float(stats['total_dist']), 2),
                late=round(float(stats['total_late']), 2), late_count=stats['late_count'],
                steps=obj.env.selected_count, rollouts=best,
                completed_rollouts=int(obj.env.finished.sum().item()))


def main():
    torch.set_num_threads(4)
    results = []
    for family, pool in [('r1', 'r102-r112'), ('rc1', 'rc102-rc108'), ('c1', 'c102-c109')]:
        prefix = family.replace('1', '101').upper()
        pure_dir = ROOT / f'result/config_E_{family}_final_weightedsum_b32_aug8/{pool}'
        soft_dir = ROOT / f'result_soft/{pool}'
        cases = [
            ('pure500', pure_dir / 'checkpoint-500.pt', True, [0], ['']),
            ('pure500_without_slack', pure_dir / 'checkpoint-500.pt', False, [0], ['']),
            ('pure1000', pure_dir / 'checkpoint-1000.pt', True, [0], ['', '_RAIN2', '_ACC2']),
            ('soft1000', soft_dir / 'checkpoint-1000.pt', False, [0, 1, 5], ['', '_RAIN2', '_ACC2']),
        ]
        for label, path, native, betas, suffixes in cases:
            obj, ignored = trainer(family, pool, path, native)
            for beta in betas:
                for suffix in suffixes:
                    row = evaluate(obj, prefix + suffix, beta, 'random')
                    row.update(checkpoint=label, ignored_weights=ignored)
                    results.append(row)
                    print(json.dumps(row), flush=True)
            if label == 'soft1000':
                for suffix in ['', '_ACC2', '_RAIN2']:
                    row = evaluate(obj, prefix + suffix, 5, 'det_conf')
                    row.update(checkpoint=label, ignored_weights=ignored)
                    results.append(row)
                    print(json.dumps(row), flush=True)
        (ROOT / 'analysis/final_runs_diagnostics.json').write_text(
            json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
