"""
extract_rl_routes.py
RL checkpoint에서 training instance best routes를 추출해 {name}_rl_cluster.json 생성.

K-means 없이 RL routing 자체를 cluster로 사용:
  RL이 찾은 route → vehicle별 node 묶음 → LLM confidence 호출 입력

Usage:
  python extract_rl_routes.py --benchmark r1 --config F
  python extract_rl_routes.py --benchmark c1 --config F
  python extract_rl_routes.py --benchmark rc1 --config F
"""
import os, sys, json, argparse, math, random
import torch

_HERE  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Legacy checkpoints from the pre-consolidation original_POMO sweep (config A-K
# pomo100 runs) still live under original_POMO/result/ -- not moved.
_LEGACY_RESULT_DIR = os.path.join(_HERE, '..', 'original_POMO', 'result')

from vrptw_env import VRPTWEnv, load_solomon, make_batch, _extract_routes
from train_vrptw import VRPTWModel, model_params, env_params, _get_rain_event, _load_model_flexible

# ── benchmark → train instances ───────────────────────────────────────────────
BENCH_TRAIN = {
    'c1':  [f'c{i:03d}' for i in range(102, 110)],
    'rc1': [f'rc{i:03d}' for i in range(102, 109)],
    'r1':  [f'r{i:03d}' for i in range(102, 113)],
}

# ── result dir pattern ────────────────────────────────────────────────────────
BENCH_TAG = {
    'c1':  'c102-c109',
    'rc1': 'rc102-rc108',
    'r1':  'r102-r112',
}

LLM_CACHE_DIR = os.path.join(_HERE, 'result_soft', 'llm_cache')


def _checkpoint_path(benchmark: str, config: str) -> str:
    tag   = BENCH_TAG[benchmark]
    rdir  = os.path.join(_LEGACY_RESULT_DIR, f'config_{config}_{benchmark}_pomo100', tag)
    ckpt  = os.path.join(rdir, 'checkpoint-last.pt')
    if not os.path.isfile(ckpt):
        # fallback: no benchmark suffix
        rdir = os.path.join(_LEGACY_RESULT_DIR, f'config_{config}_pomo100', tag)
        ckpt = os.path.join(rdir, 'checkpoint-last.pt')
    return ckpt


@torch.no_grad()
def extract_routes(inst: dict, model, env: VRPTWEnv,
                   pomo_size: int, device, max_steps: int = 600):
    """Run greedy inference (pomo_size starts, no aug) → best route."""
    env.pomo_size = pomo_size

    batch = make_batch(inst, 1, device)
    env.load_problems(batch)

    reset_state, _, _ = env.reset()
    # minimal rain/acc tokens
    reset_state.rain_tokens = torch.zeros(1, 0, 4, device=device)
    reset_state.acc_tokens  = torch.zeros(1, 0, 3, device=device)
    model.pre_forward(reset_state)

    state, reward, done = env.pre_step()
    step = 0
    while not done and step < max_steps:
        sel, _ = model(state)
        state, reward, done = env.step(sel)
        step += 1

    if reward is None:
        return None, float('-inf')

    best_idx  = int(reward[0].argmax().item())
    best_rew  = float(reward[0, best_idx].item())
    node_list = env.selected_node_list[0, best_idx].cpu().tolist()
    routes    = _extract_routes(node_list)   # list[list[int]], depot 0 제외
    return routes, best_rew


def routes_to_cluster_json(routes: list) -> dict:
    """Convert routes to {cluster_idx: {node: 1.0}} — confidence 1.0 placeholder."""
    return {
        str(k): {str(n): 1.0 for n in route if n != 0}
        for k, route in enumerate(routes)
        if any(n != 0 for n in route)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', required=True, choices=['c1', 'rc1', 'r1'])
    parser.add_argument('--config',    default='F')
    parser.add_argument('--pomo',      type=int, default=100)
    parser.add_argument('--out-dir',   default=None,
                        help='Output dir for _rl_cluster.json (default: result_soft/llm_cache)')
    parser.add_argument('--ckpt',      default=None,
                        help='Override checkpoint path (default: checkpoint-last.pt)')
    args = parser.parse_args()

    out_dir = args.out_dir or LLM_CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = args.ckpt or _checkpoint_path(args.benchmark, args.config)
    if not os.path.isfile(ckpt_path):
        print(f'[Error] checkpoint not found: {ckpt_path}')
        sys.exit(1)
    print(f'[Checkpoint] {ckpt_path}')

    # ── load model ────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)

    ep = dict(env_params)
    ep['pomo_size'] = args.pomo

    model = VRPTWModel(**model_params).to(device)
    _load_model_flexible(model, ckpt['model_state_dict'])
    model.eval()
    print(f'[Model] loaded (epoch={ckpt.get("epoch","?")})')

    env = VRPTWEnv(**ep)
    env.device = device

    data_dir  = os.path.join(_HERE, '..', 'data', 'Solomon')
    instances = BENCH_TRAIN[args.benchmark]

    for name in instances:
        sol_path = os.path.join(data_dir, f'{name.upper()}.txt')
        if not os.path.isfile(sol_path):
            print(f'  [{name}] file not found, skip')
            continue
        inst = load_solomon(sol_path)
        if inst is None:
            print(f'  [{name}] not found, skip')
            continue

        routes, reward = extract_routes(inst, model, env, args.pomo, device)
        if routes is None:
            print(f'  [{name}] inference failed, skip')
            continue

        n_vehicles = len([r for r in routes if r])
        print(f'  [{name}] reward={reward:.4f}  vehicles={n_vehicles}')

        cluster_json = routes_to_cluster_json(routes)
        out_path = os.path.join(out_dir, f'{name.upper()}_rl_cluster.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_json, f)
        print(f'           → {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
