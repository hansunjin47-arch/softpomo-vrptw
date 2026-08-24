"""
eval_events.py — Compare Pure POMO vs LLM+RL vs Ontology+LLM+RL
on HIGH-IMPACT (rain_A / acc_A) vs NO-IMPACT (rain_B / acc_B) scenarios.

Usage:
  python eval_events.py --instance c101 --mode rain   [--ckpt-pomo PATH] [--ckpt-llm PATH]
  python eval_events.py --instance c101 --mode acc
  python eval_events.py --instance c101 --mode rain --no-llm   # skip LLM models (offline)

Models evaluated (those with checkpoint paths supplied):
  pure_pomo      -- VRPTWTrainer checkpoint (trained without LLM)
  llm_rl         -- train_vrptw_llm checkpoint with use_ontology=False
  ontology_llm   -- train_vrptw_llm checkpoint with use_ontology=True

Output:
  Console table + result_llm/eval_events_<instance>_<mode>.csv
"""
from __future__ import annotations

import os
import math
import argparse
import csv

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

from VRPTWEnv import VRPTWEnv as Env
from VRPTWModel import VRPTWModel as Model
from VRPTWProblemDef import load_solomon, make_batch
from VRPTWTrainer import _make_rain_tokens, _make_acc_tokens, _get_rain_event

DATA_DIR   = os.path.join(_ROOT, "data", "Solomon")
RESULT_DIR = os.path.join(_HERE, "result_llm")

# ── default model hyperparams ──────────────────────────────────────────────────
_MODEL_PARAMS = dict(
    embedding_dim       = 128,
    sqrt_embedding_dim  = math.sqrt(128),
    encoder_layer_num   = 6,
    head_num            = 8,
    qkv_dim             = 16,
    ff_hidden_dim       = 512,
    eval_type           = 'argmax',
    logit_clipping      = 10.0,
)
_ENV_PARAMS = dict(
    problem_size    = 100,
    pomo_size       = 20,
    late_penalty    = 5.0,
    vehicle_penalty = 1.0,
)


# ── rollout helpers ────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_rollout(
    model: Model,
    env: Env,
    inst: dict,
    device: torch.device,
    *,
    rain_nodes:  list,
    rain_mult:   float,
    acc_evs:     list,
    start_nodes: list | None,
    max_steps:   int = 600,
) -> dict:
    """
    Single inference pass. Applies event TT modifications.
    Returns metrics dict.
    """
    pomo_size = len(start_nodes) if start_nodes else env.pomo_size
    N = inst['n_customers']

    batch = make_batch(inst, 1, device)
    env.pomo_size = pomo_size
    env.load_problems(batch)

    if rain_nodes:
        env.apply_rain(rain_nodes, rain_mult)

    reset_state, _, _ = env.reset()
    rain_evs = [e for e in inst.get('preset_events', []) if e['type'] == 'RAIN']
    reset_state.rain_tokens = _make_rain_tokens(inst, rain_evs, 1, device)
    reset_state.acc_tokens  = _make_acc_tokens([], N, 1, device)

    model.pre_forward(reset_state)
    state, reward, done = env.pre_step()

    # Step 1: start nodes
    if start_nodes is not None:
        start_t  = torch.tensor(start_nodes, dtype=torch.long, device=device)
        selected = start_t[None, :].expand(1, pomo_size)
        state, reward, done = env.step(selected)
    else:
        selected, _ = model(state)
        state, reward, done = env.step(selected)

    # Steps 2+: RL policy with accident TT + encoder token updates
    acc_applied  = [False] * len(acc_evs)
    acc_restored = [False] * len(acc_evs)
    active_accs  = []
    step = 1
    while not done and step < max_steps:
        cur_t = state.current_time.mean().item()
        acc_changed = False
        for i, ae in enumerate(acc_evs):
            if not acc_applied[i] and cur_t >= ae['t_start']:
                env.apply_accident(ae['node_a'], ae['node_b'], ae['multiplier'])
                active_accs.append(dict(
                    nodes=[ae['node_a'], ae['node_b']],
                    t_start=ae['t_start'], t_end=ae['t_end'],
                ))
                acc_applied[i] = True
                acc_changed = True
            elif acc_applied[i] and not acc_restored[i] and cur_t >= ae['t_end']:
                env.restore_accident(ae['node_a'], ae['node_b'])
                active_accs = [a for a in active_accs
                               if a['nodes'] != [ae['node_a'], ae['node_b']]]
                acc_restored[i] = True
                acc_changed = True

        if acc_changed:
            reset_state.acc_tokens = _make_acc_tokens(active_accs, N, 1, device)
            model.pre_forward(reset_state)

        selected, _ = model(state)
        state, reward, done = env.step(selected)
        step += 1

    if reward is None:
        reward = -env._get_travel_distance()

    # Best rollout (highest reward)
    best_idx    = int(reward[0].argmax().item())
    best_reward = float(reward[0, best_idx].item())
    distance    = float(env._get_travel_distance()[0, best_idx].item())
    total_late  = float(env.total_late[0, best_idx].item())
    n_vehicles  = int(env.depot_visit_count[0, best_idx].item())
    tw_violated = int((env.total_late[0] > 0).sum().item())  # across rollouts

    return dict(
        reward     = best_reward,
        distance   = distance,
        total_late = total_late,
        n_vehicles = n_vehicles,
        tw_violated_rollouts = tw_violated,  # how many pomo rollouts had any TW violation
        pomo_size  = pomo_size,
    )


def _load_model(ckpt_path: str | None, device: torch.device) -> Model | None:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None
    model = Model(**_MODEL_PARAMS).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def _events_from_scenario(inst: dict) -> tuple[list, float, list]:
    """Extract (rain_nodes, rain_mult, acc_evs) from preset_events in scenario file."""
    rain_evs = [e for e in inst.get('preset_events', []) if e['type'] == 'RAIN']
    raw_acc  = [e for e in inst.get('preset_events', []) if e['type'] == 'ACCIDENT']

    if rain_evs:
        rain_nodes = sorted({n for e in rain_evs for n in e['nodes']})
        rain_mult  = max(e['multiplier'] for e in rain_evs)
    else:
        rain_nodes, rain_mult = [], 1.0

    acc_evs = []
    T = float(inst['T'])
    for e in raw_acc:
        nodes = e['nodes']
        acc_evs.append(dict(
            node_a     = nodes[0] if len(nodes) >= 1 else 0,
            node_b     = nodes[1] if len(nodes) >= 2 else 0,
            multiplier = e['multiplier'],
            t_start    = e['trigger_time'] / max(T, 1.0),
            t_end      = (e['trigger_time'] + e['duration']) / max(T, 1.0),
        ))

    return rain_nodes, rain_mult, acc_evs


# ── main evaluation ────────────────────────────────────────────────────────────

def evaluate(
    instance:   str,
    mode:       str,   # 'rain' or 'acc'
    ckpt_pomo:  str | None,
    ckpt_llm:   str | None,
    ckpt_ont:   str | None,
    use_cuda:   bool = True,
    top_k:      int  = 20,
    max_steps:  int  = 600,
    no_llm:     bool = False,
):
    device = torch.device('cuda', 0) if use_cuda and torch.cuda.is_available() else torch.device('cpu')
    print(f'[Eval] device={device}  instance={instance}  mode={mode}')

    # ── Scenario files ───────────────────────────────────────────────────────
    suffix_high = f"_{mode}_A"
    suffix_no   = f"_{mode}_B"
    path_high = os.path.join(DATA_DIR, f'{instance}{suffix_high}.txt')
    path_no   = os.path.join(DATA_DIR, f'{instance}{suffix_no}.txt')
    path_base = os.path.join(DATA_DIR, f'{instance}.txt')

    if not os.path.isfile(path_high):
        print(f'[ERROR] {path_high} not found. Run generate_events.py first.')
        return
    if not os.path.isfile(path_no):
        print(f'[ERROR] {path_no} not found. Run generate_events.py first.')
        return

    inst_high = load_solomon(path_high)
    inst_no   = load_solomon(path_no)
    inst_base = load_solomon(path_base)

    rain_high, mult_high, acc_high = _events_from_scenario(inst_high)
    rain_no,   mult_no,   acc_no   = _events_from_scenario(inst_no)

    # For mode=rain: use rain events, no accident
    # For mode=acc:  use accident events, no rain
    if mode == 'rain':
        acc_high = []
        acc_no   = []
    else:  # acc
        rain_high, mult_high = [], 1.0
        rain_no,   mult_no   = [], 1.0

    # ── Load models ──────────────────────────────────────────────────────────
    env   = Env(**_ENV_PARAMS)
    m_pomo = _load_model(ckpt_pomo, device)
    m_llm  = _load_model(ckpt_llm,  device) if not no_llm else None
    m_ont  = _load_model(ckpt_ont,  device) if not no_llm else None

    if m_pomo is None:
        print('[WARN] No Pure POMO checkpoint; using random model weights')
        m_pomo = Model(**_MODEL_PARAMS).to(device)
        m_pomo.eval()

    # LLM start nodes (optional — requires API access)
    start_nodes_llm = None
    start_nodes_ont = None
    if not no_llm and (m_llm is not None or m_ont is not None):
        try:
            from VRPTWLLMModule import get_start_nodes, DEFAULT_MODEL
            from train_vrptw_llm import _preset_weather
            weather = _preset_weather(inst_base)
            start_nodes_llm = get_start_nodes(inst_base, top_k, DEFAULT_MODEL, weather,
                                              use_cot=False, use_ontology=False)
            start_nodes_ont = get_start_nodes(inst_base, top_k, DEFAULT_MODEL, weather,
                                              use_cot=False, use_ontology=True)
            print(f'[LLM] start_nodes obtained: llm={len(start_nodes_llm)}, ont={len(start_nodes_ont)}')
        except Exception as exc:
            print(f'[LLM] Could not get start nodes ({exc}); using default POMO starts')

    # ── Run evaluations ──────────────────────────────────────────────────────
    scenarios = [
        ('HIGH_IMPACT', inst_high, rain_high, mult_high, acc_high),
        ('NO_IMPACT',   inst_no,   rain_no,   mult_no,   acc_no),
        ('BASELINE',    inst_base, [],         1.0,       []),
    ]

    models = [
        ('Pure_POMO',     m_pomo, None),
        ('LLM+RL',        m_llm,  start_nodes_llm),
        ('Ontology+LLM',  m_ont,  start_nodes_ont),
    ]

    rows = []
    for sc_name, inst, rn, rm, ae in scenarios:
        for model_name, model, sn in models:
            if model is None:
                continue
            env.pomo_size = len(sn) if sn else _ENV_PARAMS['pomo_size']
            try:
                m = _run_rollout(
                    model, env, inst, device,
                    rain_nodes=rn, rain_mult=rm, acc_evs=ae,
                    start_nodes=sn, max_steps=max_steps,
                )
            except Exception as exc:
                print(f'  [SKIP] {sc_name}/{model_name}: {exc}')
                continue
            rows.append(dict(scenario=sc_name, model=model_name, **m))
            print(
                f'  {sc_name:12s} {model_name:15s} '
                f'reward={m["reward"]:7.3f}  dist={m["distance"]:7.3f}  '
                f'late={m["total_late"]:6.3f}  veh={m["n_vehicles"]:3d}'
            )

    # ── Print comparison table ───────────────────────────────────────────────
    print('\n' + '='*75)
    print(f'  IMPACT vs NO-IMPACT comparison  |  instance={instance}  mode={mode}')
    print('='*75)
    header = f"{'Scenario':12s}  {'Model':15s}  {'Reward':>8s}  {'Dist':>8s}  {'Late':>8s}  {'Veh':>4s}"
    print(header)
    print('-'*75)
    for r in rows:
        print(
            f"{r['scenario']:12s}  {r['model']:15s}  "
            f"{r['reward']:8.4f}  {r['distance']:8.4f}  "
            f"{r['total_late']:8.4f}  {r['n_vehicles']:4d}"
        )

    # Delta: HIGH - NO_IMPACT (bigger delta = more sensitive to events)
    print('\n  Impact sensitivity (HIGH - NO_IMPACT reward delta; more negative = worse under event):')
    for model_name, model, _ in models:
        if model is None:
            continue
        hi = next((r for r in rows if r['scenario']=='HIGH_IMPACT' and r['model']==model_name), None)
        no = next((r for r in rows if r['scenario']=='NO_IMPACT'   and r['model']==model_name), None)
        if hi and no:
            delta = hi['reward'] - no['reward']
            print(f'    {model_name:15s}  delta_reward={delta:+.4f}')
    print('='*75)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, f'eval_events_{instance}_{mode}.csv')
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f'\n  [Saved] {csv_path}')


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate event-scenario impact across models')
    parser.add_argument('--instance',   default='c101',  help='Solomon instance name (e.g. c101)')
    parser.add_argument('--mode',       default='rain',  choices=['rain', 'acc'])
    parser.add_argument('--ckpt-pomo',  default=None,    help='Pure POMO checkpoint path')
    parser.add_argument('--ckpt-llm',   default=None,    help='LLM+RL checkpoint path')
    parser.add_argument('--ckpt-ont',   default=None,    help='Ontology+LLM checkpoint path')
    parser.add_argument('--no-llm',     action='store_true', help='Skip LLM models (offline mode)')
    parser.add_argument('--top-k',      type=int, default=20)
    parser.add_argument('--max-steps',  type=int, default=600)
    parser.add_argument('--no-cuda',    action='store_true')
    args = parser.parse_args()

    # Try to resolve default checkpoint paths
    def _auto_ckpt(result_dir: str, tag: str) -> str | None:
        p = os.path.join(result_dir, f'checkpoint-{tag}.pt')
        return p if os.path.isfile(p) else None

    pomo_dir = os.path.join(_HERE, 'result')
    llm_dir  = os.path.join(_HERE, 'result_llm')

    ckpt_pomo = args.ckpt_pomo or _auto_ckpt(pomo_dir, 'last') or _auto_ckpt(pomo_dir, 'best')
    ckpt_llm  = args.ckpt_llm  or (_auto_ckpt(llm_dir, 'last') if not args.no_llm else None)
    ckpt_ont  = args.ckpt_ont  or ckpt_llm   # same file if ablation was same training run

    evaluate(
        instance  = args.instance,
        mode      = args.mode,
        ckpt_pomo = ckpt_pomo,
        ckpt_llm  = ckpt_llm,
        ckpt_ont  = ckpt_ont,
        use_cuda  = not args.no_cuda,
        top_k     = args.top_k,
        max_steps = args.max_steps,
        no_llm    = args.no_llm,
    )


if __name__ == '__main__':
    main()
