"""
train_vrptw_llm.py — Proposed model: Ontology + LLM + RL for VRPTW.

Model: POMO with 8-D node features (x,y,dem,tw_o,tw_c,svc,rain_affected,acc_affected).
       Both rain and accident embedded in encoder as static node features.
       Rain set at episode start; accident triggers pre_forward re-call mid-episode.
       LLM provides event-aware start-node selection and accident priority guidance.

Run:
  python train_vrptw_llm.py
  python train_vrptw_llm.py --no-llm        # pure POMO ablation (same arch)
  python train_vrptw_llm.py --no-ontology   # LLM+RL ablation (no ontology context)
  python train_vrptw_llm.py --test-only
"""
from __future__ import annotations

import os
import sys
import math
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam as Optimizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_vrptw as _base_config
from train_vrptw import model_params, optimizer_params
from vrptw_env import (
    VRPTWEnv, load_solomon, make_batch,
    AverageMeter, _extract_routes, _print_solution, _save_solution, _load_sol,
    plot_routes, plot_training_curves,
)
from VRPTWLLMModule import (
    WeatherEvent, AccidentEvent,
    get_start_nodes, get_accident_priority,
    build_accident_prompt, query_llm, parse_priority,
    DEFAULT_MODEL,
)
from VRPTWOntology import VRPTWOntology as _VRPTWOntology

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ── Dataset ───────────────────────────────────────────────────────────────────
# train_vrptw.py의 TRAIN_INSTANCES 공유 (base + event variants)
from train_vrptw import TRAIN_INSTANCES as _TRAIN_INSTANCES
TRAIN_INSTANCES = _TRAIN_INSTANCES
TEST_INSTANCES  = ["c101", "c101_rain_A", "c101_rain_B", "c101_acc_A", "c101_acc_B"]

DATA_DIR   = os.path.join(_ROOT, "data", "Solomon")
RESULT_DIR = os.path.join(_HERE, "result_llm")

# ── Hyperparameters ───────────────────────────────────────────────────────────
# model_params, optimizer_params: imported from train_vrptw.py (fully shared)

# env_params: pomo_size here only applies when LLM is off (--no-llm) or top_k_auto
# is disabled -- when LLM top-k start selection is active, pomo_size_eff follows
# top_k at runtime instead (see pomo_size_eff assignments below).
env_params = dict(
    problem_size        = _base_config.env_params['problem_size'],
    pomo_size           = 100,
    late_count_penalty  = _base_config.env_params['late_count_penalty'],
    late_penalty        = _base_config.env_params['late_penalty'],
    vehicle_penalty     = _base_config.env_params['vehicle_penalty'],
    unserved_penalty    = _base_config.env_params['unserved_penalty'],
)

# trainer_params: shared keys from train_vrptw.py; LLM-specific keys overridden below
_SHARED_TRAINER_KEYS = (
    'use_cuda', 'cuda_device_num', 'epochs', 'train_episodes',
    'train_batch_size', 'test_batch_size', 'max_steps',
    'n_mc_samples', 'curriculum_epoch',
)
trainer_params = {
    **{k: _base_config.trainer_params[k] for k in _SHARED_TRAINER_KEYS},
    'data_dir':        DATA_DIR,
    'train_instances': TRAIN_INSTANCES,
    'test_instances':  TEST_INSTANCES,
    'result_dir':      RESULT_DIR,
    'model_load':      dict(enable=False, path=None, epoch=None),
    'logging':         dict(model_save_interval=500, log_interval=10),
}

llm_params = dict(
    enabled          = True,
    model            = DEFAULT_MODEL,
    top_k            = 25,   # LLM ranks top-K nodes; pomo_size follows top_k when LLM on
    top_k_accident   = 15,
    use_cot          = True,
    use_ontology     = True,
    bias_strength    = 5.0,  # additive logit bias scale for LLM priority scores
    refresh_interval = 0,    # periodic priority refresh every N vehicle dispatches (0 = off)
    top_k_auto       = False, # if True, set top_k = ceil(total_demand/capacity) per instance
)


# ── Model ─────────────────────────────────────────────────────────────────────
# Node features (8-D): x, y, demand, tw_open, tw_close, service, rain_affected, acc_affected
#   → rain and accident embedded in encoder as static node features
#   → rain set at episode start; accident triggers pre_forward re-call mid-episode
# Decoder context: last_node_emb || load || time  (emb+2)

class VRPTWModel(nn.Module):
    def __init__(self, **mp):
        super().__init__()
        self.mp            = mp
        self.encoder       = _Encoder(**mp)
        self.decoder       = _Decoder(**mp)
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        node_xy  = reset_state.node_xy
        demand   = reset_state.node_demand
        tw_open  = reset_state.node_tw_open
        tw_close = reset_state.node_tw_close
        service  = reset_state.node_service_time

        # 6-D node feature: (x, y, demand, tw_open, tw_close, service)
        node_features = torch.cat([
            node_xy,
            demand  [:, :, None],
            tw_open [:, :, None],
            tw_close[:, :, None],
            service [:, :, None],
        ], dim=2)   # (batch, N, 6)

        rain_tokens = reset_state.rain_tokens   # (batch, R, 4) or None
        acc_tokens  = reset_state.acc_tokens    # (batch, A, 5) or None

        self.encoded_nodes = self.encoder(depot_xy, node_features,
                                          rain_tokens, acc_tokens)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state, llm_bias=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size  = state.BATCH_IDX.size(1)
        dev        = state.BATCH_IDX.device

        if state.selected_count == 0:
            # Always start at depot
            selected = torch.zeros((batch_size, pomo_size), dtype=torch.long, device=dev)
            prob     = torch.ones((batch_size, pomo_size), device=dev)

        elif state.selected_count == 1 and llm_bias is None:
            # Sample once per episode, shared across all batch instances
            n_customers = state.ninf_mask.size(-1) - 1
            starts = torch.randperm(n_customers, device=dev)[:pomo_size] + 1
            selected = starts[None, :].expand(batch_size, pomo_size)
            prob     = torch.ones((batch_size, pomo_size), device=dev)

        else:
            # LLM mode (step 1+) or standard mode (step 2+): RL decoder with optional LLM bias
            encoded_last = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(encoded_last, state.load, state.current_time,
                                 ninf_mask=state.ninf_mask, llm_bias=llm_bias)

            if self.training or self.mp['eval_type'] == 'softmax':
                while True:
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1) \
                                       .multinomial(1).squeeze(1) \
                                       .reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                               .reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index):
    b, p  = node_index.shape
    emb   = encoded_nodes.size(2)
    idx   = node_index[:, :, None].expand(b, p, emb)
    return encoded_nodes.gather(dim=1, index=idx)


class _Encoder(nn.Module):
    def __init__(self, **mp):
        super().__init__()
        emb = mp['embedding_dim']
        self.embedding_depot = nn.Linear(2, emb)   # depot: (x, y)
        self.embedding_node  = nn.Linear(6, emb)   # node: (x,y,demand,tw_open,tw_close,service)
        self.embedding_rain  = nn.Linear(4, emb)   # rain token: (node/N, mm/100, t_s/T, t_e/T)
        self.embedding_acc   = nn.Linear(3, emb)   # acc  token: (node/N, t_s/T, t_e/T)
        self.layers = nn.ModuleList([_EncoderLayer(**mp) for _ in range(mp['encoder_layer_num'])])

    def forward(self, depot_xy, node_features, rain_tokens=None, acc_tokens=None):
        emb_depot = self.embedding_depot(depot_xy)
        emb_node  = self.embedding_node(node_features)
        seq = torch.cat((emb_depot, emb_node), dim=1)
        if rain_tokens is not None and rain_tokens.size(1) > 0:
            seq = torch.cat((seq, self.embedding_rain(rain_tokens)), dim=1)
        if acc_tokens is not None and acc_tokens.size(1) > 0:
            seq = torch.cat((seq, self.embedding_acc(acc_tokens)), dim=1)
        for layer in self.layers:
            seq = layer(seq)
        return seq[:, :emb_node.size(1) + 1, :]  # routing nodes only


class _EncoderLayer(nn.Module):
    def __init__(self, **mp):
        super().__init__()
        emb  = mp['embedding_dim']
        h    = mp['head_num']
        d    = mp['qkv_dim']
        ff   = mp['ff_hidden_dim']
        self.head_num = h
        self.Wq      = nn.Linear(emb, h * d, bias=False)
        self.Wk      = nn.Linear(emb, h * d, bias=False)
        self.Wv      = nn.Linear(emb, h * d, bias=False)
        self.combine = nn.Linear(h * d, emb)
        self.norm1   = _Norm(emb)
        self.ff      = _FF(emb, ff)
        self.norm2   = _Norm(emb)

    def forward(self, x):
        q  = _reshape(self.Wq(x), self.head_num)
        k  = _reshape(self.Wk(x), self.head_num)
        v  = _reshape(self.Wv(x), self.head_num)
        mh = self.combine(_mha(q, k, v))
        x  = self.norm1(x, mh)
        x  = self.norm2(x, self.ff(x))
        return x


class _Decoder(nn.Module):
    def __init__(self, **mp):
        super().__init__()
        emb  = mp['embedding_dim']
        h    = mp['head_num']
        d    = mp['qkv_dim']
        self.head_num       = h
        self.sqrt_emb       = math.sqrt(emb)
        self.logit_clipping = mp['logit_clipping']
        # context: last_node_emb(emb) + load(1) + time(1) = emb+2
        self.Wq_last = nn.Linear(emb + 2, h * d, bias=False)
        self.Wk      = nn.Linear(emb,     h * d, bias=False)
        self.Wv      = nn.Linear(emb,     h * d, bias=False)
        self.combine = nn.Linear(h * d, emb)
        self.k = self.v = self.single_head_key = None

    def set_kv(self, encoded_nodes):
        self.k               = _reshape(self.Wk(encoded_nodes), self.head_num)
        self.v               = _reshape(self.Wv(encoded_nodes), self.head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last, load, current_time, ninf_mask, llm_bias=None):
        ctx = torch.cat([
            encoded_last,
            load         [:, :, None],
            current_time [:, :, None],
        ], dim=2)   # (batch, pomo, emb+2)

        q      = _reshape(self.Wq_last(ctx), self.head_num)
        mh_out = self.combine(_mha(q, self.k, self.v, rank3_ninf_mask=ninf_mask))
        score  = torch.matmul(mh_out, self.single_head_key) / self.sqrt_emb
        score  = self.logit_clipping * torch.tanh(score) + ninf_mask
        if llm_bias is not None:
            score = score + llm_bias   # soft additive bias — RL can still override
        return F.softmax(score, dim=2)


def _reshape(qkv, h):
    b, n, _ = qkv.shape
    return qkv.reshape(b, n, h, -1).transpose(1, 2)


def _mha(q, k, v, rank3_ninf_mask=None):
    b, h, n, d = q.shape
    score = torch.matmul(q, k.transpose(2, 3)) / (d ** 0.5)
    if rank3_ninf_mask is not None:
        score = score + rank3_ninf_mask[:, None, :, :]
    w   = torch.softmax(score, dim=3)
    out = torch.matmul(w, v)
    return out.transpose(1, 2).reshape(b, n, h * d)


class _Norm(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.norm = nn.InstanceNorm1d(emb, affine=True, track_running_stats=False)

    def forward(self, x, res):
        return self.norm((x + res).transpose(1, 2)).transpose(1, 2)


class _FF(nn.Module):
    def __init__(self, emb, ff):
        super().__init__()
        self.W1 = nn.Linear(emb, ff)
        self.W2 = nn.Linear(ff, emb)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))


# ── Event helpers ─────────────────────────────────────────────────────────────

def _get_weather(inst: dict) -> WeatherEvent | None:
    """WeatherEvent kept for API compat; weather detail now read by ontology directly."""
    rain_evs = [e for e in inst.get('preset_events', []) if e['type'] == 'RAIN']
    if not rain_evs:
        return None
    all_nodes = sorted({n for e in rain_evs for n in e['nodes']})
    start_t   = min(e['trigger_time'] for e in rain_evs)
    end_t     = max(e['trigger_time'] + e['duration'] for e in rain_evs)
    avg_mm    = sum(e['rainfall_mm'] for e in rain_evs) / len(rain_evs)
    return WeatherEvent(affected_nodes=all_nodes, start_time=start_t,
                        end_time=end_t, rainfall_mm=avg_mm)


def _make_rain_tokens(inst: dict, rain_evs: list, batch_size: int, device) -> torch.Tensor:
    """(batch, R, 4) — one token per (event, stop): [node/N, mm/100, t_s/T, t_e/T]."""
    T, N = float(inst['T']), inst['n_customers']
    rows = []
    for ev in rain_evs:
        mm  = min(ev.get('rainfall_mm', 0.0) / 100.0, 1.0)
        t_s = ev.get('trigger_time', 0.0) / max(T, 1.0)
        t_e = (ev.get('trigger_time', 0.0) + ev.get('duration', 0.0)) / max(T, 1.0)
        for n in ev.get('nodes', []):
            if 1 <= n <= N:
                rows.append([n / max(N, 1), mm, t_s, t_e])
    if not rows:
        return torch.zeros(batch_size, 0, 4, device=device)
    t = torch.tensor(rows, dtype=torch.float32, device=device)
    return t.unsqueeze(0).expand(batch_size, -1, -1)


def _make_acc_tokens(active_accs: list, N: int, batch_size: int, device) -> torch.Tensor:
    """(batch, A_total, 3) — one token per (accident, node): [node/N, t_s/T, t_e/T]."""
    if not active_accs:
        return torch.zeros(batch_size, 0, 3, device=device)
    rows = []
    for a in active_accs:
        t_s, t_e = a['t_start'], a['t_end']
        for n in a['nodes']:
            if 1 <= n <= N:
                rows.append([n / N, t_s, t_e])
    if not rows:
        return torch.zeros(batch_size, 0, 3, device=device)
    t = torch.tensor(rows, dtype=torch.float32, device=device)
    return t.unsqueeze(0).expand(batch_size, -1, -1)


def _get_rain_nodes_mult(inst: dict, weather: WeatherEvent | None) -> tuple[list, float]:
    rain_evs = [e for e in inst.get('preset_events', []) if e['type'] == 'RAIN']
    if not rain_evs:
        return [], 1.0
    nodes = sorted({n for e in rain_evs for n in e['nodes']})
    mult  = max(e['multiplier'] for e in rain_evs)
    return nodes, mult


def _get_rain_nodes_mult_evs(inst: dict) -> tuple[list, float, list]:
    """Returns (rain_nodes, max_multiplier, rain_evs_list) from preset_events."""
    rain_evs = [e for e in inst.get('preset_events', []) if e['type'] == 'RAIN']
    if not rain_evs:
        return [], 1.0, []
    nodes = sorted({n for e in rain_evs for n in e['nodes']})
    mult  = max(e['multiplier'] for e in rain_evs)
    return nodes, mult, rain_evs


def _get_accidents(inst: dict) -> list[tuple[AccidentEvent, dict]]:
    """Returns list of (AccidentEvent for LLM, feat_params dict for env only).
    multiplier is kept in feat_params and NEVER passed to AccidentEvent / LLM.
    description (e.g. '4-car-pile-up') is stored in AccidentEvent and shown to LLM.
    """
    acc_evs = [e for e in inst.get('preset_events', []) if e['type'] == 'ACCIDENT']
    T = float(inst['T'])
    result = []
    for e in acc_evs:
        nodes   = e['nodes']
        mult    = e['multiplier']
        desc    = e.get('severity', 'high')   # new: n-car description (or legacy label)
        t_start = e['trigger_time'] / max(T, 1.0)
        t_end   = (e['trigger_time'] + e['duration']) / max(T, 1.0)
        accident = AccidentEvent(
            node_a=nodes[0] if len(nodes) >= 1 else 0,
            node_b=nodes[1] if len(nodes) >= 2 else 0,
            severity=desc,                                 # LLM-visible description
            affected_nodes=nodes,
        )
        feat_params = dict(
            multiplier=mult,                               # env-internal only
            t_start=t_start,
            t_end=t_end,
        )
        result.append((accident, feat_params))
    return result


# ── Trainer ───────────────────────────────────────────────────────────────────

def _inst_tag(instances: list) -> str:
    """Base instance names only (event variants 제외) -> 'c102-c109'"""
    base = sorted(set(n.split('_')[0] for n in instances))
    return f"{base[0]}-{base[-1]}" if len(base) > 1 else base[0]


class VRPTWLLMTrainer:
    def __init__(self, env_p, model_p, opt_p, trainer_p, llm_p):
        self.env_params       = env_p
        self.model_params     = model_p
        self.optimizer_params = opt_p
        self.trainer_params   = trainer_p
        self.llm_params       = llm_p

        USE_CUDA = trainer_p.get('use_cuda', torch.cuda.is_available())
        if USE_CUDA:
            dev_num = trainer_p.get('cuda_device_num', 0)
            torch.cuda.set_device(dev_num)
            self.device = torch.device('cuda', dev_num)
        else:
            self.device = torch.device('cpu')

        self.model     = VRPTWModel(**model_p).to(self.device)
        self.env       = VRPTWEnv(**env_p)
        self.optimizer = Optimizer(self.model.parameters(), **opt_p['optimizer'])

        data_dir = trainer_p['data_dir']
        self.instance_pool = []
        self.base_pool     = []
        for iname in trainer_p['train_instances']:
            inst = load_solomon(os.path.join(data_dir, iname + '.txt'))
            self.instance_pool.append(inst)
            if not inst['preset_events']:
                self.base_pool.append(inst)
            ev_tag = f"  {len(inst['preset_events'])} event(s)" if inst['preset_events'] else ""
            print(f'  [Data] {inst["name"]}  N={inst["n_customers"]}{ev_tag}')

        # Weighted sampling by instance type (base / rain / acc)
        type_ratio = trainer_p.get('type_ratio', None)  # e.g. {'base':0.5,'rain':0.4,'acc':0.1}
        if type_ratio:
            n_base = sum(1 for i in self.instance_pool if '_rain_' not in i['name'].lower() and '_acc_' not in i['name'].lower())
            n_rain = sum(1 for i in self.instance_pool if '_rain_' in i['name'].lower())
            n_acc  = sum(1 for i in self.instance_pool if '_acc_'  in i['name'].lower())
            def _w(inst):
                name = inst['name'].lower()
                if '_acc_' in name:  return type_ratio.get('acc',  0.1) / max(n_acc,  1)
                if '_rain_' in name: return type_ratio.get('rain', 0.4) / max(n_rain, 1)
                return                      type_ratio.get('base', 0.5) / max(n_base, 1)
            self._pool_weights = [_w(i) for i in self.instance_pool]
            print(f'  [TypeRatio] base={type_ratio.get("base")} ({n_base}) '
                  f'rain={type_ratio.get("rain")} ({n_rain}) acc={type_ratio.get("acc")} ({n_acc})')
        else:
            self._pool_weights = None

        self.curriculum_epoch = trainer_p.get('curriculum_epoch', 0)
        if self.curriculum_epoch > 0:
            print(f'  [Curriculum] Phase 1: base only (1-{self.curriculum_epoch})'
                  f' → Phase 2: full (>{self.curriculum_epoch})')

        self.test_pool  = []
        self.test_paths = []
        for iname in trainer_p.get('test_instances', []):
            path = os.path.join(data_dir, iname + '.txt')
            inst = load_solomon(path)
            self.test_pool.append(inst)
            self.test_paths.append(path)
            print(f'  [Test] {inst["name"]}')

        self.start_epoch = 1
        model_load = trainer_p.get('model_load', {'enable': False})
        if model_load.get('enable'):
            ckpt = torch.load(f'{model_load["path"]}/checkpoint-{model_load["epoch"]}.pt',
                              map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            loaded_epoch = ckpt.get('epoch', model_load['epoch'])
            self.start_epoch = (int(loaded_epoch) if str(loaded_epoch).isdigit() else loaded_epoch) + 1
            print(f'[Resume] epoch {self.start_epoch}')

        base_result_dir = trainer_p.get('result_dir', 'result_llm')
        inst_tag        = _inst_tag(trainer_p['train_instances'])
        self.result_dir = os.path.join(base_result_dir, inst_tag)
        os.makedirs(self.result_dir, exist_ok=True)
        self.reward_log = []
        self.loss_log  = []

        # LLM decision cache (disk-backed: compute once, reuse across runs)
        self._start_nodes_cache: dict[str, list[int]] = {}
        self._inst_top_k: dict[str, int] = {}   # per-instance effective top_k (auto mode)
        self._llm_cache_dir = trainer_p.get('llm_cache_dir') or os.path.join(base_result_dir, 'llm_cache')
        os.makedirs(self._llm_cache_dir, exist_ok=True)

        self._force_llm_refresh = trainer_p.get('force_llm_refresh', False)
        if llm_p['enabled']:
            _test_only = trainer_p.get('test_only', False)
            all_insts = self.test_pool if _test_only else self.instance_pool + self.test_pool
            # Process base instances first so acc can reference them
            base_insts  = [i for i in all_insts if '_' not in i['name']]
            event_insts = [i for i in all_insts if '_' in i['name']]
            print(f'  [LLM] Loading/computing start nodes '
                  f'({len(base_insts)} base first, then {len(event_insts)} event)...')
            for inst in base_insts + event_insts:
                self._ensure_llm_cache(inst, force_refresh=self._force_llm_refresh)

    # ------------------------------------------------------------------
    def run(self):
        lp     = self.llm_params
        llm_on = lp['enabled']

        if _MLFLOW:
            exp_name = "SoftPOMO_proposed" if llm_on else "SoftPOMO_RL"
            run_name = ("ontology_llm" if llm_on and lp.get('use_ontology')
                        else "llm_only" if llm_on
                        else "no_llm")
            mlflow.set_experiment(exp_name)
            run_ctx = mlflow.start_run(run_name=run_name)
            run_ctx.__enter__()
            tp = self.trainer_params
            mlflow.log_params({
                "epochs": tp['epochs'], "train_episodes": tp['train_episodes'],
                "train_batch_size": tp['train_batch_size'],
                "pomo_size": self.env_params['pomo_size'],
                "problem_size": self.env_params['problem_size'],
                "late_penalty": self.env_params.get('late_penalty', 1.0),
                "vehicle_penalty": self.env_params.get('vehicle_penalty', 0.0),
                "lr": self.optimizer_params['optimizer']['lr'],
                "embedding_dim": self.model_params['embedding_dim'],
                "encoder_layers": self.model_params['encoder_layer_num'],
                "llm_enabled": llm_on,
                "use_ontology": lp.get('use_ontology', False),
                "top_k": lp.get('top_k', 0),
                "llm_model": lp.get('model', ''),
                "train_instances": ",".join(tp['train_instances']),
                "test_instances":  ",".join(tp.get('test_instances', [])),
            })

        t0 = time.time()
        train_reward_history = []
        loss_history         = []

        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            train_reward, train_loss = self._train_one_epoch(epoch)
            train_reward_history.append(train_reward)
            loss_history.append(train_loss)
            self.reward_log.append(train_reward)
            self.loss_log.append(train_loss)

            llm_tag = "" if llm_on else " [no-llm]"
            total_epochs = self.trainer_params['epochs']
            log_interval = max(1, total_epochs // 10)
            if epoch % log_interval == 0 or epoch == total_epochs:
                elapsed = time.time() - t0
                h, m = divmod(int(elapsed), 3600)
                m, s = divmod(m, 60)
                print(f'Epoch {epoch:4d}/{total_epochs}{llm_tag}  '
                      f'train_reward={train_reward:.4f}  loss={train_loss:.6f}  '
                      f'elapsed={h:02d}:{m:02d}:{s:02d}')

            if _MLFLOW:
                mlflow.log_metrics({"train/reward": train_reward,
                                    "train/loss":   train_loss}, step=epoch)

            save_interval = self.trainer_params['logging'].get('model_save_interval', 500)
            if epoch % save_interval == 0:
                self._save_ckpt(epoch)

        self._save_ckpt(self.trainer_params['epochs'], tag='last')
        total_min = (time.time() - t0) / 60
        print(f'\n*** Training Done ***  total={total_min:.1f}min  '
              f'avg_epoch={total_min/max(1, self.trainer_params["epochs"]-self.start_epoch+1)*60:.1f}s/epoch')

        plots_dir  = os.path.join(self.result_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_training_curves(train_reward_history, loss_history, plots_dir)

        if self.test_pool:
            self.model.eval()
            final_test_reward, per_inst = self._eval_test()
            print(f'  [Final Eval] test_reward={final_test_reward:.4f}')
            if _MLFLOW:
                mlflow.log_metric("test/reward_final", final_test_reward)
                for name, r in per_inst:
                    mlflow.log_metric(f"test/reward_{name}", r)
            for inst, data_path in zip(self.test_pool, self.test_paths):
                routes, best_reward = self._best_solution(inst)
                sol_routes = _load_sol(data_path, inst['n_customers'])
                _print_solution(inst, routes, best_reward, sol_routes)
                route_path = os.path.join(plots_dir, f"{inst['name']}_best_route.png")
                sol_path   = os.path.join(plots_dir, f"{inst['name']}_solution.txt")
                plot_routes(inst, routes, best_reward, route_path)
                _save_solution(inst, routes, best_reward, sol_path, sol_routes)

        if _MLFLOW:
            ckpt_path = os.path.join(self.result_dir, 'checkpoint-last.pt')
            if os.path.isfile(ckpt_path):
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
            for fname in ("train_reward.png", "test_reward.png", "loss.png"):
                p = os.path.join(plots_dir, fname)
                if os.path.isfile(p):
                    mlflow.log_artifact(p, artifact_path="plots")
            run_ctx.__exit__(None, None, None)

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _train_one_batch(self, inst: dict, batch_size: int):
        self.model.train()
        lp     = self.llm_params
        llm_on = lp['enabled']

        # ── Event setup ───────────────────────────────────────────────────
        rain_nodes, rain_mult, rain_evs = _get_rain_nodes_mult_evs(inst)
        if llm_on:
            self._ensure_llm_cache(inst)
        accident_list = _get_accidents(inst) if llm_on else []
        N             = self.env.problem_size
        if llm_on and lp.get('top_k_auto', False):
            pomo_size_eff = self._inst_top_k.get(
                inst['name'],
                math.ceil(float(inst['node_demand'].sum().item()))
            )
        else:
            pomo_size_eff = self.env_params['pomo_size']
        self.env.pomo_size = pomo_size_eff

        batch = make_batch(inst, batch_size, self.device)
        self.env.load_problems(batch)
        if rain_nodes:
            self.env.apply_rain(rain_nodes, rain_mult)

        reset_state, _, _ = self.env.reset()
        reset_state.rain_tokens = _make_rain_tokens(inst, rain_evs, batch_size, self.device)
        reset_state.acc_tokens  = _make_acc_tokens([], N, batch_size, self.device)
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(batch_size, pomo_size_eff, 0, device=self.device)
        state, reward, done = self.env.pre_step()

        # ── Step 0: depot (model always returns depot, no bias needed) ───
        selected, prob = self.model(state)   # selected_count=0 → depot
        state, reward, done = self.env.step(selected)
        prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)  # prob=1, log=0

        # ── Build initial LLM bias from cached priority list ─────────────
        # All subsequent steps (vehicle 1 first stop, all depot returns) use this bias.
        # bias_t=None → no-LLM mode falls through to standard POMO arange at step 1.
        bias_t = self._build_bias_tensor(inst) if llm_on else None

        # ── Steps 1+: RL decoder with soft LLM bias ──────────────────────
        max_steps        = self.trainer_params.get('max_steps', 600)
        refresh_interval = lp.get('refresh_interval', 0)
        acc_tt_applied   = [False] * len(accident_list)
        acc_tt_restored  = [False] * len(accident_list)
        active_accs      = []
        vehicles_dispatched = 0
        prev_dvc            = 0   # tracks depot_visit_count[0,0]

        # ── Step 1: Deterministic first-customer selection (LLM mode) ─────
        # rollout i starts at top_k[i]; no sampling, prob=1 (no gradient here)
        if llm_on:
            selected = self._det_starts(inst, pomo_size_eff).expand(batch_size, -1)
            prob     = torch.ones(batch_size, pomo_size_eff, device=self.device)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            step = 2
        else:
            step = 1

        while not done and step < max_steps:
            cur_t = state.current_time.mean().item()

            # ── Accident TT modification + encoder re-encode ──────────────
            acc_changed      = False
            acc_just_started = []
            for i, (accident, fp) in enumerate(accident_list):
                if not acc_tt_applied[i] and cur_t >= fp['t_start']:
                    self.env.apply_accident(accident.node_a, accident.node_b, fp['multiplier'])
                    active_accs.append(dict(
                        nodes=[accident.node_a, accident.node_b],
                        t_start=fp['t_start'], t_end=fp['t_end'],
                    ))
                    acc_tt_applied[i]  = True
                    acc_changed        = True
                    acc_just_started.append(accident)
                elif acc_tt_applied[i] and not acc_tt_restored[i] and cur_t >= fp['t_end']:
                    self.env.restore_accident(accident.node_a, accident.node_b)
                    active_accs = [a for a in active_accs
                                   if a['nodes'] != [accident.node_a, accident.node_b]]
                    acc_tt_restored[i] = True
                    acc_changed        = True

            if acc_changed:
                self.env.reset_state.acc_tokens = _make_acc_tokens(active_accs, N, batch_size, self.device)
                self.model.pre_forward(self.env.reset_state)

            # ── Accident-triggered LLM priority refresh ──────────────────────────
            if llm_on and acc_just_started:
                unvisited = self._get_representative_unvisited(N)
                if unvisited:
                    t0 = time.time()
                    self._refresh_priority_accident(inst, unvisited, acc_just_started[-1])
                    bias_t = self._build_bias_tensor(inst, unvisited)
                    print(f"  [LLM:acc_train] {inst['name']} refresh took {time.time()-t0:.1f}s", flush=True)

            # ── Decoder step: bias only when rollout is at depot (dispatch moment) ──
            if llm_on and bias_t is not None:
                at_depot  = (state.current_node == 0).float().unsqueeze(-1)  # (B,P,1)
                bias_now  = bias_t * at_depot  # (1,1,N+1) * (B,P,1) → (B,P,N+1)
                selected, prob = self.model(state, llm_bias=bias_now)
            else:
                selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            step += 1

            # ── Detect depot returns → periodic LLM refresh ───────────────
            if llm_on and refresh_interval > 0:
                cur_dvc = int(self.env.depot_visit_count[0, 0].item())
                newly   = cur_dvc - prev_dvc
                if newly > 0:
                    vehicles_dispatched += newly
                    prev_dvc = cur_dvc
                    if vehicles_dispatched % refresh_interval == 0:
                        unvisited = self._get_representative_unvisited(N)
                        if unvisited:
                            self._refresh_priority(inst, unvisited)
                            bias_t = self._build_bias_tensor(inst, unvisited)

        if reward is None:
            reward = self._fallback_reward()

        advantage = reward - reward.float().mean(dim=1, keepdim=True)
        advantage = advantage / (advantage.std(dim=1, keepdim=True).clamp(min=1e-6))
        log_prob  = prob_list.log().sum(dim=2)
        loss      = -(advantage * log_prob).mean()

        self.model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        max_pomo_reward, _ = reward.max(dim=1)
        return max_pomo_reward.float().mean().item(), loss.item()

    def _update_episode_stats(self, inst: dict):
        """Update cross-episode stats using best POMO rollout's outcome."""
        from VRPTWOntology import episode_tracker
        N = inst['n_customers']
        # best pomo agent (index 0 of batch, best pomo rollout)
        best_pomo = int(self.env.total_late[0].argmax().item())   # most informative rollout

        visited_flags = self.env.visited_ninf_flag[0, best_pomo, 1:].cpu().numpy()
        unserved = [c for c in range(1, N + 1) if visited_flags[c - 1] != float('-inf')]

        late_flags = self.env.total_late[0, best_pomo].item()
        # per-stop late: re-derive from node list
        node_list = self.env.selected_node_list[0, best_pomo].cpu().tolist()
        late_ids  = []
        cur_t, prev = 0.0, 0
        tt_raw = inst['tt'].numpy() * inst['T']
        tw_o_raw = self.env.depot_node_tw_close[0].cpu().numpy() * 0   # dummy
        # use env tensors for tw_close
        tw_close_raw = self.env.depot_node_tw_close[0].cpu().numpy() * float(inst['T'])
        for n in node_list:
            if n == 0:
                cur_t = 0.0; prev = 0; continue
            arr = cur_t + float(tt_raw[prev, n])
            if arr > tw_close_raw[n] + 1e-6:
                late_ids.append(n)
            cur_t = max(arr, self.env.depot_node_tw_open[0, n].item() * float(inst['T']))
            prev = n

        episode_tracker.update(inst['name'], N, unserved, late_ids)

    def _ensure_llm_cache(self, inst: dict, force_refresh: bool = False):
        """Load LLM start-node priority from disk cache; call LLM only on first encounter."""
        import json
        name = inst['name']
        if name in self._start_nodes_cache and not force_refresh:
            return
        lp = self.llm_params

        cache_path = os.path.join(self._llm_cache_dir, f'{name}.json')
        if os.path.isfile(cache_path) and not force_refresh:
            data = json.load(open(cache_path))
            self._start_nodes_cache[name] = data['start_nodes']
            print(f'  [LLM cache] loaded {name}')
            return

        # Compute effective top_k for this instance
        if lp.get('top_k_auto', False):
            import math
            eff_k = math.ceil(float(inst['node_demand'].sum().item()))
            self._inst_top_k[name] = eff_k
            print(f'  [LLM:auto-k] {name}: n_min_veh={eff_k}')
        else:
            eff_k = lp['top_k']

        # For acc instances: reuse base instance's start nodes (accident unknown at dispatch)
        is_acc = 'ACC' in inst['name'].upper()
        if is_acc:
            base_key = inst['name'].split('_')[0]   # e.g. "C102" from "C102_ACC_A"
            base_nodes = self._start_nodes_cache.get(base_key)
            if base_nodes:
                nodes = base_nodes
                if lp.get('top_k_auto', False):
                    self._inst_top_k[name] = self._inst_top_k.get(base_key, eff_k)
                print(f'  [LLM] {inst["name"]}: reusing {base_key} start nodes (acc unknown at dispatch)')
            else:
                nodes = get_start_nodes(inst, eff_k, lp['model'],
                                        use_cot=lp['use_cot'], use_ontology=lp['use_ontology'])
        else:
            nodes = get_start_nodes(inst, eff_k, lp['model'],
                                    use_cot=lp['use_cot'], use_ontology=lp['use_ontology'])
        self._start_nodes_cache[name] = nodes
        print(f'  [LLM] {name}: {nodes[:8]}{"..." if len(nodes) > 8 else ""}')

        # Persist to disk (start_nodes only; accident priority refreshed live during training)
        json.dump({'start_nodes': nodes}, open(cache_path, 'w'))
        print(f'  [LLM cache] saved {name}')

    # ── LLM bias helpers ─────────────────────────────────────────────────────────

    def _priority_scores_tensor(self, inst: dict, unvisited: list | None = None) -> torch.Tensor:
        """Convert cached priority list to per-node score tensor (N+1,).
        Rank 0 → score 1.0, rank K-1 → score ~0; unranked/depot → 0.
        Optionally zeroes out already-visited nodes for a cleaner refresh bias."""
        N      = inst['n_customers']
        name   = inst['name']
        nodes  = self._start_nodes_cache.get(name, [])
        eff_k  = self._inst_top_k.get(name, len(nodes))  # auto mode: limit to n_min_veh
        nodes  = nodes[:eff_k]
        scores = torch.zeros(N + 1, device=self.device)
        K = len(nodes)
        for rank, node in enumerate(nodes):
            if 1 <= node <= N:
                scores[node] = 1.0 - rank / max(K - 1, 1)
        if unvisited is not None:
            visited_set = set(range(1, N + 1)) - set(unvisited)
            for c in visited_set:
                scores[c] = 0.0
        return scores

    def _build_bias_tensor(self, inst: dict, unvisited: list | None = None) -> torch.Tensor:
        """(1, 1, N+1) logit bias ready to broadcast over (batch, pomo, N+1)."""
        strength = self.llm_params.get('bias_strength', 2.0)
        scores   = self._priority_scores_tensor(inst, unvisited)
        return (scores * strength)[None, None, :]

    def _det_starts(self, inst: dict, pomo_size: int) -> torch.Tensor:
        """Return (1, pomo_size) tensor: rollout i → top_k[i], no sampling."""
        top_k = list(self._start_nodes_cache.get(inst['name'], []))[:pomo_size]
        if len(top_k) < pomo_size:
            used = set(top_k)
            pad  = [n for n in range(1, inst['n_customers'] + 1) if n not in used]
            random.shuffle(pad)
            top_k = top_k + pad[:pomo_size - len(top_k)]
        return torch.tensor(top_k, dtype=torch.long, device=self.device).unsqueeze(0)

    def _get_representative_unvisited(self, N: int) -> list:
        """Unvisited customer indices from rollout (batch=0, pomo=0) as representative state."""
        visited = (self.env.visited_ninf_flag[0, 0, 1:] == float('-inf')).cpu()
        return [c for c in range(1, N + 1) if not visited[c - 1].item()]

    def _refresh_priority(self, inst: dict, unvisited: list) -> None:
        """Re-rank unvisited nodes via LLM and update in-memory cache (no CoT for speed)."""
        lp    = self.llm_params
        eff_k = min(lp['top_k'], len(unvisited))
        nodes = get_start_nodes(
            inst, eff_k, lp['model'],
            use_cot=False, use_ontology=lp['use_ontology'],
            unvisited=unvisited,
        )
        self._start_nodes_cache[inst['name']] = nodes
        print(f"  [LLM:refresh] {inst['name']} | {len(unvisited)} unvisited → {nodes[:8]}")

    def _refresh_priority_accident(self, inst: dict, unvisited: list, accident: AccidentEvent) -> None:
        """Re-rank unvisited top-k dispatch candidates given a mid-route accident.

        Context passed to LLM:
          - Unvisited nodes from the current priority cache  (remaining planned first-stops)
          - Accident edge endpoints (node_a, node_b)        (always included even if outside top-k)
        This is intentionally limited to the dispatch-candidate pool; within-route rerouting
        for non-top-k customers is handled by the RL decoder via acc_tokens.
        """
        lp = self.llm_params
        unvisited_set = set(unvisited)

        # Remaining planned dispatch candidates (top-k nodes not yet visited)
        existing_priority = list(self._start_nodes_cache.get(inst['name'], []))
        unvisited_topk = [n for n in existing_priority if n in unvisited_set]

        # Always include accident edge endpoints if still unvisited
        acc_nodes = [n for n in accident.affected_nodes if n in unvisited_set]
        acc_nodes_new = [n for n in acc_nodes if n not in set(unvisited_topk)]

        context_nodes = unvisited_topk + acc_nodes_new
        if not context_nodes:
            return

        eff_k   = len(context_nodes)
        ont_ctx = _VRPTWOntology(inst).get_context(context_nodes) if lp['use_ontology'] else {}
        prompt  = build_accident_prompt(
            inst, context_nodes, ont_ctx, accident, eff_k,
            use_cot=False, use_ontology=lp['use_ontology'],
        )
        resp     = query_llm(prompt, lp['model'])
        priority = parse_priority(resp, context_nodes, eff_k)
        if priority:
            # Re-ranked context nodes replace the front of the cache;
            # remaining unvisited non-context nodes follow (unchanged order)
            context_set = set(context_nodes)
            rest = [n for n in unvisited if n not in context_set]
            self._start_nodes_cache[inst['name']] = priority + rest
            print(f"  [LLM:acc_refresh] {inst['name']} | edge({accident.affected_nodes}) "
                  f"| {len(context_nodes)} candidates → {priority[:8]}")
        else:
            print(f"  [LLM:acc_refresh] {inst['name']} parse failed — keeping previous priority")

    # ------------------------------------------------------------------
    def _fallback_reward(self) -> torch.Tensor:
        dist     = self.env._get_travel_distance()
        N        = self.env.problem_size
        unserved = (self.env.visited_ninf_flag[:, :, 1:] == 0.0).sum(dim=2).float()
        lcp = self.env_params.get('late_count_penalty', 0.0)
        lmp = self.env_params.get('late_penalty',       0.0)
        up  = self.env_params.get('unserved_penalty', 500.0)
        return (-dist
                - lcp * self.env.n_late_stops / N
                - lmp * self.env.total_late
                - up  * unserved / N)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_test(self):
        self.model.eval()
        lp        = self.llm_params
        llm_on    = lp['enabled']
        rewards, per_inst = [], []
        orig_pomo = self.env.pomo_size
        max_steps = self.trainer_params.get('max_steps', 600)
        bs        = self.trainer_params.get('test_batch_size', 1)

        for inst in self.test_pool:
            if llm_on:
                self._ensure_llm_cache(inst)
                bias_t = self._build_bias_tensor(inst)
                pomo_size_eff = (self._inst_top_k.get(
                                     inst['name'],
                                     math.ceil(float(inst['node_demand'].sum().item())))
                                 if lp.get('top_k_auto', False)
                                 else self.env_params['pomo_size'])
            else:
                bias_t = None
                pomo_size_eff = orig_pomo

            self.env.pomo_size = pomo_size_eff
            batch = make_batch(inst, bs, self.device)
            self.env.load_problems(batch)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
            state, reward, done = self.env.pre_step()

            # Step 0: depot
            sel, _ = self.model(state)
            state, reward, done = self.env.step(sel)

            # Steps 1+: bias only at depot dispatch; RL free within each route
            # Step 1: Deterministic (LLM) or random via model (RL)
            if llm_on:
                sel   = self._det_starts(inst, pomo_size_eff).expand(bs, -1)
                state, reward, done = self.env.step(sel)
                step = 2
            else:
                step = 1

            while not done and step < max_steps:
                if llm_on and bias_t is not None:
                    at_depot = (state.current_node == 0).float().unsqueeze(-1)
                    sel, _ = self.model(state, llm_bias=bias_t * at_depot)
                else:
                    sel, _ = self.model(state)
                state, reward, done = self.env.step(sel)
                step += 1

            if reward is None:
                reward = self._fallback_reward()

            max_r, _ = reward.max(dim=1)
            rewards.append(max_r.float().mean().item())
            per_inst.append((inst['name'], max_r.float().mean().item()))

        self.env.pomo_size = orig_pomo
        return sum(rewards) / len(rewards), per_inst

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _best_solution(self, inst: dict):
        self.model.eval()
        lp     = self.llm_params
        llm_on = lp['enabled']

        # ── Event data ────────────────────────────────────────────────────
        rain_nodes, rain_mult, rain_evs = _get_rain_nodes_mult_evs(inst)
        accident_list = _get_accidents(inst) if llm_on else []
        N             = self.env.problem_size

        if llm_on:
            self._ensure_llm_cache(inst)
            bias_t = self._build_bias_tensor(inst)
            pomo_size_eff = (self._inst_top_k.get(
                                 inst['name'],
                                 math.ceil(float(inst['node_demand'].sum().item())))
                             if lp.get('top_k_auto', False)
                             else self.env_params['pomo_size'])
        else:
            bias_t = None
            pomo_size_eff = self.env_params['pomo_size']

        self.env.pomo_size = pomo_size_eff
        batch = make_batch(inst, 1, self.device)
        self.env.load_problems(batch)
        if rain_nodes:
            self.env.apply_rain(rain_nodes, rain_mult)

        reset_state, _, _ = self.env.reset()
        reset_state.rain_tokens = _make_rain_tokens(inst, rain_evs, 1, self.device)
        reset_state.acc_tokens  = _make_acc_tokens([], N, 1, self.device)
        self.model.pre_forward(reset_state)
        state, reward, done = self.env.pre_step()

        # Step 0: depot
        sel, _ = self.model(state)
        state, reward, done = self.env.step(sel)

        # Steps 1+: RL decoder with soft LLM bias + live accident refresh
        max_steps       = self.trainer_params.get('max_steps', 600)
        acc_tt_applied  = [False] * len(accident_list)
        acc_tt_restored = [False] * len(accident_list)
        active_accs     = []

        # Step 1: Deterministic first-customer selection (LLM mode)
        if llm_on:
            sel  = self._det_starts(inst, pomo_size_eff).expand(1, -1)
            state, reward, done = self.env.step(sel)
            step = 2
        else:
            step = 1

        while not done and step < max_steps:
            cur_t = state.current_time.mean().item()

            # Accident TT + encoder token update
            acc_changed      = False
            acc_just_started = []
            for i, (accident, fp) in enumerate(accident_list):
                if not acc_tt_applied[i] and cur_t >= fp['t_start']:
                    self.env.apply_accident(accident.node_a, accident.node_b, fp['multiplier'])
                    active_accs.append(dict(
                        nodes=[accident.node_a, accident.node_b],
                        t_start=fp['t_start'], t_end=fp['t_end'],
                    ))
                    acc_tt_applied[i]  = True
                    acc_changed        = True
                    acc_just_started.append(accident)
                elif acc_tt_applied[i] and not acc_tt_restored[i] and cur_t >= fp['t_end']:
                    self.env.restore_accident(accident.node_a, accident.node_b)
                    active_accs = [a for a in active_accs
                                   if a['nodes'] != [accident.node_a, accident.node_b]]
                    acc_tt_restored[i] = True
                    acc_changed        = True

            if acc_changed:
                self.env.reset_state.acc_tokens = _make_acc_tokens(active_accs, N, 1, self.device)
                self.model.pre_forward(self.env.reset_state)

            # Accident-triggered LLM bias refresh
            if llm_on and acc_just_started:
                unvisited = self._get_representative_unvisited(N)
                if unvisited:
                    self._refresh_priority_accident(inst, unvisited, acc_just_started[-1])
                    bias_t = self._build_bias_tensor(inst, unvisited)

            if llm_on and bias_t is not None:
                at_depot = (state.current_node == 0).float().unsqueeze(-1)
                sel, _ = self.model(state, llm_bias=bias_t * at_depot)
            else:
                sel, _ = self.model(state)
            state, reward, done = self.env.step(sel)
            step += 1

        if reward is None:
            reward = self._fallback_reward()

        best_idx    = int(reward[0].argmax().item())
        best_reward = float(reward[0, best_idx].item())
        node_list   = self.env.selected_node_list[0, best_idx].cpu().tolist()
        return _extract_routes(node_list), best_reward

    # ------------------------------------------------------------------
    def _save_ckpt(self, epoch, tag=None):
        name = f'checkpoint-{tag}.pt' if tag else f'checkpoint-{epoch}.pt'
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward_log':           self.reward_log,
            'loss_log':             self.loss_log,
        }, os.path.join(self.result_dir, name))


# ── Entry point ───────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    import random as _random
    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only',    action='store_true')
    parser.add_argument('--no-llm',       action='store_true')
    parser.add_argument('--no-ontology',  action='store_true')
    parser.add_argument('--base-only',    action='store_true',
                        help='Train on c102-c109 base only, test on c101 only')
    parser.add_argument('--acc-only',     action='store_true',
                        help='Train on c102-c109 base+acc, test on c101 base+acc')
    parser.add_argument('--resume',       default=None)
    parser.add_argument('--epochs',       type=int, default=trainer_params['epochs'])
    parser.add_argument('--top-k',        type=int, default=llm_params['top_k'])
    parser.add_argument('--model',        default=llm_params['model'])
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--config',       default='E',
                        choices=list(_base_config._REWARD_CONFIGS),
                        help='Reward shaping config (default: E)')
    parser.add_argument('--refresh-cache', action='store_true',
                        help='Ignore disk cache and call LLM fresh')
    parser.add_argument('--no-cot',          action='store_true',
                        help='Disable chain-of-thought for faster LLM calls')
    parser.add_argument('--bias-strength',   type=float, default=llm_params['bias_strength'],
                        help='LLM logit bias scale (default 2.0)')
    parser.add_argument('--refresh-interval', type=int,  default=llm_params['refresh_interval'],
                        help='Periodic LLM priority refresh every N vehicle dispatches (0=off)')
    parser.add_argument('--top-k-auto',      action='store_true',
                        help='Set top_k = ceil(total_demand/capacity) per instance (n_min_vehicles)')
    parser.add_argument('--benchmark', default='c1', choices=['c1', 'rc1', 'r1', 'mixed', 'c2'],
                        help='Instance benchmark set (default: c1)')
    parser.add_argument('--acc-ratio', type=float, default=0.0,
                        help='Include ACC instances in training with this sampling ratio (e.g. 0.1). '
                             'Rain ratio = 0.4, base = 1 - acc_ratio - rain_ratio.')
    parser.add_argument('--no-curriculum', action='store_true',
                        help='Disable curriculum learning (sample all types from epoch 1)')
    args = parser.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)
        print(f'[Seed] {args.seed}')

    # Apply reward config
    cfg = _base_config._REWARD_CONFIGS[args.config]
    env_params.update(cfg)
    print(f'[Config {args.config}] {cfg}')

    trainer_params['epochs'] = args.epochs
    llm_params['top_k']      = args.top_k
    llm_params['model']      = args.model

    if args.benchmark == 'rc1':
        _B = [f"rc{i:03d}" for i in range(102, 109)]
        _E = [f"{n}_rain_A" for n in _B] + [f"{n}_rain_B" for n in _B]
        trainer_params['train_instances'] = _B + _E
        trainer_params['test_instances']  = ["rc101","rc101_rain_A","rc101_rain_B","rc101_acc_A","rc101_acc_B"]
        print('[benchmark=rc1] Train: rc102-rc108 + rain  |  Test: rc101 (all scenarios)')
    elif args.benchmark == 'r1':
        _B = [f"r{i:03d}" for i in range(102, 113)]
        _E = [f"{n}_rain_A" for n in _B] + [f"{n}_rain_B" for n in _B]
        trainer_params['train_instances'] = _B + _E
        trainer_params['test_instances']  = ["r101","r101_rain_A","r101_rain_B","r101_acc_A","r101_acc_B"]
        print('[benchmark=r1] Train: r102-r112 + rain  |  Test: r101 (all scenarios)')
    elif args.benchmark == 'c2':
        _B = [f"c{i:03d}" for i in range(202, 209)]
        _E = [f"{n}_rain_A" for n in _B] + [f"{n}_rain_B" for n in _B]
        trainer_params['train_instances'] = _B + _E
        trainer_params['test_instances']  = ["c201","c201_rain_A","c201_rain_B","c201_acc_A","c201_acc_B"]
        print('[benchmark=c2] Train: c202-c208 + rain  |  Test: c201 (all scenarios)')
    elif args.benchmark == 'mixed':
        _C = [f"c{i:03d}" for i in range(102, 110)]
        _RC = [f"rc{i:03d}" for i in range(102, 109)]
        _R  = [f"r{i:03d}" for i in range(102, 113)]
        _ALL = _C + _RC + _R
        _E = [f"{n}_rain_A" for n in _ALL] + [f"{n}_rain_B" for n in _ALL]
        trainer_params['train_instances'] = _ALL + _E
        trainer_params['test_instances']  = ["c101","rc101","r101",
                                             "c101_rain_A","rc101_rain_A","r101_rain_A",
                                             "c101_acc_A", "rc101_acc_A", "r101_acc_A"]
        print('[benchmark=mixed] Train: C1+RC1+R1 + rain  |  Test: c101/rc101/r101 (all scenarios)')

    # --acc-ratio: add ACC instances to training pool with weighted sampling
    if args.acc_ratio > 0:
        _cur = trainer_params['train_instances']
        _base_names = [n for n in _cur if '_rain_' not in n and '_acc_' not in n]
        _acc_names  = [f"{n}_acc_A" for n in _base_names] + [f"{n}_acc_B" for n in _base_names]
        trainer_params['train_instances'] = _cur + _acc_names
        trainer_params['type_ratio'] = {'base': 1/3, 'rain': 1/3, 'acc': 1/3}
        print(f'[acc-ratio={args.acc_ratio}] ACC instances added to training. '
              f'Ratio base=1/3 rain=1/3 acc=1/3 (curriculum: Phase2+)')

    if args.base_only:
        _BASE = ["c102","c103","c104","c105","c106","c107","c108","c109"]
        trainer_params['train_instances']  = _BASE
        trainer_params['test_instances']   = ["c101"]
        trainer_params['curriculum_epoch'] = 0
        print('[base-only] Train: c102-c109  |  Test: c101')
    if args.acc_only:
        _BASE = ["c102","c103","c104","c105","c106","c107","c108","c109"]
        _ACC  = [f"{n}_acc_A" for n in _BASE] + [f"{n}_acc_B" for n in _BASE]
        trainer_params['train_instances']  = _BASE + _ACC
        trainer_params['test_instances']   = ["c101", "c101_acc_A", "c101_acc_B"]
        trainer_params['curriculum_epoch'] = 500
        print('[acc-only] Train: c102-c109 base+acc(24개)  |  Test: c101 base+acc')
    if args.no_curriculum:
        trainer_params['curriculum_epoch'] = 0
        print('[no-curriculum] Curriculum disabled: all instance types sampled from epoch 1')
    if args.no_llm:
        llm_params['enabled'] = False
    if args.no_ontology:
        llm_params['use_ontology'] = False
    if args.resume:
        trainer_params['model_load']['enable'] = True
        trainer_params['model_load']['path']   = args.resume

    if args.refresh_cache:
        trainer_params['force_llm_refresh'] = True
        print('[LLM] Cache refresh: ignoring disk cache, calling LLM fresh')
    if args.no_cot:
        llm_params['use_cot'] = False
        print('[LLM] CoT disabled: faster responses')
    llm_params['bias_strength']    = args.bias_strength
    llm_params['refresh_interval'] = args.refresh_interval
    llm_params['top_k_auto']       = args.top_k_auto
    if args.top_k_auto:
        print('[LLM] top_k_auto: top_k = ceil(total_demand/capacity) per instance')
    if llm_params['enabled']:
        env_params['pomo_size'] = llm_params['top_k']

    tag = ("Ontology+LLM+RL" if llm_params['enabled'] and llm_params['use_ontology']
           else "LLM+RL" if llm_params['enabled'] else "POMO (no LLM)")
    print(f"[{tag}] top_k={llm_params['top_k']}  model={llm_params['model']}")

    trainer = VRPTWLLMTrainer(
        env_params, model_params, optimizer_params, trainer_params, llm_params,
    )

    if args.test_only:
        ckpt_path = os.path.join(trainer.result_dir, 'checkpoint-last.pt')
        if not os.path.isfile(ckpt_path):
            print(f'[Test] No checkpoint at {ckpt_path}')
            return
        ckpt = torch.load(ckpt_path, map_location=trainer.device)
        trainer.model.load_state_dict(ckpt['model_state_dict'])
        trainer.model.eval()
        plots_dir = os.path.join(trainer.result_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        for inst, data_path in zip(trainer.test_pool, trainer.test_paths):
            routes, best_reward = trainer._best_solution(inst)
            sol_routes = _load_sol(data_path, inst['n_customers'])
            _print_solution(inst, routes, best_reward, sol_routes)
            route_path = os.path.join(plots_dir, f"{inst['name']}_best_route.png")
            sol_path   = os.path.join(plots_dir, f"{inst['name']}_solution.txt")
            plot_routes(inst, routes, best_reward, route_path)
            _save_solution(inst, routes, best_reward, sol_path, sol_routes)
    else:
        trainer.run()


if __name__ == '__main__':
    main()
