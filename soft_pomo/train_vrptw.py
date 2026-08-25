"""
train_vrptw.py — Pure RL baseline for VRPTW.

Model: POMO with 8-D node features (x,y,dem,tw_o,tw_c,svc,rain_affected,acc_affected).
       Both rain and accident embedded in encoder as static node features.
       Rain set at episode start; accident triggers re-call of pre_forward mid-episode.
       RL learns event-aware routing purely from state features.

Run:
  python train_vrptw.py
  python train_vrptw.py --test-only
  python train_vrptw.py --epochs 100
"""
from __future__ import annotations

import os
import sys
import math
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam as Optimizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vrptw_env import (
    VRPTWEnv, load_solomon, make_batch,
    AverageMeter, _extract_routes, _print_solution, _save_solution, _load_sol,
    plot_routes, plot_training_curves, augment_xy_data_by_8_fold,
    generate_c_type_instance,
    generate_r_type_instance,
    generate_rc_type_instance,
    generate_unified_instance,
    generate_rf_instance,
)

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ── Dataset ───────────────────────────────────────────────────────────────────
_BASE = ["c102", "c103", "c104", "c105", "c106", "c107", "c108", "c109"]
TRAIN_INSTANCES = (
    _BASE
    + [f"{n}_rain_A" for n in _BASE]
    + [f"{n}_rain_B" for n in _BASE]
)
TEST_INSTANCES = ["c101", "c101_rain_A", "c101_rain_B", "c101_acc_A", "c101_acc_B"]

DATA_DIR   = os.path.join(_ROOT, "data", "Solomon")
RESULT_DIR = os.path.join(_HERE, "result")

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_CUSTOMERS = 100
POMO_SIZE   = 100   # POMO rollouts per instance (standard POMO setting)

env_params = dict(
    problem_size        = N_CUSTOMERS,
    pomo_size           = POMO_SIZE,
    late_count_penalty  = 20.0,   # Config B
    late_penalty        = 20.0,
    vehicle_penalty     = 300.0,
    unserved_penalty    = 500.0,
)

# ── Reward configs (A–F): all 6 orderings of {K, D, Lt} ─────────────────────
#
# reward = -(D / D_max)
#          - late_penalty  × total_late        / Lt_max
#          - vehicle_penalty × depot_visits    / N
#   (late_count_penalty = 0 for all configs)
#
# Normalization reference (Solomon C1 typical values):
#   D   ≈ 1.7  (tour length normalized by T)
#   K   ≈ 10   → K/N ≈ 0.10
#   Lt  ≈ 0.5  (accumulated lateness normalized by T, mildly infeasible)
#
# D_max and Lt_max calibrated so that at reference values:
#   "neutral" contribution of each term ≈ 0.30
#   → D_max=6.0: D_ref/D_max = 1.7/6.0 ≈ 0.28
#   → K_neutral: vehicle_penalty × 0.10 = 0.30  → vehicle_penalty=3.0
#   → Lt_neutral: late_penalty × 0.5/Lt_max = 0.30  → late_penalty/Lt_max = 0.6
#     (Lt_max=5.0, late_penalty=3.0: 3.0×0.5/5.0 = 0.30)
#
# Priority weights applied as multipliers on neutral: 3 (high) / 2 (mid) / 1 (low)
#   High  → term ≈ 0.90   Mid  → term ≈ 0.57–0.60   Low  → term ≈ 0.28–0.30
#
# Verification (D≈1.7, K=10, Lt≈0.5):
#   A K>D>Lt: K=9×0.10=0.90  D=1.7/3.0=0.57  Lt=3×0.5/5=0.30  ✓
#   B K>Lt>D: K=9×0.10=0.90  Lt=6×0.5/5=0.60  D=1.7/6.0=0.28  ✓
#   C D>K>Lt: D=1.7/2.0=0.85  K=6×0.10=0.60  Lt=3×0.5/5=0.30  ✓
#   D D>Lt>K: D=1.7/2.0=0.85  Lt=6×0.5/5=0.60  K=3×0.10=0.30  ✓
#   E Lt>K>D: Lt=9×0.5/5=0.90  K=6×0.10=0.60  D=1.7/6.0=0.28  ✓
#   F Lt>D>K: Lt=9×0.5/5=0.90  D=1.7/3.0=0.57  K=3×0.10=0.30  ✓
# ── Reward configs per benchmark ─────────────────────────────────────────────
#
# Calibration reference (each benchmark's typical T-normalized values):
#   C1  (T=1236): D_ref≈0.67, K_ref=10/100=0.10, Lt_ref≈0.5
#   R1  (T=230):  D_ref≈8-10, K_ref=19/100=0.19, Lt_ref≈0.5
#   RC1 (T=240):  D_ref≈6-8,  K_ref=14/100=0.14, Lt_ref≈0.5
#
# For each config the "neutral" contribution of each term ≈ 0.30.
# Priority weights: High→×3(≈0.90), Mid→×2(≈0.60), Low→×1(≈0.30).
#
# C1 neutral: D=0.67/D_max, K=vp×0.10, Lt=lp×0.5/5.0
# R1 neutral: D=9/D_max,    K=vp×0.19, Lt=lp×0.5/5.0
# RC1 neutral: D=7/D_max,   K=vp×0.14, Lt=lp×0.5/5.0

_REWARD_CONFIGS = {
    # C1 (T=1236, K≈10, D≈0.67)
    # Neutral: D=0.67/D_max, K=vp×0.10, Lt=lp×0.5/5
    'A': dict(late_count_penalty=0.0, late_penalty=3.0, vehicle_penalty=9.0,
              D_max=3.0,  Lt_max=5.0),  # K(0.90) > D(0.22) > Lt(0.30)
    'B': dict(late_count_penalty=0.0, late_penalty=6.0, vehicle_penalty=9.0,
              D_max=6.0,  Lt_max=5.0),  # K(0.90) > Lt(0.60) > D(0.11)
    'C': dict(late_count_penalty=0.0, late_penalty=3.0, vehicle_penalty=6.0,
              D_max=2.0,  Lt_max=5.0),  # D(0.34) > K(0.60) > Lt(0.30)
    'D': dict(late_count_penalty=0.0, late_penalty=6.0, vehicle_penalty=3.0,
              D_max=2.0,  Lt_max=5.0),  # D(0.34) > Lt(0.60) > K(0.30)
    'E': dict(late_count_penalty=0.0, late_penalty=9.0, vehicle_penalty=6.0,
              D_max=6.0,  Lt_max=5.0),  # Lt(0.90) > K(0.60) > D(0.11)
    'F': dict(late_count_penalty=0.0, late_penalty=9.0, vehicle_penalty=3.0,
              D_max=3.0,  Lt_max=5.0),  # Lt(0.90) > D(0.22) > K(0.30)
    'G': dict(late_count_penalty=3.0, late_penalty=9.0, vehicle_penalty=3.0,
              D_max=3.0,  Lt_max=5.0),  # Lc(3.0) + Lt(0.90) > D(0.22) > K(0.30)
    'H': dict(late_count_penalty=0.0, late_penalty=7.0, vehicle_penalty=3.0,
              D_max=3.0,  Lt_max=5.0),  # Lt(0.70) > K(0.30) > D(0.22)  F와 동일 순서, penalty 완화
    'J': dict(late_count_penalty=0.0, late_penalty=9.0, vehicle_penalty=3.0,
              D_max=3.0,  Lt_max=5.0),  # Lt(0.90) > D(0.22) > K(0.30)  F+vp=3
    'K': dict(late_count_penalty=0.0, late_penalty=9.0, vehicle_penalty=2.0,
              D_max=2.0,  Lt_max=5.0),  # Lt(0.90) > D(0.34) > K(0.20)  F+Dmax=2
    # RF: RouteFinder-style — distance-only, hard TW masking
    'RF': dict(late_count_penalty=0.0, late_penalty=0.0, vehicle_penalty=0.0,
               D_max=1.0, Lt_max=1.0, hard_tw=True),
}

_REWARD_CONFIGS_R1 = {
    # R1 (T=230, K≈19, D_ref≈9)
    # Neutral: D=9/D_max, K=vp×0.19, Lt=lp×0.5/5
    # Each neutral ≈ 0.30 → vp_neutral≈1.6, D_max_neutral≈30, lp_neutral≈3
    # Priority multipliers: High×3(≈0.90), Mid×2(≈0.60), Low×1(≈0.30)
    'A': dict(late_count_penalty=0.0, late_penalty=3.0, vehicle_penalty=5.0,
              D_max=15.0, Lt_max=5.0),  # K(0.95)>D(0.60)>Lt(0.30)
    'B': dict(late_count_penalty=0.0, late_penalty=6.0, vehicle_penalty=5.0,
              D_max=30.0, Lt_max=5.0),  # K(0.95)>Lt(0.60)>D(0.30)
    'C': dict(late_count_penalty=0.0, late_penalty=3.0, vehicle_penalty=3.0,
              D_max=10.0, Lt_max=5.0),  # D(0.90)>K(0.57)>Lt(0.30)
    'D': dict(late_count_penalty=0.0, late_penalty=6.0, vehicle_penalty=2.0,
              D_max=10.0, Lt_max=5.0),  # D(0.90)>Lt(0.60)>K(0.38)
    'E': dict(late_count_penalty=0.0, late_penalty=9.0, vehicle_penalty=3.0,
              D_max=30.0, Lt_max=5.0),  # Lt(0.90)>K(0.57)>D(0.30)
    'F': dict(late_count_penalty=0.0, late_penalty=9.0, vehicle_penalty=2.0,
              D_max=15.0, Lt_max=5.0),  # Lt(0.90)>D(0.60)>K(0.38)
    'G': dict(late_count_penalty=3.0, late_penalty=9.0, vehicle_penalty=2.0,
              D_max=15.0, Lt_max=5.0),  # Lc(3.0) + Lt(0.90)>D(0.60)>K(0.38)
    'H': dict(late_count_penalty=0.0, late_penalty=7.0, vehicle_penalty=2.0,
              D_max=15.0, Lt_max=5.0),  # Lt(0.70)>D(0.60)>K(0.38)  F와 동일 순서, penalty 완화
    'J': dict(late_count_penalty=0.0, late_penalty=9.0, vehicle_penalty=3.0,
              D_max=15.0, Lt_max=5.0),  # Lt(0.90)>D(0.60)≈K(0.57)  F+vp=3
    'K': dict(late_count_penalty=0.0, late_penalty=9.0, vehicle_penalty=2.0,
              D_max=10.0, Lt_max=5.0),  # D(0.90)≈Lt(0.90)>K(0.38)  F+Dmax=10
    'RF': dict(late_count_penalty=0.0, late_penalty=0.0, vehicle_penalty=0.0,
               D_max=1.0, Lt_max=1.0, hard_tw=True),
}

_REWARD_CONFIGS_RC1 = {
    # RC1 (T=240, K≈14, D_ref≈7)
    # Neutral: D=7/D_max, K=vp×0.14, Lt=lp×0.5/5
    # vp_neutral≈2.1, D_max_neutral≈23, lp_neutral≈3
    'A': dict(late_count_penalty=0.0, late_penalty=3.0, vehicle_penalty=6.0,
              D_max=12.0, Lt_max=5.0),  # K(0.84)>D(0.58)>Lt(0.30)
    'B': dict(late_count_penalty=0.0, late_penalty=6.0, vehicle_penalty=6.0,
              D_max=24.0, Lt_max=5.0),  # K(0.84)>Lt(0.60)>D(0.29)
    'C': dict(late_count_penalty=0.0, late_penalty=3.0, vehicle_penalty=4.0,
              D_max=8.0,  Lt_max=5.0),  # D(0.88)>K(0.56)>Lt(0.30)
    'D': dict(late_count_penalty=0.0, late_penalty=6.0, vehicle_penalty=2.0,
              D_max=8.0,  Lt_max=5.0),  # D(0.88)>Lt(0.60)>K(0.28)
    'E': dict(late_count_penalty=0.0, late_penalty=9.0, vehicle_penalty=4.0,
              D_max=24.0, Lt_max=5.0),  # Lt(0.90)>K(0.56)>D(0.29)
    'F': dict(late_count_penalty=0.0, late_penalty=9.0, vehicle_penalty=2.0,
              D_max=12.0, Lt_max=5.0),  # Lt(0.90)>D(0.58)>K(0.28)
    'G': dict(late_count_penalty=3.0, late_penalty=9.0, vehicle_penalty=2.0,
              D_max=12.0, Lt_max=5.0),  # Lc(3.0) + Lt(0.90)>D(0.58)>K(0.28)
    'H': dict(late_count_penalty=0.0, late_penalty=7.0, vehicle_penalty=2.0,
              D_max=12.0, Lt_max=5.0),  # Lt(0.70)>D(0.58)>K(0.28)  F와 동일 순서, penalty 완화
    'J': dict(late_count_penalty=0.0, late_penalty=9.0, vehicle_penalty=3.0,
              D_max=12.0, Lt_max=5.0),  # Lt(0.90)>D(0.58)≈K(0.42)  F+vp=3
    'K': dict(late_count_penalty=0.0, late_penalty=9.0, vehicle_penalty=2.0,
              D_max=8.0,  Lt_max=5.0),  # D(0.88)≈Lt(0.90)>K(0.28)  F+Dmax=8
    'RF': dict(late_count_penalty=0.0, late_penalty=0.0, vehicle_penalty=0.0,
               D_max=1.0, Lt_max=1.0, hard_tw=True),
}

model_params = dict(
    embedding_dim      = 128,
    encoder_layer_num  = 6,
    head_num           = 8,
    qkv_dim            = 16,
    ff_hidden_dim      = 512,
    eval_type          = 'argmax',
    logit_clipping     = 10.0,
)

optimizer_params = dict(
    optimizer = dict(lr=1e-4),
)

trainer_params = dict(
    use_cuda         = True,
    cuda_device_num  = 0,
    data_dir         = DATA_DIR,
    train_instances  = TRAIN_INSTANCES,
    test_instances   = TEST_INSTANCES,
    result_dir       = RESULT_DIR,
    epochs           = 2000,
    train_episodes   = 2500,
    train_batch_size = 64,
    test_batch_size  = 1,
    curriculum_epoch = 0,     # 0 = disabled (no phase split)
    max_steps        = 600,
    n_mc_samples     = 1,
    model_load       = dict(enable=False, path=None, epoch=None),
    logging          = dict(model_save_interval=500, log_interval=10),
)


# ── Model ─────────────────────────────────────────────────────────────────────
# Node features (8-D): x, y, demand, tw_open, tw_close, service, rain_affected, acc_affected
#   → rain and accident embedded as static node features in encoder
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
        depot_xy  = reset_state.depot_xy
        node_xy   = reset_state.node_xy
        demand    = reset_state.node_demand
        tw_open   = reset_state.node_tw_open
        tw_close  = reset_state.node_tw_close
        service   = reset_state.node_service_time

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

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size  = state.BATCH_IDX.size(1)

        if state.selected_count == 0:
            selected = torch.zeros((batch_size, pomo_size), dtype=torch.long,
                                   device=state.BATCH_IDX.device)
            prob = torch.ones((batch_size, pomo_size), device=state.BATCH_IDX.device)

        elif state.selected_count == 1:
            n_customers = state.ninf_mask.size(-1) - 1  # exclude depot
            # Sample once per episode, shared across all batch instances
            starts = torch.randperm(n_customers, device=state.BATCH_IDX.device)[:pomo_size] + 1
            selected = starts[None, :].expand(batch_size, pomo_size)
            prob = torch.ones((batch_size, pomo_size), device=state.BATCH_IDX.device)

        else:
            encoded_last = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(encoded_last, state.load, state.current_time,
                                 ninf_mask=state.ninf_mask)

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
        self.embedding_depot = nn.Linear(2, emb)  # depot: (x, y)
        self.embedding_node  = nn.Linear(6, emb)  # node: (x,y,demand,tw_open,tw_close,service)
        self.embedding_rain  = nn.Linear(4, emb)  # rain token: (node/N, mm/100, t_s/T, t_e/T)
        self.embedding_acc   = nn.Linear(3, emb)  # acc  token: (node/N, t_s/T, t_e/T)
        self.layers = nn.ModuleList([_EncoderLayer(**mp) for _ in range(mp['encoder_layer_num'])])

    def forward(self, depot_xy, node_features, rain_tokens=None, acc_tokens=None):
        emb_depot = self.embedding_depot(depot_xy)     # (batch, 1, emb)
        emb_node  = self.embedding_node(node_features) # (batch, N, emb)
        seq = torch.cat([emb_depot, emb_node], dim=1)  # (batch, N+1, emb)
        if rain_tokens is not None and rain_tokens.size(1) > 0:
            seq = torch.cat([seq, self.embedding_rain(rain_tokens)], dim=1)
        if acc_tokens is not None and acc_tokens.size(1) > 0:
            seq = torch.cat([seq, self.embedding_acc(acc_tokens)], dim=1)
        for layer in self.layers:
            seq = layer(seq)
        return seq[:, :emb_node.size(1) + 1, :]  # (batch, N+1, emb) routing nodes only


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

    def forward(self, encoded_last, load, current_time, ninf_mask):
        ctx = torch.cat([
            encoded_last,
            load         [:, :, None],
            current_time [:, :, None],
        ], dim=2)   # (batch, pomo, emb+2)

        q      = _reshape(self.Wq_last(ctx), self.head_num)
        mh_out = self.combine(_mha(q, self.k, self.v, rank3_ninf_mask=ninf_mask))
        score  = torch.matmul(mh_out, self.single_head_key) / self.sqrt_emb
        score  = self.logit_clipping * torch.tanh(score) + ninf_mask
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

def _get_rain_event(inst: dict) -> tuple[list, float, list]:
    """Returns (rain_nodes, multiplier, rain_evs_raw) from preset_events."""
    rain_evs = [e for e in inst.get('preset_events', []) if e['type'] == 'RAIN']
    if not rain_evs:
        return [], 1.0, []
    nodes = sorted({n for e in rain_evs for n in e['nodes']})
    mult  = max(e['multiplier'] for e in rain_evs)
    return nodes, mult, rain_evs


def _get_accident_events(inst: dict) -> list[dict]:
    """Returns list of accident event dicts from preset_events."""
    acc_evs = [e for e in inst.get('preset_events', []) if e['type'] == 'ACCIDENT']
    if not acc_evs:
        return []
    T = float(inst['T'])
    result = []
    for e in acc_evs:
        nodes = e['nodes']
        result.append(dict(
            nodes=nodes,
            node_a=nodes[0] if len(nodes) >= 1 else 0,
            node_b=nodes[1] if len(nodes) >= 2 else 0,
            multiplier=e['multiplier'],
            t_start=e['trigger_time'] / max(T, 1.0),
            t_end=(e['trigger_time'] + e['duration']) / max(T, 1.0),
        ))
    return result




def _make_rain_tokens(inst: dict, rain_evs: list, batch_size: int, device) -> torch.Tensor:
    """(batch, R, 4) — one token per (event, affected_stop): [node/N, mm/100, t_s/T, t_e/T]."""
    T, N = float(inst.get('T', 1.0)), inst['n_customers']
    rows = []
    for ev in rain_evs:
        mm   = min(ev.get('rainfall_mm', 0.0) / 100.0, 1.0)
        t_s  = ev.get('trigger_time', 0.0) / max(T, 1.0)
        t_e  = (ev.get('trigger_time', 0.0) + ev.get('duration', 0.0)) / max(T, 1.0)
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


# ── Trainer ───────────────────────────────────────────────────────────────────

def _inst_tag(instances: list) -> str:
    """Base instance names only (event suffixes 제외) -> 'c102-c109'"""
    base = sorted(set(n.split('_')[0] for n in instances))
    return f"{base[0]}-{base[-1]}" if len(base) > 1 else base[0]


class VRPTWTrainer:
    def __init__(self, env_p, model_p, opt_p, trainer_p):
        self.env_params       = env_p
        self.model_params     = model_p
        self.optimizer_params = opt_p
        self.trainer_params   = trainer_p

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
        self._gen_rng  = np.random.default_rng(trainer_p.get('seed', 42))

        data_dir = trainer_p['data_dir']
        self.instance_pool = []
        self.base_pool     = []   # base instances only (for curriculum Phase 1)

        # Pre-generated dataset (.pt) takes priority over Solomon .txt pool
        train_dataset_path = trainer_p.get('train_dataset', None)
        if train_dataset_path and os.path.isfile(train_dataset_path):
            self._train_dataset = torch.load(train_dataset_path, weights_only=False)
            print(f'  [Dataset] loaded {len(self._train_dataset)} instances from {train_dataset_path}')
            # Global iterator: one shuffle, consume in order across all epochs
            _idx = list(range(len(self._train_dataset)))
            random.Random(trainer_p.get('seed', 42)).shuffle(_idx)
            self._dataset_iter  = iter(_idx)
            self._dataset_indices = _idx
        else:
            self._train_dataset   = None
            self._dataset_iter    = None
            self._dataset_indices = None
            for iname in trainer_p['train_instances']:
                inst = load_solomon(os.path.join(data_dir, iname + '.txt'))
                self.instance_pool.append(inst)
                if not inst['preset_events']:
                    self.base_pool.append(inst)
                print(f'  [Data] {inst["name"]}  N={inst["n_customers"]}'
                      f'  events={len(inst["preset_events"])}')

        self.curriculum_epoch = trainer_p.get('curriculum_epoch', 0)
        if self.curriculum_epoch > 0:
            print(f'  [Curriculum] Phase 1: base only (epoch 1-{self.curriculum_epoch})'
                  f' → Phase 2: full pool (epoch {self.curriculum_epoch+1}+)')

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
            ep = model_load['epoch']
            self.start_epoch = (ep + 1) if isinstance(ep, int) else 1
            print(f'[Resume] loaded epoch={ep}, start_epoch={self.start_epoch}')

        base_result_dir = trainer_p.get('result_dir', 'result')
        gen_type        = trainer_p.get('gen_type', None)
        if train_dataset_path and self._train_dataset is not None:
            inst_tag = os.path.splitext(os.path.basename(train_dataset_path))[0]
        elif gen_type:
            inst_tag = f'gen_{gen_type}'
        else:
            inst_tag = _inst_tag(trainer_p['train_instances'])
        self.result_dir = os.path.join(base_result_dir, inst_tag)
        os.makedirs(self.result_dir, exist_ok=True)
        self.reward_log = []
        self.loss_log   = []

    # ------------------------------------------------------------------
    def run(self):
        if _MLFLOW:
            mlflow.set_experiment("VRPTW_PurePOMO")
            run_ctx = mlflow.start_run(run_name="pure_pomo")
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
                "train_instances": ",".join(tp['train_instances']),
                "test_instances":  ",".join(tp.get('test_instances', [])),
            })

        t0 = time.time()
        train_reward_history = []
        loss_history         = []
        test_reward_history  = []   # [(epoch, avg_reward), ...]

        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            train_reward, train_loss = self._train_one_epoch(epoch)
            train_reward_history.append(train_reward)
            loss_history.append(train_loss)
            self.reward_log.append(train_reward)
            self.loss_log.append(train_loss)

            phase = ('P1:base' if self.curriculum_epoch > 0 and epoch <= self.curriculum_epoch
                     else 'P2:full')
            total_epochs = self.trainer_params['epochs']
            log_interval = max(1, total_epochs // 10)
            if epoch % log_interval == 0 or epoch == total_epochs:
                elapsed = time.time() - t0
                h, m = divmod(int(elapsed), 3600)
                m, s = divmod(m, 60)
                print(f'Epoch {epoch:4d}/{total_epochs}  [{phase}]  '
                      f'train_reward={train_reward:.4f}  loss={train_loss:.6f}  '
                      f'elapsed={h:02d}:{m:02d}:{s:02d}')

            if _MLFLOW:
                mlflow.log_metrics({"train/reward": train_reward,
                                    "train/loss":   train_loss}, step=epoch)

            save_interval = self.trainer_params['logging'].get('model_save_interval', 500)
            if epoch % save_interval == 0:
                self._save_ckpt(epoch)
                if self.test_pool:
                    self.model.eval()
                    avg_r, per_inst = self._eval_test()
                    self.model.train()
                    test_reward_history.append((epoch, avg_r))
                    per_str = '  '.join(f'{n}={r:.4f}' for n, r in per_inst)
                    print(f'  [Test@{epoch}] avg={avg_r:.4f}  {per_str}')
                    if _MLFLOW:
                        mlflow.log_metric("test/reward", avg_r, step=epoch)

        self._save_ckpt(self.trainer_params['epochs'], tag='last')
        total_min = (time.time() - t0) / 60
        print(f'\n*** Training Done ***  total={total_min:.1f}min  '
              f'avg_epoch={total_min/max(1, self.trainer_params["epochs"]-self.start_epoch+1)*60:.1f}s/epoch')

        plots_dir  = os.path.join(self.result_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_training_curves(train_reward_history, loss_history, plots_dir,
                             test_scores=test_reward_history or None)

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

        # Save best solutions for training instances (base only — no event variants)
        if self.instance_pool:
            self.model.eval()
            for inst in self.instance_pool:
                if any(tag in inst['name'].upper() for tag in ('RAIN', 'ACC')):
                    continue
                routes, best_reward = self._best_solution(inst)
                sol_path = os.path.join(plots_dir, f"{inst['name']}_solution.txt")
                _save_solution(inst, routes, best_reward, sol_path, sol_routes=None)
                print(f"  [TrainSol] {inst['name']}: reward={best_reward:.4f} saved")

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
        gen_type = self.trainer_params.get('gen_type', None)

        # Curriculum: Phase 1 (base only) → Phase 2 (full pool)
        if self.curriculum_epoch > 0 and epoch <= self.curriculum_epoch:
            pool = self.base_pool if self.base_pool else self.instance_pool
        else:
            pool = self.instance_pool

        gen_n_min = self.trainer_params.get('gen_n_min', None)
        gen_n_max = self.trainer_params.get('gen_n_max', None)

        ep = 0
        while ep < n_ep:
            bs = min(self.trainer_params['train_batch_size'], n_ep - ep)
            if self._train_dataset is not None:
                idx = next(self._dataset_iter, None)
                if idx is None:
                    # exhausted one full pass — re-shuffle and restart
                    random.shuffle(self._dataset_indices)
                    self._dataset_iter = iter(self._dataset_indices)
                    idx = next(self._dataset_iter)
                inst = self._train_dataset[idx]
                self.env.pomo_size = min(self.env_params['pomo_size'], inst['n_customers'])
            elif gen_type is not None:
                n_cust = (random.randint(gen_n_min, gen_n_max)
                          if gen_n_min and gen_n_max else None)
                kw = dict(n_customers=n_cust) if n_cust else {}
                if self.trainer_params.get('fixed_svc', False):
                    kw['svc_min'] = 10.0
                    kw['svc_max'] = 10.0
                if gen_type == 'C':
                    inst = generate_c_type_instance(rng=self._gen_rng, **kw)
                elif gen_type == 'R':
                    inst = generate_r_type_instance(rng=self._gen_rng, **kw)
                elif gen_type == 'RC':
                    inst = generate_rc_type_instance(rng=self._gen_rng, **kw)
                elif gen_type == 'UNIFIED':
                    t_corr_svc = self.trainer_params.get('gen_svc_t_corr', True)
                    inst = generate_unified_instance(rng=self._gen_rng, t_corr_svc=t_corr_svc, **kw)
                elif gen_type == 'RF':
                    inst = generate_rf_instance(rng=self._gen_rng, **kw)
                else:
                    inst = random.choice(pool)
                self.env.pomo_size = min(self.env_params['pomo_size'], inst['n_customers'])
            else:
                inst = random.choice(pool)
            r, l = self._train_one_batch(inst, bs)
            reward_AM.update(r, bs)
            loss_AM.update(l,  bs)
            ep += bs
        return reward_AM.avg, loss_AM.avg

    # ------------------------------------------------------------------
    def _fallback_reward(self) -> torch.Tensor:
        """Penalty when max_steps truncates episode: distance + late + unserved penalties."""
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
    def _train_one_batch(self, inst: dict, batch_size: int):
        self.model.train()
        tp = self.trainer_params

        rain_nodes, rain_mult, rain_evs = _get_rain_event(inst)
        acc_evs = _get_accident_events(inst)

        max_steps = tp.get('max_steps', 600)
        N = self.env.problem_size

        if tp.get('pomo_auto', False):
            self.env.pomo_size = math.ceil(float(inst['node_demand'].sum().item()))

        batch = make_batch(inst, batch_size, self.device)
        self.env.load_problems(batch)

        if rain_nodes:
            self.env.apply_rain(rain_nodes, rain_mult)

        reset_state, _, _ = self.env.reset()
        reset_state.rain_tokens = _make_rain_tokens(inst, rain_evs, batch_size, self.device)
        reset_state.acc_tokens  = _make_acc_tokens([], N, batch_size, self.device)
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(batch_size, self.env.pomo_size, 0, device=self.device)
        state, reward, done = self.env.pre_step()

        # Step 1: random K starts from all N customers (generalises to any start)
        rand_starts = random.sample(range(1, inst['n_customers'] + 1), self.env.pomo_size)
        start_t = torch.tensor(rand_starts, dtype=torch.long, device=self.device)
        selected = start_t[None, :].expand(batch_size, self.env.pomo_size)
        prob_list = torch.cat(
            (prob_list, torch.ones(batch_size, self.env.pomo_size, 1, device=self.device)), dim=2)
        state, reward, done = self.env.step(selected)

        acc_applied  = [False] * len(acc_evs)
        acc_restored = [False] * len(acc_evs)
        active_accs  = []
        step = 0
        while not done and step < max_steps:
            cur_t = state.current_time.mean().item()
            acc_changed = False
            for i, acc_ev in enumerate(acc_evs):
                if not acc_applied[i] and cur_t >= acc_ev['t_start']:
                    self.env.apply_accident(acc_ev['node_a'], acc_ev['node_b'], acc_ev['multiplier'])
                    active_accs.append(acc_ev)   # acc_ev already contains 'nodes'
                    acc_applied[i] = True
                    acc_changed = True
                elif acc_applied[i] and not acc_restored[i] and cur_t >= acc_ev['t_end']:
                    self.env.restore_accident(acc_ev['node_a'], acc_ev['node_b'])
                    active_accs = [a for a in active_accs if a is not acc_ev]
                    acc_restored[i] = True
                    acc_changed = True

            if acc_changed:
                self.env.reset_state.acc_tokens = _make_acc_tokens(
                    active_accs, N, batch_size, self.device)
                self.model.pre_forward(self.env.reset_state)

            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            step += 1

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

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_test(self):
        self.model.eval()
        rewards, per_inst = [], []
        for inst in self.test_pool:
            if self.trainer_params.get('pomo_auto', False):
                self.env.pomo_size = math.ceil(float(inst['node_demand'].sum().item()))
            batch = make_batch(inst, self.trainer_params.get('test_batch_size', 1), self.device)
            self.env.load_problems(batch)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
            state, reward, done = self.env.pre_step()
            step = 0
            while not done and step < self.trainer_params.get('max_steps', 600):
                sel, _ = self.model(state)
                state, reward, done = self.env.step(sel)
                step += 1
            if reward is None:
                reward = self._fallback_reward()
            max_r, _ = reward.max(dim=1)
            r = max_r.float().mean().item()
            rewards.append(r)
            per_inst.append((inst['name'], r))
        return sum(rewards) / len(rewards), per_inst

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _best_solution(self, inst: dict):
        self.model.eval()
        max_steps    = self.trainer_params.get('max_steps', 600)
        n_mc_samples = self.trainer_params.get('n_mc_samples', 1)
        use_aug      = self.trainer_params.get('use_augmentation', False)

        rain_nodes, rain_mult, rain_evs = _get_rain_event(inst)
        N = inst['n_customers']

        # Fair comparison mode: K random starts instead of full pomo_size
        eval_top_k = self.trainer_params.get('eval_top_k', None)
        if eval_top_k is not None:
            rand_starts = random.sample(range(1, N + 1), min(eval_top_k, N))
            self.env.pomo_size = len(rand_starts)
        elif self.trainer_params.get('pomo_auto', False):
            rand_starts = None
            self.env.pomo_size = math.ceil(float(inst['node_demand'].sum().item()))
        else:
            rand_starts = None
            self.env.pomo_size = self.env_params['pomo_size']

        base_batch = make_batch(inst, 1, self.device)

        if use_aug:
            # 8-fold: 좌표만 8가지 변환, TT는 등거리 변환이라 동일 → 그대로 복제
            aug_depot = augment_xy_data_by_8_fold(base_batch['depot_xy'])  # (8,1,2)
            aug_nodes = augment_xy_data_by_8_fold(base_batch['node_xy'])   # (8,N,2)
            batch = {k: v.expand(8, *v.shape[1:]).contiguous()
                     for k, v in base_batch.items()}
            batch['depot_xy'] = aug_depot
            batch['node_xy']  = aug_nodes
            aug_size = 8
        else:
            batch    = base_batch
            aug_size = 1

        use_sampling = n_mc_samples > 1
        if use_sampling:
            self.model.mp['eval_type'] = 'softmax'

        best_reward    = float('-inf')
        best_node_list = None
        best_aug_idx   = 0

        acc_evs = _get_accident_events(inst)
        N = self.env.problem_size

        try:
            for _ in range(n_mc_samples):
                self.env.load_problems(batch)
                if rain_nodes:
                    self.env.apply_rain(rain_nodes, rain_mult)
                reset_state, _, _ = self.env.reset()
                reset_state.rain_tokens = _make_rain_tokens(inst, rain_evs, aug_size, self.device)
                reset_state.acc_tokens  = torch.zeros(aug_size, 0, 4, device=self.device)
                self.model.pre_forward(reset_state)
                state, reward, done = self.env.pre_step()

                # Fair comparison: force random K start nodes
                if rand_starts is not None:
                    start_t  = torch.tensor(rand_starts, dtype=torch.long, device=self.device)
                    selected = start_t[None, :].expand(aug_size, len(rand_starts))
                    state, reward, done = self.env.step(selected)

                acc_applied  = [False] * len(acc_evs)
                acc_restored = [False] * len(acc_evs)
                active_accs  = []
                step = 0
                while not done and step < max_steps:
                    cur_t = state.current_time.mean().item()
                    acc_changed = False
                    for i, ae in enumerate(acc_evs):
                        if not acc_applied[i] and cur_t >= ae['t_start']:
                            self.env.apply_accident(ae['node_a'], ae['node_b'], ae['multiplier'])
                            active_accs.append(ae)
                            acc_applied[i] = True
                            acc_changed = True
                        elif acc_applied[i] and not acc_restored[i] and cur_t >= ae['t_end']:
                            self.env.restore_accident(ae['node_a'], ae['node_b'])
                            active_accs = [a for a in active_accs if a is not ae]
                            acc_restored[i] = True
                            acc_changed = True
                    if acc_changed:
                        reset_state.acc_tokens = _make_acc_tokens(active_accs, N, aug_size, self.device)
                        self.model.pre_forward(reset_state)

                    sel, _ = self.model(state)
                    state, reward, done = self.env.step(sel)
                    step += 1

                if reward is None:
                    reward = self._fallback_reward()

                # reward: (aug_size, pomo) — best across all augmentations × pomo
                r_flat   = reward.reshape(-1)
                best_idx = int(r_flat.argmax().item())
                if float(r_flat[best_idx]) > best_reward:
                    best_reward  = float(r_flat[best_idx])
                    best_aug_idx = best_idx // self.env.pomo_size
                    best_pomo    = best_idx %  self.env.pomo_size
                    best_node_list = self.env.selected_node_list[
                        best_aug_idx, best_pomo].cpu().tolist()
        finally:
            if use_sampling:
                self.model.mp['eval_type'] = 'argmax'

        aug_label = f"aug×{aug_size}" if use_aug else "no aug"
        print(f"  [Search] {aug_label}  mc×{n_mc_samples}"
              f"  total_rollouts={aug_size * self.env.pomo_size * n_mc_samples:,}")
        return _extract_routes(best_node_list), best_reward

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
    parser.add_argument('--test-only',      action='store_true')
    parser.add_argument('--resume',         default=None)
    parser.add_argument('--train-dataset',  default=None,
                        help='Path to pre-generated .pt dataset for reproducible training')
    parser.add_argument('--test-instances', nargs='+', default=None,
                        help='Override test instances (e.g. c101 r101 rc101)')
    parser.add_argument('--train-instances', nargs='+', default=None,
                        help='Override train instances, replacing the benchmark pool '
                             '(e.g. r102 r103 RH01 RH02). Applied after --benchmark.')
    parser.add_argument('--epochs',    type=int, default=trainer_params['epochs'])
    parser.add_argument('--n-mc',      type=int, default=trainer_params['n_mc_samples'],
                        help='MC sampling passes for best_solution (1=greedy, e.g. 128=sampling)')
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--config',    default='A', choices=list(_REWARD_CONFIGS),
                        help='Reward config: A=K>D>Lt, B=K>Lt>D, C=D>K>Lt, D=D>Lt>K, E=Lt>K>D, F=Lt>D>K, H=Lt>D>K(penalty완화)')
    parser.add_argument('--pomo',      type=int, default=None,
                        help='Override POMO_SIZE (e.g. 100 for full search)')
    parser.add_argument('--pomo-auto', action='store_true',
                        help='Set pomo_size = ceil(total_demand/capacity) per instance (min vehicles)')
    parser.add_argument('--aug',       action='store_true',
                        help='8-fold geometric augmentation at test time')
    parser.add_argument('--top-k',     type=int, default=None,
                        help='Use K random starts at test time (fair comparison with LLM)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override train_batch_size (e.g. 16 when pomo=100 to keep memory same as 64×25)')
    parser.add_argument('--tag', default=None,
                        help='Suffix appended to result dir (e.g. pomo100 → result/config_A_pomo100/)')
    parser.add_argument('--base-only', action='store_true',
                        help='Train on c102-c109 base only, test on c101 only')
    parser.add_argument('--test-in-sample', action='store_true',
                        help='Evaluate on training instances (in-sample check). '
                             'Use with --test-only to verify model on its own training data.')
    parser.add_argument('--split', default=None, choices=['tight', 'loose'],
                        help='C1 train/test split: '
                             'tight=train c102-c109 / test c101 (tightest TW), '
                             'loose=train c101-c108 / test c109 (loosest TW)')
    parser.add_argument('--acc-only',  action='store_true',
                        help='Train on c102-c109 base+acc, test on c101 base+acc')
    parser.add_argument('--with-acc',  action='store_true',
                        help='Add acc_A and acc_B events to training instances (on top of rain)')
    parser.add_argument('--events-only', action='store_true',
                        help='Remove base instances from training, keep only event (rain/acc) instances')
    parser.add_argument('--benchmark', default='c1',
                        choices=['c1', 'rc1', 'r1', 'mixed', 'c2',
                                 'c1-rand', 'rc1-rand', 'r1-rand'],
                        help='Instance benchmark set (default: c1)')
    parser.add_argument('--gen-type', default=None, choices=['C', 'R', 'RC', 'UNIFIED', 'RF'],
                        help='Random instance generation mode: C/R/RC (Solomon 1987 method). '
                             'Replaces fixed training files with on-the-fly generation.')
    parser.add_argument('--no-svc-t-corr', action='store_true', default=False,
                        help='UNIFIED generator: disable T-correlated svc (use uniform [0.040,0.080]). '
                             '구 generator 재현용.')
    parser.add_argument('--fixed-svc', action='store_true', default=False,
                        help='R/RC generator: fix svc=10 exactly (Solomon-aligned). '
                             'Default: Uniform(9,11) per customer.')
    parser.add_argument('--n-customers-min', type=int, default=None,
                        help='Min customers per generated instance (e.g. 50). '
                             'If set with --n-customers-max, N is sampled uniformly per episode.')
    parser.add_argument('--n-customers-max', type=int, default=None,
                        help='Max customers per generated instance (e.g. 200).')
    parser.add_argument('--d-max-scale', type=float, default=1.0,
                        help='Multiply D_max in the selected reward config by this factor. '
                             'Use ~3 for R1 (T=230) and ~2 for RC1 (T=240) to compensate '
                             'for smaller T scaling up D_normalized relative to C1 (T=1236).')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (default: 1e-4). Use smaller value e.g. 2e-5 for fine-tuning.')
    args = parser.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)
        print(f'[Seed] {args.seed}')

    trainer_params['epochs']           = args.epochs
    trainer_params['n_mc_samples']     = args.n_mc
    if args.train_dataset:
        trainer_params['train_dataset'] = args.train_dataset
    trainer_params['use_augmentation'] = args.aug
    if args.lr is not None:
        optimizer_params['optimizer']['lr'] = args.lr
        print(f'[LR] learning rate overridden to {args.lr}')
    if args.batch_size is not None:
        trainer_params['train_batch_size'] = args.batch_size
        print(f'[batch] train_batch_size overridden to {args.batch_size}')

    # Benchmark switching (overrides default C1 instances)
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
        print('[benchmark=mixed] Train: C1+RC1+R1  |  Test: c101/rc101/r101')
    # ── Random-event variants (rand_rain/rand_acc instead of A/B) ──────────────
    elif args.benchmark == 'r1-rand':
        _B = [f"r{i:03d}" for i in range(102, 113)]
        _E = [f"{n}_rand_rain_{j:03d}" for n in _B for j in range(1, 11)]
        trainer_params['train_instances'] = _B + _E
        _T = [f"r101_rand_rain_{j:03d}" for j in range(1, 4)] + \
             [f"r101_rand_acc_{j:03d}"  for j in range(1, 3)]
        trainer_params['test_instances'] = ["r101"] + _T
        print(f'[benchmark=r1-rand] Train: r102-r112 + rand_rain×10/inst ({len(_B)+len(_E)} total)'
              f'  |  Test: r101 + rand_rain×3 + rand_acc×2')
    elif args.benchmark == 'c1-rand':
        _B = [f"c{i:03d}" for i in range(102, 110)]
        _E = [f"{n}_rand_rain_{j:03d}" for n in _B for j in range(1, 11)]
        trainer_params['train_instances'] = _B + _E
        _T = [f"c101_rand_rain_{j:03d}" for j in range(1, 4)] + \
             [f"c101_rand_acc_{j:03d}"  for j in range(1, 3)]
        trainer_params['test_instances'] = ["c101"] + _T
        print(f'[benchmark=c1-rand] Train: c102-c109 + rand_rain×10/inst ({len(_B)+len(_E)} total)'
              f'  |  Test: c101 + rand_rain×3 + rand_acc×2')
    elif args.benchmark == 'rc1-rand':
        _B = [f"rc{i:03d}" for i in range(102, 109)]
        _E = [f"{n}_rand_rain_{j:03d}" for n in _B for j in range(1, 11)]
        trainer_params['train_instances'] = _B + _E
        _T = [f"rc101_rand_rain_{j:03d}" for j in range(1, 4)] + \
             [f"rc101_rand_acc_{j:03d}"  for j in range(1, 3)]
        trainer_params['test_instances'] = ["rc101"] + _T
        print(f'[benchmark=rc1-rand] Train: rc102-rc108 + rand_rain×10/inst ({len(_B)+len(_E)} total)'
              f'  |  Test: rc101 + rand_rain×3 + rand_acc×2')

    if args.split == 'loose':
        # train c101-c108 (includes tight), test c109 (loosest TW)
        _BASE = ["c101","c102","c103","c104","c105","c106","c107","c108"]
        trainer_params['train_instances'] = _BASE
        trainer_params['test_instances']  = ["c109"]
        trainer_params['curriculum_epoch'] = 0
        print('[split=loose] Train: c101-c108  |  Test: c109')
    elif args.split == 'tight':
        # train c102-c109 (excludes tight), test c101 (tightest TW) — same as --base-only
        _BASE = ["c102","c103","c104","c105","c106","c107","c108","c109"]
        trainer_params['train_instances'] = _BASE
        trainer_params['test_instances']  = ["c101"]
        trainer_params['curriculum_epoch'] = 0
        print('[split=tight] Train: c102-c109  |  Test: c101')

    if args.base_only:
        if args.benchmark == 'rc1':
            _BASE = [f"rc{i:03d}" for i in range(102, 109)]
            trainer_params['train_instances'] = _BASE
            trainer_params['test_instances']  = ["rc101"]
            print('[base-only] Train: rc102-rc108  |  Test: rc101')
        elif args.benchmark == 'r1':
            _BASE = [f"r{i:03d}" for i in range(102, 113)]
            trainer_params['train_instances'] = _BASE
            trainer_params['test_instances']  = ["r101"]
            print('[base-only] Train: r102-r112  |  Test: r101')
        else:
            _BASE = ["c102","c103","c104","c105","c106","c107","c108","c109"]
            trainer_params['train_instances'] = _BASE
            trainer_params['test_instances']  = ["c101"]
            print('[base-only] Train: c102-c109  |  Test: c101')
        trainer_params['curriculum_epoch'] = 0
    if args.with_acc:
        _cur = trainer_params['train_instances']
        _base_cur = [n for n in _cur if '_rain_' not in n and '_acc_' not in n]
        _acc = [f"{n}_acc_A" for n in _base_cur] + [f"{n}_acc_B" for n in _base_cur]
        trainer_params['train_instances'] = _cur + _acc
        print(f'[with-acc] Added acc_A/acc_B → total train instances: {len(trainer_params["train_instances"])}')

    if args.events_only:
        _cur = trainer_params['train_instances']
        _events = [n for n in _cur if '_rain_' in n or '_acc_' in n]
        trainer_params['train_instances'] = _events
        print(f'[events-only] Removed base instances → train instances: {len(_events)}')

    if args.acc_only:
        _BASE = ["c102","c103","c104","c105","c106","c107","c108","c109"]
        _ACC  = [f"{n}_acc_A" for n in _BASE] + [f"{n}_acc_B" for n in _BASE]
        trainer_params['train_instances']  = _BASE + _ACC
        trainer_params['test_instances']   = ["c101", "c101_acc_A", "c101_acc_B"]
        trainer_params['curriculum_epoch'] = 500
        print('[acc-only] Train: c102-c109 base+acc(24개)  |  Test: c101 base+acc')
    # --train-dataset: override test_instances to cover all three Solomon types
    if args.train_dataset:
        trainer_params['test_instances'] = ['c101', 'r101', 'rc101']
        print('[Dataset] test_instances set: c101, r101, rc101')
    if args.test_instances:
        trainer_params['test_instances'] = args.test_instances
        print(f'[--test-instances] override: {args.test_instances}')
    if args.train_instances:
        trainer_params['train_instances'] = args.train_instances
        print(f'[--train-instances] override: {len(args.train_instances)} instances')
    if args.test_in_sample:
        # evaluate on the same instances used for training (in-sample check)
        base_train = [n for n in trainer_params['train_instances']
                      if '_rain_' not in n and '_acc_' not in n]
        trainer_params['test_instances'] = base_train
        print(f'[test-in-sample] Test set = training base instances: {base_train}')
    if args.top_k is not None:
        trainer_params['eval_top_k'] = args.top_k
        print(f'[Eval] Using {args.top_k} random starts (fair comparison mode)')
    if args.resume:
        trainer_params['model_load']['enable'] = True
        if args.resume.endswith('.pt'):
            trainer_params['model_load']['path']  = os.path.dirname(args.resume)
            epoch_str = os.path.basename(args.resume).replace('checkpoint-', '').replace('.pt', '')
            trainer_params['model_load']['epoch'] = epoch_str
        else:
            trainer_params['model_load']['path']  = args.resume
            trainer_params['model_load']['epoch'] = 'last'

    # Select benchmark-specific reward config
    if args.benchmark == 'r1':
        _cfg_table = _REWARD_CONFIGS_R1
    elif args.benchmark == 'rc1':
        _cfg_table = _REWARD_CONFIGS_RC1
    else:
        _cfg_table = _REWARD_CONFIGS
    cfg = dict(_cfg_table[args.config])  # copy so original is not mutated
    if args.d_max_scale != 1.0:
        cfg['D_max'] = cfg['D_max'] * args.d_max_scale
        print(f'[D_max] scaled by {args.d_max_scale} → D_max={cfg["D_max"]:.2f}')
    env_params.update(cfg)
    if args.pomo is not None:
        env_params['pomo_size'] = args.pomo
        print(f'[POMO] pomo_size overridden to {args.pomo}')
    if args.pomo_auto:
        trainer_params['pomo_auto'] = True
        print('[POMO] pomo_auto: pomo_size = ceil(total_demand/capacity) per instance')
    # config별로 별도 result 디렉토리에 저장
    if args.gen_type is not None:
        trainer_params['gen_type'] = args.gen_type
        trainer_params['gen_svc_t_corr'] = not args.no_svc_t_corr
        trainer_params['fixed_svc'] = args.fixed_svc
        trainer_params['train_instances'] = []   # unused when gen_type is set
        if args.n_customers_min is not None or args.n_customers_max is not None:
            n_min = args.n_customers_min or args.n_customers_max
            n_max = args.n_customers_max or args.n_customers_min
            trainer_params['gen_n_min'] = n_min
            trainer_params['gen_n_max'] = n_max
            print(f'[gen-type={args.gen_type}] n_customers sampled from [{n_min}, {n_max}] per episode')
        else:
            print(f'[gen-type={args.gen_type}] Random instance generation enabled (n_customers=100)')

    cfg_folder = f'config_{args.config}' if args.tag is None else f'config_{args.config}_{args.tag}'
    trainer_params['result_dir'] = os.path.join(RESULT_DIR, cfg_folder)
    _PRIORITY = {'A':'K>D>Lt','B':'K>Lt>D','C':'D>K>Lt','D':'D>Lt>K','E':'Lt>K>D','F':'Lt>D>K','G':'Lc+Lt>D>K','H':'Lt>D>K(lp=7)','J':'Lt>D~K(vp=3)','K':'D~Lt>K(Dmax=10)','RF':'dist-only+hardTW'}
    print(f'[Config {args.config}] priority={_PRIORITY[args.config]}  '
          f'D_max={cfg["D_max"]}  Lt_max={cfg["Lt_max"]}  '
          f'late={cfg["late_penalty"]}  vehicle={cfg["vehicle_penalty"]}')

    trainer = VRPTWTrainer(env_params, model_params, optimizer_params, trainer_params)

    if args.test_only:
        if args.resume and args.resume.endswith('.pt'):
            ckpt_path = args.resume
        elif args.resume:
            ckpt_path = os.path.join(args.resume, 'checkpoint-last.pt')
        else:
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
            plot_routes(inst, routes, best_reward,
                        os.path.join(plots_dir, f"{inst['name']}_best_route.png"))
            _save_solution(inst, routes, best_reward,
                           os.path.join(plots_dir, f"{inst['name']}_solution.txt"),
                           sol_routes)
    else:
        trainer.run()


if __name__ == '__main__':
    main()
