"""
RL model architecture and training utilities for simultaneous C+R VRPTW.

Imported by CR_rl.py (training loops) and CR_llm_rl.py (inference).

Contents
--------
  Feature builders : build_node_feats, get_vehicle_feat
  Model classes    : VRPTWAttentionModel, PPOModel, PPORolloutBuffer
  Step helpers     : _get_decoder_inputs, _rollout_step, _greedy_eval
  Algorithm cores  : _collect_ppo_episode, _ppo_update
                     _collect_kool_episode
                     _single_pomo_rollout, _pomo_update_step
  Utilities        : save_ckpt, load_ckpt, set_seed
"""
from __future__ import annotations

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import copy
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from CR_env import VRPTWInstance, VRPTWEnv

# ── Static node features (encoded once) ──────────────────────────────────────
# x, y, tw_open, tw_close, service_time, demand
NODE_FEAT_DIM = 6

# ── Dynamic vehicle features (decoder context, changes each step) ─────────────
# remaining_cap, cur_time, served_ratio, vehicles_left_ratio
VEH_FEAT_DIM = 4


# ============================================================
# Feature builders
# ============================================================

def build_node_feats(inst: VRPTWInstance) -> np.ndarray:
    """Static [N+1, NODE_FEAT_DIM] node features for encoder input."""
    N           = inst.n_customers
    T           = float(inst.tw_close[0])
    coord_scale = max(float(np.max(np.abs(inst.coords))), 1.0)
    cap         = float(inst.vehicle_capacity)

    feats = np.zeros((N + 1, NODE_FEAT_DIM), dtype=np.float32)
    for i in range(N + 1):
        feats[i, 0] = float(inst.coords[i, 0])      / coord_scale
        feats[i, 1] = float(inst.coords[i, 1])      / coord_scale
        feats[i, 2] = float(inst.tw_open[i])        / T
        feats[i, 3] = float(inst.tw_close[i])       / T
        feats[i, 4] = float(inst.service_time[i])   / T
        feats[i, 5] = float(inst.demands[i])        / cap
    return feats


def get_vehicle_feat(env: VRPTWEnv) -> np.ndarray:
    """Dynamic [VEH_FEAT_DIM] vehicle features for decoder context."""
    av      = env.vehicles[env.active_idx]
    T       = env._T
    cap     = env._cap
    n_alive = sum(1 for v in env.vehicles if not v.retired)
    return np.array([
        av.remaining_cap / cap,
        av.cur_time / T,
        float(np.sum(env.served[1:])) / max(env.N, 1),  # served_ratio: problem progress
        n_alive / max(env.max_vehicles, 1),
    ], dtype=np.float32)


# ============================================================
# Attention Model (Kool et al. 2019)
# ============================================================

class VRPTWAttentionModel(nn.Module):
    """
    Encoder : Linear(node_feat) → L-layer Transformer → h^L_i, h̄
    Decoder : context = [h̄ ‖ h_last ‖ vehicle_proj]
              → W_ctx → glimpse MHA → pointer → C·tanh logits

    Static node features are encoded ONCE per instance.
    Dynamic state lives in the decoder context and feasibility mask.
    """

    def __init__(self, n_nodes: int, embed_dim: int = 128,
                 n_heads: int = 8, n_enc_layers: int = 3):
        super().__init__()
        self.n_nodes   = n_nodes
        self.embed_dim = embed_dim

        self.node_proj = nn.Linear(NODE_FEAT_DIM, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers,
                                               enable_nested_tensor=False)
        self.veh_proj = nn.Linear(VEH_FEAT_DIM, embed_dim)
        self.W_ctx    = nn.Linear(embed_dim * 3, embed_dim, bias=False)
        self.glimpse  = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.0)
        self.W_q_ptr  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k_ptr  = nn.Linear(embed_dim, embed_dim, bias=False)

        self.W_value    = nn.Linear(embed_dim * 3, embed_dim, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def encode(self, node_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """node_feats [B, N+1, NODE_FEAT_DIM] → h [B, N+1, E], h_bar [B, E]"""
        h = self.encoder(self.node_proj(node_feats))
        return h, h.mean(dim=1)

    def decode_logits(self, h, h_bar, last_idx, veh_feat, mask) -> torch.Tensor:
        B       = h.shape[0]
        h_last  = h[torch.arange(B, device=h.device), last_idx]
        veh_emb = self.veh_proj(veh_feat)
        ctx     = torch.cat([h_bar, h_last, veh_emb], dim=-1)
        q_g     = self.W_ctx(ctx).unsqueeze(1)
        key_pad = (mask == 0)
        h_g, _  = self.glimpse(q_g, h, h, key_padding_mask=key_pad)
        h_g     = h_g.squeeze(1)
        q_ptr   = self.W_q_ptr(h_g).unsqueeze(1)
        k_ptr   = self.W_k_ptr(h)
        score   = torch.bmm(q_ptr, k_ptr.transpose(1, 2)).squeeze(1) / (self.embed_dim ** 0.5)
        logits  = 10.0 * torch.tanh(score)
        logits  = logits + (mask == 0).float() * (-1e9)
        return logits

    def act(self, h, h_bar, last_idx, veh_feat, mask,
            greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.decode_logits(h, h_bar, last_idx, veh_feat, mask)
        dist   = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if greedy else dist.sample()
        return action, dist.log_prob(action)

    def log_prob_of(self, h, h_bar, last_idx, veh_feat, mask, action) -> torch.Tensor:
        logits = self.decode_logits(h, h_bar, last_idx, veh_feat, mask)
        return torch.distributions.Categorical(logits=logits).log_prob(action)


# ============================================================
# PPO model — value head extension
# ============================================================

class PPOModel(VRPTWAttentionModel):
    """Extends VRPTWAttentionModel with a value head for PPO."""

    def _context(self, h, h_bar, last_idx, veh_feat):
        B       = h.shape[0]
        h_last  = h[torch.arange(B, device=h.device), last_idx]
        veh_emb = self.veh_proj(veh_feat)
        return torch.cat([h_bar, h_last, veh_emb], dim=-1)

    def compute_value(self, h, h_bar, last_idx, veh_feat) -> torch.Tensor:
        return self.value_head(self.W_value(self._context(h, h_bar, last_idx, veh_feat))).squeeze(-1)

    def act_with_value(self, h, h_bar, last_idx, veh_feat, mask):
        logits = self.decode_logits(h, h_bar, last_idx, veh_feat, mask)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), self.compute_value(h, h_bar, last_idx, veh_feat)

    def evaluate_batch(self, h, h_bar, last_idxs, veh_feats, masks, actions):
        logits = self.decode_logits(h, h_bar, last_idxs, veh_feats, masks)
        dist   = Categorical(logits=logits)
        return dist.log_prob(actions), self.compute_value(h, h_bar, last_idxs, veh_feats), dist.entropy()


# ============================================================
# Rollout buffer (PPO)
# ============================================================

class PPORolloutBuffer:
    def __init__(self):
        self.last_idxs: List[int]        = []
        self.veh_feats: List[np.ndarray] = []
        self.masks:     List[np.ndarray] = []
        self.actions:   List[int]        = []
        self.log_probs: List[float]      = []
        self.values:    List[float]      = []
        self.rewards:   List[float]      = []
        self.dones:     List[float]      = []

    def push(self, last_idx, veh_feat, mask, action, log_prob, value, reward, done):
        self.last_idxs.append(int(last_idx))
        self.veh_feats.append(veh_feat)
        self.masks.append(mask)
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(float(done))

    def compute_gae(self, gamma, gae_lambda, last_val=0.0):
        n   = len(self.rewards)
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        vals = self.values + [last_val]
        for t in reversed(range(n)):
            delta  = self.rewards[t] + gamma * vals[t+1] * (1-self.dones[t]) - vals[t]
            gae    = delta + gamma * gae_lambda * (1-self.dones[t]) * gae
            adv[t] = gae
        return adv, adv + np.array(self.values, dtype=np.float32)

    def to_tensors(self, device):
        return (
            torch.tensor(self.last_idxs,            dtype=torch.long,    device=device),
            torch.tensor(np.array(self.veh_feats),  dtype=torch.float32, device=device),
            torch.tensor(np.array(self.masks),      dtype=torch.float32, device=device),
            torch.tensor(self.actions,              dtype=torch.long,    device=device),
            torch.tensor(self.log_probs,            dtype=torch.float32, device=device),
        )


# ============================================================
# Shared step helpers
# ============================================================

def _get_decoder_inputs(env: VRPTWEnv, device: str):
    av     = env.vehicles[env.active_idx]
    mask   = env.get_feasible_mask()
    last_t = torch.tensor([av.cur_node], device=device)
    veh_np = get_vehicle_feat(env)
    veh_t  = torch.tensor(veh_np, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.tensor(mask.astype(np.float32),     device=device).unsqueeze(0)
    return last_t, veh_t, mask_t, mask, veh_np


def _rollout_step(env, model, h, h_bar, device, greedy=False, forced_action=None):
    """One decoding step. Returns (action, log_prob, value, reward, done)."""
    last_t, veh_t, mask_t, mask, veh_np = _get_decoder_inputs(env, device)
    if not mask.any():
        return None, None, None, 0.0, True

    if forced_action is not None and mask[forced_action]:
        lp    = model.log_prob_of(h, h_bar, last_t, veh_t, mask_t,
                                  torch.tensor([forced_action], device=device)).squeeze()
        a_int = forced_action
        val   = None
    elif isinstance(model, PPOModel):
        ctx = torch.no_grad() if greedy else torch.enable_grad()
        with ctx:
            action, lp, val_t = model.act_with_value(h, h_bar, last_t, veh_t, mask_t)
        a_int = int(action.item())
        val   = float(val_t.item())
        lp    = lp.squeeze()
    else:
        if greedy:
            with torch.no_grad():
                action, lp = model.act(h, h_bar, last_t, veh_t, mask_t, greedy=True)
        else:
            action, lp = model.act(h, h_bar, last_t, veh_t, mask_t, greedy=False)
        a_int = int(action.item())
        val   = None
        lp    = lp.squeeze()

    _, reward, terminated, truncated, _ = env.step(a_int)
    return a_int, lp, val, reward, (terminated or truncated)


def _greedy_eval(env, model, h, h_bar, device, max_steps: int,
                 start_c: int | None = None) -> float:
    total = 0.0
    # POMO-style forced first action
    if start_c is not None:
        _, reward, terminated, truncated, _ = env.step(start_c)
        total += reward
        if terminated or truncated:
            return total
    for _ in range(max_steps):
        mask = env.get_feasible_mask()
        if not mask.any():
            break
        av     = env.vehicles[env.active_idx]
        last_t = torch.tensor([av.cur_node], device=device)
        veh_t  = torch.tensor(get_vehicle_feat(env), dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(mask.astype(np.float32), device=device).unsqueeze(0)
        action, _ = model.act(h, h_bar, last_t, veh_t, mask_t, greedy=True)
        _, reward, terminated, truncated, _ = env.step(int(action.item()))
        total += reward
        if terminated or truncated:
            break
    return total


# ============================================================
# PPO algorithm core
# ============================================================

def collect_ppo_episode(env, model: PPOModel, node_feats_t, device, cfg):
    env.reset()
    h, h_bar  = model.encode(node_feats_t)
    buf       = PPORolloutBuffer()
    ep_reward = 0.0

    for _ in range(cfg["max_steps"]):
        _, _, _, mask, veh_np = _get_decoder_inputs(env, device)
        if not mask.any():
            break
        action, lp, val, reward, done = _rollout_step(env, model, h, h_bar, device)
        if action is None:
            break
        ep_reward += reward
        buf.push(env.vehicles[env.active_idx].cur_node, veh_np, mask.astype(np.float32),
                 action, float(lp.item()), val, reward, float(done))
        if done:
            break

    return ep_reward, int(np.sum(env.served[1:])), buf


def ppo_update(model: PPOModel, optimizer, buf: PPORolloutBuffer,
               node_feats_t, cfg, device) -> float:
    adv_np, ret_np = buf.compute_gae(cfg["gamma"], cfg["gae_lambda"])
    adv = torch.tensor(adv_np, dtype=torch.float32, device=device)
    ret = torch.tensor(ret_np, dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    last_idxs, veh_feats, masks, actions, old_lps = buf.to_tensors(device)
    T = actions.shape[0]

    for _ in range(cfg["ppo_epochs"]):
        h_new, h_bar_new = model.encode(node_feats_t)
        h_exp     = h_new.expand(T, -1, -1)
        h_bar_exp = h_bar_new.expand(T, -1)
        lps, vals, entropy = model.evaluate_batch(h_exp, h_bar_exp, last_idxs, veh_feats, masks, actions)
        ratio = torch.exp(lps - old_lps)
        pg    = torch.max(
            -adv * ratio,
            -adv * torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]),
        )
        loss = pg.mean() + cfg["value_coef"] * F.mse_loss(vals, ret) - cfg["entropy_coef"] * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return float(loss.item())


# ============================================================
# Kool algorithm core
# ============================================================

def collect_kool_episode(env, model, node_feats_t, device, cfg, greedy=False):
    env.reset()
    ctx_mgr = torch.no_grad() if greedy else torch.enable_grad()
    with ctx_mgr:
        h, h_bar = model.encode(node_feats_t)

    log_probs:    List[torch.Tensor] = []
    total_reward = 0.0

    for _ in range(cfg["max_steps"]):
        a_int, lp, _, reward, done = _rollout_step(env, model, h, h_bar, device, greedy=greedy)
        if a_int is None:
            break
        total_reward += reward
        if not greedy:
            log_probs.append(lp)
        if done:
            break

    sum_lp = torch.stack(log_probs).sum() if log_probs and not greedy else None
    return total_reward, sum_lp


# ============================================================
# POMO algorithm core
# ============================================================

def single_pomo_rollout(env, model, h, h_bar, start_c, device, cfg):
    """One rollout with forced first action = start_c, sharing encoder h/h_bar."""
    env.reset()
    log_probs:   List[torch.Tensor] = []
    total_reward = 0.0

    a_int, lp, _, reward, done = _rollout_step(env, model, h, h_bar, device, forced_action=start_c)
    if lp is not None:
        log_probs.append(lp)
    total_reward += reward

    for _ in range(cfg["max_steps"]):
        if done:
            break
        a_int, lp, _, reward, done = _rollout_step(env, model, h, h_bar, device)
        if a_int is None:
            break
        log_probs.append(lp)
        total_reward += reward

    sum_lp = torch.stack(log_probs).sum() if log_probs else None
    return total_reward, sum_lp


def pomo_update_step(env, model, optimizer, node_feats_t, start_pool, device, cfg):
    """Encode once, run N rollouts, compute zero-mean advantage, single backward."""
    N        = len(start_pool)
    h, h_bar = model.encode(node_feats_t)

    rewards:     List[float]                  = []
    sum_lp_list: List[Optional[torch.Tensor]] = []

    for c_start in start_pool:
        reward, sum_lp = single_pomo_rollout(env, model, h, h_bar, c_start, device, cfg)
        rewards.append(reward)
        sum_lp_list.append(sum_lp)

    baseline = float(np.mean(rewards))

    optimizer.zero_grad()
    losses = []
    for i in range(N):
        if sum_lp_list[i] is None:
            continue
        adv = rewards[i] - baseline
        if abs(adv) > 1e-8:
            losses.append(-adv * sum_lp_list[i] / N)

    if losses:
        torch.stack(losses).sum().backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return rewards, baseline


# ============================================================
# Plotting utilities
# ============================================================

def plot_routes(inst, routes, total_reward: float, save_path: str) -> None:
    """Visualize best-episode routes and save to save_path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib 없음 — route plot 생략")
        return

    coords = inst.coords
    depot  = coords[0]
    colors = plt.cm.tab10.colors

    _, ax = plt.subplots(figsize=(12, 10))

    for vi, route in enumerate(routes):
        stops = route[1:-1]
        if not stops:
            continue
        color = colors[vi % len(colors)]
        xs = [coords[n][0] for n in route]
        ys = [coords[n][1] for n in route]
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.8)
        for j in range(len(route) - 1):
            x0, y0 = coords[route[j]]
            x1, y1 = coords[route[j+1]]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
        ax.scatter([coords[s][0] for s in stops],
                   [coords[s][1] for s in stops], color=color, s=20, zorder=3)
        for s in stops:
            ax.text(coords[s][0] + 0.3, coords[s][1] + 0.3, str(s),
                    fontsize=6, color=color)

    served_set = {s for r in routes for s in r[1:-1]}
    unvisited  = [n for n in range(1, inst.n_customers + 1) if n not in served_set]
    if unvisited:
        ax.scatter([coords[n][0] for n in unvisited],
                   [coords[n][1] for n in unvisited],
                   color="red", s=40, zorder=4, marker="x",
                   label=f"Unserved ({len(unvisited)})")

    ax.scatter(depot[0], depot[1], marker="*", s=300, color="black", zorder=5, label="Depot")
    ax.set_title(
        f"Best Solution — served={len(served_set)}/{inst.n_customers}"
        f"  vehicles={len(routes)}  reward={total_reward:.2f}",
        fontsize=12,
    )
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] route map saved → {save_path}")


def plot_reward_curve(history: list, algo: str, save_path: str) -> None:
    """Plot training reward convergence and save to save_path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib 없음 — reward curve 생략")
        return

    if not history:
        return

    eps  = list(range(1, len(history) + 1))
    window = min(50, max(1, len(history) // 20))
    mavg = [sum(history[max(0, i-window+1):i+1]) / min(i+1, window)
            for i in range(len(history))]

    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eps, history, color="lightgray", linewidth=0.6, label="reward")
    ax.plot(eps, mavg, color="steelblue", linewidth=2.0,
            label=f"moving avg (w={window})")
    ax.set_xlabel("Episode / Update"); ax.set_ylabel("Reward")
    ax.set_title(f"Training Convergence — {algo.upper()}  ({len(history)} steps)")
    ax.legend(); ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] reward curve saved → {save_path}")


# ============================================================
# Checkpoint & seed utilities
# ============================================================

def save_ckpt(path: str, model, optimizer, step: int, reward: float):
    torch.save({
        "step": step, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "reward": reward,
    }, path)


def load_ckpt(path: str, model, optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
