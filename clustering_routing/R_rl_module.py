"""
RL components: RolloutBuffer, AttentionActorCritic, PPO update, checkpointing,
plotting utilities, and BaselineEnv (rain-feature-augmented env for pure RL baseline).

Imported by main.py and rl_baseline.py.
"""
from __future__ import annotations

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from torch.distributions import Categorical

from R_env import WasteFleetEnv, EventType
from R_utils import TrainConfig, ZonePlan


# ============================================================
# 3) Rollout Buffer (PPO)
# ============================================================

class RolloutBuffer:
    """Stores transitions for one PPO rollout window."""

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.conf_vecs: List[np.ndarray] = []

    def push(self, obs, action, reward, done, value, log_prob, conf_vec=None):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.conf_vecs.append(conf_vec)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ============================================================
# 4) Attention Actor-Critic (Kool et al. 2019 inspired)
# ============================================================

class AttentionActorCritic(nn.Module):
    """Attention model following Kool et al. 2019 Appendix C.2, adapted for zone-based VRPTW.

    stop_feats layout: position 0 = depot, positions 1..S-1 = zone customers.
    Depot uses a separate input projection (W_0^x, b_0^x in Kool et al.).
    Decoder context = [h̄^N, h_{π_{t-1}}^N, D̂_t] where h_{π_{t-1}}^N is the
    last-visited node embedding retrieved via the is_cur_node flag (last stop feature).
    All logits (including depot) via pointer attention (tanh×10).
    Output shape: [batch, max_scope_size] where position 0 = depot.
    """

    def __init__(
        self,
        stop_feat_dim: int,
        max_scope_size: int,
        vehicle_feat_dim: int,
        global_feat_dim: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        event_feat_dim: int = 0,
        act_dim: int = 0,  # unused; kept for call-site compatibility
        llm_confidence_lambda_max: float = 1.0,
        learn_trust_gate: bool = True,
    ):
        super().__init__()
        self.stop_feat_dim = stop_feat_dim
        self.max_scope_size = max_scope_size
        self.vehicle_feat_dim = vehicle_feat_dim
        self.global_feat_dim = global_feat_dim
        self.event_feat_dim = event_feat_dim
        self.embed_dim = embed_dim

        self.max_rain_tokens = 32
        self.max_acc_tokens  = 16

        # ── Stop encoder (Kool Appendix C.2) ────────────────────────────────
        self.depot_input_proj = nn.Linear(stop_feat_dim, embed_dim)  # 8-D
        self.stop_input_proj  = nn.Linear(stop_feat_dim, embed_dim)  # 8-D
        self.rain_token_proj  = nn.Linear(4, embed_dim)   # rain tokens
        self.acc_token_proj   = nn.Linear(3, embed_dim)   # acc tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.stop_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # ── Vehicle / global encoder ─────────────────────────────────────────
        self.veh_encoder = nn.Sequential(
            nn.Linear(vehicle_feat_dim + global_feat_dim + event_feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ── Decoder projection: [h̄^N ‖ h_{π_{t-1}}^N ‖ D̂_t] → query ───────
        self.decoder_proj = nn.Linear(embed_dim * 3, embed_dim)

        # ── Pointer attention key projection (Kool et al. W_K) ───────────────
        self.ptr_key_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # ── Critic ────────────────────────────────────────────────────────────
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # ── LLM-confidence trust gate ────────────────────────────────────────
        # Learns a per-step scalar lambda_t in [0, lambda_max] that scales how
        # much the LLM confidence vector biases the action logits, conditioned
        # on the decoder context and a summary of the confidence vector itself.
        # If learn_trust_gate=False, lambda_t is fixed at lambda_max (the gate
        # network is still created but unused) — useful for isolating the
        # effect of the confidence bias itself from the learned-gate mechanism.
        self.lambda_max = llm_confidence_lambda_max
        self.learn_trust_gate = learn_trust_gate
        self.trust_gate = nn.Sequential(
            nn.Linear(embed_dim + 3, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def _split_obs(self, obs: torch.Tensor):
        B = obs.shape[0]
        stop_size = self.max_scope_size * self.stop_feat_dim
        rain_size = self.max_rain_tokens * 4
        acc_size  = self.max_acc_tokens  * 3
        offset = 0
        stop_feats  = obs[:, offset:offset+stop_size].view(B, self.max_scope_size, self.stop_feat_dim)
        offset += stop_size
        rain_tokens = obs[:, offset:offset+rain_size].view(B, self.max_rain_tokens, 4)
        offset += rain_size
        acc_tokens  = obs[:, offset:offset+acc_size].view(B, self.max_acc_tokens, 3)
        offset += acc_size
        veh_global  = obs[:, offset:]
        return stop_feats, rain_tokens, acc_tokens, veh_global

    def forward(self, obs: torch.Tensor, conf_vec: Optional[torch.Tensor] = None):
        stop_feats, rain_tokens, acc_tokens, veh_global = self._split_obs(obs)
        B = obs.shape[0]

        # Project stop features
        depot_proj = self.depot_input_proj(stop_feats[:, :1, :])   # [B, 1, D]
        cust_proj  = self.stop_input_proj(stop_feats[:, 1:, :])    # [B, S-1, D]
        stop_proj  = torch.cat([depot_proj, cust_proj], dim=1)     # [B, S, D]

        # Project event tokens (original_POMO style)
        rain_proj = self.rain_token_proj(rain_tokens)  # [B, R, D]
        acc_proj  = self.acc_token_proj(acc_tokens)    # [B, A, D]

        # Encode all together: [stop | rain_tokens | acc_tokens]
        seq      = torch.cat([stop_proj, rain_proj, acc_proj], dim=1)  # [B, S+R+A, D]
        encoded  = self.stop_encoder(seq)
        stop_emb = encoded[:, :self.max_scope_size, :]  # [B, S, D] — routing nodes only

        # Graph mean h̄^N (Kool decoder context)
        graph_mean = stop_emb.mean(dim=1)   # [B, D]

        # Last-visited node embedding h_{π_{t-1}}^N via is_cur_node flag (dim 5)
        is_cur = stop_feats[:, :, 5]                                               # [B, S]
        cur_idx = is_cur.argmax(dim=1)                                             # [B]
        last_node_emb = stop_emb[torch.arange(B, device=obs.device), cur_idx]     # [B, D]

        # Vehicle/global embedding D̂_t
        veh_emb = self.veh_encoder(veh_global)   # [B, D]

        # Decoder query [h̄^N ‖ h_{π_{t-1}}^N ‖ D̂_t]
        dec_input = torch.cat([graph_mean, last_node_emb, veh_emb], dim=-1)  # [B, 3D]
        dec_q = self.decoder_proj(dec_input)   # [B, D]

        # Pointer attention for all positions: depot (0) and customers (1..S-1)
        stop_keys = self.ptr_key_proj(stop_emb)   # [B, S, D]
        compat = torch.bmm(stop_keys, dec_q.unsqueeze(-1)).squeeze(-1)  # [B, S]
        compat = compat / (self.embed_dim ** 0.5)
        logits = torch.tanh(compat) * 10.0   # [B, S]

        # ── LLM-confidence trust gate ────────────────────────────────────────
        # lambda_t in [0, lambda_max] learned from decoder context + a summary
        # of the confidence vector, then used to bias the action logits.
        if conf_vec is None:
            conf_vec = torch.zeros(B, self.max_scope_size, device=obs.device, dtype=logits.dtype)
        if self.learn_trust_gate:
            conf_summary = torch.stack(
                [conf_vec.mean(dim=1), conf_vec.std(dim=1), conf_vec.max(dim=1).values],
                dim=-1,
            )  # [B, 3]
            lambda_t = self.lambda_max * torch.sigmoid(
                self.trust_gate(torch.cat([dec_q, conf_summary], dim=-1))
            )  # [B, 1]
        else:
            lambda_t = torch.full((B, 1), self.lambda_max, device=obs.device, dtype=logits.dtype)
        logits = logits + lambda_t * conf_vec

        value = self.critic_head(dec_q)
        return logits, value.squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        candidate_actions: List[int],  # scope-relative (0=depot, 1..S=scope stop)
        deterministic: bool = False,
        conf_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[int, float, float]:
        logits, value = self.forward(obs, conf_vec)
        logits = logits[0]

        mask = torch.full_like(logits, float("-inf"))
        for a in candidate_actions:
            mask[a] = 0.0
        masked_logits = logits + mask

        dist = Categorical(logits=masked_logits)
        if deterministic:
            action = int(masked_logits.argmax().item())
        else:
            action = int(dist.sample().item())

        log_prob = float(dist.log_prob(torch.tensor(action, device=obs.device)).item())
        return action, log_prob, float(value[0].item())

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,        # scope-relative action indices
        candidate_masks: torch.Tensor,
        conf_vecs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs, conf_vecs)
        masked_logits = logits + candidate_masks
        dist = Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


# ============================================================
# 13) PPO update
# ============================================================

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[float],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    advantages = []
    gae = 0.0
    values_ext = values + [last_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def compute_mc_returns(
    rewards: List[float],
    dones: List[float],
    last_value: float,
    gamma: float,
) -> List[float]:
    returns = []
    G = last_value
    for r, done in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1 - done)
        returns.insert(0, G)
    return returns


def kool_update(
    actor_critic: AttentionActorCritic,
    optimizer: optim.Optimizer,
    rollout: RolloutBuffer,
    candidate_masks_list: List[np.ndarray],
    r_stochastic: float,
    r_greedy: float,
    cfg: TrainConfig,
) -> float:
    """REINFORCE with greedy rollout baseline (Kool et al. 2019).

    Faithful to paper:
    - advantage = R_stochastic - R_greedy (episode-level scalar)
    - loss = -advantage * sum(log_probs)  (sum, not mean)
    - no entropy bonus
    - greedy rollout uses frozen baseline model (caller's responsibility)
    """
    advantage = r_stochastic - r_greedy

    obs_t     = torch.tensor(np.array(rollout.obs),             dtype=torch.float32, device=cfg.device)
    actions_t = torch.tensor(rollout.actions,                   dtype=torch.int64,   device=cfg.device)
    masks_t   = torch.tensor(np.array(candidate_masks_list),    dtype=torch.float32, device=cfg.device)
    conf_vecs_t = torch.tensor(np.array(rollout.conf_vecs),     dtype=torch.float32, device=cfg.device)

    log_probs, _, _ = actor_critic.evaluate(obs_t, actions_t, masks_t, conf_vecs_t)

    loss = -(advantage * log_probs.sum())

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
    optimizer.step()

    return float(loss.item())


def pomo_update(
    actor_critic: AttentionActorCritic,
    optimizer: optim.Optimizer,
    rollout_buffers: List[RolloutBuffer],
    candidate_masks_lists: List[List[np.ndarray]],
    total_rewards: List[float],
    cfg: TrainConfig,
    entropy_coef: Optional[float] = None,
    normalize_std: bool = False,
) -> float:
    """POMO-style REINFORCE: shared mean-reward baseline across K rollouts.

    Each rollout started from a different first customer (first-stop diversification).
    Advantage for rollout k = R_k - R̄  where R̄ = mean(R_1,..,R_K).
    If normalize_std=True (GRPO mode): advantage is also divided by std(R) + eps.
    Gradients from all K rollouts are accumulated before a single optimizer step.
    No value network loss — the shared baseline replaces it.
    """
    K = len(rollout_buffers)
    if K == 0:
        return 0.0

    R_bar = sum(total_rewards) / K
    if normalize_std and K > 1:
        r_std = (sum((r - R_bar) ** 2 for r in total_rewards) / K) ** 0.5
        r_denom = max(r_std, 1e-8)
    else:
        r_denom = 1.0
    ent_coef = entropy_coef if entropy_coef is not None else cfg.entropy_coef

    optimizer.zero_grad()
    total_loss = 0.0
    valid_k = 0

    for k in range(K):
        buf = rollout_buffers[k]
        masks = candidate_masks_lists[k]
        if len(buf) == 0:
            continue
        valid_k += 1

        advantage_k = (total_rewards[k] - R_bar) / r_denom

        obs_t    = torch.tensor(np.array(buf.obs),    dtype=torch.float32, device=cfg.device)
        actions_t = torch.tensor(buf.actions,          dtype=torch.int64,   device=cfg.device)
        masks_t  = torch.tensor(np.array(masks),       dtype=torch.float32, device=cfg.device)
        conf_vecs_t = torch.tensor(np.array(buf.conf_vecs), dtype=torch.float32, device=cfg.device)

        log_probs, _, entropy = actor_critic.evaluate(obs_t, actions_t, masks_t, conf_vecs_t)

        # Episode-level advantage × mean step log-prob
        # (mean instead of sum keeps gradient scale independent of episode length;
        #  existing hyperparameters lr/entropy_coef are calibrated for this scale)
        actor_loss = -advantage_k * log_probs.mean()
        loss_k = actor_loss - ent_coef * entropy.mean()
        (loss_k / K).backward()   # accumulate; divide by K to average across rollouts
        total_loss += float(loss_k.item())

    if valid_k > 0:
        nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
        optimizer.step()

    return total_loss / max(valid_k, 1)


def pomo_ppo_update(
    actor_critic: AttentionActorCritic,
    optimizer: optim.Optimizer,
    rollout_buffers: List[RolloutBuffer],
    candidate_masks_lists: List[List[np.ndarray]],
    cfg: TrainConfig,
    entropy_coef: Optional[float] = None,
) -> float:
    """POMO with PPO: K diverse rollouts + GAE step-level advantage + clipped objective.

    Combines POMO's starting-point diversity with PPO's step-level credit assignment.
    Each rollout gets its own GAE, then all K rollouts are used for PPO epochs.
    """
    K = len(rollout_buffers)
    if K == 0:
        return 0.0
    ent_coef = entropy_coef if entropy_coef is not None else cfg.entropy_coef

    # Pre-compute GAE for each rollout
    all_obs, all_actions, all_old_lp, all_adv, all_ret, all_masks = [], [], [], [], [], []
    for k in range(K):
        buf = rollout_buffers[k]
        masks = candidate_masks_lists[k]
        if len(buf) == 0:
            continue
        advantages, returns = compute_gae(
            buf.rewards, buf.values, buf.dones, 0.0, cfg.gamma, cfg.gae_lambda,
        )
        all_obs.append(np.array(buf.obs))
        all_actions.append(buf.actions)
        all_old_lp.append(buf.log_probs)
        all_adv.append(advantages)
        all_ret.append(returns)
        all_masks.append(np.array(masks))

    if not all_obs:
        return 0.0

    obs_t     = torch.tensor(np.concatenate(all_obs),     dtype=torch.float32, device=cfg.device)
    actions_t = torch.tensor(np.concatenate(all_actions), dtype=torch.int64,   device=cfg.device)
    old_lp_t  = torch.tensor(np.concatenate(all_old_lp),  dtype=torch.float32, device=cfg.device)
    adv_t     = torch.tensor(np.concatenate(all_adv),     dtype=torch.float32, device=cfg.device)
    ret_t     = torch.tensor(np.concatenate(all_ret),     dtype=torch.float32, device=cfg.device)
    masks_t   = torch.tensor(np.concatenate(all_masks),   dtype=torch.float32, device=cfg.device)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = obs_t.shape[0]
    total_loss = 0.0
    n_updates = 0

    for _ in range(cfg.ppo_epochs):
        idx = torch.randperm(n, device=cfg.device)
        for start in range(0, n, cfg.ppo_batch_size):
            mb = idx[start:start + cfg.ppo_batch_size]
            log_probs, values, entropy = actor_critic.evaluate(obs_t[mb], actions_t[mb], masks_t[mb])
            ratio = torch.exp(log_probs - old_lp_t[mb])
            adv_mb = adv_t[mb]

            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_mb
            actor_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, ret_t[mb])
            loss = actor_loss + cfg.value_coef * value_loss - ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_updates += 1

    return total_loss / max(n_updates, 1)


def ppo_update(
    actor_critic: AttentionActorCritic,
    optimizer: optim.Optimizer,
    rollout: RolloutBuffer,
    candidate_masks_list: List[np.ndarray],
    last_value: float,
    cfg: TrainConfig,
    entropy_coef: Optional[float] = None,
) -> float:
    advantages, returns = compute_gae(
        rollout.rewards, rollout.values, rollout.dones,
        last_value, cfg.gamma, cfg.gae_lambda,
    )

    obs_t = torch.tensor(np.array(rollout.obs), dtype=torch.float32, device=cfg.device)
    actions_t = torch.tensor(rollout.actions, dtype=torch.int64, device=cfg.device)
    old_logp_t = torch.tensor(rollout.log_probs, dtype=torch.float32, device=cfg.device)
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=cfg.device)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=cfg.device)
    masks_t = torch.tensor(np.array(candidate_masks_list), dtype=torch.float32, device=cfg.device)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = len(rollout)
    total_loss = 0.0
    n_updates = 0

    for _ in range(cfg.ppo_epochs):
        idx = torch.randperm(n, device=cfg.device)
        for start in range(0, n, cfg.ppo_batch_size):
            mb = idx[start:start + cfg.ppo_batch_size]
            log_probs, values, entropy = actor_critic.evaluate(
                obs_t[mb], actions_t[mb], masks_t[mb]
            )
            ratio = torch.exp(log_probs - old_logp_t[mb])
            adv_mb = adv_t[mb]

            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_mb
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, ret_t[mb])

            ent_coef = entropy_coef if entropy_coef is not None else cfg.entropy_coef
            loss = actor_loss + cfg.value_coef * value_loss - ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_updates += 1

    return total_loss / max(n_updates, 1)


# ============================================================
# 15) Checkpoint save / load
# ============================================================

def save_checkpoint(
    path: str,
    ep: int,
    global_step: int,
    actor_critic: AttentionActorCritic,
    optimizer: optim.Optimizer,
    ontology_engine=None,
    best_served: int = -1,
    best_ep_reward: float = float("-inf"),
    best_late_cnt: int = 0,
) -> None:
    import pickle
    ckpt = {
        "ep": ep,
        "global_step": global_step,
        "actor_critic": actor_critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_served": best_served,
        "best_ep_reward": best_ep_reward,
        "best_late_cnt": best_late_cnt,
        "ontology_unserved_counts": (
            ontology_engine.unserved_counts.copy() if ontology_engine else None
        ),
        "ontology_penalty_multipliers": (
            ontology_engine.penalty_multipliers.copy() if ontology_engine else None
        ),
        "ontology_episode_count": (
            ontology_engine.episode_count if ontology_engine else None
        ),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"[Checkpoint] saved → {path}  (ep={ep}, global_step={global_step})")


def load_checkpoint(
    path: str,
    actor_critic: AttentionActorCritic,
    optimizer: optim.Optimizer,
    ontology_engine,
    device: str,
) -> Tuple[int, int]:
    import pickle
    with open(path, "rb") as f:
        ckpt = pickle.load(f)

    try:
        actor_critic.load_state_dict(ckpt["actor_critic"])
    except RuntimeError as e:
        raise RuntimeError(
            f"[Checkpoint] Architecture mismatch loading '{path}'. "
            f"Likely saved with old MLP actor head (act_dim=N+1) but current model uses "
            f"pointer attention (act_dim=max_scope_size+1). Train from scratch.\n"
            f"Detail: {e}"
        ) from e
    actor_critic.to(device)
    optimizer.load_state_dict(ckpt["optimizer"])

    if ontology_engine is not None and ckpt.get("ontology_unserved_counts") is not None:
        ontology_engine.unserved_counts = ckpt["ontology_unserved_counts"]
        ontology_engine.penalty_multipliers = ckpt["ontology_penalty_multipliers"]
        ontology_engine.episode_count = ckpt["ontology_episode_count"]

    start_ep = int(ckpt["ep"]) + 1
    global_step = int(ckpt["global_step"])
    best_served = int(ckpt.get("best_served", -1))
    best_ep_reward = float(ckpt.get("best_ep_reward", float("-inf")))
    best_late_cnt = int(ckpt.get("best_late_cnt", 0))
    print(f"[Checkpoint] loaded ← {path}  (resume from ep={start_ep}, global_step={global_step}, best_served={best_served})")
    return start_ep, global_step, best_served, best_ep_reward, best_late_cnt


# ============================================================
# Plotting utilities
# ============================================================

def _plot_routing(routes: List[List[int]], env: WasteFleetEnv, cfg: TrainConfig) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib 없음 — routing plot 생략")
        return

    coords = env.inst.coords
    depot = coords[0]

    _, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.tab10.colors
    used_vehicles = [(i, r) for i, r in enumerate(routes) if len(r) > 2]

    for veh_idx, route in used_vehicles:
        color = colors[veh_idx % len(colors)]
        xs = [coords[n][0] for n in route]
        ys = [coords[n][1] for n in route]

        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.8)

        for i in range(len(route) - 1):
            x0, y0 = coords[route[i]]
            x1, y1 = coords[route[i + 1]]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                        )

        for n in route[1:-1]:
            ax.text(coords[n][0] + 0.3, coords[n][1] + 0.3, str(n),
                    fontsize=6, color=color)

        cust_x = [coords[n][0] for n in route[1:-1]]
        cust_y = [coords[n][1] for n in route[1:-1]]
        ax.scatter(cust_x, cust_y, color=color, s=20, zorder=3)

    visited = {n for _, r in used_vehicles for n in r[1:-1]}
    unvisited = [n for n in range(1, env.N + 1) if n not in visited]
    if unvisited:
        ux = [coords[n][0] for n in unvisited]
        uy = [coords[n][1] for n in unvisited]
        ax.scatter(ux, uy, color="red", s=40, zorder=4, marker="x", label=f"Unvisited ({len(unvisited)})")
        for n in unvisited:
            ax.text(coords[n][0] + 0.3, coords[n][1] + 0.3, str(n), fontsize=6, color="red")

    ax.scatter(depot[0], depot[1], marker="*", s=300, color="black", zorder=5, label="Depot")
    ax.text(depot[0] + 0.3, depot[1] + 0.3, "Depot", fontsize=8, fontweight="bold")

    served = sum(len(r) - 2 for _, r in used_vehicles)
    ax.set_title(
        f"Best Episode Routing — served={served}/{env.N}, vehicles={len(used_vehicles)}/{env.max_vehicles}",
        fontsize=12,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plots_dir = os.path.join(_HERE, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    tag  = os.path.splitext(os.path.basename(cfg.checkpoint_path))[0] if cfg.checkpoint_path else "run"
    save_path = os.path.join(plots_dir, f"{tag}_routing_best.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] routing saved → {save_path}")


def _plot_reward_curve(ep_reward_history: List[float], cfg: TrainConfig) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib 없음 — reward plot 생략")
        return

    if not ep_reward_history:
        return

    episodes = list(range(1, len(ep_reward_history) + 1))
    rewards = ep_reward_history

    window = min(20, len(rewards) // 5 or 1)
    moving_avg = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        moving_avg.append(sum(rewards[start:i + 1]) / (i - start + 1))

    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episodes, rewards, color="lightgray", linewidth=0.8, label="Episode reward")
    ax.plot(episodes, moving_avg, color="steelblue", linewidth=2.0,
            label=f"Moving avg (window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(f"Reward Convergence (PPO+Attention) — {cfg.num_episodes} episodes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plots_dir = os.path.join(_HERE, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    tag  = os.path.splitext(os.path.basename(cfg.checkpoint_path))[0] if cfg.checkpoint_path else "run"
    save_path = os.path.join(plots_dir, f"{tag}_reward_curve.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] reward curve saved → {save_path}")


# ============================================================
# BaselineEnv — rain features embedded in state (pure RL baseline)
# ============================================================

class BaselineEnv(WasteFleetEnv):
    """WasteFleetEnv augmented with event tokens and demand/capacity features.

    Per-stop dims (stop_feat_dim = 9):
      dim 0: start_slack   (tw_open - arrival) / T
      dim 1: end_slack     (tw_close - arrival) / T
      dim 2: travel        travel_time_from_cur / T
      dim 3: depot_return_slack  / T
      dim 4: served        0/1
      dim 5: is_cur_node   1 if this stop is currently occupied
      dim 6: x             normalised coordinate
      dim 7: y             normalised coordinate
      dim 8: demand_norm   demand / vehicle_capacity

    Vehicle feat dims (vehicle_feat_dim = 4):
      dim 0: zone_norm
      dim 1: retired       0/1
      dim 2: cur_time_norm
      dim 3: remaining_cap_norm  remaining_cap / vehicle_capacity

    Event tokens (separate encoder sequence positions):
      rain_tokens: (MAX_RAIN_TOKENS, 4) = [node_rel/scope, mm/100, t_start/T, t_end/T]
      acc_tokens:  (MAX_ACC_TOKENS,  3) = [node_rel/scope, t_start/T, t_end/T]
    """

    STOP_EVENT_DIM  = 0
    GLOBAL_EVENT_DIM = 0
    MAX_RAIN_TOKENS = 32
    MAX_ACC_TOKENS  = 16

    def __init__(self, *args, **kwargs):
        # Set dims before super().__init__() so _build_obs() (called inside reset())
        # uses the correct shape from the very first call.
        self.stop_feat_dim    = 9   # adds demand/capacity vs parent's 8
        self.vehicle_feat_dim = 4   # adds remaining_cap vs parent's 3
        super().__init__(*args, **kwargs)
        self.event_feat_dim   = self.GLOBAL_EVENT_DIM
        self.rain_token_size  = self.MAX_RAIN_TOKENS * 4
        self.acc_token_size   = self.MAX_ACC_TOKENS  * 3
        obs_dim = (
            self.max_scope_size * self.stop_feat_dim
            + self.rain_token_size
            + self.acc_token_size
            + self.vehicle_feat_dim
            + self.global_feat_dim
        )
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _build_obs(self) -> np.ndarray:
        depot_due = float(self.inst.tw_close[0])
        av = self.vehicles[self.active_vehicle_idx]
        cur_node = int(av.cur_node)

        rain_evs = [e for e in self.active_events
                    if e.event_type == EventType.RAIN and e.end_time > av.cur_time]
        acc_evs  = [e for e in self.active_events
                    if e.event_type == EventType.ACCIDENT and e.end_time > av.cur_time]

        scope_custs = self._get_scope_customers()
        n_stop_dims = self.stop_feat_dim   # base features only (events are separate tokens)
        stop_feats = np.zeros((self.max_scope_size, n_stop_dims), dtype=np.float32)
        tt_curr_row = self.tt_base[av.cur_node]
        tt_to_depot = self.tt_base[:, 0]

        cap = max(float(self.inst.vehicle_capacity), 1.0)

        # Position 0: depot
        depot_travel  = float(tt_curr_row[0])
        depot_arrival = av.cur_time + depot_travel
        stop_feats[0, 0] = self._norm_time(float(self.inst.tw_open[0]) - depot_arrival)
        stop_feats[0, 1] = self._norm_time(depot_due - depot_arrival)
        stop_feats[0, 2] = self._norm_time(depot_travel)
        stop_feats[0, 3] = 0.0
        stop_feats[0, 4] = 0.0
        stop_feats[0, 5] = 1.0 if cur_node == 0 else 0.0
        stop_feats[0, 6] = self._norm_coord(float(self.inst.coords[0, 0]))
        stop_feats[0, 7] = self._norm_coord(float(self.inst.coords[0, 1]))
        stop_feats[0, 8] = 0.0   # depot demand = 0

        # Positions 1..max_scope_size-1: zone customers
        for i, cust in enumerate(scope_custs[:self.max_scope_size - 1]):
            travel = float(tt_curr_row[cust])
            arrival = av.cur_time + travel
            start_slack        = float(self.inst.tw_open[cust]) - arrival
            depart             = max(arrival, float(self.inst.tw_open[cust])) + float(self.inst.service_time[cust])
            end_slack          = float(self.inst.tw_close[cust]) - arrival
            depot_return_slack = depot_due - depart - float(tt_to_depot[cust])

            stop_feats[i + 1, 0] = self._norm_time(start_slack)
            stop_feats[i + 1, 1] = self._norm_time(end_slack)
            stop_feats[i + 1, 2] = self._norm_time(travel)
            stop_feats[i + 1, 3] = self._norm_time(depot_return_slack)
            stop_feats[i + 1, 4] = float(self.served[cust])
            stop_feats[i + 1, 5] = 1.0 if cust == cur_node else 0.0
            stop_feats[i + 1, 6] = self._norm_coord(float(self.inst.coords[cust, 0]))
            stop_feats[i + 1, 7] = self._norm_coord(float(self.inst.coords[cust, 1]))
            stop_feats[i + 1, 8] = float(self.inst.demands[cust]) / cap

        # Build rain tokens: [node_rel/scope, mm/100, t_start/T, t_end/T]
        rain_toks = np.zeros((self.MAX_RAIN_TOKENS, 4), dtype=np.float32)
        tok_idx = 0
        scope_set = set(scope_custs)
        for ev in rain_evs:
            mm  = ev.rainfall_mm / 100.0
            t_s = max(0.0, ev.trigger_time - av.cur_time) / max(depot_due, 1.0)
            t_e = ev.end_time / max(depot_due, 1.0)
            for node in ev.affected_nodes:
                if node in scope_set and tok_idx < self.MAX_RAIN_TOKENS:
                    rel = (scope_custs.index(node) + 1) / max(self.max_scope_size, 1)
                    rain_toks[tok_idx] = [rel, mm, t_s, t_e]
                    tok_idx += 1

        # Build acc tokens: [node_rel/scope, t_start/T, t_end/T]
        acc_toks = np.zeros((self.MAX_ACC_TOKENS, 3), dtype=np.float32)
        tok_idx = 0
        for ev in acc_evs:
            t_s = max(0.0, ev.trigger_time - av.cur_time) / max(depot_due, 1.0)
            t_e = ev.end_time / max(depot_due, 1.0)
            for node in ev.affected_nodes:
                if tok_idx < self.MAX_ACC_TOKENS:
                    rel = ((scope_custs.index(node) + 1) / max(self.max_scope_size, 1)
                           if node in scope_set else 0.0)
                    acc_toks[tok_idx] = [rel, t_s, t_e]
                    tok_idx += 1

        vehicle_feat = np.array(
            [
                self._norm_zone(int(av.active_zone_idx)),
                float(av.retired),
                self._norm_time(float(av.cur_time)),
                float(av.remaining_cap) / cap,   # remaining capacity (normalised)
            ],
            dtype=np.float32,
        )
        global_feat = np.array(
            [self._norm_time(depot_due - av.cur_time)],
            dtype=np.float32,
        )

        return np.concatenate(
            [stop_feats.reshape(-1), rain_toks.reshape(-1), acc_toks.reshape(-1),
             vehicle_feat, global_feat]
        ).astype(np.float32)
