"""Lexicographic-preference advantage for POMO-style REINFORCE training.

Replaces the scalarized weighted-sum reward ("R = -dist - w1*late - w2*veh")
with an explicit epsilon-constraint preference: solutions with late <= eps
("feasible") always outrank solutions with late > eps ("infeasible"),
regardless of any distance/vehicle-count difference. Within each group,
`hybrid_advantage` keeps real magnitude information (vehicle-count and
distance/lateness differences scale the advantage proportionally); the
simpler `lexicographic_advantage` uses average-tie rank instead, which is
easier to reason about but discards magnitude (see its docstring).

Both accept torch.Tensor inputs (any device) or numpy arrays; the returned
advantage tensor is placed on the same device as the `late` input (falls
back to CPU for plain numpy/list input).
"""
from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _out_device(x):
    return x.device if isinstance(x, torch.Tensor) else torch.device('cpu')


class RunningStd:
    # 보상 정규화의 누적 상태. 현재 선택된 advantage 설정에 따라 사용되며 재개 시 저장/복원한다.
    """EMA-tracked scale estimate, shared across training batches.

    hybrid_advantage's own warning applies here: normalizing by a SINGLE
    batch's own std lets a degenerate batch (all instances near-identical
    cost) blow up noise to full gradient scale. An EMA over many batches
    stays stable through any single degenerate batch.

    Usage: keep one instance on the trainer (`self._adv_running_std =
    RunningStd()`), pass it into hybrid_advantage every call -- it reads the
    OLD estimate to normalize the current batch, then updates itself with
    the current batch's raw scale for future calls (same convention as
    BatchNorm running stats: normalize with history, then update history).
    """

    def __init__(self, momentum: float = 0.99, init: float = 1.0):
        self.momentum = momentum
        self._value = init
        self._initialized = False

    @property
    def value(self) -> float:
        return self._value

    def update(self, batch_std) -> float:
        batch_std = float(batch_std)
        if not np.isfinite(batch_std) or batch_std <= 0:
            return self._value
        if not self._initialized:
            self._value = batch_std
            self._initialized = True
        else:
            self._value = self.momentum * self._value + (1.0 - self.momentum) * batch_std
        return self._value


def _lex_rank_with_ties(k1: np.ndarray, k2: np.ndarray, k3: np.ndarray) -> np.ndarray:
    """Lexicographic rank (k1 primary), tied only when all 3 keys match exactly.
    Ties get the average rank of their group. Lower is better (rank 0 = best)."""
    N = len(k1)
    order = np.lexsort((k3, k2, k1))          # last arg is primary
    keys = np.stack([k1[order], k2[order], k3[order]], axis=1)

    new_group = np.ones(N, dtype=bool)
    new_group[1:] = np.any(keys[1:] != keys[:-1], axis=1)
    group_id = np.cumsum(new_group) - 1

    pos = np.arange(N, dtype=np.float64)
    avg = (np.bincount(group_id, weights=pos) / np.bincount(group_id))[group_id]

    r = np.empty(N, dtype=np.float64)
    r[order] = avg
    return r


def lexicographic_advantage(late, veh, dist, eps: float = 0.0) -> torch.Tensor:
    """(B, N) late/veh/dist, all lower-is-better -> (B, N) advantage.

    Priority: feasible (late<=eps) always beats infeasible; within feasible,
    fewer vehicles wins, ties broken by distance; within infeasible, less
    lateness wins. Rank-based -> robust to outliers, but a rollout that finds
    a much-better solution gets the same advantage as one barely better (see
    hybrid_advantage for a version that preserves magnitude).
    """
    late_np = _to_numpy(late)
    veh_np  = _to_numpy(veh)
    dist_np = _to_numpy(dist)
    B, N = late_np.shape

    infeasible = (late_np > eps).astype(np.float64)
    k1 = infeasible
    k2 = np.where(infeasible > 0, late_np, veh_np)
    k3 = np.where(infeasible > 0, 0.0, dist_np)

    ranks = np.stack([_lex_rank_with_ties(k1[b], k2[b], k3[b]) for b in range(B)], axis=0)

    mean = ranks.mean(axis=1, keepdims=True)
    std  = ranks.std(axis=1, keepdims=True)
    adv  = -(ranks - mean) / (std + 1e-8)
    return torch.as_tensor(adv, dtype=torch.float32, device=_out_device(late))


def eps_schedule(current_epoch: int, trainer_params: dict) -> float:
    """Linear eps decay from advantage_eps_start to advantage_eps_final over
    advantage_eps_decay_epochs, then holds at the final value. Never decays
    all the way to a hard 0 unless eps_final is explicitly set to 0 --
    holding at a small positive floor keeps a permanent buffer so
    fewer-vehicle exploration isn't punished into the infeasible tier for
    every imperfect attempt (see design discussion: a fixed eps=0 let a
    from-scratch C1 run get stuck at K=12 with Lt=1195 -- reducing K became
    something the policy never tried because any imprecise attempt was
    immediately catastrophic under a zero-tolerance boundary).

    Falls back to the static trainer_params['advantage_eps'] (default 0.0)
    if advantage_eps_start/advantage_eps_final aren't both set -- so this is
    opt-in and backward compatible with the old fixed-eps behavior.
    """
    eps_start = trainer_params.get('advantage_eps_start')
    eps_final = trainer_params.get('advantage_eps_final')
    if eps_start is None or eps_final is None:
        return trainer_params.get('advantage_eps', 0.0)
    decay_epochs = max(1, trainer_params.get('advantage_eps_decay_epochs', 500))
    progress = min(1.0, max(0.0, current_epoch / decay_epochs))
    return eps_start + (eps_final - eps_start) * progress


def tier_counts(late, veh, dist, eps: float, unserved=None) -> dict:
    """(t0, t1, t2, total) rollout counts for the current batch, matching
    hybrid_advantage's own tier boundaries -- for monitoring whether training
    is stuck below the feasible tier (t2), which starves veh/dist of any
    gradient signal (see hybrid_advantage's docstring)."""
    late_np     = _to_numpy(late)
    unserved_np = None if unserved is None else _to_numpy(unserved)
    not_served  = (unserved_np > 0) if unserved_np is not None else np.zeros_like(late_np, dtype=bool)
    served      = ~not_served
    feas        = served & (late_np <= eps)
    infeas      = served & (late_np > eps)
    return dict(t0=int(not_served.sum()), t1=int(infeas.sum()), t2=int(feas.sum()),
                total=int(late_np.size))


def _place_tier(scores_row, mask, values, ceiling, clip):
    """Places a z-scored `values` (lower is better) into scores_row[mask],
    guaranteed entirely below `ceiling` (the tier-above's floor) by
    construction. Returns this tier's own floor (min score) so the caller
    can chain a THIRD, still-worse tier below it -- returns a synthetic
    anchor (ceiling - 2*clip - 1) even when `mask` is empty, so chaining
    stays correct regardless of which tiers are actually populated."""
    if mask.any():
        v = values[mask]
        z = (np.zeros_like(v) if v.std() < 1e-9
             else np.clip(-(v - v.mean()) / v.std(), -clip, clip))
        scores_row[mask] = ceiling - (clip + 1.0) + z
        return scores_row[mask].min()
    return ceiling - (clip + 1.0) - clip


def hybrid_advantage(
    late, veh, dist,
    eps: float = 0.0,
    A: float = 7.0,
    clip: float = 3.0,
    running_std: "RunningStd | float | None" = None,
    unserved=None,
) -> torch.Tensor:
    """(B, N) late/veh/dist, all lower-is-better -> (B, N) advantage.

    Three strict tiers, best to worst:
      1. fully served, feasible (late<=eps): score = -A*veh + clip(z(-dist))
      2. fully served, infeasible (late>eps): ranked by late, always below tier 1
      3. NOT fully served (unserved>0, from an episode truncated at max_steps
         before every customer was visited): ranked by unserved count, always
         below tier 2

    Tier 3 matters for from-scratch training: a freshly-initialized policy
    often can't complete a tour within max_steps, and late/veh/dist computed
    on a truncated rollout say nothing about that failure -- without this
    tier, hybrid_advantage has no way to push the policy toward completing
    routes at all (confirmed empirically: omitting it produced a flat,
    non-improving reward curve, num served customers never targeted by any
    fully-observed pass through this session's pool of trainers). Passing
    `unserved=None` (the previous default) skips tier 3 entirely -- safe
    only when the caller already knows every rollout completes (e.g.
    continuing from an already-converged checkpoint).

    Requiring A > 2*clip guarantees a 1-vehicle difference always dominates
    any possible distance difference within tier 1 -- verified at the
    tightest case (1-vehicle gap): score(veh+1) - score(veh) = -A + delta,
    delta in [-2*clip, 2*clip], so the gap is always < 0 iff A > 2*clip.
    Default A=7, clip=3 satisfies this with margin 1.

    `running_std` should be a RunningStd instance shared across the whole
    training run (see its docstring for why) -- pass a plain float only for
    a one-off / offline computation. None defaults to no normalization
    (divide by 1.0), which is only safe for inspection, not for training.
    """
    late_np     = _to_numpy(late)
    veh_np      = _to_numpy(veh)
    dist_np     = _to_numpy(dist)
    unserved_np = None if unserved is None else _to_numpy(unserved)
    B, N = late_np.shape

    s = np.zeros((B, N), dtype=np.float64)
    for b in range(B):
        not_served = (unserved_np[b] > 0) if unserved_np is not None else np.zeros(N, dtype=bool)
        served     = ~not_served
        feas       = served & (late_np[b] <= eps)
        infeas     = served & (late_np[b] > eps)

        if feas.any():
            d = dist_np[b][feas]
            zd = (np.zeros_like(d) if d.std() < 1e-9
                  else np.clip(-(d - d.mean()) / d.std(), -clip, clip))
            s[b][feas] = -A * veh_np[b][feas] + zd
            ceiling = s[b][feas].min()
        else:
            ceiling = 0.0

        ceiling = _place_tier(s[b], infeas, late_np[b], ceiling, clip)
        _place_tier(s[b], not_served, unserved_np[b] if unserved_np is not None else late_np[b],
                    ceiling, clip)

    s = s - s.mean(axis=1, keepdims=True)

    if isinstance(running_std, RunningStd):
        norm_std = running_std.value
        running_std.update(s.std())
    elif running_std is None:
        norm_std = 1.0
    else:
        norm_std = float(running_std)

    return torch.as_tensor(s / (norm_std + 1e-8), dtype=torch.float32, device=_out_device(late))
