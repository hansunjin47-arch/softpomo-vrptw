"""
Pure VRPTW RL training — simultaneous C+R, no zone clustering.

Algorithms  (--algorithm / -a)
  ppo   : PPO with GAE critic baseline
  kool  : Kool et al. 2019 — REINFORCE + greedy rollout baseline
  pomo  : POMO (Kwon et al. 2020) — REINFORCE + multi-start mean baseline

Environment : CR_env.py
Model/utils : CR_rl_module.py
"""
from __future__ import annotations

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from CR_env import VRPTWEnv, load_solomon, build_tt
from CR_rl_module import (
    VRPTWAttentionModel, PPOModel,
    build_node_feats,
    collect_ppo_episode, ppo_update,
    collect_kool_episode,
    pomo_update_step,
    _greedy_eval,
    plot_routes, plot_reward_curve,
    save_ckpt, load_ckpt, set_seed,
)

# ============================================================
# ★ Dataset configuration — edit here
# ============================================================
TRAIN_INSTANCES = [
    "c102",
    "c103",
    "c104",
    "c105",
    "c106",
    "c107",
    "c108",
    "c109",
]

TEST_INSTANCES = [
    "c101",
]

# ============================================================
# Default hyperparameters
# ============================================================
DEFAULTS = dict(
    data_dir   = os.path.join(_ROOT, "data", "Solomon"),
    ckpt_dir   = os.path.join(_ROOT, "checkpoints", "pure_vrptw"),
    algorithm  = "pomo",
    seed       = 42,
    resume     = None,
    eval_only       = False,
    train_instances = TRAIN_INSTANCES,
    test_instances  = TEST_INSTANCES,

    # ── environment ────────────────────────────────────────
    visit_reward        = 2.0,
    unserved_penalty    = 20.0,   # >> vehicle_use_penalty to enforce: serve all > min vehicles
    vehicle_use_penalty = 10.0,   # charged once per vehicle dispatched
    dist_weight         = 3.0,    # multiplier on normalized travel penalty (tertiary)

    # ── model ──────────────────────────────────────────────
    embed_dim    = 128,
    n_heads      = 8,
    n_enc_layers = 3,
    lr           = 1e-4,
    max_steps    = 600,

    # ── PPO ────────────────────────────────────────────────
    ppo_episodes = 10_000,
    gamma        = 0.99,
    gae_lambda   = 0.95,
    clip_eps     = 0.2,
    value_coef   = 0.5,
    entropy_coef = 0.01,
    ppo_epochs   = 3,

    # ── Kool ───────────────────────────────────────────────
    kool_episodes      = 10_000,
    bl_update_interval = 50,

    # ── POMO ───────────────────────────────────────────────
    pomo_episodes = 10_000,  # total episodes (= updates × n_starts)
    n_starts      = 0,       # 0 = all feasible customers (paper default)

    # ── logging ────────────────────────────────────────────
    log_interval  = 100,
    save_interval = 1000,
)


# ============================================================
# Output formatting
# ============================================================

def _fmt(algo: str, ep_label: str, reward: float, env: VRPTWEnv,
         loss_label: str, loss_val: float, ep_time: float) -> str:
    served   = int(np.sum(env.served[1:]))
    unserved = env.N - served
    return (
        f"[{algo} {ep_label}]"
        f" reward={reward:10.2f}"
        f"  served={served:3d}"
        f"  unserved={unserved:3d}/{env.N}"
        f"  veh={env.vehicles_used:2d}/{env.max_vehicles:2d}"
        f"  dist={env.total_distance:10.2f}"
        f"  {loss_label}={loss_val:8.4f}"
        f"  ep_time={ep_time:.1f}s"
        f"  done={env.last_end_reason}"
    )


# ============================================================
# Training loops
# ============================================================

def train_ppo(env_pool, model: PPOModel, optimizer, node_feats_pool, cfg, device,
              start_ep, ckpt_dir):
    best_served, best_r = -1, -1e9
    best_line = ""
    reward_history = []
    t0 = time.time()

    for ep in range(start_ep, cfg["ppo_episodes"] + 1):
        idx = int(np.random.randint(len(env_pool)))
        env, node_feats_t = env_pool[idx], node_feats_pool[idx]

        model.train()
        ep_t0 = time.time()
        ep_reward, served, buf = collect_ppo_episode(env, model, node_feats_t, device, cfg)
        loss_val = ppo_update(model, optimizer, buf, node_feats_t, cfg, device)
        ep_time  = time.time() - ep_t0
        reward_history.append(ep_reward)

        is_best = served > best_served or (served == best_served and ep_reward > best_r)
        if is_best:
            best_served, best_r = served, ep_reward
            save_ckpt(os.path.join(ckpt_dir, "best.pt"), model, optimizer, ep, ep_reward)
            best_line = _fmt("PPO ", f"EP {ep:5d}|{env.inst.name}", ep_reward, env, "loss", loss_val, ep_time)
            print("[Best EP] " + best_line)

        if ep % cfg["save_interval"] == 0:
            save_ckpt(os.path.join(ckpt_dir, f"ep{ep}.pt"), model, optimizer, ep, ep_reward)

    total = time.time() - t0
    print(f"\n[Time] total={total:.0f}s  avg/ep={total/max(cfg['ppo_episodes'],1):.3f}s"
          f"  best_served={best_served}")
    if best_line:
        print("[Best EP] " + best_line)
    return reward_history


def train_kool(env_pool, model, optimizer, node_feats_pool, cfg, device,
               start_ep, ckpt_dir):
    baseline_model = copy.deepcopy(model)
    baseline_model.eval()

    best_served, best_r = -1, -1e9
    best_line = ""
    reward_history = []
    t0 = time.time()

    for ep in range(start_ep, cfg["kool_episodes"] + 1):
        idx = int(np.random.randint(len(env_pool)))
        env, node_feats_t = env_pool[idx], node_feats_pool[idx]

        model.train()
        ep_t0 = time.time()
        r_sample, sum_lp = collect_kool_episode(env, model, node_feats_t, device, cfg, greedy=False)

        ep_dist   = env.total_distance
        ep_veh    = env.vehicles_used
        ep_served = int(np.sum(env.served[1:]))
        ep_reason = env.last_end_reason
        ep_name   = env.inst.name

        with torch.no_grad():
            r_base, _ = collect_kool_episode(env, baseline_model, node_feats_t, device, cfg, greedy=True)

        advantage = r_sample - r_base
        if sum_lp is not None and abs(advantage) > 1e-8:
            loss = -advantage * sum_lp
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if ep % cfg["bl_update_interval"] == 0:
            baseline_model.load_state_dict(model.state_dict())
            baseline_model.eval()

        ep_time  = time.time() - ep_t0
        unserved = env.N - ep_served

        def _kool_line():
            return (
                f"[KOOL EP {ep:5d}|{ep_name}]"
                f" reward={r_sample:10.2f}  served={ep_served:3d}"
                f"  unserved={unserved:3d}/{env.N}  veh={ep_veh:2d}/{env.max_vehicles:2d}"
                f"  dist={ep_dist:10.2f}  adv={advantage:+8.4f}"
                f"  ep_time={ep_time:.1f}s  done={ep_reason}"
            )

        is_best = ep_served > best_served or (ep_served == best_served and r_sample > best_r)
        if is_best:
            best_served, best_r = ep_served, r_sample
            save_ckpt(os.path.join(ckpt_dir, "best.pt"), model, optimizer, ep, r_sample)
            best_line = _kool_line()
            print("[Best EP] " + best_line)

        reward_history.append(r_sample)
        if ep % cfg["save_interval"] == 0:
            save_ckpt(os.path.join(ckpt_dir, f"ep{ep}.pt"), model, optimizer, ep, r_sample)

    total = time.time() - t0
    print(f"\n[Time] total={total:.0f}s  avg/ep={total/max(cfg['kool_episodes'],1):.3f}s"
          f"  best_served={best_served}")
    if best_line:
        print("[Best EP] " + best_line)
    return reward_history


def train_pomo(env_pool, model, optimizer, node_feats_pool, start_pool_list, cfg, device,
               start_ep, ckpt_dir):
    best_r    = -1e9
    best_line = ""
    reward_history = []
    t0        = time.time()
    n_starts  = len(start_pool_list[0])          # uniform across c1xx instances
    n_updates = max(1, cfg["pomo_episodes"] // n_starts)
    start_upd = max(1, (start_ep - 1) // n_starts + 1)

    for upd in range(start_upd, n_updates + 1):
        idx          = int(np.random.randint(len(env_pool)))
        env          = env_pool[idx]
        node_feats_t = node_feats_pool[idx]
        start_pool   = start_pool_list[idx]

        model.train()
        rewards, _ = pomo_update_step(env, model, optimizer, node_feats_t, start_pool, device, cfg)
        ep = upd * n_starts
        reward_history.append(float(np.mean(rewards)))

        if upd % cfg["log_interval"] == 0:
            # Greedy eval on ALL instances → average as best metric
            ep_t0 = time.time()
            model.eval()
            gr_rs, serveds = [], []
            for ev, nf in zip(env_pool, node_feats_pool):
                ev.reset()
                with torch.no_grad():
                    h, h_bar = model.encode(nf)
                    gr_rs.append(_greedy_eval(ev, model, h, h_bar, device, cfg["max_steps"]))
                serveds.append(int(np.sum(ev.served[1:])))
            gr_r    = float(np.mean(gr_rs))
            served  = int(np.mean(serveds))
            ep_time = time.time() - ep_t0
            mean_r  = float(np.mean(rewards))

            def _pomo_line():
                per_inst = "  ".join(
                    f"{env_pool[i].inst.name}:{gr_rs[i]:.1f}(s{serveds[i]})"
                    for i in range(len(env_pool))
                )
                return (
                    f"[POMO UPD {upd:5d}/EP {ep:6d}]"
                    f" avg_reward={gr_r:8.2f}  avg_served={served}/{env_pool[0].N}"
                    f"  mean_R={mean_r:7.4f}  eval_time={ep_time:.1f}s"
                    f"\n    [{per_inst}]"
                )

            is_best = gr_r > best_r
            if is_best:
                best_r = gr_r
                save_ckpt(os.path.join(ckpt_dir, "best.pt"), model, optimizer, ep, gr_r)
                best_line = _pomo_line()
                print("[Best] " + best_line)

        if upd % max(1, cfg["save_interval"] // n_starts) == 0:
            save_ckpt(os.path.join(ckpt_dir, f"ep{ep}.pt"), model, optimizer, ep, float(np.mean(rewards)))

    total = time.time() - t0
    print(f"\n[Time] total={total:.0f}s  avg/ep={total/max(cfg['pomo_episodes'],1):.3f}s")
    if best_line:
        print("[Best] " + best_line)
    return reward_history


# ============================================================
# Post-training solution inspection
# ============================================================

def _greedy_routes(env, model, node_feats_t, device, cfg, start_pool=None):
    """
    POMO inference: run one greedy rollout per start node, return best.
    Falls back to single greedy rollout if start_pool is None.
    """
    with torch.no_grad():
        h, h_bar = model.encode(node_feats_t)

    if not start_pool:
        env.reset()
        with torch.no_grad():
            total_r = _greedy_eval(env, model, h, h_bar, device, cfg["max_steps"])
        best_env_snapshot = env
    else:
        best_r, best_env_snapshot = -1e9, None
        import copy
        for c_start in start_pool:
            env.reset()
            with torch.no_grad():
                r = _greedy_eval(env, model, h, h_bar, device, cfg["max_steps"],
                                 start_c=c_start)
            if r > best_r:
                best_r = r
                best_env_snapshot = copy.deepcopy(env)
        total_r = best_r
        env = best_env_snapshot

    routes = []
    for v in env.vehicles:
        if not v.activated:
            continue
        r = list(v.route)
        if r and r[-1] != 0:
            r.append(0)
        routes.append(r)
    return total_r, routes


def _load_sol(data_path: str, n_customers: int):
    """
    Try to find a Solomon solution file alongside data_path.
    Returns list of routes ([0, c1, c2, ..., 0]) or None.
    """
    import re
    base = os.path.splitext(data_path)[0]
    candidates = [
        base + ".sol",
        base + ".best",
        os.path.join(os.path.dirname(data_path), "solutions",
                     os.path.basename(base) + ".txt"),
        os.path.join(os.path.dirname(data_path), "solutions",
                     os.path.basename(base) + ".sol"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        routes = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if re.search(r'\d+\.\d+', line):        # skip float lines (metadata)
                    continue
                nums = list(map(int, re.findall(r'\b\d+\b', line)))
                cust = [n for n in nums if 1 <= n <= n_customers]
                if len(cust) >= 2:                       # need ≥2 stops to be a real route
                    routes.append([0] + cust + [0])
        if routes:
            print(f"[Solution] loaded from {path}")
            return routes
    return None


def _print_solution(inst, env, routes, total_reward, sol_routes=None):
    """
    Print per-route action sequence (from→to format) and
    optional comparison with a reference solution.
    """
    tt  = env.tt
    cap = inst.vehicle_capacity
    T   = float(inst.tw_close[0])

    total_dist = sum(float(tt[r[j], r[j+1]]) for r in routes for j in range(len(r)-1))
    n_served   = sum(len(r) - 2 for r in routes)

    W  = 75
    print("\n" + "=" * W)
    print("  BEST SOLUTION  (greedy decode)")
    print("=" * W)
    print(f"  vehicles={len(routes):2d}  served={n_served:3d}/{inst.n_customers}"
          f"  dist={total_dist:8.2f}  reward={total_reward:.4f}")

    if sol_routes is not None:
        sol_veh  = len(sol_routes)
        sol_dist = sum(float(tt[r[j], r[j+1]]) for r in sol_routes for j in range(len(r)-1))
        dveh  = len(routes) - sol_veh
        ddist = 100.0 * (total_dist - sol_dist) / max(sol_dist, 1e-8)
        print(f"  reference: vehicles={sol_veh:2d}  dist={sol_dist:8.2f}"
              f"  Δveh={dveh:+d}  Δdist={ddist:+.1f}%")

    print()
    for vi, route in enumerate(routes, 1):
        stops  = route[1:-1]
        rdist  = sum(float(tt[route[j], route[j+1]]) for j in range(len(route)-1))
        load   = sum(float(inst.demands[s]) for s in stops)
        print(f"  ── Vehicle {vi:2d}  [{len(stops):3d} stops | dist={rdist:7.1f}"
              f" | load={load:.0f}/{cap:.0f}]")

        t, prev = 0.0, 0
        for s in stops:
            travel    = float(tt[prev, s])
            arrival   = t + travel
            svc_start = max(arrival, float(inst.tw_open[s]))
            depart    = svc_start + float(inst.service_time[s])
            tw_ok     = " " if svc_start <= float(inst.tw_close[s]) + 1e-6 else " LATE"
            wait      = max(0.0, float(inst.tw_open[s]) - arrival)
            print(f"      {prev:3d} → {s:3d}"
                  f"  ({inst.coords[s,0]:5.1f},{inst.coords[s,1]:5.1f})"
                  f"  arr={arrival:6.1f}  wait={wait:4.1f}"
                  f"  TW=[{inst.tw_open[s]:5.1f},{inst.tw_close[s]:5.1f}]{tw_ok}"
                  f"  dem={inst.demands[s]:3.0f}")
            t, prev = depart, s

        ret_dist = float(tt[prev, 0])
        print(f"      {prev:3d} →   0  depot return  dist={ret_dist:.1f}"
              f"  arr={t+ret_dist:.1f}/{T:.0f}")
        print()
    print("=" * W)


# ============================================================
# CLI & main
# ============================================================

def _parse_args():
    import argparse
    D = DEFAULTS
    p = argparse.ArgumentParser(
        description="Pure VRPTW RL — simultaneous C+R",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval-only", action="store_true", default=False,
                   help="Skip training; load best checkpoint and evaluate on test set")
    p.add_argument("--algorithm", "-a", default=D["algorithm"], choices=["ppo", "kool", "pomo"])
    p.add_argument("--data-dir",  default=D["data_dir"])
    p.add_argument("--ckpt-dir",  default=D["ckpt_dir"])
    p.add_argument("--resume",    default=None)
    p.add_argument("--seed",         type=int,   default=D["seed"])
    p.add_argument("--lr",           type=float, default=D["lr"])
    p.add_argument("--embed-dim",    type=int,   default=D["embed_dim"])
    p.add_argument("--n-heads",      type=int,   default=D["n_heads"])
    p.add_argument("--n-enc-layers", type=int,   default=D["n_enc_layers"])
    p.add_argument("--max-steps",    type=int,   default=D["max_steps"])
    p.add_argument("--ppo-episodes",  type=int,   default=D["ppo_episodes"])
    p.add_argument("--gamma",         type=float, default=D["gamma"])
    p.add_argument("--gae-lambda",    type=float, default=D["gae_lambda"])
    p.add_argument("--clip-eps",      type=float, default=D["clip_eps"])
    p.add_argument("--value-coef",    type=float, default=D["value_coef"])
    p.add_argument("--entropy-coef",  type=float, default=D["entropy_coef"])
    p.add_argument("--ppo-epochs",    type=int,   default=D["ppo_epochs"])
    p.add_argument("--kool-episodes",      type=int, default=D["kool_episodes"])
    p.add_argument("--bl-update-interval", type=int, default=D["bl_update_interval"])
    p.add_argument("--pomo-episodes", type=int, default=D["pomo_episodes"])
    p.add_argument("--n-starts",      type=int, default=D["n_starts"])
    p.add_argument("--log-interval",  type=int, default=D["log_interval"])
    p.add_argument("--save-interval", type=int, default=D["save_interval"])
    p.add_argument("--visit-reward",        type=float, default=D["visit_reward"])
    p.add_argument("--unserved-penalty",    type=float, default=D["unserved_penalty"])
    p.add_argument("--vehicle-use-penalty", type=float, default=D["vehicle_use_penalty"])
    p.add_argument("--dist-weight",         type=float, default=D["dist_weight"])

    args = p.parse_args()

    def _resolve(name):
        if os.path.isfile(name):
            return name
        return os.path.join(args.data_dir, name if name.endswith(".txt") else name + ".txt")

    train_paths = [_resolve(n) for n in D["train_instances"]]
    test_paths  = [_resolve(n) for n in D["test_instances"]]

    return dict(
        algorithm=args.algorithm, train_paths=train_paths, test_paths=test_paths,
        data_dir=args.data_dir, ckpt_dir=args.ckpt_dir, resume=args.resume,
        seed=args.seed, eval_only=args.eval_only,
        lr=args.lr, embed_dim=args.embed_dim, n_heads=args.n_heads,
        n_enc_layers=args.n_enc_layers, max_steps=args.max_steps,
        visit_reward=args.visit_reward, unserved_penalty=args.unserved_penalty,
        vehicle_use_penalty=args.vehicle_use_penalty, dist_weight=args.dist_weight,
        ppo_episodes=args.ppo_episodes, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps, value_coef=args.value_coef, entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        kool_episodes=args.kool_episodes, bl_update_interval=args.bl_update_interval,
        pomo_episodes=args.pomo_episodes, n_starts=args.n_starts,
        log_interval=args.log_interval, save_interval=args.save_interval,
    )


def main():
    cfg    = _parse_args()
    algo   = cfg["algorithm"]
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_pool(paths):
        pool_inst, pool_env, pool_nf = [], [], []
        for dp in paths:
            inst = load_solomon(dp)
            tt   = build_tt(inst)
            env  = VRPTWEnv(
                inst=inst, tt=tt,
                visit_reward        = cfg["visit_reward"],
                unserved_penalty    = cfg["unserved_penalty"],
                vehicle_use_penalty = cfg["vehicle_use_penalty"],
                dist_weight         = cfg["dist_weight"],
            )
            nf = torch.tensor(build_node_feats(inst), dtype=torch.float32, device=device).unsqueeze(0)
            pool_inst.append(inst); pool_env.append(env); pool_nf.append(nf)
        return pool_inst, pool_env, pool_nf

    # ── Load train / test pools ───────────────────────────────────────────
    inst_pool,      env_pool,      node_feats_pool      = _load_pool(cfg["train_paths"])
    test_inst_pool, test_env_pool, test_node_feats_pool = _load_pool(cfg["test_paths"])

    # ── Model (all c1xx: same n_customers=100) ────────────────────────────
    n_nodes    = inst_pool[0].n_customers + 1
    ModelClass = PPOModel if algo == "ppo" else VRPTWAttentionModel
    model      = ModelClass(
        n_nodes=n_nodes, embed_dim=cfg["embed_dim"],
        n_heads=cfg["n_heads"], n_enc_layers=cfg["n_enc_layers"],
    ).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=cfg["lr"])

    ckpt_dir = os.path.join(cfg["ckpt_dir"], algo)
    os.makedirs(ckpt_dir, exist_ok=True)

    start_step = 1
    if cfg.get("resume"):
        start_step = load_ckpt(cfg["resume"], model, optimizer) + 1
        print(f"[Resume] step {start_step}")

    n_params   = sum(p.numel() for p in model.parameters())
    inst_names = [inst.name for inst in inst_pool]
    print(f"[{algo.upper()}] instances={inst_names}  N={inst_pool[0].n_customers}")
    print(f"[{algo.upper()}] n_nodes={n_nodes}  params={n_params:,}  device={device}\n")

    plots_dir = os.path.join(_HERE, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    tag       = f"{'_'.join(inst_names)}_{algo}" if len(inst_names) <= 3 else f"{inst_names[0]}-{inst_names[-1]}_{algo}"

    # ── Build start_pool for test instances (POMO inference) ─────────────
    def _build_start_pool(env_i, inst_i):
        env_i.reset()
        init_mask  = env_i.get_feasible_mask()
        all_starts = [c for c in range(1, inst_i.n_customers + 1) if init_mask[c]]
        cap = cfg["n_starts"]
        return all_starts if cap <= 0 else all_starts[:min(cap, len(all_starts))]

    test_start_pool_list = [
        _build_start_pool(env_i, inst_i)
        for env_i, inst_i in zip(test_env_pool, test_inst_pool)
    ] if algo == "pomo" else [None] * len(test_env_pool)

    # ── Eval-only mode: load best checkpoint, skip training ──────────────
    if cfg.get("eval_only"):
        best_ckpt = os.path.join(ckpt_dir, "best.pt")
        if not os.path.isfile(best_ckpt):
            print(f"[Eval] No checkpoint found at {best_ckpt}. Train first.")
            return
        load_ckpt(best_ckpt, model, optimizer)
        model.eval()
        print(f"[Eval] Loaded checkpoint: {best_ckpt}\n")
        for inst, env, nf, dp, sp in zip(
                test_inst_pool, test_env_pool, test_node_feats_pool,
                cfg["test_paths"], test_start_pool_list):
            total_r, routes = _greedy_routes(env, model, nf, device, cfg, start_pool=sp)
            sol_routes = _load_sol(dp, inst.n_customers)
            _print_solution(inst, env, routes, total_r, sol_routes)
            plot_routes(inst, routes, total_r,
                        os.path.join(plots_dir, f"{inst.name}_{algo}_route_eval.png"))
        return

    if algo == "ppo":
        reward_history = train_ppo(env_pool, model, optimizer, node_feats_pool, cfg, device, start_step, ckpt_dir)

    elif algo == "kool":
        reward_history = train_kool(env_pool, model, optimizer, node_feats_pool, cfg, device, start_step, ckpt_dir)

    elif algo == "pomo":
        start_pool_list = []
        for env_i, inst_i in zip(env_pool, inst_pool):
            env_i.reset()
            init_mask  = env_i.get_feasible_mask()
            all_starts = [c for c in range(1, inst_i.n_customers + 1) if init_mask[c]]
            cap        = cfg["n_starts"]
            sp = all_starts if cap <= 0 else all_starts[:min(cap, len(all_starts))]
            start_pool_list.append(sp)
        n_starts  = len(start_pool_list[0])
        n_updates = max(1, cfg["pomo_episodes"] // n_starts)
        print(f"[POMO] instances={len(inst_pool)}  n_starts={n_starts}  "
              f"episodes={cfg['pomo_episodes']:,}  updates={n_updates:,}\n")
        reward_history = train_pomo(env_pool, model, optimizer, node_feats_pool,
                                    start_pool_list, cfg, device, start_step, ckpt_dir)

    else:
        raise ValueError(f"Unknown algorithm: {algo!r}")

    # ── Reward curve ──────────────────────────────────────────────────────
    plot_reward_curve(reward_history, algo,
                      os.path.join(plots_dir, f"{tag}_reward_curve.png"))

    # ── Best solution: eval on every instance → print + plot ─────────────
    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    if os.path.isfile(best_ckpt):
        load_ckpt(best_ckpt, model, optimizer)
        model.eval()
        for inst, env, nf, dp, sp in zip(
                test_inst_pool, test_env_pool, test_node_feats_pool,
                cfg["test_paths"], test_start_pool_list):
            total_r, routes = _greedy_routes(env, model, nf, device, cfg, start_pool=sp)
            sol_routes = _load_sol(dp, inst.n_customers)
            _print_solution(inst, env, routes, total_r, sol_routes)
            plot_routes(inst, routes, total_r,
                        os.path.join(plots_dir, f"{inst.name}_{algo}_route_best.png"))


if __name__ == "__main__":
    main()
