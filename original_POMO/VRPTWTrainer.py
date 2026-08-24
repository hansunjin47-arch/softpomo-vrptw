"""
VRPTW Trainer — based on POMO CVRPTrainer.
Multi-instance Solomon training: randomly sample one instance per batch.
Output format matches CR_rl.py (per-route detail, route plots, score curve).
"""
from __future__ import annotations

import os
import re
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from VRPTWEnv import VRPTWEnv as Env
from VRPTWModel import VRPTWModel as Model
from VRPTWProblemDef import load_solomon, make_batch

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False


# ── Event helpers (no LLM dependency) ────────────────────────────────────────

def _get_rain_event(inst: dict, weather_prob: float) -> tuple:
    """
    Return (rain_nodes, multiplier) from preset events or random generation.
    Returns ([], 1.0) if no rain this episode.
    """
    rain_evs = [e for e in inst.get('preset_events', []) if e['type'] == 'RAIN']
    if rain_evs:
        nodes = sorted({n for e in rain_evs for n in e['nodes']})
        mult  = max(e['multiplier'] for e in rain_evs)
        return nodes, mult
    if random.random() > weather_prob:
        return [], 1.0
    N     = inst['n_customers']
    k     = random.randint(5, min(20, N))
    nodes = random.sample(range(1, N + 1), k)
    mult  = random.choice([1.5, 2.0, 2.5])
    return nodes, mult


def _get_accident_event(
    inst: dict, accident_prob: float, trigger_step: int, duration: int
) -> dict | None:
    """
    Return accident dict {node_a, node_b, multiplier, trigger_step, duration}
    from preset events or random generation. Returns None if no accident.
    """
    acc_evs = [e for e in inst.get('preset_events', []) if e['type'] == 'ACCIDENT']
    if acc_evs:
        e = acc_evs[0]
        nodes = e['nodes']
        return dict(
            node_a       = nodes[0] if len(nodes) >= 1 else 0,
            node_b       = nodes[1] if len(nodes) >= 2 else 0,
            multiplier   = e['multiplier'],
            trigger_step = trigger_step,
            duration     = int(e['duration']),
        )
    if random.random() > accident_prob:
        return None
    N      = inst['n_customers']
    node_a = random.randint(0, N)
    node_b = random.randint(0, N)
    while node_b == node_a:
        node_b = random.randint(0, N)
    return dict(
        node_a       = node_a,
        node_b       = node_b,
        multiplier   = random.uniform(2.0, 3.5),
        trigger_step = trigger_step,
        duration     = duration,
    )


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


# ── Utilities ─────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def _extract_routes(node_list):
    """
    Split a flat node sequence (depot=0 marks vehicle returns) into routes.
    Returns list of [0, c1, c2, ..., 0] lists.
    """
    routes, current = [], [0]
    for n in map(int, node_list):
        if n == 0:
            if len(current) > 1:
                current.append(0)
                routes.append(current)
            current = [0]
        else:
            current.append(n)
    if len(current) > 1:
        current.append(0)
        routes.append(current)
    return routes


def _print_solution(inst, routes, total_reward, sol_routes=None):
    """Print best solution per vehicle — matches CR_rl.py output format."""
    T   = float(inst['T'])
    cap = float(inst['vehicle_capacity'])
    mc  = float(inst['max_coord'])

    # denormalise all values for human-readable display
    tt       = inst['tt'].numpy() * T                               # (N+1,N+1) raw travel times
    coords   = np.vstack([inst['depot_xy'].numpy()  * mc,
                          inst['node_xy'].numpy()   * mc])          # (N+1, 2)
    tw_open  = np.concatenate([inst['depot_tw_open'].numpy()  * T,
                                inst['node_tw_open'].numpy()  * T]) # (N+1,)
    tw_close = np.concatenate([inst['depot_tw_close'].numpy() * T,
                                inst['node_tw_close'].numpy() * T]) # (N+1,)
    service  = np.concatenate([inst['depot_service'].numpy()  * T,
                                inst['node_service'].numpy()  * T]) # (N+1,)
    demands  = np.concatenate([inst['depot_demand'].numpy()   * cap,
                                inst['node_demand'].numpy()   * cap])# (N+1,)

    n_cust     = inst['n_customers']
    total_dist = sum(tt[r[j], r[j + 1]] for r in routes for j in range(len(r) - 1))
    n_served   = sum(len(r) - 2 for r in routes)

    W = 75
    print("\n" + "=" * W)
    print("  BEST SOLUTION  (greedy decode)")
    print("=" * W)
    print(f"  vehicles={len(routes):2d}  served={n_served:3d}/{n_cust}"
          f"  dist={total_dist:8.2f}  reward={total_reward:.4f}")

    if sol_routes is not None:
        sol_veh  = len(sol_routes)
        sol_dist = sum(tt[r[j], r[j + 1]] for r in sol_routes for j in range(len(r) - 1))
        dveh  = len(routes) - sol_veh
        ddist = 100.0 * (total_dist - sol_dist) / max(sol_dist, 1e-8)
        print(f"  reference: vehicles={sol_veh:2d}  dist={sol_dist:8.2f}"
              f"  Δveh={dveh:+d}  Δdist={ddist:+.1f}%")

    print()
    for vi, route in enumerate(routes, 1):
        stops = route[1:-1]
        rdist = sum(tt[route[j], route[j + 1]] for j in range(len(route) - 1))
        load  = sum(demands[s] for s in stops)
        print(f"  ── Vehicle {vi:2d}  [{len(stops):3d} stops | dist={rdist:7.1f}"
              f" | load={load:.0f}/{cap:.0f}]")

        t, prev = 0.0, 0
        for s in stops:
            travel    = float(tt[prev, s])
            arrival   = t + travel
            svc_start = max(arrival, float(tw_open[s]))
            svc       = float(service[s])
            depart    = svc_start + svc
            tw_ok     = " " if svc_start <= float(tw_close[s]) + 1e-6 else " LATE"
            wait      = max(0.0, float(tw_open[s]) - arrival)
            print(f"      {prev:3d} → {s:3d}"
                  f"  ({coords[s, 0]:5.1f},{coords[s, 1]:5.1f})"
                  f"  arr={arrival:6.1f}  wait={wait:4.1f}  dep={depart:6.1f}  svc={svc:4.0f}"
                  f"  TW=[{tw_open[s]:5.1f},{tw_close[s]:5.1f}]{tw_ok}"
                  f"  dem={demands[s]:3.0f}")
            t, prev = depart, s

        ret_dist = float(tt[prev, 0])
        print(f"      {prev:3d} →   0  depot return  dist={ret_dist:.1f}"
              f"  arr={t + ret_dist:.1f}/{T:.0f}")
        print()
    print("=" * W)


def _load_sol(data_path: str, n_customers: int):
    """Try to find a Solomon solution file alongside data_path. Returns routes or None."""
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
                if re.search(r'\d+\.\d+', line):   # skip float-only lines (metadata)
                    continue
                nums = list(map(int, re.findall(r'\b\d+\b', line)))
                cust = [n for n in nums if 1 <= n <= n_customers]
                if len(cust) >= 2:
                    routes.append([0] + cust + [0])
        if routes:
            print(f"[Solution] loaded from {path}")
            return routes
    return None


def plot_routes(inst, routes, total_reward: float, save_path: str) -> None:
    """Visualise best-episode routes and save to save_path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not available — route plot skipped")
        return

    mc     = float(inst['max_coord'])
    coords = np.vstack([inst['depot_xy'].numpy() * mc,
                        inst['node_xy'].numpy()  * mc])
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
            x1, y1 = coords[route[j + 1]]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
        ax.scatter([coords[s][0] for s in stops],
                   [coords[s][1] for s in stops], color=color, s=20, zorder=3)
        for s in stops:
            ax.text(coords[s][0] + 0.3, coords[s][1] + 0.3, str(s),
                    fontsize=6, color=color)

    served_set = {s for r in routes for s in r[1:-1]}
    unvisited  = [n for n in range(1, inst['n_customers'] + 1) if n not in served_set]
    if unvisited:
        ax.scatter([coords[n][0] for n in unvisited],
                   [coords[n][1] for n in unvisited],
                   color="red", s=40, zorder=4, marker="x",
                   label=f"Unserved ({len(unvisited)})")

    ax.scatter(depot[0], depot[1], marker="*", s=300, color="black", zorder=5, label="Depot")
    ax.set_title(
        f"{inst['name']} — served={len(served_set)}/{inst['n_customers']}"
        f"  vehicles={len(routes)}  reward={total_reward:.4f}",
        fontsize=12,
    )
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] route map saved → {save_path}")


def plot_score_curve(history: list, save_path: str) -> None:
    """Plot training score convergence (lower = better) and save to save_path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not available — score curve skipped")
        return

    if not history:
        return

    eps    = list(range(1, len(history) + 1))
    window = min(50, max(1, len(history) // 20))
    mavg   = [sum(history[max(0, i - window + 1):i + 1]) / min(i + 1, window)
              for i in range(len(history))]

    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eps, history, color="lightgray", linewidth=0.6, label="score")
    ax.plot(eps, mavg,    color="steelblue", linewidth=2.0,
            label=f"moving avg (w={window})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score (lower = better)")
    ax.set_title(f"Training Score Curve  ({len(history)} epochs)")
    ax.legend(); ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] score curve saved → {save_path}")


# ── Trainer ───────────────────────────────────────────────────────────────────

class VRPTWTrainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        self.env_params       = env_params
        self.model_params     = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params   = trainer_params

        USE_CUDA = trainer_params.get('use_cuda', torch.cuda.is_available())
        if USE_CUDA:
            dev_num = trainer_params.get('cuda_device_num', 0)
            torch.cuda.set_device(dev_num)
            self.device = torch.device('cuda', dev_num)
        else:
            self.device = torch.device('cpu')

        self.model = Model(**model_params).to(self.device)
        self.env   = Env(**env_params)
        self.optimizer = Optimizer(self.model.parameters(), **optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **optimizer_params['scheduler'])

        data_dir  = trainer_params['data_dir']
        train_ids = trainer_params['train_instances']
        self.instance_pool = []
        for iname in train_ids:
            path = os.path.join(data_dir, iname + '.txt')
            inst = load_solomon(path)
            self.instance_pool.append(inst)
            print(f'  [Data] loaded {inst["name"]}  N={inst["n_customers"]}')

        test_ids = trainer_params.get('test_instances', [])
        self.test_pool  = []
        self.test_paths = []
        for iname in test_ids:
            path = os.path.join(data_dir, iname + '.txt')
            inst = load_solomon(path)
            self.test_pool.append(inst)
            self.test_paths.append(path)
            print(f'  [Test] loaded {inst["name"]}')

        self.start_epoch = 1
        model_load = trainer_params.get('model_load', {'enable': False})
        if model_load.get('enable'):
            ckpt_path = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch'] - 1
            self.start_epoch = model_load['epoch'] + 1
            print(f'[Resume] epoch {self.start_epoch}')

        self.result_dir = trainer_params.get('result_dir', 'result')
        os.makedirs(self.result_dir, exist_ok=True)

        self.score_log = []
        self.loss_log  = []

    # ------------------------------------------------------------------
    def run(self):
        if _MLFLOW:
            mlflow.set_experiment("VRPTW_PurePOMO")
            run_ctx = mlflow.start_run(run_name="pure_pomo")
            run_ctx.__enter__()
            tp = self.trainer_params
            mlflow.log_params({
                "epochs":           tp['epochs'],
                "train_episodes":   tp['train_episodes'],
                "train_batch_size": tp['train_batch_size'],
                "pomo_size":        self.env_params['pomo_size'],
                "problem_size":     self.env_params['problem_size'],
                "late_penalty":     self.env_params.get('late_penalty', 1.0),
                "vehicle_penalty":  self.env_params.get('vehicle_penalty', 0.0),
                "lr":               self.optimizer_params['optimizer']['lr'],
                "embedding_dim":    self.model_params['embedding_dim'],
                "encoder_layers":   self.model_params['encoder_layer_num'],
                "weather_prob":     tp.get('weather_prob', 0.3),
                "accident_prob":    tp.get('accident_prob', 0.2),
                "train_instances":  ",".join(tp['train_instances']),
                "test_instances":   ",".join(tp.get('test_instances', [])),
            })

        t0 = time.time()
        score_history = []

        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            train_score, train_loss = self._train_one_epoch(epoch)
            self.scheduler.step()
            score_history.append(train_score)
            self.score_log.append(train_score)
            self.loss_log.append(train_loss)

            elapsed = time.time() - t0
            print(f'Epoch {epoch:4d}/{self.trainer_params["epochs"]}  '
                  f'score={train_score:.4f}  loss={train_loss:.6f}  '
                  f'elapsed={elapsed:.0f}s')

            if _MLFLOW:
                mlflow.log_metrics({
                    "train/score": train_score,
                    "train/loss":  train_loss,
                }, step=epoch)

            save_interval = self.trainer_params['logging'].get('model_save_interval', 500)
            if epoch % save_interval == 0:
                self._save_ckpt(epoch)

        self._save_ckpt(self.trainer_params['epochs'], tag='last')
        print('\n*** Training Done ***')

        # ── Score curve ──────────────────────────────────────────────────
        plots_dir = os.path.join(self.result_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        curve_path = os.path.join(plots_dir, 'score_curve.png')
        plot_score_curve(score_history, curve_path)

        # ── Final eval on test instances with last model ─────────────────
        if self.test_pool:
            self.model.eval()
            test_score, per_inst = self._eval_test()
            print(f'  [Final Eval] test_score={test_score:.4f}')
            if len(per_inst) > 1:
                per_str = '  '.join(f'{n}:{s:.4f}' for n, s in per_inst)
                print(f'    [{per_str}]')
            if _MLFLOW:
                mlflow.log_metric("test/score", test_score)
                for name, s in per_inst:
                    mlflow.log_metric(f"test/score_{name}", s)
            for inst, data_path in zip(self.test_pool, self.test_paths):
                routes, best_reward = self._best_solution(inst)
                sol_routes = _load_sol(data_path, inst['n_customers'])
                _print_solution(inst, routes, best_reward, sol_routes)
                route_path = os.path.join(plots_dir, f"{inst['name']}_best_route.png")
                plot_routes(inst, routes, best_reward, route_path)

        if _MLFLOW:
            ckpt_path = os.path.join(self.result_dir, 'checkpoint-last.pt')
            if os.path.isfile(ckpt_path):
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
            if os.path.isfile(curve_path):
                mlflow.log_artifact(curve_path, artifact_path="plots")
            for inst in self.test_pool:
                p = os.path.join(plots_dir, f"{inst['name']}_best_route.png")
                if os.path.isfile(p):
                    mlflow.log_artifact(p, artifact_path="plots")
            run_ctx.__exit__(None, None, None)

    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM  = AverageMeter()

        n_episodes = self.trainer_params['train_episodes']
        batch_size = self.trainer_params['train_batch_size']
        episode    = 0

        while episode < n_episodes:
            remaining  = n_episodes - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            inst = random.choice(self.instance_pool)
            avg_score, avg_loss = self._train_one_batch(inst, batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss,  batch_size)
            episode += batch_size

        return score_AM.avg, loss_AM.avg

    # ------------------------------------------------------------------
    def _train_one_batch(self, inst: dict, batch_size: int):
        self.model.train()

        tp = self.trainer_params
        weather_prob  = tp.get('weather_prob',  0.3)
        accident_prob = tp.get('accident_prob', 0.2)
        acc_step_cfg  = tp.get('accident_step', 50)
        acc_dur_cfg   = tp.get('accident_dur',  30)

        # ── Weather event → rain_affected tensor + real TT modification ──
        rain_nodes, rain_mult = _get_rain_event(inst, weather_prob)
        rain_evs = [e for e in inst.get('preset_events', []) if e['type'] == 'RAIN']

        # ── Accident event params ──────────────────────────────────────────
        acc_ev = _get_accident_event(inst, accident_prob, acc_step_cfg, acc_dur_cfg)

        batch = make_batch(inst, batch_size, self.device)
        self.env.load_problems(batch)

        if rain_nodes:
            self.env.apply_rain(rain_nodes, rain_mult)

        reset_state, _, _ = self.env.reset()
        N = self.env.problem_size
        reset_state.rain_tokens = _make_rain_tokens(inst, rain_evs, batch_size, self.device)
        reset_state.acc_tokens  = _make_acc_tokens([], N, batch_size, self.device)
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(batch_size, self.env.pomo_size, 0, device=self.device)

        state, reward, done = self.env.pre_step()
        max_steps    = self.trainer_params.get('max_steps', 500)
        acc_applied  = False
        acc_restored = False
        active_accs  = []
        step = 0
        while not done and step < max_steps:
            if acc_ev is not None:
                t = acc_ev['trigger_step']
                d = acc_ev['duration']
                if not acc_applied and step == t:
                    self.env.apply_accident(acc_ev['node_a'], acc_ev['node_b'],
                                            acc_ev['multiplier'])
                    active_accs.append(dict(
                        nodes=[acc_ev['node_a'], acc_ev['node_b']],
                        t_start=t / max(self.env.problem_size, 1),
                        t_end=(t + d) / max(self.env.problem_size, 1),
                    ))
                    self.env.reset_state.acc_tokens = _make_acc_tokens(
                        active_accs, N, batch_size, self.device)
                    self.model.pre_forward(self.env.reset_state)
                    acc_applied = True
                elif acc_applied and not acc_restored and step == t + d:
                    self.env.restore_accident()
                    active_accs = []
                    self.env.reset_state.acc_tokens = _make_acc_tokens(
                        active_accs, N, batch_size, self.device)
                    self.model.pre_forward(self.env.reset_state)
                    acc_restored = True

            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            step += 1

        if reward is None:
            reward = -self.env._get_travel_distance()

        advantage = reward - reward.float().mean(dim=1, keepdim=True)
        log_prob  = prob_list.log().sum(dim=2)
        loss      = -(advantage * log_prob).mean()

        self.model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        max_pomo_reward, _ = reward.max(dim=1)
        score = -max_pomo_reward.float().mean()
        return score.item(), loss.item()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_test(self):
        """Greedy eval on test instances. Returns (avg_score, [(name, score), ...])."""
        self.model.eval()
        scores, per_inst = [], []

        for inst in self.test_pool:
            batch = make_batch(inst, self.trainer_params.get('test_batch_size', 1), self.device)
            self.env.load_problems(batch)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = self.env.pre_step()
            max_steps = self.trainer_params.get('max_steps', 500)
            step = 0
            while not done and step < max_steps:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)
                step += 1

            if reward is None:
                reward = -self.env._get_travel_distance()

            max_pomo_reward, _ = reward.max(dim=1)
            s = -max_pomo_reward.float().mean().item()
            scores.append(s)
            per_inst.append((inst['name'], s))

        return sum(scores) / len(scores), per_inst

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _best_solution(self, inst: dict):
        """
        Run POMO rollout on a single instance.
        Returns (routes, best_reward) for the best POMO agent.
        """
        self.model.eval()
        batch = make_batch(inst, 1, self.device)
        self.env.load_problems(batch)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        state, reward, done = self.env.pre_step()
        max_steps = self.trainer_params.get('max_steps', 500)
        step = 0
        while not done and step < max_steps:
            selected, _ = self.model(state)
            state, reward, done = self.env.step(selected)
            step += 1

        if reward is None:
            reward = -self.env._get_travel_distance()

        # reward: (1, pomo_size) — pick best pomo agent
        best_pomo_idx = int(reward[0].argmax().item())
        best_reward   = float(reward[0, best_pomo_idx].item())

        node_list = self.env.selected_node_list[0, best_pomo_idx].cpu().tolist()
        routes    = _extract_routes(node_list)
        return routes, best_reward

    # ------------------------------------------------------------------
    def _save_ckpt(self, epoch, tag=None):
        name = f'checkpoint-{tag}.pt' if tag else f'checkpoint-{epoch}.pt'
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'score_log':            self.score_log,
            'loss_log':             self.loss_log,
        }, os.path.join(self.result_dir, name))
