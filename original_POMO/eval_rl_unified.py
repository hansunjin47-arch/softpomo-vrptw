"""
eval_rl_unified.py

Compact evaluation of the Exp3-UNIFIED RL model (Config F, benchmark r1,
gen-type UNIFIED, pomo=100, aug).

Runs _best_solution on each test instance with 8-fold augmentation, then
reports K / D / Lc / Lt in the same table format used by eval_hist_full_all.

Checkpoint: result/config_F_exp3_soft_unified/gen_UNIFIED/checkpoint-last.pt

Usage:
    python eval_rl_unified.py
    python eval_rl_unified.py --ckpt <path>
"""
from __future__ import annotations

import os
import sys
import argparse
import copy
import torch
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

# ── import VRPTWTrainer infrastructure ─────────────────────────────────────────
from vrptw_env import (
    VRPTWEnv, load_solomon, make_batch,
    AverageMeter, _extract_routes, _solution_stats,
    augment_xy_data_by_8_fold,
)
from train_vrptw import (
    VRPTWModel, VRPTWTrainer,
    _REWARD_CONFIGS_R1, model_params, optimizer_params,
    DATA_DIR, RESULT_DIR,
    _get_rain_event, _get_accident_events, _make_rain_tokens, _make_acc_tokens,
)

# ── Config F (R1 table) — same as Exp3-UNIFIED training ───────────────────────
_CFG_F_R1 = dict(_REWARD_CONFIGS_R1['F'])  # late=9 vp=2 D_max=15 Lt_max=5

TEST_GROUPS = {
    "C1":  ["c101",  "c101_rain_A",  "c101_acc_A"],
    "RC1": ["rc101", "rc101_rain_A", "rc101_acc_A"],
    "R1":  ["r101",  "r101_rain_A",  "r101_acc_A"],
}

DEFAULT_CKPT = os.path.join(
    _HERE, "result", "config_F_exp3_soft_unified", "gen_UNIFIED", "checkpoint-last.pt"
)


def build_trainer(ckpt_path: str) -> VRPTWTrainer:
    env_p = dict(
        problem_size=100,
        pomo_size=100,
        late_count_penalty=0.0,
        unserved_penalty=500.0,
    )
    env_p.update(_CFG_F_R1)

    trainer_p = dict(
        use_cuda=True,
        cuda_device_num=0,
        data_dir=DATA_DIR,
        train_instances=[],   # gen_type=UNIFIED — not used
        test_instances=[],
        result_dir=os.path.join(_HERE, "result", "config_F_exp3_soft_unified"),
        epochs=0,
        train_episodes=0,
        train_batch_size=64,
        test_batch_size=1,
        curriculum_epoch=0,
        max_steps=600,
        n_mc_samples=1,
        use_augmentation=True,
        model_load=dict(enable=False, path=None, epoch=None),
        logging=dict(model_save_interval=500, log_interval=10),
    )

    trainer = VRPTWTrainer(env_p, dict(model_params), dict(optimizer_params), trainer_p)

    ckpt = torch.load(ckpt_path, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt['model_state_dict'])
    trainer.model.eval()
    print(f"[Checkpoint] loaded from {ckpt_path}")
    return trainer


def eval_instance(trainer: VRPTWTrainer, inst_name: str) -> dict | None:
    path = os.path.join(DATA_DIR, inst_name + ".txt")
    if not os.path.isfile(path):
        print(f"  [WARN] {path} not found — skipped")
        return None
    inst = load_solomon(path)
    routes, _ = trainer._best_solution(inst)
    stats = _solution_stats(inst, routes)
    return {
        "name": inst_name,
        "K":  len([r for r in routes if r]),
        "D":  stats["total_dist"],
        "Lc": stats["late_count"],
        "Lt": stats["total_late"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT,
                        help="Path to checkpoint-last.pt")
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"[ERROR] Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    trainer = build_trainer(args.ckpt)

    results_dir = os.path.join(_HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "rl_unified_eval.log")
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 70)
    p("  RL Eval — Exp3-UNIFIED (Config F, R1 table, gen=UNIFIED, pomo=100, aug)")
    p("=" * 70)
    hdr = (f"{'Instance':<18} {'Scenario':<12} "
           f"{'K':>4} {'D':>8} {'Lc':>5} {'Lt':>8}")
    p(hdr)
    p("-" * 60)

    for group, instances in TEST_GROUPS.items():
        p(f"\n  -- {group} --")
        for inst_name in instances:
            scen = "base" if "_rain_" not in inst_name and "_acc_" not in inst_name else (
                "rain_A" if "_rain_A" in inst_name else "acc_A"
            )
            res = eval_instance(trainer, inst_name)
            if res is None:
                continue
            p(f"  {inst_name:<18} {scen:<12} "
              f"{res['K']:>4d} {res['D']:>8.1f} {res['Lc']:>5d} {res['Lt']:>8.1f}")

    p("\nDone.")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Saved] {log_path}")


if __name__ == "__main__":
    main()
