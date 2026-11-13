"""Trainable dispatch-only STOP offset, calibrated before training."""
import math
import torch
from experiments.concurrent_fleet_v9.policy import FleetPolicy as BasePolicy
from experiments.concurrent_fleet_v9.env import TensorFleetEnv


class FleetPolicy(BasePolicy):
    def __init__(self, **params):
        super().__init__(**params)
        self.stop_bias = torch.nn.Parameter(torch.zeros(()))

    def _phase_scores(self, encoded, frame, env, dispatch):
        scores = super()._phase_scores(encoded, frame, env, dispatch)
        if dispatch:
            # A learnable log-odds prior after clipping; NEW scores are unchanged.
            scores = torch.cat([scores[:, :-1], scores[:, -1:] + self.stop_bias], 1)
        return scores


@torch.no_grad()
def calibrate(model, cases, params, target=.15, count=32):
    if not 0 < target < 1:
        raise ValueError('STOP target must be in (0, 1)')
    if next(model.parameters()).device.type != 'cpu':
        raise ValueError('Initial calibration uses CPU only')
    mode = model.training
    model.eval()
    odds = []
    model.stop_bias.zero_()
    # Calibration samples are not training data; preserve the training RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261021)
        for inst, public in cases:
            env = TensorFleetEnv(inst, count, params, device='cpu')
            starts, enc = model.encode_events(inst, public)
            for _ in range(env.v):
                frame = env.frame()
                p = model.probabilities(frame, starts, enc, env)
                valid = frame['mask'][:, -1] & frame['mask'][:, :env.n].any(1)
                odds.append((p[valid, -1].double().log() - p[valid, :env.n].double().sum(1).log()))
                if not bool(frame['mask'][:, :env.n].any(1).all()):
                    raise ValueError('Calibration exhausted feasible NEW actions')
                action = frame['mask'][:, :env.n].float().multinomial(1).squeeze(1)
                env.step(action, validate=True)
    values = torch.cat(odds)
    if not values.numel() or not bool(torch.isfinite(values).all()):
        raise ValueError('No finite calibration states')
    lo, hi = -60., 60.
    for _ in range(70):
        mid = (lo + hi) / 2
        if float(torch.sigmoid(values + mid).mean()) < target:
            lo = mid
        else:
            hi = mid
    model.stop_bias.fill_((lo + hi) / 2)
    model.train(mode)
    after = torch.sigmoid(values + float(model.stop_bias))
    return dict(target=target, states=values.numel(), initial_bias=float(model.stop_bias),
                mean_before=float(torch.sigmoid(values).mean()), mean_after=float(after.mean()),
                p05=float(after.quantile(.05)), p95=float(after.quantile(.95)),
                state_sampling='32 uniformly forced NEW prefixes per train scenario; CPU FP32; no optimizer')
