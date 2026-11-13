"""Fleet-aware soft one-step PI guidance (not PIP-D or a feasibility oracle).

Inspired by PIP-constraint/POMO+PIP/envs/TSPTWEnv.py::_calculate_PIP_mask.
This independent implementation uses nominal travel times, not hidden events.
"""
import torch
from experiments.concurrent_fleet_stop_init.policy import FleetPolicy as BasePolicy


@torch.no_grad()
def route_risk(env, frame, chunk=1024):
    """Incremental optimistic late minutes after each candidate destination.

Each remaining customer independently takes its best capacity-eligible vehicle.
This ignores joint assignment conflicts, so it is a heuristic, not a guarantee.
Moving vehicles use nominal remaining time; actual simulator ETA is never read.
"""
    result = env.time.new_zeros(env.count, env.n + 1)
    vehicle, ready, _ = env.next_vehicle()
    rows = ((~env.dispatch_open) & ready & (~env.done)).nonzero().flatten()
    for ids in rows.split(chunk):
        b = len(ids)
        ar = torch.arange(b, device=env.device)
        chosen = vehicle[ids]
        now = env.time[ids]
        status = env.status[ids]
        moving = status == 1
        pos = torch.where(moving, env.target[ids], env.node[ids])
        nominal_arrival = env.departure[ids] + env.base[env.node[ids], env.target[ids]]
        moving_release = torch.maximum(nominal_arrival, env.open[pos]) + env.service[pos]
        # For service in progress, available is an already-observed service end.
        release = torch.where(moving, moving_release,
                              torch.where(status == 2, env.available[ids], now[:, None]))
        release = torch.maximum(release, now[:, None])
        eligible = ((status > 0) & (status < 4) & (pos != 0))
        other = release[:, :, None] + env.base[pos]
        can = eligible[:, :, None] & (env.load[ids, :, None] + 1e-5 >= env.demand)
        can[ar, chosen] = False
        other = other.masked_fill(~can, torch.inf).amin(1)
        origin = frame['node'][ids]
        arrival = now[:, None] + env.base[origin]
        load = frame['load'][ids]
        direct = arrival.masked_fill(load[:, None] + 1e-5 < env.demand, torch.inf)
        before = (torch.minimum(other, direct) - env.close).clamp(0, env.horizon)
        finish = torch.maximum(arrival, env.open) + env.service
        after_self = finish[:, :, None] + env.base[None]
        remaining_load = load[:, None] - env.demand
        after_self.masked_fill_(remaining_load[:, :, None] + 1e-5 < env.demand, torch.inf)
        after_self[:, 0] = torch.inf  # returning vehicles retire, no new depot tour
        after = (torch.minimum(after_self, other[:, None]) - env.close).clamp(0, env.horizon)
        # Candidate j is served by this vehicle, not by a hypothetical alternative.
        diag = torch.arange(env.n + 1, device=env.device)
        after[:, diag, diag] = (arrival - env.close).clamp(0, env.horizon)
        free = env.customer[ids] == 0
        increment = ((after - before[:, None]).clamp(min=0) * free[:, None]).sum(-1)
        valid = frame['mask'][ids, env.n:2*env.n+1]
        result[ids] = increment.masked_fill(~valid, 0)
    return result


class FleetPolicy(BasePolicy):
    risk_beta = 1.0
    risk_clip = 2.0

    def probabilities(self, frame, starts, encodings, env):
        # Compute once per state, even when several public-event encodings exist.
        with torch.no_grad():
            raw = route_risk(env, frame)
            cost = raw * env.params['late_penalty'] / (env.horizon * env.params['Lt_max'])
            self._risk_penalty = (self.risk_beta * cost).clamp(0, self.risk_clip)
            valid = frame['mask'][:, env.n:2*env.n+1] & (~env.dispatch_open[:, None])
            stats = torch.stack([self._risk_penalty.sum(), valid.sum(),
                                 ((self._risk_penalty > 0) & valid).sum()]).detach()
            previous = getattr(self, 'risk_stats', torch.zeros_like(stats))
            self.risk_stats = previous.to(stats.device) + stats
        return super().probabilities(frame, starts, encodings, env)

    def _phase_scores(self, encoded, frame, env, dispatch):
        scores = super()._phase_scores(encoded, frame, env, dispatch)
        if not dispatch:
            scores = torch.cat([scores[:, :env.n+1].float() - self._risk_penalty,
                                scores[:, -1:].float()], 1)
        return scores
