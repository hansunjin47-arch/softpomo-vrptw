import torch
from experiments.concurrent_fleet_v8.env import TensorFleetEnv as ReferenceEnv


class TensorFleetEnv(ReferenceEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dispatch_open = torch.ones(self.count, dtype=torch.bool, device=self.device)
        self.pending = torch.full_like(self.status, -1)
        self.dispatch_end_time = torch.full_like(self.time, -1)
        self.prebook_count = torch.zeros_like(self.k)

    @property
    def done(self):
        retired = ((self.status == 0) | (self.status == 4)).all(1)
        return retired & ~self.dispatch_open

    def frame(self):
        if 'pair_frame' in self._state_cache:
            return self._state_cache['pair_frame']
        active_dispatch = self.dispatch_open & ~self.done
        q = max(1, int(self.k.masked_fill(~active_dispatch, 0).max()))
        slots = torch.arange(q, device=self.device)[None].expand(self.count, -1)
        selected, ready, _ = self.next_vehicle()
        vehicles = torch.where(active_dispatch[:, None], slots, selected[:, None])
        status = self.status.gather(1, vehicles)
        is_pending = active_dispatch[:, None] & ((status == 1) | (status == 2))
        nodes = torch.where(is_pending, self.target.gather(1, vehicles), self.node.gather(1, vehicles))
        loads = self.load.gather(1, vehicles)
        free = self.customer == 0
        old = free[:, None, :] & (self.demand[None, None, :] <= loads[:, :, None] + 1e-5)
        old[:, :, 0] = True
        available = torch.where(active_dispatch[:, None], (status > 0) & (status < 4),
                                ready[:, None] & (slots == 0))
        old &= available[:, :, None] & ~self.done[:, None, None]
        new = free[:, 1:] & (self.demand[None, 1:] <= 1. + 1e-5)
        new &= (active_dispatch & (self.time == 0) & (self.k < self.v))[:, None]
        frame = dict(node=nodes, load=loads, time=self.time / self.horizon,
                     vehicles=vehicles, prebook=is_pending,
                     mask=torch.cat([new, old.flatten(1)], 1))
        self._state_cache['pair_frame'] = frame
        return frame

    def mask(self):
        return self.frame()['mask']

    def decode_action(self, action, frame=None):
        frame = self.frame() if frame is None else frame
        new = action < self.n
        pair = (action - self.n).clamp(min=0)
        slot = pair // (self.n + 1)
        vehicle = torch.where(new, self.k, frame['vehicles'][self.ids, slot])
        dest = torch.where(new, action + 1, pair % (self.n + 1))
        return new, vehicle.clamp(max=self.v - 1), dest

    def _depart(self, rows, vehicles, dest, charged=False):
        idx = (rows, vehicles)
        origin = self.node[idx]
        base = self.base[origin, dest]
        now = self.time[rows]
        mult = torch.ones_like(now)
        for start, end, value, zone in self.true_events:
            affected = (now >= start) & (now < end) & (zone[origin] | zone[dest])
            mult = torch.where(affected, mult.clamp(min=value), mult)
        duration = base * mult
        self.target[idx] = dest
        self.status[idx] = 1
        self.departure[idx] = now
        self.arc[idx] = duration
        self.arc_base[idx] = base
        self.arrival[idx] = now + duration
        self.available[idx] = torch.maximum(now + duration, self.open[dest]) + self.service[dest]
        if not charged:
            self.load[idx] -= torch.where(dest > 0, self.demand[dest], 0.)
        self.customer[rows[dest > 0], dest[dest > 0]] = 1
        self._state_cache.clear()

    def _settle(self):
        # Process committed departures at their own service completion time,
        # including those earlier than another vehicle's next policy decision.
        for _ in range(4 * self.n + 4 * self.v + 16):
            self._process()
            auto = (self.status == 3) & (self.pending >= 0)
            if bool(auto.any()):
                rows, vehicles = auto.nonzero(as_tuple=True)
                dest = self.pending[rows, vehicles].clone()
                self.pending[rows, vehicles] = -1
                self._depart(rows, vehicles, dest, charged=True)
                continue
            free = (self.customer[:, 1:] == 0).any(1)
            close = self.dispatch_open & ((self.k >= self.v) | ~free)
            self.dispatch_open[close] = False
            self.dispatch_end_time[close] = self.time[close]
            self._state_cache.clear()
            ready = (self.status == 3).any(1)
            advance = ~self.dispatch_open & ~self.done & ~ready
            if not bool(advance.any()):
                return
            moving_customer = ((self.status == 1) | (self.status == 2)) & (self.target != 0)
            completion = self.available.masked_fill(~moving_customer, torch.inf).amin(1)
            returns = self.arrival.masked_fill((self.status != 1) | (self.target != 0), torch.inf).amin(1)
            next_time = torch.minimum(completion, returns)
            if not bool(torch.isfinite(next_time[advance]).all()):
                raise RuntimeError('No event for live rollout')
            self.time = torch.where(advance, next_time, self.time)
            self.first_positive = torch.where((self.first_positive < 0) & (self.time > 0),
                                              self.time, self.first_positive)
        raise RuntimeError('Event loop did not settle')

    def step(self, action, validate=False):
        live = ~self.done
        if validate and not bool((self.mask()[self.ids, action] | ~live).all()):
            raise ValueError('Invalid pair action')
        new, vehicle, dest = self.decode_action(action)
        closing = live & ~new & self.dispatch_open
        self.dispatch_open[closing] = False
        self.dispatch_end_time[closing] = self.time[closing]
        defer = closing & (self.status[self.ids, vehicle] != 3)
        rows = self.ids[defer]
        v = vehicle[defer]
        d = dest[defer]
        self.pending[rows, v] = d
        self.load[rows, v] -= torch.where(d > 0, self.demand[d], 0.)
        self.customer[rows[d > 0], d[d > 0]] = 1
        self.prebook_count += defer.long()
        self.k += (new & live).long()
        self.last_vehicle = vehicle
        immediate = live & ~defer
        self._depart(self.ids[immediate], vehicle[immediate], dest[immediate])
        self._settle()

    def metrics(self):
        return dict(super().metrics(), prebook_count=self.prebook_count,
                    dispatch_end_time=self.dispatch_end_time)
