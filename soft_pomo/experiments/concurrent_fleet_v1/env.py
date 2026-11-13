"""Global-clock, single-trip, variable-fleet routing prototype.

Actions are (vehicle, customer), (-1, customer) for dispatch, or (-2, 0)
for advancing to the next event. Traffic is fixed at arc departure, matching
the legacy simulator; hidden event schedules never appear in observations.
"""
import heapq
import math
import numpy as np
from vrptw_env import _event_travel_mult


class FleetEnv:
    NEW, ADVANCE = -1, -2

    def __init__(self, instance, public_events, weight=5, k_ref=14, t_ref=1696.95,
                 environment_params=None):
        self.inst = instance
        self.public_events = public_events
        self.weight, self.k_ref, self.t_ref = weight, k_ref, t_ref
        self.n = instance['n_customers']
        self.horizon = instance['T']
        limit = instance.get('vehicle_limit')
        self.vehicle_limit = self.n if limit is None else int(limit)
        if self.vehicle_limit < 1:
            raise ValueError('vehicle_limit must be positive')
        self.reward_params = dict(dist_penalty_weight=0., D_max=1.,
            late_count_penalty=0., late_penalty=weight, Lt_max=t_ref/self.horizon,
            vehicle_penalty=self.n/k_ref, time_penalty=1., TT_max=t_ref/self.horizon,
            unserved_penalty=500., k_excess_alpha=0.)
        if environment_params is not None:
            self.reward_params.update(environment_params)
        if self.reward_params.get('hard_tw', False) or self.reward_params.get('k_excess_alpha',0):
            raise ValueError('This version supports the current soft-TW/no-k-excess config only')
        self.xy = np.concatenate([instance['depot_xy'], instance['node_xy']])
        self.demand = np.concatenate([instance['depot_demand'], instance['node_demand']])
        self.open = np.concatenate([instance['depot_tw_open'], instance['node_tw_open']]) * self.horizon
        self.close = np.concatenate([instance['depot_tw_close'], instance['node_tw_close']]) * self.horizon
        self.service = np.concatenate([instance['depot_service'], instance['node_service']]) * self.horizon
        self.base = instance['tt'].numpy() * self.horizon
        if np.any(self.demand[1:] > 1 + 1e-6):
            raise ValueError('Unsplittable demand exceeds vehicle capacity')
        self.reset()

    def reset(self):
        self.time = 0.0
        self.customer_state = np.zeros(self.n + 1, dtype=np.int64)
        self.customer_state[0] = 2  # 0 unassigned, 1 reserved, 2 served
        self.vehicles, self.events, self.trace = [], [], []
        self.serial = 0
        self.travel = self.distance = self.late = 0.0
        self.late_count = 0
        for index, event in enumerate(self.public_events):
            t = float(event['revealed_at_min'])
            if t > 0:
                self._push(t, 'disclosure', index)
        return self

    def _push(self, t, kind, vehicle):
        self.serial += 1
        heapq.heappush(self.events, (float(t), self.serial, kind, vehicle))

    @property
    def done(self):
        return bool(all(v['status'] == 'retired' for v in self.vehicles) and
                    (np.all(self.customer_state == 2) or len(self.vehicles) >= self.vehicle_limit))

    def actions(self):
        if self.done:
            return []
        free = np.flatnonzero(self.customer_state == 0).tolist()
        actions = [(self.NEW, int(n)) for n in free] if len(self.vehicles) < self.vehicle_limit else []
        for i, v in enumerate(self.vehicles):
            if v['status'] == 'ready':
                actions.append((i, 0))
                actions.extend((i, int(n)) for n in free if self.demand[n] <= v['load'] + 1e-5)
        # Waiting is permitted only while actual vehicle work is pending. This
        # avoids revealing the existence/time of future accidents via the mask.
        if any(v['status'] in ('moving', 'service') for v in self.vehicles):
            actions.append((self.ADVANCE, 0))
        assert actions, 'Nonterminal state without a feasible action'
        return actions

    def _lateness(self, node):
        value = max(0.0, self.time - float(self.close[node]))
        self.late += value
        self.late_count += int(value > 0)

    def step(self, action):
        action = tuple(action)
        if action not in self.actions():
            raise ValueError(f'Invalid action {action}')
        vehicle, node = action
        if vehicle == self.ADVANCE:
            target = self.events[0][0]
            assert target >= self.time
            self.time = target
            # Process all simultaneous physical events before the next decision.
            while self.events and self.events[0][0] <= target:
                _, _, kind, i = heapq.heappop(self.events)
                if kind == 'arrival':
                    v = self.vehicles[i]
                    v['node'] = v['target']
                    self.travel += v['arc_time']
                    self.distance += v['arc_base']
                    self._lateness(v['node'])
                    if v['node'] == 0:
                        v['status'] = 'retired'
                    else:
                        v['status'] = 'service'
                        self._push(max(self.time, float(self.open[v['node']])) +
                                   float(self.service[v['node']]), 'service_end', i)
                elif kind == 'service_end':
                    v = self.vehicles[i]
                    self.customer_state[v['node']] = 2
                    v['status'] = 'ready'
                self.trace.append(dict(time=self.time, event=kind, vehicle=i))
            return
        if vehicle == self.NEW:
            vehicle = len(self.vehicles)
            self.vehicles.append(dict(node=0, target=0, load=1.0, status='ready',
                                      departure=self.time, route=[0], departures=[]))
        v = self.vehicles[vehicle]
        base = float(self.base[v['node'], node])
        duration = base * _event_travel_mult(self.time, v['node'], node, self.inst['preset_events'])
        if node:
            self.customer_state[node] = 1
            v['load'] -= float(self.demand[node])
        v.update(target=node, status='moving', departure=self.time,
                 arc_time=duration, arc_base=base)
        v['route'].append(node)
        v['departures'].append(self.time)
        self._push(self.time + duration, 'arrival', vehicle)
        self.trace.append(dict(time=self.time, event='depart', vehicle=vehicle, customer=node))

    def observation(self):
        public = [{key: e[key] for key in ('event_type', 'affected_nodes', 'start_min',
                   'end_min', 'revealed_at_min', 'rainfall_mm_per_hour', 'vehicles_involved') if key in e}
                  for e in self.public_events if float(e['revealed_at_min']) <= self.time]
        fleet = []
        for v in self.vehicles:
            # Elapsed time is observable; simulator's realized arrival time is not.
            fleet.append([*self.xy[v['node']], *self.xy[v['target']], v['load'],
                          (self.time-v['departure'])/self.horizon,
                          *[float(v['status'] == s) for s in ('moving','service','ready','retired')]])
        return dict(time=self.time, customers=self.customer_state.copy(),
                    fleet=np.asarray(fleet, dtype=np.float32).reshape(-1, 10), public=public)

    def action_features(self, actions):
        obs = self.observation()
        rows = []
        for i, n in actions:
            origin = 0 if i < 0 else self.vehicles[i]['node']
            load = 1.0 if i < 0 else self.vehicles[i]['load']
            ev = [0.0]*6
            for e in obs['public']:
                if n in e['affected_nodes'] or origin in e['affected_nodes']:
                    rain = e['event_type'] == 'RAIN'
                    ev[0 if rain else 1] = float(e.get('rainfall_mm_per_hour',0))/30 if rain else float(e.get('vehicles_involved',0))/5
                    j = 2 if rain else 4
                    ev[j:j+2] = [float(e['start_min'])/self.horizon, float(e['end_min'])/self.horizon]
            rows.append([float(i == self.NEW), float(i == self.ADVANCE), load,
                         *self.xy[origin], float(self.base[origin,n])/self.horizon,
                         self.time/self.horizon, (float(self.close[n])-self.time)/self.horizon,
                         len(self.vehicles)/self.vehicle_limit, float(np.mean(self.customer_state[1:] == 0)), *ev])
        return np.asarray(rows, dtype=np.float32)

    def metrics(self):
        if not self.done:
            raise ValueError('Only completed trajectories may be scored')
        served = int(np.sum(self.customer_state[1:] == 2))
        p = self.reward_params
        cost = (p['dist_penalty_weight'] * self.distance/self.horizon/p['D_max']
                + p['late_count_penalty'] * self.late_count/self.n
                + p['late_penalty'] * self.late/self.horizon/p['Lt_max']
                + p['vehicle_penalty'] * len(self.vehicles)/self.n
                + p['time_penalty'] * self.travel/self.horizon/p['TT_max']
                + p['unserved_penalty'] * (self.n-served)/self.n)
        assert math.isfinite(cost)
        return dict(vehicles=len(self.vehicles), travel_time=self.travel, distance=self.distance,
                    total_late=self.late, late_count=self.late_count, served=served,
                    unserved=self.n-served, feasible=served==self.n,
                    vehicle_limit=self.vehicle_limit,
                    termination='all_served' if served==self.n else 'fleet_exhausted',
                    cost=cost, reward=-cost, finish_time=self.time,
                    routes=[v['route'] for v in self.vehicles],
                    departures=[v['departures'] for v in self.vehicles])
