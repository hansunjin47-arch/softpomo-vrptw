"""Independent hand-calculated and fixed-route environment checks (CPU)."""
import unittest
import torch
from experiments.concurrent_fleet_v1.test_env import instance
from experiments.concurrent_fleet_v5.env import TensorFleetEnv
from experiments.concurrent_fleet_v5.test_tensor import PARAMS
from experiments.concurrent_fleet_v3.run import SOURCE, load
from vrptw_env import VRPTWEnv, make_batch, _solution_stats
import json


def replay(inst, params, routes):
    env = TensorFleetEnv(inst, 1, params, 'cpu')
    positions = [1] * len(routes)
    for route in routes:
        env.step(torch.tensor([route[0] - 1]), validate=True)
    for _ in range(2 * inst['n_customers'] + 2 * len(routes) + 4):
        if bool(env.done.all()):
            return env
        vehicle, ready, _ = env.next_vehicle()
        if bool(ready[0]):
            v = int(vehicle[0])
            destination = routes[v][positions[v]] if positions[v] < len(routes[v]) else 0
            positions[v] += 1
            action = inst['n_customers'] + destination
        else:
            action = 2 * inst['n_customers'] + 1
        env.step(torch.tensor([action]), validate=True)
    raise AssertionError('Replay did not finish')


def fixture(event=False):
    inst = instance()
    xy = torch.tensor([[0., 0.], [.1, 0.], [.2, 0.], [.3, 0.]])
    inst.update(depot_xy=xy[:1], node_xy=xy[1:], tt=torch.cdist(xy, xy),
                vehicle_limit=2, max_coord=100.)
    inst['node_demand'][:] = .2
    inst['node_tw_open'][:] = torch.tensor([.2, 0., 0.])
    inst['node_tw_close'][:] = torch.tensor([1., .25, 1.])
    inst['node_service'][:] = torch.tensor([.05, 0., 0.])
    inst['depot_tw_close'][:] = .5
    if event:
        inst['preset_events'] = [dict(trigger_time=25., duration=20., nodes=[2], multiplier=2.)]
    return inst


def fixed_routes(inst):
    routes, route, remaining = [], [], 1.
    for n, demand in enumerate(inst['node_demand'].tolist(), 1):
        if demand > remaining + 1e-5:
            routes.append(route)
            route, remaining = [], 1.
        route.append(n)
        remaining -= demand
    if route:
        routes.append(route)
    assert len(routes) <= inst['vehicle_limit']
    return routes


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hand_wait_service_customer_and_depot_lateness(self):
        for event, travel, late, finish in [(False, 100., 25., 60.), (True, 110., 45., 65.)]:
            with self.subTest(event=event):
                env = replay(fixture(event), PARAMS, [[1, 2], [3]])
                m = env.metrics()
                expected = dict(vehicles=2, travel_time=travel, distance=100., total_late=late,
                                late_count=3, served=3, unserved=0, finish_time=finish)
                for key, value in expected.items():
                    self.assertAlmostEqual(float(m[key][0]), value, places=4)
                self.assertAlmostEqual(float(m['reward'][0]),
                                       -(2/14 + (travel + 5*late)/1696.95), places=5)

    def test_reservations_service_time_and_t0_dispatch(self):
        env = TensorFleetEnv(fixture(), 1, PARAMS, 'cpu')
        env.step(torch.tensor([0]), validate=True)
        self.assertEqual(float(env.time[0]), 0)
        self.assertEqual(int(env.customer[0, 1]), 1)
        self.assertFalse(bool(env.mask()[0, 0]))
        env.step(torch.tensor([2]), validate=True)
        env.step(torch.tensor([7]), validate=True)
        self.assertAlmostEqual(float(env.time[0]), 25., places=5)
        self.assertEqual(int(env.customer[0, 1]), 2)
        self.assertEqual(int(env.customer[0, 3]), 1)
        self.assertFalse(bool(env.mask()[0, :3].any()))
        self.assertFalse(bool(env.mask()[0, 6]))

    def test_early_retirement_unserved_penalty(self):
        env = replay(fixture(), PARAMS, [[1]])
        self.assertEqual(int(env.metrics()['unserved'][0]), 2)
        self.assertAlmostEqual(float(env.metrics()['cost_unserved'][0]), 1000/3, places=4)

    def test_event_does_not_retime_departed_arc(self):
        inst = fixture()
        inst['preset_events'] = [dict(trigger_time=5., duration=100., nodes=[3], multiplier=3.)]
        env = replay(inst, PARAMS, [[1, 2], [3]])
        # Outbound 0->3 departs before the event: 30, return departs at 30: 90.
        self.assertAlmostEqual(float(env.travel[0]), 160., places=4)
        self.assertAlmostEqual(float(env.time[0]), 120., places=4)

    def test_legacy_fixed_route_replay_all_train_cases(self):
        split = json.loads((SOURCE/'split.json').read_text())
        root = SOURCE.parents[2]
        params = json.loads((root/'result/rc101_learned_first_train20_20261017/late5/config.json').read_text())['environment']
        for row in split['train']:
            with self.subTest(name=row['name']):
                inst, _ = load(row)
                routes = fixed_routes(inst)
                m = replay(inst, params, routes).metrics()
                old = VRPTWEnv(**params)
                old.load_problems(make_batch(inst, 1, torch.device('cpu')))
                old.reset()
                sequence = [0] + [n for route in routes for n in route + [0]]
                for n in sequence:
                    _, reward, done = old.step(torch.full((1, params['pomo_size']), n, dtype=torch.long))
                self.assertTrue(bool(done))
                for actual, expected in [(m['travel_time'][0], old.total_travel_time[0, 0]*inst['T']),
                                         (m['total_late'][0], old.total_late[0, 0]*inst['T']),
                                         (m['late_count'][0], old.n_late_stops[0, 0]),
                                         (m['vehicles'][0], old.depot_visit_count[0, 0]),
                                         (m['reward'][0], reward[0, 0])]:
                    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-5, atol=.01)
                report = _solution_stats(inst, [[0] + route + [0] for route in routes])
                for key, reported in [('travel_time', 'total_travel_time'), ('total_late', 'total_late'),
                                      ('distance', 'total_dist'), ('late_count', 'late_count')]:
                    self.assertAlmostEqual(float(m[key][0]), float(report[reported]), delta=.02)


if __name__ == '__main__':
    unittest.main()
