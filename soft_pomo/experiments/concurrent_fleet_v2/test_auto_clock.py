import unittest
import torch
from experiments.concurrent_fleet_v1.test_env import instance
from experiments.concurrent_fleet_v2.env import FleetEnv
from experiments.concurrent_fleet_v2.policy import FleetPolicy, rollout


class Tests(unittest.TestCase):
    def test_clock_is_automatic(self):
        inst=instance();inst['vehicle_limit']=2
        e=FleetEnv(inst,[])
        self.assertFalse(hasattr(e,'ADVANCE'))
        with self.assertRaises(ValueError): e.step((-2,0))
        e.step((-1,1));self.assertEqual(e.time,0)
        e.step((-1,2))
        self.assertAlmostEqual(e.time,10,places=5)
        self.assertEqual([v['status'] for v in e.vehicles],['ready','moving'])
        self.assertEqual([v['departures'][0] for v in e.vehicles],[0,0])
        e.step((0,3))
        self.assertAlmostEqual(e.time,20,places=5)

    def test_idle_is_one_vehicle_not_global_time(self):
        inst=instance();inst['vehicle_limit']=3
        e=FleetEnv(inst,[]);e.step((-1,1))
        e.step((-1,0))
        self.assertEqual(e.time,0)
        self.assertEqual(e.dispatch_slots,1)
        e.step((-1,0))
        self.assertAlmostEqual(e.time,10,places=5)
        e.step((-1,0));e.step((-1,0))
        self.assertAlmostEqual(e.time,10,places=5)
        self.assertIn((0,3),e.actions())

    def test_all_ready_vehicles_are_decided_before_time_moves(self):
        inst=instance();inst['vehicle_limit']=2;inst['tt'][0,2]=.1
        e=FleetEnv(inst,[]);e.step((-1,1));e.step((-1,2))
        self.assertTrue(all(v['status']=='ready' for v in e.vehicles))
        t=e.time;e.step((0,0))
        self.assertEqual(e.time,t)
        self.assertIn((1,3),e.actions())

    def test_accident_processed_before_next_decision(self):
        inst=instance();inst['vehicle_limit']=1
        ev=dict(event_type='ACCIDENT',revealed_at_min=5,start_min=5,end_min=20,affected_nodes=[2])
        e=FleetEnv(inst,[ev])
        self.assertEqual(e.observation()['public'],[])
        e.step((-1,1))
        self.assertAlmostEqual(e.time,10,places=5)
        self.assertEqual(len(e.observation()['public']),1)
        self.assertEqual([r['event'] for r in e.trace],['depart','disclosure','arrival','service_end'])

    def test_fleet_exhaustion_and_success(self):
        for complete in (False,True):
            inst=instance();inst['vehicle_limit']=1;inst['node_demand'][:]=.2
            e=FleetEnv(inst,[]);e.step((-1,1))
            if complete: e.step((0,2));e.step((0,3))
            e.step((0,0));self.assertTrue(e.done)
            r=e.metrics();self.assertEqual(r['unserved'],0 if complete else 2)
            self.assertEqual(r['vehicles'],1)

    def test_rollout_and_backward(self):
        torch.manual_seed(42);torch.set_num_threads(1)
        inst=instance();inst['vehicle_limit']=2
        model=FleetPolicy();encoded=model.encode(inst)
        logs=[];rewards=[]
        for _ in range(4):
            e=FleetEnv(inst,[]);r,lp=rollout(model,e,encoded)
            self.assertLessEqual(r['vehicles'],2)
            self.assertEqual([x['time'] for x in e.trace],sorted(x['time'] for x in e.trace))
            logs.append(lp);rewards.append(r['reward'])
        r=torch.tensor(rewards);loss=-((r-r.mean())*torch.stack(logs)).mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__=='__main__': unittest.main()
