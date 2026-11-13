import unittest
import torch
from experiments.concurrent_fleet_v1.test_env import instance
from experiments.concurrent_fleet_v3.env import FleetEnv
from experiments.concurrent_fleet_v3.policy import FleetPolicy, rollout


class Tests(unittest.TestCase):
    def test_two_dispatches_stay_at_zero_then_mask_new(self):
        inst=instance();inst['vehicle_limit']=3
        e=FleetEnv(inst,[]);e.step((-1,1));e.step((-1,2))
        self.assertEqual(e.time,0)
        self.assertIn((0,-1),e.actions())
        self.assertNotIn((1,-1),e.actions())
        e.step((0,-1))
        self.assertAlmostEqual(e.time,10,places=5)
        self.assertTrue(all(i==0 and n>=0 for i,n in e.actions()))
        with self.assertRaises(ValueError): e.step((-1,3))
        e.step((0,3))
        self.assertAlmostEqual(e.time,20,places=5)
        self.assertTrue(all(i==1 for i,n in e.actions()))
        self.assertEqual([v['departures'][0] for v in e.vehicles],[0,0])

    def test_service_completion_not_arrival_orders_vehicles(self):
        inst=instance();inst['node_service'][0]=.3
        e=FleetEnv(inst,[]);e.step((-1,1));e.step((-1,2))
        self.assertIn((1,-1),e.actions())
        self.assertNotIn((0,-1),e.actions())
        e.step((1,-1))
        self.assertAlmostEqual(e.time,20,places=5)
        self.assertEqual(e.vehicles[0]['status'],'service')

    def test_disclosure_before_destination_choice(self):
        ev=dict(event_type='ACCIDENT',revealed_at_min=5,start_min=5,end_min=20,affected_nodes=[3])
        e=FleetEnv(instance(),[ev]);e.step((-1,1))
        self.assertEqual(e.observation()['public'],[])
        self.assertNotIn((0,3),e.actions())
        e.step((0,-1))
        self.assertAlmostEqual(e.time,10,places=5)
        self.assertEqual(len(e.observation()['public']),1)
        self.assertIn((0,3),e.actions())

    def test_equal_completion_times_do_not_add_waiting(self):
        inst=instance();inst['tt'][0,2]=.1
        e=FleetEnv(inst,[]);e.step((-1,1));e.step((-1,2));e.step((0,-1))
        t=e.time;e.step((0,0))
        self.assertEqual(e.time,t)
        self.assertTrue(all(i==1 for i,n in e.actions()))

    def test_insufficient_initial_fleet_penalizes_unserved(self):
        inst=instance();inst['vehicle_limit']=3
        e=FleetEnv(inst,[]);e.step((-1,1));e.step((0,-1));e.step((0,0))
        self.assertTrue(e.done)
        r=e.metrics();self.assertEqual(r['vehicles'],1);self.assertEqual(r['unserved'],2)
        self.assertAlmostEqual(r['cost'],1/14+r['travel_time']/1696.95+500*2/3)

    def test_no_wait_action_and_vehicle_cap(self):
        inst=instance();inst['vehicle_limit']=1
        e=FleetEnv(inst,[])
        self.assertNotIn((-1,0),e.actions());self.assertFalse(hasattr(e,'ADVANCE'))
        e.step((-1,1))
        self.assertEqual(e.actions(),[(0,-1)])

    def test_rollout_backward_and_temporal_invariants(self):
        torch.manual_seed(42);torch.set_num_threads(1)
        model=FleetPolicy();inst=instance();encoded=model.encode(inst)
        logs=[];rewards=[]
        for _ in range(4):
            e=FleetEnv(inst,[]);r,lp=rollout(model,e,encoded)
            self.assertEqual([x['time'] for x in e.trace],sorted(x['time'] for x in e.trace))
            self.assertTrue(all(v['departures'][0]==0 for v in e.vehicles))
            ends={(x['vehicle'],x['time']) for x in e.trace if x['event']=='service_end'}
            for v in range(len(e.vehicles)):
                for departure in e.vehicles[v]['departures'][1:]:
                    self.assertIn((v,departure),ends)
            logs.append(lp);rewards.append(r['reward'])
        r=torch.tensor(rewards);loss=-((r-r.mean())*torch.stack(logs)).mean();loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__=='__main__': unittest.main()
