import unittest
import numpy as np
import torch
from experiments.concurrent_fleet_v1.env import FleetEnv
from experiments.concurrent_fleet_v1.policy import FleetPolicy, rollout
from vrptw_env import VRPTWEnv, make_batch


def instance():
    xy = torch.tensor([[0.,0.],[.1,0.],[0.,.2],[.3,0.]])
    return dict(name='tiny', n_customers=3, T=100., vehicle_capacity=10,
                depot_xy=xy[:1],node_xy=xy[1:],depot_demand=torch.zeros(1),
                node_demand=torch.tensor([.6,.6,.4]),depot_tw_open=torch.zeros(1),
                node_tw_open=torch.zeros(3),depot_tw_close=torch.ones(1),
                node_tw_close=torch.ones(3),depot_service=torch.zeros(1),
                node_service=torch.zeros(3),tt=torch.cdist(xy,xy),preset_events=[])


class Tests(unittest.TestCase):
    def test_legacy_reward_parity_including_unserved(self):
        for complete in (False,True):
            inst=instance();inst['vehicle_limit']=1;inst['node_demand'][:]=.2
            inst['node_tw_close'][:]=.05
            params=dict(problem_size=3,pomo_size=1,vehicle_penalty=3/14,
                        late_penalty=5,Lt_max=16.9695,time_penalty=1,TT_max=16.9695,
                        unserved_penalty=500,dist_penalty_weight=0,late_count_penalty=0)
            legacy=VRPTWEnv(**params)
            legacy.load_problems(make_batch(inst,1,torch.device('cpu')));legacy.reset()
            new=FleetEnv(inst,[],environment_params=params)
            route=[0,1,2,3,0] if complete else [0,1,0]
            for n in route:
                _,reward,done=legacy.step(torch.tensor([[n]]))
            for n in route[1:]:
                new.step((-1 if not new.vehicles else 0,n))
                while new.vehicles[0]['status'] in ('moving','service'):
                    new.step((-2,0))
            self.assertTrue(done and new.done)
            self.assertAlmostEqual(float(reward),new.metrics()['reward'],places=4)

    def test_fleet_limit_failure_and_penalty(self):
        inst=instance();inst['vehicle_limit']=1
        e=FleetEnv(inst,[])
        e.step((-1,1))
        self.assertFalse(any(i==e.NEW for i,n in e.actions()))
        with self.assertRaises(ValueError):
            e.step((-1,2))
        e.step((-2,0));e.step((0,0));e.step((-2,0))
        self.assertTrue(e.done)
        r=e.metrics()
        self.assertEqual(r['vehicles'],1)
        self.assertEqual(r['unserved'],2)
        self.assertFalse(r['feasible'])
        self.assertAlmostEqual(r['cost'],1/14+r['travel_time']/1696.95+500*2/3)

    def test_last_vehicle_can_finish_remaining_customers(self):
        inst=instance();inst['vehicle_limit']=1;inst['node_demand'][:]=.2
        e=FleetEnv(inst,[])
        for action in [(-1,1),(-2,0),(0,2),(-2,0),(0,3),(-2,0),(0,0),(-2,0)]:
            e.step(action)
        self.assertTrue(e.done)
        self.assertTrue(e.metrics()['feasible'])

    def test_simultaneous_dispatch_and_reservations(self):
        e=FleetEnv(instance(),[])
        e.step((-1,1));e.step((-1,2))
        self.assertEqual(e.time,0)
        self.assertEqual([v['departure'] for v in e.vehicles],[0,0])
        self.assertEqual(e.customer_state.tolist(),[2,1,1,0])
        self.assertFalse(any(n in (1,2) for _,n in e.actions()))
        e.step((-2,0))
        self.assertAlmostEqual(e.time,10,places=5)
        self.assertEqual(e.vehicles[0]['status'],'ready')
        self.assertEqual(e.vehicles[1]['status'],'moving')
        self.assertEqual(e.customer_state[2],1)

    def test_global_disclosure_and_no_lookahead(self):
        public=dict(event_type='ACCIDENT',revealed_at_min=15,start_min=15,
                    end_min=30,affected_nodes=[3],vehicles_involved=3)
        a=FleetEnv(instance(),[public]);b=FleetEnv(instance(),[])
        self.assertEqual(a.actions(),b.actions())
        np.testing.assert_equal(a.action_features(a.actions()),b.action_features(b.actions()))
        a.step((-1,1));a.step((-1,2));a.step((-2,0));a.step((-2,0))
        self.assertEqual(a.time,15)
        self.assertEqual(len(a.observation()['public']),1)
        self.assertEqual(a.customer_state[2],1)
        a.step((-1,3))
        self.assertEqual(a.vehicles[2]['departure'],15)

    def test_capacity_and_deadlock_prevention(self):
        e=FleetEnv(instance(),[])
        self.assertNotIn((-2,0),e.actions())
        e.step((-1,1));e.step((-2,0))
        self.assertNotIn((0,2),e.actions())
        self.assertIn((0,3),e.actions())
        self.assertNotIn((-2,0),e.actions())

    def test_arc_entry_effect_and_return_cost(self):
        inst=instance()
        inst['preset_events']=[dict(type='ACCIDENT',trigger_time=5,duration=50,
                                   nodes=[1],multiplier=2)]
        e=FleetEnv(inst,[dict(event_type='ACCIDENT',revealed_at_min=5,
                            start_min=5,end_min=55,affected_nodes=[1])])
        e.step((-1,1));e.step((-2,0))
        self.assertEqual(e.time,5)
        e.step((-2,0))
        self.assertAlmostEqual(e.time,10,places=5)
        e.step((0,0));e.step((-2,0))
        self.assertAlmostEqual(e.time,30,places=5)
        self.assertAlmostEqual(e.travel,30,places=5)

    def test_future_event_does_not_change_earlier_decisions(self):
        a=instance();b=instance()
        b['preset_events']=[dict(type='ACCIDENT',trigger_time=50,duration=50,
                                nodes=[1,2,3],multiplier=10)]
        public=dict(event_type='ACCIDENT',revealed_at_min=50,start_min=50,
                    end_min=100,affected_nodes=[1,2,3],vehicles_involved=5)
        ea=FleetEnv(a,[]);eb=FleetEnv(b,[public])
        for action in [(-1,1),(-1,2),(-2,0),(0,3)]:
            self.assertEqual(ea.actions(),eb.actions())
            np.testing.assert_equal(ea.action_features(ea.actions()),eb.action_features(eb.actions()))
            np.testing.assert_equal(ea.observation()['fleet'],eb.observation()['fleet'])
            ea.step(action);eb.step(action)

    def test_time_window_wait_and_delayed_dispatch(self):
        inst=instance();inst['node_tw_open'][0]=.5
        inst['node_service'][0]=.1
        e=FleetEnv(inst,[])
        e.step((-1,1));e.step((-2,0))
        self.assertEqual(e.vehicles[0]['status'],'service')
        self.assertNotIn((0,3),e.actions())
        e.step((-1,2))
        self.assertAlmostEqual(e.vehicles[1]['departure'],10,places=5)
        while e.time<59:
            e.step((-2,0))
        self.assertAlmostEqual(e.time,60,places=4)
        self.assertEqual(e.vehicles[0]['status'],'ready')

    def test_end_to_end_gradients(self):
        torch.manual_seed(42);torch.set_num_threads(1)
        p=FleetPolicy();inst=instance();encoded=p.encode(inst)
        results=[];logs=[]
        for _ in range(4):
            env=FleetEnv(inst,[])
            r,log=rollout(p,env,encoded)
            self.assertEqual(r['served'],3)
            self.assertEqual(sorted(n for route in r['routes'] for n in route if n),[1,2,3])
            self.assertTrue(all(sum(float(inst['node_demand'][n-1]) for n in route if n)<=1+1e-6 for route in r['routes']))
            self.assertEqual([x['time'] for x in env.trace],sorted(x['time'] for x in env.trace))
            self.assertAlmostEqual(r['cost'],r['vehicles']/14+(r['travel_time']+5*r['total_late'])/1696.95)
            results.append(r['reward']);logs.append(log)
        reward=torch.tensor(results)
        loss=-((reward-reward.mean())*torch.stack(logs)).mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in p.parameters() if x.grad is not None))
        self.assertGreater(sum(float(x.grad.abs().sum()) for x in p.parameters() if x.grad is not None),0)


if __name__=='__main__':
    unittest.main()
