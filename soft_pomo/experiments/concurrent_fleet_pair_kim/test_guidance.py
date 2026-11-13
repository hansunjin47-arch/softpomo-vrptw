import unittest
import torch
from experiments.concurrent_fleet_guided.test_fixture import fixture, PARAMS, MP
from experiments.concurrent_fleet_pair.policy import FleetPolicy
from experiments.concurrent_fleet_pair_kim.run import GuidedEnv, GuidedPolicy, OnlineEnv


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.inst=fixture()
        self.inst['vehicle_limit']=3
        self.groups=[[1,2,3]]
        self.scores={0:{1:.1,2:.5,3:1.}}

    def env(self):
        return GuidedEnv(self.inst,2,PARAMS,'cpu',groups=self.groups,scores=self.scores)

    def test_dispatch_unchanged_and_guided_logits(self):
        env=self.env()
        model=GuidedPolicy(**MP)
        base=FleetPolicy(**MP)
        base.load_state_dict(model.state_dict())
        base._kv_cache={}
        model.new_customer_epsilon=base.new_customer_epsilon=.05
        starts,encoded=model.encode_events(self.inst,[])
        p=model.probabilities(env.frame(),starts,encoded,env)
        expected=base.probabilities(env.frame(),starts,encoded,env)
        torch.testing.assert_close(p,expected)
        env.step(torch.zeros(2,dtype=torch.long),True)
        self.assertTrue((env.guidance.vehicle_group[:,0]==0).all())
        p=model.probabilities(env.frame(),starts,encoded,env)
        expected=base.probabilities(env.frame(),starts,encoded,env)
        torch.testing.assert_close(p,expected)
        # Close dispatch through depot; frame now uses existing vehicles only.
        env.step(torch.full((2,),env.n,dtype=torch.long),True)
        # Independently check routing reweighting on a ready, open-customer state.
        env=self.env()
        env.step(torch.zeros(2,dtype=torch.long),True)
        env.dispatch_open[:]=False
        env.status[:,0]=3
        env.node[:,0]=1
        env._state_cache.clear()
        frame=env.frame()
        p=model.probabilities(frame,starts,encoded,env)
        expected=base.probabilities(frame,starts,encoded,env)
        bias=torch.tensor([[0.,0.,0.,0.,.5,2.5,5.]])
        expected=expected*torch.exp(bias)
        expected/=expected.sum(1,keepdim=True)
        torch.testing.assert_close(p,expected)
        (-p[:,env.n+2].log().mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue((p[~frame['mask']]==0).all())

    def test_online_disclosure_and_full_trajectory(self):
        calls=[]
        def refresh(env,t):
            calls.append((t,float(env.time[0])))
            env.update_event_confidence(self.scores,revealed_at_min=t,effective_time_min=float(env.time[0]))
        env=OnlineEnv(self.inst,1,PARAMS,'cpu',groups=self.groups,scores=self.scores,
            reports=[{'event_type':'ACCIDENT','revealed_at_min':5.}],on_disclosure=refresh)
        model=GuidedPolicy(**MP).eval()
        starts,encoded=model.encode_events(self.inst,[])
        for _ in range(30):
            if env.done.all():break
            action=model.probabilities(env.frame(),starts,encoded,env).argmax(1)
            env.step(action,True)
        self.assertTrue(env.done.all())
        self.assertEqual(len(calls),1)
        self.assertGreaterEqual(calls[0][1],5.)
        self.assertTrue(env.recorded_history)


if __name__=='__main__':unittest.main()
