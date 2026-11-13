import copy
import unittest
import torch
from experiments.concurrent_fleet_v3.run import load, SOURCE
from experiments.concurrent_fleet_bias_guided.support import read, ROOT
from experiments.concurrent_fleet_pair.env import TensorFleetEnv


class ReturnMaskTests(unittest.TestCase):
    def env(self, enabled=True):
        inst,_=load(read(SOURCE/'split.json')['train'][0])
        params=copy.deepcopy(read(ROOT/'result/rc101_learned_first_train20_20261017/late5/config.json')['environment'])
        params['protect_last_service_vehicle']=enabled
        env=TensorFleetEnv(inst,1,params,'cpu')
        env.dispatch_open[:]=False
        env.time[:]=100
        env.k[:]=2
        env.customer[:]=2
        env.customer[:,1]=0
        env.demand[1]=.2
        env.status[:,0]=3
        env.node[:,0]=2
        env.load[:,0]=.4
        env.status[:,1]=3
        env.load[:,1]=.1
        return env

    def test_sole_provider(self):
        e=self.env(); f=e.frame()
        self.assertFalse(bool(f['mask'][0,e.n]))
        self.assertTrue(bool(f['mask'][0,e.n+1]))

    def test_other_ready_or_moving_provider(self):
        for status in (1,2,3):
            e=self.env(); e.load[:,1]=.3; e.status[:,1]=status; e.target[:,1]=3
            self.assertTrue(bool(e.frame()['mask'][0,e.n]))

    def test_returning_vehicle_is_not_provider(self):
        e=self.env(); e.load[:,1]=.9; e.status[:,1]=1; e.target[:,1]=0
        self.assertFalse(bool(e.frame()['mask'][0,e.n]))

    def test_unserviceable_is_not_deadlocked(self):
        e=self.env(); e.load[:,0]=.1
        self.assertTrue(bool(e.frame()['mask'][0,e.n]))

    def test_default_and_dispatch_unchanged(self):
        e=self.env(False); self.assertTrue(bool(e.frame()['mask'][0,e.n]))
        e=self.env(); e.dispatch_open[:]=True; e.time[:]=0
        self.assertTrue(bool(e.frame()['mask'][0,e.n]))


if __name__=='__main__': unittest.main()
