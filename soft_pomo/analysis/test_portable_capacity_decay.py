import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from experiments.concurrent_fleet_pair_kim import portable as p
from experiments.concurrent_fleet_pair.env import TensorFleetEnv

class Checks(unittest.TestCase):
    def test_inputs_masks_and_training_flags(self):
        torch.set_num_threads(1)
        for case,minimum in [('c101',10),('r101',8),('rc101',9)]:
            with tempfile.TemporaryDirectory() as d:
                a=SimpleNamespace(case=case,vehicle_cost=100,method='pure',artifact=None,
                    out=Path(d)/'run',seed=42,head_beta=5.,resume=False,smoke_updates=0,
                    dispatch_capacity_mask=True,lr_decay=True)
                source,split,groups,cfg=p.inputs(a)
                inst,_=p.load(split['train'][0])
                env=TensorFleetEnv(inst,1,dict(cfg['environment'],require_dispatch_capacity=True),'cpu')
                self.assertEqual(env.minimum_dispatch_vehicles,minimum)
                for _ in range(minimum):
                    mask=env.mask();self.assertFalse(bool(mask[:,env.n:].any()))
                    env.step(mask[:,:env.n].long().argmax(1),validate=True)
                self.assertTrue(bool(env.mask()[:,env.n:].any()))
                with patch.object(p.trainer,'run'):p.train(a)
                saved=p.read(a.out/'config.json')
                self.assertTrue(saved['environment']['require_dispatch_capacity'])
                self.assertTrue(saved['trainer']['lr_decay_after500'])
if __name__=='__main__':unittest.main()
