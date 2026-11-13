import copy
import unittest
import torch
from experiments.concurrent_fleet_v1.test_env import instance
from experiments.concurrent_fleet_v3.env import FleetEnv
from experiments.concurrent_fleet_v3.policy import FleetPolicy
from experiments.concurrent_fleet_v3.gpu_batch import frame, distribution, sample, backward


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class Tests(unittest.TestCase):
    def test_scalar_and_padded_cuda_probabilities(self):
        torch.manual_seed(42)
        inst=instance()
        model=FleetPolicy().cuda()
        envs=[FleetEnv(inst,[]) for _ in range(3)]
        envs[1].step((-1,1)); envs[2].step((-1,2))
        encoded=model.encode(inst)
        batched=distribution(model,encoded,[frame(e) for e in envs])
        for i,e in enumerate(envs):
            scalar=model.distribution(e,encoded,e.actions())
            torch.testing.assert_close(batched.probs[i,:len(e.actions())],scalar.probs,atol=2e-6,rtol=1e-5)
            self.assertEqual(float(batched.probs[i,len(e.actions()):].sum()),0)

    def test_replay_gradient_matches_serial_state_gradients(self):
        torch.manual_seed(42)
        inst=instance()
        model=FleetPolicy().cuda()
        reference=copy.deepcopy(model)
        envs=[FleetEnv(inst,[]) for _ in range(8)]
        metrics,records=sample(model,inst,envs)
        rewards=torch.tensor([r['reward'] for r in metrics],device='cuda')
        adv=rewards-rewards.mean(); adv=adv/adv.std().clamp(min=1e-6)
        backward(model,inst,records,adv,batch_size=2,chunk=13)
        encoded=reference.encode(inst)
        loss=encoded.sum()*0
        for r in records:
            dist=distribution(reference,encoded,[r])
            loss=loss-adv[r['trajectory']]*dist.log_prob(torch.tensor([r['choice']],device='cuda')).sum()/16
        loss.backward()
        for p,q in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=2e-4,rtol=2e-3)
        old=[p.detach().clone() for p in model.parameters()]
        opt=torch.optim.Adam(model.parameters(),lr=1e-4)
        opt.step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(old,model.parameters())))
        self.assertTrue(all(p.is_cuda and torch.isfinite(p).all() for p in model.parameters()))


if __name__=='__main__':
    unittest.main()
