import sys
from pathlib import Path
import unittest
import math
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.concurrent_fleet_exploration.objective import dispatch_entropy, entropy_beta


class EntropyTests(unittest.TestCase):
    def test_mask_values_and_gradient(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            logits=torch.tensor([[0.,0.,0.,0.,0.,0.]]*4,device=device,requires_grad=True)
            mask=torch.tensor([[1,1,0,0,0,1],[1,1,0,0,0,0],
                               [0,0,0,0,0,1],[0,0,1,1,1,0]],device=device,dtype=torch.bool)
            probs=logits.masked_fill(~mask,-torch.inf).softmax(1)
            h,valid=dispatch_entropy(probs,mask,torch.tensor([1,1,1,0],device=device,dtype=torch.bool),
                                     torch.ones(4,device=device,dtype=torch.bool),2)
            self.assertEqual(valid.tolist(),[True,False,False,False])
            self.assertAlmostEqual(float(h[0]),-1/3*math.log(1/3)-2/3*math.log(2/3),places=6)
            self.assertTrue(torch.equal(h[1:],torch.zeros_like(h[1:])))
            (-h.mean()).backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertLess(float(logits.grad[0,-1]),0.)
            self.assertTrue(torch.equal(logits.grad[1:],torch.zeros_like(logits.grad[1:])))

    def test_schedule(self):
        self.assertEqual(entropy_beta(1,.01,500),.01)
        self.assertEqual(entropy_beta(500,.01,500),0.)
        self.assertEqual(entropy_beta(1000,.01,500),0.)


if __name__=='__main__':
    unittest.main(verbosity=2)
