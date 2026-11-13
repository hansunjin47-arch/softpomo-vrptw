"""Semantic checks for the user-requested NEW group-head bias."""
import unittest
from types import SimpleNamespace as S
from unittest.mock import patch
import torch
from experiments.concurrent_fleet_pair.policy import mix_new_customers
from experiments.concurrent_fleet_pair_kim.run import GuidedPolicy
from experiments.concurrent_fleet_pair_kim.start_head_guidance import StartHeadGuidedPolicy


class StartHeadChecks(unittest.TestCase):
    def setUp(self):
        self.model=StartHeadGuidedPolicy.__new__(StartHeadGuidedPolicy)
        torch.nn.Module.__init__(self.model)
        self.model.new_customer_epsilon=0.
        self.model.eval()
        self.env=S(n=4,dispatch_open=torch.tensor([True,False]),done=torch.tensor([False,False]),
                   guidance=S(groups=[[1,2],[3,4]]))
        self.q=torch.tensor([[.1,.2,.1,0.,.3,.3],[0.,0.,0.,0.,.6,.4]],requires_grad=True)
        self.frame={'mask':self.q.detach()>0}

    def calculate(self):
        with patch.object(GuidedPolicy,'probabilities',lambda *args:self.q):
            return self.model.probabilities(self.frame,None,None,self.env)

    def test_heads_only_and_lifecycle(self):
        p=self.calculate()
        # Equal bonus only for NEW(1), NEW(3); NEW(2) and OLD are unboosted.
        self.assertAlmostEqual(float(((p[0,0]/p[0,1])/(self.q[0,0]/self.q[0,1])).detach()),
                               float(torch.exp(torch.tensor(5.))),places=3)
        self.assertAlmostEqual(float((p[0,1]/p[0,4]).detach()),float((self.q[0,1]/self.q[0,4]).detach()),places=6)
        self.assertTrue(torch.equal(p[1],self.q[1]))
        self.assertTrue(torch.equal(p==0,self.q==0))
        self.assertTrue(torch.allclose(p.sum(1),torch.ones(2)))
        (-p[0,0].log()).backward()
        self.assertTrue(torch.isfinite(self.q.grad).all())

    def test_zero_beta_exact(self):
        self.model.new_head_beta=0
        self.assertTrue(torch.equal(self.calculate(),self.q))

    def test_epsilon_after_bias_once(self):
        biased=self.calculate()
        self.model.train();self.model.new_customer_epsilon=.05
        actual=self.calculate()
        expected=mix_new_customers(biased,self.frame['mask'],4,.05)
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(self.model.new_customer_epsilon,.05)

    def test_assigned_head_not_reenabled(self):
        self.q=self.q.detach().clone();self.q[0,0]=0
        self.q[0]/=self.q[0].sum();self.frame['mask']=self.q>0
        self.assertEqual(float(self.calculate()[0,0]),0.)

    def test_invalid_strength(self):
        for beta in (-1,float('nan'),float('inf')):
            self.model.new_head_beta=beta
            with self.assertRaises(ValueError):self.calculate()

if __name__=='__main__':unittest.main()
