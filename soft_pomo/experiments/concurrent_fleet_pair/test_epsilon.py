import unittest
import torch
from experiments.concurrent_fleet_pair.policy import mix_new_customers, FleetPolicy
from experiments.concurrent_fleet_pair.train import exploration_epsilon
from experiments.concurrent_fleet_pair.env import TensorFleetEnv
from experiments.concurrent_fleet_guided.test_fixture import fixture, PARAMS, MP


class Tests(unittest.TestCase):
    def test_mass_mask_and_identity(self):
        p = torch.tensor([[.54,.06,0.,.4], [0.,0.,0.,1.]])
        mask = p > 0
        self.assertIs(mix_new_customers(p,mask,3,0),p)
        q = mix_new_customers(p,mask,3,.2)
        torch.testing.assert_close(q,torch.tensor([[.492,.108,0.,.4],[0.,0.,0.,1.]]))
        torch.testing.assert_close(q.sum(1),p.sum(1))
        torch.testing.assert_close(q[:,:3].sum(1),p[:,:3].sum(1))

    def test_gradient_matches_mixture_formula(self):
        z = torch.tensor([.2,-1.,.7],requires_grad=True)
        p = z.softmax(0)[None]
        q = mix_new_customers(p,torch.ones_like(p,dtype=torch.bool),2,.2)
        actual = torch.autograd.grad(q[0,1].log(),z,retain_graph=True)[0]
        expected = torch.autograd.grad((.8*p[0,1]+.1*p[0,:2].sum()).log(),z)[0]
        torch.testing.assert_close(actual,expected)
        self.assertTrue(torch.isfinite(actual).all())

    def test_schedule(self):
        self.assertEqual(exploration_epsilon(.05,1,500),.05)
        self.assertEqual(exploration_epsilon(.05,500,500),0.)
        self.assertEqual(exploration_epsilon(.05,1000,500),0.)

    def test_eval_disables_exploration(self):
        torch.set_num_threads(1)
        model = FleetPolicy(**MP).eval()
        env = TensorFleetEnv(fixture(),4,PARAMS,device='cpu')
        with torch.no_grad():
            starts, enc = model.encode_events(env.inst,[])
            base = model.probabilities(env.frame(),starts,enc,env)
            model.new_customer_epsilon = 1.
            torch.testing.assert_close(base,model.probabilities(env.frame(),starts,enc,env))
            model.train()
            mixed = model.probabilities(env.frame(),starts,enc,env)
            valid = env.mask()[:,:env.n].float()
            torch.testing.assert_close(mixed[:,:env.n],valid/valid.sum(1,keepdim=True))

    def test_nonzero_floor_schedule(self):
        self.assertAlmostEqual(exploration_epsilon(.05,1,1000,.01),.05)
        self.assertAlmostEqual(exploration_epsilon(.05,500,1000,.01),.03002002002002)
        self.assertEqual(exploration_epsilon(.05,1000,1000,.01),.01)
        self.assertEqual(exploration_epsilon(.05,1200,1000,.01),.01)


if __name__ == '__main__':
    unittest.main()
