import unittest
import torch
from test_active_cluster_bias import ActiveClusterBiasTests
from run_rc101_pure500_phase2 import Phase2Trainer


class StartGateTests(ActiveClusterBiasTests):
    def setUp(self):
        super().setUp()
        self.obj.__class__ = Phase2Trainer
        self.obj._require_reference_start = False
        self.obj._reference_heads = {'test': torch.tensor([1, 3])}

    def test_matching_and_middle_starts(self):
        self.obj._require_reference_start = True
        out = self.bias([[0,1], [0,2], [0,3], [0,4]])
        torch.testing.assert_close(out[0,0], self.base[0])
        torch.testing.assert_close(out[0,2], self.base[1])
        self.assertEqual(out[0,[1,3]].count_nonzero().item(), 0)

    def test_gate_persists_and_resets_per_vehicle(self):
        self.obj._require_reference_start = True
        self.assertEqual(self.bias([[0,2,1]]).count_nonzero().item(), 0)
        torch.testing.assert_close(self.bias([[0,1,2]])[0,0], self.base[0])
        torch.testing.assert_close(self.bias([[0,2,0,3]])[0,0], self.base[1])
        self.assertEqual(self.bias([[0,1,0,4]]).count_nonzero().item(), 0)

    def test_accident_does_not_override_start_gate(self):
        self.obj._require_reference_start = True
        self.obj._acc_stage_events = {'test':[dict(t_start=.5,tensor=self.base*2)]}
        self.base = torch.zeros_like(self.base)
        out = self.bias([[0,1],[0,1],[0,2]], times=[.4,.6,.6])
        self.assertEqual(out[0,[0,2]].count_nonzero().item(),0)
        self.assertAlmostEqual(out[0,1,0].item(), .8)


if __name__ == '__main__':
    unittest.main()
