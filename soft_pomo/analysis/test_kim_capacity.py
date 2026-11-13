import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_soft_cluster import SoftClusterTrainer
from vrptw_env import load_solomon


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.inst=load_solomon(str(ROOT.parent/'data/Solomon/c109.txt'))
        self.trainer=SoftClusterTrainer.__new__(SoftClusterTrainer)

    def test_c109_retries_instead_of_returning_overload(self):
        groups=self.trainer._cluster_customers(self.inst,10)
        self.assertGreater(len(groups),10)
        self.assertEqual(sorted(n for ns in groups for n in ns),list(range(1,101)))
        for ns in groups:
            self.assertLessEqual(sum(float(self.inst['node_demand'][n-1]) for n in ns),1+1e-6)

    def test_failure_at_limit_is_explicit(self):
        self.inst['vehicle_limit']=10
        with self.assertRaisesRegex(RuntimeError,'exceeds capacity'):
            self.trainer._cluster_customers(self.inst,10)

    def test_initial_count_above_limit_rejected(self):
        self.inst['vehicle_limit']=9
        with self.assertRaises(ValueError):
            self.trainer._cluster_customers(self.inst,10)


if __name__=='__main__':unittest.main()
