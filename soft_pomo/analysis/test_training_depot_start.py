"""Regression: the real sequential training loop must charge its first arc."""
import unittest
from unittest.mock import patch
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_vrptw import VRPTWTrainer, VRPTWModel, model_params
from vrptw_env import VRPTWEnv, load_solomon


class Tests(unittest.TestCase):
    def test_training_first_arc(self):
        torch.set_num_threads(1)
        for learned in (False, True):
            with self.subTest(learned=learned):
                torch.manual_seed(42)
                trainer = VRPTWTrainer.__new__(VRPTWTrainer)
                trainer.device = torch.device('cpu')
                trainer.trainer_params = dict(max_steps=600)
                trainer._use_amp = False
                trainer.env = VRPTWEnv(problem_size=100, pomo_size=4)
                trainer.model = VRPTWModel(**dict(model_params, embedding_dim=16,
                    encoder_layer_num=1, head_num=2, qkv_dim=8, ff_hidden_dim=32,
                    learn_first_customer=learned))
                inst = load_solomon(str(ROOT/'data/rc101_event_pilot_discrete_v1/RC101_ACC_P001.txt'))
                original = trainer.env.step
                actions = []
                class Checked(Exception): pass
                def step(selected):
                    actions.append(selected.clone())
                    if len(actions) == 2:
                        env = trainer.env
                        previous = env.current_node.clone()
                        expected = env.tt[env.BATCH_IDX, previous, selected] * env._event_multiplier_arc(previous, selected, env.current_time)
                    result = original(selected)
                    if len(actions) == 2:
                        self.assertTrue((actions[0] == 0).all())
                        self.assertTrue((selected > 0).all())
                        self.assertTrue((expected > 0).all())
                        torch.testing.assert_close(env.total_travel_time, expected)
                        if not learned:
                            self.assertEqual(selected[0].unique().numel(), 4)
                        raise Checked()
                    return result
                with patch.object(trainer.env, 'step', side_effect=step):
                    with self.assertRaises(Checked):
                        trainer._train_one_batch(inst, 2)


if __name__ == '__main__': unittest.main()
