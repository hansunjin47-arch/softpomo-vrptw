"""
VRPTW Tester — based on POMO CVRPTester.
POMO inference: pomo_size rollouts per instance → best solution.
Optionally: 8-fold augmentation → best of (8 × pomo_size) solutions.
"""
import os
import torch

from VRPTWEnv import VRPTWEnv as Env
from VRPTWModel import VRPTWModel as Model
from VRPTWProblemDef import load_solomon, make_batch, augment_xy_data_by_8_fold


class VRPTWTester:
    def __init__(self, env_params, model_params, tester_params):
        self.env_params    = env_params
        self.model_params  = model_params
        self.tester_params = tester_params

        USE_CUDA = tester_params.get('use_cuda', torch.cuda.is_available())
        if USE_CUDA:
            dev_num = tester_params.get('cuda_device_num', 0)
            torch.cuda.set_device(dev_num)
            self.device = torch.device('cuda', dev_num)
        else:
            self.device = torch.device('cpu')

        self.env   = Env(**env_params)
        self.model = Model(**model_params).to(self.device)

        ckpt_path = tester_params['model_load']['path']
        ckpt      = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        print(f'[Tester] loaded checkpoint: {ckpt_path}')

    # ------------------------------------------------------------------
    def run(self, test_instance_paths: list):
        self.model.eval()
        aug = self.tester_params.get('augmentation_enable', False)
        aug_factor = self.tester_params.get('aug_factor', 8) if aug else 1

        for path in test_instance_paths:
            inst = load_solomon(path)
            score, aug_score = self._test_one_instance(inst, aug_factor)
            print(f'  {inst["name"]:8s}  '
                  f'no-aug={score:.4f}  aug={aug_score:.4f}  '
                  f'(N={inst["n_customers"]})')

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _test_one_instance(self, inst: dict, aug_factor: int):
        batch_size = self.tester_params.get('test_batch_size', 1)

        if aug_factor > 1:
            # augment xy coordinates, repeat TW data
            batch = make_batch(inst, batch_size * aug_factor, self.device)
            aug_depot = augment_xy_data_by_8_fold(
                batch['depot_xy'].reshape(aug_factor, batch_size, 1, 2)[0])
            aug_node  = augment_xy_data_by_8_fold(
                batch['node_xy'].reshape(aug_factor, batch_size, -1, 2)[0])
            batch['depot_xy'] = aug_depot
            batch['node_xy']  = aug_node
            # TW data doesn't depend on coordinate orientation → just repeat
        else:
            batch = make_batch(inst, batch_size, self.device)

        self.env.load_problems(batch)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        state, reward, done = self.env.pre_step()
        max_steps = self.tester_params.get('max_steps', 500)
        step = 0
        while not done and step < max_steps:
            selected, _ = self.model(state)
            state, reward, done = self.env.step(selected)
            step += 1

        if reward is None:
            reward = -self.env._get_travel_distance()

        # shape: (aug_factor * batch_size, pomo_size)
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)

        max_pomo,  _ = aug_reward.max(dim=2)   # (aug, batch)
        no_aug_score  = -max_pomo[0].float().mean().item()

        max_aug, _   = max_pomo.max(dim=0)     # (batch,)
        aug_score     = -max_aug.float().mean().item()

        return no_aug_score, aug_score
