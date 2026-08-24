# Original POMO CVRPTester.py — copied as-is (imports updated to VRPTW)
import torch

from VRPTWEnv import VRPTWEnv as Env
from VRPTWModel import VRPTWModel as Model


class VRPTWTester:
    def __init__(self, env_params, model_params, tester_params):
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        USE_CUDA = tester_params.get('use_cuda', torch.cuda.is_available())
        if USE_CUDA:
            cuda_device_num = tester_params.get('cuda_device_num', 0)
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
        else:
            self.device = torch.device('cpu')

        self.env = Env(**env_params)
        self.model = Model(**model_params).to(self.device)

        ckpt_path = '{path}/checkpoint-{epoch}.pt'.format(**tester_params['model_load'])
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        print(f'[Tester] Loaded checkpoint: {ckpt_path}')

    def run(self):
        score_sum = aug_score_sum = count = 0

        test_data_load = self.tester_params.get('test_data_load', {'enable': False})
        if test_data_load.get('enable'):
            self.env.use_saved_problems(test_data_load['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(batch_size)
            score_sum += score * batch_size
            aug_score_sum += aug_score * batch_size
            count += batch_size
            episode += batch_size

            print(f'episode {episode:4d}/{test_num_episode}  '
                  f'score={score:.4f}  aug_score={aug_score:.4f}')

        print(f'\n*** Test Done ***')
        print(f'  NO-AUG  score: {score_sum/count:.4f}')
        print(f'  AUG     score: {aug_score_sum/count:.4f}')

    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        aug_factor = self.tester_params['aug_factor'] if self.tester_params.get('augmentation_enable') else 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)       # best of pomo rollouts
        no_aug_score = -max_pomo_reward[0, :].float().mean()

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # best of augmentations
        aug_score = -max_aug_pomo_reward.float().mean()

        return no_aug_score.item(), aug_score.item()
