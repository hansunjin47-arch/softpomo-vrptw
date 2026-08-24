# Original POMO CVRPTrainer.py — copied as-is (imports updated to VRPTW)
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from VRPTWEnv import VRPTWEnv as Env
from VRPTWModel import VRPTWModel as Model


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VRPTWTrainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        USE_CUDA = trainer_params.get('use_cuda', torch.cuda.is_available())
        if USE_CUDA:
            cuda_device_num = trainer_params.get('cuda_device_num', 0)
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
        else:
            self.device = torch.device('cpu')

        self.model = Model(**model_params).to(self.device)
        self.env = Env(**env_params)
        self.optimizer = Optimizer(self.model.parameters(), **optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **optimizer_params['scheduler'])

        self.start_epoch = 1
        model_load = trainer_params.get('model_load', {'enable': False})
        if model_load.get('enable'):
            ckpt_path = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch'] - 1
            self.start_epoch = 1 + model_load['epoch']
            print(f'[Resume] Loaded checkpoint from epoch {model_load["epoch"]}')

        self.result_dir = trainer_params.get('result_dir', 'result')
        os.makedirs(self.result_dir, exist_ok=True)

        self.score_log = []
        self.loss_log = []

    def run(self):
        t0 = time.time()
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.scheduler.step()

            train_score, train_loss = self._train_one_epoch(epoch)
            self.score_log.append(train_score)
            self.loss_log.append(train_loss)

            elapsed = time.time() - t0
            print(f'Epoch {epoch:4d}/{self.trainer_params["epochs"]}  '
                  f'score={train_score:.4f}  loss={train_loss:.4f}  '
                  f'elapsed={elapsed:.0f}s')

            model_save_interval = self.trainer_params['logging'].get('model_save_interval', 500)
            all_done = (epoch == self.trainer_params['epochs'])
            if all_done or epoch % model_save_interval == 0:
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                }
                torch.save(ckpt, f'{self.result_dir}/checkpoint-{epoch}.pt')
                print(f'  [Saved] checkpoint-{epoch}.pt')

        print('\n*** Training Done ***')

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            episode += batch_size

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0), device=self.device)
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # best of pomo rollouts
        score_mean = -max_pomo_reward.float().mean()  # positive distance value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()
