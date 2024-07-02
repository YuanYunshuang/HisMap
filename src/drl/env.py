import random

import torch
import cv2
import numpy as np

from src.utils.paint import paint_strokes


class Environment:
    def __init__(self, renderer, batch_size, max_step, img_width, data_path, device):
        self.batch_size = batch_size
        self.max_step = max_step
        self.action_space = (13)
        self.img_width = img_width
        self.observation_space = (self.batch_size, img_width, img_width, 7)
        self.device = device
        self.renderer = renderer
        self.load_data(data_path)

    def load_data(self, data_path):
        self.hismap = cv2.cvtColor(cv2.imread(data_path), cv2.COLOR_BGR2RGB)
        self.xmax = self.hismap.shape[0] - self.img_width
        self.ymax = self.hismap.shape[1] - self.img_width

    def reset(self):
        gt = []
        for b in range(self.batch_size):
            x = random.randint(0, self.xmax)
            y = random.randint(0, self.ymax)
            gt.append(self.hismap[x:x+self.img_width, y:y+self.img_width])
        gt = np.array(np.stack(gt, axis=0))
        self.gt = torch.from_numpy(gt).permute(0, 3, 1, 2).to(self.device)
        self.tot_reward = ((self.gt.float() / 255) ** 2).mean((1, 2, 3))
        self.stepnum = 0
        self.canvas = torch.zeros([self.batch_size, 3, self.img_width, self.img_width],
                                  dtype=torch.uint8).to(self.device)
        self.lastdis = self.ini_dis = self.cal_dist()
        return self.observation()

    def step(self, action):
        self.canvas = (paint_strokes(action, self.canvas.float() / 255, self.renderer) * 255).byte()
        self.stepnum += 1
        ob = self.observation()
        done = torch.tensor([(self.stepnum == self.max_step)] * self.batch_size, device=ob.device)
        reward = self.cal_reward() # np.array([0.] * self.batch_size)
        return ob.detach(), reward, done, None

    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)

    def cal_dist(self):
        """Squired distance between two images"""
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).mean((1, 2, 3))

    def observation(self):
        """Cat canvas, gt and step_idx as observation tensor."""
        # canvas B * 3 * width * width
        # gt B * 3 * width * width
        # T B * 1 * width * width
        T = torch.ones([self.batch_size, 1, self.img_width, self.img_width],
                       dtype=torch.uint8, device=self.device) * self.stepnum
        return torch.cat((self.canvas, self.gt, T), 1)

    def cal_reward(self):
        dist = self.cal_dist()
        reward = (self.lastdis - dist) / (self.ini_dis + 1e-8)
        self.lastdis = dist
        return reward


class FastEnv():
    def __init__(self, renderer, data_path, max_episode_length=10, env_batch=64,
                 img_width=128, device='cuda', writer=None):
        self.max_episode_length = max_episode_length
        self.env_batch = env_batch
        self.env = Environment(renderer, self.env_batch, self.max_episode_length,
                               img_width, data_path, device)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.writer = writer
        self.test = False
        self.log = 0

    def save_image(self, log, step):
        for i in range(0, self.env_batch, 20):
            canvas_gt = torch.cat([
                self.env.canvas[i],
                self.env.gt[i]
            ], dim=-1).detach().cpu().numpy()
            # canvas_gt = cv2.cvtColor(canvas_gt, cv2.COLOR_BGR2RGB)
            self.writer.add_image(f'{i}/canvas_gt_{step}.png', canvas_gt, log)
        # if step == self.max_episode_length:
        #     for i in range(self.env_batch):
        #         if self.env.imgid[i] < 50:
        #             gt = cv2.cvtColor((to_numpy(self.env.gt[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
        #             canvas = cv2.cvtColor((to_numpy(self.env.canvas[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
        #             self.writer.add_image(str(self.env.imgid[i]) + '/_target.png', gt, log)
        #             self.writer.add_image(str(self.env.imgid[i]) + '/_canvas.png', canvas, log)

    def step(self, action):
        with torch.no_grad():
            ob, r, d, _ = self.env.step(action)
        if d[0]:
            if not self.test:
                self.dist = self.get_dist()
                for i in range(self.env_batch):
                    self.writer.add_scalar('train/dist', self.dist[i], self.log)
                    self.log += 1
        return ob, r, d, _

    def get_dist(self):
        return (((self.env.gt.float() - self.env.canvas.float()) / 255) ** 2).mean((1, 2, 3))

    def reset(self, test=False, episode=0):
        self.test = test
        ob = self.env.reset()
        return ob

