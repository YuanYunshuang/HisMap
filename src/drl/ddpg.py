import torch
from torch.optim import Adam
from torch import nn
from torch import autograd

from src.dlmodule.resnet import ResNet
from src.dlmodule.wgan import WGAN
from src.drl.rpm import RPM
from src.drl.util import hard_update, soft_update
from src.utils.img import img_coord
from src.utils.paint import paint_strokes


class DDPG(nn.Module):
    def __init__(self, renderer, batch_size=64, env_batch=1, max_step=40,
                 tau=0.001, discount=0.9, rmsize=800, img_size=128,
                 writer=None, resume=None, output_path=None):
        super().__init__()

        self.max_step = max_step
        self.env_batch = env_batch
        self.batch_size = batch_size
        self.img_size = img_size
        self.renderer = renderer

        self.actor = ResNet(9, 18, 65, 'BN', 'relu')  # target, canvas, stepnum, coordconv 3 + 3 + 1 + 2
        self.actor_target = ResNet(9, 18, 65, 'BN', 'relu')
        self.critic = ResNet(3 + 9, 18, 1, 'WN', 'trelu')  # add the last canvas for better prediction
        self.critic_target = ResNet(3 + 9, 18, 1, 'WN', 'trelu')
        self.wgan = WGAN()
        self.wgan_target = WGAN()
        hard_update(self.wgan_target, self.wgan)

        self.actor_optim = Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-2)
        self.wgan_optim = Adam(self.wgan.parameters(), lr=3e-4, betas=(0.5, 0.999))
        self.criterion = nn.MSELoss()

        if (resume != None):
            self.load_weights(resume)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = RPM(rmsize * max_step)

        # Hyper-parameters
        self.tau = tau
        self.discount = discount

        # Tensorboard
        self.writer = writer
        self.log = 0

        self.state = [None] * self.env_batch  # Most recent state
        self.action = [None] * self.env_batch  # Most recent action
        self.coord = nn.Parameter(img_coord(img_size).unsqueeze(0), requires_grad=False)

    def forward(self, state, target=False):
        state = torch.cat((state[:, :6].float() / 255,
                           state[:, 6:7].float() / self.max_step,
                           self.coord.expand(state.shape[0], 2, 128, 128)), 1)
        if target:
            return self.actor_target(state)
        else:
            return self.actor(state)

    def reset(self, obs, factor):
        self.state = obs
        self.noise_level = torch.rand(self.env_batch, device=obs.device) * factor

    def act(self, state, return_fix=False, noise_factor=0):
        self.eval()
        with torch.no_grad():
            action = self.forward(state)
        if noise_factor > 0:
            action = self.noise_action(noise_factor, state, action)
        self.train()
        self.action = action
        if return_fix:
            return action
        return self.action

    def save_sample(self, reward, state, done, step):
        for i in range(self.env_batch):
            self.memory.append([self.state[i],
                                self.action[i],
                                reward[i],
                                state[i],
                                done[i]])
        self.state = state

    def update_policy(self, lr):
        self.log += 1

        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]

        # Sample batch
        state, action, reward, next_state, terminate = (
            self.memory.sample_batch(self.batch_size))

        self.update_gan(next_state)

        with torch.no_grad():
            next_action = self.forward(next_state, True)
            target_q, _ = self.evaluate(next_state, next_action, True)
            target_q = self.discount * ((1 - terminate.float()).view(-1, 1)) * target_q

        cur_q, step_reward = self.evaluate(state, action)
        target_q += step_reward.detach()

        value_loss = self.criterion(cur_q, target_q)
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        action = self.forward(state)
        pre_q, _ = self.evaluate(state.detach(), action)
        policy_loss = -pre_q.mean()
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return -policy_loss, value_loss

    def update_gan(self, state):
        fake_data = state[:, :3].float() / 255
        real_data = state[:, 3:6].float() / 255

        fake_data = fake_data.detach()
        real_data = real_data.detach()
        fake = torch.cat([real_data, fake_data], 1)
        real = torch.cat([real_data, real_data], 1)
        D_real = self.wgan(real).mean()
        D_fake = self.wgan(fake).mean()
        gradient_penalty = self.gan_gradient_penalty(real, fake, real.shape[0])
        self.wgan_optim.zero_grad()
        D_cost = D_fake - D_real + gradient_penalty
        D_cost.backward()
        self.wgan_optim.step()
        soft_update(self.wgan_target, self.wgan, 0.001)

        if self.log % 20 == 0:
            self.writer.add_scalar('train/gan_fake', D_fake, self.log)
            self.writer.add_scalar('train/gan_real', D_real, self.log)
            self.writer.add_scalar('train/gan_penal', gradient_penalty, self.log)

    def gan_gradient_penalty(self, real_data, fake_data, batch_size, LAMBDA=10):
        device = real_data.device
        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
        alpha = alpha.view(batch_size, 6, self.img_size, self.img_size)
        fake_data = fake_data.view(batch_size, 6, self.img_size, self.img_size)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates.requires_grad = True
        disc_interpolates = self.wgan(interpolates)
        gradients = autograd.grad(disc_interpolates, interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def evaluate(self, state, action, target=False):
        T = state[:, 6 : 7]
        gt = state[:, 3 : 6].float() / 255
        canvas0 = state[:, :3].float() / 255
        canvas1 = paint_strokes(action, canvas0, self.renderer)
        gan_reward = self.gan_reward(canvas1, gt) - self.gan_reward(canvas0, gt)
        # L2_reward = ((canvas0 - gt) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1)
        coord = self.coord.expand(state.shape[0], 2, 128, 128)
        merged_state = torch.cat([canvas0, canvas1, gt, (T + 1).float() / self.max_step, coord], 1)
        # canvas0 is not necessarily added
        if target:
            Q = self.critic_target(merged_state)
            return (Q + gan_reward), gan_reward
        else:
            Q = self.critic(merged_state)
            if self.log % 20 == 0:
                self.writer.add_scalar('train/expect_reward', Q.mean(), self.log)
                self.writer.add_scalar('train/gan_reward', gan_reward.mean(), self.log)
            return (Q + gan_reward), gan_reward

    def gan_reward(self, fake_data, real_data):
        return self.wgan_target(torch.cat([real_data, fake_data], 1))