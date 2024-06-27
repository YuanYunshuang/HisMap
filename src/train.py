#!/usr/bin/env python3
import os
import random
import numpy as np
import argparse
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils.common import *
from src.drl.ddpg import DDPG
from src.drl.env import FastEnv
from src.drl.evaluator import Evaluator
from src.dlmodule.fcn import FCN

exp = os.path.abspath('.').split('/')[-1]
writer = SummaryWriter('../train_log/{}'.format(exp))
os.system('ln -sf ../train_log/{} ./log'.format(exp))
os.makedirs('./model', exist_ok=True)


def train(agent, env, evaluate):
    time_stamp = time.time()
    step = episode = episode_steps = 0
    observation = None
    noise_factor = args.noise_factor
    while step <= args.train_steps:
        step += 1
        episode_steps += 1
        # reset if it is the start of episode
        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)
        action = agent.act(observation, noise_factor=noise_factor)
        observation, reward, done, _ = env.step(action)
        agent.save_sample(reward, observation, done, step)
        if (episode_steps >= args.max_step and args.max_step):
            if step > args.warmup:
                # [optional] evaluate
                if episode > 0 and args.validate_interval > 0 and episode % args.validate_interval == 0:
                    reward, dist = evaluate(env, agent.select_action, debug=args.debug)
                    if args.debug: prRed('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'.format(
                        step - 1, np.mean(reward), np.mean(dist),np.var(dist))
                    )
                    writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                    writer.add_scalar('validate/var_dist', np.var(dist), step)
                    agent.save_model(args.output)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.
            tot_value_loss = 0.
            if step > args.warmup:
                if step < 10000 * args.max_step:
                    lr = (3e-4, 1e-3)
                elif step < 20000 * args.max_step:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)
                for i in range(args.episode_train_times):
                    Q, value_loss = agent.update_policy(lr)
                    tot_Q += Q.data.cpu().numpy()
                    tot_value_loss += value_loss.data.cpu().numpy()
                writer.add_scalar('train/critic_lr', lr[0], step)
                writer.add_scalar('train/actor_lr', lr[1], step)
                writer.add_scalar('train/Q', tot_Q / args.episode_train_times, step)
                writer.add_scalar('train/critic_loss', tot_value_loss / args.episode_train_times, step)
            if args.debug: prBlack('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}' \
                              .format(episode, step, train_time_interval, time.time() - time_stamp))
            time_stamp = time.time()
            # reset
            observation = None
            episode_steps = 0
            episode += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning to Paint Historical Maps')

    # hyper-parameter
    parser.add_argument('--warmup', default=400, type=int,
                        help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.95 ** 5, type=float, help='discount factor')
    parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    parser.add_argument('--imgsize', default=128, type=int, help='replay memory size')
    parser.add_argument('--env_batch', default=96, type=int, help='concurrent environment number')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
    parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise')
    parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
    parser.add_argument('--validate_episodes', default=5, type=int,
                        help='how many episode to perform during validation')
    parser.add_argument('--train_steps', default=2000000, type=int, help='total train steps')
    parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')
    parser.add_argument('--data_path', default='/code/HisMapPainter/data/3922_2017.tif', type=str, help='data path')
    parser.add_argument('--render_path', default='/code/HisMapPainter/render.pkl', type=str, help='data path')
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    parser.add_argument('--output', default='./model', type=str, help='Resuming model path for testing')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, "Paint")
    seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    renderer = FCN(args.render_path).cuda()
    fenv = FastEnv(renderer, args.data_path, args.max_step, args.env_batch, writer=writer)
    agent = DDPG(renderer, args.batch_size, args.env_batch, args.max_step,
                 args.tau, args.discount, args.rmsize, args.imgsize,
                 writer, args.resume, args.output).cuda()
    evaluate = Evaluator(args, writer)
    print('observation_space', fenv.observation_space, 'action_space', fenv.action_space)
    train(agent, fenv, evaluate)
