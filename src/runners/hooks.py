

import os
import time

import torch
from torchvision.utils import make_grid


class Hooks:
    def __init__(self, cfg):
        self.hooks = []
        if cfg is None:
            return
        for hook_cfg in cfg:
            self.hooks.append(
                globals()[hook_cfg['type']](**hook_cfg)
            )

    def __call__(self, runner, hook_stage, **kwargs):
        for hook in self.hooks:
            getattr(hook, hook_stage)(runner, **kwargs)

    def set_logger(self, logger):
        for hook in self.hooks:
            hook.set_logger(logger)


class BaseHook:
    def __init__(self, **kwargs):
        pass

    def pre_iter(self, runner, **kwargs):
        pass

    def post_iter(self, runner, **kwargs):
        pass

    def pre_epoch(self, runner, **kwargs):
        pass

    def post_epoch(self, runner, **kwargs):
        pass

    def set_logger(self, logger):
        self.logger = logger


class MemoryUsageHook(BaseHook):
    def __init__(self, device='cuda:0', **kwargs):
        super().__init__(**kwargs)
        self.device = device

    def post_iter(self, runner, **kwargs):
        memory = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        torch.cuda.empty_cache()
        runner.logger.update(memory=memory)


class TrainTimerHook(BaseHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elapsed_time = 0
        self.start_time = None
        self.mean_time_per_itr = None
        self.observations = 0

    def pre_epoch(self, runner, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()
            self.last_time = time.time()

    def post_iter(self, runner, **kwargs):
        cur_time = time.time()
        self.elapsed_time = (cur_time - self.start_time) / 3600
        # total_run_iter = (runner.total_iter * (runner.epoch - runner.start_epoch)) + runner.iter
        # time_per_iter = self.elapsed_time / total_run_iter
        time_per_iter = (cur_time - self.last_time) / 3600
        m = self.observations
        if self.mean_time_per_itr is None:
            self.mean_time_per_itr = time_per_iter
        else:
            self.mean_time_per_itr = m / (m + 1) * self.mean_time_per_itr + 1 / (m + 1) * time_per_iter
        iter_remain = runner.total_iter * (runner.total_epochs - runner.epoch + 1) - runner.iter
        time_remain = self.mean_time_per_itr * iter_remain
        runner.logger.update(t_remain=time_remain, t_used=self.elapsed_time)
        self.last_time = cur_time
        self.observations += 1


class CheckPointsHook(BaseHook):
    def __init__(self, max_ckpt=3, epoch_every=None, iter_every=None, **kwargs):
        super().__init__(**kwargs)
        self.max_ckpt = max_ckpt
        self.epoch_every = epoch_every
        self.iter_every = iter_every

    def post_epoch(self, runner, **kwargs):
        if runner.gpu_id != 0:
            return
        self.save(runner, f'epoch{runner.epoch}.pth')
        if runner.epoch > self.max_ckpt:
            if (self.epoch_every is None or not
            (runner.epoch - self.max_ckpt) % self.epoch_every == 0):
                filename = os.path.join(
                    runner.logger.logdir,
                    f'epoch{runner.epoch - self.max_ckpt}.pth')
                if os.path.exists(filename):
                    os.remove(filename)

    def post_iter(self, runner, **kwargs):
        if runner.gpu_id != 0:
            return
        if self.iter_every is not None and runner.iter % self.iter_every == 0:
            self.save(runner, f'latest.pth')

    @staticmethod
    def save(runner, name):
        save_path = os.path.join(runner.logger.logdir, name)
        print(f'Saving checkpoint to {save_path}.')
        torch.save({
            'epoch': runner.epoch,
            'model': runner.model.state_dict(),
            # 'optimizer': runner.optimizer.state_dict(),
            'lr_scheduler': runner.lr_scheduler.state_dict(),
        }, save_path)


class ImgWriterHook(BaseHook):
    def __init__(self, max_n_img=32, log_every=50, **kwargs):
        super().__init__(**kwargs)
        self.max_n_img = max_n_img
        self.log_every = log_every

    def post_iter(self, runner, **kwargs):
        out_dict = runner.model.out_dict
        steps = (runner.epoch - 1) * runner.total_iter + runner.iter
        if steps % self.log_every == 0:
            for k, v in out_dict.items():
                if 'img' in k:
                    img = out_dict[k]
                    if img.ndim == 3:
                        img = img.unsqueeze(1)
                    n_img = min(len(img), self.max_n_img)
                    img = make_grid(img[:n_img])
                    runner.writer.add_image(f"train/{k.replace('_img', '')}", img, steps)









