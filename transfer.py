import shutil
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs
from omegaconf.listconfig import ListConfig
import utils.dmc as dmc
import utils.utils as utils
from utils.logger import Logger
from utils.replay_buffer import ReplayBufferStorage, make_replay_loader
from utils.video import TrainVideoRecorder, VideoRecorder

from train import Workspace


torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.agent.obs_shape = obs_spec.shape
    if cfg.discrete_actions:
        cfg.agent.num_actions = action_spec.num_values
    else:
        cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class TransferWorkspace(Workspace):
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # setup the tasks
        assert isinstance(self.cfg['task_name'], ListConfig)
        self.num_tasks = len(self.cfg['task_name'])
        self.setup(task_id=0)

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self, task_id):
        # some assertions
        utils.assert_agent(self.cfg['agent']['agent_name'], self.cfg['pixel_obs'])

        # reset global step
        self._global_step = 0

        # create logger
        self.logger = Logger(self.work_dir)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name[task_id], self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed,
                                  self.cfg.pixel_obs, self.cfg.discrete_actions)
        self.eval_env = dmc.make(self.cfg.task_name[task_id], self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed,
                                 self.cfg.pixel_obs, self.cfg.discrete_actions)

        # delete the prev replay buffer from the disk
        if os.path.exists(self.work_dir / 'buffer'):
            shutil.rmtree(self.work_dir / 'buffer')

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None,
            fps=60 // self.cfg.action_repeat
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None,
            fps=60 // self.cfg.action_repeat
        )

        self.plot_dir = self.work_dir / 'plots'
        self.plot_dir.mkdir(exist_ok=True)
        self.model_dir = self.work_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)

        # save cfg
        utils.save_cfg(self.cfg, self.work_dir)

    def train_tasks(self):
        task_id = 0

        # train the initial task
        print(f"------- Training task {task_id+1} -------")
        self.train(task_id=task_id+1)

        while task_id < self.num_tasks-1:
            # setup the next task
            task_id += 1
            print(f"------- Training task {task_id+1} -------")
            self.setup(task_id)

            self.train()


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    root_dir = Path.cwd()
    workspace = TransferWorkspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train_tasks()


if __name__ == '__main__':
    main()
