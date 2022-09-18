"""
Copyright 2022 Sahand Rezaei-Shoshtari. All Rights Reserved.

Implementation of DDPG
https://arxiv.org/abs/1509.02971

Code is based on:
https://github.com/sfujim/TD3/blob/master/OurDDPG.py
"""

import hydra
import copy
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core import DeterministicActor, DDPGCritic
import utils.utils as utils


class DDPGAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, linear_approx, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip,
                 clipped_noise):

        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.clipped_noise = clipped_noise
        self.stddev_clip = stddev_clip
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr

        # models
        self.actor = DeterministicActor(obs_shape[0], action_shape[0],
                                        hidden_dim, linear_approx).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = DDPGCritic(obs_shape[0], action_shape[0],
                                 hidden_dim, linear_approx).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.actor_target.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        stddev = utils.schedule(self.stddev_schedule, step)
        action = self.actor(obs.float().unsqueeze(0))
        if eval_mode:
            action = action.cpu().numpy()[0]
        else:
            action = action.cpu().numpy()[0] + np.random.normal(0, stddev, size=self.action_dim)
            if step < self.num_expl_steps:
                action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
        return action.astype(np.float32)

    def observe(self, obs, action):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        action = torch.as_tensor(action, device=self.device).float().unsqueeze(0)

        q = self.critic(obs, action)

        return {
            'state': obs.cpu().numpy()[0],
            'value': q.cpu().numpy()[0]
        }

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            if self.clipped_noise:
                # Select action according to policy and add clipped noise
                stddev = utils.schedule(self.stddev_schedule, step)
                noise = (torch.randn_like(action) * stddev).clamp(-self.stddev_clip, self.stddev_clip)

                next_action = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            else:
                next_action = self.actor_target(next_obs)

            # Compute the target Q value
            target_Q = self.critic_target(next_obs, next_action)
            target_Q = reward + discount * target_Q

        # Get current Q estimates
        current_Q = self.critic(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q'] = current_Q.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        # Compute actor loss
        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics['actor_loss'] = actor_loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _ = utils.to_torch(
            batch, self.device)

        obs = obs.float()
        next_obs = next_obs.float()

        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor (delayed)
        if step % self.update_every_steps == 0:
            metrics.update(self.update_actor(obs.detach(), step))

            # update target networks
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
            utils.soft_update_params(self.actor, self.actor_target, self.critic_target_tau)

        return metrics

    def save(self, model_dir, step):
        model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.actor.state_dict(), f'{model_save_dir}/actor.pt')
        torch.save(self.critic.state_dict(), f'{model_save_dir}/critic.pt')

    def load(self, model_dir, step):
        print(f"Loading the model from {model_dir}, step: {step}")
        model_load_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')

        self.actor.load_state_dict(
            torch.load(f'{model_load_dir}/actor.pt', map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f'{model_load_dir}/critic.pt', map_location=self.device)
        )
