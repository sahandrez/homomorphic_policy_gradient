"""
Copyright 2022 Sahand Rezaei-Shoshtari. All Rights Reserved.

Official implementation of DHPG.
Author: Sahand Rezaei-Shoshtari
"""

import copy

import hydra
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.drqv2 import RandomShiftsAug
from models.cnn import PixelEncoder
from models.core import DeterministicActor, DDPGCritic, RewardPredictor, ActionEncoder, StateEncoder
from models.transition_model import make_transition_model
import utils.utils as utils


_ABSTRACTION_TYPES = ['base']
_HPG_UPDATE_TYPES = ['base', 'double_add', 'double_ind_together', 'double_ind_cyclic']


class HPGAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim, abstraction_type,
                 hpg_update_type, transition_model_type, pixel_obs, hidden_dim, linear_approx,
                 matching_dims, critic_target_tau, num_expl_steps, update_every_steps,
                 stddev_schedule, stddev_clip, clipped_noise, use_aug, homomorphic_coef):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.clipped_noise = clipped_noise
        self.action_dim = action_shape[0]
        self.transition_model_type = transition_model_type
        self.lax_bisim_coef = homomorphic_coef
        self.pixel_obs = pixel_obs
        self.matching_dims = matching_dims
        self.abstraction_type = abstraction_type
        self.hpg_update_type = hpg_update_type
        self.use_aug = use_aug

        state_dim = feature_dim if pixel_obs else obs_shape[0]

        assert abstraction_type in _ABSTRACTION_TYPES, f"Available abstraction types are {_ABSTRACTION_TYPES}."
        assert hpg_update_type in _HPG_UPDATE_TYPES, f"Available HPG update types are {_HPG_UPDATE_TYPES}."
        if use_aug:
            assert pixel_obs, "Image augmentation is for pixel observations."

        self.abstract_state_dim = obs_shape[0] if matching_dims and not pixel_obs else feature_dim
        self.abstract_action_dim = action_shape[0] if matching_dims else feature_dim

        # models
        self.actor = DeterministicActor(state_dim, action_shape[0],
                                        hidden_dim, linear_approx).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = DDPGCritic(state_dim, action_shape[0],
                                 hidden_dim, linear_approx).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.abstract_critic = DDPGCritic(self.abstract_state_dim, self.abstract_action_dim,
                                          hidden_dim, linear_approx).to(self.device)
        self.abstract_critic_target = copy.deepcopy(self.abstract_critic)

        self.action_encoder = ActionEncoder(state_dim, self.action_dim,
                                            self.abstract_action_dim, hidden_dim).to(self.device)
        self.state_encoder = StateEncoder(state_dim, self.abstract_state_dim,
                                          hidden_dim).to(self.device)
        self.transition_model = make_transition_model(
            transition_model_type, self.abstract_state_dim, self.abstract_action_dim, hidden_dim
        ).to(device)
        self.reward_predictor = RewardPredictor(self.abstract_state_dim, hidden_dim).to(device)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.abstract_critic_optimizer = torch.optim.Adam(self.abstract_critic.parameters(), lr=lr)
        self.action_encoder_optimizer = torch.optim.Adam(self.action_encoder.parameters(), lr=lr)
        self.state_encoder_optimizer = torch.optim.Adam(self.state_encoder.parameters(), lr=lr)
        self.reward_predictor_optimizer = torch.optim.Adam(self.reward_predictor.parameters(), lr=lr)
        self.transition_model_optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=lr)

        # pixel observations
        if pixel_obs:
            self.pixel_encoder = PixelEncoder(obs_shape, feature_dim).to(self.device)
            self.pixel_encoder_optimizer = torch.optim.Adam(self.pixel_encoder.parameters(), lr=lr)

        # data augmentation
        if use_aug and pixel_obs:
            self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()
        self.actor_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.abstract_critic.train(training)
        self.action_encoder.train(training)
        self.state_encoder.train(training)
        self.reward_predictor.train(training)
        if self.pixel_obs:
            self.pixel_encoder.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        if self.pixel_obs:
            obs = self.pixel_encoder(obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        action = self.actor(obs)
        if eval_mode:
            action = action.cpu().numpy()[0]
        else:
            action = action.cpu().numpy()[0] + np.random.normal(0, stddev, size=self.action_dim)
            if step < self.num_expl_steps:
                action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
        action = action.astype(np.float32)

        return action.astype(np.float32)

    def observe(self, obs, action):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        action = torch.as_tensor(action, device=self.device).float().unsqueeze(0)

        if self.pixel_obs:
            obs = self.pixel_encoder(obs)

        z = self.state_encoder(obs)
        abstract_action = self.action_encoder(obs, action)
        next_z = self.transition_model.sample_prediction(z, abstract_action)
        q = self.critic(obs, action)
        q_abstract = self.abstract_critic(z, abstract_action)

        return {
            'state': obs.cpu().numpy()[0],
            'abstract_state': z.cpu().numpy()[0],
            'abstract_next_state': next_z.cpu().numpy()[0],
            'abstract_action': abstract_action.cpu().numpy()[0],
            'value': q.cpu().numpy()[0],
            'value_abstract': q_abstract.cpu().numpy()[0]
        }

    def get_lax_bisim(self, abstract_state, abstract_action, reward, discount):
        # Sample random states across episodes at random
        batch_size = abstract_state.size(0)
        perm = np.random.permutation(batch_size)

        abstract_state_2 = abstract_state[perm]
        abstract_action_2 = abstract_action[perm]

        with torch.no_grad():
            pred_next_latent_mu_1, pred_next_latent_sigma_1 = self.transition_model(abstract_state, abstract_action)
            reward_2 = reward[perm]
        if pred_next_latent_sigma_1 is None:
            pred_next_latent_sigma_1 = torch.zeros_like(pred_next_latent_mu_1)
        if pred_next_latent_mu_1.ndim == 2:                             # shape (B, Z), no ensemble
            pred_next_latent_mu_2 = pred_next_latent_mu_1[perm]
            pred_next_latent_sigma_2 = pred_next_latent_sigma_1[perm]
        elif pred_next_latent_mu_1.ndim == 3:                           # shape (E, B, Z), using an ensemble
            pred_next_latent_mu_2 = pred_next_latent_mu_1[:, perm, :]
            pred_next_latent_sigma_2 = pred_next_latent_sigma_1[:, perm, :]
        else:
            raise NotImplementedError

        z_dist = F.smooth_l1_loss(abstract_state, abstract_state_2, reduction='none').mean(dim=1)
        r_dist = F.smooth_l1_loss(reward, reward_2, reduction='none').mean(dim=1)
        # abstract_action_dist = F.smooth_l1_loss(abstract_action, abstract_action_2,
        #                                         reduction='none').mean(dim=1)

        if self.transition_model_type in ['', 'deterministic']:
            transition_dist = F.smooth_l1_loss(pred_next_latent_mu_1, pred_next_latent_mu_2, reduction='none')
        else:
            transition_dist = torch.sqrt(
                (pred_next_latent_mu_1 - pred_next_latent_mu_2).pow(2) +
                (pred_next_latent_sigma_1 - pred_next_latent_sigma_2).pow(2)
            )
        transition_dist = transition_dist.mean(dim=-1)

        lax_bisimilarity = r_dist + discount.squeeze() * transition_dist
        lax_bisim_loss = (z_dist - lax_bisimilarity).pow(2).mean()
        # lax_bisim_loss = (z_dist + abstract_action_dist - lax_bisimilarity).pow(2).mean()
        return lax_bisim_loss

    def get_transition_reward_loss(self, abstract_state, abstract_action, reward, abstract_next_state):
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(abstract_state, abstract_action)
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        # transition loss
        diff = (pred_next_latent_mu - abstract_next_state.detach()) / pred_next_latent_sigma
        transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))

        # reward loss
        pred_next_latent = self.transition_model.sample_prediction(abstract_state, abstract_action)
        pred_next_reward = self.reward_predictor(pred_next_latent)
        reward_loss = F.mse_loss(pred_next_reward, reward)

        return transition_loss, reward_loss

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        # Abstract states
        z = self.state_encoder(obs)
        with torch.no_grad():
            next_z = self.state_encoder(next_obs)
        # Abstract actions
        abstract_action = self.action_encoder(obs, action)
        # Compute MDP homomorphism loss
        lax_bisim_loss = self.get_lax_bisim(z, abstract_action, reward, discount)
        # Compute transition and reward loss
        transition_loss, reward_loss = self.get_transition_reward_loss(z, abstract_action, reward, next_z)

        # Compute critic loss
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

        loss = critic_loss + self.lax_bisim_coef * lax_bisim_loss + transition_loss + reward_loss

        # Optimize the critic
        if self.pixel_obs:
            self.pixel_encoder_optimizer.zero_grad(set_to_none=True)
        self.state_encoder_optimizer.zero_grad(set_to_none=True)
        self.action_encoder_optimizer.zero_grad(set_to_none=True)
        self.reward_predictor_optimizer.zero_grad(set_to_none=True)
        self.transition_model_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        self.transition_model_optimizer.step()
        self.reward_predictor_optimizer.step()
        self.action_encoder_optimizer.step()
        self.state_encoder_optimizer.step()
        if self.pixel_obs:
            self.pixel_encoder_optimizer.step()

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q'] = current_Q.mean().item()
        metrics['critic_loss'] = critic_loss.item()
        metrics['transition_loss'] = transition_loss.item()
        metrics['reward_loss'] = reward_loss.item()
        metrics['homomorphic_loss'] = lax_bisim_loss.item()

        return metrics

    def update_abstract_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            # Abstract states and actions
            z = self.state_encoder(obs)
            next_z = self.state_encoder(next_obs)
            abstract_action = self.action_encoder(obs, action)

            if self.clipped_noise:
                # Select action according to policy and add clipped noise
                stddev = utils.schedule(self.stddev_schedule, step)
                noise = (torch.randn_like(abstract_action) * stddev).clamp(-self.stddev_clip, self.stddev_clip)

                next_abstract_action = (self.action_encoder(next_obs, self.actor_target(next_obs)) + noise).clamp(-1.0, 1.0)
            else:
                next_abstract_action = self.action_encoder(next_obs, self.actor_target(next_obs))

            # Compute the target Q value
            target_Q = self.abstract_critic_target(next_z, next_abstract_action)
            target_Q = reward + discount * target_Q

        # Get current Q estimates
        current_Q = self.abstract_critic(z, abstract_action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.abstract_critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.abstract_critic_optimizer.step()

        # Compute the value equivalence between the two critics
        with torch.no_grad():
            Q = self.critic(obs, action)
            abstract_Q = self.abstract_critic(z, abstract_action)
            value_equivalence = torch.abs(Q - abstract_Q).mean()

        metrics['abs_critic_target_q'] = target_Q.mean().item()
        metrics['abs_critic_q'] = current_Q.mean().item()
        metrics['abs_critic_loss'] = critic_loss.item()
        metrics['value_equivalence'] = value_equivalence.item()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        # Compute actor loss
        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        if self.hpg_update_type == 'double_add':
            # Does not update the actor here but uses the HPG theorem to reduce variance
            abstract_actor_loss, metrics_update = self.update_abstract_actor(obs, step, perform_update=False)
            metrics.update(metrics_update)
            actor_loss += abstract_actor_loss

        # Optimize the actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics['actor_loss'] = actor_loss.item()

        return metrics

    def update_abstract_actor(self, obs, step, perform_update=True):
        metrics = dict()

        with torch.no_grad():
            z = self.state_encoder(obs)

        # Compute actor loss
        actor_loss = -self.abstract_critic(z, self.action_encoder(obs, self.actor(obs))).mean()

        metrics['abs_actor_loss'] = actor_loss.item()

        if perform_update:
            # Optimize the actor
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            return metrics
        else:
            return actor_loss, metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _ = utils.to_torch(
            batch, self.device)

        obs = obs.float()
        next_obs = next_obs.float()

        # encode
        if self.pixel_obs:
            # image augmentation
            if self.use_aug:
                obs = self.aug(obs)
                next_obs = self.aug(next_obs)

            obs = self.pixel_encoder(obs)
            with torch.no_grad():
                next_obs = self.pixel_encoder(next_obs)

        metrics['batch_reward'] = reward.mean().item()

        # update critic, encoder, reward and transition model
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update the abstract critic
        metrics.update(self.update_abstract_critic(obs, action, reward, discount, next_obs, step))

        if step % self.update_every_steps == 0:
            if self.hpg_update_type == 'base':
                # update actor only based on HPG
                metrics.update(self.update_abstract_actor(obs.detach(), step))
            else:
                # update actor based on DPG and HPG (depending on the HPG update type)
                metrics.update(self.update_actor(obs.detach(), step))

        # update actor based on HPG independently of DPG
        if self.hpg_update_type == 'double_ind_together' and step % self.update_every_steps == 0:
            metrics.update(self.update_abstract_actor(obs.detach(), step, perform_update=True))
        elif self.hpg_update_type == 'double_ind_cyclic' and step % self.update_every_steps == 1:
            metrics.update(self.update_abstract_actor(obs.detach(), step, perform_update=True))

        # update target networks
        if step % self.update_every_steps == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
            utils.soft_update_params(self.abstract_critic, self.abstract_critic_target, self.critic_target_tau)
            utils.soft_update_params(self.actor, self.actor_target, self.critic_target_tau)

        return metrics

    def save(self, model_dir, step):
        model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.actor.state_dict(), f'{model_save_dir}/actor.pt')
        torch.save(self.critic.state_dict(), f'{model_save_dir}/critic.pt')
        torch.save(self.abstract_critic.state_dict(), f'{model_save_dir}/abstract_critic.pt')
        torch.save(self.state_encoder.state_dict(), f'{model_save_dir}/state_encoder.pt')
        torch.save(self.action_encoder.state_dict(), f'{model_save_dir}/action_encoder.pt')
        torch.save(self.transition_model.state_dict(), f'{model_save_dir}/transition_model.pt')
        torch.save(self.reward_predictor.state_dict(), f'{model_save_dir}/reward_predictor.pt')
        if self.pixel_obs:
            torch.save(self.pixel_encoder.state_dict(), f'{model_save_dir}/pixel_encoder.pt')

    def load(self, model_dir, step):
        print(f"Loading the model from {model_dir}, step: {step}")
        model_load_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')

        self.actor.load_state_dict(
            torch.load(f'{model_load_dir}/actor.pt', map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f'{model_load_dir}/critic.pt', map_location=self.device)
        )
        self.abstract_critic.load_state_dict(
            torch.load(f'{model_load_dir}/abstract_critic.pt', map_location=self.device)
        )
        self.state_encoder.load_state_dict(
            torch.load(f'{model_load_dir}/state_encoder.pt', map_location=self.device)
        )
        self.action_encoder.load_state_dict(
            torch.load(f'{model_load_dir}/action_encoder.pt', map_location=self.device)
        )
        self.transition_model.load_state_dict(
            torch.load(f'{model_load_dir}/transition_model.pt', map_location=self.device)
        )
        self.reward_predictor.load_state_dict(
            torch.load(f'{model_load_dir}/reward_predictor.pt', map_location=self.device)
        )
        if self.pixel_obs:
            self.pixel_encoder.load_state_dict(
                torch.load(f'{model_load_dir}/pixel_encoder.pt', map_location=self.device)
            )
