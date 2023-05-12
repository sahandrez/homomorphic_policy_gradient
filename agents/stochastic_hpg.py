"""
Copyright 2023 Sahand Rezaei-Shoshtari. All Rights Reserved.

Official implementation of Stochastic DHPG.
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
from models.core import StochasticActor, Critic, RewardPredictor, ActionEncoder, StateEncoder
from models.transition_model import make_transition_model
import utils.utils as utils


_ABSTRACTION_TYPES = ['base']
_HPG_UPDATE_TYPES = ['base', 'double']


class StochasticHPGAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 abstraction_type, hpg_update_type, transition_model_type,
                 pixel_obs, hidden_dim, linear_approx, init_temperature, alpha_lr,
                 actor_log_std_min, actor_log_std_max, matching_dims, critic_target_tau,
                 num_expl_steps, update_every_steps, clipped_noise, use_aug, homomorphic_coef,
                 lifting_weight, lifting_repeat_obs, stddev_schedule):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.clipped_noise = clipped_noise
        self.action_dim = action_shape[0]
        self.transition_model_type = transition_model_type
        self.lax_bisim_coef = homomorphic_coef
        self.pixel_obs = pixel_obs
        self.matching_dims = matching_dims
        self.abstraction_type = abstraction_type
        self.hpg_update_type = hpg_update_type
        self.use_aug = use_aug
        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.lifting_weight = lifting_weight
        self.lifting_repeat_obs = lifting_repeat_obs
        self.actor_log_std_min = actor_log_std_min
        self.actor_log_std_max = actor_log_std_max
        self.stddev_schedule = stddev_schedule

        state_dim = feature_dim if pixel_obs else obs_shape[0]

        assert abstraction_type in _ABSTRACTION_TYPES, f"Available abstraction types are {_ABSTRACTION_TYPES}."
        assert hpg_update_type in _HPG_UPDATE_TYPES, f"Available HPG update types are {_HPG_UPDATE_TYPES}."
        if use_aug:
            assert pixel_obs, "Image augmentation is for pixel observations."

        self.abstract_state_dim = obs_shape[0] if matching_dims and not pixel_obs else feature_dim
        self.abstract_action_dim = action_shape[0]

        # models
        self.actor = StochasticActor(state_dim, action_shape[0], hidden_dim, linear_approx,
                                     actor_log_std_min, actor_log_std_max).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.abstract_actor = StochasticActor(self.abstract_state_dim, self.abstract_action_dim, hidden_dim,
                                              linear_approx, actor_log_std_min, actor_log_std_max).to(device)
        self.abstract_actor_target = copy.deepcopy(self.abstract_actor)

        self.critic = Critic(state_dim, action_shape[0],
                             hidden_dim, linear_approx).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.abstract_critic = Critic(self.abstract_state_dim, self.abstract_action_dim,
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

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.abstract_actor_optimizer = torch.optim.Adam(self.abstract_actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.abstract_critic_optimizer = torch.optim.Adam(self.abstract_critic.parameters(), lr=lr)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.action_encoder_optimizer = torch.optim.Adam(self.action_encoder.parameters(), lr=lr)
        self.state_encoder_optimizer = torch.optim.Adam(self.state_encoder.parameters(), lr=lr)
        self.reward_predictor_optimizer = torch.optim.Adam(self.reward_predictor.parameters(), lr=lr)
        self.transition_model_optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=lr)

        # pixel observations
        if pixel_obs:
            self.pixel_encoder = PixelEncoder(obs_shape, feature_dim).to(self.device)
            self.pixel_encoder_optimizer = torch.optim.Adam(self.pixel_encoder.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()
        self.abstract_critic_target.train()
        self.actor_target.train()
        self.abstract_actor_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.abstract_actor.train(training)
        self.critic.train(training)
        self.abstract_critic.train(training)
        self.action_encoder.train(training)
        self.state_encoder.train(training)
        self.reward_predictor.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, step)
        if self.pixel_obs:
            obs = self.pixel_encoder(obs)

        if eval_mode:
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            action = mu.clamp(-1, 1).cpu().numpy()[0]
        else:
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            action = pi.clamp(-1, 1).cpu().numpy()[0]
            if step < self.num_expl_steps:
                action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
        return action.astype(np.float32)

    def observe(self, obs, action):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        action = torch.as_tensor(action, device=self.device).float().unsqueeze(0)

        if self.pixel_obs:
            obs = self.pixel_encoder(obs)

        z = self.state_encoder(obs)
        abstract_action = self.action_encoder(obs, action)
        next_z = self.transition_model.sample_prediction(z, abstract_action)
        q, _ = self.critic(obs, action)
        q_abstract, _ = self.abstract_critic(z, abstract_action)

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
            _, policy_action, log_pi, _ = self.actor_target(next_obs)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (discount * target_Q)

        # Get current Q estimates
        Q1, Q2 = self.critic(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

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
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
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

            _, abstract_policy_action, abstract_log_pi, _ = self.abstract_actor_target(next_z)

            # Compute the target Q value
            target_Q1, target_Q2 = self.abstract_critic_target(next_z, abstract_policy_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha.detach() * abstract_log_pi
            target_Q = reward + (discount * target_Q)

        # Get current Q estimates
        Q1, Q2 = self.abstract_critic(z, abstract_action)

        # Compute critic loss
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # Optimize the critic
        self.abstract_critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.abstract_critic_optimizer.step()

        # Compute the value equivalence between the two critics
        with torch.no_grad():
            Q1, Q2 = self.critic(obs, action)
            Q = torch.min(Q1, Q2)
            abstract_Q1, abstract_Q2 = self.abstract_critic(z, abstract_action)
            abstract_Q = torch.min(abstract_Q1, abstract_Q2)
            value_equivalence = torch.abs(Q - abstract_Q).mean()

        metrics['abs_critic_target_q'] = target_Q.mean().item()
        metrics['abs_critic_q1'] = Q1.mean().item()
        metrics['abs_critic_q2'] = Q2.mean().item()
        metrics['abs_critic_loss'] = critic_loss.item()
        metrics['value_equivalence'] = value_equivalence.item()

        return metrics

    def update_actor_and_alpha(self, obs, step):
        metrics = dict()

        # Compute actor loss
        _, pi, log_pi, log_std = self.actor(obs)
        Q1, Q2 = self.critic(obs, pi)
        Q = torch.min(Q1, Q2)

        actor_loss = (self.alpha.detach() * log_pi - Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        # Optimize the actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # optimize alpha
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.log_alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.log_alpha_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_target_ent'] = self.target_entropy.item()
        metrics['actor_ent'] = entropy.mean().item()
        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha_value'] = self.alpha.item()

        return metrics

    def update_abstract_actor(self, obs, step, lift_towards_actor=True):
        metrics = dict()

        batch_size = obs.shape[0]

        # Compute abstract actor loss
        with torch.no_grad():
            z = self.state_encoder(obs)

        _, abstract_pi, abstract_log_pi, _ = self.abstract_actor(z)
        Q1, Q2 = self.abstract_critic(z, abstract_pi)
        Q = torch.min(Q1, Q2)

        abstract_actor_loss = (self.alpha.detach() * abstract_log_pi - Q).mean()
        metrics['abs_actor_loss'] = abstract_actor_loss.item()

        if lift_towards_actor:
            # Computer policy lifting loss (moves abstract actor towards actual actor)
            # Repeat obs in a new dimension to create the samples needed for estimation
            obs_repeated = obs.unsqueeze(1).repeat(1, self.lifting_repeat_obs, 1)
            obs_repeated = obs_repeated.view(int(batch_size * self.lifting_repeat_obs), -1)
            with torch.no_grad():
                z_repeated = self.state_encoder(obs_repeated)
                _, pi, _, _ = self.actor_target(obs_repeated)
                pi_transformed = self.action_encoder(obs_repeated, pi)

            # Compute Kantorovich distance between the two policies
            # policy_lifting_loss = torch.sqrt(
            #     (mu_transformed - abstract_mu).pow(2) +
            #     (torch.exp(log_std) - torch.exp(abstract_log_std)).pow(2)
            # ).mean()

            _, abstract_pi, _, _ = self.abstract_actor(z_repeated)

            pi_transformed = pi_transformed.view(batch_size, self.lifting_repeat_obs, -1)
            abstract_pi = abstract_pi.view(batch_size, self.lifting_repeat_obs, -1)
            policy_lifting_loss = F.mse_loss(pi_transformed.mean(dim=1), abstract_pi.mean(dim=1))
            policy_lifting_loss += F.mse_loss(pi_transformed.std(dim=1), abstract_pi.std(dim=1))
            metrics['lifting_loss_abs_to_act'] = policy_lifting_loss.item()

            loss = abstract_actor_loss + self.lifting_weight * policy_lifting_loss
        else:
            loss = abstract_actor_loss

        # Optimize the abstract actor
        self.abstract_actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.abstract_actor_optimizer.step()

        return metrics

    def update_actor_by_lifting(self, obs, step):
        metrics = dict()

        batch_size = obs.shape[0]

        # Computer policy lifting loss (moves actual actor towards abstract actor)
        # Repeat obs in a new dimension to create the samples needed for estimation
        obs_repeated = obs.unsqueeze(1).repeat(1, self.lifting_repeat_obs, 1)
        obs_repeated = obs_repeated.view(int(batch_size * self.lifting_repeat_obs), -1)

        _, pi, _, _ = self.actor(obs_repeated)
        pi_transformed = self.action_encoder(obs_repeated, pi)
        with torch.no_grad():
            z_repeated = self.state_encoder(obs_repeated)
            _, abstract_pi, _, _ = self.abstract_actor_target(z_repeated)

        pi_transformed = pi_transformed.view(batch_size, self.lifting_repeat_obs, -1)
        abstract_pi = abstract_pi.view(batch_size, self.lifting_repeat_obs, -1)
        policy_lifting_loss = F.mse_loss(pi_transformed.mean(dim=1), abstract_pi.mean(dim=1))
        policy_lifting_loss += F.mse_loss(pi_transformed.std(dim=1), abstract_pi.std(dim=1))

        metrics['lifting_loss_act_to_abs'] = policy_lifting_loss.item()

        # Optimize the actual actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_lifting_loss.backward()
        self.actor_optimizer.step()

        return metrics

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
            if self.hpg_update_type == 'double':
                # update actual actor based on PG and abstract actor based on HPG and policy lifting loss
                metrics.update(self.update_actor_and_alpha(obs.detach(), step))
                metrics.update(self.update_abstract_actor(obs.detach(), step, lift_towards_actor=True))
            elif self.hpg_update_type == 'base':
                # update abstract actor based on HPG and update actual actor based on the policy lifting loss
                metrics.update(self.update_abstract_actor(obs.detach(), step, lift_towards_actor=False))
                metrics.update(self.update_actor_by_lifting(obs.detach(), step))
            else:
                raise NotImplementedError

        # update target networks
        if step % self.update_every_steps == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
            utils.soft_update_params(self.abstract_critic, self.abstract_critic_target, self.critic_target_tau)
            utils.soft_update_params(self.actor, self.actor_target, self.critic_target_tau)
            utils.soft_update_params(self.abstract_actor, self.abstract_actor_target, self.critic_target_tau)

        return metrics

    def save(self, model_dir, step):
        model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.actor.state_dict(), f'{model_save_dir}/actor.pt')
        torch.save(self.critic.state_dict(), f'{model_save_dir}/critic.pt')
        torch.save(self.abstract_actor.state_dict(), f'{model_save_dir}/abstract_actor.pt')
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
        self.abstract_actor.load_state_dict(
            torch.load(f'{model_load_dir}/abstract_actor.pt', map_location=self.device)
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
