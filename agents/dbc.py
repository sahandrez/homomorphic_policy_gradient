"""
Implementation of DBC
https://arxiv.org/abs/2006.10742
"""

import hydra
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn import PixelEncoder
from models.core import StochasticActor, Critic
from models.core import RewardPredictor
from models.transition_model import make_transition_model
import utils.utils as utils


class DBCAgent:
    def __init__(self, obs_shape, action_shape, device, lr, transition_model_type,
                 beta, feature_dim, hidden_dim, linear_approx, init_temperature,
                 alpha_lr, alpha_beta, actor_log_std_min, actor_log_std_max,
                 actor_update_freq, critic_target_tau, critic_target_update_freq,
                 encoder_tau, weight_lambda, num_expl_steps, bisim_coef):
        self.device = device
        self.action_dim = action_shape[0]
        self.num_expl_steps = num_expl_steps
        self.critic_target_update_freq = critic_target_update_freq
        self.critic_target_tau = critic_target_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.transition_model_type = transition_model_type
        self.bisim_coef = bisim_coef
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.actor_log_std_min = actor_log_std_min
        self.actor_log_std_max = actor_log_std_max
        self.lr = lr
        self.beta = beta
        self.alpha_lr = alpha_lr
        self.alpha_beta = alpha_beta
        self.init_temperature = init_temperature
        self.weight_lambda = weight_lambda

        # models
        self.pixel_encoder = PixelEncoder(obs_shape, feature_dim).to(device)
        self.actor = StochasticActor(feature_dim, action_shape[0], hidden_dim, linear_approx,
                                     actor_log_std_min, actor_log_std_max).to(device)

        self.critic = Critic(feature_dim, action_shape[0], hidden_dim, linear_approx).to(device)
        self.critic_target = Critic(feature_dim, action_shape[0], hidden_dim, linear_approx).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.transition_model = make_transition_model(
            transition_model_type, feature_dim, action_shape[0], hidden_dim
        ).to(device)
        self.reward_predictor = RewardPredictor(feature_dim, hidden_dim).to(device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.pixel_encoder_opt = torch.optim.Adam(self.pixel_encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr,betas=(beta, 0.999))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(beta, 0.999))
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999))
        self.transition_reward_opt = torch.optim.Adam(
            list(self.reward_predictor.parameters()) + list(self.transition_model.parameters()),
            lr=lr, weight_decay=weight_lambda,
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.pixel_encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.reward_predictor.train(training)
        self.transition_model.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.pixel_encoder(obs.unsqueeze(0))

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

        obs = self.pixel_encoder(obs)
        q, _ = self.critic(obs, action)

        return {
            'state': obs.cpu().numpy()[0],
            'value': q.cpu().numpy()[0]
        }

    def get_bisim_loss(self, obs, action, reward, discount):
        # Sample random states across episodes at random
        batch_size = obs.size(0)
        perm = np.random.permutation(batch_size)
        obs2 = obs[perm]

        with torch.no_grad():
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(obs, action)
            reward2 = reward[perm]
        if pred_next_latent_sigma1 is None:
            pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
        if pred_next_latent_mu1.ndim == 2:      # shape (B, Z), no ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
        elif pred_next_latent_mu1.ndim == 3:    # shape (B, E, Z), using an ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
        else:
            raise NotImplementedError

        z_dist = F.smooth_l1_loss(obs, obs2, reduction='none')
        r_dist = F.smooth_l1_loss(reward, reward2, reduction='none')
        if self.transition_model_type == '':
            transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none')
        else:
            transition_dist = torch.sqrt(
                (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
                (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
            )

        bisimilarity = r_dist + discount * transition_dist
        bisim_loss = (z_dist - bisimilarity).pow(2).mean()
        return bisim_loss

    def get_transition_reward_loss(self, obs, action, reward, next_obs):
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(obs, action)
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        diff = (pred_next_latent_mu - next_obs.detach()) / pred_next_latent_sigma
        transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        pred_next_latent = self.transition_model.sample_prediction(obs, action)
        pred_next_reward = self.reward_predictor(pred_next_latent)
        reward_loss = F.mse_loss(pred_next_reward, reward)
        return transition_loss, reward_loss

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        # compute critic loss
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)

            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # compute bisim loss
        bisim_loss = self.get_bisim_loss(obs, action, reward, discount)

        # compute transition and reward loss
        transition_loss, reward_loss = self.get_transition_reward_loss(obs, action, reward, next_obs)

        loss = critic_loss + self.bisim_coef * bisim_loss + transition_loss + reward_loss

        # optimize encoder and critic
        self.pixel_encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        self.transition_reward_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.transition_reward_opt.zero_grad()
        self.critic_opt.step()
        self.pixel_encoder_opt.step()

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()
        metrics['transition_loss'] = transition_loss.item()
        metrics['reward_loss'] = reward_loss.item()
        metrics['bisim_loss'] = bisim_loss.item()

        return metrics

    def update_actor_and_alpha(self, obs, step):
        metrics = dict()

        _, pi, log_pi, log_std = self.actor(obs)
        Q1, Q2 = self.critic(obs, pi)
        Q = torch.min(Q1, Q2)

        actor_loss = (self.alpha.detach() * log_pi - Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

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

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _ = utils.to_torch(
            batch, self.device)

        # encode
        obs = self.pixel_encoder(obs)
        with torch.no_grad():
            next_obs = self.pixel_encoder(next_obs)

        metrics['batch_reward'] = reward.mean().item()

        # update critic, encoder, reward and transition model
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        if step % self.actor_update_freq == 0:
            metrics.update(self.update_actor_and_alpha(obs.detach(), step))

        # update critic target
        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)

        return metrics

    def save(self, model_dir, step):
        model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.actor.state_dict(), f'{model_save_dir}/actor.pt')
        torch.save(self.critic.state_dict(), f'{model_save_dir}/critic.pt')
        torch.save(self.pixel_encoder.state_dict(), f'{model_save_dir}/pixel_encoder.pt')
        torch.save(self.transition_model.state_dict(), f'{model_save_dir}/transition_model.pt')
        torch.save(self.reward_predictor.state_dict(), f'{model_save_dir}/reward_predictor.pt')

    def load(self, model_dir, step):
        print(f"Loading the model from {model_dir}, step: {step}")
        model_load_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')

        self.actor.load_state_dict(
            torch.load(f'{model_load_dir}/actor.pt', map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f'{model_load_dir}/critic.pt', map_location=self.device)
        )
        self.pixel_encoder.load_state_dict(
            torch.load(f'{model_load_dir}/pixel_encoder.pt', map_location=self.device)
        )
        self.transition_model.load_state_dict(
            torch.load(f'{model_load_dir}/transition_model.pt', map_location=self.device)
        )
        self.reward_predictor.load_state_dict(
            torch.load(f'{model_load_dir}/reward_predictor.pt', map_location=self.device)
        )
