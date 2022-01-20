import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils as utils


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


class ActionEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, abstract_action_dim, hidden_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, abstract_action_dim),
        )

        self.apply(utils.weight_init)

    def forward(self, state, action):
        # State-dependent action-embedding
        sa = torch.cat([state, action], dim=1)
        abstract_action = self.encoder(sa)
        return torch.tanh(abstract_action)


class StateEncoder(nn.Module):
    def __init__(self, state_dim, abstract_state_dim, hidden_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, abstract_state_dim)
        )

        self.apply(utils.weight_init)

    def forward(self, state):
        abstract_state = self.encoder(state)
        return abstract_state


class RewardPredictor(nn.Module):
    def __init__(self, abstract_state_dim, hidden_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(abstract_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(utils.weight_init)

    def forward(self, abstract_state):
        return self.fc(abstract_state)


class DeterministicActor(nn.Module):
    """Original TD3 and DDPG actor."""
    def __init__(self, feature_dim, action_dim, hidden_dim, linear_approx):
        super(DeterministicActor, self).__init__()

        if linear_approx:
            self.policy = nn.Linear(feature_dim, action_dim)
        else:
            self.policy = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, action_dim)
            )

        self.apply(utils.weight_init)

    def forward(self, state):
        a = self.policy(state)
        return torch.tanh(a)


class DrQActor(nn.Module):
    """TD3 actor used in DRQ-v2."""
    def __init__(self, feature_dim, action_dim, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))

        self.apply(utils.weight_init)

    def forward(self, state, std):
        mu = self.policy(state)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class StochasticActor(nn.Module):
    """SAC actor used in SAC-AE and DBC."""
    def __init__(self, feature_dim, action_dim, hidden_dim,
                 linear_approx, log_std_min, log_std_max):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        if linear_approx:
            self.policy = nn.Linear(feature_dim, 2 * action_dim)
        else:
            self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, 2 * action_dim))

        self.apply(utils.weight_init)

    def forward(self, state, compute_pi=True, compute_log_pi=True):
        mu, log_std = self.policy(state).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class Critic(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim, linear_approx):
        super(Critic, self).__init__()

        if linear_approx:
            self.Q1_net = nn.Linear(feature_dim + action_dim, 1)
            self.Q2_net = nn.Linear(feature_dim + action_dim, 1)
        else:
            # Q1 architecture
            self.Q1_net = nn.Sequential(
                nn.Linear(feature_dim + action_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1)
            )

            # Q2 architecture
            self.Q2_net = nn.Sequential(
                nn.Linear(feature_dim + action_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1)
            )

        self.apply(utils.weight_init)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.Q1_net(sa)
        q2 = self.Q2_net(sa)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = self.Q1_net(sa)
        return q1


class DDPGCritic(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim, linear_approx):
        super(DDPGCritic, self).__init__()

        if linear_approx:
            self.Q = nn.Linear(feature_dim + action_dim, 1)

        else:
            self.Q = nn.Sequential(
                nn.Linear(feature_dim + action_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1)
            )

        self.apply(utils.weight_init)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = self.Q(sa)
        return q
