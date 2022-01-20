import random
import torch
import torch.nn as nn

import utils.utils as utils


class DeterministicTransitionModel(nn.Module):

    def __init__(self, z_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)

        self.apply(utils.weight_init)
        print("Deterministic transition model chosen.")

    def forward(self, abstract_state, abstract_action):
        za = torch.cat([abstract_state, abstract_action], dim=1)
        za = self.fc(za)
        mu = self.fc_mu(za)
        sigma = None
        return mu, sigma

    def sample_prediction(self, abstract_state, abstract_action):
        mu, sigma = self.forward(abstract_state, abstract_action)
        return mu


class ProbabilisticTransitionModel(nn.Module):

    def __init__(self, abstract_state_dim, abstract_action_dim, hidden_dim,
                 announce=True, max_sigma=1e1, min_sigma=1e-4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(abstract_state_dim + abstract_action_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, abstract_state_dim)
        self.fc_sigma = nn.Linear(hidden_dim, abstract_state_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert(self.max_sigma >= self.min_sigma)

        self.apply(utils.weight_init)

        if announce:
            print("Probabilistic transition model chosen.")

    def forward(self, abstract_state, abstract_action):
        za = torch.cat([abstract_state, abstract_action], dim=1)
        za = self.fc(za)
        mu = self.fc_mu(za)
        sigma = torch.sigmoid(self.fc_sigma(za))                                # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma      # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, abstract_state, abstract_action):
        mu, sigma = self.forward(abstract_state, abstract_action)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps


class EnsembleOfProbabilisticTransitionModels(nn.Module):

    def __init__(self, abstract_state_dim, abstract_action_dim, hidden_dim, ensemble_size=5):
        super().__init__()
        self.models = nn.ModuleList([ProbabilisticTransitionModel(abstract_state_dim, abstract_action_dim,
                                                                  hidden_dim, announce=False)
                                     for _ in range(ensemble_size)])
        print("Ensemble of probabilistic transition models chosen.")

    def __call__(self, abstract_state, abstract_action):
        za = torch.cat([abstract_state, abstract_action], dim=1)
        mu_sigma_list = [model.forward(abstract_state, abstract_action) for model in self.models]
        mus, sigmas = zip(*mu_sigma_list)
        mus, sigmas = torch.stack(mus), torch.stack(sigmas)
        return mus, sigmas

    def sample_prediction(self, abstract_state, abstract_action):
        model = random.choice(self.models)
        return model.sample_prediction(abstract_state, abstract_action)


_AVAILABLE_TRANSITION_MODELS = {'': DeterministicTransitionModel,
                                'deterministic': DeterministicTransitionModel,
                                'probabilistic': ProbabilisticTransitionModel,
                                'ensemble': EnsembleOfProbabilisticTransitionModels}


def make_transition_model(transition_model_type, encoder_feature_dim, action_dim, hidden_dim):
    assert transition_model_type in _AVAILABLE_TRANSITION_MODELS
    return _AVAILABLE_TRANSITION_MODELS[transition_model_type](
        encoder_feature_dim, action_dim, hidden_dim
    )
