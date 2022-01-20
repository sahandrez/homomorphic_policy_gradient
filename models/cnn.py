import torch
import torch.nn as nn

import utils.utils as utils


class PixelEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU()
        )
        if feature_dim <= 10:
            self.trunk = nn.Sequential(
                nn.Linear(self.repr_dim, 50),
                nn.LayerNorm(50),
                nn.Linear(50, feature_dim),
                nn.Tanh()
            )
        else:
            self.trunk = nn.Sequential(
                nn.Linear(self.repr_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.Tanh()
            )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.trunk(h)
        return h


class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.deconvnet = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=1),
            nn.ReLU(), nn.ConvTranspose2d(32, 32, 3, stride=1),
            nn.ReLU(), nn.ConvTranspose2d(32, 32, 3, stride=1),
            nn.ReLU(), nn.ConvTranspose2d(32, obs_shape[0], 3, stride=2, output_padding=1)
        )

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, self.repr_dim),
            nn.ReLU()
        )

        self.apply(utils.weight_init)

    def forward(self, z):
        h = self.trunk(z)
        deconv = h.view(-1, 32, 35, 35)
        obs = self.deconvnet(deconv)
        return obs
