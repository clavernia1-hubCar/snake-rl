"""Dueling CNN DQN network architecture."""

import torch
import torch.nn as nn


class DuelingCNN(nn.Module):
    """
    Dueling DQN with CNN backbone for grid-based observations.

    Input:  (batch, n_channels, grid_size, grid_size)
    Output: (batch, action_dim) Q-values

    Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))

    Receptive field: kernel=5 on first layer, kernel=3 on remaining three.
    RF = 5 + 2 + 2 + 2 = 11, covering a 10x10 grid fully.
    """

    def __init__(self, grid_size: int = 10, n_channels: int = 7, action_dim: int = 4, hidden: int = 512):
        super().__init__()

        # CNN backbone
        # Layer 1: kernel=5 gives RF=5; with padding=2 spatial dims are preserved
        # Layers 2-4: kernel=3, padding=1 each adds 2 to RF → total RF = 5+2+2+2 = 11
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        flat_dim = 64 * grid_size * grid_size

        # Shared FC after CNN
        self.trunk = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(),
        )

        # Value stream: V(s) → scalar
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Advantage stream: A(s,a) → action_dim values
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, H, W)
        features = self.cnn(x)                          # (batch, 64, H, W)
        features = features.flatten(start_dim=1)        # (batch, 64*H*W)
        shared = self.trunk(features)                   # (batch, hidden)
        value = self.value_stream(shared)               # (batch, 1)
        advantage = self.advantage_stream(shared)       # (batch, action_dim)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
