"""Actor-Critic CNN for PPO — grid-size agnostic via AdaptiveAvgPool."""

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """
    Shared CNN backbone with separate actor and critic heads.

    AdaptiveAvgPool2d makes the backbone accept any input grid size, so the
    same network weights can be used across curriculum stages (6×6 → 10×10).

    Architecture mirrors the DuelingCNN receptive field (RF=11) but outputs
    both a policy (logits) and a value estimate.
    """

    def __init__(
        self,
        n_channels: int = 8,
        action_dim: int = 4,
        hidden_dim: int = 512,
    ):
        super().__init__()

        # CNN backbone — RF grows: 5 → 7 → 9 → 11
        # Expects fixed 10×10 input (curriculum stages are padded to 10×10 by the trainer).
        # Spatial dims preserved throughout (padding=same), flattened before FC.
        self.backbone = nn.Sequential(
            nn.Conv2d(n_channels, 32,  kernel_size=5, padding=2),  # (N, 32,  10, 10)
            nn.ReLU(inplace=True),
            nn.Conv2d(32,  64,  kernel_size=3, padding=1),          # (N, 64,  10, 10)
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, kernel_size=3, padding=1),          # (N, 128, 10, 10)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),          # (N, 128, 10, 10)
            nn.ReLU(inplace=True),
            nn.Flatten(),                                           # (N, 12800)
            nn.Linear(128 * 10 * 10, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.actor_head  = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small actor output → near-uniform initial policy
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        # Unit gain for critic
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.zeros_(self.critic_head.bias)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        obs:          (N, C, H, W)
        action_masks: (N, A) bool — True = valid action
        Returns:      logits (N, A), values (N,)
        """
        features = self.backbone(obs)
        logits   = self.actor_head(features)
        values   = self.critic_head(features).squeeze(-1)

        if action_masks is not None:
            logits = logits.masked_fill(~action_masks, -1e8)

        return logits, values

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_masks: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs, action_masks)
        dist    = torch.distributions.Categorical(logits=logits)
        actions = dist.mode if deterministic else dist.sample()
        return actions, dist.log_prob(actions), dist.entropy(), values

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs, action_masks)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values
