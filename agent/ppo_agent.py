"""PPO agent with GAE advantage estimation and action masking."""

import os
import numpy as np
import torch
import torch.nn as nn

from agent.actor_critic import ActorCritic


class PPOAgent:
    """
    Proximal Policy Optimization agent.

    Key properties vs the DQN:
    - On-policy: always trains on fresh rollouts — no replay buffer drift
    - Entropy bonus: directly penalises deterministic (circling) policies
    - GAE: multi-step advantage estimation with variance-bias trade-off
    - Action masking: logit masking keeps the policy within safe actions
    """

    def __init__(
        self,
        n_channels: int = 8,
        action_dim: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.02,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        hidden_dim: int = 512,
        device: str | None = None,
    ):
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs     = n_epochs

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.network = ActorCritic(n_channels, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, eps=1e-5
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act_batch(
        self,
        obs_batch: np.ndarray,
        masks_batch: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect actions, log_probs, values for a batch of observations.
        obs_batch:   (N, C, H, W)
        masks_batch: (N, 4) bool
        Returns:     actions (N,), log_probs (N,), values (N,)
        """
        obs_t   = torch.from_numpy(obs_batch).float().to(self.device)
        masks_t = (
            torch.from_numpy(masks_batch).bool().to(self.device)
            if masks_batch is not None else None
        )
        actions, log_probs, _, values = self.network.get_action_and_value(obs_t, masks_t)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> int:
        """Single-env action selection."""
        obs_t   = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        masks_t = (
            torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
            if mask is not None else None
        )
        actions, _, _, _ = self.network.get_action_and_value(obs_t, masks_t, deterministic)
        return int(actions[0].cpu())

    # ------------------------------------------------------------------
    # Advantage estimation
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rewards:      np.ndarray,  # (T, N)
        values:       np.ndarray,  # (T, N)
        dones:        np.ndarray,  # (T, N) — 1.0 if step t ended an episode
        next_values:  np.ndarray,  # (N,)   — V(obs after last rollout step)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generalised Advantage Estimation (Schulman et al., 2015).

        δ_t  = r_t + γ * V(s_{t+1}) * (1 − done_t) − V(s_t)
        A_t  = δ_t + γλ * (1 − done_t) * A_{t+1}
        R_t  = A_t + V(s_t)

        done_t = 1 means the episode ended at step t; we do NOT bootstrap
        across that boundary in either the delta or the recursion.
        """
        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        last_gae   = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            nxt_val        = next_values if t == T - 1 else values[t + 1]
            nxt_nonterminal = 1.0 - dones[t]          # zero-out across episode boundary

            delta    = rewards[t] + self.gamma * nxt_val * nxt_nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nxt_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        states:        np.ndarray,  # (B, C, H, W)
        actions:       np.ndarray,  # (B,)
        old_log_probs: np.ndarray,  # (B,)
        advantages:    np.ndarray,  # (B,)
        returns:       np.ndarray,  # (B,)
        masks:         np.ndarray,  # (B, 4) bool
        minibatch_size: int = 512,
    ) -> dict:
        """
        PPO-Clip update: n_epochs passes of minibatch gradient steps.
        Advantage normalisation is done once before any epoch.
        """
        B = len(states)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totals   = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0)
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = np.random.permutation(B)

            for start in range(0, B, minibatch_size):
                idx = perm[start: start + minibatch_size]

                mb_obs  = torch.from_numpy(states[idx]).float().to(self.device)
                mb_act  = torch.from_numpy(actions[idx]).long().to(self.device)
                mb_olp  = torch.from_numpy(old_log_probs[idx]).float().to(self.device)
                mb_adv  = torch.from_numpy(advantages[idx]).float().to(self.device)
                mb_ret  = torch.from_numpy(returns[idx]).float().to(self.device)
                mb_mask = torch.from_numpy(masks[idx]).bool().to(self.device)

                log_probs, entropy, values = self.network.evaluate_actions(
                    mb_obs, mb_act, mb_mask
                )

                # PPO-Clip surrogate
                ratio      = torch.exp(log_probs - mb_olp)
                surrogate1 = ratio * mb_adv
                surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss  = nn.functional.mse_loss(values, mb_ret)
                ent_mean    = entropy.mean()
                loss        = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (log_probs - mb_olp)).mean().item()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"]  += value_loss.item()
                totals["entropy"]     += ent_mean.item()
                totals["approx_kl"]   += approx_kl
                n_updates += 1

        return {k: v / n_updates for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Learning rate schedule
    # ------------------------------------------------------------------

    def set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str, episode: int = 0) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "network":   self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode":   episode,
        }, path)

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt.get("episode", 0)
