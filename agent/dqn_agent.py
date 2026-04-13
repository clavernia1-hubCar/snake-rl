"""Dueling Double DQN agent with CNN and n-step returns."""

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.network import DuelingCNN
from agent.replay_buffer import PrioritizedReplayBuffer


class DQNAgent:
    """
    Dueling Double DQN agent with Prioritized Experience Replay and n-step returns.

    Double DQN: online network selects the best next action,
                target network evaluates its value.
    N-step returns: credit for rewards propagates back n steps.
    """

    def __init__(
        self,
        grid_size: int = 10,
        n_channels: int = 7,
        action_dim: int = 4,
        lr: float = 0.0003,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 64,
        train_frequency: int = 4,
        warmup_steps: int = 2000,
        buffer_capacity: int = 100_000,
        buffer_alpha: float = 0.6,
        buffer_beta_start: float = 0.4,
        n_step: int = 5,
        hidden_dim: int = 512,
        device: str | None = None,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.train_frequency = train_frequency
        self.warmup_steps = warmup_steps
        self.n_step = n_step

        if device:
            selected = device
        elif torch.cuda.is_available():
            selected = "cuda"
        elif torch.backends.mps.is_available():
            selected = "mps"
        else:
            selected = "cpu"
        self.device = torch.device(selected)

        # Networks (n_channels=7: head, body, food, dir×4 one-hot)
        self.online_net = DuelingCNN(grid_size, n_channels, action_dim, hidden_dim).to(self.device)
        self.target_net = DuelingCNN(grid_size, n_channels, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # Huber loss, per-sample

        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=buffer_alpha,
            beta_start=buffer_beta_start,
        )

        # N-step buffer: stores raw transitions before computing n-step return
        self._n_step_buf: deque = deque()

        self._step_count = 0

        # Gamma^n for n-step target bootstrap
        self._gamma_n = gamma ** n_step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection with action masking. obs: (n_channels, H, W)."""
        valid = self._valid_mask(obs)
        if np.random.random() < epsilon:
            valid_indices = np.where(valid)[0]
            return int(np.random.choice(valid_indices))
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(obs_t).cpu().numpy()[0]
        q[~valid] = -np.inf
        return int(q.argmax())

    def act_batch(self, obs_batch: np.ndarray, epsilon: float) -> np.ndarray:
        """Vectorized epsilon-greedy with action masking. obs_batch: (N, n_channels, H, W)."""
        obs_t = torch.from_numpy(obs_batch).float().to(self.device)
        with torch.no_grad():
            q_batch = self.online_net(obs_t).cpu().numpy()  # (N, 4)

        valid_batch = self._valid_mask_batch(obs_batch)  # (N, 4) bool

        # Greedy: mask invalid Q-values, take argmax
        q_masked = q_batch.copy()
        q_masked[~valid_batch] = -np.inf
        greedy = q_masked.argmax(axis=1)

        # Random: sample uniformly from valid actions per env
        N = len(obs_batch)
        random_actions = np.zeros(N, dtype=np.int64)
        for i in range(N):
            valid_idx = np.where(valid_batch[i])[0]
            random_actions[i] = np.random.choice(valid_idx)

        explore = np.random.random(N) < epsilon
        return np.where(explore, random_actions, greedy).astype(np.int64)

    def _valid_mask_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Vectorized valid-action masks for a batch. obs_batch: (N, C, H, W) → (N, 4) bool."""
        N, _, H, _ = obs_batch.shape
        head_flat = obs_batch[:, 0, :, :].reshape(N, -1).argmax(axis=1)  # (N,)
        hr = head_flat // H
        hc = head_flat % H
        flood = obs_batch[:, 3, :, :]  # (N, H, W)

        # Neighbor coords for UP=0, RIGHT=1, DOWN=2, LEFT=3
        dr = np.array([-1, 0, 1,  0])
        dc = np.array([ 0, 1, 0, -1])
        nr = hr[:, None] + dr[None, :]  # (N, 4)
        nc = hc[:, None] + dc[None, :]  # (N, 4)

        in_bounds = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < H)
        nr_clip = np.clip(nr, 0, H - 1)
        nc_clip = np.clip(nc, 0, H - 1)

        # Gather flood fill values at neighbor positions
        n_idx = np.arange(N)[:, None]
        flood_vals = flood[n_idx, nr_clip, nc_clip]  # (N, 4)

        valid = in_bounds & (flood_vals > 0)
        # Fallback: if fully enclosed, allow all (unavoidable death)
        valid[~valid.any(axis=1)] = True
        return valid

    def _valid_mask(self, obs: np.ndarray) -> np.ndarray:
        """
        Derive valid action mask from the observation.
        Channel 0 = head, Channel 3 = flood fill (1.0 where reachable from head).
        A move is valid iff the destination cell is in the flood fill.
        Directions: UP=0, RIGHT=1, DOWN=2, LEFT=3.
        """
        H = obs.shape[1]
        hr, hc = np.unravel_index(obs[0].argmax(), (H, H))
        flood = obs[3]

        neighbors = [(hr - 1, hc), (hr, hc + 1), (hr + 1, hc), (hr, hc - 1)]
        valid = np.array(
            [0 <= nr < H and 0 <= nc < H and flood[nr, nc] > 0
             for nr, nc in neighbors],
            dtype=bool,
        )
        # If fully enclosed, allow all actions (unavoidable death — let Q-values decide)
        if not valid.any():
            valid[:] = True
        return valid

    def store(self, state, action: int, reward: float, next_state, done: bool,
              n_step_buf: deque | None = None) -> None:
        """Buffer a transition, computing n-step returns before adding to PER.

        n_step_buf: external per-env deque for vectorized training. Uses the
                    agent's internal buffer when None (single-env mode).
        """
        buf = n_step_buf if n_step_buf is not None else self._n_step_buf
        buf.append((state, action, reward, next_state, done))
        self._step_count += 1

        if len(buf) >= self.n_step:
            self._commit_front(buf)

        if done:
            while buf:
                self._commit_front(buf)

    def train_step(self) -> float | None:
        """Sample a batch and perform one gradient update."""
        if self._step_count < self.warmup_steps:
            return None
        if not self.buffer.ready(self.batch_size):
            return None
        if self._step_count % self.train_frequency != 0:
            return None

        states, actions, rewards, next_states, dones, weights, leaf_indices = \
            self.buffer.sample(self.batch_size)

        states_t      = torch.from_numpy(states).to(self.device)
        actions_t     = torch.from_numpy(actions).to(self.device)
        rewards_t     = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t       = torch.from_numpy(dones).to(self.device)
        weights_t     = torch.from_numpy(weights).to(self.device)

        # Current Q values
        q_current = self.online_net(states_t)
        q_current = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: online net picks action, target net evaluates
        # Use gamma^n for n-step bootstrap
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            q_target = rewards_t + self._gamma_n * q_next * (~dones_t)

        # Per-sample Huber loss weighted by IS weights
        td_errors = q_target - q_current
        loss_per_sample = self.loss_fn(q_current, q_target)
        loss = (loss_per_sample * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities in the buffer
        self.buffer.update_priorities(leaf_indices, td_errors.detach().cpu().numpy())

        # Soft update target network
        self._soft_update_target()

        return float(loss.item())

    def anneal_beta(self, fraction: float) -> None:
        self.buffer.anneal_beta(fraction)

    def save(self, path: str, episode: int = 0) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "step_count": self._step_count,
            "episode":    episode,
        }, path)

    def load(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._step_count = checkpoint.get("step_count", 0)
        self.target_net.eval()
        return checkpoint.get("episode", 0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _commit_front(self, buf: deque | None = None) -> None:
        """Compute n-step return for the oldest buffered transition and add to PER."""
        if buf is None:
            buf = self._n_step_buf
        if not buf:
            return
        obs_0, action_0, _, _, _ = buf[0]
        R = 0.0
        final_next_obs = None
        final_done = False
        for i, (_, _, r, n_obs, d) in enumerate(buf):
            R += (self.gamma ** i) * r
            final_next_obs = n_obs
            final_done = d
            if d:
                break  # episode ended; don't look past terminal
        self.buffer.add(obs_0, action_0, R, final_next_obs, final_done)
        buf.popleft()

    def _soft_update_target(self) -> None:
        """Polyak averaging: target ← tau*online + (1-tau)*target."""
        for target_p, online_p in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)
