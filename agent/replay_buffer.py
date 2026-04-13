"""Prioritized Experience Replay buffer using a sum-tree."""

import numpy as np


class SumTree:
    """
    Binary sum-tree for O(log n) priority updates and sampling.

    Leaves store priorities; internal nodes store sums.
    Data is stored in a separate circular array.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        # Leaf nodes are at indices capacity-1 to 2*capacity-2
        if idx >= self.capacity - 1:
            return idx
        left  = 2 * idx + 1
        right = left + 1
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        leaf_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(leaf_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        """Sample by value s in [0, total]. Returns (leaf_idx, priority, data)."""
        leaf_idx = self._retrieve(0, s)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, float(self.tree[leaf_idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Transitions with higher TD error are sampled more frequently.
    Importance sampling weights correct for the sampling bias.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        epsilon: float = 1e-6,
    ):
        self.tree = SumTree(capacity)
        self.alpha = alpha          # priority exponent (0=uniform, 1=full PER)
        self.beta = beta_start
        self.beta_end = beta_end
        self.epsilon = epsilon      # small constant to avoid zero priority
        self._max_priority = 1.0

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        transition = (state, action, reward, next_state, done)
        priority = self._max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a batch of transitions.

        Returns:
            (states, actions, rewards, next_states, dones, weights, leaf_indices)
        """
        batch = []
        leaf_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            leaf_idx, priority, data = self.tree.get(s)
            # Guard against None (unfilled slots)
            while data is None:
                s = np.random.uniform(0, self.tree.total)
                leaf_idx, priority, data = self.tree.get(s)
            batch.append(data)
            leaf_indices[i] = leaf_idx
            priorities[i] = priority

        # Importance sampling weights
        n = self.tree.n_entries
        probs = priorities / self.tree.total
        probs = np.maximum(probs, 1e-10)
        weights = (n * probs) ** (-self.beta)
        weights /= weights.max()  # normalize so max weight = 1

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=bool),
            np.array(weights,     dtype=np.float32),
            leaf_indices,
        )

    def update_priorities(self, leaf_indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(leaf_indices, priorities):
            self.tree.update(int(idx), float(priority))
        self._max_priority = max(self._max_priority, float(priorities.max()))

    def anneal_beta(self, fraction: float) -> None:
        """Anneal beta linearly. Call with fraction = current_step / total_steps."""
        self.beta = min(self.beta_end, self.beta + fraction * (self.beta_end - self.beta))

    def __len__(self) -> int:
        return self.tree.n_entries

    def ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size
