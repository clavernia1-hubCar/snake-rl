"""Gymnasium-compatible Snake environment with 8-channel grid observation."""

import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from environment.snake_game import SnakeGame, Direction


# Channels: head, body_age, food, flood_fill, dir_up, dir_right, dir_down, dir_left
N_CHANNELS = 8


class SnakeEnv(gym.Env):
    """
    Snake environment following the Gymnasium API.

    Observation: float32 tensor of shape (8, grid_size, grid_size)
        Channel 0: snake head   — 1.0 at head cell
        Channel 1: body age     — 1.0 at head, decays linearly to 1/length at tail.
                                  Tells the agent when each segment will vacate.
        Channel 2: food         — 1.0 at food cell
        Channel 3: flood fill   — 1.0 at every cell reachable from the head via BFS
                                  (tail excluded from obstacles — it will move).
                                  Critical for trap-avoidance.
        Channels 4-7: direction one-hot (UP, RIGHT, DOWN, LEFT)
                       — entire channel filled with 1.0 if that direction is active

    Action: Discrete(4) — 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.
    Reward: shaped reward with BFS distance guidance and flood-fill trap penalty.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, grid_size: int = 10, render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self._seed = seed
        self._flood_cache: np.ndarray | None = None

        self.max_steps = grid_size * grid_size  # 100 steps per food — keeps episodes short

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_CHANNELS, grid_size, grid_size), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.game = SnakeGame(grid_size=grid_size, seed=seed)
        self._renderer = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.game = SnakeGame(grid_size=self.grid_size, seed=seed)
        else:
            self.game.reset()
        self._flood_cache = None
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        prev_score = self.game.score
        total_cells = self.grid_size * self.grid_size

        game_over, score, _ = self.game.step(action)
        self._flood_cache = None  # invalidate — board has changed

        ate_food = score > prev_score
        won = self.game.snake_length() >= total_cells
        timed_out = self.game.steps_since_food >= self.max_steps

        if won:
            reward = 500.0
            game_over = True
        elif ate_food:
            progress = self.game.snake_length() / total_cells
            reward = 10.0 + progress * 40.0  # +10 early, up to +50 near completion
        elif game_over:
            reward = -10.0
        elif timed_out:
            reward = -10.0
            game_over = True
        else:
            # Small step penalty — makes circling unprofitable,
            # forces the agent to eat food rather than hover near it.
            reward = -0.01

            # Flood fill trap penalty — punish moves that seal off space
            flood_map = self._get_flood()
            free_cells = total_cells - self.game.snake_length()
            if free_cells > 0:
                flood_ratio = float(flood_map.sum()) / free_cells
                if flood_ratio < 0.5:
                    # Proportional penalty: 0 at ratio=0.5, up to -1.0 at ratio=0
                    reward -= (0.5 - flood_ratio) * 2.0

        terminated = game_over
        truncated = False

        obs = self._get_obs()
        won_flag = won if not game_over else (self.game.snake_length() >= total_cells)
        info = {"score": self.game.score, "length": self.game.snake_length(), "won": won_flag}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            from environment.renderer import Renderer
            self._renderer = Renderer(grid_size=self.grid_size, cell_size=30, caption="Snake RL — Training")
            self._renderer.init()
        self._renderer.draw(self.game, fps=self.metadata["render_fps"])

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Grid observation (8 channels)
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((N_CHANNELS, self.grid_size, self.grid_size), dtype=np.float32)

        hr, hc = self.game.get_head()
        fr, fc = self.game.get_food()

        # Channel 0: head
        obs[0, hr, hc] = 1.0

        # Channel 1: body age — 1.0 at head, linear decay to 1/length at tail
        n = self.game.snake_length()
        for i, (r, c) in enumerate(self.game.body):
            obs[1, r, c] = (n - i) / n

        # Channel 2: food
        obs[2, fr, fc] = 1.0

        # Channel 3: flood fill accessibility from head
        obs[3] = self._get_flood()

        # Channels 4-7: direction one-hot broadcast across full grid
        obs[4 + int(self.game.direction), :, :] = 1.0

        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_flood(self) -> np.ndarray:
        """Return (or compute and cache) the flood fill map."""
        if self._flood_cache is None:
            self._flood_cache = self._flood_fill()
        return self._flood_cache

    def _flood_fill(self) -> np.ndarray:
        """
        BFS from head through non-body cells.
        Tail is excluded from obstacles because it will vacate next step.
        Returns binary (0/1) accessibility map.
        """
        head = self.game.get_head()
        accessible = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        body_list = list(self.game.body)
        body_set = set(body_list[:-1])  # all segments except tail

        visited: set = {head}
        queue: deque = deque([head])

        while queue:
            r, c = queue.popleft()
            accessible[r, c] = 1.0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                pos = (nr, nc)
                if (0 <= nr < self.grid_size
                        and 0 <= nc < self.grid_size
                        and pos not in visited
                        and pos not in body_set):
                    visited.add(pos)
                    queue.append(pos)

        return accessible

    def _bfs_to_food(self) -> int:
        """
        BFS shortest path from head to food through non-body cells.
        Returns -1 if food is unreachable (snake is trapped).
        """
        head = self.game.get_head()
        food = self.game.get_food()

        if head == food:
            return 0

        body_list = list(self.game.body)
        body_set = set(body_list[:-1])  # tail excluded

        visited: set = {head}
        queue: deque = deque([(head, 0)])

        while queue:
            (r, c), dist = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                pos = (nr, nc)
                if pos == food:
                    return dist + 1
                if (0 <= nr < self.grid_size
                        and 0 <= nc < self.grid_size
                        and pos not in visited
                        and pos not in body_set):
                    visited.add(pos)
                    queue.append((pos, dist + 1))

        return -1  # unreachable
