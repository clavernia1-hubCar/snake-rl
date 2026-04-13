"""Greedy evaluation: runs episodes with epsilon=0 and returns stats."""

import numpy as np
from environment.snake_env import SnakeEnv
from agent.dqn_agent import DQNAgent


def evaluate(agent: DQNAgent, grid_size: int = 20, n_episodes: int = 10) -> dict:
    """
    Run `n_episodes` with greedy policy (epsilon=0).

    Returns:
        dict with keys: mean_score, max_score, mean_length, scores
    """
    env = SnakeEnv(grid_size=grid_size)
    scores = []
    lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        scores.append(info["score"])
        lengths.append(info["length"])

    env.close()
    return {
        "mean_score":  float(np.mean(scores)),
        "max_score":   float(np.max(scores)),
        "mean_length": float(np.mean(lengths)),
        "scores":      scores,
    }
