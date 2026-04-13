"""Main training loop for the Snake DQN agent — vectorized multi-env edition."""

import os
import csv
from collections import deque

import numpy as np
from tqdm import tqdm
from gymnasium.vector import SyncVectorEnv

from environment.snake_env import SnakeEnv, N_CHANNELS
from agent.dqn_agent import DQNAgent
from training.evaluator import evaluate


def linear_epsilon(episode: int, start: float, end: float, decay_episodes: int) -> float:
    if episode >= decay_episodes:
        return end
    return start - (start - end) * (episode / decay_episodes)


def _make_env_fn(grid_size: int):
    def _make():
        return SnakeEnv(grid_size=grid_size)
    return _make


def train(cfg: dict, resume: str | None = None) -> DQNAgent:
    env_cfg   = cfg["environment"]
    agent_cfg = cfg["agent"]
    buf_cfg   = cfg["replay_buffer"]
    train_cfg = cfg["training"]

    grid_size = env_cfg["grid_size"]
    num_envs  = train_cfg.get("num_envs", 1)

    # Vectorized environments
    envs = SyncVectorEnv([_make_env_fn(grid_size) for _ in range(num_envs)])

    agent = DQNAgent(
        grid_size=grid_size,
        n_channels=N_CHANNELS,
        action_dim=4,
        lr=agent_cfg["lr"],
        gamma=agent_cfg["gamma"],
        tau=agent_cfg["tau"],
        batch_size=agent_cfg["batch_size"],
        train_frequency=agent_cfg["train_frequency"],
        warmup_steps=agent_cfg["warmup_steps"],
        buffer_capacity=buf_cfg["capacity"],
        buffer_alpha=buf_cfg["alpha"],
        buffer_beta_start=buf_cfg["beta_start"],
        n_step=agent_cfg.get("n_step", 5),
        hidden_dim=agent_cfg.get("hidden_dim", 512),
    )

    if resume:
        global_episode = agent.load(resume)
        print(f"Resumed from checkpoint: {resume} (episode {global_episode})")

    total_episodes  = train_cfg["total_episodes"]
    checkpoint_dir  = train_cfg["checkpoint_dir"]
    log_dir         = train_cfg["log_dir"]
    eval_frequency  = train_cfg["eval_frequency"]
    eval_episodes   = train_cfg["eval_episodes"]
    log_frequency   = train_cfg.get("log_frequency", 10)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path    = os.path.join(log_dir, "training_log.csv")
    append_mode = resume is not None and os.path.exists(log_path)
    csv_file    = open(log_path, "a" if append_mode else "w", newline="")
    csv_writer  = csv.writer(csv_file)
    if not append_mode:
        csv_writer.writerow(["episode", "score", "length", "epsilon", "loss",
                             "eval_mean_score", "eval_max_score"])

    # Per-env n-step buffers (agent.store uses these instead of its internal one)
    n_step_bufs = [deque() for _ in range(num_envs)]

    obs, _ = envs.reset()  # shape: (num_envs, 7, H, W)

    if not resume:
        global_episode = 0
    best_eval_score = -float("inf")
    episode_losses  = []

    print(f"Training with {num_envs} parallel environments.")
    pbar = tqdm(total=total_episodes, desc="Training", unit="ep")

    while global_episode < total_episodes:
        epsilon = linear_epsilon(
            global_episode,
            agent_cfg["epsilon_start"],
            agent_cfg["epsilon_end"],
            agent_cfg["epsilon_decay_episodes"],
        )

        # One batched forward pass for all envs
        actions = agent.act_batch(obs, epsilon)  # (num_envs,)

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        dones = terminated | truncated

        for i in range(num_envs):
            agent.store(
                obs[i], int(actions[i]), float(rewards[i]),
                next_obs[i], bool(dones[i]),
                n_step_bufs[i],
            )
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            if dones[i]:
                agent.anneal_beta(1.0 / total_episodes)

                # gymnasium VectorEnv puts terminal info in infos["final_info"]
                final_infos = infos.get("final_info", [None] * num_envs)
                ep_score  = 0
                ep_length = 1
                if final_infos is not None and final_infos[i] is not None:
                    ep_score  = final_infos[i].get("score",  0)
                    ep_length = final_infos[i].get("length", 1)

                global_episode += 1
                pbar.update(1)

                # Eval
                eval_mean = eval_max = ""
                if global_episode % eval_frequency == 0:
                    eval_result = evaluate(agent, grid_size=grid_size, n_episodes=eval_episodes)
                    eval_mean   = round(eval_result["mean_score"], 2)
                    eval_max    = int(eval_result["max_score"])
                    pbar.set_postfix({
                        "score": ep_score,
                        "eval":  eval_mean,
                        "eps":   f"{epsilon:.3f}",
                    })
                    if eval_result["mean_score"] > best_eval_score:
                        best_eval_score = eval_result["mean_score"]
                        agent.save(os.path.join(checkpoint_dir, "best.pt"), episode=global_episode)

                # CSV logging
                if global_episode % log_frequency == 0:
                    mean_loss = float(np.mean(episode_losses[-50:])) if episode_losses else 0.0
                    csv_writer.writerow([
                        global_episode, ep_score, ep_length,
                        round(epsilon, 4), round(mean_loss, 6),
                        eval_mean, eval_max,
                    ])
                    csv_file.flush()

                # Periodic checkpoint
                if global_episode % 500 == 0:
                    agent.save(os.path.join(checkpoint_dir, f"episode_{global_episode}.pt"), episode=global_episode)

                if global_episode >= total_episodes:
                    break

        obs = next_obs

    pbar.close()
    csv_file.close()
    envs.close()

    agent.save(os.path.join(checkpoint_dir, "final.pt"), episode=global_episode)
    print(f"\nTraining complete. Best eval score: {best_eval_score:.2f}")
    print(f"Models saved to: {checkpoint_dir}/")
    print(f"Logs saved to:   {log_path}")

    return agent
