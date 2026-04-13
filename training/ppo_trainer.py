"""PPO training loop with curriculum learning (small grid → 10×10)."""

import os
import csv
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv

from environment.snake_env import SnakeEnv, N_CHANNELS
from agent.ppo_agent import PPOAgent


# ---------------------------------------------------------------------------
# Observation padding wrapper (curriculum support)
# ---------------------------------------------------------------------------

class _PadObs(gym.ObservationWrapper):
    """
    Zero-pad a smaller-grid SnakeEnv observation to (N_CHANNELS, 10, 10).

    The game plays on an H×H sub-grid (top-left corner of the 10×10 frame).
    Cells outside the sub-grid stay at 0, which the flood-fill channel also
    does naturally — so the action mask correctly blocks moves into the padding.
    Direction channels (4-7) are re-broadcast to the full 10×10 area to
    match the behaviour of a full-size env.
    """
    _TARGET = 10

    def __init__(self, env: SnakeEnv):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(N_CHANNELS, self._TARGET, self._TARGET),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        H = obs.shape[1]
        if H == self._TARGET:
            return obs
        out = np.zeros((N_CHANNELS, self._TARGET, self._TARGET), dtype=np.float32)
        out[:, :H, :H] = obs
        # Re-broadcast active direction channel to the full TARGET×TARGET area
        for c in range(4, N_CHANNELS):
            if obs[c, 0, 0] > 0.5:
                out[c, :, :] = 1.0
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env_fn(grid_size: int):
    def _make():
        env = SnakeEnv(grid_size=grid_size)
        if grid_size < _PadObs._TARGET:
            env = _PadObs(env)
        return env
    return _make


def _action_masks(obs_batch: np.ndarray) -> np.ndarray:
    """
    Compute valid-action masks from a batch of observations.
    Channel 0 = head position, Channel 3 = flood-fill accessibility.
    Directions: UP=0, RIGHT=1, DOWN=2, LEFT=3.

    Returns (N, 4) bool — True = action is safe.
    """
    N, _, H, _ = obs_batch.shape
    head_flat = obs_batch[:, 0].reshape(N, -1).argmax(axis=1)
    hr = head_flat // H
    hc = head_flat % H
    flood = obs_batch[:, 3]                     # (N, H, H)

    dr = np.array([-1,  0,  1,  0])
    dc = np.array([ 0,  1,  0, -1])
    nr = hr[:, None] + dr[None, :]              # (N, 4)
    nc = hc[:, None] + dc[None, :]

    in_bounds  = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < H)
    nr_c = np.clip(nr, 0, H - 1)
    nc_c = np.clip(nc, 0, H - 1)
    flood_vals = flood[np.arange(N)[:, None], nr_c, nc_c]

    valid = in_bounds & (flood_vals > 0)
    valid[~valid.any(axis=1)] = True            # fallback: fully enclosed
    return valid


def _single_mask(obs: np.ndarray) -> np.ndarray:
    """Action mask for a single (C, H, W) observation."""
    H  = obs.shape[1]
    hr, hc = np.unravel_index(obs[0].argmax(), (H, H))
    flood  = obs[3]
    neighbors = [(hr - 1, hc), (hr, hc + 1), (hr + 1, hc), (hr, hc - 1)]
    valid = np.array(
        [0 <= r < H and 0 <= c < H and flood[r, c] > 0 for r, c in neighbors],
        dtype=bool,
    )
    if not valid.any():
        valid[:] = True
    return valid


def evaluate_ppo(agent: PPOAgent, grid_size: int, n_episodes: int = 100) -> dict:
    """Run n_episodes deterministically and return score stats."""
    scores = []
    for _ in range(n_episodes):
        # Use the same wrapper as training so the network always sees 10×10
        env = _make_env_fn(grid_size)()
        obs, _ = env.reset()
        done = False
        while not done:
            mask = _single_mask(obs)   # obs is already 10×10 after wrapper
            action = agent.act(obs, mask=mask, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        scores.append(info["score"])
        env.close()
    return {
        "mean_score":   float(np.mean(scores)),
        "max_score":    int(np.max(scores)),
        "median_score": float(np.median(scores)),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_ppo(cfg: dict, resume: str | None = None) -> PPOAgent:
    """
    PPO training with curriculum:

      6×6 → 7×7 → 8×8 → 9×9 → 10×10

    The network uses AdaptiveAvgPool so the same weights carry over across
    stages without any re-initialisation.  Each stage runs until:
      (a) eval mean score >= graduate_score, or
      (b) max_updates for the stage are exhausted.
    """
    agent_cfg  = cfg["agent"]
    train_cfg  = cfg["training"]
    curriculum = cfg.get("curriculum", [{"grid_size": 10, "graduate_score": 9999}])

    agent = PPOAgent(
        n_channels    = N_CHANNELS,
        action_dim    = 4,
        lr            = agent_cfg["lr"],
        gamma         = agent_cfg["gamma"],
        gae_lambda    = agent_cfg["gae_lambda"],
        clip_eps      = agent_cfg["clip_eps"],
        value_coef    = agent_cfg["value_coef"],
        entropy_coef  = agent_cfg["entropy_coef"],
        max_grad_norm = agent_cfg["max_grad_norm"],
        n_epochs      = agent_cfg["n_epochs"],
        hidden_dim    = agent_cfg.get("hidden_dim", 512),
    )

    global_episode = 0
    if resume:
        global_episode = agent.load(resume)
        print(f"Resumed from {resume} (episode {global_episode})")

    num_envs        = train_cfg["num_envs"]
    rollout_steps   = train_cfg["rollout_steps"]
    minibatch_size  = train_cfg["minibatch_size"]
    eval_frequency  = train_cfg["eval_frequency"]    # updates between evals
    eval_episodes   = train_cfg["eval_episodes"]
    checkpoint_dir  = train_cfg["checkpoint_dir"]
    log_dir         = train_cfg["log_dir"]
    total_updates   = train_cfg["total_updates"]
    init_lr         = agent_cfg["lr"]

    # Entropy decay: start high for exploration, taper to floor by end of training
    ent_start = agent_cfg.get("entropy_coef_start", agent_cfg.get("entropy_coef", 0.1))
    ent_end   = agent_cfg.get("entropy_coef_end",   0.02)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path   = os.path.join(log_dir, "ppo_log.csv")
    csv_file   = open(log_path, "a" if resume else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not resume:
        csv_writer.writerow([
            "update", "episode", "stage", "grid_size",
            "mean_ep_score", "policy_loss", "value_loss",
            "entropy", "approx_kl",
            "eval_mean", "eval_max",
        ])

    best_eval     = -float("inf")
    global_update = 0

    for stage_idx, stage in enumerate(curriculum):
        grid_size      = stage["grid_size"]
        grad_score     = stage.get("graduate_score", float("inf"))
        stage_max_upd  = stage.get("max_updates", total_updates)

        print(
            f"\n{'='*60}\n"
            f"Stage {stage_idx+1}/{len(curriculum)}: {grid_size}×{grid_size}  "
            f"(graduate at mean score ≥ {grad_score})\n"
            f"{'='*60}"
        )

        envs   = SyncVectorEnv([_make_env_fn(grid_size) for _ in range(num_envs)])
        obs, _ = envs.reset()

        # Rolling window for episode stats (last 200 completed episodes)
        recent_scores  = []
        recent_lengths = []
        stage_update   = 0
        graduated      = False

        pbar = tqdm(total=stage_max_upd, desc=f"Stage {stage_idx+1} ({grid_size}×{grid_size})", unit="upd")

        while stage_update < stage_max_upd and global_update < total_updates:

            # ---- Linear LR + entropy annealing ----
            frac = max(0.0, 1.0 - global_update / total_updates)
            agent.set_lr(init_lr * frac)
            agent.entropy_coef = ent_end + (ent_start - ent_end) * frac

            # ---- Allocate rollout buffers ----
            # Observations are always padded to 10×10 by _PadObs wrapper
            C, H = N_CHANNELS, 10
            obs_buf  = np.empty((rollout_steps, num_envs, C, H, H), dtype=np.float32)
            act_buf  = np.empty((rollout_steps, num_envs), dtype=np.int64)
            rew_buf  = np.empty((rollout_steps, num_envs), dtype=np.float32)
            val_buf  = np.empty((rollout_steps, num_envs), dtype=np.float32)
            lp_buf   = np.empty((rollout_steps, num_envs), dtype=np.float32)
            done_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
            mask_buf = np.empty((rollout_steps, num_envs, 4), dtype=bool)

            # ---- Collect rollout ----
            for t in range(rollout_steps):
                masks                  = _action_masks(obs)
                actions, log_probs, values = agent.act_batch(obs, masks)

                next_obs, rewards, terminated, truncated, infos = envs.step(actions)
                dones = terminated | truncated

                obs_buf[t]  = obs
                act_buf[t]  = actions
                rew_buf[t]  = rewards
                val_buf[t]  = values
                lp_buf[t]   = log_probs
                done_buf[t] = dones.astype(np.float32)
                mask_buf[t] = masks

                final_infos = infos.get("final_info", [None] * num_envs)
                for i, done in enumerate(dones):
                    if done:
                        global_episode += 1
                        fi = final_infos[i] if final_infos is not None else None
                        if fi is not None:
                            recent_scores.append(fi.get("score", 0))
                            recent_lengths.append(fi.get("length", 1))

                obs = next_obs

            # Bootstrap values for the obs after the last step
            next_masks               = _action_masks(obs)
            _, _, next_values        = agent.act_batch(obs, next_masks)

            # ---- GAE ----
            advantages, returns = agent.compute_gae(
                rew_buf, val_buf, done_buf, next_values
            )

            # ---- Flatten and update ----
            T, N = rollout_steps, num_envs

            def flat(x):
                return x.reshape(T * N, *x.shape[2:])

            metrics = agent.update(
                states        = flat(obs_buf),
                actions       = flat(act_buf),
                old_log_probs = flat(lp_buf),
                advantages    = flat(advantages),
                returns       = flat(returns),
                masks         = flat(mask_buf),
                minibatch_size = minibatch_size,
            )

            stage_update  += 1
            global_update += 1
            pbar.update(1)

            mean_score = float(np.mean(recent_scores[-200:])) if recent_scores else 0.0

            eval_mean = eval_max = ""

            # ---- Periodic evaluation ----
            if stage_update % eval_frequency == 0:
                eval_result = evaluate_ppo(agent, grid_size=grid_size, n_episodes=eval_episodes)
                eval_mean   = round(eval_result["mean_score"], 2)
                eval_max    = int(eval_result["max_score"])

                pbar.set_postfix({
                    "score": f"{mean_score:.2f}",
                    "eval":  eval_mean,
                    "ent":   f"{metrics['entropy']:.3f}",
                    "kl":    f"{metrics['approx_kl']:.4f}",
                })

                if eval_result["mean_score"] > best_eval:
                    best_eval = eval_result["mean_score"]
                    agent.save(os.path.join(checkpoint_dir, "best.pt"), episode=global_episode)

                if eval_result["mean_score"] >= grad_score:
                    print(
                        f"\n  *** GRADUATED from {grid_size}×{grid_size}! "
                        f"eval_mean={eval_mean} >= {grad_score} ***"
                    )
                    graduated = True

            # ---- CSV logging ----
            if stage_update % 50 == 0 or graduated:
                csv_writer.writerow([
                    global_update, global_episode, stage_idx + 1, grid_size,
                    round(mean_score, 3),
                    round(metrics["policy_loss"], 6),
                    round(metrics["value_loss"], 6),
                    round(metrics["entropy"], 6),
                    round(metrics["approx_kl"], 6),
                    eval_mean, eval_max,
                ])
                csv_file.flush()

            # ---- Periodic checkpoint ----
            if global_update % 1000 == 0:
                agent.save(
                    os.path.join(checkpoint_dir, f"update_{global_update}.pt"),
                    episode=global_episode,
                )

            if graduated or global_update >= total_updates:
                break

        pbar.close()
        envs.close()

        if global_update >= total_updates:
            print("Reached total_updates budget — stopping.")
            break

    # ---- Final checkpoint ----
    agent.save(os.path.join(checkpoint_dir, "final.pt"), episode=global_episode)
    csv_file.close()

    # ---- Final evaluation on 10×10 ----
    print(f"\nFinal evaluation on 10×10 ({eval_episodes} episodes)…")
    final_eval = evaluate_ppo(agent, grid_size=10, n_episodes=eval_episodes)
    print(
        f"  mean={final_eval['mean_score']:.2f}  "
        f"max={final_eval['max_score']}  "
        f"median={final_eval['median_score']:.2f}"
    )
    print(f"  Best eval score during training: {best_eval:.2f}")
    print(f"  Checkpoints: {checkpoint_dir}/")

    return agent
