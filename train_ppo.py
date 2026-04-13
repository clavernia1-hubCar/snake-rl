"""Entry point for PPO Snake training."""

import argparse
import yaml

from training.ppo_trainer import train_ppo


def main():
    parser = argparse.ArgumentParser(description="Train Snake PPO agent")
    parser.add_argument("--config", default="config/ppo_config.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint .pt to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    curriculum = cfg.get("curriculum", [{"grid_size": 10}])
    grid_sizes = [s["grid_size"] for s in curriculum]
    print(f"PPO training | curriculum: {' → '.join(str(g) for g in grid_sizes)}")
    print(f"Total updates: {cfg['training']['total_updates']}")
    print(f"Envs: {cfg['training']['num_envs']}  "
          f"Rollout steps: {cfg['training']['rollout_steps']}  "
          f"Minibatch: {cfg['training']['minibatch_size']}")

    train_ppo(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
