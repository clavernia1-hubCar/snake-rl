"""Entry point for training the Snake RL agent."""

import argparse
import yaml

from training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train Snake DQN agent")
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Training with config: {args.config}")
    print(f"Grid size: {cfg['environment']['grid_size']}  "
          f"Episodes: {cfg['training']['total_episodes']}")

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
