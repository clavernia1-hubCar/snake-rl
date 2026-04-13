"""Watch a trained Snake agent play, with optional stats output."""

import argparse
import yaml

import pygame

from environment.snake_env import SnakeEnv
from environment.renderer import Renderer
from agent.dqn_agent import DQNAgent


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake agent")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt",
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", default="config/default_config.yaml",
                        help="Config file (used for grid_size)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to watch")
    parser.add_argument("--fps", type=int, default=15,
                        help="Playback speed (frames per second)")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Exploration rate during evaluation (0=greedy)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    grid_size = cfg["environment"]["grid_size"]

    agent = DQNAgent(grid_size=grid_size, n_channels=7, action_dim=4)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    env = SnakeEnv(grid_size=grid_size)
    renderer = Renderer(grid_size=grid_size, cell_size=30, caption="Snake RL — Evaluation")
    renderer.init()

    scores = []
    running = True

    for ep in range(args.episodes):
        if not running:
            break
        obs, _ = env.reset()
        done = False
        ep_score = 0

        while not done and running:
            # Handle quit events between steps
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False

            action = agent.act(obs, epsilon=args.epsilon)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_score = info["score"]

            if not renderer.draw(env.game, fps=args.fps):
                running = False

        scores.append(ep_score)
        print(f"Episode {ep+1}: score={ep_score}  length={info['length']}")

    renderer.close()
    env.close()

    if scores:
        print(f"\nResults over {len(scores)} episode(s):")
        print(f"  Mean score: {sum(scores)/len(scores):.2f}")
        print(f"  Max score:  {max(scores)}")


if __name__ == "__main__":
    main()
