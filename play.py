"""Human-playable Snake. Use arrow keys or WASD."""

import pygame
import sys
from environment.snake_game import SnakeGame, Direction
from environment.renderer import Renderer


def main():
    grid_size = 20
    cell_size = 30
    fps = 10

    game = SnakeGame(grid_size=grid_size)
    renderer = Renderer(grid_size=grid_size, cell_size=cell_size, caption="Snake — Human Play")
    renderer.init()

    key_to_action = {
        pygame.K_UP:    Direction.UP,
        pygame.K_w:     Direction.UP,
        pygame.K_DOWN:  Direction.DOWN,
        pygame.K_s:     Direction.DOWN,
        pygame.K_LEFT:  Direction.LEFT,
        pygame.K_a:     Direction.LEFT,
        pygame.K_RIGHT: Direction.RIGHT,
        pygame.K_d:     Direction.RIGHT,
    }

    action = Direction.RIGHT
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in key_to_action:
                    action = key_to_action[event.key]

        game_over, score, _ = game.step(int(action))

        if not renderer.draw(game, fps=fps):
            running = False

        if game_over:
            print(f"Game over! Score: {score}")
            game.reset()
            action = Direction.RIGHT

    renderer.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
