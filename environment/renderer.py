"""Pygame renderer for Snake. Decoupled from game logic."""

import pygame
import numpy as np
from environment.snake_game import EMPTY, SNAKE, FOOD, HEAD

# Colors
BLACK      = (  0,   0,   0)
WHITE      = (255, 255, 255)
GREEN      = ( 34, 177,  76)
DARK_GREEN = ( 20, 120,  50)
RED        = (200,  50,  50)
BLUE_HEAD  = ( 50, 130, 220)
GRAY       = ( 40,  40,  40)


class Renderer:
    """
    Renders a SnakeGame via pygame.

    Usage:
        renderer = Renderer(grid_size=20, cell_size=30)
        renderer.init()
        renderer.draw(game)
        renderer.close()
    """

    def __init__(self, grid_size: int = 20, cell_size: int = 30, caption: str = "Snake RL"):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.caption = caption
        self.window_size = grid_size * cell_size
        self.screen = None
        self.font = None
        self.clock = None

    def init(self) -> None:
        """Initialize pygame window. Call once before drawing."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size + 40))
        pygame.display.set_caption(self.caption)
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.clock = pygame.time.Clock()

    def draw(self, game, fps: int = 15) -> bool:
        """
        Draw the current game state.

        Returns:
            False if the window was closed (quit event), True otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.screen.fill(BLACK)
        self._draw_grid(game.get_board())
        self._draw_score(game.score, game.snake_length())
        pygame.display.flip()
        self.clock.tick(fps)
        return True

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _draw_grid(self, board: np.ndarray) -> None:
        cs = self.cell_size
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(c * cs, r * cs, cs - 1, cs - 1)
                cell = board[r][c]
                if cell == HEAD:
                    pygame.draw.rect(self.screen, BLUE_HEAD, rect, border_radius=4)
                elif cell == SNAKE:
                    pygame.draw.rect(self.screen, GREEN, rect, border_radius=3)
                elif cell == FOOD:
                    # Draw food as a circle
                    cx = c * cs + cs // 2
                    cy = r * cs + cs // 2
                    pygame.draw.circle(self.screen, RED, (cx, cy), cs // 2 - 2)
                else:
                    pygame.draw.rect(self.screen, GRAY, rect)

    def _draw_score(self, score: int, length: int) -> None:
        y_offset = self.grid_size * self.cell_size
        bg_rect = pygame.Rect(0, y_offset, self.window_size, 40)
        pygame.draw.rect(self.screen, (20, 20, 20), bg_rect)
        text = self.font.render(f"Score: {score}   Length: {length}", True, WHITE)
        self.screen.blit(text, (10, y_offset + 10))
