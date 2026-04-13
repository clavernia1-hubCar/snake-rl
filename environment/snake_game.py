"""Pure Snake game logic. No RL, no rendering dependencies."""

import numpy as np
from collections import deque
from enum import IntEnum


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# Delta (row, col) for each direction
DIRECTION_DELTA = {
    Direction.UP:    (-1,  0),
    Direction.RIGHT: ( 0,  1),
    Direction.DOWN:  ( 1,  0),
    Direction.LEFT:  ( 0, -1),
}

# Cell type constants for the board array
EMPTY = 0
SNAKE = 1
FOOD  = 2
HEAD  = 3


class SnakeGame:
    """
    Snake game logic.

    Coordinate system: (row, col), origin top-left.
    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT (same as Direction enum).
    """

    def __init__(self, grid_size: int = 20, seed: int | None = None):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the game to initial state."""
        mid = self.grid_size // 2
        # Snake starts as 3 cells long, heading right
        self.direction = Direction.RIGHT
        self.body: deque[tuple[int, int]] = deque([
            (mid, mid),
            (mid, mid - 1),
            (mid, mid - 2),
        ])
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self._build_board()
        self._place_food()

    def step(self, action: int) -> tuple[bool, int, tuple[int, int]]:
        """
        Advance the game by one step.

        Args:
            action: Direction int (0=UP,1=RIGHT,2=DOWN,3=LEFT).
                    Opposite-direction moves are ignored (snake keeps heading).

        Returns:
            (game_over, score, new_head_pos)
        """
        new_dir = Direction(action)
        # Prevent 180-degree reversal
        if not self._is_opposite(new_dir, self.direction):
            self.direction = new_dir

        dr, dc = DIRECTION_DELTA[self.direction]
        head_r, head_c = self.body[0]
        new_head = (head_r + dr, head_c + dc)

        self.steps += 1
        self.steps_since_food += 1

        # Wall collision
        if not self._in_bounds(new_head):
            return True, self.score, new_head

        new_r, new_c = new_head

        # Self collision (ignore tail — it will move away unless eating)
        body_set = set(self.body)
        if new_head in body_set and new_head != self.body[-1]:
            return True, self.score, new_head

        ate_food = (new_r == self.food_pos[0] and new_c == self.food_pos[1])

        # Move snake
        self.body.appendleft(new_head)
        if ate_food:
            self.score += 1
            self.steps_since_food = 0
            self.board[new_r][new_c] = HEAD
            # Update old head to body
            old_r, old_c = self.body[1]
            self.board[old_r][old_c] = SNAKE
            food_placed = self._place_food()
            if not food_placed:
                # Board is full — player wins
                return True, self.score, new_head
        else:
            tail = self.body.pop()
            self.board[tail[0]][tail[1]] = EMPTY
            old_r, old_c = self.body[1] if len(self.body) > 1 else new_head
            if len(self.body) > 1:
                self.board[old_r][old_c] = SNAKE
            self.board[new_r][new_c] = HEAD

        return False, self.score, new_head

    def get_board(self) -> np.ndarray:
        """Return a copy of the current board (grid_size x grid_size)."""
        return self.board.copy()

    def get_head(self) -> tuple[int, int]:
        return self.body[0]

    def get_food(self) -> tuple[int, int]:
        return self.food_pos

    def get_direction(self) -> Direction:
        return self.direction

    def snake_length(self) -> int:
        return len(self.body)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_board(self) -> None:
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for i, (r, c) in enumerate(self.body):
            self.board[r][c] = HEAD if i == 0 else SNAKE

    def _place_food(self) -> bool:
        """Place food on a random empty cell. Returns False if board is full."""
        empty_mask = self.board == EMPTY
        empty_coords = np.argwhere(empty_mask)
        if len(empty_coords) == 0:
            return False
        idx = self.rng.integers(len(empty_coords))
        r, c = empty_coords[idx]
        self.food_pos = (int(r), int(c))
        self.board[r][c] = FOOD
        return True

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    @staticmethod
    def _is_opposite(d1: Direction, d2: Direction) -> bool:
        return (d1 == Direction.UP   and d2 == Direction.DOWN)  or \
               (d1 == Direction.DOWN and d2 == Direction.UP)    or \
               (d1 == Direction.LEFT and d2 == Direction.RIGHT) or \
               (d1 == Direction.RIGHT and d2 == Direction.LEFT)
