"""Tiny built-in games used to generate training videos.

Each game lives on a 16x16 grid and renders to a 64x64 RGB frame (4 px per
cell). Games record the true action at every step, so we can measure how well
the unsupervised latent actions line up with the real controls. The world
model itself never sees these labels.
"""

from __future__ import annotations

import numpy as np

GRID = 16
CELL = 4
FRAME = GRID * CELL  # 64


def _hex(c: str) -> np.ndarray:
    c = c.lstrip("#")
    return np.array([int(c[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def _upscale(grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(grid, CELL, axis=0), CELL, axis=1)


class Game:
    name = "game"
    actions: list[str] = []

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.last_action = 0
        self.reset()

    def reset(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def step(self, action: int) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def render(self) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def policy(self) -> int:
        """A sticky random policy. Subclasses add a bit of intent."""
        if self.rng.random() < 0.75:
            return self.last_action
        return int(self.rng.integers(len(self.actions)))


class Platformer(Game):
    name = "platformer"
    actions = ["noop", "left", "right", "jump"]

    SKY, CLOUD = _hex("#1E2A44"), _hex("#3A4A6B")
    PLATFORM, GRASS, DIRT = _hex("#8B5E34"), _hex("#2F9E66"), _hex("#6B4423")
    BODY, HEAD = _hex("#F5B83D"), _hex("#FFD27A")
    COIN, ENEMY = _hex("#FFE066"), _hex("#F07167")

    def reset(self) -> None:
        self.solid = np.zeros((GRID, GRID), dtype=bool)
        self.solid[14:, :] = True
        for _ in range(2):
            row = int(self.rng.integers(9, 12))
            start = int(self.rng.integers(1, 11))
            length = int(self.rng.integers(3, 6))
            self.solid[row, start : start + length] = True
        self.x, self.y, self.up = 2, 13, 0
        self.enemy_x, self.enemy_dir, self.tick = 12, -1, 0
        self._spawn_coin()

    def _free(self, r: int, c: int) -> bool:
        return 0 <= r < GRID and 0 <= c < GRID and not self.solid[r, c]

    def _spawn_coin(self) -> None:
        while True:
            r, c = int(self.rng.integers(5, 13)), int(self.rng.integers(0, GRID))
            if self._free(r, c):
                self.coin = (r, c)
                return

    def _on_ground(self) -> bool:
        return not self._free(self.y + 1, self.x)

    def step(self, action: int) -> None:
        self.last_action = action
        if action in (1, 2):
            nx = self.x + (-1 if action == 1 else 1)
            if self._free(self.y, nx) and self._free(self.y - 1, nx):
                self.x = nx
        if action == 3 and self._on_ground() and self.up == 0:
            self.up = 3
        if self.up > 0:
            if self._free(self.y - 2, self.x):
                self.y -= 1
                self.up -= 1
            else:
                self.up = 0
        elif not self._on_ground():
            self.y += 1

        self.tick += 1
        if self.tick % 2 == 0:
            nx = self.enemy_x + self.enemy_dir
            if not 6 <= nx <= 15:
                self.enemy_dir *= -1
                nx = self.enemy_x + self.enemy_dir
            self.enemy_x = nx
        if self.enemy_x == self.x and self.y == 13:
            self.x, self.y, self.up = 2, 13, 0
        if self.coin[1] == self.x and self.coin[0] in (self.y, self.y - 1):
            self._spawn_coin()

    def policy(self) -> int:
        if self.rng.random() < 0.75:
            return self.last_action
        return int(self.rng.choice(4, p=[0.15, 0.25, 0.35, 0.25]))

    def render(self) -> np.ndarray:
        g = np.empty((GRID, GRID, 3), dtype=np.uint8)
        g[:] = self.SKY
        g[2, 2:5] = self.CLOUD
        g[3, 10:13] = self.CLOUD
        g[self.solid] = self.PLATFORM
        g[14, :] = self.GRASS
        g[15, :] = self.DIRT
        g[self.coin] = self.COIN
        g[13, self.enemy_x] = self.ENEMY
        g[self.y, self.x] = self.BODY
        if self.y - 1 >= 0:
            g[self.y - 1, self.x] = self.HEAD
        return _upscale(g)


class Paddle(Game):
    name = "paddle"
    actions = ["noop", "up", "down"]

    BG, LINE = _hex("#0B0F17"), _hex("#2A3140")
    PADDLE, BALL = _hex("#E8ECF2"), _hex("#F5B83D")

    def reset(self) -> None:
        self.py, self.ay = 6, 6
        self._serve()

    def _serve(self) -> None:
        self.bx, self.by = 8, int(self.rng.integers(3, 13))
        self.dx = int(self.rng.choice([-1, 1]))
        self.dy = int(self.rng.choice([-1, 1]))

    def step(self, action: int) -> None:
        self.last_action = action
        if action == 1:
            self.py = max(0, self.py - 1)
        elif action == 2:
            self.py = min(GRID - 3, self.py + 1)
        if self.rng.random() < 0.7:
            target = self.by - 1
            self.ay += int(np.sign(target - self.ay))
            self.ay = int(np.clip(self.ay, 0, GRID - 3))

        ny = self.by + self.dy
        if not 0 <= ny < GRID:
            self.dy *= -1
            ny = self.by + self.dy
        if self.dx < 0 and self.bx == 2 and self.py <= ny < self.py + 3:
            self.dx = 1
        if self.dx > 0 and self.bx == 13 and self.ay <= ny < self.ay + 3:
            self.dx = -1
        nx = self.bx + self.dx
        if not 0 <= nx < GRID:
            self._serve()
            return
        self.bx, self.by = nx, ny

    def policy(self) -> int:
        if self.rng.random() < 0.6:
            centre = self.py + 1
            if self.by < centre:
                return 1
            if self.by > centre:
                return 2
            return 0
        return super().policy()

    def render(self) -> np.ndarray:
        g = np.empty((GRID, GRID, 3), dtype=np.uint8)
        g[:] = self.BG
        g[::2, 8] = self.LINE
        g[self.py : self.py + 3, 1] = self.PADDLE
        g[self.ay : self.ay + 3, 14] = self.PADDLE
        g[self.by, self.bx] = self.BALL
        return _upscale(g)


class Snake(Game):
    name = "snake"
    actions = ["noop", "up", "down", "left", "right"]
    DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}

    BG, BODY = _hex("#13261B"), _hex("#34D399")
    HEAD, APPLE = _hex("#B7F5D8"), _hex("#F07167")

    def reset(self) -> None:
        self.body = [(8, 6), (8, 5), (8, 4)]
        self.heading = 4
        self._spawn_apple()

    def _spawn_apple(self) -> None:
        while True:
            p = (int(self.rng.integers(0, GRID)), int(self.rng.integers(0, GRID)))
            if p not in self.body:
                self.apple = p
                return

    def _next_head(self, heading: int) -> tuple[int, int]:
        dr, dc = self.DIRS[heading]
        return self.body[0][0] + dr, self.body[0][1] + dc

    def _safe(self, heading: int) -> bool:
        r, c = self._next_head(heading)
        return 0 <= r < GRID and 0 <= c < GRID and (r, c) not in self.body[:-1]

    def step(self, action: int) -> None:
        self.last_action = action
        if action in self.DIRS and action != self.OPPOSITE[self.heading]:
            self.heading = action
        head = self._next_head(self.heading)
        r, c = head
        if not (0 <= r < GRID and 0 <= c < GRID) or head in self.body[:-1]:
            self.reset()
            return
        self.body.insert(0, head)
        if head == self.apple:
            self._spawn_apple()
        else:
            self.body.pop()

    def policy(self) -> int:
        options = [a for a in self.DIRS if a != self.OPPOSITE[self.heading] and self._safe(a)]
        if not options:
            return 0
        if self.rng.random() < 0.6:
            hr, hc = self.body[0]
            ar, ac = self.apple
            toward = []
            if ar < hr:
                toward.append(1)
            if ar > hr:
                toward.append(2)
            if ac < hc:
                toward.append(3)
            if ac > hc:
                toward.append(4)
            toward = [a for a in toward if a in options]
            if toward:
                return int(self.rng.choice(toward))
        if self._safe(self.heading) and self.rng.random() < 0.6:
            return 0
        return int(self.rng.choice(options))

    def render(self) -> np.ndarray:
        g = np.empty((GRID, GRID, 3), dtype=np.uint8)
        g[:] = self.BG
        g[self.apple] = self.APPLE
        for p in self.body[1:]:
            g[p] = self.BODY
        g[self.body[0]] = self.HEAD
        return _upscale(g)


GAMES: dict[str, type[Game]] = {g.name: g for g in (Platformer, Paddle, Snake)}


def make(name: str, seed: int = 0) -> Game:
    if name not in GAMES:
        raise ValueError(f"Unknown game '{name}'. Choose from: {', '.join(GAMES)}")
    return GAMES[name](np.random.default_rng(seed))
