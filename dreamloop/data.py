"""Gameplay datasets: frames plus (optional) true actions, stored as .npz.

frames:  uint8  [episodes, steps, 64, 64, 3]
actions: int16  [episodes, steps]  action taken after frame t (-1 = unknown)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import games


@dataclass
class Dataset:
    frames: np.ndarray
    actions: np.ndarray
    game: str
    action_names: list[str]

    @property
    def has_actions(self) -> bool:
        return bool((self.actions[:, :-1] >= 0).all())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frames=self.frames,
            actions=self.actions,
            game=np.array(self.game),
            action_names=np.array(self.action_names, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        d = np.load(path, allow_pickle=True)
        return cls(
            frames=d["frames"],
            actions=d["actions"],
            game=str(d["game"]),
            action_names=[str(a) for a in d["action_names"]],
        )

    def __repr__(self) -> str:
        e, t = self.frames.shape[:2]
        return f"Dataset(game={self.game!r}, episodes={e}, steps={t}, labelled={self.has_actions})"


def generate(game: str, episodes: int, steps: int, seed: int = 0) -> Dataset:
    frames = np.empty((episodes, steps, games.FRAME, games.FRAME, 3), dtype=np.uint8)
    actions = np.full((episodes, steps), -1, dtype=np.int16)
    names: list[str] = []
    for e in range(episodes):
        g = games.make(game, seed=seed * 100_003 + e)
        names = g.actions
        for t in range(steps):
            frames[e, t] = g.render()
            if t < steps - 1:
                a = g.policy()
                actions[e, t] = a
                g.step(a)
    return Dataset(frames, actions, game, names)
