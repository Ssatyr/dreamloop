"""Render the GIFs used in README.md from trained runs.

    python scripts/readme_gifs.py --platformer runs/platformer-small --paddle runs/paddle-small

Writes to assets/:
  platformer.gif   real game vs. dream, driven by actions read from the real video
  long.gif         a long rollout (the full held-out episode) to show drift
  random.gif       paddle: after a point the real game serves at random; the dream can't know where
  keys.gif         platformer dreamed from held keys only (as in `play`): right, then jump held down
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dreamloop.play import Session
from dreamloop.rollout import WorldModel, make_gif


def keys_gif(wm: WorldModel, out: Path, plan: list[tuple[str, int]], scale: int = 4, seed: int = 1) -> Path:
    """Dream from held keys only, like `dreamloop play`, with the held key written on each frame."""
    session = Session(wm, temperature=0.0, seed=seed)
    frames = [(session.reset(), "start")]
    for key, n in plan:
        for _ in range(n):
            frames.append((session.step(wm.bindings[key]), f"holding {key}"))
    size, bar = 64 * scale, 18
    images = []
    for frame, label in frames:
        canvas = Image.new("RGB", (size, size + bar), (15, 18, 24))
        canvas.paste(Image.fromarray(frame).resize((size, size), Image.NEAREST), (0, bar))
        ImageDraw.Draw(canvas).text((6, 4), f"dream, {label}", fill=(232, 236, 242))
        images.append(canvas)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = wm.cfg.get("fps", 15)
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
    return out


def first_serve(wm: WorldModel) -> tuple[int, int]:
    """(episode, step) of the first held-out serve: the ball reappears in the centre column."""
    ball = np.array([245, 184, 61])
    for e, ep in enumerate(wm.seed_frames):
        cols = [np.nonzero((np.abs(f.astype(int) - ball).max(-1) < 30).any(0))[0] for f in ep]
        for t in range(wm.context + 1, len(ep)):
            if len(cols[t]) and len(cols[t - 1]) and abs(int(cols[t].mean()) - int(cols[t - 1].mean())) > 8:
                return e, t
    return 0, wm.context


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--platformer", default="runs/platformer-small")
    p.add_argument("--paddle", default="runs/paddle-small")
    p.add_argument("--out", default="assets")
    args = p.parse_args()
    out = Path(args.out)

    wm = WorldModel(args.platformer)
    print("saved", make_gif(wm, out / "platformer.gif", episode=0, steps=90))
    # Episodes are fixed in advance (the first held-out ones), not picked for looking good.
    print("saved", make_gif(wm, out / "long.gif", episode=1, steps=156, scale=3))
    # seed 0 starts away from the enemy (touching it, the dreamed player vanishes: another failure)
    print("saved", keys_gif(wm, out / "keys.gif", [("jump", 16), ("right", 5)], scale=3, seed=0))

    wm = WorldModel(args.paddle)
    e, t = first_serve(wm)
    # start a few steps before the serve, so the GIF shows the point being lost and the new serve
    start = max(0, t - wm.context - 6)
    wm.seed_frames = wm.seed_frames[:, start:]
    wm.seed_actions = wm.seed_actions[:, start:]
    print("saved", make_gif(wm, out / "random.gif", episode=e, steps=30, scale=3), f"(episode {e}, serve at step {t})")


if __name__ == "__main__":
    main()
