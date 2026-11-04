"""`dreamloop record` (experimental): capture a screen region as unlabelled gameplay.

Needs `pip install mss`. On macOS, allow your terminal under
System Settings > Privacy & Security > Screen Recording.
No key presses are logged: the latent action model learns the controls from video.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from .data import Dataset

SIZE = 64


def record(region: str, minutes: float, out: str, fps: int = 15, episode_len: int = 200, name: str = "recorded") -> Dataset:
    try:
        import mss
    except ImportError as err:  # pragma: no cover - optional dependency
        raise SystemExit("Recording needs mss: pip install 'dreamloop[record]'") from err

    x, y, w, h = (int(v) for v in region.split(","))
    box = {"left": x, "top": y, "width": w, "height": h}
    total = int(minutes * 60 * fps)
    frames = []
    print(f"Recording {w}x{h} at ({x}, {y}) for {minutes} min. Play now; Ctrl+C stops early.")
    with mss.mss() as screen:
        next_t = time.perf_counter()
        try:
            for i in range(total):
                shot = screen.grab(box)
                img = Image.frombytes("RGB", shot.size, shot.rgb).resize((SIZE, SIZE), Image.BOX)
                frames.append(np.asarray(img, dtype=np.uint8))
                if i % (fps * 10) == 0:
                    print(f"  {i / fps:5.0f} s captured", flush=True)
                next_t += 1 / fps
                time.sleep(max(0.0, next_t - time.perf_counter()))
        except KeyboardInterrupt:
            print("Stopped early.")

    n = len(frames) // episode_len
    if n < 2:
        raise SystemExit(f"Only {len(frames)} frames captured; record at least {2 * episode_len / fps:.0f} seconds.")
    arr = np.stack(frames[: n * episode_len]).reshape(n, episode_len, SIZE, SIZE, 3)
    ds = Dataset(arr, np.full(arr.shape[:2], -1, dtype=np.int16), name, [])
    ds.save(out)
    print(f"Saved {ds} to {out}")
    return ds
