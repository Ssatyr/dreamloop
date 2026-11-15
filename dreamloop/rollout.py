"""Load a trained run, roll the world model forward, make GIFs, measure drift."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw

from .models import Dynamics, LatentActionModel, Tokenizer


class WorldModel:
    def __init__(self, run_dir: str | Path):
        run = Path(run_dir)
        meta = json.loads((run / "config.json").read_text())
        cfg = meta["config"]
        self.game, self.action_names, self.cfg = meta["game"], meta["action_names"], cfg
        self.tok = Tokenizer(cfg["tok_ch"], cfg["tok_levels"])
        self.tok.load_weights(str(run / "tokenizer.safetensors"))
        self.lam = LatentActionModel(cfg["lam_ch"], cfg["lam_levels"])
        self.lam.load_weights(str(run / "lam.safetensors"))
        self.dyn = Dynamics(self.tok.vocab, self.lam.num_actions, cfg["context"], 64,
                            cfg["dyn_d"], cfg["dyn_layers"], cfg["dyn_heads"])
        self.dyn.load_weights(str(run / "dynamics.safetensors"))
        for m in (self.tok, self.lam, self.dyn):
            m.eval()
        seeds = np.load(run / "seeds.npz")
        self.seed_frames, self.seed_actions = seeds["frames"], seeds["actions"]
        metrics_path = run / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        report = metrics.get("actions") or {}
        self.bindings: dict[str, int] = report.get("bindings", {})
        self.context = cfg["context"]

    @staticmethod
    def _f(frames: np.ndarray) -> mx.array:
        return mx.array(frames.astype(np.float32) / 255.0)

    def encode(self, frames: np.ndarray) -> np.ndarray:
        return np.array(self.tok.encode(self._f(frames))).astype(np.int32)

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        img = self.tok.decode(mx.array(tokens))
        return np.array(img * 255).round().clip(0, 255).astype(np.uint8)

    def infer_actions(self, f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
        return np.array(self.lam.action_ids(self._f(f0), self._f(f1))).astype(np.int32)

    def next_tokens(self, ctx: np.ndarray, action: np.ndarray, steps: int = 4, temperature: float = 0.0) -> np.ndarray:
        out = self.dyn.generate(mx.array(ctx), mx.array(action), steps=steps, temperature=temperature)
        return np.array(out).astype(np.int32)

    def latent_for_true(self, true_actions: np.ndarray) -> np.ndarray:
        lookup = np.array([self.bindings.get(name, 0) for name in self.action_names], dtype=np.int32)
        return lookup[true_actions]

    def rollout(self, episodes: np.ndarray, true_actions: np.ndarray | None, horizon: int,
                temperature: float = 0.0, latent_actions: np.ndarray | None = None) -> np.ndarray:
        """Dream `horizon` frames for each episode, starting from its first C real frames.

        Controls come from `latent_actions` [B, horizon] if given; else from the key
        bindings when true actions exist (like a player pressing keys); otherwise from
        latent actions inferred on the real video.
        """
        c = self.context
        b = episodes.shape[0]
        ctx = np.stack([self.encode(ep[:c]) for ep in episodes])
        dreamed = []
        for i in range(horizon):
            t = c + i
            if latent_actions is not None:
                action = latent_actions[:, i].astype(np.int32)
            elif true_actions is not None and self.bindings:
                action = self.latent_for_true(true_actions[:, t - 1])
            else:
                action = self.infer_actions(episodes[:, t - 1], episodes[:, t])
            nxt = self.next_tokens(ctx, action, temperature=temperature)
            dreamed.append(self.decode(nxt))
            ctx = np.concatenate([ctx[:, 1:], nxt[:, None]], axis=1)
        return np.stack(dreamed, axis=1).reshape(b, horizon, *episodes.shape[2:])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) / 255 - b.astype(np.float32) / 255) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def _changed(a: np.ndarray, b: np.ndarray, tol: int = 10) -> np.ndarray:
    """[..., H, W, 3] uint8 frames -> [..., H, W] bool: pixels that differ by more than tol/255."""
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1) > tol


def moving_psnr(real: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> float:
    """PSNR over only the pixels that changed since `ref`, in the real frame or in the prediction.

    Whole-frame PSNR is useless on these games: ~98% of pixels are static background, so
    even a frozen picture scores above 20 dB. Here a frozen picture scores ~5 dB, a dream
    one step behind ~10 dB, and the tokenizer's own reconstruction (the best case) ~28 dB.
    """
    mask = _changed(real, ref) | _changed(pred, ref)
    if not mask.any():
        return 99.0
    diff = (real.astype(np.float32) - pred.astype(np.float32)) / 255
    mse = float(np.mean(diff[mask] ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def evaluate_drift(wm: WorldModel, horizon: int = 64, threshold_db: float = 20.0, seed: int = 0) -> dict:
    """How long do dreams stay close to the real game, and do the controls matter?

    Compares three rollouts from the same held-out starts, all scored with moving_psnr:
    the dream with the real controls, the dream with random actions, and a frozen picture.
    """
    c = wm.context
    horizon = min(horizon, wm.seed_frames.shape[1] - c)
    real = wm.seed_frames[:, c : c + horizon]
    ref = wm.seed_frames[:, c - 1]
    dreams = wm.rollout(wm.seed_frames, wm.seed_actions, horizon)
    rng = np.random.default_rng(seed)
    random_actions = rng.integers(0, wm.lam.num_actions, (len(real), horizon))
    random_dreams = wm.rollout(wm.seed_frames, None, horizon, latent_actions=random_actions)

    def curve(pred: np.ndarray) -> list[float]:
        return [round(moving_psnr(real[:, h], pred[:, h], ref), 2) for h in range(horizon)]

    def until(values: list[float]) -> int:
        return next((h for h, v in enumerate(values) if v < threshold_db), horizon)

    dream_curve = curve(dreams)
    random_curve = curve(random_dreams)
    frozen_curve = curve(np.repeat(ref[:, None], horizon, axis=1))
    steps = until(dream_curve)
    fps = wm.cfg.get("fps", 15)
    at = min(4, horizon) - 1  # Genie reports its controllability metric at step 4
    return {
        "metric": "moving-pixel PSNR (pixels that changed since the last real frame)",
        "moving_psnr_by_step": dream_curve,
        "random_actions_psnr_by_step": random_curve,
        "frozen_psnr_by_step": frozen_curve,
        "full_frame_psnr_by_step": [round(psnr(real[:, h], dreams[:, h]), 2) for h in range(horizon)],
        "steps_until_drift": steps,
        "seconds_until_drift": round(steps / fps, 1),
        "frozen_steps_until_drift": until(frozen_curve),
        "control_gain_db_at_4": round(dream_curve[at] - random_curve[at], 2),
        "threshold_db": threshold_db,
        "horizon": horizon,
    }


def drift_summary(drift: dict) -> str:
    """Human-readable lines for `dreamloop eval` and `dreamloop demo`."""
    marks = [h for h in (1, 4, 16, 32, 64) if h <= drift["horizon"]]

    def row(name: str, key: str) -> str:
        return f"  {name:<22}" + "  ".join(f"{drift[key][h - 1]:5.1f}" for h in marks)

    capped = " (never drifted within the test)" if drift["steps_until_drift"] == drift["horizon"] else ""
    return "\n".join([
        "Moving-pixel PSNR (dB), only pixels that change; steps " + ", ".join(map(str, marks)) + ":",
        row("dream, real controls", "moving_psnr_by_step"),
        row("dream, random actions", "random_actions_psnr_by_step"),
        row("frozen picture", "frozen_psnr_by_step"),
        f"Stays above {drift['threshold_db']:.0f} dB for {drift['steps_until_drift']} steps "
        f"({drift['seconds_until_drift']} s){capped}; frozen picture: {drift['frozen_steps_until_drift']} steps.",
        f"Controls matter: {drift['control_gain_db_at_4']:+.1f} dB at step 4 vs. random actions "
        "(about 0 means the actions do nothing).",
    ])


def _label(img: Image.Image, text: str, x: int) -> None:
    ImageDraw.Draw(img).text((x + 6, 4), text, fill=(232, 236, 242))


def make_gif(wm: WorldModel, out: str | Path, episode: int = 0, steps: int = 90, scale: int = 4,
             temperature: float = 0.0) -> Path:
    """Side-by-side GIF: real game on the left, the network's dream on the right."""
    c = wm.context
    ep = wm.seed_frames[episode : episode + 1]
    acts = wm.seed_actions[episode : episode + 1]
    steps = min(steps, ep.shape[1] - c)
    dream = wm.rollout(ep, acts, steps, temperature=temperature)[0]
    real = ep[0, c : c + steps]
    size, gap, bar = 64 * scale, 8, 18
    frames = []
    for r, d in zip(real, dream):
        canvas = Image.new("RGB", (size * 2 + gap, size + bar), (15, 18, 24))
        canvas.paste(Image.fromarray(r).resize((size, size), Image.NEAREST), (0, bar))
        canvas.paste(Image.fromarray(d).resize((size, size), Image.NEAREST), (size + gap, bar))
        _label(canvas, "real game", 0)
        _label(canvas, "dreamed by dreamloop", size + gap)
        frames.append(canvas)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = wm.cfg.get("fps", 15)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return out
