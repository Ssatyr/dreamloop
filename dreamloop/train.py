"""Train tokenizer -> latent actions -> dynamics, then save a playable run."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .data import Dataset
from .models import Dynamics, LatentActionModel, Tokenizer, count_params


@dataclass
class Config:
    # tokenizer
    tok_ch: int = 48
    tok_levels: list[int] = field(default_factory=lambda: [8, 5, 5, 5])
    tok_steps: int = 3000
    # latent action model
    lam_ch: int = 32
    num_actions: int = 8
    lam_steps: int = 3000
    # dynamics
    dyn_d: int = 192
    dyn_layers: int = 4
    dyn_heads: int = 6
    dyn_steps: int = 6000
    context: int = 4
    ctx_noise: float = 0.05
    # shared
    batch: int = 32
    lr: float = 3e-4
    seed: int = 0
    fps: int = 15


PRESETS: dict[str, dict] = {
    # a couple of minutes on CPU; checks that everything runs
    "smoke": dict(tok_ch=16, tok_steps=60, lam_ch=16, lam_steps=60, dyn_d=64, dyn_layers=2,
                  dyn_heads=4, dyn_steps=80, context=2, batch=8),
    # a first real result
    "small": dict(),
    # the README preset: bigger models, many more steps
    "overnight": dict(tok_ch=64, tok_steps=12000, lam_ch=48, lam_steps=12000, dyn_d=320,
                      dyn_layers=8, dyn_heads=8, dyn_steps=40000, context=6),
}


def make_config(preset: str, **overrides) -> Config:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {', '.join(PRESETS)}")
    values = {**PRESETS[preset], **{k: v for k, v in overrides.items() if v is not None}}
    return Config(**values)


def _schedule(lr: float, steps: int):
    warm = max(1, min(200, steps // 10))
    return optim.join_schedules(
        [optim.linear_schedule(lr * 0.01, lr, warm), optim.cosine_decay(lr, max(1, steps - warm), lr * 0.1)],
        [warm],
    )


def _fit(name: str, model: nn.Module, loss_fn, batch_fn, steps: int, lr: float, log_every: int = 100) -> float:
    opt = optim.AdamW(learning_rate=_schedule(lr, steps), weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(*batch):
        loss, grads = loss_and_grad(model, *batch)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    print(f"\n[{name}] {count_params(model) / 1e6:.2f}M params, {steps} steps")
    start, running = time.time(), None
    for i in range(1, steps + 1):
        loss = step(*batch_fn())
        mx.eval(state, loss)
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if i % log_every == 0 or i == steps:
            rate = i / (time.time() - start)
            eta = (steps - i) / rate
            print(f"  step {i:>6}/{steps}  loss {running:.4f}  {rate:5.1f} it/s  eta {eta / 60:5.1f} min", flush=True)
    return time.time() - start


def _to_float(frames: np.ndarray) -> mx.array:
    return mx.array(frames.astype(np.float32) / 255.0)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) / 255 - b.astype(np.float32) / 255) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def encode_all(tok: Tokenizer, frames: np.ndarray, batch: int = 256) -> np.ndarray:
    flat = frames.reshape(-1, *frames.shape[2:])
    out = []
    for i in range(0, len(flat), batch):
        ids = tok.encode(_to_float(flat[i : i + batch]))
        mx.eval(ids)
        out.append(np.array(ids))
    return np.concatenate(out).reshape(*frames.shape[:2], -1).astype(np.int32)


def infer_all_actions(lam: LatentActionModel, frames: np.ndarray, batch: int = 256) -> np.ndarray:
    e, t = frames.shape[:2]
    f0 = frames[:, :-1].reshape(-1, *frames.shape[2:])
    f1 = frames[:, 1:].reshape(-1, *frames.shape[2:])
    out = []
    for i in range(0, len(f0), batch):
        ids = lam.action_ids(_to_float(f0[i : i + batch]), _to_float(f1[i : i + batch]))
        mx.eval(ids)
        out.append(np.array(ids))
    return np.concatenate(out).reshape(e, t - 1).astype(np.int32)


def action_report(true: np.ndarray, latent: np.ndarray, names: list[str], k: int) -> dict:
    """How well do unsupervised latent actions line up with the real controls?"""
    table = np.zeros((len(names), k), dtype=np.int64)
    np.add.at(table, (true.ravel(), latent.ravel()), 1)
    total = table.sum()
    return {
        "table": table.tolist(),
        # each latent action, read as its most common true action
        "purity": float(table.max(axis=0).sum() / total),
        # each true action, read as its most common latent action
        "consistency": float(table.max(axis=1).sum() / total),
        "bindings": {names[a]: int(table[a].argmax()) for a in range(len(names))},
        "usage": (table.sum(axis=0) / total).round(4).tolist(),
    }


def train(data_path: str, out_dir: str, cfg: Config) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mx.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    ds = Dataset.load(data_path)
    print(ds)
    holdout = max(1, min(8, len(ds.frames) // 8))
    train_frames, test_frames = ds.frames[:-holdout], ds.frames[-holdout:]
    e, t = train_frames.shape[:2]
    if t <= cfg.context + 1:
        raise ValueError(f"Episodes need more than {cfg.context + 1} steps for context={cfg.context}.")
    timings: dict[str, float] = {}

    # 1. Tokenizer ---------------------------------------------------------
    tok = Tokenizer(cfg.tok_ch, cfg.tok_levels)

    def tok_batch():
        idx = rng.integers(0, e, cfg.batch), rng.integers(0, t, cfg.batch)
        return (_to_float(train_frames[idx]),)

    def tok_loss(model, x):
        return mx.mean((model(x) - x) ** 2)

    timings["tokenizer"] = _fit("tokenizer", tok, tok_loss, tok_batch, cfg.tok_steps, cfg.lr)
    tok.save_weights(str(out / "tokenizer.safetensors"))
    sample = test_frames.reshape(-1, *test_frames.shape[2:])[:256]
    recon = np.array(tok.decode(tok.encode(_to_float(sample))) * 255).round().clip(0, 255).astype(np.uint8)
    tok_psnr = _psnr(sample, recon)
    print(f"  reconstruction PSNR on held-out frames: {tok_psnr:.1f} dB")

    # 2. Latent actions -----------------------------------------------------
    lam = LatentActionModel(cfg.lam_ch, cfg.num_actions)

    def lam_batch():
        ei, ti = rng.integers(0, e, cfg.batch), rng.integers(0, t - 1, cfg.batch)
        return _to_float(train_frames[ei, ti]), _to_float(train_frames[ei, ti + 1])

    def lam_loss(model, f0, f1):
        pred = model(f0, f1)
        moved = (mx.abs(f1 - f0).sum(axis=-1, keepdims=True) > 0).astype(mx.float32)
        return mx.mean((1 + 10 * moved) * (pred - f1) ** 2)

    timings["latent_actions"] = _fit("latent actions", lam, lam_loss, lam_batch, cfg.lam_steps, cfg.lr)
    lam.save_weights(str(out / "lam.safetensors"))
    latent = infer_all_actions(lam, train_frames)
    report = None
    if ds.has_actions:
        report = action_report(ds.actions[:-holdout, :-1], latent, ds.action_names, lam.num_actions)
        print(f"  latent action purity {report['purity']:.1%}, consistency {report['consistency']:.1%}")
        print(f"  key bindings: {report['bindings']}")

    # 3. Dynamics -----------------------------------------------------------
    tokens = encode_all(tok, train_frames)
    n = tokens.shape[-1]
    dyn = Dynamics(tok.vocab, lam.num_actions, cfg.context, n, cfg.dyn_d, cfg.dyn_layers, cfg.dyn_heads)

    def dyn_batch():
        ei = rng.integers(0, e, cfg.batch)
        ti = rng.integers(cfg.context, t, cfg.batch)
        window = np.stack([tokens[a, b - cfg.context : b + 1] for a, b in zip(ei, ti)])
        ctx, target = window[:, :-1].copy(), window[:, -1]
        noise = rng.random(ctx.shape) < rng.uniform(0, cfg.ctx_noise, (cfg.batch, 1, 1))
        ctx[noise] = rng.integers(0, tok.vocab, int(noise.sum()))
        ratio = np.cos(rng.random(cfg.batch) * math.pi / 2)
        k = np.maximum(1, np.ceil(ratio * n)).astype(int)
        scores = rng.random((cfg.batch, n))
        mask = scores < np.sort(scores, axis=1)[np.arange(cfg.batch), k - 1][:, None] + 1e-12
        action = latent[ei, ti - 1]
        return mx.array(ctx), mx.array(target), mx.array(mask), mx.array(action)

    def dyn_loss(model, ctx, target, mask, action):
        logits = model(ctx, mx.where(mask, model.mask_id, target), action)
        ce = nn.losses.cross_entropy(logits, target, reduction="none")
        m = mask.astype(mx.float32)
        return (ce * m).sum() / m.sum()

    timings["dynamics"] = _fit("dynamics", dyn, dyn_loss, dyn_batch, cfg.dyn_steps, cfg.lr)
    dyn.save_weights(str(out / "dynamics.safetensors"))

    # Save everything play/dream needs ----------------------------------------
    np.savez_compressed(out / "seeds.npz", frames=test_frames, actions=ds.actions[-holdout:])
    (out / "config.json").write_text(json.dumps({"game": ds.game, "action_names": ds.action_names,
                                                  "config": asdict(cfg)}, indent=2))
    summary = {
        "game": ds.game,
        "tokenizer_psnr_db": round(tok_psnr, 2),
        "params_m": {
            "tokenizer": round(count_params(tok) / 1e6, 2),
            "latent_actions": round(count_params(lam) / 1e6, 2),
            "dynamics": round(count_params(dyn) / 1e6, 2),
        },
        "train_minutes": {k: round(v / 60, 1) for k, v in timings.items()},
        "actions": report,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved run to {out}/  (total {sum(timings.values()) / 60:.1f} min)")
    return summary
