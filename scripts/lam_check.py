"""Check whether a trained latent action model (LAM) has collapsed, and why.

Three questions:
  1. Encoder: before rounding, does each dial give different keys different numbers?
  2. Decoder: if we force each action code, does its predicted frame change, and how much?
  3. Keys -> codes: for each real key, which codes did the LAM pick?

Usage:
    python scripts/lam_check.py --run runs/platformer-small
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from dreamloop.data import Dataset
from dreamloop.models import LatentActionModel
from dreamloop.train import Config, make_config


def load_lam(run: Path, preset: str) -> LatentActionModel:
    cfg_path = run / "config.json"
    if cfg_path.exists():
        cfg = Config(**json.loads(cfg_path.read_text())["config"])
    else:  # config.json is only written when the whole run finishes
        cfg = make_config(preset)
    lam = LatentActionModel(cfg.lam_ch, cfg.lam_levels)
    lam.load_weights(str(run / "lam.safetensors"))
    lam.eval()
    return lam


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", default=None, help="defaults to data/<game>.npz, game read from the run's config.json")
    p.add_argument("--preset", default="small", help="used only if the run has no config.json yet")
    p.add_argument("--n", type=int, default=256)
    args = p.parse_args()

    run = Path(args.run)
    cfg_path = run / "config.json"
    game = json.loads(cfg_path.read_text())["game"] if cfg_path.exists() else run.name.split("-")[0]
    data = args.data or f"data/{game}.npz"
    lam = load_lam(run, args.preset)
    ds = Dataset.load(data)
    k = lam.num_actions

    rng = np.random.default_rng(0)
    e, t = ds.frames.shape[:2]
    ei, ti = rng.integers(0, e, args.n), rng.integers(0, t - 1, args.n)
    f0 = mx.array(ds.frames[ei, ti].astype(np.float32) / 255)
    f1 = mx.array(ds.frames[ei, ti + 1].astype(np.float32) / 255)
    true = ds.actions[ei, ti]

    # 1. Encoder: the numbers (one per dial) it squeezes each frame pair into.
    h = lam.pair_enc(mx.concatenate([f0, f1], axis=-1) * 2 - 1).mean(axis=(1, 2))
    raw = np.array(lam.action_norm(lam.to_action(h)))  # [n, dials]
    # Same squashing FSQ does, so each number sits on its dial's ruler (e.g. -1 .. 1 for 3 levels);
    # FSQ then rounds it to the nearest whole number.
    levels = np.array(lam.fsq.levels)
    half_l = (levels - 1) * (1 + 1e-3) / 2
    offset = np.where(levels % 2 == 0, 0.5, 0.0)
    ruler = np.tanh(raw + np.arctanh(offset / half_l)) * half_l - offset
    ids = np.array(lam.action_ids(f0, f1)).ravel()

    dials = range(len(levels))
    rulers = ", ".join(f"dial {j + 1}: {-(l // 2)} .. {-(l // 2) + l - 1}" for j, l in zip(dials, levels))
    print(f"1. ENCODER  (each dial's number before rounding; {rulers})")
    print("            " + "".join(f"{'dial ' + str(j + 1):>16}" for j in dials))
    print("   spread   " + "".join(f"{'std ' + format(ruler[:, j].std(), '.3f'):>16}" for j in dials))
    for a, name in enumerate(ds.action_names):
        sel = ruler[true == a]
        if len(sel):
            cells = "".join(f"{format(sel[:, j].mean(), '+.2f') + ' ± ' + format(sel[:, j].std(), '.2f'):>16}" for j in dials)
            print(f"   {name:>6}  {cells}   ({len(sel)} pairs)")
    counts = np.bincount(ids, minlength=k)
    print(f"   codes used: {dict((c, int(n)) for c, n in enumerate(counts) if n)}")

    # 2. Decoder: same first frame, force each code, measure how much the output moves.
    b = min(64, args.n)
    h0 = lam.frame_enc(f0[:b] * 2 - 1)
    preds = []
    for c in range(k):
        code = lam.fsq.indices_to_codes(mx.full((b,), c, dtype=mx.int32))
        preds.append(lam.dec(h0 + lam.action_emb(code)[:, None, None, :]))
    preds = mx.stack(preds)
    spread = np.array((preds.max(axis=0) - preds.min(axis=0)).max(axis=-1)) * 255
    moved = np.array((mx.abs(f1[:b] - f0[:b]).sum(axis=-1) > 0).astype(mx.float32))

    print(f"\n2. DECODER  (same frame, try all {k} codes)")
    print(f"   biggest pixel change between codes: {spread.max():.1f} / 255")
    print(f"   pixels that change by more than 10/255: {(spread > 10).mean():.2%}")
    print(f"   for scale, pixels that really change between frames: {moved.mean():.2%}")

    # 3. Keys -> codes, from the full training set (saved in metrics.json when the run finished).
    metrics_path = run / "metrics.json"
    report = json.loads(metrics_path.read_text()).get("actions") if metrics_path.exists() else None
    if report:
        table = np.array(report["table"])
        print("\n3. KEYS -> CODES  (share of each key's presses that got each code)")
        print("     code:" + "".join(f"{c:>5}" for c in range(table.shape[1])))
        for name, row in zip(ds.action_names, table):
            print(f"   {name:>6}:" + "".join(f"{v / max(row.sum(), 1):>5.0%}" for v in row))
        print(f"   purity {report['purity']:.1%}, consistency {report['consistency']:.1%}, "
              f"bindings {report['bindings']}")


if __name__ == "__main__":
    main()
