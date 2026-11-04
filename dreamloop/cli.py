"""dreamloop command line: data, train, dream, eval, play, record, demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import games


def _data(args) -> None:
    from .data import generate

    ds = generate(args.game, args.episodes, args.steps, args.seed)
    out = args.out or f"data/{args.game}.npz"
    ds.save(out)
    print(f"Saved {ds} to {out}")


def _train(args) -> None:
    from .train import make_config, train

    cfg = make_config(args.preset, tok_steps=args.tok_steps, lam_steps=args.lam_steps,
                      dyn_steps=args.dyn_steps, batch=args.batch, seed=args.seed)
    train(args.data, args.out, cfg)


def _dream(args) -> None:
    from .rollout import WorldModel, make_gif

    wm = WorldModel(args.run)
    out = make_gif(wm, args.out or Path(args.run) / "dream.gif", episode=args.episode,
                   steps=args.steps, temperature=args.temperature)
    print(f"Saved {out}")


def _eval(args) -> None:
    from .rollout import WorldModel, evaluate_drift

    wm = WorldModel(args.run)
    drift = evaluate_drift(wm, horizon=args.horizon, threshold_db=args.threshold)
    metrics_path = Path(args.run) / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["drift"] = drift
    metrics_path.write_text(json.dumps(metrics, indent=2))
    curve = drift["psnr_by_step"]
    marks = [h for h in (1, 4, 16, 32, 64) if h <= len(curve)]
    print("PSNR vs. the real game: " + ", ".join(f"step {h}: {curve[h - 1]:.1f} dB" for h in marks))
    print(f"Stays above {args.threshold:.0f} dB for {drift['steps_until_drift']} steps "
          f"({drift['seconds_until_drift']} s). Saved to {metrics_path}")


def _play(args) -> None:
    from .play import serve

    serve(args.run, port=args.port, temperature=args.temperature, open_browser=not args.no_browser)


def _record(args) -> None:
    from .record import record

    record(args.region, args.minutes, args.out, fps=args.fps, name=args.name)


def _demo(args) -> None:
    from .data import generate
    from .rollout import WorldModel, evaluate_drift, make_gif
    from .train import make_config, train

    data_path = f"data/{args.game}.npz"
    run = args.out or f"runs/{args.game}-{args.preset}"
    episodes, steps = (12, 48) if args.preset == "smoke" else (args.episodes, args.steps)
    generate(args.game, episodes, steps).save(data_path)
    train(data_path, run, make_config(args.preset))
    wm = WorldModel(run)
    drift = evaluate_drift(wm)
    metrics_path = Path(run) / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["drift"] = drift
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Dream stays above {drift['threshold_db']:.0f} dB PSNR for {drift['steps_until_drift']} steps "
          f"({drift['seconds_until_drift']} s)")
    print(f"Saved {make_gif(wm, Path(run) / 'dream.gif')}")
    print(f"\nNext: dreamloop play --run {run}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dreamloop", description="Train a playable world model on your Mac.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data", help="generate gameplay from a built-in game")
    s.add_argument("--game", choices=list(games.GAMES), default="platformer")
    s.add_argument("--episodes", type=int, default=120)
    s.add_argument("--steps", type=int, default=160)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(fn=_data)

    s = sub.add_parser("train", help="train tokenizer, latent actions and dynamics")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--preset", default="small", choices=["smoke", "small", "overnight"])
    for name in ("tok-steps", "lam-steps", "dyn-steps", "batch"):
        s.add_argument(f"--{name}", type=int)
    s.add_argument("--seed", type=int)
    s.set_defaults(fn=_train)

    s = sub.add_parser("dream", help="save a side-by-side GIF: real vs. dreamed")
    s.add_argument("--run", required=True)
    s.add_argument("--out")
    s.add_argument("--episode", type=int, default=0)
    s.add_argument("--steps", type=int, default=90)
    s.add_argument("--temperature", type=float, default=0.0)
    s.set_defaults(fn=_dream)

    s = sub.add_parser("eval", help="measure how long dreams stay close to the real game")
    s.add_argument("--run", required=True)
    s.add_argument("--horizon", type=int, default=64)
    s.add_argument("--threshold", type=float, default=20.0)
    s.set_defaults(fn=_eval)

    s = sub.add_parser("play", help="play the dream in your browser")
    s.add_argument("--run", required=True)
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=_play)

    s = sub.add_parser("record", help="(experimental) record a screen region as training data")
    s.add_argument("--region", required=True, help="x,y,width,height in screen points")
    s.add_argument("--minutes", type=float, default=20)
    s.add_argument("--fps", type=int, default=15)
    s.add_argument("--name", default="recorded")
    s.add_argument("--out", default="data/recorded.npz")
    s.set_defaults(fn=_record)

    s = sub.add_parser("demo", help="data + train + eval + GIF in one go")
    s.add_argument("--game", choices=list(games.GAMES), default="platformer")
    s.add_argument("--preset", default="small", choices=["smoke", "small", "overnight"])
    s.add_argument("--episodes", type=int, default=120)
    s.add_argument("--steps", type=int, default=160)
    s.add_argument("--out")
    s.set_defaults(fn=_demo)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
