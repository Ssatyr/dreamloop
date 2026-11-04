# dreamloop

**Record a simple game. Train a world model overnight on your MacBook. Then play the dream.**

Every frame you see in `dreamloop play` is imagined by a neural network. The real game isn't running.

> Draft: every [bracketed] value below is a placeholder until the training runs are done.

![real game vs. dreamed by dreamloop]([dream.gif])

| Training time on M1 Pro | World model size | Playback | Gameplay used |
| --- | --- | --- | --- |
| [__] h | [__]M params | [__] fps | [__] min |

## How it works

1. **Record.** Gameplay video from a built-in game, or (experimental) a screen region of any simple game. No action labels are needed.
2. **Learn.** Three small networks in [MLX](https://github.com/ml-explore/mlx):
   - a **tokenizer** that turns each 64×64 frame into an 8×8 grid of discrete tokens (FSQ),
   - a **latent action model** that invents 8 discrete "actions" that best explain how one frame becomes the next,
   - a **dynamics transformer** that predicts the next frame's tokens from the last few frames plus an action (MaskGIT decoding).
3. **Play.** Your keys pick latent actions. The network imagines every next frame, live in your browser.

## Quick start

Apple Silicon Mac with 16 GB+ of memory, Python 3.10+.

```bash
git clone https://github.com/Ssatyr/dreamloop
cd dreamloop
pip install -e ".[dev]"

dreamloop demo --game platformer --preset smoke   # a few minutes: checks everything runs
dreamloop demo --game platformer --preset small   # a first real result
dreamloop play --run runs/platformer-small        # opens the dream in your browser
```

Step by step:

```bash
dreamloop data  --game paddle --episodes 120 --steps 160
dreamloop train --data data/paddle.npz --out runs/paddle --preset overnight
dreamloop eval  --run runs/paddle      # how long the dream stays close to the real game
dreamloop dream --run runs/paddle      # side-by-side GIF for your README
dreamloop play  --run runs/paddle
```

Built-in games: `platformer`, `paddle`, `snake`.

## Do the invented actions match the real controls?

The latent action model never sees which key was pressed. Built-in games record the true actions anyway, so we can check:

| Game | Purity | Consistency |
| --- | --- | --- |
| platformer | [__]% | [__]% |
| paddle | [__]% | [__]% |
| snake | [__]% | [__]% |

*Purity:* how often a latent action means one real action. *Consistency:* how often a real action gets the same latent action. Neither can reach 100%, because some moves look identical on screen (walking into a wall looks like standing still).

## When dreams go wrong

World models drift, forget and improvise. The failures show what the network actually learned.

| Long rollouts | Off-screen memory | Rare actions |
| --- | --- | --- |
| [GIF] seconds until drift: [__] | [GIF] | [GIF] |

## Record your own game (experimental)

```bash
pip install -e ".[record]"
dreamloop record --region 100,100,480,480 --minutes 20
dreamloop train --data data/recorded.npz --out runs/mygame --preset small
```

On macOS, allow your terminal under System Settings › Privacy & Security › Screen Recording.

## Why

Interactive world models like DeepMind's Genie are among the most exciting ideas in AI right now, and most of them are out of reach. dreamloop is the pocket-sized version: small enough to train on a laptop and read in an afternoon.

## Built on

- [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391) (Bruce et al., 2024)
- [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505) (Mentzer et al., 2023)
- [MaskGIT](https://arxiv.org/abs/2202.04200) (Chang et al., 2022)
- [TinyWorlds](https://github.com/AlmondGod/tinyworlds)

MIT license.
