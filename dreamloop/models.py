"""The three networks of a Genie-style world model, written in MLX.

1. Tokenizer          64x64 frame  ->  8x8 grid of discrete tokens (FSQ)
2. LatentActionModel  (frame t, frame t+1)  ->  one of K latent actions
3. Dynamics           past tokens + latent action  ->  next frame's tokens

Images are channels-last (NHWC), which is what MLX convolutions expect.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al., 2023). No codebook, no collapse."""

    def __init__(self, levels: list[int]):
        super().__init__()
        self.levels = [int(v) for v in levels]
        self.dim = len(self.levels)
        self.codebook_size = int(np.prod(self.levels))
        self.basis = [int(b) for b in np.concatenate([[1], np.cumprod(self.levels[:-1])])]

    def _consts(self):
        L = mx.array(self.levels, dtype=mx.float32)
        half_width = mx.array([v // 2 for v in self.levels], dtype=mx.float32)
        return L, half_width

    def __call__(self, z: mx.array) -> mx.array:
        L, half_width = self._consts()
        half_l = (L - 1) * (1 + 1e-3) / 2
        offset = mx.where(L % 2 == 0, 0.5, 0.0)
        shift = mx.arctanh(offset / half_l)
        bounded = mx.tanh(z + shift) * half_l - offset
        quantized = bounded + mx.stop_gradient(mx.round(bounded) - bounded)
        return quantized / half_width

    def codes_to_indices(self, codes: mx.array) -> mx.array:
        _, half_width = self._consts()
        digits = mx.round(codes * half_width + half_width).astype(mx.int32)
        return (digits * mx.array(self.basis, dtype=mx.int32)).sum(axis=-1)

    def indices_to_codes(self, idx: mx.array) -> mx.array:
        L, half_width = self._consts()
        digits = (idx[..., None] // mx.array(self.basis, dtype=mx.int32)) % L.astype(mx.int32)
        return (digits.astype(mx.float32) - half_width) / half_width


def _upsample(x: mx.array) -> mx.array:
    return mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch, pytorch_compatible=True)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch, pytorch_compatible=True)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.c1(nn.silu(self.n1(x)))
        h = self.c2(nn.silu(self.n2(h)))
        return x + h


class Encoder(nn.Module):
    """64x64 -> 8x8 with three stride-2 convolutions."""

    def __init__(self, in_ch: int, ch: int, out_dim: int, n_res: int = 1):
        super().__init__()
        self.down = [
            nn.Conv2d(in_ch, ch, 4, stride=2, padding=1),
            nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),
            nn.Conv2d(ch * 2, ch * 2, 4, stride=2, padding=1),
        ]
        self.res = [ResBlock(ch * 2) for _ in range(n_res)]
        self.out = nn.Conv2d(ch * 2, out_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        for conv in self.down:
            x = nn.silu(conv(x))
        for block in self.res:
            x = block(x)
        return self.out(x)


class Decoder(nn.Module):
    """8x8 -> 64x64 with nearest upsampling + convolutions (no checkerboards)."""

    def __init__(self, in_dim: int, ch: int, out_ch: int = 3, n_res: int = 1):
        super().__init__()
        self.inp = nn.Conv2d(in_dim, ch * 2, 3, padding=1)
        self.res = [ResBlock(ch * 2) for _ in range(n_res)]
        self.up = [
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
            nn.Conv2d(ch * 2, ch, 3, padding=1),
            nn.Conv2d(ch, ch, 3, padding=1),
        ]
        self.out = nn.Conv2d(ch, out_ch, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.inp(x)
        for block in self.res:
            x = block(x)
        for conv in self.up:
            x = nn.silu(conv(_upsample(x)))
        return mx.sigmoid(self.out(x))


class Tokenizer(nn.Module):
    def __init__(self, ch: int = 64, levels: list[int] = (8, 5, 5, 5), n_res: int = 1):
        super().__init__()
        self.fsq = FSQ(list(levels))
        self.enc = Encoder(3, ch, self.fsq.dim, n_res)
        self.dec = Decoder(self.fsq.dim, ch, 3, n_res)

    @property
    def vocab(self) -> int:
        return self.fsq.codebook_size

    def __call__(self, x: mx.array) -> mx.array:
        return self.dec(self.fsq(self.enc(x * 2 - 1)))

    def encode(self, x: mx.array) -> mx.array:
        """[B, 64, 64, 3] floats in [0, 1] -> [B, 64] token ids."""
        idx = self.fsq.codes_to_indices(self.fsq(self.enc(x * 2 - 1)))
        return idx.reshape(idx.shape[0], -1)

    def decode(self, idx: mx.array) -> mx.array:
        side = int(math.isqrt(idx.shape[-1]))
        codes = self.fsq.indices_to_codes(idx.reshape(idx.shape[0], side, side))
        return self.dec(codes)


class LatentActionModel(nn.Module):
    """Learns K discrete 'actions' that best explain how frame t becomes t+1."""

    def __init__(self, ch: int = 48, num_actions: int = 8, n_res: int = 1):
        super().__init__()
        self.fsq = FSQ([num_actions])
        self.pair_enc = Encoder(6, ch, ch * 2, n_res)
        self.to_action = nn.Linear(ch * 2, 1)
        self.frame_enc = Encoder(3, ch, ch * 2, n_res)
        self.action_emb = nn.Linear(1, ch * 2)
        self.dec = Decoder(ch * 2, ch, 3, n_res)

    @property
    def num_actions(self) -> int:
        return self.fsq.codebook_size

    def infer(self, f0: mx.array, f1: mx.array) -> mx.array:
        h = self.pair_enc(mx.concatenate([f0, f1], axis=-1) * 2 - 1).mean(axis=(1, 2))
        return self.fsq(self.to_action(h))

    def __call__(self, f0: mx.array, f1: mx.array) -> mx.array:
        action = self.infer(f0, f1)
        h = self.frame_enc(f0 * 2 - 1) + self.action_emb(action)[:, None, None, :]
        return self.dec(h)

    def action_ids(self, f0: mx.array, f1: mx.array) -> mx.array:
        return self.fsq.codes_to_indices(self.infer(f0, f1))


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiHeadAttention(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.ln1(x)
        x = x + self.attn(h, h, h)
        return x + self.fc2(nn.gelu(self.fc1(self.ln2(x))))


class Dynamics(nn.Module):
    """Masked-token transformer: sees C past frames, fills in the next one."""

    def __init__(
        self,
        vocab: int,
        num_actions: int,
        context: int = 4,
        tokens_per_frame: int = 64,
        d: int = 256,
        layers: int = 6,
        heads: int = 8,
    ):
        super().__init__()
        self.vocab, self.mask_id = vocab, vocab
        self.context, self.n = context, tokens_per_frame
        self.tok = nn.Embedding(vocab + 1, d)
        self.space = nn.Embedding(tokens_per_frame, d)
        self.time = nn.Embedding(context + 1, d)
        self.act = nn.Embedding(num_actions, d)
        self.blocks = [Block(d, heads) for _ in range(layers)]
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def __call__(self, ctx: mx.array, target: mx.array, action: mx.array) -> mx.array:
        """ctx [B, C, N] ids, target [B, N] ids (mask_id = unknown), action [B] -> logits [B, N, V]."""
        b = ctx.shape[0]
        x = mx.concatenate([ctx, target[:, None, :]], axis=1)
        h = self.tok(x)
        h = h + self.space(mx.arange(self.n))[None, None] + self.time(mx.arange(self.context + 1))[None, :, None]
        a = self.act(action)[:, None, None, :]
        h = mx.concatenate([h[:, :-1], h[:, -1:] + a], axis=1)
        h = h.reshape(b, (self.context + 1) * self.n, -1)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln(h[:, -self.n :]))

    def generate(self, ctx: mx.array, action: mx.array, steps: int = 4, temperature: float = 0.0) -> mx.array:
        """MaskGIT decoding: commit the most confident tokens first."""
        b, n = ctx.shape[0], self.n
        target = mx.full((b, n), self.mask_id, dtype=mx.int32)
        known = mx.zeros((b, n), dtype=mx.bool_)
        for s in range(steps):
            logits = self(ctx, target, action)
            if temperature > 0:
                sampled = mx.random.categorical(logits / temperature).astype(mx.int32)
            else:
                sampled = mx.argmax(logits, axis=-1).astype(mx.int32)
            probs = mx.softmax(logits, axis=-1)
            conf = mx.take_along_axis(probs, sampled[..., None], axis=-1).squeeze(-1)
            sampled = mx.where(known, target, sampled)
            remaining = int(n * math.cos(math.pi / 2 * (s + 1) / steps))
            if remaining == 0:
                return sampled
            conf = mx.where(known, mx.array(float("inf")), conf)
            ranks = mx.argsort(mx.argsort(conf, axis=-1), axis=-1)
            still_masked = ranks < remaining
            target = mx.where(still_masked, self.mask_id, sampled)
            known = mx.logical_not(still_masked)
        return target


def count_params(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.parameters()))
