import mlx.core as mx
import numpy as np

from dreamloop import games
from dreamloop.data import generate
from dreamloop.models import FSQ, Dynamics, LatentActionModel, Tokenizer


def test_games_render_64x64():
    for name in games.GAMES:
        g = games.make(name, seed=1)
        for _ in range(50):
            g.step(g.policy())
        frame = g.render()
        assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8


def test_dataset_shapes():
    ds = generate("paddle", episodes=2, steps=10)
    assert ds.frames.shape == (2, 10, 64, 64, 3)
    assert ds.has_actions and (ds.actions[:, -1] == -1).all()


def test_fsq_roundtrip():
    fsq = FSQ([8, 5, 5, 5])
    codes = fsq(mx.random.normal((16, 4)) * 3)
    idx = fsq.codes_to_indices(codes)
    assert int(idx.min()) >= 0 and int(idx.max()) < fsq.codebook_size
    assert mx.allclose(fsq.indices_to_codes(idx), codes)


def test_model_shapes():
    x = mx.random.uniform(shape=(2, 64, 64, 3))
    tok = Tokenizer(16)
    ids = tok.encode(x)
    assert ids.shape == (2, 64)
    assert tok.decode(ids).shape == (2, 64, 64, 3)
    lam = LatentActionModel(16, num_actions=8)
    assert lam(x, x).shape == (2, 64, 64, 3)
    assert lam.action_ids(x, x).shape == (2,)
    dyn = Dynamics(tok.vocab, 8, context=2, d=32, layers=1, heads=4)
    ctx = mx.zeros((2, 2, 64), dtype=mx.int32)
    out = dyn.generate(ctx, mx.array([0, 3]), steps=3)
    assert out.shape == (2, 64) and int(out.max()) < tok.vocab
