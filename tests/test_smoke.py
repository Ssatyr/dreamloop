import mlx.core as mx
import numpy as np

from dreamloop import games
from dreamloop.data import generate
from dreamloop.models import FSQ, Dynamics, LatentActionModel, Tokenizer
from dreamloop.rollout import moving_psnr


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
    lam = LatentActionModel(16, levels=[3, 3])
    assert lam.num_actions == 9
    assert lam(x, x).shape == (2, 64, 64, 3)
    assert lam.action_ids(x, x).shape == (2,)
    dyn = Dynamics(tok.vocab, 8, context=2, d=32, layers=1, heads=4)
    ctx = mx.zeros((2, 2, 64), dtype=mx.int32)
    out = dyn.generate(ctx, mx.array([0, 3]), steps=3)
    assert out.shape == (2, 64) and int(out.max()) < tok.vocab


def test_moving_psnr_ignores_static_background():
    ref = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    real = ref.copy()
    real[0, 10:14, 20:24] = 255  # a 4x4 sprite appears; the other 99.6% of pixels stay put
    assert moving_psnr(real, ref, ref) < 1  # a frozen picture gets every moving pixel wrong
    assert moving_psnr(real, real, ref) == 99.0  # a perfect prediction


def test_bind_keys_prefers_specific_codes():
    from dreamloop.train import bind_keys

    # 'left' often does nothing (blocked by a wall), so its most common code is the
    # "nothing happened" code 0 that noop also uses; it should still bind to code 1.
    table = [[60, 0], [52, 48]]  # rows: noop, left; columns: codes 0, 1
    assert bind_keys(table, ["noop", "left"]) == {"noop": 0, "left": 1}


def test_idle_repeats_last_action_when_noop_has_no_code_of_its_own():
    from dreamloop.play import idle_action

    assert idle_action({"noop": 4, "left": 2, "right": 6}) == 4  # platformer: "nothing happened"
    assert idle_action({"noop": 2, "up": 8, "right": 2}) is None  # snake: noop = "keep going"
