import numpy as np
import torch

from memo2master.degrade.chain import Degrader
from memo2master.models.unet import Restorer
from memo2master.train.losses import MultiResSTFTLoss, compressed_complex_loss

SR = 44_100


def _music_like(seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    tones = sum(np.sin(2 * np.pi * f * t) for f in (110, 220, 330, 440, 880, 1760))
    clicks = (rng.random(len(t)) < 0.0005) * rng.standard_normal(len(t)) * 3
    x = tones / 6 * 0.3 + clicks * 0.1 + rng.standard_normal(len(t)) * 0.01
    return x.astype(np.float32)


def test_degrader_shape_finite_and_seeded():
    deg = Degrader("configs/degrade.yaml")
    clean = _music_like()
    for seed in range(8):
        a, params = deg(clean, np.random.default_rng(seed))
        assert a.shape == clean.shape
        assert np.isfinite(a).all()
    a1, _ = deg(clean, np.random.default_rng(123))
    a2, _ = deg(clean, np.random.default_rng(123))
    np.testing.assert_allclose(a1, a2, atol=1e-6)


def test_degrader_stays_time_aligned():
    """Light chain (no room / codec latency) should keep the signal in sync."""
    deg = Degrader("configs/degrade.yaml")
    clean = _music_like()
    for seed in range(6):
        out, _ = deg(clean, np.random.default_rng(seed))
        corr = np.correlate(out[SR : 2 * SR], clean[SR - 2000 : 2 * SR + 2000], mode="valid")
        lag = int(np.argmax(np.abs(corr))) - 2000
        assert abs(lag) < 1200, f"seed {seed}: lag {lag} samples"


def test_restorer_roundtrip_and_backward():
    model = Restorer(channels=(8, 16, 16, 32, 32), attn_layers=1, attn_heads=2)
    wav = torch.from_numpy(np.stack([_music_like(2.0, s) for s in range(2)]))
    out, spec = model(wav)
    assert out.shape == wav.shape
    # head is zero-initialized, so an untrained model is ~identity
    assert torch.allclose(out, wav, atol=1e-3)
    loss = MultiResSTFTLoss()(out, wav * 0.9) + compressed_complex_loss(spec, model.stft(wav * 0.9))
    loss.backward()
    assert torch.isfinite(loss)
