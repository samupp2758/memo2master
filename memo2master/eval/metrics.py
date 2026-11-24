"""Objective metrics for synthetic (paired) evaluation.

For real voice memos there is no clean reference; plan on:
  * Frechet Audio Distance (FAD) between restored memos and the clean CCM set
  * Blind listening tests (MUSHRA-style) with musicians
"""

from __future__ import annotations

import numpy as np


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (higher is better)."""
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-9)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target**2) + 1e-9) / (np.sum(noise**2) + 1e-9)))


def log_spectral_distance(estimate: np.ndarray, reference: np.ndarray,
                          n_fft: int = 2048, hop: int = 512) -> float:
    """Mean log-spectral distance in dB (lower is better). Sensitive to missing highs."""
    def mag(x):
        frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
        return np.abs(np.fft.rfft(frames, axis=-1)) ** 2 + 1e-10

    E, R = mag(estimate), mag(reference)
    return float(np.mean(np.sqrt(np.mean((10 * np.log10(R / E)) ** 2, axis=-1))))
