"""Losses for the regression baseline.

Stage 2 (later): add HiFi-GAN-style multi-period / multi-scale discriminators
plus feature-matching loss to escape the "muffled average" of pure regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResSTFTLoss(nn.Module):
    """Spectral convergence + log-magnitude L1, averaged over several resolutions."""

    def __init__(self, resolutions=((512, 128), (1024, 256), (2048, 512), (4096, 1024))):
        super().__init__()
        self.resolutions = resolutions
        for n_fft, _ in resolutions:
            self.register_buffer(f"win{n_fft}", torch.hann_window(n_fft), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_fft, hop in self.resolutions:
            w = getattr(self, f"win{n_fft}")
            P = torch.stft(pred.float(), n_fft, hop, window=w, return_complex=True).abs()
            T = torch.stft(target.float(), n_fft, hop, window=w, return_complex=True).abs()
            sc = torch.norm(T - P, p="fro") / (torch.norm(T, p="fro") + 1e-7)
            mag = F.l1_loss(torch.log(P + 1e-5), torch.log(T + 1e-5))
            loss = loss + sc + mag
        return loss / len(self.resolutions)


def compressed_complex_loss(spec_pred: torch.Tensor, spec_target: torch.Tensor,
                            mag_weight: float = 0.7) -> torch.Tensor:
    """MSE on compressed complex spectrogram + on its magnitude (phase-aware)."""
    cplx = F.mse_loss(spec_pred.float(), spec_target.float())
    mag = F.mse_loss(spec_pred.float().norm(dim=1), spec_target.float().norm(dim=1))
    return (1 - mag_weight) * cplx + mag_weight * mag
