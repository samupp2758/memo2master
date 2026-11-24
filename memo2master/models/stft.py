"""STFT <-> compressed complex spectrogram.

Representation: (B, 2, F, T) with channels = real, imag of |X|^c * e^{j*phase}.
Power-law compression (c ~ 0.3) evens out the huge dynamic range of music
spectra so quiet high-frequency detail is not ignored by the loss.
The Nyquist bin is dropped so F is a power of two (n_fft/2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompressedSTFT(nn.Module):
    def __init__(self, n_fft: int = 2048, hop: int = 512, compress: float = 0.3):
        super().__init__()
        self.n_fft, self.hop, self.c = n_fft, hop, compress
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # (B, N) -> (B, 2, F, T)
        X = torch.stft(wav, self.n_fft, self.hop, window=self.window, return_complex=True)
        X = X[:, :-1]  # drop Nyquist
        mag = X.abs().clamp_min(1e-8)
        Xc = mag.pow(self.c) * (X / mag)
        return torch.stack([Xc.real, Xc.imag], dim=1)

    def inverse(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        Xc = torch.complex(spec[:, 0].float(), spec[:, 1].float())
        mag = Xc.abs().clamp_min(1e-8)
        X = mag.pow(1.0 / self.c) * (Xc / mag)
        X = torch.nn.functional.pad(X, (0, 0, 0, 1))  # restore Nyquist (zero)
        return torch.istft(X, self.n_fft, self.hop, window=self.window, length=length)
