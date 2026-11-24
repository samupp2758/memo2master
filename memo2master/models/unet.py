"""Spectrogram U-Net for memo -> master restoration.

Design choices (see README for the why):
  * Input/output are compressed complex spectrograms (real+imag), so phase is
    modeled instead of reusing the noisy phase.
  * Output = input + predicted residual (mapping, not a multiplicative mask),
    so the network can *create* energy where the codec removed it
    (bandwidth extension). A pure mask can't: anything times zero is zero.
  * Frequency is downsampled at every level; time only at the first few, which
    keeps temporal resolution for transients.
  * A small transformer over time at the bottleneck adds long-range context
    (sustained notes, room tails).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from memo2master.models.stft import CompressedSTFT


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.GELU(),
    )


def _resample_kw(time_stride: int) -> dict:
    """Halve frequency always; halve time only when time_stride == 2."""
    kt, pt = (4, 1) if time_stride > 1 else (1, 0)
    return dict(kernel_size=(4, kt), stride=(2, time_stride), padding=(1, pt))


class TimeAttention(nn.Module):
    """Self-attention along time, independently per frequency row."""

    def __init__(self, ch: int, heads: int, layers: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(ch, heads, ch * 2, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        y = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
        y = self.enc(y)
        return y.reshape(b, f, t, c).permute(0, 3, 1, 2)


class SpecUNet(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256, 256), time_strides=(2, 2, 2, 1, 1),
                 attn_layers: int = 2, attn_heads: int = 4):
        super().__init__()
        assert len(channels) == len(time_strides)
        self.time_factor = 1
        for s in time_strides:
            self.time_factor *= s
        self.freq_factor = 2 ** len(channels)

        self.stem = conv_block(2, channels[0])
        self.downs, self.enc = nn.ModuleList(), nn.ModuleList()
        cin = channels[0]
        for cout, ts in zip(channels, time_strides):
            self.downs.append(nn.Conv2d(cin, cout, **_resample_kw(ts)))
            self.enc.append(conv_block(cout, cout))
            cin = cout

        self.bottleneck = TimeAttention(cin, attn_heads, attn_layers) if attn_layers else nn.Identity()

        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        skips = [channels[0]] + list(channels[:-1])
        for skip_ch, ts in zip(reversed(skips), reversed(time_strides)):
            self.ups.append(nn.ConvTranspose2d(cin, skip_ch, **_resample_kw(ts)))
            self.dec.append(conv_block(skip_ch * 2, skip_ch))
            cin = skip_ch
        self.head = nn.Conv2d(cin, 2, 1)
        nn.init.zeros_(self.head.weight)  # start as identity: output == input
        nn.init.zeros_(self.head.bias)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:  # (B, 2, F, T)
        t = spec.shape[-1]
        pad_t = (-t) % self.time_factor
        x = F.pad(spec, (0, pad_t))

        h = self.stem(x)
        skips = [h]
        for down, enc in zip(self.downs, self.enc):
            h = enc(down(h))
            skips.append(h)
        skips.pop()  # deepest output is the bottleneck input, not a skip
        h = self.bottleneck(h)
        for up, dec in zip(self.ups, self.dec):
            h = up(h)
            h = dec(torch.cat([h, skips.pop()], dim=1))
        out = x + self.head(h)  # residual mapping
        return out[..., :t]


class Restorer(nn.Module):
    """Waveform in -> waveform out, wrapping STFT + SpecUNet."""

    def __init__(self, n_fft: int = 2048, hop: int = 512, compress: float = 0.3, **unet_kwargs):
        super().__init__()
        self.stft = CompressedSTFT(n_fft, hop, compress)
        self.unet = SpecUNet(**unet_kwargs)
        assert (n_fft // 2) % self.unet.freq_factor == 0, "n_fft/2 must divide by 2**levels"

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec_in = self.stft(wav)
        spec_out = self.unet(spec_in)
        return self.stft.inverse(spec_out, wav.shape[-1]), spec_out
