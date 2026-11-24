"""Restore a full-length voice memo (any length, any format ffmpeg reads).

    python -m memo2master.infer.run --ckpt runs/baseline/last.pt --input memo.m4a --output memo_master.wav

The file is processed in overlapping chunks and crossfaded back together.
One global input gain is used for the whole file (no per-chunk normalization),
so there is no level pumping between chunks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from memo2master.audio import SAMPLE_RATE, load_any, normalize_loudness, save
from memo2master.data.dataset import INPUT_RMS_DB
from memo2master.models.unet import Restorer
from memo2master.train.train import pick_device


@torch.no_grad()
def restore(model: Restorer, audio: np.ndarray, device: torch.device,
            chunk_s: float, overlap_s: float, batch: int = 4) -> np.ndarray:
    chunk, overlap = int(chunk_s * SAMPLE_RATE), int(overlap_s * SAMPLE_RATE)
    hop = chunk - overlap
    n = len(audio)
    padded = np.pad(audio, (0, (hop - (n - chunk) % hop) % hop if n > chunk else chunk - n))
    starts = list(range(0, len(padded) - chunk + 1, hop))

    fade = np.ones(chunk, dtype=np.float32)
    ramp = np.sin(np.linspace(0, np.pi / 2, overlap, dtype=np.float32)) ** 2
    fade[:overlap], fade[-overlap:] = ramp, ramp[::-1]

    out = np.zeros(len(padded), dtype=np.float32)
    norm = np.zeros(len(padded), dtype=np.float32)
    for i in tqdm(range(0, len(starts), batch), desc="restore"):
        idx = starts[i : i + batch]
        x = torch.from_numpy(np.stack([padded[s : s + chunk] for s in idx])).to(device)
        y = model(x)[0].float().cpu().numpy()
        for s, yi in zip(idx, y):
            out[s : s + chunk] += yi * fade
            norm[s : s + chunk] += fade
    return (out / np.maximum(norm, 1e-6))[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--overlap-seconds", type=float, default=1.0)
    ap.add_argument("--target-lufs", type=float, default=-14.0)
    args = ap.parse_args()

    device = pick_device()
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["config"]
    model = Restorer(**cfg["model"]).to(device).eval()
    model.load_state_dict(ck["model"])

    audio = load_any(args.input)
    rms = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9)
    audio = audio * 10 ** ((INPUT_RMS_DB - rms) / 20)

    restored = restore(model, audio, device, cfg["data"]["chunk_seconds"], args.overlap_seconds)
    restored = normalize_loudness(restored, SAMPLE_RATE, args.target_lufs)
    save(args.output, restored)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
