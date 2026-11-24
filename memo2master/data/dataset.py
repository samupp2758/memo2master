"""Paired (degraded, clean) chunks, degraded on the fly in DataLoader workers.

Only clean audio lives on disk; every time a chunk is drawn it gets a fresh
random degradation, so the model never sees the same damage twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from memo2master.audio import SAMPLE_RATE
from memo2master.degrade.chain import Degrader

INPUT_RMS_DB = -23.0  # input chunks are scaled to this; target gets the same gain


def scale_pair(degraded: np.ndarray, clean: np.ndarray, rms_db: float = INPUT_RMS_DB):
    """Apply one gain to both so the degraded input sits at a fixed RMS.

    Using the same gain keeps the level relationship, so the model learns to
    undo AGC / limiting rather than having it normalized away.
    """
    cur = 20 * np.log10(np.sqrt(np.mean(degraded**2)) + 1e-9)
    gain = 10 ** (np.clip(rms_db - cur, -30, 30) / 20)
    return degraded * gain, clean * gain


class PairedChunks(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        clean_root: str | Path,
        split: str,
        degrade_config: str | Path,
        chunk_seconds: float = 4.0,
        length: int = 10_000,
        deterministic: bool = False,
    ):
        rows = [json.loads(l) for l in Path(manifest).read_text().splitlines()]
        self.paths = [Path(clean_root) / r["path"] for r in rows if r["split"] == split]
        if not self.paths:
            raise ValueError(f"No segments for split '{split}' in {manifest}")
        self.chunk = int(chunk_seconds * SAMPLE_RATE)
        self.length = length
        self.deterministic = deterministic  # fixed pairs for validation
        self.degrade_config = degrade_config
        self._degrader: Degrader | None = None  # built lazily inside each worker

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self._degrader is None:
            self._degrader = Degrader(self.degrade_config)
        seed = [idx] if self.deterministic else [torch.initial_seed() % 2**32, idx]
        rng = np.random.default_rng(seed)

        path = self.paths[rng.integers(len(self.paths))]
        frames = sf.info(str(path)).frames
        start = int(rng.integers(0, max(1, frames - self.chunk)))
        clean, _ = sf.read(str(path), start=start, frames=self.chunk, dtype="float32")
        if clean.ndim > 1:
            clean = clean.mean(axis=1)
        if len(clean) < self.chunk:
            clean = np.pad(clean, (0, self.chunk - len(clean)))

        degraded, _ = self._degrader(clean, rng)
        degraded, clean = scale_pair(degraded, clean)
        return torch.from_numpy(degraded.astype(np.float32)), torch.from_numpy(clean.astype(np.float32))
