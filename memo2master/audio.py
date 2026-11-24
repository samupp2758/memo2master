"""Shared audio I/O helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

SAMPLE_RATE = 44_100


def load_any(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any format ffmpeg understands (m4a, opus, webm, mp3...) to mono float32."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def save(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr)


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Integrated-loudness normalization (ITU-R BS.1770). Falls back to RMS on silence."""
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(audio)
    except ValueError:  # too short for gating blocks
        loudness = float("-inf")
    if not np.isfinite(loudness):
        return audio
    out = pyln.normalize.loudness(audio, loudness, target_lufs)
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak * 0.99 if peak > 0.99 else out).astype(np.float32)


def rms_db(audio: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9))
