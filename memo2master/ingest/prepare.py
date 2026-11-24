"""Turn raw downloads into clean, loudness-normalized FLAC segments + a manifest.

    python -m memo2master.ingest.prepare --raw data/raw --out data/clean

Each source video is split into fixed-length segments; mostly-silent segments
are dropped. Train/val/test split is assigned per *video* (by hash of its id)
so no song leaks across splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from memo2master.audio import SAMPLE_RATE, load_any, normalize_loudness, save

AUDIO_EXTS = {".webm", ".m4a", ".opus", ".mp3", ".wav", ".flac", ".ogg", ".mp4"}


def assign_split(source_id: str, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.sha1(source_id.encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def active_fraction(seg: np.ndarray, sr: int, floor_db: float) -> float:
    """Fraction of 100 ms windows louder than floor_db (dBFS RMS)."""
    win = sr // 10
    n = len(seg) // win
    if n == 0:
        return 0.0
    frames = seg[: n * win].reshape(n, win)
    db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    return float(np.mean(db > floor_db))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/clean"))
    ap.add_argument("--segment-seconds", type=float, default=30.0)
    ap.add_argument("--target-lufs", type=float, default=-18.0)
    ap.add_argument("--silence-floor-db", type=float, default=-50.0)
    ap.add_argument("--min-active", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    args = ap.parse_args()

    files = sorted(p for p in args.raw.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        done = {json.loads(l)["source_id"] for l in manifest_path.read_text().splitlines()}

    seg_len = int(args.segment_seconds * SAMPLE_RATE)
    with manifest_path.open("a") as mf:
        for f in tqdm(files, desc="prepare"):
            source_id = f.stem
            if source_id in done:
                continue
            try:
                audio = load_any(f)
            except Exception as e:  # noqa: BLE001
                print(f"! failed to decode {f.name}: {e}")
                continue
            audio = normalize_loudness(audio, SAMPLE_RATE, args.target_lufs)
            split = assign_split(source_id, args.val_frac, args.test_frac)

            kept = 0
            for i in range(len(audio) // seg_len):
                seg = audio[i * seg_len : (i + 1) * seg_len]
                if active_fraction(seg, SAMPLE_RATE, args.silence_floor_db) < args.min_active:
                    continue
                rel = Path(split) / f"{source_id}_{i:04d}.flac"
                save(args.out / rel, seg)
                mf.write(json.dumps({
                    "path": str(rel), "source_id": source_id, "split": split,
                    "seconds": args.segment_seconds,
                }) + "\n")
                kept += 1
            mf.flush()
            tqdm.write(f"{source_id}: {kept} segments -> {split}")


if __name__ == "__main__":
    main()
