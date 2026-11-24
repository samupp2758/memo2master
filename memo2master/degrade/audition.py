"""Listen to the degradation chain before training anything.

    python -m memo2master.degrade.audition --clean data/clean --n 12 --out auditions/
    python -m memo2master.degrade.audition --file some_song.flac --n 5 --out auditions/

Writes NN_clean.wav / NN_degraded.wav pairs plus params.json describing what was
applied to each. Compare the degraded clips against your real voice memos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from memo2master.audio import SAMPLE_RATE, load_any, save
from memo2master.degrade.chain import Degrader


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clean", type=Path, help="prepared clean dir (with manifest.jsonl)")
    src.add_argument("--file", type=Path, help="any single audio file")
    ap.add_argument("--config", type=Path, default=Path("configs/degrade.yaml"))
    ap.add_argument("--out", type=Path, default=Path("auditions"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    degrader = Degrader(args.config)
    chunk = int(args.seconds * SAMPLE_RATE)

    if args.file:
        audio = load_any(args.file)
        sources = [(args.file.name, audio)] * args.n
    else:
        rows = [json.loads(l) for l in (args.clean / "manifest.jsonl").read_text().splitlines()]
        picks = rng.choice(len(rows), size=min(args.n, len(rows)), replace=False)
        sources = [(rows[i]["path"], sf.read(str(args.clean / rows[i]["path"]), dtype="float32")[0])
                   for i in picks]

    all_params = []
    for k, (name, audio) in enumerate(sources):
        start = int(rng.integers(0, max(1, len(audio) - chunk)))
        clean = audio[start : start + chunk]
        degraded, params = degrader(clean, rng)
        save(args.out / f"{k:02d}_clean.wav", clean)
        save(args.out / f"{k:02d}_degraded.wav", degraded)
        all_params.append({"index": k, "source": name, "start_s": start / SAMPLE_RATE, **params})
        print(f"[{k:02d}] {name}: {', '.join(params)}")

    (args.out / "params.json").write_text(json.dumps(all_params, indent=2, default=float))
    print(f"\nWrote {len(sources)} pairs to {args.out}/")


if __name__ == "__main__":
    main()
