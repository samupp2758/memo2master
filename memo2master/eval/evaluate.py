"""Score a checkpoint on the held-out test split (seeded synthetic degradations).

    python -m memo2master.eval.evaluate --ckpt runs/baseline/last.pt --n 200

Reports metrics for the degraded input and for the model output, so you can see
the improvement, not just an absolute number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from memo2master.data.dataset import PairedChunks
from memo2master.eval.metrics import log_spectral_distance, si_sdr
from memo2master.models.unet import Restorer
from memo2master.train.train import pick_device


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    device = pick_device()
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["config"]
    model = Restorer(**cfg["model"]).to(device).eval()
    model.load_state_dict(ck["model"])

    dc = cfg["data"]
    ds = PairedChunks(dc["manifest"], dc["clean_root"], args.split, dc["degrade_config"],
                      dc["chunk_seconds"], length=args.n, deterministic=True)

    rows = []
    for i in range(len(ds)):
        noisy, clean = ds[i]
        pred = model(noisy[None].to(device))[0][0].float().cpu().numpy()
        noisy, clean = noisy.numpy(), clean.numpy()
        rows.append([si_sdr(noisy, clean), si_sdr(pred, clean),
                     log_spectral_distance(noisy, clean), log_spectral_distance(pred, clean)])
    r = np.array(rows).mean(axis=0)
    print(f"SI-SDR  input {r[0]:6.2f} dB -> output {r[1]:6.2f} dB  (+{r[1] - r[0]:.2f})")
    print(f"LSD     input {r[2]:6.2f} dB -> output {r[3]:6.2f} dB")


if __name__ == "__main__":
    main()
