"""Train the spectrogram U-Net.

    python -m memo2master.train.train --config configs/train.yaml
    python -m memo2master.train.train --config configs/train.yaml --resume runs/baseline/last.pt
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from memo2master.data.dataset import PairedChunks
from memo2master.models.unet import Restorer
from memo2master.train.losses import MultiResSTFTLoss, compressed_complex_loss


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: dict) -> Restorer:
    return Restorer(**cfg["model"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tc, dc = cfg["train"], cfg["data"]
    run_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    device = pick_device()
    print(f"device: {device}")

    common = dict(manifest=dc["manifest"], clean_root=dc["clean_root"],
                  degrade_config=dc["degrade_config"], chunk_seconds=dc["chunk_seconds"])
    train_ds = PairedChunks(split="train", length=tc["steps"] * tc["batch_size"], **common)
    val_ds = PairedChunks(split="val", length=dc["val_items"], deterministic=True, **common)
    loader_kw = dict(batch_size=tc["batch_size"], num_workers=tc["num_workers"],
                     pin_memory=device.type == "cuda", persistent_workers=tc["num_workers"] > 0)
    train_dl = DataLoader(train_ds, shuffle=False, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = build_model(cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / tc["warmup_steps"])
                                              * 0.5 * (1 + math.cos(math.pi * min(s / tc["steps"], 1.0))))
    mrstft = MultiResSTFTLoss().to(device)
    w = cfg["loss"]

    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"resumed from step {step}")

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(tc["amp"])
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    def autocast():
        return torch.autocast("cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()

    def compute_loss(noisy, clean):
        with autocast():
            pred_wav, pred_spec = model(noisy)
        target_spec = model.stft(clean)
        l_mr = mrstft(pred_wav, clean)
        l_cx = compressed_complex_loss(pred_spec, target_spec)
        return w["mrstft"] * l_mr + w["complex"] * l_cx, l_mr, l_cx

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        total, n = 0.0, 0
        for noisy, clean in val_dl:
            loss, _, _ = compute_loss(noisy.to(device), clean.to(device))
            total += loss.item() * len(noisy); n += len(noisy)
        model.train()
        return total / max(n, 1)

    log = (run_dir / "log.csv").open("a")
    model.train()
    t0 = time.time()
    for noisy, clean in train_dl:
        if step >= tc["steps"]:
            break
        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
        loss, l_mr, l_cx = compute_loss(noisy, clean)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt); scaler.update(); sched.step()
        step += 1

        if step % tc["log_every"] == 0:
            rate = tc["log_every"] / (time.time() - t0); t0 = time.time()
            print(f"step {step:>7} loss {loss.item():.4f} mr {l_mr.item():.4f} "
                  f"cx {l_cx.item():.4f} lr {sched.get_last_lr()[0]:.2e} {rate:.1f} it/s")
            log.write(f"{step},train,{loss.item():.5f},{l_mr.item():.5f},{l_cx.item():.5f}\n"); log.flush()
        if step % tc["val_every"] == 0:
            v = validate()
            print(f"step {step:>7} VAL loss {v:.4f}")
            log.write(f"{step},val,{v:.5f},,\n"); log.flush()
        if step % tc["ckpt_every"] == 0 or step == tc["steps"]:
            ck = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                      step=step, config=cfg)
            torch.save(ck, run_dir / "last.pt")
            torch.save(dict(model=model.state_dict(), step=step, config=cfg), run_dir / f"step{step}.pt")


if __name__ == "__main__":
    main()
