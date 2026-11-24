# memo2master

Turn phone **voice memos of live worship-band sets** into **production-grade recordings**.

A rebuild of my senior project. The idea: take professionally produced live
recordings of a CCM (contemporary Christian music) band, **synthetically degrade**
them so they sound like a phone voice memo from the back of the room, and train a
**spectrogram U-Net** to undo the damage. Then point the model at real voice memos
of college bands playing the same style.

```
 YouTube (clean, produced)                                 real voice memos
          │                                                        │
          ▼                                                        ▼
   ingest: download → 44.1 kHz mono FLAC → segments         infer: chunk + crossfade
          │                                                        ▲
          ▼                                                        │
   degrade (Pedalboard, random every chunk) ──► (degraded, clean) ──► Spectrogram U-Net
```

---

## Approach

### 1. Data: clean audio from YouTube, damage made on the fly

Only clean audio is stored. Each time the data loader draws a 4-second chunk, a
**freshly randomized Pedalboard chain** turns it into a fake voice memo. So the
model never sees the same degradation twice, and disk usage stays at 1× the clean
corpus no matter how long you train.

The chain follows the physical signal path of a phone recording a live band:

| # | Stage | Implementation | Imitates |
|---|---|---|---|
| 1 | Room | `Convolution` with real room impulse responses, or `Reverb` fallback | chapel / gym reverb |
| 2 | Distance EQ | `HighShelfFilter` + `LowShelfFilter` | standing far from the PA |
| 3 | Noise | real noise beds or 1/f colored noise, optional mains hum, random SNR | crowd, HVAC, clapping |
| 4 | Phone mic | `HighpassFilter` + random `PeakFilter`s | tiny capsule, no lows |
| 5 | AGC | slow envelope-following gain (numpy) | voice-memo "breathing" |
| 6 | Compressor | `Compressor`, fast attack, high ratio | phone dynamics processing |
| 7 | Limiter | `Limiter` | brick-walled loudness |
| 8 | Clipping | `Gain` → `Clipping` → `Gain` | kick / chorus overloading the mic |
| 9 | Bandwidth | `Resample` to 8–32 kHz and back | lost highs |
| 10 | Codec | `MP3Compressor` (rarely `GSMFullRateCompressor`) | lossy encoding artifacts |

Every stage fires with probability `p`, and every parameter is sampled from the
ranges in [`configs/degrade.yaml`](configs/degrade.yaml). The output stays
sample-aligned with the clean input: impulse responses are trimmed to their
direct-path peak.

> **Domain gap is the main risk.** The model only learns to undo the damage it was
> trained on. Always run the **audition** step and A/B the fake memos against
> real ones before training. Dropping real RIRs into `data/rirs/` and real crowd
> or room noise into `data/noise/` helps a lot.

### 2. Model: spectrogram U-Net, done carefully

[`memo2master/models/unet.py`](memo2master/models/unet.py)

- **Input/output:** compressed complex STFT (2048 window, 512 hop at 44.1 kHz,
  magnitude^0.3), stored as real and imaginary channels. Phase is modeled, not
  copied from the noisy input, which keeps drum transients sharp.
- **Residual mapping, not a mask:** `output = input + U-Net(input)`. A
  multiplicative mask can't rebuild highs the codec removed (anything × 0 = 0).
  Mapping lets the model do bandwidth extension.
- **Asymmetric downsampling:** frequency halves at every level (1024 → 32 bins);
  time halves only at the first three levels.
- **Bottleneck:** a small transformer along time, for long-range context like
  sustained pads and room tails.
- **Identity start:** the output layer is zero-initialized, so training starts
  from "output = input" and only learns the correction.

### 3. Loss

- Multi-resolution STFT loss (spectral convergence + log-magnitude, at 4 resolutions)
- Compressed-complex MSE (phase-aware)

**Planned stage 2:** HiFi-GAN-style multi-period and multi-scale discriminators
with feature-matching loss. Pure regression losses produce a clean but slightly
muffled "average" sound, and adversarial training is what brings back realistic
air and transients.

### 4. Evaluation

- **Synthetic test split** (held out per *video*, so no song leaks): SI-SDR and
  log-spectral distance, reported as input → output improvement.
- **Real voice memos** (no reference exists). Planned: Fréchet Audio Distance
  against the clean set, plus blind listening tests with musicians.

---

## Project layout

```
configs/
  degrade.yaml        # the degradation chain: stages, probabilities, ranges
  train.yaml          # model / loss / training hyperparameters
sources/
  sources.txt         # YouTube URLs (videos, playlists, channels)
memo2master/
  audio.py            # ffmpeg decode, save, loudness normalization
  ingest/
    download.py       # yt-dlp audio-only download with resume archive
    prepare.py        # → 44.1k mono FLAC, LUFS-normalized 30 s segments, per-video split
  degrade/
    chain.py          # randomized Pedalboard degradation chain
    audition.py       # write clean/degraded example pairs to listen to
  data/
    dataset.py        # on-the-fly paired chunks for PyTorch
  models/
    stft.py           # compressed complex STFT and its inverse
    unet.py           # SpecUNet + Restorer (waveform in/out wrapper)
  train/
    losses.py         # multi-res STFT + compressed complex losses
    train.py          # training loop (CUDA / MPS / CPU, bf16, checkpoints)
  eval/
    metrics.py        # SI-SDR, log-spectral distance
    evaluate.py       # score a checkpoint on the test split
  infer/
    run.py            # restore a full-length memo with overlap-add
tests/
  test_pipeline.py    # degrader shape/alignment/determinism, model round-trip
data/                 # (gitignored) raw/, clean/, rirs/, noise/
runs/                 # (gitignored) checkpoints and logs
```

---

## Quickstart

Requires **Python 3.10–3.13** (PyTorch and Pedalboard have no 3.14+ wheels yet)
and **ffmpeg** on your PATH.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

### 1. Download clean recordings

Add URLs to `sources/sources.txt` (only recordings you have permission to use), then:

```bash
python -m memo2master.ingest.download --sources sources/sources.txt --out data/raw
```

### 2. Prepare segments

```bash
python -m memo2master.ingest.prepare --raw data/raw --out data/clean
```

### 3. Listen before you train

```bash
python -m memo2master.degrade.audition --clean data/clean --n 12 --out auditions/
```

Open `auditions/` and compare `NN_degraded.wav` against real voice memos. Tune
`configs/degrade.yaml` until they sound like the same kind of bad.

### 4. Train

```bash
python -m memo2master.train.train --config configs/train.yaml
```

The defaults (batch 8, 4 s chunks, bf16) target a 16 GB GPU. Degradation runs on
the CPU in data-loader workers, so raise `num_workers` if the GPU is starved.

### 5. Evaluate and restore

```bash
python -m memo2master.eval.evaluate --ckpt runs/baseline/last.pt
```

```bash
python -m memo2master.infer.run --ckpt runs/baseline/last.pt --input memo.m4a --output memo_master.wav
```

---

## Roadmap

- [x] Ingest pipeline (yt-dlp → segmented FLAC + manifest)
- [x] Randomized Pedalboard degradation chain + audition tool
- [x] Spectrogram U-Net baseline (complex, residual mapping, time attention)
- [x] Training loop, evaluation, long-file inference
- [ ] Collect real RIRs + noise beds; tune degradation against real memos
- [ ] Stage 2: GAN discriminators (multi-period / multi-scale) + feature matching
- [ ] Re-recorded pairs: play clean tracks through a PA, record on a phone, align, fine-tune
- [ ] FAD + listening-test harness for real memos
- [ ] Stereo output
- [ ] v2: stem separation → per-stem restoration → learned automix

## Data and permissions

Clean training audio comes from recordings used with the permission of the
performing band. No audio is committed to this repo (see `.gitignore`), and
model outputs of copyrighted songs are for research and personal use.
