"""Randomized Pedalboard chain that turns clean audio into a fake phone voice memo.

Order follows the physical signal path:
    room -> distance EQ -> noise -> phone mic -> AGC -> compressor -> limiter
    -> clipping -> bandwidth (resample) -> codec

Output is sample-aligned with the input (RIRs are trimmed to their direct-path
peak), so (degraded, clean) pairs can be compared frame-by-frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pedalboard as pb
import soundfile as sf
import yaml

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}


def _u(rng: np.random.Generator, lo_hi: list[float]) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _list_audio(folder: str | Path | None) -> list[Path]:
    if not folder:
        return []
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def _load_mono(path: Path, sr: int) -> np.ndarray:
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if file_sr != sr:  # simple linear resample; IRs/noise beds don't need better
        n = int(round(len(audio) * sr / file_sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def _colored_noise(rng: np.random.Generator, n: int, alpha: float) -> np.ndarray:
    """1/f^alpha noise: 0 = white, 1 = pink, 2 = brown."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec /= f ** (alpha / 2)
    x = np.fft.irfft(spec, n)
    return (x / (np.std(x) + 1e-9)).astype(np.float32)


class Degrader:
    def __init__(self, config: dict[str, Any] | str | Path):
        if not isinstance(config, dict):
            config = yaml.safe_load(Path(config).read_text())
        self.cfg = config
        self.sr = int(config.get("sample_rate", 44_100))
        self.st = config["stages"]
        self.rir_paths = _list_audio(config.get("rir_dir"))
        self.noise_paths = _list_audio(config.get("noise_dir"))
        self._rir_cache: dict[Path, np.ndarray] = {}

    # ---- individual stages -------------------------------------------------

    def _rir(self, rng: np.random.Generator) -> np.ndarray:
        path = self.rir_paths[rng.integers(len(self.rir_paths))]
        if path not in self._rir_cache:
            ir = _load_mono(path, self.sr)
            peak = int(np.argmax(np.abs(ir)))
            ir = ir[peak:]  # align direct path to t=0 so output stays in sync
            ir /= np.sqrt(np.sum(ir**2)) + 1e-9
            self._rir_cache[path] = ir.astype(np.float32)
        return self._rir_cache[path]

    def _noise(self, rng: np.random.Generator, n: int, params: dict) -> np.ndarray:
        if self.noise_paths and rng.random() < 0.8:
            path = self.noise_paths[rng.integers(len(self.noise_paths))]
            noise = _load_mono(path, self.sr)
            if len(noise) < n:
                noise = np.tile(noise, int(np.ceil(n / len(noise))))
            start = int(rng.integers(0, len(noise) - n + 1))
            noise = noise[start : start + n]
            params["noise_src"] = path.name
        else:
            alpha = float(rng.uniform(0.0, 2.0))
            noise = _colored_noise(rng, n, alpha)
            params["noise_src"] = f"colored(alpha={alpha:.2f})"
        if rng.random() < self.st["noise"].get("hum_p", 0.0):
            t = np.arange(n) / self.sr
            base = float(rng.choice([50.0, 60.0]))
            hum = sum(np.sin(2 * np.pi * base * k * t) / k for k in range(1, 6))
            noise = noise + 0.5 * hum.astype(np.float32) / np.std(hum)
            params["hum_hz"] = base
        return noise.astype(np.float32)

    def _agc(self, x: np.ndarray, rng: np.random.Generator, params: dict) -> np.ndarray:
        c = self.st["agc"]
        window_s = _u(rng, c["window_s"])
        max_gain = _u(rng, c["max_gain_db"])
        target = _u(rng, c["target_db"])
        hop = self.sr // 100  # 10 ms frames
        n_frames = int(np.ceil(len(x) / hop))
        padded = np.pad(x, (0, n_frames * hop - len(x)))
        frame_db = 10 * np.log10(np.mean(padded.reshape(n_frames, hop) ** 2, axis=1) + 1e-10)
        coef = np.exp(-1.0 / max(window_s * 100, 1.0))
        env = np.empty_like(frame_db)
        acc = frame_db[0]
        for i, v in enumerate(frame_db):
            acc = coef * acc + (1 - coef) * v
            env[i] = acc
        gain_db = np.clip(target - env, -max_gain, max_gain)
        gain = 10 ** (np.interp(np.arange(len(x)) / hop, np.arange(n_frames), gain_db) / 20)
        params["agc"] = dict(window_s=window_s, max_gain_db=max_gain, target_db=target)
        return (x * gain).astype(np.float32)

    # ---- main entry --------------------------------------------------------

    def __call__(
        self, clean: np.ndarray, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, dict]:
        rng = rng or np.random.default_rng()
        st, sr = self.st, self.sr
        x = np.asarray(clean, dtype=np.float32).copy()
        n = len(x)
        params: dict[str, Any] = {}

        def on(name: str) -> bool:
            return name in st and rng.random() < st[name]["p"]

        # 1. room
        if on("room"):
            if self.rir_paths:
                mix = _u(rng, st["room"]["rir_mix"])
                x = pb.Convolution(self._rir(rng), mix=mix, sample_rate=sr)(x, sr)
                params["room"] = dict(type="rir", mix=mix)
            else:
                fr = st["room"]["fallback_reverb"]
                p = dict(room_size=_u(rng, fr["room_size"]), damping=_u(rng, fr["damping"]),
                         wet_level=_u(rng, fr["wet_level"]))
                x = pb.Reverb(**p, dry_level=1.0 - p["wet_level"], width=0.0)(x, sr)
                params["room"] = dict(type="reverb", **p)

        # 2. distance / balance EQ
        if on("distance_eq"):
            tilt = _u(rng, st["distance_eq"]["tilt_db"])
            low = _u(rng, st["distance_eq"]["low_shelf_db"])
            x = pb.Pedalboard([
                pb.HighShelfFilter(cutoff_frequency_hz=3000, gain_db=tilt),
                pb.LowShelfFilter(cutoff_frequency_hz=250, gain_db=low),
            ])(x, sr)
            params["distance_eq"] = dict(tilt_db=tilt, low_shelf_db=low)

        # 3. noise in the room
        if on("noise"):
            snr = _u(rng, st["noise"]["snr_db"])
            noise = self._noise(rng, n, params)
            sig_p = np.mean(x**2) + 1e-10
            noise *= np.sqrt(sig_p / (10 ** (snr / 10)) / (np.mean(noise**2) + 1e-10))
            x = x + noise
            params["snr_db"] = snr

        # 4. phone mic response
        if on("phone_mic"):
            c = st["phone_mic"]
            hp = _u(rng, c["highpass_hz"])
            chain = [pb.HighpassFilter(cutoff_frequency_hz=hp)]
            peaks = []
            for _ in range(int(rng.integers(c["num_peaks"][0], c["num_peaks"][1] + 1))):
                pk = dict(cutoff_frequency_hz=float(np.exp(rng.uniform(*np.log(c["peak_hz"])))),
                          gain_db=_u(rng, c["peak_gain_db"]), q=_u(rng, c["peak_q"]))
                chain.append(pb.PeakFilter(**pk))
                peaks.append(pk)
            x = pb.Pedalboard(chain)(x, sr)
            params["phone_mic"] = dict(highpass_hz=hp, peaks=peaks)

        # 5. AGC
        if on("agc"):
            x = self._agc(x, rng, params)

        # 6-7. dynamics
        if on("compressor"):
            c = st["compressor"]
            p = dict(threshold_db=_u(rng, c["threshold_db"]), ratio=_u(rng, c["ratio"]),
                     attack_ms=_u(rng, c["attack_ms"]), release_ms=_u(rng, c["release_ms"]))
            x = pb.Compressor(**p)(x, sr)
            params["compressor"] = p
        if on("limiter"):
            c = st["limiter"]
            p = dict(threshold_db=_u(rng, c["threshold_db"]), release_ms=_u(rng, c["release_ms"]))
            x = pb.Limiter(**p)(x, sr)
            params["limiter"] = p

        # 8. clipping
        if on("clipping"):
            drive = _u(rng, st["clipping"]["drive_db"])
            x = pb.Pedalboard([pb.Gain(drive), pb.Clipping(threshold_db=-0.5), pb.Gain(-drive)])(x, sr)
            params["clipping_drive_db"] = drive

        # 9. bandwidth loss
        if on("bandwidth"):
            target_sr = float(rng.choice(st["bandwidth"]["sample_rates"]))
            x = pb.Resample(target_sample_rate=target_sr)(x, sr)
            params["bandwidth_sr"] = target_sr

        # 10. codec
        if on("gsm"):
            x = pb.GSMFullRateCompressor()(x, sr)
            params["codec"] = "gsm"
        elif on("codec"):
            q = _u(rng, st["codec"]["mp3_vbr_quality"])
            x = pb.MP3Compressor(vbr_quality=q)(x, sr)
            params["codec"] = dict(mp3_vbr_quality=q)

        x = np.nan_to_num(x[:n], nan=0.0, posinf=0.0, neginf=0.0)
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        return x.astype(np.float32), params
