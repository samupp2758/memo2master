"""Download audio-only streams from YouTube with yt-dlp.

    python -m memo2master.ingest.download --sources sources/sources.txt --out data/raw

Keeps a download archive so re-running only fetches new videos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yt_dlp


def read_sources(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def download(urls: list[str], out_dir: Path, min_minutes: float, max_minutes: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def length_filter(info, *, incomplete):
        dur = info.get("duration")
        if dur is None:
            return None
        if dur < min_minutes * 60 or dur > max_minutes * 60:
            return f"skipping {info.get('id')}: {dur / 60:.1f} min outside range"
        return None

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "download_archive": str(out_dir / "archive.txt"),
        "writeinfojson": True,
        "ignoreerrors": True,
        "match_filter": length_filter,
        "quiet": False,
        "noprogress": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download(urls)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=Path, default=Path("sources/sources.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--min-minutes", type=float, default=2.0)
    ap.add_argument("--max-minutes", type=float, default=240.0)
    args = ap.parse_args()

    urls = read_sources(args.sources)
    if not urls:
        raise SystemExit(f"No URLs in {args.sources}")
    download(urls, args.out, args.min_minutes, args.max_minutes)


if __name__ == "__main__":
    main()
