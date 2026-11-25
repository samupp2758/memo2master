"""Download audio-only streams from YouTube with yt-dlp.

    python -m memo2master.ingest.download --sources sources/sources.txt --out data/raw

Safe to stop and re-run: a download archive (data/raw/archive.txt) records
finished videos, so later runs only fetch what's missing.

Filters (all configurable):
  * length between --min-minutes and --max-minutes (default 1-90 min)
  * titles matching --reject-title: compilations/playlists (they repeat songs
    that have their own videos, which would leak the same recording across
    train/test splits) and talk videos (interviews, recaps, vlogs)

Polite by default: random 5-15 s pause between videos to avoid YouTube's
bot checks on large channel downloads.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yt_dlp

DEFAULT_REJECT = (
    r"playlist|top songs|compilation|interview|recap|behind the|what is worship"
    r"|learns how|welcome home service"
)


def read_sources(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def js_runtimes() -> dict:
    """yt-dlp needs a JS runtime to solve YouTube's challenges; use whatever is installed."""
    return {name: {} for name in ("deno", "node", "bun") if shutil.which(name)}


def download(urls: list[str], out_dir: Path, min_minutes: float, max_minutes: float,
             reject_title: str, sleep_min: float, sleep_max: float,
             max_downloads: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    reject = re.compile(reject_title, re.IGNORECASE) if reject_title else None
    skipped = (out_dir / "skipped.tsv").open("a")

    def video_filter(info, *, incomplete):
        vid, title, dur = info.get("id"), info.get("title") or "", info.get("duration")
        reason = None
        if reject and reject.search(title):
            reason = "title"
        elif dur is not None and not (min_minutes * 60 <= dur <= max_minutes * 60):
            reason = f"length {dur / 60:.1f} min"
        if reason:
            skipped.write(f"{vid}\t{reason}\t{title}\n"); skipped.flush()
            return f"skip {vid} ({reason}): {title}"
        return None

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "download_archive": str(out_dir / "archive.txt"),
        "writeinfojson": True,
        "ignoreerrors": True,
        "match_filter": video_filter,
        "sleep_interval": sleep_min,
        "max_sleep_interval": sleep_max,
        "sleep_interval_requests": 1,
        "retries": 10,
        "fragment_retries": 10,
        "noprogress": True,
    }
    if max_downloads:
        opts["max_downloads"] = max_downloads
    runtimes = js_runtimes()
    if runtimes:
        opts["js_runtimes"] = runtimes

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download(urls)
    except yt_dlp.utils.MaxDownloadsReached:
        print(f"stopped after --max-downloads {max_downloads}")
    finally:
        skipped.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=Path("sources/sources.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--min-minutes", type=float, default=1.0)
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--reject-title", default=DEFAULT_REJECT,
                    help="regex; matching titles are skipped ('' to disable)")
    ap.add_argument("--sleep-min", type=float, default=5.0, help="seconds between videos (min)")
    ap.add_argument("--sleep-max", type=float, default=15.0, help="seconds between videos (max)")
    ap.add_argument("--max-downloads", type=int, help="stop after N new downloads (for testing)")
    args = ap.parse_args()

    urls = read_sources(args.sources)
    if not urls:
        raise SystemExit(f"No URLs in {args.sources}")
    download(urls, args.out, args.min_minutes, args.max_minutes, args.reject_title,
             args.sleep_min, args.sleep_max, args.max_downloads)


if __name__ == "__main__":
    main()
