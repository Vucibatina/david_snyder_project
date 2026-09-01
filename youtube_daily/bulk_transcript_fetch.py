#!/usr/bin/env python3
"""
Bulk transcript fetcher — pulls every video transcript from a YouTube channel
using yt-dlp and stores them in data/<ChannelHandle>/<video_id>.txt (same
format as fetch_youtube_videos.py).

Pulls both the channel's "Videos" tab and its "Live/Streams" tab (past
broadcasts), deduped by video ID, so long-form webinar/replay content isn't
missed. Videos gated behind a paid channel membership tier (Gold/Platinum/
Diamond, etc.) cannot be fetched — that's a YouTube-side paywall, not
something yt-dlp or cookies can get around without a subscribed account.

Run (from this directory, using this project's venv which has yt-dlp installed):
    cd /Users/vuk/projects/youtube_daily
    ./venv/bin/python bulk_transcript_fetch.py                    # David Snyder (default)
    ./venv/bin/python bulk_transcript_fetch.py @SomebodyElse      # any channel handle
    ./venv/bin/python bulk_transcript_fetch.py @Handle1 @Handle2  # multiple channels

Resume-safe: videos with an existing data/<ChannelHandle>/<video_id>.txt are skipped,
so it's safe to rerun periodically to pick up newly uploaded/unlocked videos.

Age-restricted videos need a valid cookies.txt (Netscape format) next to this
script, exported from a browser logged into YouTube — see the 'cookiefile'
option below. Re-export if you start seeing "Sign in to confirm your age" errors.

Output lands in this project's own data/<ChannelHandle>/ directory (see BASE_DIR
below). For the David Snyder NLP channel, transcripts are additionally kept in
sync with a separate project data store at:
    /Users/vuk/projects/data/DavidSnyder/youtube
Copy new files over after a run with (non-destructive, skips existing):
    cp -n /Users/vuk/projects/youtube_daily/data/DavidSnyderNLP/*.txt \\
        /Users/vuk/projects/data/DavidSnyder/youtube/
"""

import os
import re
import sys
import time
from datetime import datetime

try:
    import yt_dlp
except ImportError:
    print("yt-dlp not installed. Run: pip install yt-dlp")
    sys.exit(1)

import tempfile
from pathlib import Path

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MIN_DURATION_SECONDS = 61  # skip YouTube Shorts


# ── Helpers ───────────────────────────────────────────────────────────────────

def transcript_path(video_id: str, channel_dir: str) -> str:
    return os.path.join(channel_dir, f"{video_id}.txt")


def already_stored(video_id: str, channel_dir: str) -> bool:
    return os.path.exists(transcript_path(video_id, channel_dir))


def save_transcript(video_id, channel_name, published_at, title, transcript_text, channel_dir):
    """Write data/<channel>/<video_id>.txt — same format as fetch_youtube_videos.py."""
    base = os.path.realpath(BASE_DIR)
    real_dir = os.path.realpath(os.path.abspath(channel_dir))
    if real_dir == base or not real_dir.startswith(base + os.sep):
        raise ValueError(f"Refusing to write: '{channel_dir}' is not a subdirectory of BASE_DIR '{BASE_DIR}'")
    os.makedirs(channel_dir, exist_ok=True)
    with open(transcript_path(video_id, channel_dir), 'w', encoding='utf-8') as f:
        f.write(f"{channel_name}\n")
        f.write(f"{published_at}\n")
        f.write(f"{title}\n")
        f.write(f"{transcript_text}\n")


def parse_vtt(vtt_text: str) -> str:
    """
    Convert VTT subtitle text to clean plain text.
    Auto-generated captions repeat the same line as the window scrolls —
    de-duplicate consecutive identical lines to avoid repetition.
    """
    blocks = re.split(r'\n{2,}', vtt_text.strip())
    lines = []
    for block in blocks:
        for part in block.strip().splitlines():
            part = part.strip()
            if not part:
                continue
            if part.startswith('WEBVTT') or '-->' in part or re.match(r'^\d+$', part):
                continue
            clean = re.sub(r'<[^>]+>', '', part).strip()
            if clean:
                lines.append(clean)

    deduped = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    return ' '.join(deduped)


def fmt_date(upload_date: str) -> str:
    """Convert YYYYMMDD → YYYY-MMM-DD for the header line."""
    try:
        return datetime.strptime(upload_date, '%Y%m%d').strftime('%Y-%b-%d')
    except Exception:
        return upload_date


# ── Core Fetcher ──────────────────────────────────────────────────────────────

def fetch_channel(channel_handle: str):
    if not channel_handle.startswith('@'):
        channel_handle = f'@{channel_handle}'

    subdir = channel_handle.lstrip('@')
    if not subdir:
        print(f"ERROR: channel handle '{channel_handle}' produces empty subdirectory — skipping")
        return
    channel_dir = os.path.join(BASE_DIR, subdir)
    print(f"\n{'─'*60}")
    print(f"Channel : {channel_handle}")
    print(f"Data dir: {channel_dir}")
    print(f"{'─'*60}")

    # ── Phase 1: discover all video IDs across the Videos and Live/Streams tabs ──
    print("Discovering videos...")
    entries = []
    seen_ids = set()
    channel_name = channel_handle
    for tab in ('videos', 'streams'):
        tab_url = f"https://www.youtube.com/{channel_handle}/{tab}"
        with yt_dlp.YoutubeDL({'extract_flat': True, 'quiet': True, 'ignoreerrors': True}) as ydl:
            info = ydl.extract_info(tab_url, download=False)
        if not info:
            print(f"  {tab}: tab not found — skipping")
            continue
        tab_entries = info.get('entries', [])
        channel_name = info.get('channel') or info.get('uploader') or channel_name
        added = 0
        for e in tab_entries:
            vid = e.get('id')
            if not vid or vid in seen_ids:
                continue
            seen_ids.add(vid)
            entries.append(e)
            added += 1
        print(f"  {tab}: {len(tab_entries)} entries ({added} new)")

    if not entries:
        print(f"CANNOT FIND {channel_handle}")
        return

    print(f"Found {len(entries)} total unique entries for '{channel_name}'")

    # filter out already-stored and obvious Shorts (duration available in flat mode)
    todo = []
    skipped_exists = 0
    skipped_short = 0
    for e in entries:
        vid = e.get('id')
        if not vid:
            continue
        dur = e.get('duration') or 0
        if dur and dur < MIN_DURATION_SECONDS:
            skipped_short += 1
            continue
        if already_stored(vid, channel_dir):
            skipped_exists += 1
            continue
        todo.append(e)

    print(f"To fetch: {len(todo)}  |  Already stored: {skipped_exists}  |  Shorts filtered: {skipped_short}\n")
    if not todo:
        print("Nothing new to fetch.")
        return

    # ── Phase 2: fetch subtitle VTT per video, parse, save ───────────────────
    stats = {'saved': 0, 'no_sub': 0, 'short': 0, 'error': 0}
    total = len(todo)
    width = len(str(total))

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            'skip_download': True,
            'writeautomaticsub': True,
            'writesubtitles': True,
            'subtitleslangs': ['en'],
            'subtitlesformat': 'vtt',
            'outtmpl': os.path.join(tmpdir, '%(id)s.%(ext)s'),
            'cookiefile': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cookies.txt'),
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
        }

        for idx, entry in enumerate(todo, 1):
            video_id = entry['id']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            prefix = f"[{idx:{width}}/{total}]"

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    vinfo = ydl.extract_info(video_url, download=True)
            except Exception as exc:
                print(f"{prefix} ERROR   {video_id}: {exc}")
                stats['error'] += 1
                continue

            if not vinfo:
                stats['error'] += 1
                continue

            title = vinfo.get('title', video_id)
            duration = vinfo.get('duration') or 0
            published = fmt_date(vinfo.get('upload_date', ''))
            short_title = title[:52] + ('…' if len(title) > 52 else '')

            # second duration check (flat mode may have missed it)
            if duration < MIN_DURATION_SECONDS:
                print(f"{prefix} SHORT   {short_title} ({duration}s)")
                stats['short'] += 1
                continue

            # find the downloaded VTT file
            vtt_file = Path(tmpdir) / f"{video_id}.en.vtt"
            if not vtt_file.exists():
                candidates = list(Path(tmpdir).glob(f"{video_id}*.vtt"))
                vtt_file = candidates[0] if candidates else None

            if vtt_file and vtt_file.exists():
                raw = vtt_file.read_text(encoding='utf-8', errors='replace')
                transcript = parse_vtt(raw)
                vtt_file.unlink()
            else:
                transcript = None

            if transcript:
                save_transcript(video_id, channel_name, published, title, transcript, channel_dir)
                print(f"{prefix} OK      {short_title}  ({len(transcript):,} chars)")
                stats['saved'] += 1
            else:
                print(f"{prefix} NO_SUB  {short_title}")
                stats['no_sub'] += 1

            time.sleep(0.3)

    print(f"""
Summary for {channel_name}
  Saved:            {stats['saved']}
  No subtitle:      {stats['no_sub']}
  Skipped (Short):  {stats['short']}
  Errors:           {stats['error']}
  Already existed:  {skipped_exists}
""")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # handles = sys.argv[1:] if len(sys.argv) > 1 else ['@DavidSnyderNLP']
    # handles = sys.argv[1:] if len(sys.argv) > 1 else ['@KetoKamp']
    handles = sys.argv[1:] if len(sys.argv) > 1 else [
        '@BryanJohnson',
        '@DrGundry',
        '@DoctorMike',
        '@BobbyParrish',
        '@GlucoseRevolution',
        '@diabe_tech',
        '@wimhof1',
        '@TomBilyeu',
        '@TheDiaryOfACEO',
    ]

    for handle in handles:
        try:
            fetch_channel(handle)
        except KeyboardInterrupt:
            print("\nInterrupted — files saved so far are in data/")
            sys.exit(0)

    existing = len(list(Path(BASE_DIR).rglob('*.txt'))) if Path(BASE_DIR).exists() else 0
    print(f"Total transcripts in data/: {existing}")
