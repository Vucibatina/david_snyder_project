"""Parsers for the two source transcript formats.

youtube/*.txt:
    line 1: presenter name
    line 2: publish date, "YYYY-Mon-DD" (e.g. "2024-Oct-15")
    line 3: title
    line 4+: "Kind: captions Language: en " prefix immediately followed by the
             full transcript body (no internal line breaks in practice, but we
             don't rely on that).

videos/*.txt:
    plain transcript text, no header. Title is derived from the filename.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

_CAPTION_PREFIX_RE = re.compile(r"^\s*Kind:\s*captions\s*Language:\s*[a-zA-Z-]+\s*", re.IGNORECASE)
_DATE_FORMAT = "%Y-%b-%d"


@dataclass
class ParsedDocument:
    source_type: str  # "youtube" | "lecture"
    teacher: str
    title: str
    publish_date: Optional[str]  # ISO date string, or None
    video_id: Optional[str]
    file_name: str
    file_relpath: str
    text: str


def _humanize_filename(stem: str) -> str:
    title = re.sub(r"[_\-]+", " ", stem).strip()
    title = re.sub(r"\s+", " ", title)
    return title


def parse_youtube_date(date_str: str) -> Optional[str]:
    date_str = date_str.strip()
    try:
        return datetime.strptime(date_str, _DATE_FORMAT).date().isoformat()
    except ValueError:
        return None


def parse_youtube_file(path: Path, file_relpath: str, teacher: str) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.split("\n")

    presenter = lines[0].strip() if len(lines) > 0 and lines[0].strip() else teacher
    date_str = lines[1].strip() if len(lines) > 1 else ""
    title = lines[2].strip() if len(lines) > 2 and lines[2].strip() else path.stem

    rest = "\n".join(lines[3:]) if len(lines) > 3 else ""
    body = _CAPTION_PREFIX_RE.sub("", rest, count=1).strip()

    return ParsedDocument(
        source_type="youtube",
        teacher=presenter or teacher,
        title=title,
        publish_date=parse_youtube_date(date_str),
        video_id=path.stem,
        file_name=path.name,
        file_relpath=file_relpath,
        text=body,
    )


def parse_lecture_file(path: Path, file_relpath: str, teacher: str) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    body = raw.strip()

    return ParsedDocument(
        source_type="lecture",
        teacher=teacher,
        title=_humanize_filename(path.stem),
        publish_date=None,
        video_id=None,
        file_name=path.name,
        file_relpath=file_relpath,
        text=body,
    )
