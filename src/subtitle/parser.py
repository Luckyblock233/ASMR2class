"""SRT 解析（回放模式用，FR-3.3 / M3）。

双语 SRT 条目解析为 (start, end, source, translation?)：
条目内第一行为原文，第二行（如有）为译文。
"""

from __future__ import annotations

from pathlib import Path

import pysrt

from .writer import SubtitleEntry


def load_srt(path: str | Path) -> list[SubtitleEntry]:
    subs = pysrt.open(str(path), encoding="utf-8")
    entries = []
    for sub in subs:
        lines = sub.text.splitlines()
        source = lines[0] if lines else ""
        translation = lines[1] if len(lines) > 1 else None
        entries.append(
            SubtitleEntry(
                start=sub.start.ordinal / 1000,
                end=sub.end.ordinal / 1000,
                source=source,
                translation=translation,
            )
        )
    return entries
