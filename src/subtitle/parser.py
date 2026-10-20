"""SRT 解析（回放模式用，FR-3.3 / M3）。

内部双语 SRT（本工具生成）：条目内第一行为原文，第二行为译文（保持向后兼容）。
外部 SRT（external=True）：按完整块读取并保留多行，整块文本都作为原文。
"""

from __future__ import annotations

from pathlib import Path

import pysrt

from .writer import SubtitleEntry


def load_srt(path: str | Path, external: bool = False) -> list[SubtitleEntry]:
    """解析 SRT 为句段列表。

    external=False：内部双语格式（第一行原文、第二行译文）。
    external=True：外部字幕，整块多行原文合并保留（不拆译文）。
    """
    subs = pysrt.open(str(path), encoding="utf-8")
    entries = []
    for sub in subs:
        lines = sub.text.splitlines()
        if external:
            source = "\n".join(line for line in lines if line.strip())
            translation = None
        else:
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
