"""字幕文件生成（FR-2）：txt / 原文 SRT / 双语 SRT / 分析 JSON / 滚动记忆。"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .document import Segment

# 兼容既有导入；句段模型已移动到 document.py。
SubtitleEntry = Segment


def srt_timestamp(seconds: float) -> str:
    """秒 → SRT 时间戳 00:00:01,200（负值归零）。"""
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_txt(entries: list[SubtitleEntry], path: Path) -> None:
    """完整转写纯文本（FR-2.1）：每行一句原文。"""
    path.write_text(
        "\n".join(e.source for e in entries) + "\n", encoding="utf-8"
    )


def write_srt(entries: list[SubtitleEntry], path: Path, bilingual: bool) -> None:
    """原文或双语 SRT；优先沿用句段稳定 ID。"""
    blocks = []
    for index, e in enumerate(entries, 1):
        lines = [
            str(e.id or index),
            f"{srt_timestamp(e.start)} --> {srt_timestamp(e.end)}",
            e.source,
        ]
        if bilingual and e.translation:
            lines.append(e.translation)
        blocks.append("\n".join(lines))
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_analysis_json(
    entries: list[SubtitleEntry],
    path: Path,
    *,
    language_profile: str,
    source_lang: str,
    target_lang: str,
    prompt_template_version: int = 0,
    grammar_prompt_template_version: int = 0,
    grammar_provider: str = "",
    grammar_model: str = "",
) -> None:
    """语法分析 JSON：通用信封固定，语言相关字段原样透传。"""
    payload = {
        "version": "1.0",
        "language_profile": language_profile,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "prompt_template_version": prompt_template_version,
        "grammar_prompt_template_version": grammar_prompt_template_version,
        "grammar_provider": grammar_provider,
        "grammar_model": grammar_model,
        "segments": [
            {
                "id": e.id or index,
                "start": srt_timestamp(e.start),
                "end": srt_timestamp(e.end),
                "source": e.source,
                "translation": e.translation,
                "translation_status": e.translation_status,
                "translation_error": e.translation_error,
                "asr_ms": e.asr_ms,
                "translate_ms": e.translate_ms,
                **e.analysis_payload(),
            }
            for index, e in enumerate(entries, 1)
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\s]+', "-", name).strip("-") or "session"


def make_session_dir(output_root: Path, session_name: str) -> Path:
    """output/<YYYYMMDD>-<会话名>/，重名自动追加时间后缀。"""
    date = time.strftime("%Y%m%d")
    base = output_root / f"{date}-{sanitize_filename(session_name)}"
    path = base
    if path.exists():
        path = output_root / f"{date}-{sanitize_filename(session_name)}-{time.strftime('%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    return path
