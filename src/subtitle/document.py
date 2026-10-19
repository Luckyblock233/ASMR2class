"""实时与回放共用的字幕文档及稳定句段 ID。"""

from __future__ import annotations

import copy
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ANALYSIS_NONE = "none"
ANALYSIS_PENDING = "pending"
ANALYSIS_READY = "ready"
ANALYSIS_FAILED = "failed"


@dataclass
class Segment:
    """一个字幕句段；语法字段的内部结构完全由语言档案和 prompt 决定。"""

    start: float
    end: float
    source: str
    translation: str | None = None
    id: int = 0
    structure: Any = None
    grammar_points: list[Any] = field(default_factory=list)
    vocabulary: list[Any] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    analysis_status: str = ANALYSIS_NONE
    analysis_error: str = ""

    @property
    def segment_id(self) -> int:
        return self.id

    @property
    def has_analysis(self) -> bool:
        return (
            self.structure is not None
            or bool(self.grammar_points)
            or bool(self.vocabulary)
            or bool(self.extra)
        )

    def analysis_payload(self) -> dict[str, Any]:
        return {
            "structure": copy.deepcopy(self.structure),
            "grammar_points": copy.deepcopy(self.grammar_points),
            "vocabulary": copy.deepcopy(self.vocabulary),
            "extra": copy.deepcopy(self.extra),
        }

    def snapshot(self) -> "Segment":
        return copy.deepcopy(self)


class SessionDocument:
    """按稳定 ID 保存句段，允许工作线程乱序合并语法结果。"""

    def __init__(
        self,
        *,
        language_profile: str = "",
        source_lang: str = "",
        target_lang: str = "",
        prompt_template_version: int = 0,
    ):
        self.language_profile = language_profile
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.prompt_template_version = prompt_template_version
        self.grammar_prompt_template_version = 0
        self.grammar_provider = ""
        self.grammar_model = ""
        self._segments: list[Segment] = []
        self._by_id: dict[int, Segment] = {}
        self._next_id = 1
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._segments)

    @property
    def segments(self) -> list[Segment]:
        with self._lock:
            return [s.snapshot() for s in self._segments]

    def get(self, segment_id: int) -> Segment | None:
        with self._lock:
            segment = self._by_id.get(segment_id)
            return segment.snapshot() if segment is not None else None

    def add(self, start: float, end: float, source: str, translation: str | None) -> int:
        with self._lock:
            segment_id = self._next_id
            self._next_id += 1
            segment = Segment(start, end, source, translation, id=segment_id)
            self._segments.append(segment)
            self._by_id[segment_id] = segment
            return segment_id

    def add_loaded(self, segment: Segment) -> None:
        """加载已有句段；保留文件中的 ID，并恢复下一个可用 ID。"""
        with self._lock:
            if segment.id <= 0:
                segment.id = self._next_id
            if segment.id in self._by_id:
                raise ValueError(f"字幕句段 ID 重复：{segment.id}")
            self._next_id = max(self._next_id, segment.id + 1)
            if segment.has_analysis:
                segment.analysis_status = ANALYSIS_READY
            self._segments.append(segment)
            self._by_id[segment.id] = segment

    def set_analysis_state(self, segment_id: int, status: str, error: str = "") -> Segment:
        with self._lock:
            segment = self._require(segment_id)
            segment.analysis_status = status
            segment.analysis_error = error
            return segment.snapshot()

    def apply_analysis(self, segment_id: int, payload: dict[str, Any]) -> Segment:
        with self._lock:
            segment = self._require(segment_id)
            segment.structure = copy.deepcopy(payload.get("structure"))
            segment.grammar_points = copy.deepcopy(payload.get("grammar_points") or [])
            segment.vocabulary = copy.deepcopy(payload.get("vocabulary") or [])
            segment.extra = copy.deepcopy(payload.get("extra") or {})
            segment.analysis_status = ANALYSIS_READY
            segment.analysis_error = ""
            return segment.snapshot()

    def update_grammar_metadata(
        self,
        *,
        template_version: int,
        provider_id: str = "",
        model: str = "",
    ) -> None:
        with self._lock:
            self.grammar_prompt_template_version = template_version
            self.grammar_provider = provider_id
            self.grammar_model = model

    def _require(self, segment_id: int) -> Segment:
        try:
            return self._by_id[segment_id]
        except KeyError as e:
            raise KeyError(f"未知字幕句段 ID：{segment_id}") from e


def parse_srt_timestamp(value: str) -> float:
    """把 00:00:01,200 转成秒。"""
    try:
        hours, minutes, rest = value.strip().split(":", 2)
        seconds, millis = rest.replace(".", ",").split(",", 1)
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000
    except (ValueError, TypeError) as e:
        raise ValueError(f"非法 SRT 时间戳：{value!r}") from e


def _document_from_analysis(path: Path) -> SessionDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError(f"分析文件格式错误：{path}")
    doc = SessionDocument(
        language_profile=str(payload.get("language_profile") or ""),
        source_lang=str(payload.get("source_lang") or ""),
        target_lang=str(payload.get("target_lang") or ""),
        prompt_template_version=int(payload.get("prompt_template_version") or 0),
    )
    doc.grammar_prompt_template_version = int(payload.get("grammar_prompt_template_version") or 0)
    doc.grammar_provider = str(payload.get("grammar_provider") or "")
    doc.grammar_model = str(payload.get("grammar_model") or "")
    for index, raw in enumerate(payload["segments"], 1):
        if not isinstance(raw, dict):
            raise ValueError(f"分析文件第 {index} 句不是对象：{path}")
        segment = Segment(
            id=int(raw.get("id") or index),
            start=parse_srt_timestamp(str(raw["start"])),
            end=parse_srt_timestamp(str(raw["end"])),
            source=str(raw.get("source") or ""),
            translation=raw.get("translation"),
            structure=raw.get("structure"),
            grammar_points=list(raw.get("grammar_points") or []),
            vocabulary=list(raw.get("vocabulary") or []),
            extra=dict(raw.get("extra") or {}),
        )
        doc.add_loaded(segment)
    return doc


def load_session_document(session_dir: str | Path) -> SessionDocument:
    """从会话目录恢复文档；优先分析 JSON，缺失时退回双语/原文 SRT。"""
    directory = Path(session_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"会话目录不存在：{directory}")
    analysis_files = sorted(directory.glob("analysis.*.json"))
    if len(analysis_files) > 1:
        raise ValueError(f"会话目录存在多个分析文件，无法确定语言：{directory}")
    if analysis_files:
        return _document_from_analysis(analysis_files[0])

    from .parser import load_srt  # 延迟导入，避免 document/parser 循环依赖

    srt_path = directory / "bilingual.srt"
    if not srt_path.is_file():
        srt_path = directory / "original.srt"
    if not srt_path.is_file():
        raise FileNotFoundError(f"会话目录缺少 analysis JSON 或 SRT：{directory}")
    doc = SessionDocument()
    for entry in load_srt(srt_path):
        doc.add(entry.start, entry.end, entry.source, entry.translation)
    return doc
