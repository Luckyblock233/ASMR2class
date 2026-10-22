"""会话记录器：收集句段、接收乱序语法结果并在会话结束落盘。"""

from __future__ import annotations

import logging
from pathlib import Path

from ..nlp.context_memory import RollingMemory
from ..profiles.loader import LanguageProfile
from .document import Segment, SessionDocument
from .writer import make_session_dir, write_analysis_json, write_srt, write_txt

log = logging.getLogger(__name__)


class SessionRecorder:
    def __init__(self, profile: LanguageProfile, prompt_template_version: int = 0):
        self.profile = profile
        self.prompt_template_version = prompt_template_version
        self.document = SessionDocument(
            language_profile=profile.name,
            source_lang=profile.source_lang,
            target_lang=profile.target_lang,
            prompt_template_version=prompt_template_version,
        )

    def add(
        self,
        start: float,
        end: float,
        source: str,
        translation: str | None,
        *,
        asr_ms: float = 0.0,
    ) -> int:
        """登记首次显示的句子并返回稳定 ID，供异步语法结果回写。"""
        return self.document.add(start, end, source, translation, asr_ms=asr_ms)

    def add_pending(self, start: float, end: float, source: str, *, asr_ms: float = 0.0) -> int:
        """ASR 句首次出现时登记「翻译在途」占位句段并返回稳定 ID。"""
        from .document import TRANSLATION_PENDING

        return self.document.add(
            start, end, source, None,
            translation_status=TRANSLATION_PENDING,
            asr_ms=asr_ms,
        )

    def update_translation(
        self,
        segment_id: int,
        translation: str | None,
        *,
        status: str,
        error: str = "",
        translate_ms: float = 0.0,
    ) -> Segment:
        """翻译完成/失败回调：按 ID 更新译文与状态（失败只影响该句）。"""
        return self.document.update_translation(
            segment_id, translation, status=status, error=error, translate_ms=translate_ms
        )

    @property
    def entries(self) -> list[Segment]:
        return self.document.segments

    def get_segment(self, segment_id: int) -> Segment | None:
        return self.document.get(segment_id)

    def save(
        self,
        output_root: Path,
        session_name: str = "",
        memory: RollingMemory | None = None,
    ) -> Path | None:
        """保存全部会话文件，返回输出目录；无内容时不产生目录。

        session_name 为空时默认用当前语言对（<源>-<目标>，如 ja-zh；auto 源为 auto-zh）。
        """
        entries = self.entries
        if not entries:
            return None
        if not session_name:
            session_name = f"{self.profile.source_lang}-{self.profile.target_lang}"
        out_dir = make_session_dir(output_root, session_name)
        write_txt(entries, out_dir / "transcript.txt")
        write_srt(entries, out_dir / "original.srt", bilingual=False)
        write_srt(entries, out_dir / "bilingual.srt", bilingual=True)
        # 文件名标注源语言与目标语言；源语言自动识别时为 analysis.auto.<tgt>.json
        write_analysis_json(
            entries,
            out_dir / f"analysis.{self.profile.source_lang}.{self.profile.target_lang}.json",
            language_profile=self.profile.name,
            source_lang=self.profile.source_lang,
            target_lang=self.profile.target_lang,
            prompt_template_version=self.prompt_template_version,
            grammar_prompt_template_version=self.document.grammar_prompt_template_version,
            grammar_provider=self.document.grammar_provider,
            grammar_model=self.document.grammar_model,
        )
        if memory is not None:
            memory.save(out_dir / "memory.json", language_profile=self.profile.name)
        log.info("会话文件已保存到 %s", out_dir)
        return out_dir
