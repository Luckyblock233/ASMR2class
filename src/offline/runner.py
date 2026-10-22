"""真正的离线批处理（M3 阶段 5）。

与 FileSource 实时模拟的边界：本模块不使用 RingBuffer、不用墙上时钟做完成判断；
解码为拉取式生成器，消费速度即背压（不丢样本）。

管线（逐文件）：
  PyAV 流式解码 → SegmentProcessor（VAD/ASR 断句，显式 finish/EOF）
  → Blocker（offline=20 句、autoflush=False）
  → 顺序 Translator（保持历史/记忆顺序，消费线程内同步翻译）
  → 有界 GrammarAnalyzer → drain 完成后 SessionRecorder.save()

EOF 顺序：解码结束 → ASR flush → Blocker flush → 翻译全部完成
→ Grammar drain → 保存 → （全部文件结束后）关闭 Provider。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from ..asr.engine import ASREngine
from ..asr.segmenter import SAMPLE_RATE, SegmentProcessor, Sentence
from ..config import GlobalConfig
from ..nlp.blocker import Blocker
from ..nlp.prompt_engine import PromptEngine
from ..nlp.translate import Translator
from ..profiles.loader import LanguageProfile
from ..subtitle.document import (
    TRANSLATION_FAILED,
    TRANSLATION_NONE,
    TRANSLATION_READY,
)
from ..subtitle.session import SessionRecorder

log = logging.getLogger(__name__)


class OfflineRunner:
    """逐文件离线转写翻译；单文件失败隔离，Ctrl+C 取消并保存已完成部分。"""

    def __init__(
        self,
        cfg: GlobalConfig,
        profile: LanguageProfile,
        *,
        output_root: Path | None = None,
        engine=None,
        vad_fn=None,
        decoder=None,
        slots=None,
        debug=None,
        grammar_enabled: bool = False,
        session_name: str = "",
        cancel_event: threading.Event | None = None,
        on_event=None,
    ):
        self.cfg = cfg
        self.profile = profile
        self.output_root = output_root or cfg.root / "output"
        self.vad_fn = vad_fn
        self.slots = slots
        self.debug = debug
        self.grammar_enabled = grammar_enabled
        self.session_name = session_name
        # 可取消标志（UI 批处理窗口的「取消」按钮）与进度事件回调
        self._cancel = cancel_event or threading.Event()
        self._on_event = on_event or (lambda kind, **kw: None)
        # 可注入 seam：测试用假引擎/假解码器，不依赖真实 whisper/媒体文件
        self._engine = engine
        self._decoder = decoder
        if self._decoder is None:
            from ..audio.decoder import decode_audio

            self._decoder = decode_audio

    @property
    def engine(self):
        if self._engine is None:
            self._engine = ASREngine(self.cfg.asr)
        return self._engine

    def run(self, inputs: list[str]) -> int:
        """逐文件处理；返回 0 = 全部成功，1 = 有失败，130 = 用户取消。"""
        total = len(inputs)
        failures = 0
        started_all = time.monotonic()
        for index, path in enumerate(inputs, 1):
            if self._cancel.is_set():
                return 130
            print(f"\n== [{index}/{total}] {path} ==")
            self._on_event("file_start", index=index, total=total, path=path)
            try:
                out_dir = self.run_file(path)
            except KeyboardInterrupt:
                print("\n⚠ 用户取消（Ctrl+C）")
                self._on_event("cancelled", path=path)
                return 130
            except Exception as e:  # noqa: BLE001 —— 单文件失败不阻塞后续文件
                failures += 1
                print(f"✗ 处理失败：{type(e).__name__}: {e}")
                log.exception("离线批处理文件失败：%s", path)
                self._on_event("file_error", path=path, error=str(e))
                continue
            self._on_event("file_done", path=path, out_dir=str(out_dir) if out_dir else "")
        elapsed = time.monotonic() - started_all
        print(f"\n批处理完成：{total - failures}/{total} 成功，耗时 {elapsed:.1f}s")
        self._on_event("all_done", total=total, failures=failures)
        return 1 if failures else 0

    def run_file(self, path: str) -> Path | None:
        """处理单个文件；返回输出目录（无内容时为 None）。抛异常由 run() 隔离。"""
        from ..audio.decoder import media_duration_sec

        duration = media_duration_sec(path)
        duration_text = f"{duration:.1f}s" if duration else "未知时长"
        print(f"时长：{duration_text}")

        # 每个文件一个会话：Translator/RollingMemory/Recorder 随文件重建
        translator = None
        if self.slots is not None:
            translator = Translator(
                self.slots.translation_provider,
                PromptEngine(self.cfg.prompts_dir),
                self.profile,
                self.cfg.context,
                debug=self.debug,
                lease=self.slots.translation_lease,
            )
        tpl_version = 0
        if translator is not None:
            try:
                tpl_path = translator.prompts.resolve(
                    self.profile.prompt_path("translate"), "translate"
                )
                tpl_version = translator.prompts.template_version(tpl_path)
            except Exception:  # noqa: BLE001
                pass
        recorder = SessionRecorder(self.profile, prompt_template_version=tpl_version)

        grammar = None
        if self.slots is not None and self.grammar_enabled:
            from ..nlp.grammar import GrammarAnalyzer

            grammar = GrammarAnalyzer(
                self.slots.grammar_provider,
                PromptEngine(self.cfg.prompts_dir),
                self.profile,
                recorder.document,
                lease=self.slots.grammar_lease,
            )

        # 块配置：离线 20 句上限实际生效，禁用墙上时钟 autoflush
        block_cfg = self.cfg.context.merged(self.profile.context).block
        offline_block_cfg = type(block_cfg)(
            max_sentences=block_cfg.max_sentences_offline,
            max_sentences_offline=block_cfg.max_sentences_offline,
            silence_gap_sec=block_cfg.silence_gap_sec,
        )

        sentence_count = [0]
        started = time.monotonic()

        def on_sentence(s: Sentence) -> None:
            """ASR 句出现：登记占位并喂 Blocker（与实时同一套按 ID 回写）。"""
            sentence_count[0] += 1
            if translator is not None:
                segment_id = recorder.add_pending(s.start, s.end, s.text, asr_ms=s.asr_ms)
            else:
                segment_id = recorder.add(s.start, s.end, s.text, None, asr_ms=s.asr_ms)
            s.segment_id = segment_id
            self._on_event("sentence", count=sentence_count[0], text=s.text)
            if blocker is not None:
                blocker.feed(s)

        def on_block(block: list[Sentence]) -> None:
            """顺序翻译（消费线程内同步执行：保持历史/滚动记忆顺序）。"""
            assert translator is not None
            t0 = time.perf_counter()
            results = translator.translate_block([s.text for s in block])
            block_ms = (time.perf_counter() - t0) * 1000
            bound = translator.bound
            for s, r in zip(block, results):
                ok = bool(r.translation)
                if not bound:
                    status, error = TRANSLATION_NONE, ""
                elif ok:
                    status, error = TRANSLATION_READY, ""
                else:
                    status, error = TRANSLATION_FAILED, "翻译失败或服务不可用"
                recorder.update_translation(
                    s.segment_id, r.translation,
                    status=status, error=error, translate_ms=block_ms,
                )
                if grammar is not None and grammar.available:
                    grammar.submit(s.segment_id)
            print(f"  已翻译 {sentence_count[0]} 句（最近块 {block_ms:.0f}ms）", flush=True)
            self._on_event("block_done", count=sentence_count[0], block_ms=block_ms)

        blocker = (
            Blocker(offline_block_cfg, on_block, autoflush=False)
            if translator is not None
            else None
        )
        processor = SegmentProcessor(
            self.engine,
            self.cfg.segmentation,
            language_hint=self.profile.language_hint,
            on_sentence=on_sentence,
            vad_fn=self.vad_fn,
        )

        # 拉取式解码：消费速度即背压，长文件不丢音频
        position = 0
        cancelled = False
        try:
            for chunk in self._decoder(path):
                if self._cancel.is_set():
                    cancelled = True
                    break
                processor.feed(chunk, start_sample=position)
                position += len(chunk)
        except KeyboardInterrupt:
            print("\n⚠ 取消当前文件，保存已完成部分…")
            cancelled = True
        # EOF：ASR flush → Blocker flush → 翻译完成 → Grammar drain → 保存
        processor.finish()
        if blocker is not None:
            blocker.flush()
        if grammar is not None:
            grammar.drain()
        elapsed = time.monotonic() - started
        out_dir = recorder.save(
            self.output_root,
            self.session_name or Path(path).stem,
            memory=translator.memory
            if translator is not None and translator.cfg.memory.enabled
            else None,
        )
        print(
            f"✓ {sentence_count[0]} 句，耗时 {elapsed:.1f}s"
            + (f"，已保存：{out_dir}" if out_dir else "（无有效句段）")
        )
        if cancelled:
            raise KeyboardInterrupt  # 已保存部分内容，向上终止批处理
        return out_dir
