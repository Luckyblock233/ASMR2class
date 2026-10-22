"""流式识别管线（FR-1.2）。

音频回调 → RingBuffer → ASRStreamer 线程 → SegmentProcessor（VAD 断句 + 转写）
→ 产出带会话时间戳的句子。

断句/转写逻辑在 segmenter.py（与离线批处理共用）；本类只负责实时采集侧：
RingBuffer 消费、暂停语义（§4.3：暂停丢弃积压，恢复从当前音频继续）。
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from ..config import SegmentationConfig
from .engine import ASREngine
from .segmenter import SAMPLE_RATE, SegmentProcessor, Sentence

__all__ = ["ASRStreamer", "Sentence", "SAMPLE_RATE"]


class ASRStreamer:
    """从 RingBuffer 消费音频，断句并转写。"""

    def __init__(
        self,
        engine: ASREngine,
        seg_cfg: SegmentationConfig,
        language_hint: str | None = None,
        on_sentence: Callable[[Sentence], None] | None = None,
        vad_fn=None,
        metrics=None,
    ):
        self.engine = engine
        self.seg_cfg = seg_cfg
        self._language_hint = language_hint
        self.on_sentence = on_sentence or (lambda s: None)
        self._stop = threading.Event()
        self._paused = threading.Event()  # 置位 = 暂停：丢弃积压，不产出新句
        self._thread: threading.Thread | None = None
        self._vad_fn = vad_fn
        # 运行时指标采集（telemetry.RuntimeMetrics，可选）：只写快照，不触碰 UI
        self._metrics = metrics
        self._processor: SegmentProcessor | None = None

    @property
    def language_hint(self) -> str | None:
        return self._language_hint

    @language_hint.setter
    def language_hint(self, value: str | None) -> None:
        """档案切换时更新 ASR 语种提示；必须同步到正在运行的处理器，
        否则流式线程仍用旧语种（处理器在 start 时创建，不会自动跟随）。"""
        self._language_hint = value
        if self._processor is not None:
            self._processor.language_hint = value

    def _new_processor(self) -> SegmentProcessor:
        self._processor = SegmentProcessor(
            self.engine,
            self.seg_cfg,
            language_hint=self._language_hint,
            on_sentence=self.on_sentence,
            vad_fn=self._vad_fn,
            metrics=self._metrics,
        )
        return self._processor

    # ---- 生命周期 ----

    def start(self, ring) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(ring,), name="asr-streamer", daemon=True
        )
        self._thread.start()

    def stop(self, flush: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30 if flush else 5)
            self._thread = None

    def set_paused(self, paused: bool) -> None:
        """实时暂停/恢复（交接文档 §4.3）。

        暂停期间丢弃新到音频、不产出新句；已在途的翻译/语法自然完成。
        恢复时从当前音频继续，不补识别暂停期间的旧音频。
        """
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    # ---- 内部 ----

    def _run(self, ring) -> None:
        processor = self._new_processor()
        block = int(SAMPLE_RATE * 0.1)  # 100ms 读取粒度

        while not self._stop.is_set() or ring.total_written() - ring.tell() > 0:
            chunk, pos = ring.read(block, timeout=0.2)
            if self._paused.is_set():
                # 暂停：消费并丢弃积压音频，恢复后从当前位置重新断句
                processor = self._new_processor()
                continue
            if len(chunk) == 0:
                if self._stop.is_set():
                    break
                continue
            processor.feed(chunk, start_sample=pos)

        # 停止后冲刷：pending 中残余语音不再等静音，直接转写
        processor.finish()
