"""VAD 断句 + ASR 转写的共用处理器（实时 ASRStreamer 与离线批处理共用）。

从 ASRStreamer 提炼：调用方按音频到达顺序喂 chunk（16kHz float32 mono），
处理器内部累积 pending、周期性跑 VAD、切句并转写，产出带会话秒级时间戳的句子。
EOF 必须显式调用 finish() 冲刷尾句——不用固定时长/sleep 作为完成判断。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import SegmentationConfig
from .engine import ASREngine

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_VAD_CHECK_SAMPLES = 3200  # 每累积 200ms 新音频跑一次 VAD
_PAD_SEC = 0.2  # 句首/句尾保留的上下文


@dataclass
class Sentence:
    """一句识别结果，时间戳为会话绝对秒。"""

    start: float
    end: float
    text: str
    segment_id: int = 0  # 主流程登记字幕文档后回填；Blocker 必须原样携带到翻译回调
    asr_ms: float = 0.0  # 该句 ASR 转写耗时（毫秒）


class SegmentProcessor:
    """有状态的断句-转写处理器；单线程消费，不做并发。"""

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
        self.language_hint = language_hint
        self.on_sentence = on_sentence or (lambda s: None)
        self._vad_fn = vad_fn
        self._metrics = metrics
        self._pending = np.empty(0, dtype=np.float32)
        self._pending_start: int | None = None  # pending 起点的绝对样本位置
        self._since_vad = 0

    def _get_vad(self):
        if self._vad_fn is not None:
            return self._vad_fn
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        options = VadOptions(
            min_silence_duration_ms=self.seg_cfg.silence_ms,
            speech_pad_ms=int(_PAD_SEC * 1000),
        )

        def vad(audio: np.ndarray) -> list[dict]:
            return get_speech_timestamps(audio, vad_options=options)

        self._vad_fn = vad
        return vad

    def feed(self, chunk: np.ndarray, start_sample: int | None = None) -> None:
        """喂入一段连续音频；首块需给 start_sample 作为时间基准。"""
        if self._pending_start is None:
            self._pending_start = start_sample or 0
        max_samples = int(self.seg_cfg.max_segment_sec * SAMPLE_RATE)
        silence_samples = int(self.seg_cfg.silence_ms * SAMPLE_RATE / 1000)
        vad = self._get_vad()

        self._pending = np.concatenate([self._pending, chunk])
        self._since_vad += len(chunk)
        if self._since_vad < _VAD_CHECK_SAMPLES and len(self._pending) < max_samples:
            return
        self._since_vad = 0

        pending = self._pending
        speech = vad(pending)
        if self._metrics is not None:
            self._metrics.set_vad_active(bool(speech))
        if not speech:
            # 全是静音：丢弃超出保留窗口的部分
            if len(pending) > max_samples:
                drop = len(pending) - silence_samples
                self._pending = pending[drop:]
                self._pending_start += drop
            return

        last_end = speech[-1]["end"]
        trailing_silence = len(pending) - last_end
        force = len(pending) >= max_samples
        if not force and trailing_silence < silence_samples:
            return  # 句子尚未结束

        # 切句：[首个语音段起点, 最后语音段终点]，含 VAD 自带 padding
        seg_start = speech[0]["start"]
        seg_end = last_end
        utter = pending[seg_start:seg_end]
        utter_offset_sec = (self._pending_start + seg_start) / SAMPLE_RATE
        self._transcribe_and_emit(utter, utter_offset_sec)

        # 句尾之后的静音保留给下一句（其起点即下一段 pending 的起点）
        self._pending = pending[seg_end:]
        self._pending_start += seg_end

    def finish(self) -> None:
        """EOF 冲刷：pending 中残余语音不再等静音，直接转写。"""
        if len(self._pending) == 0:
            return
        speech = self._get_vad()(self._pending)
        if speech:
            seg_start = speech[0]["start"]
            utter = self._pending[seg_start : speech[-1]["end"]]
            offset = ((self._pending_start or 0) + seg_start) / SAMPLE_RATE
            self._transcribe_and_emit(utter, offset)
        self._pending = np.empty(0, dtype=np.float32)

    def _transcribe_and_emit(self, utter: np.ndarray, offset_sec: float) -> None:
        if len(utter) < SAMPLE_RATE * 0.2:  # 过短的碎片跳过
            return
        started = time.perf_counter()
        try:
            text, rel_start, rel_end = self.engine.transcribe(
                utter, language=self.language_hint
            )
        except Exception:
            log.exception("ASR 转写失败，跳过该段")
            return
        asr_ms = (time.perf_counter() - started) * 1000
        if not text:
            return
        if self._metrics is not None:
            self._metrics.record_asr(asr_ms)
        dur = len(utter) / SAMPLE_RATE
        sentence = Sentence(
            start=offset_sec + rel_start,
            end=offset_sec + (rel_end if rel_end > rel_start else dur),
            text=text,
            asr_ms=asr_ms,
        )
        self.on_sentence(sentence)
