"""流式识别管线（FR-1.2）。

音频回调 → RingBuffer → VAD 断句线程 → ASR 转写 → 产出带会话时间戳的句子。

断句策略：累积 pending 音频，周期性运行 Silero VAD（faster-whisper 内置）；
当语音段后出现 ≥ silence_ms 的静音，或 pending 超过 max_segment_sec 时切句。
切句时保留少量重叠上下文，避免词被切断。
"""

from __future__ import annotations

import logging
import threading
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


class ASRStreamer:
    """从 RingBuffer 消费音频，断句并转写。"""

    def __init__(
        self,
        engine: ASREngine,
        seg_cfg: SegmentationConfig,
        language_hint: str | None = None,
        on_sentence: Callable[[Sentence], None] | None = None,
        vad_fn=None,
    ):
        self.engine = engine
        self.seg_cfg = seg_cfg
        self.language_hint = language_hint
        self.on_sentence = on_sentence or (lambda s: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # VAD 后端（可插拔，见 asr/vad.py）；None = 内置 Silero
        self._vad_fn = vad_fn

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

    # ---- 内部 ----

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

    def _run(self, ring) -> None:
        vad = self._get_vad()
        block = int(SAMPLE_RATE * 0.1)  # 100ms 读取粒度
        pending = np.empty(0, dtype=np.float32)
        pending_start = ring.tell()  # pending 起点的绝对样本位置
        max_samples = int(self.seg_cfg.max_segment_sec * SAMPLE_RATE)
        silence_samples = int(self.seg_cfg.silence_ms * SAMPLE_RATE / 1000)
        since_vad = 0

        while not self._stop.is_set() or ring.total_written() - ring.tell() > 0:
            chunk, _pos = ring.read(block, timeout=0.2)
            if len(chunk) == 0:
                if self._stop.is_set():
                    break
                continue
            pending = np.concatenate([pending, chunk])
            since_vad += len(chunk)
            if since_vad < _VAD_CHECK_SAMPLES and len(pending) < max_samples:
                continue
            since_vad = 0

            speech = vad(pending)
            if not speech:
                # 全是静音：丢弃超出保留窗口的部分
                if len(pending) > max_samples:
                    drop = len(pending) - silence_samples
                    pending = pending[drop:]
                    pending_start += drop
                continue

            last_end = speech[-1]["end"]
            trailing_silence = len(pending) - last_end
            force = len(pending) >= max_samples
            if not force and trailing_silence < silence_samples:
                continue  # 句子尚未结束

            # 切句：[首个语音段起点, 最后语音段终点]，含 VAD 自带 padding
            seg_start = speech[0]["start"]
            seg_end = last_end
            utter = pending[seg_start:seg_end]
            utter_offset_sec = (pending_start + seg_start) / SAMPLE_RATE
            self._transcribe_and_emit(utter, utter_offset_sec)

            # 句尾之后的静音保留给下一句（其起点即下一段 pending 的起点）
            pending = pending[seg_end:]
            pending_start += seg_end

        # 停止后冲刷：pending 中残余语音不再等静音，直接转写
        if len(pending) > 0:
            speech = vad(pending)
            if speech:
                seg_start = speech[0]["start"]
                utter = pending[seg_start : speech[-1]["end"]]
                self._transcribe_and_emit(utter, (pending_start + seg_start) / SAMPLE_RATE)

    def _transcribe_and_emit(self, utter: np.ndarray, offset_sec: float) -> None:
        if len(utter) < SAMPLE_RATE * 0.2:  # 过短的碎片跳过
            return
        try:
            text, rel_start, rel_end = self.engine.transcribe(
                utter, language=self.language_hint
            )
        except Exception:
            log.exception("ASR 转写失败，跳过该段")
            return
        if not text:
            return
        dur = len(utter) / SAMPLE_RATE
        sentence = Sentence(
            start=offset_sec + rel_start,
            end=offset_sec + (rel_end if rel_end > rel_start else dur),
            text=text,
        )
        self.on_sentence(sentence)
