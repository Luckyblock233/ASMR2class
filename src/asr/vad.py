"""VAD：固定使用 faster-whisper 内置 Silero VAD（零下载）。

接口：callable(audio: np.ndarray) -> [{"start": int, "end": int}, ...]（样本单位）
"""

from __future__ import annotations

import numpy as np

SpeechSegment = dict  # {"start": int, "end": int}（样本单位）


class SileroVad:
    """faster-whisper 内置 Silero VAD。"""

    def __init__(self, min_silence_ms: int, pad_ms: int):
        from faster_whisper.vad import VadOptions

        self._options = VadOptions(
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=pad_ms,
        )

    def __call__(self, audio: np.ndarray) -> list[SpeechSegment]:
        from faster_whisper.vad import get_speech_timestamps

        return get_speech_timestamps(audio, vad_options=self._options)


def create_silero_vad(min_silence_ms: int, pad_ms: int = 200) -> SileroVad:
    """创建 Silero VAD（供 ASRStreamer / 离线 runner 注入）。"""
    return SileroVad(min_silence_ms=min_silence_ms, pad_ms=pad_ms)
