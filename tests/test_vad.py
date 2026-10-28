"""Silero VAD（固定后端）测试。"""

from src.asr.vad import SileroVad, create_silero_vad


def test_create_silero_vad_passthrough():
    vad = create_silero_vad(min_silence_ms=300, pad_ms=150)
    assert isinstance(vad, SileroVad)
    assert vad._options.min_silence_duration_ms == 300  # 断句配置透传
    assert vad._options.speech_pad_ms == 150
