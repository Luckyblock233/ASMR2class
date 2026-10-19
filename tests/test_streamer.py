"""流式管线测试：真实 RingBuffer + 假 VAD（能量法）+ 假 ASR 引擎。

验证断句、时间戳、停止冲刷，不依赖 GPU/模型下载。
"""

import threading
import time

import numpy as np
import pytest

from src.asr.streamer import SAMPLE_RATE, ASRStreamer, Sentence
from src.audio.base import RingBuffer
from src.config import SegmentationConfig


class FakeEngine:
    """把每段音频转写为固定文本，记录收到的音频长度。"""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None):
        dur = len(audio) / SAMPLE_RATE
        self.calls.append((dur, language))
        return f"テスト{len(self.calls)}", 0.0, dur


def energy_vad(audio: np.ndarray, frame: int = 480) -> list[dict]:
    """确定性假 VAD：RMS 高于阈值的连续帧合并为语音段。"""
    segments = []
    start = None
    for i in range(0, len(audio) - frame + 1, frame):
        loud = float(np.sqrt(np.mean(audio[i : i + frame] ** 2))) > 0.05
        if loud and start is None:
            start = i
        elif not loud and start is not None:
            segments.append({"start": start, "end": i})
            start = None
    if start is not None:
        segments.append({"start": start, "end": len(audio)})
    return segments


def make_audio(pattern: list[tuple[str, float]]) -> np.ndarray:
    """按 [('speech', 秒) | ('silence', 秒), ...] 生成测试音频。"""
    rng = np.random.default_rng(42)
    parts = []
    for kind, sec in pattern:
        n = int(sec * SAMPLE_RATE)
        if kind == "speech":
            parts.append((rng.standard_normal(n) * 0.3).astype(np.float32))
        else:
            parts.append(np.zeros(n, dtype=np.float32))
    return np.concatenate(parts)


def run_streamer(audio: np.ndarray, seg_cfg: SegmentationConfig, engine=None):
    """把音频实时喂入管线，收集产出句子。"""
    ring = RingBuffer(SAMPLE_RATE * 60)
    engine = engine or FakeEngine()
    sentences: list[Sentence] = []
    done = threading.Event()
    streamer = ASRStreamer(
        engine, seg_cfg, language_hint="ja", on_sentence=lambda s: sentences.append(s)
    )
    streamer._vad_fn = energy_vad  # 注入确定性 VAD

    block = 480  # 30ms
    streamer.start(ring)
    for i in range(0, len(audio), block):
        ring.write(audio[i : i + block])
        time.sleep(0.001)
    streamer.stop()
    return sentences, engine


def test_two_sentences_segmented():
    audio = make_audio([("silence", 0.5), ("speech", 1.5), ("silence", 1.0),
                        ("speech", 1.5), ("silence", 1.0)])
    seg = SegmentationConfig(silence_ms=500, max_segment_sec=15)
    sentences, engine = run_streamer(audio, seg)
    assert len(sentences) == 2
    # 第一句应起于 ~0.5s，第二句起于 ~3.0s（误差容忍 0.3s）
    assert sentences[0].start == pytest.approx(0.5, abs=0.3)
    assert sentences[1].start == pytest.approx(3.0, abs=0.3)
    assert all(lang == "ja" for _dur, lang in engine.calls)


def test_max_segment_force_cut():
    # 连续 20s 语音 > max_segment_sec=5，应被强制切成多段
    audio = make_audio([("speech", 8.0), ("silence", 1.0)])
    seg = SegmentationConfig(silence_ms=500, max_segment_sec=5)
    sentences, _ = run_streamer(audio, seg)
    assert len(sentences) >= 2


def test_flush_on_stop():
    # 句尾无足够静音，停止时应冲刷残余语音
    audio = make_audio([("speech", 1.5)])
    seg = SegmentationConfig(silence_ms=500, max_segment_sec=30)
    sentences, _ = run_streamer(audio, seg)
    assert len(sentences) == 1
    assert sentences[0].text == "テスト1"


def test_silence_only_produces_nothing():
    audio = make_audio([("silence", 3.0)])
    seg = SegmentationConfig(silence_ms=500, max_segment_sec=15)
    sentences, engine = run_streamer(audio, seg)
    assert sentences == []
    assert engine.calls == []
