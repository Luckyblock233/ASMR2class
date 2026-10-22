"""离线批处理（阶段 5）：PyAV 解码器与 OfflineRunner 编排。"""

import json
import math
import re
import wave
from pathlib import Path

import numpy as np
import pytest

from src.audio.decoder import AudioDecodeError, decode_audio, media_duration_sec

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_wav(path: Path, seconds: float = 2.0, rate: int = 16000, tone: bool = True) -> None:
    """生成测试 wav：正弦或静音。"""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    samples = (0.3 * np.sin(2 * math.pi * 440 * t)) if tone else np.zeros(n)
    pcm16 = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16.tobytes())


def test_decode_wav_streams_16k_mono_chunks(tmp_path):
    wav = tmp_path / "音频 测试.wav"  # 中文 + 空格路径
    _write_wav(wav, seconds=1.0)
    chunks = list(decode_audio(wav, chunk_ms=100))
    assert len(chunks) == 10
    total = np.concatenate(chunks)
    assert total.dtype == np.float32
    assert 15500 <= len(total) <= 16500  # 1 秒 16kHz（重采样误差容限）
    assert np.abs(total).max() > 0.1  # 有信号而非全零


def test_decode_missing_and_corrupt_file(tmp_path):
    with pytest.raises(AudioDecodeError, match="不存在"):
        list(decode_audio(tmp_path / "missing.mp3"))
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a media file")
    with pytest.raises(AudioDecodeError):
        list(decode_audio(bad))


def test_media_duration(tmp_path):
    wav = tmp_path / "a.wav"
    _write_wav(wav, seconds=2.0)
    duration = media_duration_sec(wav)
    assert duration is not None and 1.8 <= duration <= 2.2
    assert media_duration_sec(tmp_path / "missing.wav") is None


# ---- OfflineRunner 编排（假引擎/假解码，不依赖真实 whisper/媒体）----


class FakeEngine:
    """每段语音转写为递增句号。"""

    def __init__(self):
        self.calls = 0

    def transcribe(self, utter, language=None):
        self.calls += 1
        dur = len(utter) / 16000
        return (f"句{self.calls}", 0.0, dur)


def fake_vad(audio: np.ndarray) -> list[dict]:
    """能量即语音的假 VAD。"""
    nz = np.nonzero(np.abs(audio) > 0.01)[0]
    if len(nz) == 0:
        return []
    return [{"start": int(nz[0]), "end": int(nz[-1]) + 1}]


def fake_decode_factory(chunks_per_file):
    def decode(path):
        yield from (c.copy() for c in chunks_per_file)

    return decode


def alternating_audio(sentences: int) -> list[np.ndarray]:
    """每句 = 1s 语音 + 1s 静音，切成 100ms 块。"""
    speech = np.full(16000, 0.5, dtype=np.float32)
    silence = np.zeros(16000, dtype=np.float32)
    blocks = []
    for _ in range(sentences):
        blocks += [speech, silence]
    return [b[i : i + 1600] for b in blocks for i in range(0, 16000, 1600)]


class EchoProvider:
    """从 prompt 中提取句 id，逐句回译；记录每次块大小与顺序。"""

    def __init__(self):
        from src.config import ProviderConfig

        self.cfg = ProviderConfig(id="echo", capabilities=["translation", "grammar"])
        self.online = True
        self.block_sizes: list[int] = []
        self.sources: list[str] = []

    def chat(self, prompt, *, system=None, json_mode=False):
        from src.nlp.llm_client import ChatResult

        m = re.search(r"\[\{\"id\".*\}\]", prompt, re.DOTALL)
        items = json.loads(m.group(0))
        self.block_sizes.append(len(items))
        self.sources.extend(i["source"] for i in items)
        payload = {
            "sentences": [
                {"id": i["id"], "source": i["source"], "translation": f"译-{i['source']}"}
                for i in items
            ]
        }
        return ChatResult(json.dumps(payload, ensure_ascii=False))


class GrammarProvider(EchoProvider):
    """翻译/语法共槽桩：prompt 含句 id JSON 走翻译契约，否则走语法契约。"""

    def chat(self, prompt, *, system=None, json_mode=False):
        from src.nlp.llm_client import ChatResult

        if re.search(r"\[\{\"id\".*\}\]", prompt, re.DOTALL):
            return super().chat(prompt, system=system, json_mode=json_mode)
        return ChatResult(json.dumps({
            "structure": {"pattern": "测试"},
            "grammar_points": [],
            "vocabulary": [],
            "extra": {},
        }))


class FakeSlots:
    """最小槽位实现：固定 provider + 直通租约。"""

    def __init__(self, provider):
        self._provider = provider

    def translation_provider(self):
        return self._provider

    def grammar_provider(self):
        return self._provider

    def translation_lease(self):
        return self._lease()

    def grammar_lease(self):
        return self._lease()

    def _lease(self):
        import contextlib

        @contextlib.contextmanager
        def cm():
            yield self._provider

        return cm()


def _make_runner(tmp_path, provider=None, sentences=3, **kwargs):
    from src.config import load_config
    from src.offline.runner import OfflineRunner
    from src.profiles.loader import get_profile

    cfg = load_config(PROJECT_ROOT)
    profile = get_profile(cfg.languages_dir, "ja")
    slots = FakeSlots(provider) if provider is not None else None
    return OfflineRunner(
        cfg,
        profile,
        output_root=tmp_path,
        engine=FakeEngine(),
        vad_fn=fake_vad,
        decoder=fake_decode_factory(alternating_audio(sentences)),
        slots=slots,
        **kwargs,
    )


def test_offline_pipeline_end_to_end(tmp_path):
    """解码→断句→顺序翻译→语法→保存：五类文件齐全，译文按 ID 回写。"""
    provider = GrammarProvider()  # 翻译与语法共用同一桩（语法返回结构 JSON）
    runner = _make_runner(tmp_path, provider, sentences=4, grammar_enabled=True)
    out_dir = runner.run_file("fake.mp4")
    assert out_dir is not None
    names = {p.name for p in out_dir.iterdir()}
    assert {"transcript.txt", "original.srt", "bilingual.srt", "analysis.ja.zh.json", "memory.json"} <= names

    payload = json.loads((out_dir / "analysis.ja.zh.json").read_text(encoding="utf-8"))
    assert len(payload["segments"]) == 4
    assert payload["segments"][0]["source"] == "句1"
    assert payload["segments"][0]["translation"] == "译-句1"
    assert payload["segments"][0]["translation_status"] == "ready"
    # 语法分析全部完成才保存
    assert all(s["structure"] == {"pattern": "测试"} for s in payload["segments"])
    # 顺序翻译：块内顺序与源句顺序一致
    assert provider.sources == ["句1", "句2", "句3", "句4"]


def test_offline_block_limit_20_effective(tmp_path):
    """离线块上限 max_sentences_offline=20 实际生效（25 句 → 20+5 两块）。"""
    provider = EchoProvider()
    runner = _make_runner(tmp_path, provider, sentences=25)
    runner.run_file("long.mp4")
    assert provider.block_sizes == [20, 5]
    assert len(provider.sources) == 25


def test_offline_failure_isolation_and_cancel(tmp_path):
    """单文件失败不阻塞后续；取消返回 130。"""
    runner = _make_runner(tmp_path, sentences=2)

    def decoder_per_file(path):
        if "bad" in path:
            raise AudioDecodeError("无法打开媒体文件")
        yield from alternating_audio(2)

    runner._decoder = decoder_per_file
    rc = runner.run([str(tmp_path / "bad.mp4"), str(tmp_path / "good.mp4")])
    assert rc == 1  # 有失败但整体完成
    saved = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(saved) == 1  # 只有 good 产出

    def cancel_decoder(path):
        yield np.zeros(1600, dtype=np.float32)
        raise KeyboardInterrupt

    runner._decoder = cancel_decoder
    assert runner.run([str(tmp_path / "x.mp4")]) == 130
