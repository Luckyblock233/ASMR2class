"""GPU/设备解析相关单元测试（不依赖真实 GPU）。"""

from src.asr.engine import ASREngine, _find_cuda_dll_dirs, resolve_device
from src.config import ASRConfig


def test_cpu_mode_forced():
    device, ct, downgraded, _reason = resolve_device(ASRConfig(device="cpu"))
    assert device == "cpu"
    assert ct == "int8"
    assert not downgraded  # 用户显式选择不算降级


def test_no_cuda_falls_back(monkeypatch):
    import src.asr.engine as eng

    monkeypatch.setattr(eng, "_cuda_available", lambda: False)
    device, ct, downgraded, reason = resolve_device(ASRConfig(device="cuda"))
    assert device == "cpu"
    assert ct == "int8"
    assert downgraded
    assert "CUDA" in reason


def test_low_vram_falls_back(monkeypatch):
    import src.asr.engine as eng

    monkeypatch.setattr(eng, "_cuda_available", lambda: True)
    monkeypatch.setattr(eng, "_cuda_free_vram_mb", lambda: 1000)  # 低于 2000 保留线
    device, _ct, downgraded, reason = resolve_device(ASRConfig(device="auto"))
    assert device == "cpu"
    assert downgraded
    assert "1000" in reason


def test_sufficient_vram_uses_cuda(monkeypatch):
    import src.asr.engine as eng

    monkeypatch.setattr(eng, "_cuda_available", lambda: True)
    monkeypatch.setattr(eng, "_cuda_free_vram_mb", lambda: 5000)
    device, ct, downgraded, _ = resolve_device(ASRConfig(device="auto"))
    assert device == "cuda"
    assert ct == "int8_float16"
    assert not downgraded


def test_find_cuda_dll_dirs(tmp_path):
    (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    (tmp_path / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
    found = _find_cuda_dll_dirs([str(tmp_path), str(tmp_path / "nonexist")])
    assert len(found) == 2
    assert any("cublas" in str(d) for d in found)
    assert any("cudnn" in str(d) for d in found)


def test_find_cuda_dll_dirs_empty(tmp_path):
    assert _find_cuda_dll_dirs([str(tmp_path)]) == []


class _FakeSeg:
    def __init__(self, text, start=0.0, end=1.0, no_speech_prob=0.0, avg_logprob=0.0, compression_ratio=1.0):
        self.text, self.start, self.end = text, start, end
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio


class _FakeModel:
    def __init__(self, segments):
        self._segments = segments
        self.kwargs = {}

    def transcribe(self, audio, **kw):
        self.kwargs.update(kw)
        return iter(self._segments), None


def _bare_engine(segments):
    """绕过 __init__ 构造仅含 transcribe 依赖的引擎实例。"""
    engine = ASREngine.__new__(ASREngine)
    engine.cfg = ASRConfig()
    engine._model = _FakeModel(segments)
    return engine


def test_hallucination_outro_phrase_dropped():
    """静音段幻觉出的视频结尾高频字幕句被门控丢弃（正常语音段不受影响）。"""
    segs = [
        _FakeSeg(" 最後まで視聴してくださって本当にありがとうございます",
                 no_speech_prob=0.9, avg_logprob=-1.5),
        _FakeSeg(" 今日は天気がいいですね", start=1.0, end=2.0,
                 no_speech_prob=0.05, avg_logprob=-0.2),
    ]
    engine = _bare_engine(segs)
    text, start, end = engine.transcribe(None, language="ja")
    assert text == "今日は天気がいいですね"
    assert (start, end) == (1.0, 2.0)
    # 幻觉句全被丢弃时返回空文本（streamer 对空文本自然跳过）
    engine2 = _bare_engine([segs[0]])
    assert engine2.transcribe(None)[0] == ""


def test_repetitive_hallucination_dropped_by_compression_ratio():
    engine = _bare_engine([_FakeSeg(" ひ", compression_ratio=9.7, avg_logprob=-0.1)])
    assert engine.transcribe(None)[0] == ""


def test_condition_on_previous_disabled_by_default():
    engine = _bare_engine([])
    engine.transcribe(None)
    assert engine._model.kwargs["condition_on_previous_text"] is False


def test_thresholds_from_config():
    """阈值可在 config 中调紧/调松。"""
    engine = _bare_engine([_FakeSeg(" 微妙な段", no_speech_prob=0.7, avg_logprob=-0.8)])
    engine.cfg.log_prob_threshold = -0.5  # 调严：-0.8 < -0.5 → 丢弃
    assert engine.transcribe(None)[0] == ""
    engine.cfg.log_prob_threshold = -1.0  # 默认：-0.8 >= -1.0 → 保留
    assert engine.transcribe(None)[0] == "微妙な段"


def test_hallucination_phrase_blacklist():
    """幻觉短语黑名单：形态固定的结尾客套话即使置信度正常也丢弃（含变体匹配）。"""
    variants = [
        " 最後まで視聴してくださって 本当にありがとうございます",   # 完整形态
        " 後まで視聴してくださって 本当にありがとうございます",     # 缺字变体
        " 最後まで視聴してくださって ありがとうございました",       # 时态变体
        " Thank you for watching!",                                 # 英文，大小写/空白归一
    ]
    for v in variants:
        engine = _bare_engine([_FakeSeg(v, no_speech_prob=0.1, avg_logprob=-0.1)])
        assert engine.transcribe(None)[0] == "", f"未拦截：{v}"
    # 正常内容不含黑名单短语：不受影响
    engine = _bare_engine([_FakeSeg(" 今日はありがとうと言いたい", avg_logprob=-0.1)])
    assert engine.transcribe(None)[0] == "今日はありがとうと言いたい"


def test_hallucination_phrases_configurable():
    """黑名单可在 config 中覆盖（误杀时删条目）。"""
    engine = _bare_engine([_FakeSeg(" チャンネル登録お願いします", avg_logprob=-0.1)])
    engine.cfg.hallucination_phrases = ["別のフレーズ"]
    assert engine.transcribe(None)[0] == "チャンネル登録お願いします"


# ==================== 幻觉三层压制 ====================


def test_transcribe_params_passed_to_engine():
    """置信度阈值/温度回退链/语种/复读链开关原样传入 transcribe（不再只做事后自判）。"""
    engine = _bare_engine([])
    engine.transcribe(None, language="ja")
    kw = engine._model.kwargs
    assert kw["language"] == "ja"
    assert kw["condition_on_previous_text"] is False
    assert kw["vad_filter"] is False  # VAD 前置过滤已在断句层完成
    assert kw["no_speech_threshold"] == engine.cfg.no_speech_threshold
    assert kw["log_prob_threshold"] == engine.cfg.log_prob_threshold
    assert kw["compression_ratio_threshold"] == engine.cfg.compression_ratio_threshold
    assert kw["temperature"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 温度回退链


def test_param_spelling_matches_faster_whisper_signature():
    """参数拼写验证：faster-whisper 是 log_prob_threshold（openai-whisper 是
    logprob_threshold）；拼错会被静默忽略，这里用真实签名守住。"""
    import inspect

    from faster_whisper import WhisperModel

    params = set(inspect.signature(WhisperModel.transcribe).parameters)
    for name in (
        "language", "beam_size", "vad_filter", "condition_on_previous_text",
        "temperature", "no_speech_threshold", "log_prob_threshold",
        "compression_ratio_threshold",
    ):
        assert name in params, f"faster-whisper transcribe 缺少参数 {name}（拼写漂移？）"


def test_fold_repetitions():
    from src.asr.engine import fold_repetitions

    # 整段复读 → 折叠为一份
    assert fold_repetitions("谢谢观看 谢谢观看 谢谢观看") == "谢谢观看"
    assert fold_repetitions("ですね。ですね。ですね。ですね。") == "ですね。"
    # 尾部复读 → 折叠复读部分（黏着语：原句结尾的首次出现保留）
    assert fold_repetitions("今日はいい天気ですね ですね ですね ですね") == "今日はいい天気ですね"
    # 正常文本不动
    assert fold_repetitions("今日は天気がいいですね") == "今日は天気がいいですね"
    assert fold_repetitions("わくわく") == "わくわく"  # 单元 <2 字符不折叠
    assert fold_repetitions("短い") == "短い"
    assert fold_repetitions("") == ""


def test_looped_blacklist_phrase_dropped_after_folding():
    """复读循环产出的黑名单句（如结尾客套话 ×N）：折叠后仍命中 → 整段丢弃。"""
    engine = _bare_engine([
        _FakeSeg(" thank you for watching " * 4, avg_logprob=-0.1, no_speech_prob=0.1),
    ])
    assert engine.transcribe(None)[0] == ""

    # 非黑名单复读：折叠保留一份（不整段丢弃）
    engine = _bare_engine([
        _FakeSeg("とても とても とても 楽しい", avg_logprob=-0.1),
    ])
    assert engine.transcribe(None)[0] == "とても 楽しい"
