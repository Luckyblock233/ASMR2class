"""模型下载镜像回退测试（直连失败自动切 hf-mirror）。"""

import pytest

from src.asr.engine import _is_network_error


def make_chain():
    """模拟 huggingface_hub 的异常链：LocalEntryNotFoundError ← ConnectTimeout。"""
    import httpx

    inner = httpx.ConnectTimeout("timed out")
    try:
        raise inner
    except httpx.ConnectTimeout as e:
        try:
            from huggingface_hub.errors import LocalEntryNotFoundError

            raise LocalEntryNotFoundError("cannot find snapshot") from e
        except LocalEntryNotFoundError as e2:
            return e2


def test_network_error_chain_detected():
    assert _is_network_error(make_chain())


def test_network_error_winerror_text():
    assert _is_network_error(OSError("[WinError 10060] 连接尝试失败"))


def test_non_network_error_not_detected():
    assert not _is_network_error(RuntimeError("Library cublas64_12.dll is not found"))
    assert not _is_network_error(ValueError("bad config"))


def test_mirror_fallback_on_network_error(monkeypatch, tmp_path):
    """直连失败 + 模型未缓存 + 未设 HF_ENDPOINT → 自动走镜像下载并用本地路径加载。"""
    import src.asr.engine as eng
    from src.asr.engine import ASREngine
    from src.config import ASRConfig

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(eng, "resolve_device", lambda cfg: ("cpu", "int8", False, ""))
    monkeypatch.setattr(eng, "_cuda_free_vram_mb", lambda: None)
    monkeypatch.setattr(eng, "_model_cached", lambda size: False)

    calls = {"whisper": [], "mirror": []}

    class FakeWhisperModel:
        def __init__(self, size_or_path, device=None, compute_type=None):
            calls["whisper"].append(size_or_path)
            if len(calls["whisper"]) == 1:
                raise OSError("[WinError 10060] 连接超时")  # 直连失败

        def transcribe(self, audio, beam_size=1):
            return iter([]), None

    monkeypatch.setattr(eng, "_download_via_mirror",
                        lambda size: calls["mirror"].append(size) or "/fake/medium")
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
    # engine 内部是 `from faster_whisper import WhisperModel` 延迟导入，替换模块属性即可

    engine = ASREngine(ASRConfig(device="cpu", model_size="medium"), progress=lambda m: None)
    assert calls["mirror"] == ["medium"]
    assert calls["whisper"] == ["medium", "/fake/medium"]  # 第二次用镜像下载的本地路径
    assert engine.info.device == "cpu"


def test_mirror_fallback_also_fails_gives_guidance(monkeypatch):
    import src.asr.engine as eng
    from src.asr.engine import ASREngine, ModelLoadError
    from src.config import ASRConfig

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(eng, "resolve_device", lambda cfg: ("cpu", "int8", False, ""))
    monkeypatch.setattr(eng, "_cuda_free_vram_mb", lambda: None)
    monkeypatch.setattr(eng, "_model_cached", lambda size: False)

    class FailWhisper:
        def __init__(self, *a, **kw):
            raise OSError("timed out")

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FailWhisper)
    monkeypatch.setattr(eng, "_download_via_mirror",
                        lambda size: (_ for _ in ()).throw(OSError("mirror down")))

    with pytest.raises(ModelLoadError, match="HF_ENDPOINT"):
        ASREngine(ASRConfig(device="cpu", model_size="medium"), progress=lambda m: None)
