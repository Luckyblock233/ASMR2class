"""可插拔 VAD 后端测试（silero / whisper_asmr）。"""

import numpy as np
import pytest

from src.asr.vad import (
    SileroVad,
    WhisperAsmrVad,
    _gpu_fallback_reason,
    _resolve_providers,
    _timestamps,
    create_vad_fn,
)
from src.config import SegmentationConfig, VADConfig, load_config


def _logits_from_probs(probs: np.ndarray) -> np.ndarray:
    """概率 → logits（fake session 输出用）。"""
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class FakeSession:
    """按脚本逐块返回帧 logits（每块 1500 帧）。"""

    def __init__(self, blocks: list[np.ndarray]):
        self._blocks = [_logits_from_probs(b) for b in blocks]
        self.calls = 0

    def get_inputs(self):
        class _In:
            name = "input_features"

        return [_In()]

    def run(self, _outputs, feed):
        assert feed["input_features"].shape == (1, 80, 3000)  # ONNX 输入形状契约
        block = self._blocks[min(self.calls, len(self._blocks) - 1)]
        self.calls += 1
        return [block[np.newaxis, :]]


def make_vad(blocks, **kw) -> WhisperAsmrVad:
    return WhisperAsmrVad.__new__(WhisperAsmrVad).__class__(
        "dummy.onnx", session=FakeSession(blocks), **kw
    )


def test_whisper_asmr_speech_silence_segments():
    """1s 静音 + 1s 语音 + 1s 静音（3s 音频补零到 30s）：检出中间语音段。"""
    probs = np.full(1500, 0.01)
    probs[50:100] = 0.9  # 第 50-100 帧（1s-2s）为语音
    vad = make_vad([probs], threshold=0.5, min_silence_ms=500, pad_ms=200, min_speech_ms=250)
    segments = vad(np.zeros(16000 * 3, dtype=np.float32))
    assert len(segments) == 1
    # 帧 → 样本换算 + padding：起点约 (50-10)*320，终点约 (100+10)*320
    assert segments[0]["start"] == 40 * 320
    assert segments[0]["end"] == 110 * 320


def test_whisper_asmr_hysteresis_ignores_brief_dip():
    """短暂低于负阈值（< min_silence）不切段：滞后窗口内语音恢复。"""
    probs = np.full(1500, 0.01)
    probs[50:200] = 0.9
    probs[100:110] = 0.05  # 200ms 短暂低谷（< 500ms min_silence）
    vad = make_vad([probs], threshold=0.5, min_silence_ms=500, pad_ms=0, min_speech_ms=250)
    segments = vad(np.zeros(16000, dtype=np.float32))
    assert len(segments) == 1
    assert segments[0]["start"] == 50 * 320
    assert segments[0]["end"] == 200 * 320


def test_whisper_asmr_short_speech_dropped():
    """短于 min_speech 的语音突发被丢弃（防喷麦/爆音误触发）。"""
    probs = np.full(1500, 0.01)
    probs[50:55] = 0.9  # 100ms，小于默认 min_speech 250ms
    vad = make_vad([probs], threshold=0.5, min_silence_ms=100, pad_ms=0, min_speech_ms=250)
    assert vad(np.zeros(16000, dtype=np.float32)) == []


def test_timestamps_pure_function():
    """分段纯函数：双阈值 + 最小静音 + padding 邻段钳制。"""
    probs = np.full(200, 0.01)
    probs[20:40] = 0.9
    probs[60:80] = 0.9
    segs = _timestamps(probs, threshold=0.5, neg_threshold=0.35,
                       min_speech_frames=5, min_silence_frames=10, pad_frames=5)
    assert segs == [{"start": 15, "end": 45}, {"start": 55, "end": 85}]


def test_config_vad_section(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text("""\
vad:
  backend: whisper_asmr
  threshold: 0.6
  model_path: "models/vad.onnx"
  download: {hf_repo: "org/repo", hf_file: "model.onnx"}
""", encoding="utf-8")
    cfg = load_config(tmp_path)
    assert cfg.vad.backend == "whisper_asmr"
    assert cfg.vad.threshold == 0.6
    assert cfg.vad.model_path == str(tmp_path / "models" / "vad.onnx")  # 相对路径锚定项目根
    assert cfg.vad.download["hf_repo"] == "org/repo"


def test_config_vad_defaults():
    cfg = VADConfig()
    assert cfg.backend == "silero"
    assert cfg.device == "auto"


# ---- 推理设备选择（whisper_asmr GPU/CPU）----

def test_resolve_providers_auto_prefers_cuda():
    providers = _resolve_providers("auto", ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]  # CUDA 优先，CPU 兜底


def test_resolve_providers_auto_cpu_only_wheel():
    # CPU 版 onnxruntime（无 CUDA EP）→ 静默用 CPU
    assert _resolve_providers("auto", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_resolve_providers_explicit_cpu():
    assert _resolve_providers("cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_resolve_providers_cuda_requested_but_unavailable():
    # 显式要 cuda 但 wheel 不支持 → 回退 CPU（警告文案由 _gpu_fallback_reason 负责）
    assert _resolve_providers("cuda", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_resolve_providers_unknown_device():
    with pytest.raises(ValueError, match="未知 vad.device"):
        _resolve_providers("tpu", ["CPUExecutionProvider"])


def test_gpu_fallback_reason_cuda_ep_present():
    # CUDA EP 存在但初始化失败 → 提示版本钉 1.26（CUDA 13 构建不匹配）
    msg = _gpu_fallback_reason(cuda_ep_available=True, gpu_wheel_installed=True)
    assert "1.26" in msg and "CUDA 13" in msg


def test_gpu_fallback_reason_gpu_wheel_overwritten():
    # 装过 gpu 版但无 CUDA EP → 文件被 uv sync 装回的 CPU 版覆盖
    msg = _gpu_fallback_reason(cuda_ep_available=False, gpu_wheel_installed=True)
    assert "覆盖" in msg and "enable-vad-gpu" in msg


def test_gpu_fallback_reason_plain_cpu_wheel():
    msg = _gpu_fallback_reason(cuda_ep_available=False, gpu_wheel_installed=False)
    assert "onnxruntime-gpu==1.26" in msg


def test_config_vad_device(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text("""\
vad:
  backend: whisper_asmr
  device: cuda
""", encoding="utf-8")
    assert load_config(tmp_path).vad.device == "cuda"


def test_create_vad_fn_silero():
    vad = create_vad_fn(VADConfig(), SegmentationConfig(silence_ms=300))
    assert isinstance(vad, SileroVad)
    assert vad._options.min_silence_duration_ms == 300  # 断句配置透传


def test_create_vad_fn_unknown_backend():
    with pytest.raises(ValueError, match="未知 VAD 后端"):
        create_vad_fn(VADConfig(backend="no-such"), SegmentationConfig())


def test_create_vad_fn_whisper_asmr_needs_model_path():
    with pytest.raises(ValueError, match="model_path"):
        create_vad_fn(VADConfig(backend="whisper_asmr"), SegmentationConfig())


# ---- 增量推理（流式性能优化）回归测试 ----

_SR = 16000


def _noise(seconds, seed=42):
    """确定性白噪声（宽带能量，fake session 可稳定区分语音/静音）。"""
    rng = np.random.default_rng(seed)
    return (0.3 * rng.standard_normal(int(seconds * _SR))).astype(np.float32)


class DeterministicSession:
    """由输入特征确定性推导帧概率（能量高 → 语音），保证同一窗口永远输出相同概率。"""

    # 宽带白噪声帧能量 ≈ +68，静音 ≈ -70：阈值取 0 稳定区分（特征经窗口最大值归一，
    # 绝对能量随窗口内容漂移，故不用绝对阈值标定）
    def __init__(self):
        self.forwards = 0
        self._col_threshold = 0.0

    def get_inputs(self):
        class _In:
            name = "input_features"

        return [_In()]

    def run(self, _outputs, feed):
        self.forwards += 1
        colsum = feed["input_features"][0].sum(axis=0)  # (3000,)
        frames = colsum.reshape(1500, 2).mean(axis=1)
        probs = np.where(frames > self._col_threshold, 0.9, 0.01)
        p = np.clip(probs, 1e-6, 1 - 1e-6)
        return [(np.log(p / (1 - p)))[np.newaxis, :]]


def make_inc_vad(session, **kw):
    args = dict(threshold=0.5, min_silence_ms=500, pad_ms=200, min_speech_ms=250,
                recompute_ms=1000, session=session)
    args.update(kw)
    return WhisperAsmrVad("dummy.onnx", **args)


def test_incremental_matches_full_recompute():
    """增量模式（200ms 步进投喂）与全量重算模式输出的分段结果完全一致。"""
    audio = np.concatenate([_noise(2.0), np.zeros(int(1.5 * _SR), dtype=np.float32)])
    full = make_inc_vad(DeterministicSession())(audio)

    session = DeterministicSession()
    inc = make_inc_vad(session)
    result = []
    for end in range(int(0.2 * _SR), len(audio) + 1, int(0.2 * _SR)):
        result = inc(audio[:end])
    assert result == full
    # 节流生效：前向次数远少于调用次数（17 次调用）
    assert session.forwards < 17 / 2


def test_incremental_cache_hit_no_forward():
    """同一音频重复调用：缓存命中，不触发前向。"""
    session = DeterministicSession()
    vad = make_inc_vad(session)
    audio = _noise(1.0)
    vad(audio)
    assert session.forwards == 1
    assert vad(audio) == vad(audio)
    assert session.forwards == 1


def test_incremental_throttle_interval():
    """新增音频不足 recompute_ms 复用缓存；累计达到间隔才重算。"""
    session = DeterministicSession()
    vad = make_inc_vad(session, recompute_ms=1000)
    vad(_noise(1.0))
    vad(_noise(1.5))  # +0.5s < 1s：缓存
    vad(_noise(1.8))  # +0.3s < 1s：缓存
    assert session.forwards == 1
    vad(_noise(2.2))  # 累计 +1.2s ≥ 1s：重算
    assert session.forwards == 2


def test_incremental_cut_confirm_bypasses_throttle():
    """句尾静音够切句时：即使新增不足间隔也立即重算（切句点必须基于新鲜概率）。"""
    session = DeterministicSession()
    vad = make_inc_vad(session, recompute_ms=10000, min_silence_ms=500)
    speech_then_silence = np.concatenate([_noise(1.0), np.zeros(int(0.6 * _SR), dtype=np.float32)])
    vad(speech_then_silence)  # 首次：缓存显示句尾静音 0.6s ≥ 0.5s
    assert session.forwards == 1
    vad(np.concatenate([speech_then_silence, np.zeros(int(0.1 * _SR), dtype=np.float32)]))
    assert session.forwards == 2  # 新增仅 0.1s，但切句确认触发重算


def test_incremental_no_speech_no_cut_bypass():
    """纯静音/无语音音频：没有待切句段，切句确认旁路不触发，节流正常生效。"""
    session = DeterministicSession()
    vad = make_inc_vad(session, recompute_ms=10000)
    silence = np.zeros(int(3.0 * _SR), dtype=np.float32)
    vad(silence[:_SR])
    assert session.forwards == 1
    for end in range(_SR + 3200, len(silence) + 1, 3200):
        vad(silence[:end])
    assert session.forwards == 1  # 全程无语音 → 无切句确认 → 无重算


def test_incremental_cache_invalidation():
    """pending 前移/内容变化：前缀校验和失配 → 全量重算，结果仍与全量模式一致。"""
    audio_a = np.concatenate([_noise(1.0, seed=1), np.zeros(_SR, dtype=np.float32)])
    audio_b = np.concatenate([_noise(1.0, seed=2), np.zeros(_SR, dtype=np.float32)])
    session = DeterministicSession()
    vad = make_inc_vad(session)
    vad(audio_a)
    # 同长度但内容变化 → 失配重算
    assert vad(audio_b) == make_inc_vad(DeterministicSession())(audio_b)
    # pending 前移（变短）→ 失配重算
    short = audio_b[: int(0.5 * _SR)]
    assert vad(short) == make_inc_vad(DeterministicSession())(short)


def test_incremental_chunk_boundary_finalized():
    """超过 30s 的音频：已完整块定稿不再重算，只重算尾部未满块。"""
    session = DeterministicSession()
    vad = make_inc_vad(session, recompute_ms=0)  # 每次都允许重算
    vad(_noise(35.0))   # 1 个完整块 + 5s 尾块 → 2 次前向
    assert session.forwards == 2
    vad(_noise(36.0))   # 新增 1s：只重算尾块（完整块已定稿）
    assert session.forwards == 3
    vad(_noise(65.0))   # 跨过第二个完整块：第 2 块定稿 + 新尾块 → 2 次前向
    assert session.forwards == 5
