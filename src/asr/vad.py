"""可插拔 VAD 后端（模型可插拔思路延伸到 VAD 层）。

后端统一接口：callable(audio: np.ndarray) -> [{"start": int, "end": int}, ...]（样本单位）

- silero        ：faster-whisper 内置 Silero VAD（默认，零下载）
- whisper_asmr  ：Whisper-Vad-EncDec-ASMR（耳语/软语专用 ONNX，20ms 帧，
  whisper-base encoder + 非自回归 decoder；首次使用按 vad.download 配置自动下载）

模型/仓库信息全部来自 config 的 vad 节，代码零硬编码具体仓库名。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import SegmentationConfig, VADConfig

log = logging.getLogger(__name__)

# whisper_asmr 模型结构参数（随仓库 model_metadata.json：30s 块、20ms 帧）
_CHUNK_SAMPLES = 480000      # 30s @ 16kHz
_FRAME_SAMPLES = 320         # 20ms @ 16kHz
_FEATURE_FRAMES = 3000       # ONNX 输入 [1, 80, 3000]

SpeechSegment = dict  # {"start": int, "end": int}（样本单位）


def _resolve_providers(device: str, available: list[str]) -> list[str]:
    """按 vad.device 决定 onnxruntime 执行提供者优先级（仅 whisper_asmr 使用）。

    返回列表按优先级排列，onnxruntime 初始化失败会自动尝试下一个（CUDA→CPU 兜底）。
    """
    cuda, cpu = "CUDAExecutionProvider", "CPUExecutionProvider"
    if device == "cpu":
        return [cpu]
    if device in ("auto", "cuda"):
        if cuda in available:
            return [cuda, cpu]
        return [cpu]
    raise ValueError(f"未知 vad.device：{device}（可选 auto / cuda / cpu）")


def _gpu_fallback_reason(cuda_ep_available: bool, gpu_wheel_installed: bool) -> str:
    """GPU 未生效时的针对性诊断（纯函数，便于测试）。"""
    if cuda_ep_available:
        return (
            "CUDA EP 初始化失败：常见原因是装到 onnxruntime-gpu>=1.27（CUDA 13 构建），"
            "本项目 CUDA 栈为 12，请换装 onnxruntime-gpu==1.26.*；或缺 cuDNN 9 / CUDA 12 运行时 DLL"
        )
    if gpu_wheel_installed:
        return (
            "onnxruntime-gpu 已安装但模块文件被 CPU 版覆盖（uv run / uv sync 会按锁文件"
            "装回 CPU 版 onnxruntime）。修复：运行 scripts/enable-vad-gpu.ps1，"
            "之后用 .venv\\Scripts\\python 或 uv run --no-sync 启动"
        )
    return "当前是 CPU 版 onnxruntime；GPU 推理请按 README 换装 onnxruntime-gpu==1.26.*"


def _gpu_wheel_installed() -> bool:
    from importlib.metadata import PackageNotFoundError, version

    try:
        version("onnxruntime-gpu")
        return True
    except PackageNotFoundError:
        return False


class SileroVad:
    """faster-whisper 内置 Silero VAD（当前默认行为）。"""

    def __init__(self, min_silence_ms: int, pad_ms: int):
        from faster_whisper.vad import VadOptions

        self._options = VadOptions(
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=pad_ms,
        )

    def __call__(self, audio: np.ndarray) -> list[SpeechSegment]:
        from faster_whisper.vad import get_speech_timestamps

        return get_speech_timestamps(audio, vad_options=self._options)


class WhisperAsmrVad:
    """Whisper-Vad-EncDec-ASMR ONNX 后端（耳语/软语高精度，CPU 推理 ~100x 实时）。

    流式增量推理（性能优化）：
    - ONNX 输入是静态 30s 窗口，无法缩短；但完整 30s 块的概率一旦算出即定稿缓存，
      之后只对尾部未满块重算，早期音频不再重复编码
    - 重算节流：新增音频 < recompute_ms 时直接复用缓存概率（一次 whisper-base encoder
      前向约百毫秒级，每 200ms 全量重算是流式变慢的根因）
    - 切句确认不受节流影响：缓存概率显示「有语音且句尾静音已够切句」且又有新音频时立即重算，
      保证 streamer 的切句点永远基于新鲜概率，分段结果与全量模式一致（缓存中无语音则无需旁路）
    - 缓存判据 (audio_len, 前缀校验和)：pending 被截断/前移/内容变化 → 失配全量重算

    session 可注入（测试用 fake session 产出预定帧概率）。
    """

    _CHECKSUM_SAMPLES = 1600  # 前缀校验和的取样窗口（头/尾各 100ms）

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.5,
        min_silence_ms: int = 500,
        pad_ms: int = 200,
        min_speech_ms: int = 250,
        recompute_ms: int = 1000,
        device: str = "auto",
        session=None,
    ):
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.01)  # 双阈值滞后，防抖
        self.min_speech_frames = max(1, min_speech_ms // 20)
        self.min_silence_frames = max(1, min_silence_ms // 20)
        self.min_silence_samples = self.min_silence_frames * _FRAME_SAMPLES
        self.pad_frames = max(0, pad_ms // 20)
        self.recompute_samples = max(0, recompute_ms * 16)  # ms → 16kHz 样本
        if session is None:
            import onnxruntime as ort

            providers = _resolve_providers(device, ort.get_available_providers())
            session = ort.InferenceSession(str(model_path), providers=providers)
            self.active_provider = session.get_providers()[0]
            # GPU 未生效时给出针对性诊断。auto 且从未装过 GPU 版 = 用户没想用 GPU，保持安静；
            # 显式 cuda 或装过 gpu 版却回退 = 异常状态，必须警告
            gpu_wheel = _gpu_wheel_installed()
            if (
                device != "cpu"
                and self.active_provider != "CUDAExecutionProvider"
                and (device == "cuda" or gpu_wheel)
            ):
                log.warning(
                    "whisper_asmr VAD 未启用 GPU（回退 CPU）：%s",
                    _gpu_fallback_reason("CUDAExecutionProvider" in providers, gpu_wheel),
                )
            log.info("whisper_asmr VAD 推理提供者：%s", self.active_provider)
        else:
            self.active_provider = "injected"
        self._session = session
        self._input_name = session.get_inputs()[0].name
        from faster_whisper.feature_extractor import FeatureExtractor

        self._fe = FeatureExtractor()  # 与 WhisperFeatureExtractor 等价的 log-mel
        self._reset_cache()

    # ---- 增量缓存 ----

    def _reset_cache(self) -> None:
        self._cached_len = 0            # 缓存概率对应的音频长度（样本）
        self._head_sum = 0.0            # 前缀校验和（头部窗口）
        self._tail_sum = 0.0            # 前缀校验和（缓存末尾窗口）
        self._probs: np.ndarray | None = None  # 全窗口帧概率（含定稿块 + 尾部块）
        self._final_probs: list[np.ndarray] = []  # 已定稿的完整 30s 块
        self._last_speech_end = 0       # 缓存概率中最后语音帧位置（样本）

    def _checksum(self, audio: np.ndarray, length: int) -> tuple[float, float]:
        n = min(self._CHECKSUM_SAMPLES, length)
        head = float(audio[:n].sum())
        tail = float(audio[length - n : length].sum()) if length >= self._CHECKSUM_SAMPLES else head
        return head, tail

    def _is_extension(self, audio: np.ndarray) -> bool:
        """audio 是否为缓存音频的前缀扩展（pending 截断/前移/内容变化则为 False）。"""
        if self._probs is None or len(audio) < self._cached_len:
            return False
        return self._checksum(audio, self._cached_len) == (self._head_sum, self._tail_sum)

    def _forward_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """单个 30s 窗口（不足补零）→ 帧概率。"""
        if len(chunk) < _CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, _CHUNK_SAMPLES - len(chunk)))
        feat = self._fe(chunk)[:, :_FEATURE_FRAMES][np.newaxis, :, :]
        logits = self._session.run(None, {self._input_name: feat})[0][0]
        return 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    def _recompute_from(self, audio: np.ndarray, start: int) -> None:
        """从 start（30s 块边界）起重算并更新缓存；之前的定稿块直接复用。"""
        blocks: list[np.ndarray] = []
        off = start
        while off < len(audio):
            chunk = audio[off : off + _CHUNK_SAMPLES]
            probs = self._forward_chunk(chunk)
            off += _CHUNK_SAMPLES
            if off <= len(audio):
                # 完整 30s 块：定稿，此后不再重算
                self._final_probs.append(probs)
            else:
                blocks.append(probs)  # 尾部未满块：下次有新音频需重算
        all_blocks = [*self._final_probs, *blocks]
        self._probs = (
            np.concatenate(all_blocks) if all_blocks else np.empty(0, dtype=np.float32)
        )
        self._cached_len = len(audio)
        self._head_sum, self._tail_sum = self._checksum(audio, len(audio))
        speech_frames = np.nonzero(self._probs >= self.threshold)[0]
        self._last_speech_end = (
            min(int(speech_frames[-1] + 1) * _FRAME_SAMPLES, len(audio))
            if len(speech_frames)
            else 0
        )

    def _frame_probs(self, audio: np.ndarray) -> np.ndarray:
        """增量入口：缓存有效且新增不足节流间隔 → 复用缓存，否则按需重算。"""
        if not self._is_extension(audio):
            self._reset_cache()
            self._recompute_from(audio, 0)
            return self._probs
        new = len(audio) - self._cached_len
        # 切句确认：缓存中有语音、且句尾静音已够切句时才需要新鲜概率；
        # 缓存里尚无语音（_last_speech_end == 0）则没有待切句段，不受此旁路影响
        cut_imminent = (
            self._last_speech_end > 0
            and self._cached_len - self._last_speech_end >= self.min_silence_samples
        )
        if new >= self.recompute_samples or (new > 0 and cut_imminent):
            boundary = len(self._final_probs) * _CHUNK_SAMPLES
            self._recompute_from(audio, boundary)
        return self._probs

    def __call__(self, audio: np.ndarray) -> list[SpeechSegment]:
        probs = self._frame_probs(np.asarray(audio, dtype=np.float32))
        segments = _timestamps(
            probs,
            threshold=self.threshold,
            neg_threshold=self.neg_threshold,
            min_speech_frames=self.min_speech_frames,
            min_silence_frames=self.min_silence_frames,
            pad_frames=self.pad_frames,
        )
        return [
            {"start": s["start"] * _FRAME_SAMPLES, "end": s["end"] * _FRAME_SAMPLES}
            for s in segments
        ]


def _timestamps(
    probs: np.ndarray,
    threshold: float,
    neg_threshold: float,
    min_speech_frames: int,
    min_silence_frames: int,
    pad_frames: int,
) -> list[SpeechSegment]:
    """Silero 风格双阈值滞后分段（帧单位）。纯函数，便于测试。"""
    speeches: list[SpeechSegment] = []
    triggered = False
    start = 0
    temp_end = 0
    for i, p in enumerate(probs):
        if p >= threshold and not triggered:
            triggered, start = True, i
            continue
        if p < neg_threshold and triggered:
            if not temp_end:
                temp_end = i
            if i - temp_end >= min_silence_frames:
                if temp_end - start >= min_speech_frames:
                    speeches.append({"start": start, "end": temp_end})
                triggered, temp_end = False, 0
        elif p >= threshold and temp_end:
            temp_end = 0  # 语音恢复，取消暂记结束点
    if triggered and len(probs) - start >= min_speech_frames:
        speeches.append({"start": start, "end": len(probs)})

    for i, s in enumerate(speeches):
        prev_end = speeches[i - 1]["end"] if i else 0
        next_start = speeches[i + 1]["start"] if i < len(speeches) - 1 else len(probs)
        s["start"] = max(prev_end, s["start"] - pad_frames)
        s["end"] = min(next_start, s["end"] + pad_frames)
    return speeches


def ensure_vad_model(cfg: VADConfig) -> Path:
    """whisper_asmr 模型不存在时按 vad.download 配置自动下载（镜像回退）。"""
    path = Path(cfg.model_path)
    if path.is_file():
        return path
    dl = cfg.download or {}
    from ..nlp.providers.download import _download_hf, _download_modelscope

    errors: list[str] = []
    for source in ("hf", "modelscope"):
        repo, file = dl.get(f"{source}_repo"), dl.get(f"{source}_file")
        if not repo or not file:
            continue
        try:
            print(f"下载 VAD 模型：{source} {repo}/{file} → {path}")
            if source == "hf":
                _download_hf(repo, file, path)
            else:
                _download_modelscope(repo, file, path)
            print(f"✓ VAD 模型下载完成：{path}")
            return path
        except Exception as e:  # noqa: BLE001 —— 换下一个源
            errors.append(f"{source}: {type(e).__name__}: {e}")
    raise RuntimeError(
        f"VAD 模型不存在且下载失败：{path}\n  " + ("\n  ".join(errors) or "vad.download 未配置")
    )


def create_vad_fn(vad_cfg: VADConfig, seg_cfg: SegmentationConfig, pad_ms: int = 200):
    """按配置创建 VAD 后端（统一 callable 接口，供 ASRStreamer 注入）。"""
    if vad_cfg.backend == "silero":
        return SileroVad(min_silence_ms=seg_cfg.silence_ms, pad_ms=pad_ms)
    if vad_cfg.backend == "whisper_asmr":
        if not vad_cfg.model_path:
            raise ValueError("vad.backend=whisper_asmr 需要配置 vad.model_path")
        path = ensure_vad_model(vad_cfg)
        return WhisperAsmrVad(
            path,
            threshold=vad_cfg.threshold,
            min_silence_ms=seg_cfg.silence_ms,
            pad_ms=pad_ms,
            recompute_ms=vad_cfg.recompute_ms,
            device=vad_cfg.device,
        )
    raise ValueError(f"未知 VAD 后端：{vad_cfg.backend}（可选 silero / whisper_asmr）")
