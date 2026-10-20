"""faster-whisper 引擎封装（§4.2 GPU 使用策略 + 幻觉三层压制）。

- device=auto：检测 CUDA + 剩余显存，满足预算用 GPU（int8_float16），否则 CPU（int8）
- device=cpu：纯 CPU 模式
- GPU 初始化失败自动回落 CPU 并提示（§7 稳定性）
- 禁止事项由配置约束：本模块不加载任何 LLM，模型档位仅来自 config

幻觉压制（模型外围，三道防线按序串联）：
1. VAD 前置过滤：断句/裁剪在 SegmentProcessor 完成，本模块输入只有语音段
2. 解码参数约束：显式语种、condition_on_previous_text=False、置信度阈值
   与温度回退链原样传入 transcribe()（低质量窗口由引擎内部门控+重试）
3. 后处理兜底：段级元数据再过滤 + 幻觉短语黑名单 + 连续重复段折叠
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import ASRConfig

log = logging.getLogger(__name__)

# 各档位模型首次下载的近似大小（用于提示用户耐心等待）
MODEL_DOWNLOAD_SIZE_MB = {
    "tiny": 75,
    "base": 145,
    "small": 466,
    "medium": 1500,
}

# os.add_dll_directory 返回的句柄必须保持引用，否则目录会被移出搜索路径
_DLL_DIR_HANDLES: list = []


def _find_cuda_dll_dirs(roots: list[str]) -> list[Path]:
    """在 site-packages 根目录列表中查找 nvidia pip 包的 DLL 目录（nvidia/*/bin 通配）。

    ctranslate2 需要 cublas/cudnn；onnxruntime-gpu 的 CUDA EP 另外需要
    cuda_runtime/cufft/curand 等，通配扫描一次覆盖全部（含未来新增组件）。
    """
    found: list[Path] = []
    for root in roots:
        found.extend(sorted(Path(root).glob("nvidia/*/bin")))
    return [d for d in found if d.is_dir()]


def setup_cuda_dll_dirs() -> list[str]:
    """Windows：把 pip 安装的 nvidia-*-cu12 包的 DLL 目录加入搜索路径。

    ctranslate2 的 CUDA 支持依赖 cuBLAS 12 / cuDNN 9 的动态库，
    pip 包 nvidia-cublas-cu12 / nvidia-cudnn-cu12 会把 DLL 装在
    site-packages/nvidia/<name>/bin 下，但不在系统 PATH 中。
    返回成功注册的目录列表（空列表 = 非 Windows 或未安装这些包）。
    """
    if sys.platform != "win32":
        return []
    import site

    roots: list[str] = []
    try:
        roots += site.getsitepackages()
    except Exception:
        pass
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass

    added: list[str] = []
    for d in _find_cuda_dll_dirs(roots):
        s = str(d)
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(s))
            os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")
            added.append(s)
        except (OSError, AttributeError):
            log.debug("注册 DLL 目录失败：%s", s, exc_info=True)
    if added:
        log.info("已注册 CUDA DLL 目录：%s", added)
    return added


@dataclass
class EngineInfo:
    device: str  # "cuda" | "cpu"
    model_size: str
    compute_type: str
    downgraded: bool = False  # 是否发生了降级（用于 UI 提示）
    reason: str = ""  # 降级原因


def _cuda_free_vram_mb() -> int | None:
    """查询 GPU 空闲显存（MB）。pynvml 不可用或无 GPU 时返回 None。"""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(mem.free // (1024 * 1024))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _model_cached(model_size: str) -> bool:
    """模型是否已下载到本地 HF 缓存（仅启发式判断，用于下载提示）。"""
    if os.path.sep in model_size or "/" in model_size:
        return Path(model_size).exists()  # 本地路径形式的模型
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshot_dir = hf_home / "hub" / f"models--Systran--faster-whisper-{model_size}" / "snapshots"
    try:
        for snap in snapshot_dir.iterdir():
            # 关键权重文件存在即视为已缓存
            if (snap / "model.bin").is_file() or any(snap.glob("*.bin")):
                return True
    except OSError:
        pass
    return False


def resolve_device(cfg: ASRConfig) -> tuple[str, str, bool, str]:
    """根据配置与硬件决定 (device, compute_type, downgraded, reason)。"""
    if cfg.device == "cpu":
        return "cpu", cfg.compute_type or "int8", False, "配置指定纯 CPU 模式"

    if not _cuda_available():
        return "cpu", "int8", cfg.device == "cuda", "未检测到可用 CUDA 设备"

    free = _cuda_free_vram_mb()
    if free is not None and free < cfg.vram_min_free_mb:
        return (
            "cpu",
            "int8",
            True,
            f"剩余显存 {free}MB 低于保留线 {cfg.vram_min_free_mb}MB，已降级到 CPU 模式",
        )
    return "cuda", cfg.compute_type or "int8_float16", False, ""


def _is_network_error(exc: BaseException) -> bool:
    """异常链中是否包含网络连接错误（模型下载失败的典型原因）。"""
    NETWORK_NAMES = {
        "LocalEntryNotFoundError", "ConnectTimeout", "ConnectError",
        "ReadTimeout", "ConnectError", "TimeoutError",
    }
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in NETWORK_NAMES:
            return True
        if isinstance(cur, OSError) and ("10060" in str(cur) or "timed out" in str(cur)):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


MIRROR_ENDPOINT = "https://hf-mirror.com"


_LOOP_RE = re.compile(r"(.{2,}?)(?: ?\1){2,}", re.DOTALL)


def fold_repetitions(text: str) -> str:
    """折叠连续重复段（自回归复读循环的产物）。

    任意位置（句首/句中/句尾）的 unit × k（k≥3）折叠为一份；单元最短 2 字符，
    减少拟声词/正常重复（"わくわく"）的误折叠。可处理空格分隔与无空格形态。
    """
    norm = re.sub(r"\s+", " ", text).strip()
    if len(norm) < 8:
        return norm
    while True:
        m = _LOOP_RE.search(norm)
        if not m:
            return norm
        unit = m.group(1).strip()
        log.info("折叠复读段：%r ×%d", unit, norm[m.start():m.end()].count(m.group(1)))
        # 单元与前文无空格（日语等黏着语）：首次出现属于原句结尾，只折叠其后的复读
        if m.start() > 0 and norm[m.start() - 1] != " ":
            norm = (norm[: m.start()] + m.group(1) + norm[m.end():]).strip()
        else:
            norm = re.sub(
                r"\s+", " ", (norm[: m.start()] + " " + unit + " " + norm[m.end():])
            ).strip()


class ModelLoadError(RuntimeError):
    """模型既无法下载也无法加载时的最终错误（附带用户可操作的指引）。"""


def _download_via_mirror(model_size: str) -> str:
    """通过 hf-mirror 镜像下载模型，返回本地路径。"""
    from huggingface_hub import HfApi

    api = HfApi(endpoint=MIRROR_ENDPOINT)
    return api.snapshot_download(f"Systran/faster-whisper-{model_size}")


class ASREngine:
    """faster-whisper 模型持有者与转写入口。

    progress: 进度回调（默认打印到控制台），各加载阶段都会汇报，便于定位卡点。
    """

    def __init__(self, cfg: ASRConfig, progress=None):
        import numpy as np
        from faster_whisper import WhisperModel  # 延迟导入：加快 CLI 启动

        self._progress = progress or (lambda msg: print(msg, flush=True))
        self.cfg = cfg

        self._progress("[1/4] 检测 GPU 与显存…")
        dll_dirs = setup_cuda_dll_dirs()
        if sys.platform == "win32" and not dll_dirs:
            log.debug("未发现 pip 版 CUDA 运行时目录（nvidia-cublas-cu12 / nvidia-cudnn-cu12）")
        t0 = time.monotonic()
        device, compute_type, downgraded, reason = resolve_device(cfg)
        free = _cuda_free_vram_mb()
        vram_msg = f"，空闲显存 {free}MB" if free is not None else ""
        self._progress(
            f"      → 目标设备 {device}（{compute_type}）{vram_msg}，耗时 {time.monotonic() - t0:.1f}s"
        )

        if not _model_cached(cfg.model_size):
            size_mb = MODEL_DOWNLOAD_SIZE_MB.get(cfg.model_size)
            hint = f"约 {size_mb}MB" if size_mb else "数百 MB"
            self._progress(
                f"      模型 '{cfg.model_size}' 首次使用需下载（{hint}），"
                "下载期间下方应有进度条；国内网络建议设置 "
                "HF_ENDPOINT=https://hf-mirror.com 与 HF_HUB_DISABLE_XET=1"
            )

        def _load(dev: str, ct: str):
            t = time.monotonic()
            self._progress(f"[2/4] 加载模型 {cfg.model_size}（device={dev}, {ct}）…")
            try:
                model = WhisperModel(cfg.model_size, device=dev, compute_type=ct)
            except Exception as e:
                # 直连 HF 失败且模型未缓存：自动切换 hf-mirror 镜像重试一次
                if (
                    _is_network_error(e)
                    and not _model_cached(cfg.model_size)
                    and "/" not in cfg.model_size
                    and not os.environ.get("HF_ENDPOINT")
                ):
                    self._progress(
                        f"      ✗ 直连 HuggingFace 失败（{type(e).__name__}），"
                        f"自动切换到镜像 {MIRROR_ENDPOINT} 重试下载…"
                    )
                    try:
                        local_path = _download_via_mirror(cfg.model_size)
                    except Exception as e2:
                        raise ModelLoadError(
                            f"模型 '{cfg.model_size}' 下载失败（直连与镜像均不可用）。\n"
                            "请检查网络，或手动设置镜像后重试：\n"
                            "  PowerShell: $env:HF_ENDPOINT=\"https://hf-mirror.com\"; "
                            "$env:HF_HUB_DISABLE_XET=\"1\"\n"
                            "  Linux/WSL:  export HF_ENDPOINT=https://hf-mirror.com "
                            "HF_HUB_DISABLE_XET=1"
                        ) from e2
                    model = WhisperModel(local_path, device=dev, compute_type=ct)
                else:
                    raise
            self._progress(f"      → 模型加载完成，耗时 {time.monotonic() - t:.1f}s")
            # 预热自测：部分环境（如缺 cuBLAS 的 WSL）构造成功但首次推理才失败
            t = time.monotonic()
            self._progress(f"[3/4] 预热自检（{dev}）…")
            model.transcribe(np.zeros(1600, dtype=np.float32), beam_size=1)
            self._progress(f"      → 自检通过，耗时 {time.monotonic() - t:.1f}s")
            return model

        try:
            self._model = _load(device, compute_type)
        except Exception as e:
            if device == "cpu":
                self._progress(f"      ✗ CPU 模型加载失败：{e!r}")
                raise
            self._progress(f"      ✗ GPU 初始化/自检失败：{e!r}")
            if sys.platform == "win32" and not dll_dirs and "dll" in str(e).lower():
                self._progress(
                    "      提示：缺少 CUDA 运行时 DLL。请运行 uv sync 安装 "
                    "nvidia-cublas-cu12 与 nvidia-cudnn-cu12 后重试"
                )
            log.warning("GPU 初始化/自检失败（%s），自动回落 CPU int8", e)
            self._progress("      自动回落到 CPU（int8）重新加载…")
            device, compute_type, downgraded, reason = (
                "cpu", "int8", True, f"GPU 初始化失败：{e}",
            )
            try:
                self._model = _load("cpu", "int8")
            except ModelLoadError:
                raise
            except Exception as e2:
                raise ModelLoadError(
                    f"CPU 兜底加载也失败：{e2}\n"
                    "若为网络问题，请设置镜像后重试（详见 README「快速开始」）。"
                ) from e2
        self._progress("[4/4] ASR 引擎就绪")
        self.info = EngineInfo(
            device=device,
            model_size=cfg.model_size,
            compute_type=compute_type,
            downgraded=downgraded,
            reason=reason,
        )

    def transcribe(
        self, audio, language: str | None = None
    ) -> tuple[str, float, float]:
        """转写一段已断句的音频，返回 (文本, 起始秒, 结束秒)（相对该段音频）。

        参数拼写注意（faster-whisper vs openai-whisper）：log_prob_threshold
        拼错成 logprob_threshold 会被静默忽略——tests/test_engine.py 有签名级验证。
        置信度阈值传给引擎原生门控（触发温度回退重试）；本层另做事后过滤兜底。
        """
        segments, _info = self._model.transcribe(
            audio,
            language=language,  # 显式语种：静音段不走自动检测
            beam_size=self.cfg.beam_size,
            vad_filter=False,  # 断句已在 streamer 完成（VAD 前置过滤）
            condition_on_previous_text=self.cfg.condition_on_previous,
            temperature=list(self.cfg.temperature_fallback),  # 温度回退链
            no_speech_threshold=self.cfg.no_speech_threshold,
            log_prob_threshold=self.cfg.log_prob_threshold,
            compression_ratio_threshold=self.cfg.compression_ratio_threshold,
        )
        texts: list[str] = []
        start: float | None = None
        end = 0.0
        for seg in segments:
            if self._is_hallucination(seg):
                continue
            texts.append(seg.text)
            if start is None:
                start = seg.start
            end = seg.end
        text = fold_repetitions("".join(texts))
        if text and self._phrase_hit(text):
            # 复读循环的产物折叠后仍命中黑名单：整段丢弃
            log.info("丢弃折叠后仍命中黑名单的段落：%r", text)
            return "", (start or 0.0), end
        return text, (start or 0.0), end

    def _phrase_hit(self, text: str) -> bool:
        """幻觉短语黑名单：归一化（去空白、全小写）后包含匹配。"""
        normalized = "".join(text.split()).lower()
        return any(
            "".join(p.split()).lower() in normalized
            for p in self.cfg.hallucination_phrases
        )

    def _is_hallucination(self, seg) -> bool:
        """Whisper 幻觉段判定：质量门控（与原版跳过规则一致）+ 幻觉短语黑名单。"""
        no_speech = getattr(seg, "no_speech_prob", 0.0) or 0.0
        avg_logprob = getattr(seg, "avg_logprob", 0.0) or 0.0
        compression = getattr(seg, "compression_ratio", 0.0) or 0.0
        quality_drop = (
            (no_speech > self.cfg.no_speech_threshold
             and avg_logprob < self.cfg.log_prob_threshold)
            or compression > self.cfg.compression_ratio_threshold
        )
        phrase_hit = self._phrase_hit(seg.text)
        if quality_drop or phrase_hit:
            log.info(
                "丢弃疑似幻觉段：%r（no_speech=%.2f, logprob=%.2f, compression=%.1f, 短语命中=%s）",
                seg.text, no_speech, avg_logprob, compression, phrase_hit,
            )
            return True
        return False
