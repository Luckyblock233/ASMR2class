"""运行时指标快照层（M3 后续：C 栏状态显示的数据源）。

采集线程（音频回调/ASR/翻译 worker）只通过 RuntimeMetrics 写计数与耗时快照，
不直接访问任何 Qt 控件；UI 每秒读取一次 MetricsSnapshot 刷新显示。

系统资源采样由 SystemSampler 提供：
- CPU/RAM 用 psutil；未安装时报 None（UI 显示 N/A）
- GPU/VRAM 用 NVML（nvidia-ml-py）；无 NVIDIA/NVML 时报 None
- 采样只被 UI 定时器低频调用（1s），不做网络请求
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemSample:
    """一次系统资源采样；None = 该项不可用（显示 N/A）。"""

    cpu_percent: float | None = None
    ram_percent: float | None = None
    ram_used_mb: float | None = None
    gpu_percent: float | None = None
    vram_used_mb: float | None = None
    vram_free_mb: float | None = None


@dataclass(frozen=True)
class MetricsSnapshot:
    """某一时刻的运行时指标快照（采集线程写、UI 读，均为原子整体替换）。"""

    rms: float = 0.0                    # 最近音频块 RMS 电平
    vad_active: bool = False            # 最近 VAD 判定是否有语音
    asr_count: int = 0                  # ASR 完成句数
    asr_last_ms: float = 0.0            # 最近一句 ASR 耗时
    asr_avg_ms: float = 0.0             # ASR 平均耗时
    translate_count: int = 0            # 翻译完成句数（含失败）
    translate_failed: int = 0           # 翻译失败句数
    translate_last_ms: float = 0.0      # 最近翻译块耗时
    translate_avg_ms: float = 0.0       # 翻译块平均耗时
    system: SystemSample = SystemSample()


class RuntimeMetrics:
    """线程安全的指标采集器：写方只更新计数，读方取不可变快照。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rms = 0.0
        self._vad_active = False
        self._asr_count = 0
        self._asr_total_ms = 0.0
        self._asr_last_ms = 0.0
        self._translate_count = 0
        self._translate_failed = 0
        self._translate_total_ms = 0.0
        self._translate_last_ms = 0.0

    # ---- 采集线程写入 ----

    def record_audio_block(self, rms: float) -> None:
        with self._lock:
            self._rms = rms

    def set_vad_active(self, active: bool) -> None:
        with self._lock:
            self._vad_active = bool(active)

    def record_asr(self, elapsed_ms: float) -> None:
        with self._lock:
            self._asr_count += 1
            self._asr_total_ms += elapsed_ms
            self._asr_last_ms = elapsed_ms

    def record_translation(self, elapsed_ms: float, ok: bool) -> None:
        with self._lock:
            self._translate_count += 1
            if not ok:
                self._translate_failed += 1
            self._translate_total_ms += elapsed_ms
            self._translate_last_ms = elapsed_ms

    # ---- UI 读取 ----

    def snapshot(self, system: SystemSample | None = None) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                rms=self._rms,
                vad_active=self._vad_active,
                asr_count=self._asr_count,
                asr_last_ms=self._asr_last_ms,
                asr_avg_ms=self._asr_total_ms / self._asr_count if self._asr_count else 0.0,
                translate_count=self._translate_count,
                translate_failed=self._translate_failed,
                translate_last_ms=self._translate_last_ms,
                translate_avg_ms=(
                    self._translate_total_ms / self._translate_count
                    if self._translate_count
                    else 0.0
                ),
                system=system or SystemSample(),
            )


class SystemSampler:
    """CPU/RAM（psutil）与 GPU/VRAM（NVML）采样；任一不可用则对应字段为 None。

    只应被低频（约 1s）调用；所有异常内部吞掉并降级为 N/A，不影响主流程。
    """

    def __init__(self, gpu_index: int = 0) -> None:
        self._gpu_index = gpu_index
        self._psutil = None
        self._nvml = None
        self._nvml_handle = None
        try:
            import psutil

            psutil.cpu_percent(interval=None)  # 首次调用预热（返回自上次调用的均值）
            self._psutil = psutil
        except Exception:  # noqa: BLE001 —— 无 psutil 时 CPU/RAM 显示 N/A
            log.info("psutil 不可用，CPU/RAM 指标将显示 N/A")
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:  # noqa: BLE001 —— 无 NVIDIA/NVML 时 GPU 指标显示 N/A
            self._nvml = None
            self._nvml_handle = None
            log.info("NVML 不可用（无 NVIDIA GPU 或未装驱动），GPU/VRAM 指标将显示 N/A")

    @property
    def gpu_available(self) -> bool:
        return self._nvml_handle is not None

    def sample(self) -> SystemSample:
        cpu = ram_percent = ram_used = None
        gpu = vram_used = vram_free = None
        if self._psutil is not None:
            try:
                cpu = float(self._psutil.cpu_percent(interval=None))
                mem = self._psutil.virtual_memory()
                ram_percent = float(mem.percent)
                ram_used = float(mem.used) / (1024 * 1024)
            except Exception:  # noqa: BLE001
                pass
        if self._nvml is not None and self._nvml_handle is not None:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu = float(util.gpu)
                info = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                vram_used = float(info.used) / (1024 * 1024)
                vram_free = float(info.free) / (1024 * 1024)
            except Exception:  # noqa: BLE001
                pass
        return SystemSample(
            cpu_percent=cpu,
            ram_percent=ram_percent,
            ram_used_mb=ram_used,
            gpu_percent=gpu,
            vram_used_mb=vram_used,
            vram_free_mb=vram_free,
        )

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
            self._nvml = None
            self._nvml_handle = None
