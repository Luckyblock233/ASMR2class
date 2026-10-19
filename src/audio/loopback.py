"""WASAPI loopback 系统音内录（仅 Windows，pyaudiowpatch）。

pyaudiowpatch 只在 Windows 提供；非 Windows 环境 start() 会给出明确指引（§9 风险对策）。
loopback 设备原生采样率/声道数由系统决定，此处统一重采样为 16kHz mono。
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from .base import AudioCallback

log = logging.getLogger(__name__)

LOOPBACK_GUIDE = """\
loopback（系统音内录）不可用。请依次排查：
  1. 确认运行在 Windows 且已安装 pyaudiowpatch：uv sync
  2. 在「设置 > 系统 > 声音」中确认默认输出设备工作正常
  3. 部分声卡需开启「立体声混音」或更换输出设备
  4. 仍不可用时可在 config.yaml 将 audio.source 改为 mic 使用麦克风兜底
"""


class LoopbackUnavailableError(RuntimeError):
    pass


def list_loopback_devices() -> list[dict]:
    """列出所有可内录的输出设备（WASAPI loopback）。仅 Windows。"""
    if sys.platform != "win32":
        raise LoopbackUnavailableError(LOOPBACK_GUIDE)
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as e:
        raise LoopbackUnavailableError(LOOPBACK_GUIDE) from e
    pa = pyaudio.PyAudio()
    try:
        default_name = ""
        try:
            wasapi = pa.get_host_api_info_by_type(pa.paWASAPI)
            default_name = pa.get_device_info_by_index(
                wasapi["defaultOutputDevice"]
            )["name"]
        except Exception:
            pass
        devices = []
        for d in pa.get_loopback_device_info_generator():
            prefix = default_name.split("(")[0].strip()
            devices.append(
                {
                    "index": d["index"],
                    "name": d["name"],
                    "channels": d["maxInputChannels"],
                    "sample_rate": int(d["defaultSampleRate"]),
                    "is_default": bool(prefix) and d["name"].startswith(prefix),
                }
            )
        return devices
    finally:
        pa.terminate()


def resample_to_16k_mono(data: np.ndarray, src_rate: int, channels: int) -> np.ndarray:
    """线性插值重采样到 16kHz 单声道（轻量实现，避免引入 scipy）。"""
    if data.ndim == 1:
        data = data.reshape(-1, channels) if channels > 1 else data
    if channels > 1 and data.ndim == 2:
        data = data.mean(axis=1)
    data = np.ascontiguousarray(data, dtype=np.float32)
    if src_rate == 16000:
        return data
    n_out = int(round(len(data) * 16000 / src_rate))
    if n_out <= 0:
        return np.empty(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)


class LoopbackSource:
    def __init__(self, sample_rate: int = 16000, block_ms: int = 30, device_name: str = ""):
        if sample_rate != 16000:
            raise ValueError("loopback 内部统一 16kHz 输出")
        self.block_ms = block_ms
        self.device_name = device_name
        self._stream = None
        self._pa = None

    def start(self, on_audio: AudioCallback) -> None:
        if sys.platform != "win32":
            raise LoopbackUnavailableError(LOOPBACK_GUIDE)
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as e:
            raise LoopbackUnavailableError(LOOPBACK_GUIDE) from e

        self._pa = pyaudio.PyAudio()
        try:
            loopback = self._find_loopback_device()
        except Exception as e:
            self._pa.terminate()
            self._pa = None
            raise LoopbackUnavailableError(LOOPBACK_GUIDE) from e

        native_rate = int(loopback["defaultSampleRate"])
        channels = int(loopback["maxInputChannels"])
        blocksize = int(native_rate * self.block_ms / 1000)
        pa_continue = pyaudio.paContinue  # 模块级常量，非 PyAudio 实例属性

        def _callback(in_data, frame_count, time_info, status):
            # 回调在 C 线程执行：任何异常都会导致采集流被中止，必须兜底
            try:
                data = np.frombuffer(in_data, dtype=np.float32)
                pcm = resample_to_16k_mono(data, native_rate, channels)
                on_audio(pcm)
            except Exception:
                log.exception("loopback 音频回调异常")
            return (None, pa_continue)

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=native_rate,
            frames_per_buffer=blocksize,
            input=True,
            input_device_index=loopback["index"],
            stream_callback=_callback,
        )
        self._stream.start_stream()
        log.info("loopback 采集已启动：%s（%d Hz, %d ch）", loopback["name"], native_rate, channels)

    def is_active(self) -> bool:
        """采集流是否仍在工作（回调线程崩溃/被系统中止后为 False）。"""
        try:
            return self._stream is not None and self._stream.is_active()
        except Exception:
            return False

    def _find_loopback_device(self) -> dict:
        if self.device_name:
            for d in self._pa.get_loopback_device_info_generator():
                if self.device_name.lower() in d["name"].lower():
                    return d
            raise LoopbackUnavailableError(
                f"未找到名为 '{self.device_name}' 的 loopback 设备\n{LOOPBACK_GUIDE}"
            )
        # 优先匹配默认输出设备对应的 loopback，失败则退回第一个可用设备
        try:
            wasapi = self._pa.get_host_api_info_by_type(self._pa.paWASAPI)
            default_out = self._pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
            prefix = default_out["name"].split("(")[0].strip()
            for d in self._pa.get_loopback_device_info_generator():
                if prefix and d["name"].startswith(prefix):
                    return d
        except Exception:
            log.debug("匹配默认输出设备失败，退回第一个 loopback 设备", exc_info=True)
        return next(self._pa.get_loopback_device_info_generator())

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
