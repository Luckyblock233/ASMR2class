"""麦克风采集（sounddevice，跨平台）。

sounddevice 依赖 PortAudio 动态库，因此延迟到 start() 时才 import，
使无音频设备的环境（CI / WSL）仍可导入本模块与运行其余测试。
"""

from __future__ import annotations

import logging

import numpy as np

from .base import AudioCallback

log = logging.getLogger(__name__)


def list_input_devices() -> list[dict]:
    """列出可用麦克风（输入）设备。"""
    import sounddevice as sd  # 延迟导入

    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"],
         "sample_rate": d["default_samplerate"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


class MicSource:
    def __init__(self, sample_rate: int = 16000, block_ms: int = 30):
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self._stream = None

    def start(self, on_audio: AudioCallback) -> None:
        import sounddevice as sd  # 延迟导入：无 PortAudio 环境可安全 import 本模块

        blocksize = int(self.sample_rate * self.block_ms / 1000)

        def _callback(indata, frames, time_info, status):
            if status:
                log.warning("麦克风采集状态：%s", status)
            on_audio(indata[:, 0].astype(np.float32))

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=_callback,
        )
        self._stream.start()
        log.info("麦克风采集已启动（%d Hz, %d ms/块）", self.sample_rate, self.block_ms)

    def is_active(self) -> bool:
        """采集流是否仍在工作。"""
        try:
            return self._stream is not None and self._stream.active
        except Exception:
            return False

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
