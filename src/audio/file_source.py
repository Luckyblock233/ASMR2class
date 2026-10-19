"""文件实时模拟源。

按真实时间节奏把文件音频推入管线，用于：
- 无音频设备环境（WSL/CI）的端到端验证
- 未来 FR-2.3 离线批处理的输入源雏形
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import AudioCallback
from .loopback import resample_to_16k_mono

log = logging.getLogger(__name__)


class FileSource:
    def __init__(self, input_path: str | Path, realtime: bool = True, speed: float = 1.0):
        self.input_path = Path(input_path)
        self.realtime = realtime
        self.speed = speed
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.finished = threading.Event()

    def start(self, on_audio: AudioCallback) -> None:
        data, rate = sf.read(self.input_path, dtype="float32", always_2d=True)
        pcm = resample_to_16k_mono(data, rate, data.shape[1])

        def _run():
            block = 480  # 30ms @16k
            for i in range(0, len(pcm), block):
                if self._stop_event.is_set():
                    break
                on_audio(pcm[i : i + block])
                if self.realtime:
                    time.sleep(block / 16000 / self.speed)
            self.finished.set()

        self._thread = threading.Thread(target=_run, name="file-source", daemon=True)
        self._thread.start()
        log.info("文件模拟源已启动：%s（%.1fs）", self.input_path, len(pcm) / 16000)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
