"""音频采集抽象与环形缓冲。

所有采集源（麦克风 / WASAPI loopback / 文件模拟）实现同一接口：
start() 后通过回调持续推送 float32 mono 16kHz PCM 块，stop() 停止。
RingBuffer 供 ASR 流式线程读取，旧数据自动丢弃（防长会话内存增长，§9）。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol

import numpy as np

AudioCallback = Callable[[np.ndarray], None]
"""回调参数：float32 mono PCM 块，采样率 16kHz。"""


class AudioSource(Protocol):
    """音频采集源接口。"""

    def start(self, on_audio: AudioCallback) -> None: ...
    def stop(self) -> None: ...


class RingBuffer:
    """线程安全的 float32 环形缓冲，容量之外的旧数据自动覆盖。"""

    def __init__(self, capacity_samples: int):
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._write_pos = 0
        self._written = 0  # 累计写入样本数（单调递增，作为绝对时间基准）
        self._read_pos = 0  # 已消费到的绝对样本位置
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    @property
    def capacity(self) -> int:
        return self._capacity

    def write(self, data: np.ndarray) -> None:
        data = np.ascontiguousarray(data, dtype=np.float32).ravel()
        with self._cond:
            orig_n = len(data)
            n = orig_n
            if n > self._capacity:  # 块比缓冲还大：只留尾部
                data = data[-self._capacity :]
                n = self._capacity
            # 保留部分的首个样本的绝对位置
            start_abs = self._written + (orig_n - n)
            pos = start_abs % self._capacity
            first = min(n, self._capacity - pos)
            self._buf[pos : pos + first] = data[:first]
            if first < n:
                self._buf[: n - first] = data[first:]
            self._written += orig_n
            self._write_pos = self._written % self._capacity
            # 读指针落后超过容量 = 数据被覆盖，强制对齐到最旧可用处
            if self._written - self._read_pos > self._capacity:
                self._read_pos = self._written - self._capacity
            self._cond.notify_all()

    def read(self, n: int, timeout: float | None = None) -> tuple[np.ndarray, int]:
        """读取 n 个样本，返回 (数据, 起始绝对样本位置)。

        数据不足时阻塞等待，timeout 后返回当前已有的部分（可能为空）。
        """
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._written - self._read_pos >= n, timeout=timeout
            )
            avail = self._written - self._read_pos
            n = n if ok else min(n, avail)
            if n <= 0:
                return np.empty(0, dtype=np.float32), self._read_pos
            start = self._read_pos
            pos = start % self._capacity
            first = min(n, self._capacity - pos)
            out = np.empty(n, dtype=np.float32)
            out[:first] = self._buf[pos : pos + first]
            if first < n:
                out[first:] = self._buf[: n - first]
            self._read_pos += n
            return out, start

    def peek(self, n: int) -> tuple[np.ndarray, int]:
        """查看但不消费最近最多 n 个样本，返回 (数据, 起始绝对样本位置)。"""
        with self._lock:
            avail = self._written - self._read_pos
            n = min(n, avail)
            if n <= 0:
                return np.empty(0, dtype=np.float32), self._read_pos
            start = self._written - n
            out = np.empty(n, dtype=np.float32)
            for i in range(n):
                out[i] = self._buf[(start + i) % self._capacity]
            return out, start

    def tell(self) -> int:
        """当前已消费的绝对样本位置。"""
        with self._lock:
            return self._read_pos

    def total_written(self) -> int:
        with self._lock:
            return self._written
