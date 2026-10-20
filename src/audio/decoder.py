"""媒体文件流式音频解码（M3 阶段 5：离线批处理专用）。

PyAV 流式解码为 16kHz 单声道 float32 块；逐块产出、不整文件读入。
与 FileSource（实时模拟采集源）职责分离：本模块面向可靠批处理，
不使用 RingBuffer、不做墙上时钟模拟。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
DEFAULT_CHUNK_MS = 100


class AudioDecodeError(RuntimeError):
    """解码失败（无音轨/文件损坏/格式不支持），附中文原因。"""


def decode_audio(
    path: str | Path,
    *,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Iterator[np.ndarray]:
    """流式产出音频块（float32 mono 一维数组）。

    - 无音轨/损坏文件/格式不支持：抛 AudioDecodeError
    - 中文/空格路径：PyAV 接受 str 路径，天然支持
    - 调用方以 for 循环拉取即自然背压（不产生中间整文件缓冲）
    """
    import av  # 延迟导入：批处理模式才需要 PyAV

    p = Path(path)
    if not p.is_file():
        raise AudioDecodeError(f"文件不存在：{p}")
    try:
        container = av.open(str(p))
    except Exception as e:  # noqa: BLE001 —— av 异常类型繁多，统一包装
        raise AudioDecodeError(f"无法打开媒体文件：{p}（{type(e).__name__}: {e}）") from e

    chunk_samples = int(sample_rate * chunk_ms / 1000)
    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise AudioDecodeError(f"文件无音轨：{p}")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        carry = np.empty(0, dtype=np.float32)
        try:
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    pcm = np.asarray(resampled.to_ndarray(), dtype=np.float32).ravel()
                    carry = np.concatenate([carry, pcm])
                    while len(carry) >= chunk_samples:
                        yield carry[:chunk_samples]
                        carry = carry[chunk_samples:]
            # flush 重采样器残留
            for resampled in resampler.resample(None):
                pcm = np.asarray(resampled.to_ndarray(), dtype=np.float32).ravel()
                carry = np.concatenate([carry, pcm])
            if len(carry) > 0:
                yield carry
        except AudioDecodeError:
            raise
        except Exception as e:  # noqa: BLE001
            raise AudioDecodeError(f"解码中断：{p}（{type(e).__name__}: {e}）") from e
    finally:
        container.close()


def media_duration_sec(path: str | Path) -> float | None:
    """媒体总时长（秒，用于进度显示）；不可知时返回 None。"""
    import av

    try:
        container = av.open(str(path))
        try:
            if container.duration:
                return container.duration / 1_000_000
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is not None and stream.duration is not None and stream.time_base:
                return float(stream.duration * stream.time_base)
            return None
        finally:
            container.close()
    except Exception:  # noqa: BLE001
        return None
