"""语义边界分块器（FR-C2）。

成块触发（三者任一）：
1. 新句子与上一句的静音间隙 > silence_gap_sec（语义边界）
2. 块内句数达到 max_sentences
3. 空闲超时：最后一句结束后 silence_gap_sec 内没有新句（等不到"下一句"的块尾）

注：场景切换信号暂无来源（需视频/章节信息），预留 trigger 参数待 M3+ 接入。
实时模式与离线批处理共用本类（FR-C2.5）；离线模式把 max_sentences 放宽到 20 即可。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from ..asr.streamer import Sentence
from ..config import BlockConfig

log = logging.getLogger(__name__)

_CHECK_INTERVAL = 0.1  # 空闲检测周期（秒）


class Blocker:
    def __init__(
        self,
        cfg: BlockConfig,
        on_block: Callable[[list[Sentence]], None],
        clock: Callable[[], float] | None = None,
        autoflush: bool = True,
    ):
        self.cfg = cfg
        self.on_block = on_block
        self._pending: list[Sentence] = []
        self._lock = threading.Lock()
        self._closed = False
        import time

        self._clock = clock or time.monotonic
        self._last_feed_wall: float | None = None
        self._timer: threading.Thread | None = None
        if autoflush:
            self._timer = threading.Thread(
                target=self._idle_watch, name="blocker-idle", daemon=True
            )
            self._timer.start()

    def feed(self, s: Sentence) -> None:
        with self._lock:
            if self._closed:
                return
            # 触发 1：静音间隙 = 语义边界
            if self._pending and (s.start - self._pending[-1].end) > self.cfg.silence_gap_sec:
                self._flush_locked()
            self._pending.append(s)
            self._last_feed_wall = self._clock()
            # 触发 2：句数上限
            if len(self._pending) >= self.cfg.max_sentences:
                self._flush_locked()

    def flush(self) -> None:
        """冲刷残余块（会话结束时调用）。"""
        with self._lock:
            self._flush_locked()
            self._closed = True

    def close(self) -> None:
        self.flush()

    # ---- 内部 ----

    def _flush_locked(self) -> None:
        if not self._pending:
            return
        block = self._pending
        self._pending = []
        log.debug("成块：%d 句（%.1fs → %.1fs）", len(block), block[0].start, block[-1].end)
        self.on_block(block)

    def _idle_watch(self) -> None:
        """触发 3：空闲超时成块。墙上时钟近似会话时间（实时模式下一致）。"""
        idle = self.cfg.silence_gap_sec + 0.5
        while True:
            with self._lock:
                if self._closed:
                    return
                due = (
                    self._pending
                    and self._last_feed_wall is not None
                    and (self._clock() - self._last_feed_wall) >= idle
                )
            if due:
                with self._lock:
                    self._flush_locked()
            threading.Event().wait(_CHECK_INTERVAL)
