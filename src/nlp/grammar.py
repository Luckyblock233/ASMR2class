"""异步语法分析：语法槽 Provider → 外部模板 → 按稳定句段 ID 回写。"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable

from ..profiles.loader import LanguageProfile
from ..subtitle.document import (
    ANALYSIS_FAILED,
    ANALYSIS_PENDING,
    ANALYSIS_READY,
    Segment,
    SessionDocument,
)
from .prompt_engine import PromptEngine
from .translate import _extract_json

log = logging.getLogger(__name__)


class GrammarContractError(ValueError):
    """语法 Provider 返回值不符合通用 JSON 契约。"""


def validate_analysis_payload(value: Any) -> dict[str, Any]:
    """只校验通用容器；语言专属字段及内部结构一律透传。"""
    if not isinstance(value, dict):
        raise GrammarContractError("语法响应必须是 JSON 对象")
    structure = value.get("structure")
    if structure is not None and not isinstance(structure, (dict, list)):
        raise GrammarContractError("structure 必须是对象、数组或 null")
    grammar_points = value.get("grammar_points")
    vocabulary = value.get("vocabulary")
    extra = value.get("extra")
    grammar_points = [] if grammar_points is None else grammar_points
    vocabulary = [] if vocabulary is None else vocabulary
    extra = {} if extra is None else extra
    if not isinstance(grammar_points, list):
        raise GrammarContractError("grammar_points 必须是数组")
    if not isinstance(vocabulary, list):
        raise GrammarContractError("vocabulary 必须是数组")
    if not isinstance(extra, dict):
        raise GrammarContractError("extra 必须是对象")
    return {
        "structure": structure,
        "grammar_points": grammar_points,
        "vocabulary": vocabulary,
        "extra": extra,
    }


class GrammarAnalyzer:
    """有界异步分析器；回调只携带句段快照，不暴露可变文档内部状态。"""

    def __init__(
        self,
        provider_getter,
        prompts: PromptEngine,
        profile: LanguageProfile,
        document: SessionDocument,
        *,
        on_update: Callable[[Segment], None] | None = None,
        max_workers: int = 2,
        max_pending: int = 64,
    ):
        self._get_provider = provider_getter if callable(provider_getter) else lambda: provider_getter
        self.prompts = prompts
        self.profile = profile
        self.document = document
        self.on_update = on_update
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="grammar")
        self._capacity = threading.BoundedSemaphore(max_pending)
        self._pending: set[int] = set()
        self._futures: set[Future] = set()
        self._lock = threading.RLock()
        self._closed = False

    @property
    def available(self) -> bool:
        return self._get_provider() is not None

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def submit(self, segment_id: int) -> bool:
        """提交自动或手动分析；重复、已就绪、关闭或队列已满时返回 False。"""
        with self._lock:
            if self._closed or segment_id in self._pending:
                return False
            segment = self.document.get(segment_id)
            if segment is None or (segment.analysis_status == ANALYSIS_READY and segment.has_analysis):
                return False
            if not self._capacity.acquire(blocking=False):
                snapshot = self.document.set_analysis_state(
                    segment_id, ANALYSIS_FAILED, "语法分析队列已满，请稍后重试"
                )
                self._notify(snapshot)
                return False
            self._pending.add(segment_id)
            snapshot = self.document.set_analysis_state(segment_id, ANALYSIS_PENDING)
            self._notify(snapshot)
            try:
                future = self._executor.submit(self._run, segment_id)
                self._futures.add(future)
            except Exception:
                self._pending.discard(segment_id)
                self._capacity.release()
                snapshot = self.document.set_analysis_state(
                    segment_id, ANALYSIS_FAILED, "无法提交语法分析任务"
                )
                self._notify(snapshot)
                raise
            future.add_done_callback(lambda f, sid=segment_id: self._complete(sid, f))
        return True

    def _run(self, segment_id: int) -> None:
        segment = self.document.get(segment_id)
        if segment is None:
            return
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                provider = self._get_provider()
                if provider is None:
                    raise GrammarContractError("语法槽未绑定或 Provider 不可用")
                template = self.prompts.resolve(self.profile.prompt_path("grammar"), "grammar")
                prompt = self.prompts.render(
                    template,
                    source_lang=self.profile.source_lang,
                    target_lang=self.profile.target_lang,
                    sentence=segment.source,
                    source=segment.source,
                    translation=segment.translation or "",
                    level_hint=str(self.profile.raw.get("level_hint") or ""),
                )
                response = provider.chat(prompt, json_mode=True)
                payload = validate_analysis_payload(_extract_json(response.content))
                cfg = getattr(provider, "cfg", None)
                self.document.update_grammar_metadata(
                    template_version=self.prompts.template_version(template),
                    provider_id=str(getattr(cfg, "id", "") or ""),
                    model=str(getattr(cfg, "model", "") or ""),
                )
                snapshot = self.document.apply_analysis(segment_id, payload)
                self._notify(snapshot)
                return
            except Exception as e:  # 单句失败不得影响其他句或主字幕流
                last_error = e
                log.info("语法分析失败（句 %s，第 %s 次）：%s", segment_id, attempt + 1, e)
        snapshot = self.document.set_analysis_state(
            segment_id,
            ANALYSIS_FAILED,
            f"{type(last_error).__name__}: {last_error}" if last_error else "语法分析失败",
        )
        self._notify(snapshot)

    def _complete(self, segment_id: int, future: Future) -> None:
        if future.cancelled():
            try:
                snapshot = self.document.set_analysis_state(
                    segment_id, ANALYSIS_FAILED, "语法分析已取消"
                )
                self._notify(snapshot)
            except KeyError:
                pass
        with self._lock:
            self._pending.discard(segment_id)
            self._futures.discard(future)
            self._capacity.release()

    def drain(self, timeout: float = 10.0) -> bool:
        """停止新任务并限期等待；超时取消排队任务，运行中调用安全结束后才返回。"""
        with self._lock:
            self._closed = True
            futures = list(self._futures)
        deadline = time.monotonic() + max(0.0, timeout)
        timed_out = False
        for future in futures:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                future.result(timeout=remaining)
            except FutureTimeoutError:
                timed_out = True
                future.cancel()
            except Exception as e:  # _run 已兜底，这里仅防回调外的意外异常
                log.debug("语法任务结束时异常：%s", e)
        with self._lock:
            remaining_futures = list(self._futures)
        for future in remaining_futures:
            future.cancel()
        # Python 无法强制终止正在执行的 Provider 调用；必须等它退出，调用方才能
        # 安全保存并关闭 Registry，避免迟到回写和请求使用已关闭的本地进程。
        self._executor.shutdown(wait=True, cancel_futures=True)
        return not timed_out

    def _notify(self, segment: Segment) -> None:
        if self.on_update is None:
            return
        try:
            self.on_update(segment)
        except Exception as e:  # UI 回调不得反向破坏分析 worker
            log.debug("语法状态回调异常：%s", e)
