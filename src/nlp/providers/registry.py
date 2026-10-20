"""Provider 注册表：加载配置、按 id 寻址、健康检查、本地托管 provider 生命周期。

安全热换绑（M3 后续阶段 2）：
- 每个 Provider **实例**是一条带 generation/state/active_calls 的记录；
  同 id 配置编辑或换绑产生新代，正在退休的旧代不会被复活。
- 消费方统一经 ``acquire()`` 租约使用 Provider；租约释放必须在 finally 中执行。
- 换绑发布后旧代进入 retiring：只有当不再被任何槽位引用且 active_calls == 0
  才关闭；控制面（rebind/release_unused/close_all）在锁外同步关闭，
  请求 worker 释放最后一个租约时由回收线程异步关闭，避免阻塞翻译/语法 worker。
- close()/start()/健康检查等待一律不持有 Registry 全局锁。
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from ...config import GlobalConfig, ProviderConfig
from .base import ProviderBase, ProviderUnavailableError
from .llama_cpp_local import LlamaCppLocalProvider
from .openai_compatible import OpenAICompatibleProvider

log = logging.getLogger(__name__)

STATE_ACTIVE = "active"
STATE_RETIRING = "retiring"
STATE_CLOSED = "closed"


def create_provider(cfg: ProviderConfig, popen=None) -> ProviderBase:
    """按 type 构造 Provider。popen 仅供测试注入。"""
    if cfg.type == "llama_cpp_server" and cfg.local and cfg.local.managed:
        return LlamaCppLocalProvider(cfg, popen=popen)
    if cfg.type in ("openai_compatible", "vllm", "llama_cpp_server"):
        return OpenAICompatibleProvider(cfg)
    raise ProviderUnavailableError(f"Provider '{cfg.id}' 类型未知：{cfg.type}")


class _Record:
    """一个 Provider 实例代际：状态机 active → retiring → closed。"""

    __slots__ = ("provider_id", "instance", "generation", "state", "active_calls")

    def __init__(self, provider_id: str, instance: ProviderBase, generation: int):
        self.provider_id = provider_id
        self.instance = instance
        self.generation = generation
        self.state = STATE_ACTIVE
        self.active_calls = 0


def _close_safely(instance: ProviderBase) -> None:
    try:
        instance.close()
    except Exception:  # noqa: BLE001
        log.debug("关闭 Provider 失败", exc_info=True)


class ProviderRegistry:
    """线程安全的 Provider 注册表（id → 当前代 Provider 实例记录）。"""

    def __init__(self, configs: list[ProviderConfig], popen=None):
        self._configs = {c.id: c for c in configs}
        self._records: dict[str, _Record] = {}
        self._retiring: list[_Record] = []
        self._generation = 0
        self._popen = popen
        self._lock = threading.Condition(threading.Lock())
        self._closed = False

    @classmethod
    def from_config(cls, cfg: GlobalConfig, popen=None) -> "ProviderRegistry":
        return cls(cfg.providers, popen=popen)

    def ids(self) -> list[str]:
        return list(self._configs)

    def config(self, provider_id: str) -> ProviderConfig | None:
        return self._configs.get(provider_id)

    # ---- 实例代际 ----

    def _get_or_create_locked(self, provider_id: str) -> _Record | None:
        rec = self._records.get(provider_id)
        if rec is not None and rec.state == STATE_ACTIVE:
            return rec
        cfg = self._configs.get(provider_id)
        if cfg is None:
            return None
        try:
            inst = create_provider(cfg, popen=self._popen)
        except ProviderUnavailableError as e:
            log.warning("Provider 构造失败：%s", e)
            return None
        self._generation += 1
        rec = _Record(provider_id, inst, self._generation)
        self._records[provider_id] = rec
        return rec

    def get(self, provider_id: str) -> ProviderBase | None:
        """按 id 寻址（懒构造）；未知 id 或构造失败返回 None。

        供状态显示/健康检查等只读路径；实际请求必须走 acquire() 租约。
        """
        if not provider_id:
            return None
        with self._lock:
            rec = self._get_or_create_locked(provider_id)
            return rec.instance if rec is not None else None

    @contextmanager
    def acquire(self, provider_id: str) -> Iterator[ProviderBase | None]:
        """租约：持有期间实例不会被回收；未绑定/不可用 yield None。"""
        rec = None
        if provider_id:
            with self._lock:
                if not self._closed:
                    rec = self._get_or_create_locked(provider_id)
                    if rec is not None:
                        rec.active_calls += 1
        if rec is None:
            yield None
            return
        try:
            yield rec.instance
        finally:
            self._release(rec)

    def _release(self, rec: _Record) -> None:
        """释放租约；worker 释放退休代的最后一个租约时异步关闭，不阻塞请求线程。"""
        retire_async = None
        with self._lock:
            rec.active_calls -= 1
            self._lock.notify_all()
            if rec.state == STATE_RETIRING and rec.active_calls == 0 and rec in self._retiring:
                self._retiring.remove(rec)
                rec.state = STATE_CLOSED
                retire_async = rec
        if retire_async is not None:
            threading.Thread(
                target=_close_safely,
                args=(retire_async.instance,),
                name=f"provider-reaper-{retire_async.provider_id}",
                daemon=True,
            ).start()

    def _retire_locked(self, provider_id: str) -> _Record | None:
        """当前代转退休；无活动租约时返回待关闭记录（调用方在锁外关闭）。"""
        rec = self._records.pop(provider_id, None)
        if rec is None:
            return None
        rec.state = STATE_RETIRING
        if rec.active_calls > 0:
            self._retiring.append(rec)
            log.info("Provider '%s' 第 %d 代进入退休（在途请求 %d 个）",
                     provider_id, rec.generation, rec.active_calls)
            return None
        rec.state = STATE_CLOSED
        return rec

    def retire(self, provider_id: str) -> None:
        """同 id 配置编辑后作废旧代：下次 acquire 创建新代，旧代排空后关闭。"""
        with self._lock:
            rec = self._retire_locked(provider_id)
        if rec is not None:
            log.info("Provider '%s' 配置已更新，回收第 %d 代", provider_id, rec.generation)
            _close_safely(rec.instance)

    def update_config(self, cfg: ProviderConfig) -> None:
        """新增/替换 provider 配置并作废旧代（同 id 编辑创建新代）。"""
        with self._lock:
            self._configs[cfg.id] = cfg
        self.retire(cfg.id)

    def remove_config(self, provider_id: str) -> None:
        """删除 provider 配置并退役当前代（删除前调用方须先解除槽位绑定）。"""
        with self._lock:
            self._configs.pop(provider_id, None)
        self.retire(provider_id)

    def check_health(self, provider_id: str) -> str | None:
        """健康检查；返回 None = 可用，否则中文原因。未知 id 也返回原因。

        ensure_available（含本地进程启动与等待）在锁外执行，不阻塞其他线程。
        """
        cfg = self._configs.get(provider_id)
        if cfg is None:
            return f"Provider '{provider_id}' 不存在"
        inst = self.get(provider_id)
        if inst is None:
            return f"Provider '{provider_id}' 构造失败（详见日志）"
        try:
            inst.ensure_available()
        except ProviderUnavailableError as e:
            return str(e)
        except Exception as e:  # noqa: BLE001
            return f"Provider '{provider_id}' 健康检查异常：{type(e).__name__}: {e}"
        return None

    def release_unused(self, in_use_ids: set[str]) -> None:
        """换绑后回收：不再被任何槽位引用的本地托管 provider 转退休/关闭。"""
        to_close: list[_Record] = []
        with self._lock:
            for pid in list(self._records):
                if pid in in_use_ids:
                    continue
                cfg = self._configs.get(pid)
                is_managed_local = (
                    cfg is not None and cfg.local is not None and cfg.local.managed
                )
                if not is_managed_local:
                    continue
                rec = self._retire_locked(pid)
                if rec is not None:
                    to_close.append(rec)
        for rec in to_close:
            log.info("Provider '%s' 已不再被任何槽位绑定，回收资源", rec.provider_id)
            _close_safely(rec.instance)

    def close_all(self, timeout: float = 30.0) -> None:
        """退出收尾：等待在途租约排空（不超过 timeout），然后关闭全部实例。

        调用方必须先停止生产者（采集/ASR/Blocker/Grammar），再调用本方法；
        超时后仍强制关闭并告警（本地进程不能随应用退出而泄漏）。
        """
        deadline = time.monotonic() + timeout
        with self._lock:
            self._closed = True
            while True:
                active = sum(r.active_calls for r in self._records.values())
                active += sum(r.active_calls for r in self._retiring)
                if active <= 0:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning("关闭 Registry 时仍有 %d 个在途请求，强制回收", active)
                    break
                self._lock.wait(remaining)
            to_close = [r.instance for r in self._records.values()]
            to_close += [r.instance for r in self._retiring]
            self._records.clear()
            self._retiring.clear()
        for inst in to_close:
            _close_safely(inst)

    # ---- 测试/诊断 ----

    def active_calls(self, provider_id: str) -> int:
        """该 provider 当前代 + 退休代的在途租约总数。"""
        with self._lock:
            total = 0
            rec = self._records.get(provider_id)
            if rec is not None:
                total += rec.active_calls
            total += sum(
                r.active_calls for r in self._retiring if r.provider_id == provider_id
            )
            return total

    def snapshot(self) -> dict[str, Any]:
        """C 栏状态显示用的只读快照（不做网络健康请求）。

        status 统一语义：running（运行中）/ degraded（退休排空中）/
        error（最近请求失败，last_error 附原因）。未实例化的配置不在其中，
        由调用方按"○ 停止"显示。
        """
        with self._lock:
            out = {}
            for pid, rec in self._records.items():
                online = bool(getattr(rec.instance, "online", False))
                last_error = str(getattr(rec.instance, "last_error", "") or "")
                if rec.state == STATE_RETIRING:
                    status = "degraded"
                elif not online:
                    status = "error"
                else:
                    status = "running"
                out[pid] = {
                    "generation": rec.generation,
                    "state": rec.state,
                    "active_calls": rec.active_calls,
                    "online": online,
                    "status": status,
                    "last_error": last_error,
                }
            return out
