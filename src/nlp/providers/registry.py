"""Provider 注册表：加载配置、按 id 寻址、健康检查、本地托管 provider 生命周期。

热切换就绪：槽位持有的只是 provider id，Provider 对象统一由 Registry 按 id
寻址；改绑 = 更新槽位→id 映射，不触碰进行中的请求。
"""

from __future__ import annotations

import logging
import threading

from ...config import GlobalConfig, ProviderConfig
from .base import ProviderBase, ProviderUnavailableError
from .llama_cpp_local import LlamaCppLocalProvider
from .openai_compatible import OpenAICompatibleProvider

log = logging.getLogger(__name__)


def create_provider(cfg: ProviderConfig, popen=None) -> ProviderBase:
    """按 type 构造 Provider。popen 仅供测试注入。"""
    if cfg.type == "llama_cpp_server" and cfg.local and cfg.local.managed:
        return LlamaCppLocalProvider(cfg, popen=popen)
    if cfg.type in ("openai_compatible", "vllm", "llama_cpp_server"):
        return OpenAICompatibleProvider(cfg)
    raise ProviderUnavailableError(f"Provider '{cfg.id}' 类型未知：{cfg.type}")


class ProviderRegistry:
    """线程安全的 Provider 注册表（id → 懒构造的 Provider 实例）。"""

    def __init__(self, configs: list[ProviderConfig], popen=None):
        self._configs = {c.id: c for c in configs}
        self._instances: dict[str, ProviderBase] = {}
        self._popen = popen
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: GlobalConfig, popen=None) -> "ProviderRegistry":
        return cls(cfg.providers, popen=popen)

    def ids(self) -> list[str]:
        return list(self._configs)

    def config(self, provider_id: str) -> ProviderConfig | None:
        return self._configs.get(provider_id)

    def get(self, provider_id: str) -> ProviderBase | None:
        """按 id 寻址（懒构造）；未知 id 或构造失败返回 None。"""
        if not provider_id:
            return None
        with self._lock:
            if provider_id in self._instances:
                return self._instances[provider_id]
            cfg = self._configs.get(provider_id)
            if cfg is None:
                return None
            try:
                inst = create_provider(cfg, popen=self._popen)
            except ProviderUnavailableError as e:
                log.warning("Provider 构造失败：%s", e)
                return None
            self._instances[provider_id] = inst
            return inst

    def check_health(self, provider_id: str) -> str | None:
        """健康检查；返回 None = 可用，否则中文原因。未知 id 也返回原因。"""
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
        """关闭不再被任何槽位引用的本地托管 provider（释放显存）。"""
        with self._lock:
            for pid, inst in list(self._instances.items()):
                cfg = self._configs.get(pid)
                is_managed_local = (
                    cfg is not None and cfg.local is not None and cfg.local.managed
                )
                if is_managed_local and pid not in in_use_ids:
                    log.info("Provider '%s' 已不再被任何槽位绑定，回收资源", pid)
                    inst.close()
                    del self._instances[pid]

    def close_all(self) -> None:
        with self._lock:
            for inst in self._instances.values():
                try:
                    inst.close()
                except Exception:  # noqa: BLE001
                    log.debug("关闭 Provider 失败", exc_info=True)
            self._instances.clear()
