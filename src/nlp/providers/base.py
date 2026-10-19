"""Provider 抽象基类：统一 OpenAI 兼容 chat/completions 接口。

llm_client 只认该接口，不感知本地/云端差异（模型可插拔架构 §1.1 统一接口）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...config import ProviderConfig
from ..llm_client import ChatResult


class ProviderUnavailableError(RuntimeError):
    """Provider 不可用（未配置完整/本地服务拉不起/健康检查失败），附带中文原因。"""


class ProviderBase(ABC):
    """模型提供方统一接口。实现类不得缓存导致无法换绑的状态。"""

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg

    @abstractmethod
    def chat(self, prompt: str, *, system: str | None = None, json_mode: bool = False) -> ChatResult:
        """发送一次对话请求（语义与 LLMClient.chat 一致）。"""

    @property
    @abstractmethod
    def online(self) -> bool:
        """当前是否在线（请求级状态，由底层客户端维护）。"""

    @abstractmethod
    def ensure_available(self) -> None:
        """启动时健康检查；不可用抛 ProviderUnavailableError（附原因）。"""

    def close(self) -> None:
        """释放资源（本地托管 provider 回收子进程）。云端 provider 默认无操作。"""
