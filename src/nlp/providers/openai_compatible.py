"""OpenAI 兼容云端 Provider（也用于用户自托管的 vLLM 端点：type=vllm 不托管进程）。"""

from __future__ import annotations

import logging

from ...config import LLMConfig, ProviderConfig
from ..llm_client import ChatResult, LLMClient
from .base import ProviderBase, ProviderUnavailableError

log = logging.getLogger(__name__)


class OpenAICompatibleProvider(ProviderBase):
    """包装 LLMClient；ensure_available 为一次轻量 ping（短超时、不重试）。"""

    def __init__(self, cfg: ProviderConfig):
        super().__init__(cfg)
        if not cfg.is_complete:
            raise ProviderUnavailableError(
                f"Provider '{cfg.id}' 配置不完整：需要 base_url / model（api_key 视服务商要求）"
            )
        self._client = LLMClient(cfg.to_llm_config(), params=cfg.params)

    def chat(self, prompt: str, *, system: str | None = None, json_mode: bool = False) -> ChatResult:
        return self._client.chat(prompt, system=system, json_mode=json_mode)

    @property
    def online(self) -> bool:
        return self._client.online

    def ensure_available(self) -> None:
        ping_cfg = LLMConfig(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            model=self.cfg.model,
            timeout_sec=min(self.cfg.timeout_sec, 10.0),
            max_retries=0,
        )
        try:
            LLMClient(ping_cfg).chat("ping")
        except Exception as e:  # noqa: BLE001 —— 任何失败都归为不可用，原因带给用户
            raise ProviderUnavailableError(
                f"Provider '{self.cfg.id}' 健康检查失败：{type(e).__name__}: {e}"
            ) from e
