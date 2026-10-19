"""OpenAI 兼容 LLM 客户端（FR-1.5/1.6）。

- base_url / api_key / model 三项接入任意 OpenAI 兼容服务，无默认服务商。
- 指数退避重试；连续失败后标记离线，首次成功请求自动恢复在线。
- api_key 绝不写入日志。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from ..config import LLMConfig

log = logging.getLogger(__name__)


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1        # 实际发起的网络请求次数（含重试）
    latency_sec: float = 0.0
    json_mode: bool = False  # 本次请求是否使用了 response_format=json_object

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """线程安全的 OpenAI 兼容客户端，内置退避重试与在线状态跟踪。

    params：Provider 声明的请求参数透传（temperature/max_tokens 等标准参数覆盖默认值，
    其余非标准键如 enable_thinking 走 extra_body）。
    """

    def __init__(
        self,
        config: LLMConfig,
        client: OpenAI | None = None,
        params: dict | None = None,
    ):
        if not config.is_complete:
            raise ValueError("LLM 配置不完整：需要 base_url / api_key / model 三项")
        self.config = config
        self.params = dict(params or {})
        self._client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_sec,
            max_retries=0,  # 重试由本类统一管理（指数退避）
        )
        self._online = True
        self._json_supported = True  # 服务商是否支持 response_format=json_object（400 时自动退化）
        self._lock = threading.Lock()

    @property
    def online(self) -> bool:
        return self._online

    def _set_online(self, value: bool) -> None:
        with self._lock:
            if self._online != value:
                self._online = value
                log.warning("LLM 连接状态切换为：%s", "ONLINE" if value else "OFFLINE")

    def chat(self, prompt: str, *, system: str | None = None, json_mode: bool = False) -> ChatResult:
        """发送一次对话请求，带指数退避重试。全部失败时抛出最后一次异常。

        json_mode=True 时优先使用 response_format: json_object；
        服务商不支持（400）时本次及以后自动退化为普通请求（模板内需自带 JSON 格式说明）。
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        use_json = json_mode and self._json_supported
        last_exc: Exception | None = None
        attempts = 0
        t0 = time.monotonic()
        # Provider 声明的请求参数：标准键覆盖默认值，非标准键走 extra_body
        STANDARD_KEYS = ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "seed")
        std_params = {"temperature": 0.2}
        std_params.update({k: v for k, v in self.params.items() if k in STANDARD_KEYS})
        extra = {k: v for k, v in self.params.items() if k not in STANDARD_KEYS}
        for attempt in range(self.config.max_retries + 1):
            try:
                kwargs: dict = dict(std_params)
                if extra:
                    kwargs["extra_body"] = extra
                if use_json:
                    kwargs["response_format"] = {"type": "json_object"}
                attempts += 1
                resp = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    **kwargs,
                )
                self._set_online(True)
                usage = getattr(resp, "usage", None)
                return ChatResult(
                    content=(resp.choices[0].message.content or "").strip(),
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    attempts=attempts,
                    latency_sec=time.monotonic() - t0,
                    json_mode=use_json,
                )
            except (APIConnectionError, APITimeoutError) as e:
                last_exc = e
                self._set_online(False)
            except APIStatusError as e:
                # 服务商不支持 response_format：退化后原地重试，不计入退避次数
                if use_json and e.status_code == 400:
                    log.info("服务商不支持 response_format=json_object，退化为普通请求")
                    self._json_supported = False
                    use_json = False
                    continue
                last_exc = e
                # 4xx（鉴权/参数错误）不重试，直接抛出
                if 400 <= e.status_code < 500:
                    raise
                self._set_online(False)
            if attempt < self.config.max_retries:
                delay = 2**attempt
                log.info("LLM 请求失败（第 %d 次），%ds 后重试", attempt + 1, delay)
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc
