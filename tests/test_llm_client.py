from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from src.config import LLMConfig
from src.nlp.llm_client import LLMClient


def make_cfg(max_retries=2):
    return LLMConfig(
        base_url="https://example.test/v1",
        api_key="sk-test",
        model="test-model",
        max_retries=max_retries,
    )


def make_response(content="译文", prompt_tokens=10, completion_tokens=5):
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    msg = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


class FakeCompletions:
    """模拟 openai chat.completions.create，按队列抛出异常或返回结果。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeOpenAI:
    def __init__(self, script):
        self.completions = FakeCompletions(script)
        self.chat = SimpleNamespace(completions=self.completions)


def conn_err():
    return APIConnectionError(request=None)


def test_success_returns_content_and_tokens(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = FakeOpenAI([make_response("你好")])
    client = LLMClient(make_cfg(), client=fake)
    r = client.chat("prompt")
    assert r.content == "你好"
    assert r.total_tokens == 15
    assert client.online


def test_retry_then_success(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = FakeOpenAI([conn_err(), make_response("ok")])
    client = LLMClient(make_cfg(max_retries=2), client=fake)
    r = client.chat("prompt")
    assert r.content == "ok"
    assert fake.completions.calls == 2
    assert client.online  # 恢复后自动回到 ONLINE


def test_all_retries_fail_marks_offline(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = FakeOpenAI([conn_err(), conn_err(), conn_err()])
    client = LLMClient(make_cfg(max_retries=2), client=fake)
    with pytest.raises(APIConnectionError):
        client.chat("prompt")
    assert not client.online
    assert fake.completions.calls == 3  # 1 + 2 次重试


def test_4xx_no_retry(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    req = httpx.Request("POST", "https://example.test/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    err = APIStatusError("bad request", response=resp, body=None)
    fake = FakeOpenAI([err])
    client = LLMClient(make_cfg(max_retries=3), client=fake)
    with pytest.raises(APIStatusError):
        client.chat("prompt")
    assert fake.completions.calls == 1  # 4xx 不重试


def test_incomplete_config_rejected():
    with pytest.raises(ValueError, match="不完整"):
        LLMClient(LLMConfig(base_url="", api_key="", model=""))
