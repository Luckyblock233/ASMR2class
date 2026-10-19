"""debug 模式输出测试。"""

import json
from pathlib import Path

from src.nlp.llm_client import ChatResult
from src.nlp.prompt_engine import PromptEngine
from src.nlp.translate import Translator
from src.profiles.loader import get_profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def block_json(pairs, **extra):
    return json.dumps({
        "sentences": [{"id": i, "source": s, "translation": t} for i, s, t in pairs],
        "glossary_updates": extra.get("glossary_updates", []),
        "style_updates": extra.get("style_updates", ""),
    }, ensure_ascii=False)


class StubLLM:
    online = True

    def __init__(self, script):
        self.script = list(script)

    def chat(self, prompt, **kw):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResult(item, prompt_tokens=10, completion_tokens=5,
                          attempts=1, latency_sec=0.5, json_mode=True)


def make(script):
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    dumps: list[str] = []
    t = Translator(
        StubLLM(script), PromptEngine(PROJECT_ROOT / "config" / "prompts"),
        profile, debug=dumps.append,
    )
    return t, dumps


def test_debug_dump_contains_prompt_and_response():
    t, dumps = make([block_json([(1, "おはよう", "早上好")],
                                glossary_updates=[{"term": "x", "translation": "y"}])])
    t.translate_block(["おはよう"])
    assert len(dumps) == 1
    out = dumps[0]
    assert "LLM 块翻译 #1" in out
    assert "[请求 prompt]" in out and "おはよう" in out  # prompt 全文含待译句
    assert "[响应]" in out and "早上好" in out          # 原始响应
    assert "tokens 10+5=15" in out and "json_mode=开" in out
    assert "[对齐] 成功 1/1 句" in out
    assert "+术语 x→y" in out                            # 记忆回写差异
    assert "おはよう  →  早上好" in out                  # 最终输出


def test_debug_dump_shows_fallback_chain():
    bad = "not json at all {broken"
    t, dumps = make([bad, bad, block_json([(3, "x", "译")])])
    t.translate_block(["x"])
    out = dumps[0]
    assert "尝试 3/3" in out          # 整块 + 重试 + 逐句兜底
    assert "逐句兜底" in out
    assert out.count("[失败]") == 2   # 前两次失败原因可见


def test_no_debug_no_overhead():
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    t = Translator(
        StubLLM([block_json([(1, "a", "甲")])]),
        PromptEngine(PROJECT_ROOT / "config" / "prompts"), profile,
    )
    results = t.translate_block(["a"])
    assert results[0].translation == "甲"  # debug=None 时行为不变
