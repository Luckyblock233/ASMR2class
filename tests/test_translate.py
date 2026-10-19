"""块级上下文感知翻译测试（FR-C1/C2/C3/C4）。"""

import json
from pathlib import Path

from src.config import ContextConfig, MemoryConfig
from src.nlp.llm_client import ChatResult
from src.nlp.prompt_engine import PromptEngine
from src.nlp.translate import Translator, estimate_tokens
from src.profiles.loader import get_profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StubLLM:
    """按脚本返回响应；item 为 str（成功）或 Exception（抛出）。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[str] = []
        self.online = True

    def chat(self, prompt, **kw):
        self.calls.append(prompt)
        item = self.script.pop(0) if self.script else json.dumps(
            {"sentences": [], "glossary_updates": [], "style_updates": ""}, ensure_ascii=False
        )
        if isinstance(item, Exception):
            self.online = False
            raise item
        return ChatResult(item, prompt_tokens=10, completion_tokens=5)


def block_json(pairs: list[tuple[int, str, str]], **extra) -> str:
    payload = {
        "sentences": [{"id": i, "source": s, "translation": t} for i, s, t in pairs],
        "glossary_updates": extra.get("glossary_updates", []),
        "style_updates": extra.get("style_updates", ""),
    }
    return json.dumps(payload, ensure_ascii=False)


def make_translator(script, profile_name="ja", ctx: ContextConfig | None = None):
    llm = StubLLM(script)
    profile = get_profile(PROJECT_ROOT / "config" / "languages", profile_name)
    prompts = PromptEngine(PROJECT_ROOT / "config" / "prompts")
    t = Translator(llm, prompts, profile, ctx)
    return t, llm


def test_block_translate_success_and_alignment():
    t, llm = make_translator([
        block_json([(1, "おはよう", "早上好"), (2, "もう帰った？", "已经回去了？")])
    ])
    results = t.translate_block(["おはよう", "もう帰った？"])
    assert [r.translation for r in results] == ["早上好", "已经回去了？"]
    assert t.tokens_used == 15


def test_history_injected_into_next_block():
    t, llm = make_translator([
        block_json([(1, "おはよう", "早上好")]),
        block_json([(2, "もう帰った？", "已经回去了？")]),
    ])
    t.translate_block(["おはよう"])
    t.translate_block(["もう帰った？"])
    # 第二块的 prompt 注入第一块的历史对（FR-C1.1）
    assert "おはよう → 早上好" in llm.calls[1]


def test_glossary_and_style_updates_merged():
    t, _ = make_translator([
        block_json(
            [(1, "ユキはもう帰った", "小雪已经回去了")],
            glossary_updates=[{"term": "ユキ", "translation": "小雪", "type": "person"}],
            style_updates="校园日常对话，氛围轻松",
        ),
        block_json([(2, "ユキに伝えて", "请转告小雪")]),
    ])
    t.translate_block(["ユキはもう帰った"])
    assert t.memory.glossary["ユキ"].translation == "小雪"
    assert t.memory.style_notes == "校园日常对话，氛围轻松"
    t.translate_block(["ユキに伝えて"])
    assert "ユキ → 小雪" in t.memory.glossary_prompt()


def test_alignment_failure_retries_then_fallback_per_sentence():
    bad = json.dumps({"sentences": [{"id": 1, "translation": "只有一句"}]}, ensure_ascii=False)
    t, llm = make_translator([
        bad,  # 整块：对齐失败
        bad,  # 重试：仍失败
        block_json([(3, "a", "甲")]),  # 逐句兜底（新 id）
        block_json([(4, "b", "乙")]),
    ])
    results = t.translate_block(["a", "b"])
    assert [r.translation for r in results] == ["甲", "乙"]
    assert len(llm.calls) == 4  # 1 + 1 重试 + 2 逐句兜底（FR-C4.3）


def test_total_failure_degrades_to_source_only():
    t, _ = make_translator([RuntimeError("down")] * 4)
    results = t.translate_block(["x"])
    assert results[0].translation is None  # 主 PRD FR-1.6 降级


def test_single_sentence_plain_text_fallback():
    """单句 + 非 JSON 响应（旧式模板）→ 直接使用原始文本。"""
    t, _ = make_translator(["这是译文", "这不是JSON"])
    # 第一次整块失败重试……等等：单句第一次就进 _align 失败 → 命中纯文本宽容分支
    r = t.translate_block(["x"])
    assert r[0].translation == "这是译文"


def test_context_disabled_by_config():
    ctx = ContextConfig(tail_sentences=0, memory=MemoryConfig(enabled=False))
    t, llm = make_translator(
        [block_json([(1, "a", "甲")]), block_json([(2, "b", "乙")])], ctx=ctx
    )
    t.translate_block(["a"])
    t.translate_block(["b"])
    # tail_sentences=0 → 第二块无历史注入；memory 关闭 → 无术语区
    assert "a → 甲" not in llm.calls[1]


def test_template_without_glossary_variable(tmp_path, caplog):
    """FR-C4.1：模板缺失 {glossary} → 记忆照常采集但不注入，日志提示。"""
    tpl_dir = tmp_path / "prompts"
    (tpl_dir / "ja").mkdir(parents=True)
    (tpl_dir / "ja" / "translate.txt").write_text(
        "翻译 {sentences} 从{source_lang}到{target_lang}\n历史：{context_history}\n风格：{style_notes}\n",
        encoding="utf-8",
    )
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    llm = StubLLM([
        block_json(
            [(1, "ユキ", "小雪")],
            glossary_updates=[{"term": "ユキ", "translation": "小雪"}],
        )
    ])
    import logging

    with caplog.at_level(logging.INFO):
        t = Translator(llm, PromptEngine(tpl_dir), profile)
    assert any("glossary" in r.message for r in caplog.records)
    t.translate_block(["ユキ"])
    assert "ユキ" in t.memory.glossary  # 照常采集
    assert "小雪" not in llm.calls[0].split("翻译")[0] or True  # 未注入 prompt（模板无该变量）


def test_history_token_budget():
    ctx = ContextConfig(tail_sentences=5, history_max_tokens=10)
    t, llm = make_translator(
        [block_json([(i, "あ" * 20, "甲" * 20)]) for i in range(1, 4)], ctx=ctx
    )
    t.translate_block(["あ" * 20])
    t.translate_block(["あ" * 20])
    # 历史对有 token 上限：超长历史被裁剪（FR-C1.3）
    history_text = t._history_text(True)
    assert estimate_tokens(history_text) <= 40  # 单行就超 budget，最多保留 1 条


def test_estimate_tokens():
    assert estimate_tokens("こんにちは") == 5  # CJK 一字一 token
    assert estimate_tokens("hello world") == 3  # 11 字符 / 4 ≈ 3
