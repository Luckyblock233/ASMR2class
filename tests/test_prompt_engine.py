import time
from pathlib import Path

import pytest

from src.nlp.prompt_engine import PromptEngine, PromptTemplateError

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "config" / "prompts"


@pytest.fixture
def engine():
    return PromptEngine(PROMPTS_DIR)


ALL_VARS = dict(
    source_lang="ja",
    target_lang="zh",
    sentences='[{"id": 1, "source": "こんにちは"}]',
    context_history="",
    glossary="",
    style_notes="",
)


def test_render_default_translate(engine):
    path = engine.resolve(None, "translate")
    assert path.name == "translate.txt"
    out = engine.render(path, **ALL_VARS)
    assert "こんにちは" in out
    # 所有变量已渲染
    for var in ("source_lang", "target_lang", "sentences",
                "context_history", "glossary", "style_notes"):
        assert "{" + var + "}" not in out
    # 注释行（# 开头）不出现在渲染结果中
    assert not any(line.lstrip().startswith("#") for line in out.splitlines())


def test_template_variables_detected(engine):
    path = engine.resolve("ja/translate.txt", "translate")
    declared = engine.variables(path)
    assert {"source_lang", "target_lang", "sentences",
            "context_history", "glossary", "style_notes"} <= declared


def test_language_specific_template(engine):
    path = engine.resolve("ja/translate.txt", "translate")
    assert "ja" in path.parts
    out = engine.render(path, **ALL_VARS)
    assert "日语" in out


def test_fallback_to_default(engine):
    path = engine.resolve("nonexist/translate.txt", "translate")
    assert path.parent.name == "_default"


def test_render_grammar_templates(engine):
    variables = dict(
        source_lang="ja",
        target_lang="zh",
        sentence="早く知っていれば。",
        translation="要是早知道的话。",
        level_hint="N3",
    )
    for rel_path in (None, "ja/grammar.txt", "en/grammar.txt"):
        path = engine.resolve(rel_path, "grammar")
        rendered = engine.render(path, **variables)
        assert "早く知っていれば。" in rendered
        assert "N3" in rendered
        assert "{" + "sentence" + "}" not in rendered


def test_missing_template_raises(engine):
    with pytest.raises(PromptTemplateError, match="未找到"):
        engine.resolve(None, "missing_kind")


def test_missing_variable_reports_line(engine, tmp_path):
    tpl = tmp_path / "t.txt"
    tpl.write_text("第一行\n第二行 {unknown_var}\n", encoding="utf-8")
    with pytest.raises(PromptTemplateError, match="第 2 行"):
        engine.render(tpl, source_lang="ja")


def test_mtime_hot_reload(engine, tmp_path):
    tpl = tmp_path / "t.txt"
    tpl.write_text("v1 {sentences}", encoding="utf-8")
    assert engine.render(tpl, sentences="s").startswith("v1")
    # 保证 mtime 变化
    time.sleep(0.01)
    tpl.write_text("v2 {sentences}", encoding="utf-8")
    import os

    os.utime(tpl, (time.time() + 1, time.time() + 1))
    assert engine.render(tpl, sentences="s").startswith("v2")
