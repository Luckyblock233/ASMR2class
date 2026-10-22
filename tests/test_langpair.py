"""自由语言对模式：语言包合成、pair 模板解析、会话文件名语言标注。"""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from src.config import load_config
from src.nlp.prompt_engine import PromptEngine
from src.profiles.loader import (
    AUTO_LANG,
    build_profile,
    get_profile,
    load_profiles,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANGS = PROJECT_ROOT / "config" / "languages"


def _packs():
    return load_profiles(LANGS)


def test_build_profile_overrides_target_keeps_pack():
    ja = _packs()["ja"]
    p = build_profile(_packs(), "ja", "en")
    assert p.source_lang == "ja" and p.target_lang == "en"
    assert p.prompts == ja.prompts          # 模板/语法卡来自源语言包
    assert p.raw.get("grammar_card") == ja.raw.get("grammar_card")
    assert ja.target_lang == "zh"           # 原包不被修改


def test_build_profile_auto_source():
    p = build_profile(_packs(), AUTO_LANG, "zh")
    assert p.source_lang == "auto"
    assert p.language_hint is None          # ASR 自动检测
    assert p.prompts == {}                  # 走 _default 模板
    assert p.raw["grammar_card"]            # 通用语法卡存在


def test_build_profile_unknown_source_raises():
    with pytest.raises(KeyError):
        build_profile(_packs(), "fr", "zh")


def test_resolve_pair_prefers_pair_dir(tmp_path):
    prompts = tmp_path / "prompts"
    (prompts / "_default").mkdir(parents=True)
    (prompts / "ja").mkdir()
    (prompts / "ja-en").mkdir()
    (prompts / "_default" / "translate.txt").write_text("默认", encoding="utf-8")
    (prompts / "ja" / "translate.txt").write_text("包内", encoding="utf-8")
    (prompts / "ja-en" / "translate.txt").write_text("对定制", encoding="utf-8")
    engine = PromptEngine(prompts)

    # pair 目录优先
    assert "ja-en" in str(engine.resolve_pair("translate", "ja/translate.txt", "ja", "en"))
    # 无 pair 目录回落包内
    assert "ja" in str(engine.resolve_pair("translate", "ja/translate.txt", "ja", "zh"))
    # auto 源跳过 pair 目录，包内路径为空时落 _default
    assert "_default" in str(engine.resolve_pair("translate", None, "auto", "zh"))


def test_session_filename_marks_language_pair(tmp_path):
    """保存的分析文件名含源语言与目标语言；auto 源标注 auto。"""
    from src.subtitle.session import SessionRecorder

    ja = get_profile(LANGS, "ja")
    recorder = SessionRecorder(build_profile(_packs(), "ja", "en"))
    recorder.add(0, 1, "原文", "translation")
    out = recorder.save(tmp_path, "pair")
    assert (out / "analysis.ja.en.json").is_file()

    recorder = SessionRecorder(build_profile(_packs(), AUTO_LANG, "zh"))
    recorder.add(0, 1, "何か", "什么")
    out = recorder.save(tmp_path, "auto")
    assert (out / "analysis.auto.zh.json").is_file()
    payload = json.loads((out / "analysis.auto.zh.json").read_text(encoding="utf-8"))
    assert payload["source_lang"] == "auto" and payload["target_lang"] == "zh"


def test_resolve_profile_free_pair_and_auto():
    """main._resolve_profile：自由语言对 / auto 源 / 档案默认。"""
    from src.main import _resolve_profile

    cfg = load_config(PROJECT_ROOT)
    base = dict(profile=None, auto_lang=False, source_lang="", target_lang="")
    p = _resolve_profile(Namespace(**{**base, "source_lang": "en", "target_lang": "ja"}), cfg)
    assert (p.source_lang, p.target_lang) == ("en", "ja")
    p = _resolve_profile(Namespace(**{**base, "auto_lang": True, "target_lang": "zh"}), cfg)
    assert p.source_lang == "auto" and p.target_lang == "zh"
    p = _resolve_profile(Namespace(**{**base, "profile": "ja"}), cfg)
    assert (p.source_lang, p.target_lang) == ("ja", "zh")  # 档案默认对不变


def test_replay_profile_from_auto_document(tmp_path):
    """auto 源会话回放：档案按文档语言对合成，不要求档案名存在于语言包。"""
    from src.main import _resolve_replay_profile

    from src.subtitle.document import SessionDocument

    cfg = load_config(PROJECT_ROOT)
    doc = SessionDocument(language_profile="auto-zh", source_lang="auto", target_lang="zh")
    doc.add(0, 1, "hello", "你好")
    p = _resolve_replay_profile(Namespace(profile=None), cfg, doc)
    assert p.source_lang == "auto" and p.target_lang == "zh"


def test_zh_pack_templates_resolve_and_render():
    """中文语言包：translate/grammar 模板可解析、变量可渲染、语法卡字段与模板契约一致。"""
    zh = _packs()["zh"]
    engine = PromptEngine(PROJECT_ROOT / "config" / "prompts")

    t = engine.resolve_pair("translate", zh.prompt_path("translate"), "zh", "en")
    assert t.parent.name == "zh"
    rendered = engine.render(
        t, source_lang="zh", target_lang="en", sentences="[]",
        context_history="", glossary="", style_notes="",
    )
    assert "中文" in rendered and "en" in rendered

    g = engine.resolve_pair("grammar", zh.prompt_path("grammar"), "zh", "en")
    assert g.parent.name == "zh"
    rendered = engine.render(
        g, source_lang="zh", target_lang="en",
        sentence="我把作业写完了", translation="I finished my homework",
        level_hint="HSK 1-4",
    )
    assert "把" in rendered  # 虚词讲解提示在模板中
    # 语法卡声明的 badge/列字段与模板契约一致（hsk）
    card = zh.raw["grammar_card"]["sections"]
    gp = next(s for s in card if s["key"] == "grammar_points")
    assert gp["badge"]["field"] == "hsk"


def test_session_dir_default_name_is_language_pair(tmp_path):
    """未指定会话名时，目录名默认 <源>-<目标>；auto 源为 auto-<目标>。"""
    from src.subtitle.session import SessionRecorder

    recorder = SessionRecorder(build_profile(_packs(), "ja", "en"))
    recorder.add(0, 1, "原文", "translation")
    out = recorder.save(tmp_path)
    assert out.name.endswith("-ja-en")

    recorder = SessionRecorder(build_profile(_packs(), AUTO_LANG, "zh"))
    recorder.add(0, 1, "何か", "什么")
    out = recorder.save(tmp_path)
    assert out.name.endswith("-auto-zh")
