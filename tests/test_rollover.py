"""语言档案运行中切换的会话滚动（交接文档 §4.2）。"""

from pathlib import Path

from src.config import load_config
from src.main import rollover_profile
from src.nlp.prompt_engine import PromptEngine
from src.nlp.providers.registry import ProviderRegistry
from src.nlp.slots import ModelSlots, resolve_slots
from src.nlp.translate import Translator
from src.profiles.loader import get_profile
from src.subtitle.session import SessionRecorder

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FakeStreamer:
    """rollover 只依赖 set_paused/language_hint 两个接口。"""

    def __init__(self):
        self.paused_states: list[bool] = []
        self.language_hint = None

    def set_paused(self, paused: bool) -> None:
        self.paused_states.append(paused)


def _make_runtime(tmp_path, *, translate: bool = True):
    cfg = load_config(PROJECT_ROOT)
    ja = get_profile(cfg.languages_dir, "ja")
    registry = ProviderRegistry(cfg.providers)
    slots = ModelSlots(registry, resolve_slots(cfg.slots, ja.slots, registry))
    recorder = SessionRecorder(ja)
    recorder.add(0.0, 1.0, "おはよう", "早上好")
    translator = (
        Translator(lambda: None, PromptEngine(cfg.prompts_dir), ja, cfg.context)
        if translate
        else None
    )
    from src.nlp.blocker import Blocker

    on_block_calls = []
    blocker = (
        Blocker(
            cfg.context.merged(ja.context).block,
            lambda block: on_block_calls.append(block),
            autoflush=False,
        )
        if translate
        else None
    )
    runtime = {
        "profile": ja,
        "translator": translator,
        "grammar": None,
        "recorder": recorder,
        "blocker": blocker,
        "on_block": (lambda block: on_block_calls.append(block)) if translate else None,
        "on_grammar_update": None,
    }
    streamer = FakeStreamer()
    return cfg, slots, runtime, streamer


def test_rollover_saves_old_session_and_rebuilds(tmp_path):
    cfg, slots, runtime, streamer = _make_runtime(tmp_path)
    old_recorder = runtime["recorder"]
    old_translator = runtime["translator"]
    old_blocker = runtime["blocker"]

    new_profile, err, saved_dir = rollover_profile(
        runtime, get_profile(cfg.languages_dir, "en"),
        cfg=cfg, slots=slots, streamer=streamer,
        output_root=tmp_path, session_name="",
    )
    assert err is None and new_profile.name == "en"

    # 1. 旧会话完整保存（单语言文件，不混写）
    saved = list(tmp_path.glob("*/analysis.ja.zh.json"))
    assert len(saved) == 1
    content = saved[0].read_text(encoding="utf-8")
    assert "おはよう" in content and "早上好" in content
    assert not list(tmp_path.glob("*/analysis.en.json"))

    # 2. 运行态整体重建
    assert runtime["profile"].name == "en"
    assert runtime["recorder"] is not old_recorder
    assert len(runtime["recorder"].document) == 0
    assert runtime["recorder"].document.language_profile == "en"
    assert runtime["translator"] is not old_translator
    assert runtime["translator"].profile.source_lang == "en"
    assert runtime["blocker"] is not old_blocker
    assert runtime["grammar"] is not None  # slots 可用时始终创建
    runtime["grammar"].drain()

    # 3. 暂停/恢复成对发生；ASR 语种提示更新
    assert streamer.paused_states == [True, False]
    assert streamer.language_hint == "en"


def test_rollover_without_translation(tmp_path):
    cfg, slots, runtime, streamer = _make_runtime(tmp_path, translate=False)
    new_profile, err, saved_dir = rollover_profile(
        runtime, get_profile(cfg.languages_dir, "en"), cfg=cfg, slots=slots, streamer=streamer, output_root=tmp_path,
    )
    assert err is None
    assert runtime["translator"] is None  # 不翻译模式不创建 Translator
    assert runtime["blocker"] is None
    assert runtime["profile"].name == "en"
    runtime["grammar"].drain()


def test_rollover_unknown_profile_keeps_runtime(tmp_path):
    """未知语言包在 build_profile 阶段报错，rollover 不会被调用。"""
    import pytest

    from src.profiles.loader import build_profile, load_profiles

    cfg, slots, runtime, streamer = _make_runtime(tmp_path)
    with pytest.raises(KeyError):
        build_profile(load_profiles(cfg.languages_dir), "no-such-lang", "zh")
    assert runtime["profile"].name == "ja"
    assert streamer.paused_states == []
    assert list(tmp_path.iterdir()) == []


def test_rollover_same_profile_is_noop(tmp_path):
    cfg, slots, runtime, streamer = _make_runtime(tmp_path)
    profile, err, _dir = rollover_profile(
        runtime, runtime["profile"], cfg=cfg, slots=slots, streamer=streamer, output_root=tmp_path,
    )
    assert err is None and profile.name == "ja"
    assert streamer.paused_states == []
    assert list(tmp_path.iterdir()) == []


def test_rollover_flushes_pending_block(tmp_path):
    """切换时 Blocker 残余句先用旧 Translator 冲刷，再重建。"""
    cfg, slots, runtime, streamer = _make_runtime(tmp_path)
    blocker = runtime["blocker"]
    from src.asr.streamer import Sentence

    # 与真实流程一致：ASR 句先登记文档（占位），再喂 Blocker
    runtime["recorder"].add(1.5, 2.0, "残りの文", None)
    blocker.feed(Sentence(1.5, 2.0, "残りの文"))
    new_profile, err, saved_dir = rollover_profile(
        runtime, get_profile(cfg.languages_dir, "en"), cfg=cfg, slots=slots, streamer=streamer, output_root=tmp_path,
    )
    assert err is None
    # 残余句已随旧会话保存（无 Provider，仅原文）
    saved = list(tmp_path.glob("*/analysis.ja.zh.json"))
    assert len(saved) == 1
    assert "残りの文" in saved[0].read_text(encoding="utf-8")
    runtime["grammar"].drain()


def test_rollover_returns_saved_dir_and_resume_control(tmp_path):
    """返回旧会话保存目录；resume=False 时保持暂停，由调用方恢复（GUI 竞态防护）。"""
    cfg, slots, runtime, streamer = _make_runtime(tmp_path)
    new_profile, err, saved_dir = rollover_profile(
        runtime, get_profile(cfg.languages_dir, "en"), cfg=cfg, slots=slots, streamer=streamer,
        output_root=tmp_path, resume=False,
    )
    assert err is None
    assert saved_dir and (tmp_path / saved_dir.split("/")[-1].split("\\")[-1]).exists() \
        or Path(saved_dir).exists()
    assert "analysis.ja.zh.json" in str(list(Path(saved_dir).iterdir()))
    assert streamer.paused_states == [True]  # 未恢复
    streamer.set_paused(False)             # 调用方（GUI 应用完毕）恢复
    runtime["grammar"].drain()


def test_stylesheets_balanced_and_valid():
    """回归：生成的样式表不得有非法花括号（f-string 转义泄漏曾导致解析失败）。"""
    from src.ui.appearance import THEMES, AppearanceSpec

    for theme in THEMES.values():
        spec = AppearanceSpec(theme=theme)
        for css in (spec.block_stylesheet, spec.bars_stylesheet, spec.menu_stylesheet):
            assert css.count("{") == css.count("}"), css
            assert "}}" not in css and "{{" not in css


def test_exit_save_uses_current_profile_not_startup(tmp_path):
    """回归：切换档案后退出，保存目录名与分析文件语言对都属于当前档案。"""
    from src.main import save_current_session

    cfg, slots, runtime, streamer = _make_runtime(tmp_path)
    en = get_profile(cfg.languages_dir, "en")
    new_profile, err, _ = rollover_profile(
        runtime, en, cfg=cfg, slots=slots, streamer=streamer, output_root=tmp_path,
    )
    assert err is None
    runtime["recorder"].add(0.0, 1.0, "hello", "你好")

    out_dir = save_current_session(runtime, tmp_path / "exit")
    assert out_dir is not None
    assert out_dir.name.endswith("-en-zh")                  # 目录名是当前语言对
    assert (out_dir / "analysis.en.zh.json").is_file()      # 文件语言对也是当前档案
    assert not list(out_dir.glob("analysis.ja.*.json"))     # 没有启动档案的残留
    runtime["grammar"].drain()
