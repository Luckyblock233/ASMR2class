from pathlib import Path

import pytest
import yaml

from src.config import (
    API_KEY_ENV,
    LLMConfig,
    SlotsConfig,
    UIConfig,
    find_project_root,
    load_config,
    save_provider_config,
    save_ui_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_example_template_defaults(tmp_path):
    """入库模板 config.example.yaml 的默认值（用户本地 config.yaml 不影响本断言）。"""
    import shutil

    (tmp_path / "config").mkdir()
    shutil.copy(PROJECT_ROOT / "config" / "config.example.yaml",
                tmp_path / "config" / "config.example.yaml")
    cfg = load_config(tmp_path)
    assert cfg.asr.model_size == "small"
    assert cfg.asr.vram_budget_mb == 2500
    assert cfg.audio.sample_rate == 16000
    assert cfg.default_profile == "ja"
    assert cfg.context.tail_sentences == 3
    assert cfg.context.block.max_sentences == 8
    assert cfg.ui.overlay is True
    # 模板的 providers/slots 节全部注释掉：默认无 Provider、无绑定（向导负责注册）
    assert cfg.providers == []
    assert cfg.slots.translation == ""


def test_no_hardcoded_llm_provider():
    """FR-1.5：代码中的 LLM 默认配置不得预设任何服务商。"""
    default = LLMConfig()
    assert default.base_url == ""
    assert default.api_key == ""
    assert default.model == ""
    assert not default.is_complete


def test_env_overrides_api_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "sk-test-env-key")
    cfg = load_config(PROJECT_ROOT)
    assert cfg.llm.api_key == "sk-test-env-key"


def test_llm_merged_profile_override():
    base = LLMConfig(base_url="https://a.example/v1", api_key="k1", model="m1")
    merged = base.merged({"base_url": "https://b.example/v1", "model": "m2"})
    assert merged.base_url == "https://b.example/v1"
    assert merged.model == "m2"
    assert merged.api_key == "k1"  # 未覆盖项回落全局
    assert base.merged(None) is base
    assert base.merged({}) is base


def test_find_project_root(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text("{}", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path.resolve()


def test_find_project_root_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        find_project_root(tmp_path)


def test_first_run_creates_config_from_example(tmp_path):
    """首次运行：config.yaml 缺失时自动从 example 复制创建。"""
    from src.config import ensure_config_file

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.example.yaml").write_text(
        "llm:\n  base_url: ''\n  api_key: ''\n  model: ''\nasr:\n  model_size: small\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert (tmp_path / "config" / "config.yaml").is_file()
    assert cfg.asr.model_size == "small"
    assert not cfg.llm.is_complete  # example 模板不含服务商
    # 已存在时不覆盖用户配置
    (tmp_path / "config" / "config.yaml").write_text(
        "llm:\n  base_url: https://user.example/v1\n  api_key: k\n  model: m\n",
        encoding="utf-8",
    )
    ensure_config_file(tmp_path)
    assert "user.example" in (tmp_path / "config" / "config.yaml").read_text(encoding="utf-8")


def test_find_root_with_example_only(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.example.yaml").write_text("{}", encoding="utf-8")
    assert find_project_root(tmp_path) == tmp_path.resolve()


def _write_example(tmp_path: Path, extra: str = "") -> Path:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.example.yaml").write_text(
        "llm:\n  base_url: ''\n  api_key: ''\n  model: ''\n" + extra,
        encoding="utf-8",
    )
    return tmp_path


def test_ui_appearance_defaults(tmp_path):
    cfg = load_config(_write_example(tmp_path))
    ui = cfg.ui
    assert ui.overlay is True
    assert ui.theme == "dark"
    assert ui.background_opacity is None
    assert ui.effective_background_opacity == 0.85  # 迁移自旧 opacity
    assert ui.show_bar_a and ui.show_bar_b and ui.show_bar_c
    assert ui.auto_scroll is True and ui.compact_mode is False
    assert ui.ui_language == "zh_CN"
    assert ui.view_capacity == 200
    # 逐项颜色/字体配置已移除，由主题预定义
    assert not hasattr(ui, "source_color")
    assert not hasattr(ui, "background_color")


def test_ui_opacity_migration_and_explicit_background_opacity(tmp_path):
    cfg = load_config(_write_example(tmp_path, "ui:\n  opacity: 0.5\n"))
    assert cfg.ui.effective_background_opacity == 0.5

    cfg = load_config(
        _write_example(tmp_path / "b", "ui:\n  opacity: 0.5\n  background_opacity: 0.3\n")
    )
    assert cfg.ui.effective_background_opacity == 0.3  # 新字段优先


def test_ui_legacy_color_fields_ignored_on_load(tmp_path):
    """旧配置的颜色/字体字段被忽略（主题取代），加载不报错。"""
    cfg = load_config(
        _write_example(
            tmp_path,
            "ui:\n  background_color: '#112233'\n  source_color: '#FFFFFF'\n  theme: light\n",
        )
    )
    assert cfg.ui.theme == "light"


def test_ui_merged_profile_appearance_override():
    base = UIConfig()
    assert base.merged(None) is base
    assert base.merged({}) is base
    merged = base.merged(
        {"theme": "light", "background_opacity": 0.6, "unknown_key": 1, "auto_scroll": False}
    )
    assert merged.theme == "light"
    assert merged.effective_background_opacity == 0.6
    assert merged.auto_scroll is False
    assert base.theme == "dark"  # 原对象不被修改


def test_save_ui_config_atomic_and_preserves_other_sections(tmp_path):
    root = _write_example(tmp_path)
    cfg = load_config(root)
    path = cfg.config_file
    # 预置一个无关字段和 ui 节的旧字段，验证保存后保留无关节、清掉旧颜色字段
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["custom_section"] = {"keep": True}
    raw["ui"] = {"opacity": 0.7, "background_color": "#112233"}  # 旧字段应被清除
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    ui = UIConfig(theme="light", background_opacity=0.6, view_capacity=100)
    save_ui_config(cfg, ui)

    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["custom_section"] == {"keep": True}
    assert saved["ui"]["opacity"] == 0.7  # 旧字段保留作迁移输入
    assert saved["ui"]["theme"] == "light"
    assert saved["ui"]["background_opacity"] == 0.6
    assert saved["ui"]["view_capacity"] == 100
    assert "background_color" not in saved["ui"]  # 旧颜色字段不再写回
    assert not list(path.parent.glob("*.tmp"))  # 原子写不留临时文件


def test_save_ui_config_roundtrips_every_field(tmp_path):
    """回归：UIConfig 所有可持久化字段保存→重载后一致（曾漏写 bubble_fade_delay_sec）。

    新增 UIConfig 字段时本测试强制同步 save_ui_config 的写回映射。
    """
    import dataclasses

    root = _write_example(tmp_path)
    cfg = load_config(root)
    ui = UIConfig(
        overlay=False,
        theme="light",
        background_opacity=0.6,
        line_spacing=1.4,
        window_width=800,
        window_height=600,
        show_bar_a=False,
        show_bar_b=False,
        show_bar_c=False,
        auto_scroll=False,
        compact_mode=True,
        ui_language="en",
        view_capacity=100,
        latency_good_ms=300,
        latency_warn_ms=900,
        ram_warn_percent=70,
        bubble_theme="light",
        bubble_opacity=0.5,
        bubble_font_size=22,
        bubble_fade="hide",
        bubble_fade_min=0.4,
        bubble_rms_threshold=0.05,
        bubble_idle_text="♪",
        bubble_fade_delay_sec=3.5,
        bubble_idle_hold_sec=4.5,
    )
    save_ui_config(cfg, ui)
    reloaded = load_config(root).ui
    skipped = {"opacity"}  # 旧字段：仅作迁移输入，不写回
    for f in dataclasses.fields(UIConfig):
        if f.name in skipped:
            continue
        expected = getattr(ui, f.name)
        if f.name == "background_opacity":
            expected = ui.effective_background_opacity  # 写回的是生效值
        actual = getattr(reloaded, f.name)
        assert actual == pytest.approx(expected), f"字段 {f.name} 未持久化"


def test_ui_zero_float_values_respected(tmp_path):
    """回归：0 是合法配置值（淡出延时 0 = 立即触发），加载不得被默认值顶替。"""
    root = _write_example(tmp_path)
    cfg = load_config(root)
    raw = yaml.safe_load(cfg.config_file.read_text(encoding="utf-8"))
    raw.setdefault("ui", {}).update(
        {
            "bubble_fade_delay_sec": 0.0,
            "bubble_idle_hold_sec": 0.0,
            "bubble_fade_min": 0.0,
        }
    )
    cfg.config_file.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    ui = load_config(root).ui
    assert ui.bubble_fade_delay_sec == 0.0
    assert ui.bubble_idle_hold_sec == 0.0
    assert ui.bubble_fade_min == 0.0


def test_save_provider_config_preserves_fields_and_skips_env_key(tmp_path, monkeypatch):
    root = _write_example(tmp_path)
    cfg = load_config(root)
    path = cfg.config_file
    path.write_text(
        yaml.safe_dump(
            {
                "providers": [
                    {
                        "id": "p1",
                        "name": "旧名",
                        "type": "openai_compatible",
                        "base_url": "https://old.example/v1",
                        "api_key": "file-key",
                        "model": "old-model",
                        "capabilities": ["translation"],
                        "timeout_sec": 45,
                        "max_retries": 5,
                        "custom_future_field": {"nested": [1, 2]},
                    }
                ],
                "slots": {"translation": "p1", "grammar": ""},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(API_KEY_ENV, "sk-env-injected")
    cfg = load_config(root)
    provider = next(p for p in cfg.providers if p.id == "p1")
    provider.name = "新名"
    assert provider.api_key == "file-key"  # env 只填充空 key，不覆盖文件值
    provider.api_key = "sk-env-injected"  # 模拟运行态中密钥来自环境变量注入
    save_provider_config(cfg, provider, SlotsConfig(translation="p1", grammar=""))

    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    entry = saved["providers"][0]
    assert entry["name"] == "新名"
    assert entry["api_key"] == ""  # 环境变量注入的密钥不落盘
    assert entry["timeout_sec"] == 45 and entry["max_retries"] == 5  # 字段不丢失
    assert entry["custom_future_field"] == {"nested": [1, 2]}  # 未知字段保留
    assert saved["slots"] == {"translation": "p1", "grammar": ""}
