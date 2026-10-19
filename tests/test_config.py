from pathlib import Path

import pytest

from src.config import API_KEY_ENV, LLMConfig, find_project_root, load_config

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
