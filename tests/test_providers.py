"""Provider 配置解析与槽位绑定解析测试（模型可插拔架构）。"""

from pathlib import Path

import pytest

from src.config import load_config
from src.nlp.providers import (
    CAP_GLOSSARY_CONSTRAINT,
    CAP_GRAMMAR,
    CAP_TRANSLATION,
    filter_for_slot,
    slot_error,
)
from src.nlp.slots import ModelSlots, resolve_slots
from src.nlp.providers.registry import ProviderRegistry

PROVIDERS_YAML = """\
providers:
  - id: cloud-a
    name: "云端 A"
    type: openai_compatible
    base_url: "https://a.example/v1"
    api_key: "k1"
    model: "m1"
    capabilities: [translation, grammar]
  - id: local-tr
    name: "本地翻译"
    type: llama_cpp_server
    base_url: "http://127.0.0.1:8080/v1"
    api_key: "EMPTY"
    model: "local-tr"
    capabilities: [translation, glossary_constraint]
    params:
      temperature: 0
      max_tokens: 1024
      enable_thinking: false
    local:
      managed: true
      gguf_path: "models/x.gguf"
      server_bin: "llama-server"
      server_args: "--port 8080"
      vram_budget_mb: 1500
      download:
        hf_repo: "org/repo"
        hf_file: "x.gguf"
slots:
  translation: local-tr
  grammar: cloud-a
"""


def _write_config(tmp_path: Path, body: str) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text(body, encoding="utf-8")
    return tmp_path


def test_parse_providers_and_slots(tmp_path):
    cfg = load_config(_write_config(tmp_path, PROVIDERS_YAML))
    assert len(cfg.providers) == 2
    local = next(p for p in cfg.providers if p.id == "local-tr")
    assert local.type == "llama_cpp_server"
    assert local.capabilities == [CAP_TRANSLATION, CAP_GLOSSARY_CONSTRAINT]
    assert local.params == {"temperature": 0, "max_tokens": 1024, "enable_thinking": False}
    assert local.local.managed is True
    assert local.local.gguf_path == str(tmp_path / "models" / "x.gguf")  # 相对路径锚定到项目根
    assert local.local.download["hf_repo"] == "org/repo"
    assert cfg.slots.translation == "local-tr"
    assert cfg.slots.grammar == "cloud-a"


def test_legacy_llm_synthesizes_default_provider(tmp_path):
    body = """\
llm:
  base_url: "https://a.example/v1"
  api_key: "k1"
  model: "m1"
"""
    cfg = load_config(_write_config(tmp_path, body))
    assert [p.id for p in cfg.providers] == ["default"]
    p = cfg.providers[0]
    assert p.capabilities == [CAP_TRANSLATION, CAP_GRAMMAR]
    assert p.base_url == "https://a.example/v1"
    assert cfg.slots.translation == "default"
    assert cfg.slots.grammar == "default"


def test_legacy_llm_incomplete_no_synthesis(tmp_path):
    cfg = load_config(_write_config(tmp_path, "llm:\n  base_url: ''\n  api_key: ''\n  model: ''\n"))
    assert cfg.providers == []
    assert cfg.slots.translation == ""


def test_env_key_fills_empty_provider_api_key(tmp_path, monkeypatch):
    body = PROVIDERS_YAML.replace('api_key: "k1"', 'api_key: ""')
    monkeypatch.setenv("ASR2CLASS_LLM_API_KEY", "sk-env")
    cfg = load_config(_write_config(tmp_path, body))
    cloud = next(p for p in cfg.providers if p.id == "cloud-a")
    assert cloud.api_key == "sk-env"
    local = next(p for p in cfg.providers if p.id == "local-tr")
    assert local.api_key == "EMPTY"  # 已有 key 不被覆盖


def test_capability_filtering(tmp_path):
    cfg = load_config(_write_config(tmp_path, PROVIDERS_YAML))
    assert [p.id for p in filter_for_slot(cfg.providers, "translation")] == ["cloud-a", "local-tr"]
    # 翻译专用模型（无 grammar 能力）选不进语法槽
    assert [p.id for p in filter_for_slot(cfg.providers, "grammar")] == ["cloud-a"]


def test_slot_error_messages(tmp_path):
    cfg = load_config(_write_config(tmp_path, PROVIDERS_YAML))
    local = next(p for p in cfg.providers if p.id == "local-tr")
    assert slot_error(local, "grammar") is not None
    assert slot_error(local, "translation") is None
    assert slot_error(None, "translation") is not None


def test_resolve_slots_and_profile_override(tmp_path):
    cfg = load_config(_write_config(tmp_path, PROVIDERS_YAML))
    registry = ProviderRegistry.from_config(cfg)
    res = resolve_slots(cfg.slots, {}, registry)
    assert res.provider_id("translation") == "local-tr"
    assert res.provider_id("grammar") == "cloud-a"
    # 档案覆盖优先于全局
    res2 = resolve_slots(cfg.slots, {"translation": "cloud-a"}, registry)
    assert res2.provider_id("translation") == "cloud-a"
    assert res2.provider_id("grammar") == "cloud-a"  # 未覆盖项沿用全局
    # 能力不符 / 未知 id → None + 原因
    res3 = resolve_slots(cfg.slots, {"grammar": "local-tr"}, registry)
    assert res3.provider_id("grammar") is None
    assert "grammar" in res3.errors["grammar"]
    res4 = resolve_slots(cfg.slots, {"translation": "no-such"}, registry)
    assert res4.provider_id("translation") is None
    assert "不存在" in res4.errors["translation"]


def test_model_slots_getter_and_status(tmp_path):
    cfg = load_config(_write_config(tmp_path, PROVIDERS_YAML))
    registry = ProviderRegistry.from_config(cfg)
    slots = ModelSlots(registry, resolve_slots(cfg.slots, {}, registry))
    assert slots.translation_provider() is not None
    assert slots.grammar_provider() is not None
    lines = slots.status_lines()
    assert any("翻译槽" in ln and "local-tr" in ln for ln in lines)
    # 未绑定时 getter 返回 None 且状态行有提示
    empty = ModelSlots(registry, resolve_slots(type(cfg.slots)(), {}, registry))
    assert empty.translation_provider() is None
    assert any("未绑定" in ln for ln in empty.status_lines())


def test_no_hardcoded_provider_names():
    """验收：src/ 代码中不得出现具体模型名/服务商名（配置示例除外）。"""
    import re

    src = Path(__file__).resolve().parent.parent / "src"
    # 具体模型/服务商标识（示例配置里的名字），代码中一律不得出现
    banned = re.compile(
        r"index-translate|deepseek|qwen|sakura|gpt-|claude-|llama-\d|"
        r"api\.openai\.com|api\.deepseek\.com",
        re.IGNORECASE,
    )
    offenders = []
    for f in src.rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if banned.search(line):
                offenders.append(f"{f.name}:{i}: {line.strip()}")
    assert not offenders, "代码中出现硬编码模型/服务商名：\n" + "\n".join(offenders)
