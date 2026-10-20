"""语言档案加载器（FR-4.6）：扫描 config/languages/*.yaml 动态加载。

新增一门学习语言 = 新增一个 YAML 档案 + 对应 prompt 模板目录，零代码改动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LanguageProfile:
    name: str  # 文件名（不含扩展名），如 ja
    language: str
    display_name: str
    source_lang: str
    target_lang: str
    language_hint: str | None = None  # ASR 显式语种；None = auto
    llm_override: dict[str, Any] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)  # 相对 config/prompts/ 的路径
    subtitle: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)  # context 配置覆盖（FR-C1.2）
    slots: dict[str, Any] = field(default_factory=dict)  # 槽位绑定覆盖（档案优先于全局 slots）
    appearance: dict[str, Any] = field(default_factory=dict)  # 外观覆盖（合并到全局 ui 配置）
    style_preset: str = ""  # 档案预设风格，非空则优先于 LLM 推断并锁定（§8 风险对策）
    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def prompt_path(self, kind: str) -> str | None:
        return self.prompts.get(kind)


def _parse(path: Path) -> LanguageProfile:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    pair = raw.get("language_pair") or {}
    asr = raw.get("asr") or {}
    return LanguageProfile(
        name=path.stem,
        language=str(raw.get("language") or path.stem),
        display_name=str(raw.get("display_name") or path.stem),
        source_lang=str(pair.get("source") or ""),
        target_lang=str(pair.get("target") or ""),
        language_hint=asr.get("language_hint") or None,
        llm_override=raw.get("llm") or {},
        prompts=raw.get("prompts") or {},
        subtitle=raw.get("subtitle") or {},
        context=raw.get("context") or {},
        slots=raw.get("slots") or {},
        appearance=raw.get("appearance") or {},
        style_preset=str(raw.get("style_preset") or ""),
        raw=raw,
        path=path,
    )


def load_profiles(languages_dir: Path) -> dict[str, LanguageProfile]:
    """扫描目录加载全部语言档案，key 为文件名 stem。"""
    profiles: dict[str, LanguageProfile] = {}
    if not languages_dir.is_dir():
        return profiles
    for path in sorted(languages_dir.glob("*.yaml")):
        p = _parse(path)
        profiles[p.name] = p
    return profiles


def get_profile(languages_dir: Path, name: str) -> LanguageProfile:
    profiles = load_profiles(languages_dir)
    if name not in profiles:
        available = ", ".join(profiles) or "(无)"
        raise KeyError(f"语言档案 '{name}' 不存在，可用档案：{available}")
    return profiles[name]
