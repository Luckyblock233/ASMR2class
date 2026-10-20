"""UI 文案资源加载层（M3 阶段 3：中/英界面文案）。

文案集中在 config/locales/<lang>.yaml（扁平 key → 文本），代码统一用 tr(key)。
缺失 key 安全回退：当前语言 → zh_CN → key 本身（不会崩溃、不会出现空白）。
语言档案自己的语法卡字段标题仍由档案决定，不经本层。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "zh_CN"

_lock = threading.Lock()
_current = DEFAULT_LANGUAGE
_bundles: dict[str, dict[str, str]] = {}
_locales_dir: Path | None = None


def _load_bundle(locales_dir: Path, language: str) -> dict[str, str]:
    path = locales_dir / f"{language}.yaml"
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001 —— 文案文件损坏不应 crash UI
        log.warning("文案文件 %s 加载失败：%s", path, e)
        return {}
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def set_language(language: str, locales_dir: Path | None = None) -> None:
    """切换 UI 语言；locales_dir 首次必须给出（之后缓存复用）。"""
    global _current, _locales_dir
    with _lock:
        if locales_dir is not None:
            _locales_dir = locales_dir
        _current = language or DEFAULT_LANGUAGE
        if _locales_dir is not None:
            for lang in {_current, DEFAULT_LANGUAGE}:
                if lang not in _bundles:
                    _bundles[lang] = _load_bundle(_locales_dir, lang)


def current_language() -> str:
    with _lock:
        return _current


def tr(key: str) -> str:
    """取文案；缺失时回退 zh_CN，再回退 key 本身。"""
    with _lock:
        for lang in (_current, DEFAULT_LANGUAGE):
            text = _bundles.get(lang, {}).get(key)
            if text:
                return text
    return key


def reload() -> None:
    """重新加载文案文件（开发期改 yaml 后热更新）。"""
    with _lock:
        langs = list(_bundles)
        _bundles.clear()
        if _locales_dir is not None:
            for lang in langs:
                _bundles[lang] = _load_bundle(_locales_dir, lang)


def _reset_for_testing() -> None:
    global _current, _locales_dir
    with _lock:
        _current = DEFAULT_LANGUAGE
        _bundles.clear()
        _locales_dir = None
