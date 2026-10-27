"""语法分析的结构化文本（Markdown）导出。

由语言档案的 grammar_card 区块声明驱动（tree/cards/table 与字段标签），
语言相关字段不写死；供分析笔记窗口预览与「导出」共用同一渲染。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..profiles.loader import LanguageProfile
from .document import Segment, SessionDocument

_MISSING = object()

_DEFAULT_SECTIONS = {
    "structure": ("句子结构", "tree"),
    "grammar_points": ("语法点", "cards"),
    "vocabulary": ("重点词汇", "table"),
}


def _has_content(value: Any) -> bool:
    return value is not _MISSING and value is not None and value != "" and value != [] and value != {}


def section_declarations(profile: LanguageProfile) -> list[dict[str, Any]]:
    """档案声明的语法卡区块（Qt 无关，与 GrammarCardPopup 同语义）。"""
    raw = getattr(profile, "raw", None) or {}
    card_config = raw.get("grammar_card") or {}
    sections = card_config.get("sections")
    if not isinstance(sections, list) or not sections:
        sections = list(_DEFAULT_SECTIONS)

    declarations: list[dict[str, Any]] = []
    for item in sections:
        if isinstance(item, str):
            item = {"key": item}
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key") or "")
        if not key:
            continue
        default_title, default_renderer = _DEFAULT_SECTIONS.get(key, (key, "tree"))
        declaration = dict(item)
        declaration["key"] = key
        declaration["title"] = str(item.get("title") or default_title)
        declaration["renderer"] = str(item.get("renderer") or default_renderer)
        declarations.append(declaration)
    return declarations


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value)


def _tree_md(value: Any, indent: int = 0) -> list[str]:
    """任意嵌套 dict/list → markdown 无序列表。"""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(child, (Mapping, list, tuple)):
                lines.append(f"{pad}- **{key}**")
                lines.extend(_tree_md(child, indent + 1))
            else:
                lines.append(f"{pad}- **{key}**：{_fmt(child)}")
    elif isinstance(value, (list, tuple)):
        for child in value:
            if isinstance(child, (Mapping, list, tuple)):
                lines.extend(_tree_md(child, indent))
            else:
                lines.append(f"{pad}- {_fmt(child)}")
    else:
        lines.append(f"{pad}- {_fmt(value)}")
    return lines


def _cards_md(value: Any, declaration: Mapping[str, Any]) -> list[str]:
    items = value if isinstance(value, (list, tuple)) else [value]
    title_field = str(declaration.get("title_field") or "")
    badge = declaration.get("badge") or {}
    badge_field = str(badge.get("field") or "") if isinstance(badge, Mapping) else ""
    body_fields = declaration.get("body_fields")

    lines: list[str] = []
    for index, item in enumerate(items, 1):
        item_map = item if isinstance(item, Mapping) else {}
        title = _fmt(item_map.get(title_field)) if title_field else ""
        if not title:
            title = _fmt(item) if not item_map else f"{index}."
        badge_value = item_map.get(badge_field) if badge_field else None
        badge_text = f" `{badge_value}`" if badge_value not in (None, "") else ""
        lines.append(f"#### {title}{badge_text}")
        if isinstance(body_fields, list):
            for entry in body_fields:
                if isinstance(entry, str):
                    key, label = entry, entry
                elif isinstance(entry, Mapping):
                    key = str(entry.get("key") or "")
                    label = str(entry.get("label") or key)
                else:
                    continue
                if key:
                    lines.append(f"- {label}：{_fmt(item_map.get(key, ''))}")
        else:
            for key, val in item_map.items():
                if str(key) in (title_field, badge_field):
                    continue
                lines.append(f"- {key}：{_fmt(val)}")
        lines.append("")
    return lines


def _table_md(value: Any, declaration: Mapping[str, Any]) -> list[str]:
    rows = list(value) if isinstance(value, (list, tuple)) else [value]
    columns: list[tuple[str, str]] = []
    declared = declaration.get("columns")
    if isinstance(declared, list):
        for entry in declared:
            if isinstance(entry, str):
                columns.append((entry, entry))
            elif isinstance(entry, Mapping):
                key = str(entry.get("key") or "")
                if key:
                    columns.append((key, str(entry.get("label") or key)))
    if not columns:
        for row in rows:
            if isinstance(row, Mapping):
                for key in row:
                    if str(key) not in [c[0] for c in columns]:
                        columns.append((str(key), str(key)))
    if not columns:
        return []

    lines = ["| " + " | ".join(label for _, label in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        row_map = row if isinstance(row, Mapping) else {}
        cells = [_fmt(row_map.get(key, "")) for key, _ in columns]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def segment_to_markdown(segment: Segment, profile: LanguageProfile) -> str:
    """单句分析 → markdown 片段。"""
    lines = [f"### [{_fmt_time(segment.start)}] {segment.source}", ""]
    if segment.translation:
        lines += [f"> {segment.translation}", ""]
    payload = segment.analysis_payload()
    extra = payload.get("extra") or {}
    declared_keys = set()
    for declaration in section_declarations(profile):
        key = declaration["key"]
        declared_keys.add(key)
        value = payload.get(key, extra.get(key, _MISSING))
        if not _has_content(value):
            continue
        lines += [f"#### {declaration['title']}", ""]
        renderer = declaration["renderer"]
        if renderer == "cards":
            lines += _cards_md(value, declaration)
        elif renderer == "table":
            lines += _table_md(value, declaration)
        else:
            lines += _tree_md(value) + [""]
    # 未声明的 extra 自定义键：通用树兜底，不丢任何分析字段
    rest = {k: v for k, v in extra.items() if k not in declared_keys and _has_content(v)}
    if rest:
        lines += ["#### 其他", ""] + _tree_md(rest) + [""]
    return "\n".join(lines).rstrip() + "\n"


def document_to_markdown(document: SessionDocument, profile: LanguageProfile) -> str:
    """全部已分析句（按识别顺序）→ 完整 markdown 文档。"""
    analyzed = [s for s in document.segments if s.has_analysis]
    header = [
        f"# 语法分析笔记（{document.source_lang} → {document.target_lang}）",
        "",
        f"共 {len(analyzed)} 句已分析",
        "",
        "---",
        "",
    ]
    body = [segment_to_markdown(s, profile) for s in analyzed]
    return "\n".join(header) + "\n---\n\n".join(body)


def export_markdown(
    document: SessionDocument, profile: LanguageProfile, path: str | Path
) -> Path:
    """写入 markdown 文件（原子：临时文件 + 替换）。"""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(document_to_markdown(document, profile), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _fmt_time(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
