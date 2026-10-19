"""滚动记忆（FR-C3）：实体登记表 + 风格摘要。

- 新实体由 LLM 在翻译响应中附带登记（glossary_updates），本地合并去重，同 term 后登记覆盖先登记
- 风格摘要由 LLM 推断微调（style_updates 非空则整体替换），受 style_max_chars 限制
- 容量控制：登记表上限 glossary_max_entries，按 LRU（last_used_block）淘汰
- 持久化：memory.json（FR-C3.6），随会话保存/恢复
- 语言档案 style_preset 非空时：预设值优先并锁定，LLM 的 style_updates 被忽略（§8 风险对策）
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class GlossaryEntry:
    term: str
    translation: str
    type: str = ""
    last_used_block: int = 0


@dataclass
class RollingMemory:
    glossary_max_entries: int = 100
    style_max_chars: int = 200
    style_preset: str = ""  # 档案预设风格；非空则锁定
    glossary: dict[str, GlossaryEntry] = field(default_factory=dict)
    style_notes: str = ""
    block_counter: int = 0

    def __post_init__(self):
        if self.style_preset and not self.style_notes:
            self.style_notes = self.style_preset[: self.style_max_chars]

    # ---- 合并更新（FR-C3.4）----

    def begin_block(self) -> int:
        self.block_counter += 1
        return self.block_counter

    def merge_updates(
        self,
        glossary_updates: list[dict] | None,
        style_updates: str | None,
    ) -> None:
        for u in glossary_updates or []:
            term = str(u.get("term") or "").strip()
            trans = str(u.get("translation") or "").strip()
            if not term or not trans:
                continue
            self.glossary[term] = GlossaryEntry(  # 同 term 后登记覆盖
                term=term,
                translation=trans,
                type=str(u.get("type") or ""),
                last_used_block=self.block_counter,
            )
        self._evict()
        if style_updates and not self.style_preset:
            self.style_notes = str(style_updates).strip()[: self.style_max_chars]

    def _evict(self) -> None:
        while len(self.glossary) > self.glossary_max_entries:
            # LRU：淘汰 last_used_block 最小（最久未命中）的条目
            oldest = min(self.glossary.values(), key=lambda e: e.last_used_block)
            del self.glossary[oldest.term]
            log.debug("记忆淘汰：%s", oldest.term)

    # ---- prompt 注入（FR-C3.3）----

    def mark_hits(self, text: str) -> None:
        """文本中命中的条目刷新 last_used_block（LRU 的"近期命中"依据）。"""
        for entry in self.glossary.values():
            if entry.term in text:
                entry.last_used_block = self.block_counter

    def glossary_prompt(self) -> str:
        if not self.glossary:
            return ""
        lines = [f"- {e.term} → {e.translation}" + (f"（{e.type}）" if e.type else "")
                 for e in self.glossary.values()]
        return "\n".join(lines)

    def glossary_inline(self) -> str:
        """instTrans 硬性要求用的一行式术语对照：A→B、C→D。"""
        return "、".join(f"{e.term}→{e.translation}" for e in self.glossary.values())

    def style_prompt(self) -> str:
        return self.style_notes

    # ---- 手动编辑接口（FR-C3.7）----

    def set_term(self, term: str, translation: str, type: str = "") -> None:
        self.glossary[term] = GlossaryEntry(
            term=term, translation=translation, type=type,
            last_used_block=self.block_counter,
        )
        self._evict()

    def remove_term(self, term: str) -> bool:
        return self.glossary.pop(term, None) is not None

    def set_style(self, notes: str) -> None:
        self.style_notes = notes[: self.style_max_chars]

    def clear(self) -> None:
        self.glossary.clear()
        self.style_notes = self.style_preset  # 回到档案预设（无预设则清空）

    # ---- 持久化（FR-C3.6）----

    def to_dict(self) -> dict:
        return {
            "glossary": [
                {
                    "term": e.term,
                    "translation": e.translation,
                    "type": e.type,
                    "last_used_block": e.last_used_block,
                }
                for e in self.glossary.values()
            ],
            "style_notes": self.style_notes,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "RollingMemory":
        mem = cls(**kwargs)
        for d in data.get("glossary") or []:
            mem.glossary[d["term"]] = GlossaryEntry(
                term=d["term"],
                translation=d.get("translation", ""),
                type=d.get("type", ""),
                last_used_block=int(d.get("last_used_block", 0)),
            )
        mem.style_notes = str(data.get("style_notes") or mem.style_notes)
        return mem

    def save(self, path: Path, language_profile: str = "") -> None:
        payload = {"language_profile": language_profile, **self.to_dict()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, **kwargs) -> "RollingMemory":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, **kwargs)
