"""滚动记忆（FR-C3）：实体登记表 + 风格摘要。

- 新实体由 LLM 在翻译响应中附带登记（glossary_updates），本地合并去重，同 term 后登记覆盖先登记
- 风格摘要由 LLM 推断微调（style_updates 非空则整体替换），受 style_max_chars 限制
- 容量控制：登记表上限 glossary_max_entries，按 LRU（last_used_block）淘汰
- 持久化：memory.json（FR-C3.6），随会话保存/恢复；保存为原子替换
- 语言档案 style_preset 非空时：预设值优先并锁定，LLM 的 style_updates 被忽略（§8 风险对策）

M3 阶段 6（记忆编辑）：
- 全部读写/prompt 快照/保存共用一把可重入锁，GUI 与翻译线程可并发操作
- load(strict=True) 做结构校验：畸形 JSON/空字段/重复术语/语言档案不匹配都报错
- 加载后重建不变量：容量淘汰、风格长度截断、block_counter 至少恢复到最大 last_used_block
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


class MemoryValidationError(ValueError):
    """memory.json 校验失败（带条目位置信息）。"""


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
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        if self.style_preset and not self.style_notes:
            self.style_notes = self.style_preset[: self.style_max_chars]

    # ---- 合并更新（FR-C3.4）----

    def begin_block(self) -> int:
        with self._lock:
            self.block_counter += 1
            return self.block_counter

    def merge_updates(
        self,
        glossary_updates: list[dict] | None,
        style_updates: str | None,
    ) -> None:
        with self._lock:
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
        with self._lock:
            for entry in self.glossary.values():
                if entry.term in text:
                    entry.last_used_block = self.block_counter

    def prompt_snapshot(self) -> tuple[str, str]:
        """一次取走 (glossary_prompt, style_prompt)，两者保证来自同一时刻。"""
        with self._lock:
            return self.glossary_prompt(), self.style_prompt()

    def glossary_prompt(self) -> str:
        with self._lock:
            if not self.glossary:
                return ""
            lines = [f"- {e.term} → {e.translation}" + (f"（{e.type}）" if e.type else "")
                     for e in self.glossary.values()]
            return "\n".join(lines)

    def glossary_inline(self) -> str:
        """instTrans 硬性要求用的一行式术语对照：A→B、C→D。"""
        with self._lock:
            return "、".join(f"{e.term}→{e.translation}" for e in self.glossary.values())

    def style_prompt(self) -> str:
        with self._lock:
            return self.style_notes

    # ---- 手动编辑接口（FR-C3.7 / M3 阶段 6）----

    def set_term(self, term: str, translation: str, type: str = "") -> None:
        term = term.strip()
        translation = translation.strip()
        if not term or not translation:
            raise MemoryValidationError("术语和译文都不能为空")
        with self._lock:
            self.glossary[term] = GlossaryEntry(
                term=term, translation=translation, type=type.strip(),
                last_used_block=self.block_counter,
            )
            self._evict()

    def remove_term(self, term: str) -> bool:
        with self._lock:
            return self.glossary.pop(term, None) is not None

    def set_style(self, notes: str) -> None:
        """人工编辑风格；style_preset 锁定时拒绝（§4.6：不暗中覆盖档案预设）。"""
        if self.style_preset:
            raise MemoryValidationError("当前语言档案已锁定风格预设，不能人工修改")
        with self._lock:
            self.style_notes = notes.strip()[: self.style_max_chars]

    @property
    def style_locked(self) -> bool:
        return bool(self.style_preset)

    def clear(self) -> None:
        with self._lock:
            self.glossary.clear()
            self.style_notes = self.style_preset  # 回到档案预设（无预设则清空）

    def glossary_snapshot(self) -> list[GlossaryEntry]:
        """GUI 展示用的条目副本列表。"""
        with self._lock:
            return [GlossaryEntry(**vars(e)) for e in self.glossary.values()]

    # ---- 持久化（FR-C3.6）----

    def to_dict(self) -> dict:
        with self._lock:
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
    def from_dict(cls, data: dict, *, strict: bool = False, **kwargs) -> "RollingMemory":
        mem = cls(**kwargs)
        glossary = data.get("glossary") or []
        if strict and not isinstance(glossary, list):
            raise MemoryValidationError("glossary 必须是数组")
        seen: set[str] = set()
        for i, d in enumerate(glossary):
            if not isinstance(d, dict):
                raise MemoryValidationError(f"glossary 第 {i + 1} 条不是对象")
            term = str(d.get("term") or "").strip()
            translation = str(d.get("translation") or "").strip()
            if strict:
                if not term:
                    raise MemoryValidationError(f"glossary 第 {i + 1} 条 term 为空")
                if not translation:
                    raise MemoryValidationError(f"glossary 第 {i + 1} 条（{term}）translation 为空")
                if term in seen:
                    raise MemoryValidationError(f"glossary 术语重复：{term}")
                if "last_used_block" in d and int(d.get("last_used_block") or 0) < 0:
                    raise MemoryValidationError(f"glossary 第 {i + 1} 条 last_used_block 非法")
            if not term:
                continue  # 宽松模式跳过空条目
            seen.add(term)
            mem.glossary[term] = GlossaryEntry(
                term=term,
                translation=translation,
                type=str(d.get("type") or ""),
                last_used_block=max(0, int(d.get("last_used_block", 0) or 0)),
            )
        mem.style_notes = str(data.get("style_notes") or mem.style_notes)
        # 不变量重建：容量淘汰、风格长度截断、LRU 计数恢复
        mem._evict()
        mem.style_notes = mem.style_notes[: mem.style_max_chars]
        mem.block_counter = max(
            (e.last_used_block for e in mem.glossary.values()), default=0
        )
        return mem

    def save(self, path: Path, language_profile: str = "") -> None:
        """原子写回（同目录临时文件 + 替换），可用于现有会话的 memory.json。"""
        payload = {"language_profile": language_profile, **self.to_dict()}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        strict: bool = False,
        expected_profile: str = "",
        **kwargs,
    ) -> "RollingMemory":
        """加载 memory.json；strict 时结构/档案不匹配都抛 MemoryValidationError。"""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise MemoryValidationError(f"JSON 格式错误：第 {e.lineno} 行：{e.msg}") from e
        if not isinstance(data, dict):
            raise MemoryValidationError("memory.json 根节点必须是对象")
        file_profile = str(data.get("language_profile") or "")
        if strict and expected_profile and file_profile and file_profile != expected_profile:
            raise MemoryValidationError(
                f"记忆属于语言档案 {file_profile}，不能载入当前 {expected_profile} 会话"
            )
        return cls.from_dict(data, strict=strict, **kwargs)
