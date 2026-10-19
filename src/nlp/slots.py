"""模型槽位：翻译槽/语法槽的绑定解析与热替换（模型可插拔架构 §1/§4）。

绑定解析优先级：语言档案 slots 覆盖 > 全局 slots。
槽位只持有 provider id，Provider 对象由 Registry 按 id 寻址——
rebind 只换映射，进行中的请求跑完自然结束，本地托管 Provider
被换绑出所有槽位后自动回收子进程。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from ..config import SlotsConfig
from .providers import (
    SLOT_GRAMMAR,
    SLOT_NAMES,
    SLOT_TRANSLATION,
    ProviderBase,
    ProviderRegistry,
    slot_error,
)

log = logging.getLogger(__name__)

ALL_SLOTS = (SLOT_TRANSLATION, SLOT_GRAMMAR)


@dataclass
class SlotResolution:
    """一次槽位绑定解析的结果：槽位 → (provider id 或 None, 不可用原因)。"""

    bindings: dict[str, str | None] = field(default_factory=dict)  # slot → provider id | None
    errors: dict[str, str] = field(default_factory=dict)           # slot → 不可用原因（可绑定时无条目）

    def provider_id(self, slot: str) -> str | None:
        return self.bindings.get(slot)


def resolve_slots(
    global_slots: SlotsConfig,
    profile_slots: dict | None,
    registry: ProviderRegistry,
) -> SlotResolution:
    """档案覆盖 > 全局；逐槽校验 id 存在与能力匹配。"""
    override = profile_slots or {}
    res = SlotResolution()
    for slot in ALL_SLOTS:
        pid = str(override.get(slot) or "") or (
            global_slots.translation if slot == SLOT_TRANSLATION else global_slots.grammar
        )
        pid = pid.strip()
        if not pid:
            res.bindings[slot] = None
            res.errors[slot] = f"{SLOT_NAMES.get(slot, slot)}未绑定"
            continue
        cfg = registry.config(pid)
        err = slot_error(cfg, slot) if cfg else f"Provider '{pid}' 不存在"
        if err:
            res.bindings[slot] = None
            res.errors[slot] = err
        else:
            res.bindings[slot] = pid
    return res


class ModelSlots:
    """槽位运行时引用：消费者每次通过 getter 取当前 Provider，支持热替换。"""

    def __init__(self, registry: ProviderRegistry, resolution: SlotResolution):
        self._registry = registry
        self._resolution = resolution
        self._lock = threading.Lock()

    @property
    def resolution(self) -> SlotResolution:
        return self._resolution

    def provider(self, slot: str) -> ProviderBase | None:
        """取槽位当前绑定的 Provider（未绑定/构造失败返回 None）。"""
        with self._lock:
            pid = self._resolution.bindings.get(slot)
        if not pid:
            return None
        return self._registry.get(pid)

    def translation_provider(self) -> ProviderBase | None:
        return self.provider(SLOT_TRANSLATION)

    def grammar_provider(self) -> ProviderBase | None:
        return self.provider(SLOT_GRAMMAR)

    def rebind(self, resolution: SlotResolution) -> None:
        """换绑：新请求立即走新 Provider；不再被引用的本地 Provider 回收。"""
        with self._lock:
            old = self._resolution
            self._resolution = resolution
        if old.bindings != resolution.bindings:
            log.info("槽位绑定变更：%s → %s", old.bindings, resolution.bindings)
        in_use = {pid for pid in resolution.bindings.values() if pid}
        self._registry.release_unused(in_use)

    def status_lines(self) -> list[str]:
        """启动时打印的两槽状态。"""
        lines: list[str] = []
        for slot in ALL_SLOTS:
            name = SLOT_NAMES.get(slot, slot)
            pid = self._resolution.bindings.get(slot)
            if pid:
                cfg = self._registry.config(pid)
                label = cfg.name if cfg else pid
                lines.append(f"{name}：{label}（{pid}）")
            else:
                reason = self._resolution.errors.get(slot, "未绑定")
                hint = "，对应功能停用" if slot == SLOT_GRAMMAR else "，将仅显示原文"
                lines.append(f"{name}：{reason}{hint}")
        return lines
