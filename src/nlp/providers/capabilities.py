"""Provider 能力标签与槽位过滤（模型可插拔架构 §1.1 能力声明）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ProviderConfig

CAP_TRANSLATION = "translation"            # 可绑翻译槽
CAP_GRAMMAR = "grammar"                    # 可绑语法槽（支持结构化 JSON 输出）
CAP_GLOSSARY_CONSTRAINT = "glossary_constraint"  # 术语表硬约束（instTrans 映射）

SLOT_TRANSLATION = "translation"
SLOT_GRAMMAR = "grammar"

# 槽位 → 必需能力标签
SLOT_REQUIRED_CAP = {
    SLOT_TRANSLATION: CAP_TRANSLATION,
    SLOT_GRAMMAR: CAP_GRAMMAR,
}

SLOT_NAMES = {SLOT_TRANSLATION: "翻译槽", SLOT_GRAMMAR: "语法槽"}


def supports(provider: ProviderConfig, capability: str) -> bool:
    return capability in provider.capabilities


def slot_error(provider: ProviderConfig | None, slot: str) -> str | None:
    """该 provider 能否绑到该槽位；返回 None = 可以，否则返回中文原因。"""
    name = SLOT_NAMES.get(slot, slot)
    if provider is None:
        return f"{name}未绑定"
    cap = SLOT_REQUIRED_CAP.get(slot)
    if cap and not supports(provider, cap):
        return f"Provider '{provider.id}' 不具备 {cap} 能力，无法绑到{name}"
    return None


def filter_for_slot(providers: list[ProviderConfig], slot: str) -> list[ProviderConfig]:
    """按槽位过滤可选 Provider（能力声明不匹配的被排除）。"""
    cap = SLOT_REQUIRED_CAP.get(slot)
    if not cap:
        return list(providers)
    return [p for p in providers if supports(p, cap)]
