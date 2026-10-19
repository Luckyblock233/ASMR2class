"""nlp.providers：模型提供方可插拔架构（Provider Registry + 统一接口）。"""

from .base import ProviderBase, ProviderUnavailableError
from .capabilities import (
    CAP_GLOSSARY_CONSTRAINT,
    CAP_GRAMMAR,
    CAP_TRANSLATION,
    SLOT_GRAMMAR,
    SLOT_NAMES,
    SLOT_TRANSLATION,
    filter_for_slot,
    slot_error,
    supports,
)
from .registry import ProviderRegistry, create_provider

__all__ = [
    "ProviderBase",
    "ProviderUnavailableError",
    "ProviderRegistry",
    "create_provider",
    "CAP_TRANSLATION",
    "CAP_GRAMMAR",
    "CAP_GLOSSARY_CONSTRAINT",
    "SLOT_TRANSLATION",
    "SLOT_GRAMMAR",
    "SLOT_NAMES",
    "supports",
    "slot_error",
    "filter_for_slot",
]
