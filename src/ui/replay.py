"""无媒体依赖的会话重载预览；完整音视频播放器在后续 M3 阶段实现。"""

from __future__ import annotations

from ..profiles.loader import LanguageProfile
from ..subtitle.document import SessionDocument


def run_session_preview(document: SessionDocument, profile: LanguageProfile, opacity: float = 0.85) -> int:
    """显示已保存句段和语法卡，不初始化 ASR 或任何 Provider。"""
    from PyQt6.QtWidgets import QApplication

    from .overlay import create_overlay

    overlay = create_overlay(
        display_order=profile.subtitle.get("display_order", "source_first"),
        opacity=opacity,
        profile=profile,
        read_only=True,
    )
    for segment in document.segments:
        overlay.upsert_segment.emit(segment)
    overlay.setWindowTitle(f"ASR2class 会话重载 - {profile.display_name}")
    overlay.show()
    app = QApplication.instance()
    if app is None:  # create_overlay 正常情况下已保证实例存在
        raise RuntimeError("QApplication 未初始化")
    return app.exec()
