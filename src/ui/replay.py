"""会话重载与媒体回放（M3 阶段 4）。

- 无媒体：只用字幕悬浮窗预览句块/语法卡（既有行为）。
- 有媒体（--media 或界面「打开媒体」）：ReplayWindow 负责媒体与播放控制，
  悬浮窗负责句块/语法卡；本模块作为 replay controller 连接二者：
  positionChanged → segment_at 高亮；句块单击 → seek；双击 → 语法卡。
"""

from __future__ import annotations

from pathlib import Path

from ..profiles.loader import LanguageProfile
from ..subtitle.document import SessionDocument


def run_session_preview(
    document: SessionDocument,
    profile: LanguageProfile,
    opacity: float = 0.85,
    media_path: str | None = None,
    ui=None,
) -> int:
    """重载已保存会话，不初始化 ASR 或任何 Provider。"""
    from PyQt6.QtWidgets import QApplication, QFileDialog

    from .i18n import tr
    from .notebook import create_notebook_window
    from .overlay import create_overlay
    from .player import create_replay_window

    overlay = create_overlay(
        display_order=profile.subtitle.get("display_order", "source_first"),
        opacity=opacity,
        profile=profile,
        read_only=True,
        replay_mode=True,
        ui=ui,
        notebook_factory=lambda: create_notebook_window(lambda: document, lambda: profile),
    )
    for segment in document.segments:
        overlay.upsert_segment.emit(segment)
    overlay.setWindowTitle(f"ASR2class 会话重载 - {profile.display_name}")

    # ---- replay controller：连接字幕窗与媒体窗 ----
    player_box = {"window": None}

    def attach_media(path: str) -> None:
        if player_box["window"] is None:
            window = create_replay_window(
                document,
                on_seek_highlight=overlay.subtitle_view.set_current_segment,
            )
            window.show()
            player_box["window"] = window
        player_box["window"].load_media(path)

    # 单击句块 → seek（回放单击语义，§4.1）
    overlay.seek_requested.connect(
        lambda sid: player_box["window"] is not None
        and player_box["window"].seek_to_segment(sid)
    )
    # 双击有分析句块 → 语法卡（独立手势）
    overlay.grammar_requested.connect(
        lambda sid: overlay.show_grammar(document.get(sid))
    )

    def open_media_dialog() -> None:
        path, _ = QFileDialog.getOpenFileName(
            overlay, tr("replay.open_media"), "",
            "媒体文件 (*.mp4 *.mkv *.mp3 *.wav *.m4a *.flac);;所有文件 (*)",
        )
        if path:
            attach_media(path)

    overlay.add_menu_action(tr("replay.open_media") + "…", open_media_dialog)
    if media_path:
        attach_media(media_path)

    overlay.show()
    app = QApplication.instance()
    if app is None:  # create_overlay 正常情况下已保证实例存在
        raise RuntimeError("QApplication 未初始化")
    return app.exec()


def document_from_srt(srt_path: str | Path) -> SessionDocument:
    """外部 SRT 重载为会话文档（完整块读取，保留多行原文）。"""
    from ..subtitle.parser import load_srt

    document = SessionDocument()
    for entry in load_srt(srt_path, external=True):
        document.add(entry.start, entry.end, entry.source, entry.translation)
    return document
