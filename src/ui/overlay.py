"""桌面悬浮字幕窗：现有窗口壳＋M3 结构化句块与语法卡。"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

from ..subtitle.document import Segment

log = logging.getLogger(__name__)

DISPLAY_ORDERS = ("source_first", "translation_first", "source_only", "translation_only")


@dataclass
class OverlayLine:
    start: float
    end: float
    source: str
    translation: str | None = None


class SubtitleBuffer:
    """保留给既有调用方的纯文本数据层；新窗口使用 SubtitleView。"""

    def __init__(self, display_order: str = "source_first", capacity: int = 50):
        self.display_order = display_order if display_order in DISPLAY_ORDERS else "source_first"
        self.lines: deque[OverlayLine] = deque(maxlen=capacity)

    def add(self, line: OverlayLine) -> None:
        self.lines.append(line)

    def render(self) -> str:
        out = []
        for ln in self.lines:
            src, tgt = ln.source, ln.translation or ""
            if self.display_order == "source_only":
                rows = [src]
            elif self.display_order == "translation_only":
                rows = [tgt] if tgt else [src]
            elif self.display_order == "translation_first":
                rows = [tgt, src] if tgt else [src]
            else:
                rows = [src, tgt] if tgt else [src]
            out.append("\n".join(rows))
        return "\n\n".join(out)


class OverlayUnavailableError(RuntimeError):
    pass


_APP_REF = None  # 保持 QApplication 引用，防止被 GC 后 Qt 崩溃


def create_overlay(display_order: str = "source_first", opacity: float = 0.85, profile=None, read_only=False):
    """创建悬浮窗（延迟导入 PyQt6，确保 QApplication 已存在）。"""
    try:
        from PyQt6.QtCore import Qt, pyqtSignal
        from PyQt6.QtGui import QAction, QIcon
        from PyQt6.QtWidgets import (
            QHBoxLayout,
            QPushButton,
            QSystemTrayIcon,
            QMenu,
            QVBoxLayout,
            QWidget,
            QStyle,
            QApplication,
        )
        from .grammar_card import GrammarCardPopup
        from .subtitle_view import SubtitleView
    except ImportError as e:
        raise OverlayUnavailableError(f"PyQt6 不可用：{e}") from e

    import sys as _sys

    if QApplication.instance() is None:
        global _APP_REF
        _APP_REF = QApplication(_sys.argv)

    class OverlayWindow(QWidget):
        add_sentence = pyqtSignal(float, float, str, object)  # 旧接口兼容
        upsert_segment = pyqtSignal(object)
        analysis_ready = pyqtSignal(object)
        analysis_requested = pyqtSignal(int)
        set_status = pyqtSignal(str)

        def __init__(self):
            super().__init__()
            self.profile = profile
            self._drag_pos = None
            self._click_through = False
            self._opacity = max(0.2, min(1.0, opacity))
            self._legacy_id = 0
            self.grammar_card = None

            self.setWindowTitle("ASR2class 字幕")
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.resize(720, 480)
            self.setMinimumSize(400, 240)
            self.setWindowOpacity(self._opacity)

            self.subtitle_view = SubtitleView(
                display_order=display_order,
                source_lang=profile.source_lang if profile is not None else "",
                read_only=read_only,
                parent=self,
            )
            self.text = self.subtitle_view  # 旧属性名保留
            # 主程序只在语法槽成功绑定后开启手动分析，避免发出无人接收的请求。
            self.subtitle_view.set_grammar_enabled(False)
            self.subtitle_view.sentence_activated.connect(self._show_grammar)
            self.subtitle_view.analysis_requested.connect(self.analysis_requested.emit)

            bar = QHBoxLayout()
            self.status_label = QPushButton("")
            self.status_label.setFlat(True)
            self.status_label.setStyleSheet("color: #9f9; font-size: 10px;")
            btn_style = "QPushButton { font-size: 11px; padding: 2px 4px; }"
            self.btn_less = QPushButton("透明+")
            self.btn_more = QPushButton("透明-")
            self.btn_through = QPushButton("穿透")
            self.btn_close = QPushButton("×")
            for b in (self.btn_less, self.btn_more, self.btn_through, self.btn_close):
                b.setStyleSheet(btn_style)
            self.btn_less.clicked.connect(lambda: self._adjust_opacity(-0.1))
            self.btn_more.clicked.connect(lambda: self._adjust_opacity(0.1))
            self.btn_through.clicked.connect(lambda: self.set_click_through(True))
            self.btn_close.clicked.connect(self.close)
            bar.addWidget(self.status_label)
            bar.addStretch()
            bar.addWidget(self.btn_less)
            bar.addWidget(self.btn_more)
            bar.addWidget(self.btn_through)
            bar.addWidget(self.btn_close)

            layout = QVBoxLayout()
            layout.setContentsMargins(4, 4, 4, 4)
            layout.addLayout(bar)
            layout.addWidget(self.subtitle_view)
            self.setLayout(layout)

            style = QApplication.style()
            icon = (
                style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
                if style is not None
                else QIcon()
            )
            self.tray = QSystemTrayIcon(icon, self)
            menu = QMenu(self)
            self.tray_menu = menu
            act_through = QAction("关闭点击穿透", self)
            act_through.triggered.connect(lambda: self.set_click_through(False))
            act_quit = QAction("退出", self)
            act_quit.triggered.connect(self.close)
            menu.addAction(act_through)
            menu.addAction(act_quit)
            self.tray.setContextMenu(menu)
            self.tray.setToolTip("ASR2class 字幕悬浮窗")
            self.tray.show()

            self.add_sentence.connect(self._on_legacy_sentence)
            self.upsert_segment.connect(self.subtitle_view.upsert_segment)
            self.analysis_ready.connect(self.subtitle_view.upsert_segment)
            self.set_status.connect(self._on_status)

        def _on_legacy_sentence(self, start: float, end: float, source: str, translation) -> None:
            self._legacy_id += 1
            self.subtitle_view.upsert_segment(
                Segment(start, end, source, translation, id=self._legacy_id)
            )

        def _show_grammar(self, segment: Segment) -> None:
            if self.profile is None or not segment.has_analysis:
                return
            if self.grammar_card is None:
                self.grammar_card = GrammarCardPopup()
            self.grammar_card.show_segment(segment, self.profile, anchor=self.subtitle_view)

        def set_grammar_enabled(self, enabled: bool) -> None:
            self.subtitle_view.set_grammar_enabled(enabled)

        def _on_status(self, status: str) -> None:
            online = status == "ONLINE"
            self.status_label.setText("● " + status)
            self.status_label.setStyleSheet(
                f"color: {'#9f9' if online else '#f66'}; font-size: 10px;"
            )

        def _adjust_opacity(self, delta: float) -> None:
            self._opacity = max(0.2, min(1.0, self._opacity + delta))
            self.setWindowOpacity(self._opacity)

        def set_click_through(self, on: bool) -> None:
            self._click_through = on
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, on)
            for b in (self.btn_less, self.btn_more, self.btn_through, self.btn_close):
                b.setVisible(not on)
            self.show()
            self.tray.showMessage(
                "ASR2class",
                "点击穿透已开启，右键托盘图标恢复" if on else "点击穿透已关闭",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()

        def mouseMoveEvent(self, event):
            if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self.move(event.globalPosition().toPoint() - self._drag_pos)
                event.accept()

        def mouseReleaseEvent(self, event):
            self._drag_pos = None

        def closeEvent(self, event):
            if self.grammar_card is not None:
                self.grammar_card.close()
            self.tray.hide()
            QApplication.quit()
            super().closeEvent(event)

    return OverlayWindow()
