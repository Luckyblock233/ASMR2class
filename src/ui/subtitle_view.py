"""按稳定句段 ID 更新的字幕滚动视图。

本模块只负责 D 区句块及其语法分析交互；工作线程应通过 Qt 排队信号把
:class:`~src.subtitle.document.Segment` 快照交给 GUI 线程，不要直接操作控件。
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.subtitle.document import (
    ANALYSIS_FAILED,
    ANALYSIS_PENDING,
    ANALYSIS_READY,
    Segment,
)

DISPLAY_ORDERS = (
    "source_first",
    "translation_first",
    "source_only",
    "translation_only",
)


def _normalized_order(order: str) -> str:
    return order if order in DISPLAY_ORDERS else "source_first"


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class SentenceBlock(QFrame):
    """单个字幕句块；只有已完成且含分析数据的句块可激活。"""

    activated = pyqtSignal(object)
    analysis_requested = pyqtSignal(int)

    def __init__(
        self,
        segment: Segment,
        *,
        display_order: str = "source_first",
        source_lang: str = "",
        grammar_enabled: bool = True,
        read_only: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._segment = segment.snapshot()
        self._display_order = _normalized_order(display_order)
        self._source_lang = source_lang
        self._grammar_enabled = grammar_enabled
        self._read_only = read_only
        self._clickable = False

        self.setObjectName("sentenceBlock")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.setProperty("hovered", False)
        self.setStyleSheet(
            'SentenceBlock[hovered="true"] {'
            " background-color: rgba(255, 255, 255, 24); border-radius: 4px; }"
        )

        self.meta_label = QLabel(self)
        self.meta_label.setObjectName("sentenceMeta")
        self.meta_label.setStyleSheet("color: #808080; background: transparent;")

        self.status_label = QLabel(self)
        self.status_label.setObjectName("sentenceAnalysisStatus")
        self.status_label.setStyleSheet("color: #808080; background: transparent;")

        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.addWidget(self.meta_label)
        meta_row.addStretch(1)
        meta_row.addWidget(self.status_label)

        self.source_label = QLabel(self)
        self.source_label.setObjectName("sentenceSource")
        self.source_label.setWordWrap(True)
        self.source_label.setStyleSheet("color: #FFFFFF; background: transparent;")

        self.translation_label = QLabel(self)
        self.translation_label.setObjectName("sentenceTranslation")
        self.translation_label.setWordWrap(True)
        self.translation_label.setStyleSheet("color: #B8D8FF; background: transparent;")

        self.text_layout = QVBoxLayout()
        self.text_layout.setContentsMargins(0, 0, 0, 0)
        self.text_layout.setSpacing(2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)
        layout.addLayout(meta_row)
        layout.addLayout(self.text_layout)

        self.analysis_action = QAction("为此句生成语法分析", self)
        self.analysis_action.setObjectName("generateAnalysisAction")
        self.analysis_action.triggered.connect(self._request_analysis)

        self._refresh()

    @property
    def segment_id(self) -> int:
        return self._segment.id

    @property
    def is_clickable(self) -> bool:
        return self._clickable

    def set_segment(self, segment: Segment) -> None:
        """原位替换句段快照，不改变句块在滚动区中的位置。"""
        self._segment = segment.snapshot()
        self._refresh()

    def set_display_order(self, order: str) -> None:
        self._display_order = _normalized_order(order)
        self._refresh_text()

    def set_interaction(self, *, grammar_enabled: bool, read_only: bool) -> None:
        self._grammar_enabled = grammar_enabled
        self._read_only = read_only
        self._refresh_analysis_action()

    def _refresh(self) -> None:
        segment = self._segment
        language = f" [{self._source_lang}]" if self._source_lang else ""
        self.meta_label.setText(f"[{_format_timestamp(segment.start)}]{language}")
        self._refresh_text()
        self._refresh_status()
        self._refresh_analysis_action()

        self._clickable = segment.analysis_status == ANALYSIS_READY and segment.has_analysis
        self.setCursor(
            Qt.CursorShape.PointingHandCursor
            if self._clickable
            else Qt.CursorShape.ArrowCursor
        )
        if not self._clickable:
            self._set_hovered(False)

    def _refresh_text(self) -> None:
        source = self._segment.source or ""
        translation = self._segment.translation or ""
        has_translation = bool(translation)
        self.source_label.setText(source)
        # translation_only 在没有译文时也必须有可读内容。
        self.translation_label.setText(translation if has_translation else source)

        while self.text_layout.count():
            self.text_layout.takeAt(0)

        order = self._display_order
        self.source_label.setVisible(order != "translation_only")
        self.translation_label.setVisible(
            order not in ("source_only",)
            and (has_translation or order == "translation_only")
        )

        if order == "translation_only":
            widgets = [self.translation_label]
        elif order == "source_only":
            widgets = [self.source_label]
        elif order == "translation_first" and has_translation:
            widgets = [self.translation_label, self.source_label]
        elif order == "source_first" and has_translation:
            widgets = [self.source_label, self.translation_label]
        else:
            # 任一双语顺序在缺译文时都回退为单行原文。
            widgets = [self.source_label]

        for widget in widgets:
            self.text_layout.addWidget(widget)

    def _refresh_status(self) -> None:
        status = self._segment.analysis_status
        text = ""
        tooltip = ""
        color = "#808080"
        if status == ANALYSIS_PENDING:
            text = "语法分析中…"
            tooltip = "正在生成语法分析"
        elif status == ANALYSIS_READY:
            text = "语法分析可用" if self._segment.has_analysis else "语法分析已完成"
            tooltip = "单击查看语法分析" if self._segment.has_analysis else "语法分析已完成"
        elif status == ANALYSIS_FAILED:
            text = "语法分析失败"
            tooltip = self._segment.analysis_error or "语法分析失败，可右键重试"
            color = "#FF6B6B"

        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))
        self.status_label.setToolTip(tooltip)
        self.status_label.setStyleSheet(f"color: {color}; background: transparent;")
        self.setToolTip(tooltip)

    def _refresh_analysis_action(self) -> None:
        status = self._segment.analysis_status
        enabled = True
        hint = "为此句生成语法分析"
        if self._read_only:
            enabled = False
            hint = "只读模式下不能生成语法分析"
        elif not self._grammar_enabled:
            enabled = False
            hint = "请先在设置中绑定语法分析模型"
        elif status == ANALYSIS_PENDING:
            enabled = False
            hint = "语法分析正在进行，请稍候"
        elif status == ANALYSIS_READY:
            enabled = False
            hint = "该句已有语法分析" if self._segment.has_analysis else "该句语法分析已完成"
        elif status == ANALYSIS_FAILED:
            hint = "重新为此句生成语法分析"

        self.analysis_action.setEnabled(enabled)
        self.analysis_action.setToolTip(hint)
        self.analysis_action.setStatusTip(hint)

    def _request_analysis(self) -> None:
        if self.analysis_action.isEnabled():
            self.analysis_requested.emit(self._segment.id)

    def _show_context_menu(self, position) -> None:
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        menu.addAction(self.analysis_action)
        menu.exec(self.mapToGlobal(position))

    def _set_hovered(self, hovered: bool) -> None:
        hovered = hovered and self._clickable
        if self.property("hovered") == hovered:
            return
        self.setProperty("hovered", hovered)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def enterEvent(self, event) -> None:
        if self._clickable:
            self._set_hovered(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._set_hovered(False)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if (
            self._clickable
            and event.button() == Qt.MouseButton.LeftButton
            and self.rect().contains(event.position().toPoint())
        ):
            # 每次激活都发出新快照，接收方不能修改句块内部数据。
            self.activated.emit(self._segment.snapshot())
            event.accept()
            return
        super().mouseReleaseEvent(event)


class SubtitleView(QWidget):
    """字幕滚动区，按 ID 原位合并实时、回放及语法分析更新。"""

    sentence_activated = pyqtSignal(object)
    analysis_requested = pyqtSignal(int)

    def __init__(
        self,
        display_order: str = "source_first",
        source_lang: str = "",
        read_only: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.display_order = _normalized_order(display_order)
        self.source_lang = source_lang
        self._grammar_enabled = True
        self._read_only = bool(read_only)
        self._blocks: dict[int, SentenceBlock] = {}
        self._order: list[int] = []

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("subtitleScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._container = QWidget()
        self._container.setObjectName("subtitleContainer")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        self._layout.addStretch(1)
        self.scroll_area.setWidget(self._container)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)

    def upsert_segment(self, segment: Segment) -> None:
        """新增句块，或按稳定 ID 原位更新已有句块。"""
        snapshot = segment.snapshot()
        block = self._blocks.get(snapshot.id)
        if block is not None:
            block.set_segment(snapshot)
        else:
            block = SentenceBlock(
                snapshot,
                display_order=self.display_order,
                source_lang=self.source_lang,
                grammar_enabled=self._grammar_enabled,
                read_only=self._read_only,
                parent=self._container,
            )
            block.activated.connect(self.sentence_activated.emit)
            block.analysis_requested.connect(self.analysis_requested.emit)
            self._blocks[snapshot.id] = block
            self._order.append(snapshot.id)
            # 尾部 stretch 始终保留在最后，新句仍按到达顺序追加。
            self._layout.insertWidget(len(self._order) - 1, block)
            block.show()

        self._scroll_to_bottom()
        QTimer.singleShot(0, self._scroll_to_bottom)

    def set_display_order(self, order: str) -> None:
        self.display_order = _normalized_order(order)
        for block in self._blocks.values():
            block.set_display_order(self.display_order)

    def set_grammar_enabled(self, enabled: bool) -> None:
        self._grammar_enabled = bool(enabled)
        self._refresh_interaction()

    def set_read_only(self, read_only: bool) -> None:
        self._read_only = bool(read_only)
        self._refresh_interaction()

    def clear(self) -> None:
        for block in self._blocks.values():
            self._layout.removeWidget(block)
            block.hide()
            block.deleteLater()
        self._blocks.clear()
        self._order.clear()
        self.scroll_area.verticalScrollBar().setValue(0)

    def _refresh_interaction(self) -> None:
        for block in self._blocks.values():
            block.set_interaction(
                grammar_enabled=self._grammar_enabled,
                read_only=self._read_only,
            )

    def _scroll_to_bottom(self) -> None:
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
