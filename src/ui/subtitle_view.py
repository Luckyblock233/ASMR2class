"""按稳定句段 ID 更新的字幕滚动视图（M3 阶段 3 完整 D 区）。

职责边界：
- 工作线程应通过 Qt 排队信号把 Segment 快照交给 GUI 线程，不直接操作控件。
- 实时模式：单击有分析的句块弹语法卡；右键菜单提供手动分析/外观设置。
- 回放模式（replay_mode=True）：单击任意句块只发 seek_requested(segment_id)，
  双击或右键「生成语法分析」发 grammar/analysis 信号——一次手势不触发两个结果
  （见交接文档 §4.1）。
- 滚动：仅新句且跟随开启时自动到底；语法/译文原位更新不抢滚动条；
  用户上翻暂停跟随并显示「回到底部」。
- 容量：实时视图超过 capacity 淘汰最旧句块控件（只删控件，不动 SessionDocument）。
- 视觉层次（参考 LiveTrans）：译文更大加粗为主阅读对象；每句行尾跟
  ASR/TL 延迟小字（阈值着色），不再只在 C 栏显示平均延迟。
"""

from __future__ import annotations

from html import escape

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.subtitle.document import (
    ANALYSIS_FAILED,
    ANALYSIS_PENDING,
    ANALYSIS_READY,
    TRANSLATION_FAILED,
    TRANSLATION_PENDING,
    Segment,
)
from src.ui.appearance import AppearanceSpec

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
    """单个字幕句块；实时模式只有已完成且含分析数据的句块可激活。"""

    activated = pyqtSignal(object)          # 实时：含分析句块被单击（弹语法卡）
    analysis_requested = pyqtSignal(int)    # 右键「生成语法分析」
    seek_requested = pyqtSignal(int)        # 回放：单击句块
    grammar_requested = pyqtSignal(int)     # 回放：双击句块（有分析时）
    appearance_requested = pyqtSignal()     # 右键「外观设置…」

    def __init__(
        self,
        segment: Segment,
        *,
        display_order: str = "source_first",
        source_lang: str = "",
        grammar_enabled: bool = True,
        read_only: bool = False,
        replay_mode: bool = False,
        appearance: AppearanceSpec | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._segment = segment.snapshot()
        self._display_order = _normalized_order(display_order)
        self._source_lang = source_lang
        self._grammar_enabled = grammar_enabled
        self._read_only = read_only
        self._replay_mode = replay_mode
        self._appearance = appearance or AppearanceSpec()
        self._clickable = False
        self._current = False

        self.setObjectName("sentenceBlock")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.setProperty("hovered", False)
        self.setProperty("current", False)

        self.meta_label = QLabel(self)
        self.meta_label.setObjectName("sentenceMeta")

        self.status_label = QLabel(self)
        self.status_label.setObjectName("sentenceAnalysisStatus")

        # 语法分析进度条（仅 ANALYSIS_PENDING 时显示）：浮动在状态文字正下方，
        # 不占用布局高度（不改变句块高度），宽度不超过状态文字。
        self.progress = QProgressBar(self)
        self.progress.setObjectName("analysisProgress")
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setVisible(False)

        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.addWidget(self.meta_label)
        meta_row.addStretch(1)
        meta_row.addWidget(self.status_label)

        self.source_label = QLabel(self)
        self.source_label.setObjectName("sentenceSource")
        self.source_label.setWordWrap(True)
        self.source_label.setTextFormat(Qt.TextFormat.RichText)

        self.translation_label = QLabel(self)
        self.translation_label.setObjectName("sentenceTranslation")
        self.translation_label.setWordWrap(True)
        self.translation_label.setTextFormat(Qt.TextFormat.RichText)

        self.text_layout = QVBoxLayout()
        self.text_layout.setContentsMargins(0, 0, 0, 0)
        self.text_layout.setSpacing(2)

        layout = QVBoxLayout(self)
        self._block_layout = layout
        layout.setSpacing(2)
        layout.addLayout(meta_row)
        layout.addLayout(self.text_layout)

        self.analysis_action = QAction("为此句生成语法分析", self)
        self.analysis_action.setObjectName("generateAnalysisAction")
        self.analysis_action.triggered.connect(self._request_analysis)
        self.appearance_action = QAction("外观设置…", self)
        self.appearance_action.setObjectName("appearanceAction")
        self.appearance_action.triggered.connect(self.appearance_requested.emit)

        self._apply_appearance()
        self._refresh()

    @property
    def segment_id(self) -> int:
        return self._segment.id

    @property
    def is_clickable(self) -> bool:
        return self._clickable

    @property
    def is_current(self) -> bool:
        return self._current

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

    def set_appearance(self, spec: AppearanceSpec) -> None:
        self._appearance = spec
        self._apply_appearance()
        self._refresh()  # 阈值/主题变化需重渲染文本（句内延迟小字等）

    def set_current(self, current: bool) -> None:
        """回放当前句高亮。"""
        if self._current == current:
            return
        self._current = current
        self.setProperty("current", current)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _apply_appearance(self) -> None:
        spec = self._appearance
        self.setStyleSheet(spec.block_stylesheet)  # 分隔线/悬停/当前句高亮随主题
        self.meta_label.setStyleSheet(spec.meta_style)
        self.source_label.setStyleSheet(spec.source_style)
        self.translation_label.setStyleSheet(spec.translation_style)
        self._block_layout.setContentsMargins(*spec.block_margins)
        self.text_layout.setSpacing(spec.block_spacing)

    def _refresh(self) -> None:
        segment = self._segment
        language = f" [{self._source_lang}]" if self._source_lang else ""
        self.meta_label.setText(f"[{_format_timestamp(segment.start)}]{language}")
        self._refresh_text()
        self._refresh_status()
        self._refresh_analysis_action()

        if self._replay_mode:
            self._clickable = True  # 回放中任意句块可单击 seek
        else:
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
        if not translation:
            # 翻译占位：ASR 句先显示原文与占位，翻译完成后按 ID 原位替换
            if self._segment.translation_status == TRANSLATION_PENDING:
                translation = "翻译中…"
            elif self._segment.translation_status == TRANSLATION_FAILED:
                translation = "⚠ 翻译失败"
        has_translation = bool(translation)
        # 句内延迟小字（阈值着色，参考 LiveTrans：ASR/TL 跟在每句后面）
        source_html = escape(source)
        if self._segment.asr_ms > 0:
            source_html += (
                f' <span style="{self._appearance.latency_span_style(self._segment.asr_ms)}">'
                f"ASR {self._segment.asr_ms:.0f}ms</span>"
            )
        translation_html = escape(translation)
        if has_translation and self._segment.translate_ms > 0:
            translation_html += (
                f' <span style="{self._appearance.latency_span_style(self._segment.translate_ms)}">'
                f"TL {self._segment.translate_ms:.0f}ms</span>"
            )
        self.source_label.setText(source_html)
        # translation_only 在没有译文时也必须有可读内容。
        self.translation_label.setText(translation_html if has_translation else escape(source))

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

    # 分析阶段 →（进度值, 阶段说明）；与 grammar.py 的 STAGE_* 对应
    _STAGE_PROGRESS = {
        "queued": (10, "排队中"),
        "rendering": (35, "正在渲染模板"),
        "requesting": (65, "正在调用模型（发送并等待响应）"),
        "parsing": (85, "正在解析响应"),
    }

    def _refresh_status(self) -> None:
        parts: list[str] = []
        tooltips: list[str] = []
        color = self._appearance.theme.meta_color
        error_color = self._appearance.theme.status_err
        translation_status = self._segment.translation_status
        if translation_status == TRANSLATION_FAILED:
            parts.append("翻译失败")
            tooltips.append(self._segment.translation_error or "翻译失败，恢复后自动续翻")
            color = error_color
        status = self._segment.analysis_status
        if status == ANALYSIS_PENDING:
            parts.append("语法分析中…")
            stage = self._segment.analysis_stage
            value, stage_label = self._STAGE_PROGRESS.get(stage, (10, "排队中"))
            tooltips.append(f"正在生成语法分析（{stage_label}）")
            # 进度条 hover 显示当前阶段
            self.progress.setValue(value)
            self.progress.setToolTip(f"语法分析：{stage_label}")
            self.progress.setVisible(True)
            self._position_progress()
        else:
            self.progress.setVisible(False)
        if status == ANALYSIS_READY:
            parts.append("语法分析可用" if self._segment.has_analysis else "语法分析已完成")
            tooltips.append("单击查看语法分析" if self._segment.has_analysis else "语法分析已完成")
        elif status == ANALYSIS_FAILED:
            parts.append("语法分析失败")
            # 失败原因（含分类）在 hover 可见，便于针对性调整
            tooltips.append(self._segment.analysis_error or "语法分析失败，可右键重试")
            color = error_color

        text = " · ".join(parts)
        tooltip = "；".join(tooltips)
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))
        self.status_label.setStyleSheet(f"color: {color}; background: transparent;")
        # tooltip 只挂在状态文字上（进行中的进度条另有自己的阶段提示）；
        # 其他位置不设 tooltip，避免空 tooltip 黑框
        self.status_label.setToolTip(tooltip)

    def _position_progress(self) -> None:
        """进度条浮动在「语法分析中…」状态文字正下方，不占布局高度。"""
        if not self.progress.isVisibleTo(self):
            return
        margins = self._block_layout.contentsMargins()
        width = max(60, min(self.status_label.sizeHint().width(), 240))
        width = min(width, self.width() - margins.left() - margins.right())
        x = self.width() - margins.right() - width
        y = margins.top() + self.status_label.sizeHint().height() + 1
        self.progress.setGeometry(x, y, width, 4)

    def resizeEvent(self, event) -> None:
        self._position_progress()
        super().resizeEvent(event)

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
        menu.addSeparator()
        menu.addAction(self.appearance_action)  # 两处外观入口打开同一个设置页
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
            if self._replay_mode:
                # 回放：单击只 seek，不弹卡（§4.1 手势分离）
                self.seek_requested.emit(self._segment.id)
            else:
                # 每次激活都发出新快照，接收方不能修改句块内部数据。
                self.activated.emit(self._segment.snapshot())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if (
            self._replay_mode
            and event.button() == Qt.MouseButton.LeftButton
            and self._segment.has_analysis
        ):
            # 回放：双击（或右键菜单）独立触发语法卡
            self.grammar_requested.emit(self._segment.id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class SubtitleView(QWidget):
    """字幕滚动区，按 ID 原位合并实时、回放及语法分析更新。"""

    sentence_activated = pyqtSignal(object)
    analysis_requested = pyqtSignal(int)
    seek_requested = pyqtSignal(int)
    grammar_requested = pyqtSignal(int)
    appearance_requested = pyqtSignal()

    def __init__(
        self,
        display_order: str = "source_first",
        source_lang: str = "",
        read_only: bool = False,
        *,
        replay_mode: bool = False,
        auto_scroll: bool = True,
        capacity: int | None = None,
        appearance: AppearanceSpec | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.display_order = _normalized_order(display_order)
        self.source_lang = source_lang
        self._grammar_enabled = True
        self._read_only = bool(read_only)
        self._replay_mode = replay_mode
        self._auto_scroll = auto_scroll
        self._capacity = capacity
        self._appearance = appearance or AppearanceSpec()
        self._blocks: dict[int, SentenceBlock] = {}
        self._order: list[int] = []
        self._evicted_up_to = 0  # 容量淘汰水位线：≤ 该 ID 的迟到更新直接忽略
        self._current_id: int | None = None
        # 回放批量加载不自动跳底；实时模式默认跟随
        self._follow = not replay_mode and auto_scroll
        self._programmatic_scroll = False

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("subtitleScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # 全链路透明：滚动区/视口/容器不带自己的背景色，
        # 悬浮窗背景统一由外层 AppearanceSpec 的 RGBA 绘制（窗口与字幕区一体）
        self.scroll_area.setAutoFillBackground(False)
        self.scroll_area.viewport().setAutoFillBackground(False)
        self.scroll_area.setStyleSheet(
            "QScrollArea#subtitleScrollArea { background: transparent; border: none; }"
        )
        self.scroll_area.viewport().setStyleSheet("background: transparent;")

        self._container = QWidget()
        self._container.setObjectName("subtitleContainer")
        self._container.setAutoFillBackground(False)
        self._container.setStyleSheet("QWidget#subtitleContainer { background: transparent; }")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        self._layout.addStretch(1)
        self.scroll_area.setWidget(self._container)

        self.back_to_bottom = QToolButton(self)
        self.back_to_bottom.setObjectName("backToBottom")
        # 回放模式：定位到当前播放句（底部是最后一句而非正在播放的）
        self.back_to_bottom.setText("回到当前句 ↓" if replay_mode else "回到底部 ↓")
        self.back_to_bottom.setVisible(False)
        self.back_to_bottom.clicked.connect(self._on_back_button)

        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.valueChanged.connect(self._on_scroll_value_changed)
        scrollbar.rangeChanged.connect(self._on_scroll_range_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)

    # ---- 滚动跟随 ----

    @property
    def following(self) -> bool:
        return self._follow

    def _on_back_button(self) -> None:
        if self._replay_mode:
            self.scroll_to_current()
        else:
            self.scroll_to_bottom()

    def scroll_to_bottom(self) -> None:
        """回到底部并恢复跟随。"""
        self._follow = True
        self.back_to_bottom.setVisible(False)
        self._scroll_to_bottom()
        QTimer.singleShot(0, self._scroll_to_bottom)

    def scroll_to_current(self) -> None:
        """回放：定位到当前播放句并恢复跟随（当前句变化时才滚动）。"""
        self._follow = True
        self.back_to_bottom.setVisible(False)
        block = self._blocks.get(self._current_id) if self._current_id is not None else None
        if block is not None:
            self._ensure_visible(block)

    def _ensure_visible(self, block) -> None:
        """程序化滚动（不触发"用户上翻"判定）。"""
        self._programmatic_scroll = True
        self.scroll_area.ensureWidgetVisible(block)
        self._programmatic_scroll = False

    def _scroll_to_bottom(self) -> None:
        scrollbar = self.scroll_area.verticalScrollBar()
        self._programmatic_scroll = True
        scrollbar.setValue(scrollbar.maximum())
        self._programmatic_scroll = False

    def _on_scroll_value_changed(self, value: int) -> None:
        if self._programmatic_scroll:
            return
        scrollbar = self.scroll_area.verticalScrollBar()
        at_bottom = value >= scrollbar.maximum() - 4
        if not at_bottom and self._follow:
            self._follow = False  # 用户上翻：暂停跟随，不被新句抢回
        elif at_bottom and not self._follow:
            self._follow = True
        self.back_to_bottom.setVisible(not self._follow and self._auto_scroll)

    def _on_scroll_range_changed(self, _minimum: int, _maximum: int) -> None:
        # 内容增长时保持贴底（仅跟随开启；原位更新不打断用户阅读位置除外——
        # 原位更新通常不改变内容高度，改变时维持跟随语义）
        if self._follow:
            self._scroll_to_bottom()

    # ---- 句段 ----

    def upsert_segment(self, segment: Segment) -> None:
        """新增句块，或按稳定 ID 原位更新已有句块。"""
        snapshot = segment.snapshot()
        block = self._blocks.get(snapshot.id)
        if block is not None:
            block.set_segment(snapshot)
            return  # 原位更新不抢滚动条
        if snapshot.id <= self._evicted_up_to:
            return  # 已淘汰句段的迟到更新：不重建乱序控件（文档仍完整）
        block = SentenceBlock(
            snapshot,
            display_order=self.display_order,
            source_lang=self.source_lang,
            grammar_enabled=self._grammar_enabled,
            read_only=self._read_only,
            replay_mode=self._replay_mode,
            appearance=self._appearance,
            parent=self._container,
        )
        block.activated.connect(self.sentence_activated.emit)
        block.analysis_requested.connect(self.analysis_requested.emit)
        block.seek_requested.connect(self.seek_requested.emit)
        block.grammar_requested.connect(self.grammar_requested.emit)
        block.appearance_requested.connect(self.appearance_requested.emit)
        self._blocks[snapshot.id] = block
        self._order.append(snapshot.id)
        # 尾部 stretch 始终保留在最后，新句仍按到达顺序追加。
        self._layout.insertWidget(self._layout.count() - 1, block)
        block.show()
        self._evict_if_needed()
        if self._follow and self._auto_scroll:
            self._scroll_to_bottom()
            QTimer.singleShot(0, self._scroll_to_bottom)

    def add_system_message(self, text: str) -> None:
        """系统提示行（如「已切换语言档案」）：不参与句段 ID 体系。"""
        label = QLabel(f"— {text} —", self._container)
        label.setObjectName("systemMessage")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            f"color: {self._appearance.theme.meta_color}; background: transparent; font-size: 12px;"
        )
        self._layout.insertWidget(self._layout.count() - 1, label)
        label.show()
        if self._follow and self._auto_scroll:
            self._scroll_to_bottom()

    def set_current_segment(self, segment_id: int | None) -> None:
        """回放当前句高亮；ID 变化才更新，句间空白传 None 清除。"""
        if segment_id == self._current_id:
            return
        if self._current_id in self._blocks:
            self._blocks[self._current_id].set_current(False)
        self._current_id = segment_id
        block = self._blocks.get(segment_id) if segment_id is not None else None
        if block is not None:
            block.set_current(True)
            if self._follow:
                self._ensure_visible(block)

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

    def set_auto_scroll(self, enabled: bool) -> None:
        self._auto_scroll = bool(enabled)
        if self._auto_scroll and self._follow:
            self.scroll_to_bottom()
        elif not self._auto_scroll:
            self.back_to_bottom.setVisible(False)

    def set_appearance(self, spec: AppearanceSpec) -> None:
        self._appearance = spec
        # 条目间距随紧凑模式调整；分隔线本身在句块 stylesheet 中
        self._layout.setSpacing(3 if spec.compact_mode else 6)
        for block in self._blocks.values():
            block.set_appearance(spec)

    def clear(self) -> None:
        """清空显示：只移除控件，不删除文档/记忆/已落盘文件。"""
        for block in self._blocks.values():
            self._layout.removeWidget(block)
            block.hide()
            block.deleteLater()
        self._blocks.clear()
        self._order.clear()
        self._current_id = None
        # 系统提示行一并清除
        for i in reversed(range(self._layout.count())):
            item = self._layout.itemAt(i)
            widget = item.widget()
            if widget is not None and widget.objectName() == "systemMessage":
                self._layout.removeWidget(widget)
                widget.hide()
                widget.deleteLater()
        self.scroll_area.verticalScrollBar().setValue(0)
        self._follow = not self._replay_mode and self._auto_scroll
        self.back_to_bottom.setVisible(False)

    def _evict_if_needed(self) -> None:
        """实时视图容量上限：淘汰最旧句块控件（文档不受影响）。"""
        if not self._capacity or self._capacity <= 0:
            return
        while len(self._order) > self._capacity:
            oldest = self._order.pop(0)
            block = self._blocks.pop(oldest, None)
            self._evicted_up_to = max(self._evicted_up_to, oldest)
            if block is None:
                continue
            if self._current_id == oldest:
                self._current_id = None
            self._layout.removeWidget(block)
            block.hide()
            block.deleteLater()

    def _refresh_interaction(self) -> None:
        for block in self._blocks.values():
            block.set_interaction(
                grammar_enabled=self._grammar_enabled,
                read_only=self._read_only,
            )
