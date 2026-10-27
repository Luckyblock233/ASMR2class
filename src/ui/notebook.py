"""分析笔记窗口：悬浮窗风格（半透明主题化），已分析单句集中查看。

- 默认折叠：每个句块只显示时间 + 原句 + 译文（如有）；点击句块弹出与
  悬浮窗一致的语法卡（GrammarCardPopup）。
- 排序可选：识别顺序（句段 ID）/ 分析顺序（analysis_seq，分析完成先后）。
- 导出 Markdown：与 analysis_md 渲染同源。
文档/档案经 getter 注入，档案切换后刷新即取当前会话。
"""

from __future__ import annotations

from typing import Callable

from ..config import UIConfig

SORT_RECOGNITION = "recognition"
SORT_ANALYSIS = "analysis"


def create_notebook_window(
    document_getter: Callable,
    profile_getter: Callable,
    ui: UIConfig | None = None,
):
    """构建分析笔记窗口（延迟导入 PyQt6）。"""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QComboBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QSizeGrip,
        QVBoxLayout,
        QWidget,
        QFileDialog,
    )

    from ..subtitle.analysis_md import export_markdown
    from .appearance import AppearanceSpec
    from .grammar_card import GrammarCardPopup
    from .i18n import tr
    from .subtitle_view import SubtitleView

    appearance = AppearanceSpec.from_ui(ui or UIConfig())

    class NotebookWindow(QWidget):
        """半透明无边框窗口，复用悬浮窗的 D 区句块与语法卡。"""

        def __init__(self):
            super().__init__()
            self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
            self.setWindowTitle(tr("notebook.title"))
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.resize(560, 480)
            self.setMinimumSize(400, 240)
            self._drag_pos = None
            self._sort_mode = SORT_RECOGNITION
            self._card: GrammarCardPopup | None = None

            # 与悬浮窗一致的半透明主题背景
            self.central = QWidget(self)
            self.central.setObjectName("centralHost")
            self.central.setStyleSheet(
                f"QWidget#centralHost {{ background: {appearance.background_rgba};"
                " border-radius: 8px; }"
                + appearance.bars_stylesheet
            )

            self.header = QLabel("", self.central)
            self.header.setStyleSheet(appearance.meta_style + " font-weight: 600;")

            self.sort_combo = QComboBox(self.central)
            self.sort_combo.setObjectName("notebookSort")
            self.sort_combo.addItem(tr("notebook.sort.recognition"), SORT_RECOGNITION)
            self.sort_combo.addItem(tr("notebook.sort.analysis"), SORT_ANALYSIS)
            self.sort_combo.activated.connect(self._on_sort_changed)

            self.btn_refresh = QPushButton(tr("notebook.refresh"), self.central)
            self.btn_export = QPushButton(tr("notebook.export"), self.central)
            self.btn_close = QPushButton("×", self.central)
            self.btn_refresh.clicked.connect(self.refresh)
            self.btn_export.clicked.connect(self._export)
            self.btn_close.clicked.connect(self.close)

            top_row = QHBoxLayout()
            top_row.addWidget(self.header, 1)
            top_row.addWidget(QLabel(tr("notebook.sort"), self.central))
            top_row.addWidget(self.sort_combo)
            top_row.addWidget(self.btn_refresh)
            top_row.addWidget(self.btn_export)
            top_row.addWidget(self.btn_close)

            self.view = SubtitleView(
                display_order="source_first",
                read_only=False,
                auto_scroll=False,
                appearance=appearance,
                parent=self.central,
            )
            self.view.set_grammar_enabled(False)  # 右键不再触发分析，只看卡
            self.view.sentence_activated.connect(self._open_card)

            layout = QVBoxLayout(self.central)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.addLayout(top_row)
            layout.addWidget(self.view, 1)
            bottom = QHBoxLayout()
            bottom.setContentsMargins(0, 0, 0, 0)
            bottom.addStretch(1)
            bottom.addWidget(
                QSizeGrip(self.central),
                0,
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
            )
            layout.addLayout(bottom)

            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.addWidget(self.central)
            self.refresh()

        # ---- 数据 ----

        def _sorted_segments(self) -> list:
            document = document_getter()
            if document is None:
                return []
            analyzed = [s for s in document.segments if s.has_analysis]
            if self._sort_mode == SORT_ANALYSIS:
                analyzed.sort(key=lambda s: (s.analysis_seq or s.id))
            else:
                analyzed.sort(key=lambda s: s.id)
            return analyzed

        def refresh(self) -> None:
            document = document_getter()
            profile = profile_getter()
            if document is None or profile is None:
                self.header.setText(tr("notebook.empty"))
                self.view.clear()
                return
            segments = self._sorted_segments()
            if not segments:
                self.header.setText(tr("notebook.empty"))
                self.view.clear()
                return
            self.header.setText(
                f"{profile.display_name}（{document.source_lang} → {document.target_lang}）"
                f" · {tr('notebook.count')} {len(segments)}"
            )
            if profile is not None:
                self.view.source_lang = profile.source_lang
                self.view.set_display_order(
                    profile.subtitle.get("display_order", "source_first")
                )
            self.view.clear()
            for segment in segments:
                self.view.upsert_segment(segment)

        def _open_card(self, segment) -> None:
            profile = profile_getter()
            if profile is None or not segment.has_analysis:
                return
            if self._card is None:
                self._card = GrammarCardPopup()
            self._card.show_segment(segment, profile, anchor=self)

        def _on_sort_changed(self, _index: int) -> None:
            self._sort_mode = self.sort_combo.currentData() or SORT_RECOGNITION
            self.refresh()

        def _export(self) -> None:
            document = document_getter()
            profile = profile_getter()
            if document is None or profile is None:
                return
            if not any(s.has_analysis for s in document.segments):
                QMessageBox.information(self, tr("notebook.export"), tr("notebook.empty"))
                return
            default = f"analysis-notes.{document.source_lang}-{document.target_lang}.md"
            path, _ = QFileDialog.getSaveFileName(
                self, tr("notebook.export"), default, "Markdown (*.md)"
            )
            if not path:
                return
            try:
                out = export_markdown(document, profile, path)
            except OSError as e:
                QMessageBox.warning(self, tr("notebook.export"), str(e))
                return
            QMessageBox.information(
                self, tr("notebook.export"), tr("notebook.export_done") + f"\n{out}"
            )

        def closeEvent(self, event) -> None:
            if self._card is not None:
                self._card.close()
            super().closeEvent(event)

        # ---- 拖动（仅顶部行空白处）----

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

    return NotebookWindow()
