"""由语言档案声明驱动的语法学习卡。

卡片只实现通用的 tree/cards/table 与徽章渲染；字段、标题及颜色均来自
``LanguageProfile.raw['grammar_card']['sections']``，新增语言无需修改本模块。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QCursor, QGuiApplication, QPalette
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.subtitle.document import Segment

CARD_WIDTH = 420
CARD_BACKGROUND = "#2A2A2A"
_DEFAULT_SECTIONS = {
    "structure": ("句子结构", "tree"),
    "grammar_points": ("语法点", "cards"),
    "vocabulary": ("重点词汇", "table"),
}
_MISSING = object()


def _format_value(value: Any, *, pretty: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            default=str,
        )
    except (TypeError, ValueError):
        return str(value)


def _has_content(value: Any) -> bool:
    return value is not _MISSING and value is not None and value != "" and value != [] and value != {}


def _safe_color(value: Any) -> QColor:
    color = QColor(str(value or "#808080"))
    return color if color.isValid() else QColor("#808080")


class GrammarCardPopup(QWidget):
    """固定宽度、不透明且由 Qt Popup 管理关闭行为的语法卡。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("grammarCardPopup")
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setFixedWidth(CARD_WIDTH)
        self.setWindowOpacity(1.0)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(CARD_BACKGROUND))
        self.setPalette(palette)
        self.setStyleSheet(
            f"QWidget#grammarCardPopup {{ background-color: {CARD_BACKGROUND}; color: #FFFFFF; }}"
            "QWidget#grammarCardContent, QWidget#grammarHeader, QWidget#grammarCards {"
            " background: transparent; }"
            "QLabel { background: transparent; color: #FFFFFF; }"
            "QLabel#grammarSectionTitle { color: #B8D8FF; font-weight: 600; }"
            "QLabel#grammarTranslation, QLabel#grammarFieldLabel { color: #B8B8B8; }"
            "QFrame#grammarCardItem { background-color: #323232;"
            " border: 1px solid #454545; border-radius: 4px; }"
            "QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }"
            "QTreeWidget, QTableWidget { background-color: #2A2A2A; color: #FFFFFF;"
            " border: 1px solid #454545; gridline-color: #454545; }"
            "QHeaderView::section { background-color: #383838; color: #FFFFFF;"
            " border: none; padding: 4px; }"
            "QPushButton { background-color: #3A3A3A; color: #FFFFFF;"
            " border: 1px solid #505050; border-radius: 4px; padding: 5px; }"
            "QPushButton:disabled { color: #777777; background-color: #303030; }"
        )

        self._segment: Segment | None = None
        self._content: QWidget | None = None
        self._section_widgets: dict[str, QWidget] = {}

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("grammarCardScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        self.glossary_button = QPushButton("＋生词本（M4）", self)
        self.glossary_button.setObjectName("glossaryButton")
        self.glossary_button.setEnabled(False)
        self.glossary_button.setToolTip("生词本功能将在 M4 提供")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self.scroll_area)
        layout.addWidget(self.glossary_button)

    def show_segment(
        self,
        segment: Segment,
        profile,
        anchor: QWidget | QPoint | QRect | None = None,
    ) -> None:
        """重建卡片内容并显示；anchor 可为句块、全局点或全局矩形。"""
        self._segment = segment.snapshot()
        self._build_content(profile)
        anchor_rect = self._anchor_rect(anchor)
        screen = QGuiApplication.screenAt(anchor_rect.center()) or QGuiApplication.primaryScreen()
        self.ensurePolished()
        if self._content is not None:
            self._content.ensurePolished()
            self._content.adjustSize()
        self._resize_for_screen(screen.availableGeometry())
        self.move(self._popup_position(anchor_rect, screen.availableGeometry()))
        self.scroll_area.verticalScrollBar().setValue(0)
        self.show()
        self.raise_()

    def _build_content(self, profile) -> None:
        old_content = self.scroll_area.takeWidget()
        if old_content is not None:
            old_content.deleteLater()

        self._section_widgets = {}
        content = QWidget()
        content.setObjectName("grammarCardContent")
        self._content = content
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QWidget(content)
        header.setObjectName("grammarHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        self.source_label = QLabel(self._segment.source or "", header)
        self.source_label.setObjectName("grammarSource")
        self.source_label.setWordWrap(True)
        self.translation_label = QLabel(
            self._segment.translation or "（暂无译文）", header
        )
        self.translation_label.setObjectName("grammarTranslation")
        self.translation_label.setWordWrap(True)
        header_layout.addWidget(self.source_label)
        header_layout.addWidget(self.translation_label)
        layout.addWidget(header)

        payload = self._segment.analysis_payload()
        extra = payload.get("extra") or {}
        for declaration in self._section_declarations(profile):
            key = declaration["key"]
            value = payload.get(key, extra.get(key, _MISSING))
            if not _has_content(value):
                continue
            renderer = declaration.pop("renderer")
            if renderer == "cards":
                body = self._render_cards(value, declaration)
            elif renderer == "table":
                body = self._render_table(value, declaration)
            else:
                # 未知或省略的渲染器均使用通用树，避免语言档案导致卡片崩溃。
                body = self._render_tree(value)
            section = self._make_section(declaration["title"], body, key)
            self._section_widgets[key] = section
            layout.addWidget(section)

        layout.addStretch(1)
        self.scroll_area.setWidget(content)

    def _section_declarations(self, profile) -> list[dict[str, Any]]:
        raw = getattr(profile, "raw", None) or {}
        card_config = raw.get("grammar_card") or {}
        sections = card_config.get("sections", _MISSING)
        if sections is _MISSING or sections is None:
            sections = list(_DEFAULT_SECTIONS)
        elif isinstance(sections, (str, Mapping)):
            sections = [sections]
        elif not isinstance(sections, list):
            sections = list(_DEFAULT_SECTIONS)

        declarations: list[dict[str, Any]] = []
        for item in sections:
            if isinstance(item, str):
                item = {"key": item}
            if not isinstance(item, Mapping):
                continue
            key = str(item.get("key") or "")
            if not key:
                continue
            default_title, default_renderer = _DEFAULT_SECTIONS.get(
                key, (key, "tree")
            )
            declaration = dict(item)
            declaration["key"] = key
            declaration["title"] = str(item.get("title") or default_title)
            declaration["renderer"] = str(item.get("renderer") or default_renderer)
            declarations.append(declaration)
        return declarations

    def _make_section(self, title: str, body: QWidget, key: str) -> QWidget:
        section = QWidget(self._content)
        section.setObjectName("grammarSection")
        section.setProperty("section_key", key)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        title_label = QLabel(title, section)
        title_label.setObjectName("grammarSectionTitle")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(body)
        return section

    def _render_tree(self, value: Any) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setObjectName("grammarTree")
        tree.setColumnCount(2)
        tree.setHeaderLabels(["项目", "内容"])
        tree.setHeaderHidden(True)
        tree.setIndentation(14)
        tree.setWordWrap(True)
        tree.setEditTriggers(QTreeWidget.EditTrigger.NoEditTriggers)
        tree.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        tree.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        def add_node(parent: QTreeWidgetItem | None, label: str, node_value: Any) -> None:
            text = "" if isinstance(node_value, (Mapping, list, tuple)) else _format_value(node_value)
            item = QTreeWidgetItem([str(label), text])
            if parent is None:
                tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            if isinstance(node_value, Mapping):
                for child_key, child_value in node_value.items():
                    add_node(item, str(child_key), child_value)
            elif isinstance(node_value, (list, tuple)):
                for index, child_value in enumerate(node_value, 1):
                    add_node(item, f"[{index}]", child_value)

        if isinstance(value, Mapping):
            for key, child_value in value.items():
                add_node(None, str(key), child_value)
        elif isinstance(value, (list, tuple)):
            for index, child_value in enumerate(value, 1):
                add_node(None, f"[{index}]", child_value)
        else:
            add_node(None, "内容", value)

        tree.expandAll()
        tree.doItemsLayout()
        fallback_height = tree.fontMetrics().lineSpacing() + 8

        def item_height(item: QTreeWidgetItem) -> int:
            height = tree.visualItemRect(item).height() or fallback_height
            for index in range(item.childCount()):
                height += item_height(item.child(index))
            return height

        content_height = sum(
            item_height(tree.topLevelItem(index))
            for index in range(tree.topLevelItemCount())
        )
        tree.setFixedHeight(max(32, content_height + 2 * tree.frameWidth() + 2))
        return tree

    def _render_cards(self, value: Any, declaration: Mapping[str, Any]) -> QWidget:
        container = QWidget()
        container.setObjectName("grammarCards")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        items = value if isinstance(value, (list, tuple)) else [value]
        title_field = str(declaration.get("title_field") or "")
        badge = declaration.get("badge") or {}
        badge_field = str(badge.get("field") or "") if isinstance(badge, Mapping) else ""
        body_fields = declaration.get("body_fields")

        for index, item in enumerate(items, 1):
            card = QFrame(container)
            card.setObjectName("grammarCardItem")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 6, 8, 6)
            card_layout.setSpacing(4)

            item_map = item if isinstance(item, Mapping) else {}
            title_value = item_map.get(title_field) if title_field else None
            if title_value in (None, ""):
                title_value = (
                    _format_value(item)
                    if not item_map
                    else f"{declaration['title']} {index}"
                )

            title_row = QHBoxLayout()
            title_row.setContentsMargins(0, 0, 0, 0)
            title_label = QLabel(_format_value(title_value), card)
            title_label.setObjectName("grammarCardTitle")
            title_label.setWordWrap(True)
            title_row.addWidget(title_label, 1)

            badge_value = item_map.get(badge_field) if badge_field else None
            if badge_value not in (None, ""):
                colors = badge.get("colors") or {}
                color_value = colors.get(str(badge_value), "#808080")
                title_row.addWidget(self._make_badge(_format_value(badge_value), color_value), 0)
            card_layout.addLayout(title_row)

            fields = self._card_body_fields(body_fields, item_map, title_field, badge_field)
            for field_key, field_label in fields:
                label = QLabel(f"{field_label}：", card)
                label.setObjectName("grammarFieldLabel")
                value_label = QLabel(
                    _format_value(item_map.get(field_key, ""), pretty=True), card
                )
                value_label.setObjectName("grammarFieldValue")
                value_label.setWordWrap(True)
                card_layout.addWidget(label)
                card_layout.addWidget(value_label)

            layout.addWidget(card)
        return container

    @staticmethod
    def _card_body_fields(
        declared: Any,
        item: Mapping[str, Any],
        title_field: str,
        badge_field: str,
    ) -> list[tuple[str, str]]:
        if isinstance(declared, list):
            fields: list[tuple[str, str]] = []
            for entry in declared:
                if isinstance(entry, str):
                    fields.append((entry, entry))
                elif isinstance(entry, Mapping):
                    key = str(entry.get("key") or "")
                    if key:
                        fields.append((key, str(entry.get("label") or key)))
            return fields
        return [
            (str(key), str(key))
            for key in item
            if str(key) not in (title_field, badge_field)
        ]

    def _make_badge(self, text: str, color_value: Any) -> QLabel:
        color = _safe_color(color_value)
        badge = QLabel(text)
        badge.setObjectName("grammarBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"QLabel#grammarBadge {{ background-color: {color.name()}; color: #FFFFFF;"
            " border-radius: 3px; padding: 1px 5px; font-size: 11px; }"
        )
        return badge

    def _render_table(self, value: Any, declaration: Mapping[str, Any]) -> QTableWidget:
        rows = list(value) if isinstance(value, (list, tuple)) else [value]
        columns = self._table_columns(declaration.get("columns"), rows)

        table = QTableWidget()
        table.setObjectName("grammarTable")
        table.setColumnCount(len(columns))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels([label for _, label in columns])
        table.verticalHeader().hide()
        table.setWordWrap(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for row_index, row in enumerate(rows):
            row_map = row if isinstance(row, Mapping) else {}
            for column_index, (key, _) in enumerate(columns):
                if row_map:
                    cell_value = row_map.get(key, "")
                elif len(columns) == 1:
                    cell_value = row
                else:
                    cell_value = ""
                table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(_format_value(cell_value)),
                )

        table.resizeRowsToContents()
        header_height = table.horizontalHeader().sizeHint().height()
        rows_height = sum(table.rowHeight(index) for index in range(table.rowCount()))
        table.setFixedHeight(max(42, header_height + rows_height + 2 * table.frameWidth() + 2))
        return table

    @staticmethod
    def _table_columns(declared: Any, rows: list[Any]) -> list[tuple[str, str]]:
        columns: list[tuple[str, str]] = []
        if isinstance(declared, list):
            for entry in declared:
                if isinstance(entry, str):
                    columns.append((entry, entry))
                elif isinstance(entry, Mapping):
                    key = str(entry.get("key") or "")
                    if key:
                        columns.append((key, str(entry.get("label") or key)))
        if columns:
            return columns

        # 未声明列时按数据中出现顺序生成通用表格，不写死任何语言字段。
        for row in rows:
            if isinstance(row, Mapping):
                for key in row:
                    key_text = str(key)
                    if key_text not in [column[0] for column in columns]:
                        columns.append((key_text, key_text))
        return columns or [("value", "内容")]

    def _anchor_rect(self, anchor: QWidget | QPoint | QRect | None) -> QRect:
        if isinstance(anchor, QWidget):
            return QRect(anchor.mapToGlobal(QPoint(0, 0)), anchor.size())
        if isinstance(anchor, QRect):
            return QRect(anchor)
        if isinstance(anchor, QPoint):
            return QRect(anchor, anchor)
        point = QCursor.pos()
        return QRect(point, point)

    def _resize_for_screen(self, available: QRect) -> None:
        margins = self.layout().contentsMargins()
        fixed_height = (
            margins.top()
            + margins.bottom()
            + self.layout().spacing()
            + self.glossary_button.sizeHint().height()
        )
        max_card_height = max(100, available.height() - 16)
        max_scroll_height = max(60, max_card_height - fixed_height)
        content_height = 0
        if self._content is not None and self._content.layout() is not None:
            content_height = self._content.layout().sizeHint().height()
        scroll_height = min(max_scroll_height, max(48, content_height + 4))
        self.scroll_area.setFixedHeight(scroll_height)
        self.adjustSize()
        self.resize(CARD_WIDTH, min(self.sizeHint().height(), max_card_height))

    def _popup_position(self, anchor: QRect, available: QRect) -> QPoint:
        spacing = 8
        x = anchor.right() + 1 + spacing
        y = anchor.center().y() - self.height() // 2
        if x + self.width() > available.right() + 1:
            x = anchor.left() - self.width() - spacing

        max_x = available.right() - self.width() + 1
        max_y = available.bottom() - self.height() + 1
        x = min(max(x, available.left()), max(available.left(), max_x))
        y = min(max(y, available.top()), max(available.top(), max_y))
        return QPoint(x, y)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)
