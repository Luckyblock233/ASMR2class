"""GrammarCardPopup 的档案声明、通用渲染器及 Popup 行为测试。"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QTableWidget,
    QTreeWidget,
)

from src.profiles.loader import LanguageProfile, get_profile
from src.subtitle.document import ANALYSIS_READY, Segment
from src.ui.grammar_card import GrammarCardPopup

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _segment(**overrides) -> Segment:
    values = {
        "id": 7,
        "start": 1.0,
        "end": 2.0,
        "source": "测试原文",
        "translation": "测试译文",
        "analysis_status": ANALYSIS_READY,
    }
    values.update(overrides)
    return Segment(**values)


def _section_title(section) -> str:
    return section.findChild(QLabel, "grammarSectionTitle").text()


def test_actual_ja_profile_uses_generic_renderers_without_declaration(qapp):
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    profile.raw = dict(profile.raw)
    profile.raw.pop("grammar_card", None)
    segment = _segment(
        structure={"custom": {"nested": True}},
        grammar_points=[{"heading": "自定义标题", "detail": "自定义讲解"}],
        vocabulary=[{"entry": "词条", "gloss": "释义"}],
    )
    popup = GrammarCardPopup()
    popup.show_segment(segment, profile)
    qapp.processEvents()

    assert popup.isVisible()
    assert popup.width() == 420
    assert popup.windowOpacity() == 1.0
    assert "#2A2A2A" in popup.styleSheet()
    assert popup.windowFlags() & Qt.WindowType.Popup
    assert popup.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert popup.source_label.text() == "测试原文"
    assert popup.translation_label.text() == "测试译文"
    assert set(popup._section_widgets) == {"structure", "grammar_points", "vocabulary"}

    tree = popup._section_widgets["structure"].findChild(QTreeWidget, "grammarTree")
    assert tree is not None
    assert tree.topLevelItem(0).text(0) == "custom"
    cards = popup._section_widgets["grammar_points"].findChildren(QFrame, "grammarCardItem")
    assert len(cards) == 1
    field_values = [label.text() for label in cards[0].findChildren(QLabel, "grammarFieldValue")]
    assert "自定义标题" in field_values and "自定义讲解" in field_values
    table = popup._section_widgets["vocabulary"].findChild(QTableWidget, "grammarTable")
    assert table is not None
    assert table.item(0, 0).text() == "词条"
    assert table.item(0, 1).text() == "释义"
    assert popup.glossary_button.text() == "＋生词本（M4）"
    assert not popup.glossary_button.isEnabled()
    popup.close()


def test_ja_temporary_raw_declaration_controls_titles_fields_and_badge(qapp):
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    profile.raw = {
        **profile.raw,
        "grammar_card": {
            "sections": [
                "structure",
                {
                    "key": "grammar_points",
                    "title": "自定义语法标题",
                    "renderer": "cards",
                    "title_field": "heading",
                    "body_fields": [{"key": "detail", "label": "自定义讲解"}],
                    "badge": {"field": "rank", "colors": {"基础": "#123456"}},
                },
                {
                    "key": "vocabulary",
                    "title": "自定义词汇标题",
                    "renderer": "table",
                    "columns": [
                        {"key": "entry", "label": "自定义词"},
                        {"key": "gloss", "label": "自定义义"},
                    ],
                },
            ]
        },
    }
    segment = _segment(
        structure=[{"part": "一段", "role": "自定义成分"}],
        grammar_points=[{"heading": "标题甲", "detail": "讲解甲", "rank": "基础"}],
        vocabulary=[{"entry": "词甲", "gloss": "义甲", "ignored": "不渲染"}],
    )
    popup = GrammarCardPopup()
    popup.show_segment(segment, profile)
    qapp.processEvents()

    assert _section_title(popup._section_widgets["structure"]) == "句子结构"
    assert _section_title(popup._section_widgets["grammar_points"]) == "自定义语法标题"
    assert _section_title(popup._section_widgets["vocabulary"]) == "自定义词汇标题"
    card = popup._section_widgets["grammar_points"].findChild(QFrame, "grammarCardItem")
    assert card.findChild(QLabel, "grammarCardTitle").text() == "标题甲"
    assert card.findChild(QLabel, "grammarFieldLabel").text() == "自定义讲解："
    badge = card.findChild(QLabel, "grammarBadge")
    assert badge.text() == "基础"
    assert "#123456" in badge.styleSheet()

    table = popup._section_widgets["vocabulary"].findChild(QTableWidget, "grammarTable")
    assert table.columnCount() == 2
    assert table.horizontalHeaderItem(0).text() == "自定义词"
    assert table.horizontalHeaderItem(1).text() == "自定义义"
    assert table.item(0, 0).text() == "词甲"
    assert table.item(0, 1).text() == "义甲"
    popup.close()


def test_fictional_language_custom_extra_fields_need_no_code_change(qapp):
    profile = LanguageProfile(
        name="fictional",
        language="fictional",
        display_name="虚构语言",
        source_lang="xx",
        target_lang="yy",
        raw={
            "grammar_card": {
                "sections": [
                    {"key": "clauses", "title": "分句甲", "renderer": "tree"},
                    {
                        "key": "notes",
                        "title": "笔记甲",
                        "renderer": "cards",
                        "title_field": "caption",
                        "body_fields": [{"key": "description", "label": "说明甲"}],
                        "badge": {"field": "tier", "colors": {"初阶": "#654321"}},
                    },
                    {
                        "key": "lexicon",
                        "title": "词表甲",
                        "renderer": "table",
                        "columns": [{"key": "token", "label": "形式甲"}],
                    },
                ]
            }
        },
    )
    segment = _segment(
        extra={
            "clauses": {"alpha": {"omega": "任意值"}},
            "notes": [
                {"caption": "虚构标题", "description": "虚构说明", "tier": "初阶"}
            ],
            "lexicon": [{"token": "虚构形式"}],
        }
    )
    popup = GrammarCardPopup()
    popup.show_segment(segment, profile)
    qapp.processEvents()

    assert set(popup._section_widgets) == {"clauses", "notes", "lexicon"}
    tree = popup._section_widgets["clauses"].findChild(QTreeWidget, "grammarTree")
    assert tree.topLevelItem(0).text(0) == "alpha"
    card = popup._section_widgets["notes"].findChild(QFrame, "grammarCardItem")
    assert card.findChild(QLabel, "grammarCardTitle").text() == "虚构标题"
    assert card.findChild(QLabel, "grammarFieldValue").text() == "虚构说明"
    assert "#654321" in card.findChild(QLabel, "grammarBadge").styleSheet()
    table = popup._section_widgets["lexicon"].findChild(QTableWidget, "grammarTable")
    assert table.horizontalHeaderItem(0).text() == "形式甲"
    assert table.item(0, 0).text() == "虚构形式"
    popup.close()


def test_tall_card_scrolls_stays_on_screen_and_escape_closes(qapp):
    profile = LanguageProfile(
        name="fictional",
        language="fictional",
        display_name="虚构语言",
        source_lang="xx",
        target_lang="yy",
        raw={"grammar_card": {"sections": ["structure"]}},
    )
    segment = _segment(
        structure={
            f"item_{index}": {"description": "很长的内容" * 12}
            for index in range(45)
        }
    )
    popup = GrammarCardPopup()
    screen = qapp.primaryScreen().availableGeometry()
    popup.show_segment(segment, profile, anchor=QPoint(screen.right(), screen.bottom()))
    qapp.processEvents()

    geometry = popup.frameGeometry()
    assert geometry.left() >= screen.left()
    assert geometry.top() >= screen.top()
    assert geometry.right() <= screen.right()
    assert geometry.bottom() <= screen.bottom()
    assert popup.scroll_area.verticalScrollBar().maximum() > 0

    QTest.keyClick(popup, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not popup.isVisible()
