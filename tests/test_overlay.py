"""悬浮窗测试（FR-3.1/3.2）。Qt 用 offscreen 平台，无需显示器。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from src.ui.overlay import OverlayLine, SubtitleBuffer

LINES = [
    OverlayLine(0.0, 1.0, "おはよう", "早上好"),
    OverlayLine(1.5, 2.5, "もう帰った？", None),  # 降级句
]


def test_buffer_source_first():
    buf = SubtitleBuffer("source_first")
    for ln in LINES:
        buf.add(ln)
    text = buf.render()
    assert "おはよう\n早上好" in text
    assert "もう帰った？" in text  # 无译文句只显示原文


def test_buffer_translation_first():
    buf = SubtitleBuffer("translation_first")
    for ln in LINES:
        buf.add(ln)
    assert "早上好\nおはよう" in buf.render()


def test_buffer_only_modes():
    buf = SubtitleBuffer("source_only")
    for ln in LINES:
        buf.add(ln)
    assert "早上好" not in buf.render()

    buf = SubtitleBuffer("translation_only")
    for ln in LINES:
        buf.add(ln)
    text = buf.render()
    assert "おはよう" not in text.split("\n")[0] or "早上好" in text
    # 降级句回退显示原文
    assert "もう帰った？" in text


def test_buffer_capacity_scrolls():
    buf = SubtitleBuffer("source_first", capacity=3)
    for i in range(5):
        buf.add(OverlayLine(i, i + 1, f"句{i}", f"译{i}"))
    text = buf.render()
    assert "句0" not in text and "句1" not in text
    assert "句4" in text


def test_invalid_order_falls_back():
    buf = SubtitleBuffer("bogus")
    assert buf.display_order == "source_first"


# ---- Qt 窗口（offscreen）----

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_overlay_window_feeds_and_controls(qapp):
    from src.ui.overlay import create_overlay

    w = create_overlay(display_order="source_first", opacity=0.8)
    w.show()
    w.add_sentence.emit(0.0, 1.0, "おはよう", "早上好")
    w.add_sentence.emit(1.5, 2.5, "テスト", None)
    qapp.processEvents()
    text = "\n".join(
        label.text()
        for block in w.text._blocks.values()
        for label in (block.source_label, block.translation_label)
    )
    assert "おはよう" in text and "早上好" in text and "テスト" in text
    assert not w.text._blocks[1].analysis_action.isEnabled()

    # 透明度可调且有下限
    w._adjust_opacity(-0.9)
    assert w.windowOpacity() == pytest.approx(0.2)
    w._adjust_opacity(0.5)

    # 穿透模式隐藏控制按钮
    w.set_click_through(True)
    assert not w.btn_close.isVisible()
    w.set_click_through(False)
    assert w.btn_close.isVisible()

    # 置顶 + 无边框
    from PyQt6.QtCore import Qt

    flags = w.windowFlags()
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert flags & Qt.WindowType.FramelessWindowHint

    # 状态行
    w.set_status.emit("OFFLINE")
    qapp.processEvents()
    assert "OFFLINE" in w.status_label.text()
    w.close()


def test_overlay_upsert_ready_segment_opens_grammar_card(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    from src.profiles.loader import LanguageProfile
    from src.subtitle.document import ANALYSIS_READY, Segment
    from src.ui.overlay import create_overlay

    profile = LanguageProfile(
        name="test",
        language="test",
        display_name="测试语言",
        source_lang="xx",
        target_lang="zh",
        raw={"grammar_card": {"sections": ["structure"]}},
    )
    w = create_overlay(profile=profile)
    w.show()
    w.upsert_segment.emit(
        Segment(
            1.0,
            2.0,
            "待分析原文",
            "参考译文",
            id=9,
            structure={"custom": "分析结果"},
            analysis_status=ANALYSIS_READY,
        )
    )
    qapp.processEvents()

    block = w.text._blocks[9]
    assert block.is_clickable
    QTest.mouseClick(block, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    assert w.grammar_card is not None
    assert w.grammar_card.isVisible()
    assert w.grammar_card.source_label.text() == "待分析原文"
    assert set(w.grammar_card._section_widgets) == {"structure"}
    w.close()
