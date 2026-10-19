"""SubtitleView 的显示顺序、原位更新及语法交互测试。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

from src.subtitle.document import (
    ANALYSIS_FAILED,
    ANALYSIS_NONE,
    ANALYSIS_PENDING,
    ANALYSIS_READY,
    Segment,
)
from src.ui.subtitle_view import SentenceBlock, SubtitleView


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _segment(
    segment_id: int = 1,
    *,
    source: str = "原文",
    translation: str | None = "译文",
    status: str = ANALYSIS_NONE,
    structure=None,
    error: str = "",
) -> Segment:
    return Segment(
        id=segment_id,
        start=float(segment_id - 1),
        end=float(segment_id),
        source=source,
        translation=translation,
        structure=structure,
        analysis_status=status,
        analysis_error=error,
    )


def _text_widgets(block: SentenceBlock) -> list[QWidget]:
    return [
        block.text_layout.itemAt(index).widget()
        for index in range(block.text_layout.count())
    ]


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ("source_first", ("source", "translation")),
        ("translation_first", ("translation", "source")),
        ("source_only", ("source",)),
        ("translation_only", ("translation",)),
    ],
)
def test_four_display_orders_and_missing_translation_fallback(qapp, order, expected):
    view = SubtitleView(display_order=order, source_lang="ja")
    view.upsert_segment(_segment(1))
    view.upsert_segment(_segment(2, source="只有原文", translation=None))
    view.show()
    qapp.processEvents()

    labels = {"source": view._blocks[1].source_label, "translation": view._blocks[1].translation_label}
    assert _text_widgets(view._blocks[1]) == [labels[name] for name in expected]
    assert view._blocks[1].meta_label.text() == "[00:00:00] [ja]"

    fallback = view._blocks[2]
    if order == "translation_only":
        assert _text_widgets(fallback) == [fallback.translation_label]
        assert fallback.translation_label.text() == "只有原文"
    else:
        assert _text_widgets(fallback) == [fallback.source_label]
        assert fallback.source_label.text() == "只有原文"
    view.close()


def test_set_display_order_rerenders_without_reordering_blocks(qapp):
    view = SubtitleView()
    view.upsert_segment(_segment(1, source="一"))
    view.upsert_segment(_segment(2, source="二"))
    first = view._blocks[1]
    second = view._blocks[2]

    view.set_display_order("translation_first")
    assert _text_widgets(first) == [first.translation_label, first.source_label]
    assert view._layout.itemAt(0).widget() is first
    assert view._layout.itemAt(1).widget() is second

    view.set_display_order("not-a-mode")
    assert view.display_order == "source_first"
    assert _text_widgets(second) == [second.source_label, second.translation_label]
    view.close()


def test_upsert_updates_in_place_and_clear_removes_all_blocks(qapp):
    view = SubtitleView()
    view.upsert_segment(_segment(1, source="旧文本"))
    view.upsert_segment(_segment(2, source="第二句"))
    first = view._blocks[1]
    second = view._blocks[2]

    view.upsert_segment(
        _segment(
            1,
            source="新文本",
            translation="新译文",
            status=ANALYSIS_READY,
            structure={"custom": "分析"},
        )
    )
    qapp.processEvents()

    assert len(view._blocks) == 2
    assert view._blocks[1] is first
    assert first.source_label.text() == "新文本"
    assert first.translation_label.text() == "新译文"
    assert view._layout.itemAt(0).widget() is first
    assert view._layout.itemAt(1).widget() is second

    view.clear()
    qapp.processEvents()
    assert not view._blocks and not view._order
    assert view._layout.count() == 1  # 仅保留尾部 stretch
    view.close()


def test_only_ready_block_with_analysis_activates_and_emits_snapshot(qapp):
    view = SubtitleView()
    blocks = [
        _segment(1, status=ANALYSIS_NONE),
        _segment(2, status=ANALYSIS_PENDING),
        _segment(3, status=ANALYSIS_FAILED),
        _segment(4, status=ANALYSIS_READY),
        _segment(5, status=ANALYSIS_READY, structure={"custom": "分析"}),
    ]
    for segment in blocks:
        view.upsert_segment(segment)
    received = []
    view.sentence_activated.connect(received.append)
    view.resize(520, 520)
    view.show()
    qapp.processEvents()

    for segment in blocks:
        block = view._blocks[segment.id]
        assert block.is_clickable is (segment.id == 5)
        expected_cursor = (
            Qt.CursorShape.PointingHandCursor
            if segment.id == 5
            else Qt.CursorShape.ArrowCursor
        )
        assert block.cursor().shape() == expected_cursor
        QTest.mouseClick(block, Qt.MouseButton.LeftButton)

    assert len(received) == 1
    assert received[0] == blocks[-1]
    assert received[0] is not view._blocks[5]._segment
    received[0].source = "接收方修改"
    assert view._blocks[5].source_label.text() == "原文"
    view.close()


def test_context_action_states_request_ids_and_status_hints(qapp):
    view = SubtitleView()
    view.upsert_segment(_segment(1, status=ANALYSIS_NONE))
    view.upsert_segment(_segment(2, status=ANALYSIS_FAILED, error="超时"))
    view.upsert_segment(_segment(3, status=ANALYSIS_PENDING))
    view.upsert_segment(_segment(4, status=ANALYSIS_READY, structure={"custom": "完成"}))
    requests = []
    view.analysis_requested.connect(requests.append)

    assert view._blocks[1].analysis_action.isEnabled()
    assert view._blocks[2].analysis_action.isEnabled()
    assert not view._blocks[3].analysis_action.isEnabled()
    assert "正在" in view._blocks[3].analysis_action.toolTip()
    assert not view._blocks[4].analysis_action.isEnabled()
    assert "已有" in view._blocks[4].analysis_action.toolTip()
    assert view._blocks[2].status_label.toolTip() == "超时"

    view._blocks[1].analysis_action.trigger()
    view._blocks[2].analysis_action.trigger()
    view._blocks[3].analysis_action.trigger()
    assert requests == [1, 2]

    view.set_grammar_enabled(False)
    assert all(not block.analysis_action.isEnabled() for block in view._blocks.values())
    assert "绑定语法分析模型" in view._blocks[1].analysis_action.toolTip()

    view.set_grammar_enabled(True)
    view.set_read_only(True)
    assert all(not block.analysis_action.isEnabled() for block in view._blocks.values())
    assert "只读" in view._blocks[1].analysis_action.toolTip()

    view.set_read_only(False)
    assert view._blocks[1].analysis_action.isEnabled()
    assert view._blocks[1].contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    view.close()


def test_upsert_scrolls_to_bottom(qapp):
    view = SubtitleView()
    view.resize(360, 100)
    view.show()
    for index in range(1, 16):
        view.upsert_segment(_segment(index, source=f"句{index}"))
    qapp.processEvents()

    scrollbar = view.scroll_area.verticalScrollBar()
    assert scrollbar.maximum() > 0
    assert scrollbar.value() == scrollbar.maximum()
    view.close()
