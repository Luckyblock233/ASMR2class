"""分析笔记与 Markdown 导出测试。"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from src.profiles.loader import get_profile
from src.subtitle.analysis_md import (
    document_to_markdown,
    export_markdown,
    section_declarations,
    segment_to_markdown,
)
from src.subtitle.document import SessionDocument

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JA = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
ZH = get_profile(PROJECT_ROOT / "config" / "languages", "zh")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _document_with_analysis():
    doc = SessionDocument(language_profile="ja", source_lang="ja", target_lang="zh")
    s1 = doc.add(0.0, 1.5, "彼は行っています", "他正要去")
    doc.apply_analysis(s1, {
        "structure": {"type": "主谓句", "clauses": [{"text": "彼は", "role": "主语", "note": "提示助词"}]},
        "grammar_points": [{"name": "〜ています", "explanation": "进行时", "example": "食べています", "jlpt": "N5"}],
        "vocabulary": [{"word": "行く", "reading": "いく", "pos": "动词", "meaning": "去", "jlpt": "N5"}],
        "extra": {},
    })
    doc.add(2.0, 3.0, "未分析句", "未分析")  # 无分析：不应出现在笔记中
    s3 = doc.add(4.0, 5.0, "第二句分析", "译文二")
    doc.apply_analysis(s3, {"structure": None, "grammar_points": [], "vocabulary": [], "extra": {}})
    # extra 携带自定义键（未知区块也应导出）
    doc.apply_analysis(s3, {"structure": None, "grammar_points": [], "vocabulary": [],
                            "extra": {"custom_note": "自定义"}})
    return doc


def test_section_declarations_from_profile():
    decls = section_declarations(JA)
    keys = [d["key"] for d in decls]
    assert keys == ["structure", "grammar_points", "vocabulary"]
    assert decls[1]["renderer"] == "cards"
    assert decls[1]["badge"]["field"] == "jlpt"


def test_segment_markdown_renders_all_sections():
    doc = _document_with_analysis()
    md = segment_to_markdown(doc.get(1), JA)
    assert "### [00:00:00] 彼は行っています" in md
    assert "> 他正要去" in md
    assert "句子结构" in md and "主谓句" in md and "提示助词" in md  # tree
    assert "〜ています" in md and "N5" in md and "进行时" in md    # cards + badge
    assert "| 词 | 假名读音 |" in md and "いく" in md                # table


def test_document_markdown_order_and_filtering(tmp_path):
    doc = _document_with_analysis()
    md = document_to_markdown(doc, JA)
    assert "未分析句" not in md          # 无分析句不进笔记
    assert md.index("彼は行っています") < md.index("第二句分析")  # 识别顺序
    assert "自定义" in md                # extra 自定义键经兜底导出

    out = export_markdown(doc, JA, tmp_path / "notes.md")
    assert out.is_file()
    assert "语法分析笔记" in out.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))  # 原子写不留临时文件


def test_notebook_window_shows_analyzed_in_order(qapp, tmp_path):
    """笔记窗口：仅已分析句、按识别顺序、点击句块弹语法卡。"""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    from src.ui import i18n
    from src.ui.notebook import create_notebook_window

    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    doc = _document_with_analysis()
    w = create_notebook_window(lambda: doc, lambda: JA)
    w.show()
    qapp.processEvents()
    ids = list(w.view._order)
    assert ids == [1, 3]  # 识别顺序，未分析句不在其中
    assert "已分析" in w.header.text()

    block = w.view._blocks[1]
    assert block.is_clickable
    QTest.mouseClick(block, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert w._card is not None and w._card.isVisible()
    assert "行っています" in w._card.source_label.text()

    # 新分析后刷新
    s4 = doc.add(6.0, 7.0, "新分析句", "新译文")
    doc.apply_analysis(s4, {"structure": {"type": "判断句"}, "grammar_points": [],
                            "vocabulary": [], "extra": {}})
    w.refresh()
    assert 4 in w.view._blocks
    w.close()


def test_notebook_sort_by_analysis_order(qapp):
    """分析顺序排序：按 analysis_seq（分析完成先后），与识别顺序不同。"""
    from src.ui import i18n
    from src.ui.notebook import SORT_ANALYSIS, create_notebook_window

    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    doc = SessionDocument(language_profile="ja", source_lang="ja", target_lang="zh")
    s1 = doc.add(0.0, 1.0, "第一句", "一")
    s2 = doc.add(2.0, 3.0, "第二句", "二")
    payload = {"structure": {"type": "t"}, "grammar_points": [], "vocabulary": [], "extra": {}}
    doc.apply_analysis(s2, payload)  # 第二句先分析完成
    doc.apply_analysis(s1, payload)
    assert doc.get(s1).analysis_seq > doc.get(s2).analysis_seq

    w = create_notebook_window(lambda: doc, lambda: JA)
    w.show()
    assert list(w.view._order) == [1, 2]          # 默认识别顺序
    idx = w.sort_combo.findData(SORT_ANALYSIS)
    w.sort_combo.setCurrentIndex(idx)
    w.sort_combo.activated.emit(idx)
    qapp.processEvents()
    assert list(w.view._order) == [2, 1]          # 分析顺序：s2 先完成
    w.close()


def test_notebook_empty_state(qapp):
    from src.ui import i18n
    from src.ui.notebook import create_notebook_window

    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    doc = SessionDocument()
    doc.add(0, 1, "无分析", None)
    w = create_notebook_window(lambda: doc, lambda: JA)
    w.show()
    assert "尚无" in w.header.text()
    assert not w.view._blocks
    w.close()


def test_overlay_notebook_button(qapp):
    """A 栏笔记按钮：提供 factory 时显示并打开窗口。"""
    from src.ui import i18n
    from src.ui.overlay import create_overlay

    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    w = create_overlay()
    w.show()
    assert not w.btn_notebook.isVisible()  # 无 factory 隐藏
    w.close()

    doc = _document_with_analysis()
    from src.ui.notebook import create_notebook_window

    w = create_overlay(notebook_factory=lambda: create_notebook_window(lambda: doc, lambda: JA))
    w.show()
    assert w.btn_notebook.isVisible()
    w.btn_notebook.click()
    qapp.processEvents()
    assert w._notebook_window is not None and w._notebook_window.isVisible()
    assert "彼は行っています" in w._notebook_window.view._blocks[1].source_label.text()
    w._notebook_window.close()
    w.close()
