"""无媒体会话重载预览的 offscreen 集成冒烟。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from src.profiles.loader import LanguageProfile
from src.subtitle.document import SessionDocument
from src.ui.replay import run_session_preview


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_session_preview_restores_blocks_and_opens_card(qapp, monkeypatch):
    from src.ui import overlay as overlay_module

    document = SessionDocument(
        language_profile="test", source_lang="xx", target_lang="zh"
    )
    segment_id = document.add(1.0, 2.0, "重载原文", "重载译文")
    document.apply_analysis(segment_id, {"structure": {"custom": "重载分析"}})
    profile = LanguageProfile(
        name="test",
        language="test",
        display_name="测试语言",
        source_lang="xx",
        target_lang="zh",
        raw={"grammar_card": {"sections": ["structure"]}},
    )

    created = []
    real_create_overlay = overlay_module.create_overlay

    def capture_overlay(*args, **kwargs):
        window = real_create_overlay(*args, **kwargs)
        created.append(window)
        return window

    monkeypatch.setattr(overlay_module, "create_overlay", capture_overlay)
    errors = []

    def verify_and_close():
        try:
            assert len(created) == 1
            window = created[0]
            assert window.windowTitle() == "ASR2class 会话重载 - 测试语言"
            block = window.text._blocks[segment_id]
            assert block.source_label.text() == "重载原文"
            assert not block.analysis_action.isEnabled()  # 重载为只读模式
            # §4.1 手势分离：回放单击 = seek（不弹卡），双击 = 语法卡
            seeks = []
            window.seek_requested.connect(seeks.append)
            QTest.mouseClick(block, Qt.MouseButton.LeftButton)
            assert seeks == [segment_id]
            assert window.grammar_card is None or not window.grammar_card.isVisible()
            QTest.mouseDClick(block, Qt.MouseButton.LeftButton)
            assert window.grammar_card is not None
            assert window.grammar_card.isVisible()
            assert window.grammar_card.source_label.text() == "重载原文"
        except Exception as exc:  # 回调内异常先保存，确保 Qt 事件循环能退出
            errors.append(exc)
        finally:
            for window in created:
                window.close()
            QApplication.quit()

    QTimer.singleShot(0, verify_and_close)
    assert run_session_preview(document, profile) == 0
    if errors:
        raise errors[0]
