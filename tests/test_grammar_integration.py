"""语法分析链路集成验证：右键手动分析 → worker → 回写 → 句块可点击弹卡。

复现用户报告的场景：请求量很少/悬浮窗看不到语法分析模块。
"""

import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from src.config import ProviderConfig
from src.nlp.grammar import GrammarAnalyzer
from src.nlp.llm_client import ChatResult
from src.nlp.prompt_engine import PromptEngine
from src.profiles.loader import LanguageProfile
from src.subtitle.document import ANALYSIS_READY, Segment, SessionDocument
from src.ui.overlay import create_overlay

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class FakeGrammarProvider:
    """语法槽桩：返回合法分析 payload，记录调用次数。"""

    def __init__(self):
        self.cfg = ProviderConfig(id="g", capabilities=["grammar"])
        self.online = True
        self.calls = 0

    def chat(self, prompt, *, system=None, json_mode=False):
        self.calls += 1
        return ChatResult(json.dumps({
            "structure": {"pattern": "〜ています"},
            "grammar_points": [{"name": "〜ています", "explanation": "进行时"}],
            "vocabulary": [{"word": "行く", "meaning": "去"}],
            "extra": {},
        }, ensure_ascii=False))


def _profile():
    return LanguageProfile(
        name="ja", language="ja", display_name="日语",
        source_lang="ja", target_lang="zh",
        prompts={"grammar": "ja/grammar.txt"},
        raw={"grammar_card": {"sections": [
            {"key": "structure", "title": "结构", "renderer": "tree"},
            {"key": "grammar_points", "title": "语法点", "renderer": "cards",
             "title_field": "name",
             "body_fields": [{"key": "explanation", "label": "讲解"}]},
            {"key": "vocabulary", "title": "词汇", "renderer": "table",
             "columns": [{"key": "word", "label": "词"},
                         {"key": "meaning", "label": "释义"}]},
        ]}},
    )


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        QApplication.processEvents()
        time.sleep(0.02)
    return False


def test_manual_analysis_full_path(qapp):
    """右键「为此句生成语法分析」→ 实际调用 provider → 句块回写可点击 → 弹卡。"""
    provider = FakeGrammarProvider()
    document = SessionDocument(language_profile="ja", source_lang="ja", target_lang="zh")
    overlay = create_overlay(profile=_profile())
    overlay.show()

    analyzer = GrammarAnalyzer(
        lambda: provider,
        PromptEngine(os.path.join(PROJECT_ROOT, "config", "prompts")),
        _profile(),
        document,
        on_update=lambda seg: overlay.analysis_ready.emit(seg),
    )
    overlay.analysis_requested.connect(analyzer.submit)
    overlay.set_grammar_enabled(analyzer.available)

    sid = document.add(0.0, 1.0, "彼は行っています", "他正要去")
    overlay.upsert_segment.emit(document.get(sid))
    qapp.processEvents()

    block = overlay.subtitle_view._blocks[sid]
    assert block.analysis_action.isEnabled()  # 右键菜单可用
    block.analysis_action.trigger()  # 模拟右键点击菜单项
    assert _wait(lambda: provider.calls >= 1)
    assert _wait(lambda: block._segment.analysis_status == ANALYSIS_READY)

    # 回写后句块可点击，单击弹卡
    assert block.is_clickable
    QTest.mouseClick(block, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert overlay.grammar_card is not None
    assert overlay.grammar_card.isVisible()
    assert "行っています" in overlay.grammar_card.source_label.text()
    analyzer.drain()
    overlay.close()


def test_analysis_not_submitted_when_slot_unbound(qapp):
    """语法槽未绑定：菜单项置灰并提示，不产生请求（这是设计行为，不是 bug）。"""
    provider = None
    document = SessionDocument()
    overlay = create_overlay(profile=_profile())
    analyzer = GrammarAnalyzer(
        lambda: provider,
        PromptEngine(os.path.join(PROJECT_ROOT, "config", "prompts")),
        _profile(),
        document,
        on_update=lambda seg: overlay.analysis_ready.emit(seg),
    )
    overlay.analysis_requested.connect(analyzer.submit)
    overlay.set_grammar_enabled(analyzer.available)  # False

    sid = document.add(0.0, 1.0, "テスト", "测试")
    overlay.upsert_segment.emit(document.get(sid))
    qapp.processEvents()
    block = overlay.subtitle_view._blocks[sid]
    assert not block.analysis_action.isEnabled()
    assert "绑定语法分析模型" in block.analysis_action.toolTip()
    assert analyzer.submit(sid) is True  # submit 接受任务…
    assert _wait(lambda: block._segment.analysis_status == "failed")
    assert "未绑定" in block.status_label.toolTip()  # …但以明确失败状态告终
    analyzer.drain()
    overlay.close()
