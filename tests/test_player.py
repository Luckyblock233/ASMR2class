"""媒体回放（阶段 4）：ReplaySync 同步逻辑、外部多行 SRT、回放窗口冒烟。"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from src.subtitle.document import SessionDocument
from src.subtitle.parser import load_srt
from src.ui.player import ReplaySync

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _document():
    doc = SessionDocument(language_profile="ja", source_lang="ja", target_lang="zh")
    doc.add(0.0, 1.0, "一", "1")
    doc.add(2.0, 3.0, "二", "2")
    doc.add(5.0, 6.0, "三", "3")
    return doc


def test_replay_sync_position_transitions():
    sync = ReplaySync(_document())
    assert sync.on_position(0.5) == (1, True)    # 进入句 1
    assert sync.on_position(0.8) == (1, False)   # 同句内移动不重复触发
    assert sync.on_position(1.5) == (None, True)  # 句间空白清除高亮
    assert sync.on_position(1.6) == (None, False)
    assert sync.on_position(2.5) == (2, True)
    assert sync.on_position(5.5) == (3, True)
    assert sync.on_position(10.0) == (None, True)  # 越过末尾


def test_replay_sync_seek_ms():
    sync = ReplaySync(_document())
    assert sync.seek_position_ms(2) == 2000
    assert sync.seek_position_ms(999) is None


def test_external_srt_multiline_preserved(tmp_path):
    """外部 SRT 完整块读取、保留多行；内部双语格式保持两行语义。"""
    srt = tmp_path / "ext.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n第一行\n第二行\n第三行\n\n"
        "2\n00:00:04,000 --> 00:00:05,000\n单行\n",
        encoding="utf-8",
    )
    external = load_srt(srt, external=True)
    assert external[0].source == "第一行\n第二行\n第三行"
    assert external[0].translation is None
    assert external[1].source == "单行"

    internal = load_srt(srt)  # 内部双语格式：第一行原文、第二行译文
    assert internal[0].source == "第一行"
    assert internal[0].translation == "第二行"


def test_document_from_srt(tmp_path):
    from src.ui.replay import document_from_srt

    srt = tmp_path / "ext.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n多行甲\n多行乙\n\n",
        encoding="utf-8",
    )
    doc = document_from_srt(srt)
    assert len(doc) == 1
    assert doc.get(1).source == "多行甲\n多行乙"
    assert doc.segment_at(1.5).id == 1


class FakePlayer:
    """假媒体时钟：记录 seek，不提供真实解码。"""

    def __init__(self):
        self.positions: list[int] = []
        self.stopped = False

    def setPosition(self, ms: int) -> None:
        self.positions.append(ms)

    def duration(self) -> int:
        return 0

    def stop(self) -> None:
        self.stopped = True


def test_replay_window_seek_and_highlight_with_fake_clock(qapp):
    """ReplayWindow 接线：句块 seek → setPosition；位置更新 → 高亮回调。"""
    from src.ui.i18n import set_language
    from src.ui.player import create_replay_window

    set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    highlights = []
    window = create_replay_window(_document(), on_seek_highlight=highlights.append)
    window.player = FakePlayer()  # 替换为假时钟

    assert window.seek_to_segment(2) is True
    assert window.player.positions == [2000]
    assert window.seek_to_segment(999) is False

    window._on_position(500)
    window._on_position(800)
    window._on_position(1500)  # 句间空白
    assert highlights == [1, None]
    window.close()


def test_replay_window_missing_media_shows_error(qapp, tmp_path):
    from src.ui.i18n import set_language
    from src.ui.player import create_replay_window

    set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    window = create_replay_window(_document())
    assert window.load_media(tmp_path / "不存在.mp4") is False
    assert window.error_label.text()  # 媒体加载错误有可见提示
    window.close()


def test_replay_window_close_stops_playback(qapp):
    """回归：关闭媒体窗口必须停止播放（否则音频在后台继续）。"""
    from src.ui.i18n import set_language
    from src.ui.player import create_replay_window

    set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    window = create_replay_window(_document())
    stopped = []
    window.player.stop = lambda: stopped.append(True)
    window.close()
    assert stopped == [True]


def test_back_button_jumps_to_current_in_replay(qapp):
    """回放模式的跟随按钮是「回到当前句」而非回到底部。"""
    from src.ui.subtitle_view import SubtitleView

    view = SubtitleView(replay_mode=True, read_only=True)
    view.resize(360, 120)
    view.show()
    assert "当前句" in view.back_to_bottom.text()

    doc = _document()
    for seg in doc.segments:
        view.upsert_segment(seg)
    # 扩充内容使可滚动
    from src.subtitle.document import Segment

    for i in range(4, 40):
        view.upsert_segment(Segment(id=i, start=i, end=i + 0.5, source=f"句{i}"))
    qapp.processEvents()
    view.set_current_segment(20)
    view.scroll_area.verticalScrollBar().setValue(0)  # 用户上翻
    qapp.processEvents()
    assert not view.following

    view.back_to_bottom.click()
    qapp.processEvents()
    assert view.following
    bar = view.scroll_area.verticalScrollBar()
    assert bar.value() != bar.maximum()  # 定位到第 20 句而非底部
    view.close()
