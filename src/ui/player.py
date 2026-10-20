"""媒体回放（M3 阶段 4）：QMediaPlayer 播放 + 媒体时钟高亮 + 单击 seek。

分层：
- ReplaySync：Qt 无关的同步逻辑（位置 → 当前句 ID、句 ID → seek 毫秒），
  offscreen 测试用假时钟驱动，不把模块可导入当作真实编解码验收。
- ReplayWindow：QMediaPlayer + QAudioOutput + QVideoWidget 的播放控制窗口。
- run_replay：由 replay controller 连接字幕悬浮窗（句块/语法卡）与播放窗口。
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..subtitle.document import SessionDocument

log = logging.getLogger(__name__)


class ReplaySync:
    """媒体位置 ↔ 字幕句段的同步逻辑（纯 Python，可脱离 Qt 测试）。"""

    def __init__(self, document: SessionDocument):
        self.document = document
        self.current_id: int | None = None

    def on_position(self, seconds: float) -> tuple[int | None, bool]:
        """位置更新：返回 (当前句 ID, 是否变化)。句间空白返回 (None, 是否清除)。"""
        segment = self.document.segment_at(seconds)
        new_id = segment.id if segment is not None else None
        changed = new_id != self.current_id
        self.current_id = new_id
        return new_id, changed

    def seek_position_ms(self, segment_id: int) -> int | None:
        """单击句块 seek 目标（毫秒）；未知 ID 返回 None。"""
        segment = self.document.get(segment_id)
        if segment is None:
            return None
        return round(segment.start * 1000)


def create_replay_window(document, on_seek_highlight=None):
    """构建回放窗口（延迟导入 PyQt6/QtMultimedia）。"""
    from PyQt6.QtCore import Qt, QUrl
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    from PyQt6.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
        QFileDialog,
    )

    from .i18n import tr

    class ReplayWindow(QWidget):
        """媒体播放控制窗口；字幕/语法卡由悬浮窗负责，经回调连接。"""

        def __init__(self):
            super().__init__()
            from PyQt6.QtCore import Qt as _Qt

            self.setAttribute(_Qt.WidgetAttribute.WA_QuitOnClose, False)
            self.setWindowTitle(tr("replay.title"))
            self.resize(720, 480)
            self._sync = ReplaySync(document)
            self._on_seek_highlight = on_seek_highlight

            self.player = QMediaPlayer(self)
            self.audio = QAudioOutput(self)
            self.player.setAudioOutput(self.audio)
            self.video = QVideoWidget(self)
            self.player.setVideoOutput(self.video)

            self.btn_play = QPushButton(tr("replay.play"), self)
            self.btn_play.clicked.connect(self.toggle_play)
            self.btn_open = QPushButton(tr("replay.open_media"), self)
            self.btn_open.clicked.connect(self.open_media)
            self.slider = QSlider(Qt.Orientation.Horizontal, self)
            self.slider.sliderMoved.connect(self._slider_seek)
            self.time_label = QLabel("00:00 / 00:00", self)
            self.error_label = QLabel("", self)
            self.error_label.setStyleSheet("color: #FF6B6B;")
            self.error_label.setWordWrap(True)

            controls = QHBoxLayout()
            controls.addWidget(self.btn_play)
            controls.addWidget(self.btn_open)
            controls.addWidget(self.slider, 1)
            controls.addWidget(self.time_label)

            layout = QVBoxLayout(self)
            layout.addWidget(self.video, 1)
            layout.addWidget(self.error_label)
            layout.addLayout(controls)

            self.player.positionChanged.connect(self._on_position)
            self.player.durationChanged.connect(self._on_duration)
            self.player.errorOccurred.connect(self._on_error)

        # ---- 播放控制 ----

        def load_media(self, path: str | Path) -> bool:
            """加载媒体文件；不存在直接报错显示，不崩溃。"""
            p = Path(path)
            if not p.is_file():
                self._show_error(tr("replay.media_missing") + f"：{p}")
                return False
            self.error_label.clear()
            self.player.setSource(QUrl.fromLocalFile(str(p)))
            return True

        def open_media(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, tr("replay.open_media"), "",
                "媒体文件 (*.mp4 *.mkv *.mp3 *.wav *.m4a *.flac);;所有文件 (*)",
            )
            if path:
                self.load_media(path)

        def toggle_play(self) -> None:
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
            else:
                self.player.play()

        def seek_to_segment(self, segment_id: int) -> bool:
            ms = self._sync.seek_position_ms(segment_id)
            if ms is None:
                return False
            self.player.setPosition(ms)
            return True

        # ---- 时钟同步 ----

        def _on_position(self, ms: int) -> None:
            seconds = ms / 1000
            segment_id, changed = self._sync.on_position(seconds)
            if changed and self._on_seek_highlight is not None:
                self._on_seek_highlight(segment_id)  # None = 句间空白，清除高亮
            if self.player.duration() > 0:
                if not self.slider.isSliderDown():
                    self.slider.setValue(ms)
                self.time_label.setText(
                    f"{_fmt_clock(seconds)} / {_fmt_clock(self.player.duration() / 1000)}"
                )

        def _on_duration(self, ms: int) -> None:
            self.slider.setRange(0, max(0, ms))

        def _slider_seek(self, value: int) -> None:
            self.player.setPosition(value)

        def _on_error(self, _error, message: str = "") -> None:
            self._show_error(message or tr("replay.media_error"))

        def _show_error(self, text: str) -> None:
            self.error_label.setText(text)
            log.warning("回放媒体错误：%s", text)

    return ReplayWindow()


def _fmt_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
