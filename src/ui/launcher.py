"""应用启动入口窗口（无 CLI 模式参数时显示）。

三种模式进入不同 UI：
- 实时（麦克风/系统音内录）→ 悬浮字幕窗
- 离线批处理（一个或多个媒体文件）→ 批处理进度窗口（batch_window.py）
- 回放（会话目录或外部 SRT + 可选媒体）→ 回放窗口 + 字幕悬浮窗

CLI 带模式参数（--source/--batch/--replay-session/--srt 等）时跳过启动器，
脚本化用法不受影响。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LaunchParams:
    """启动器收集的启动参数。"""

    mode: str = "realtime"          # realtime | batch | replay
    source_lang: str = ""           # 源语言包名（或 "auto" 自动识别）
    target_lang: str = ""           # 目标语言
    source: str = "mic"             # 实时：mic | loopback
    loopback_device: str = ""       # 实时：loopback 设备名关键词
    grammar: bool = False           # 自动后台语法分析
    debug: bool = False             # 完整 LLM 调试输出
    files: list[str] = field(default_factory=list)  # 批处理：媒体文件列表
    session_dir: str = ""           # 回放：会话目录
    media: str = ""                 # 回放：媒体文件
    srt: str = ""                   # 回放：外部 SRT


def _enum_loopback_devices() -> list[str]:
    """枚举可内录的输出设备名（仅 Windows；其他平台/无依赖时返回空列表）。"""
    try:
        from ..audio.loopback import list_loopback_devices

        return [d["name"] for d in list_loopback_devices()]
    except Exception:  # noqa: BLE001 —— 枚举失败不阻塞启动器
        return []


def create_launcher(profiles: list, *, default_profile: str = "", default_source: str = "mic"):
    """构建启动器对话框（延迟导入 PyQt6）。"""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QPushButton,
        QRadioButton,
        QVBoxLayout,
        QWidget,
    )

    from .i18n import tr

    class LauncherWindow(QDialog):
        def __init__(self):
            super().__init__()
            from PyQt6.QtCore import Qt as _Qt

            self.setAttribute(_Qt.WidgetAttribute.WA_QuitOnClose, False)
            self.setWindowTitle(tr("launcher.title"))
            self.resize(560, 480)
            self.result_params: LaunchParams | None = None

            layout = QVBoxLayout(self)

            # 模式选择
            mode_row = QHBoxLayout()
            self.rb_realtime = QRadioButton(tr("launcher.mode.realtime"))
            self.rb_batch = QRadioButton(tr("launcher.mode.batch"))
            self.rb_replay = QRadioButton(tr("launcher.mode.replay"))
            self.rb_realtime.setChecked(True)
            for rb in (self.rb_realtime, self.rb_batch, self.rb_replay):
                rb.toggled.connect(self._sync_mode_pages)
                mode_row.addWidget(rb)
            mode_row.addStretch(1)
            layout.addLayout(mode_row)

            form = QFormLayout()
            from ..profiles.loader import AUTO_LANG

            default_pack = next(
                (p for p in profiles if p.name == default_profile),
                profiles[0] if profiles else None,
            )
            # 源语言：自动识别 + 各语言包；目标语言：各语言包
            self.src_combo = QComboBox()
            self.src_combo.addItem(tr("overlay.lang.auto"), AUTO_LANG)
            for p in profiles:
                self.src_combo.addItem(p.display_name, p.language)
            idx = self.src_combo.findData(
                default_pack.source_lang if default_pack else ""
            )
            self.src_combo.setCurrentIndex(max(1, idx))  # 默认跟随档案源语言（非 auto）
            form.addRow(tr("launcher.source_lang"), self.src_combo)
            self.tgt_combo = QComboBox()
            for p in profiles:
                self.tgt_combo.addItem(p.display_name, p.language)
            idx = self.tgt_combo.findData(
                default_pack.target_lang if default_pack else ""
            )
            self.tgt_combo.setCurrentIndex(max(0, idx))
            form.addRow(tr("launcher.target_lang"), self.tgt_combo)
            layout.addLayout(form)

            # ---- 实时参数 ----
            self.page_realtime = QWidget(self)
            rt = QFormLayout(self.page_realtime)
            self.source_combo = QComboBox()
            self.source_combo.addItem(tr("launcher.source.mic"), "mic")
            self.source_combo.addItem(tr("launcher.source.loopback"), "loopback")
            idx = self.source_combo.findData(default_source)
            if idx >= 0:
                self.source_combo.setCurrentIndex(idx)
            rt.addRow(tr("launcher.source"), self.source_combo)
            # 内录设备：可编辑下拉框——下拉枚举可选，也可直接输入关键词
            self.loopback_edit = QComboBox()
            self.loopback_edit.setObjectName("loopbackCombo")
            self.loopback_edit.setEditable(True)
            self.loopback_edit.addItem("")  # 空 = 跟随系统默认输出设备
            for name in _enum_loopback_devices():
                self.loopback_edit.addItem(name)
            self.loopback_edit.lineEdit().setPlaceholderText(tr("launcher.loopback_hint"))
            rt.addRow(tr("launcher.loopback"), self.loopback_edit)
            self.grammar_check = QCheckBox(tr("launcher.grammar"))
            self.debug_check = QCheckBox(tr("launcher.debug"))
            rt.addRow(self.grammar_check)
            rt.addRow(self.debug_check)
            layout.addWidget(self.page_realtime)

            # ---- 批处理参数 ----
            self.page_batch = QWidget(self)
            bt = QVBoxLayout(self.page_batch)
            self.file_list = QListWidget()
            bt.addWidget(self.file_list, 1)
            file_btns = QHBoxLayout()
            btn_add = QPushButton(tr("launcher.files.add"))
            btn_remove = QPushButton(tr("launcher.files.remove"))
            btn_add.clicked.connect(self._add_files)
            btn_remove.clicked.connect(self._remove_file)
            file_btns.addWidget(btn_add)
            file_btns.addWidget(btn_remove)
            file_btns.addStretch(1)
            bt.addLayout(file_btns)
            layout.addWidget(self.page_batch)

            # ---- 回放参数 ----
            self.page_replay = QWidget(self)
            rp = QFormLayout(self.page_replay)
            self.session_edit = QLineEdit()
            self.media_edit = QLineEdit()
            self.srt_edit = QLineEdit()
            rp.addRow(
                tr("launcher.replay.session"),
                self._with_browse(self.session_edit, "dir"),
            )
            rp.addRow(
                tr("launcher.replay.media"),
                self._with_browse(self.media_edit, "media"),
            )
            rp.addRow(
                tr("launcher.replay.srt"), self._with_browse(self.srt_edit, "srt")
            )
            layout.addWidget(self.page_replay)

            self.error_label = QLabel("")
            self.error_label.setStyleSheet("color: #C62828;")
            layout.addWidget(self.error_label)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                self,
            )
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("launcher.start"))
            buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("launcher.quit"))
            buttons.accepted.connect(self._on_start)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

            self._sync_mode_pages()

        def _with_browse(self, edit: QLineEdit, kind: str) -> QWidget:
            container = QWidget(self)
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(edit, 1)
            btn = QPushButton(tr("launcher.browse"))
            btn.clicked.connect(lambda: self._browse(edit, kind))
            row.addWidget(btn)
            return container

        def _browse(self, edit: QLineEdit, kind: str) -> None:
            if kind == "dir":
                path = QFileDialog.getExistingDirectory(self, tr("launcher.replay.session"))
            elif kind == "srt":
                path, _ = QFileDialog.getOpenFileName(
                    self, tr("launcher.replay.srt"), "", "SRT (*.srt)"
                )
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, tr("launcher.replay.media"), "",
                    "媒体文件 (*.mp4 *.mkv *.mp3 *.wav *.m4a *.flac);;所有文件 (*)",
                )
            if path:
                edit.setText(path)

        def _add_files(self) -> None:
            paths, _ = QFileDialog.getOpenFileNames(
                self, tr("launcher.files.add"), "",
                "媒体文件 (*.mp4 *.mkv *.mp3 *.wav *.m4a *.flac);;所有文件 (*)",
            )
            for p in paths:
                self.file_list.addItem(p)

        def _remove_file(self) -> None:
            for item in self.file_list.selectedItems():
                self.file_list.takeItem(self.file_list.row(item))

        def _sync_mode_pages(self) -> None:
            self.page_realtime.setVisible(self.rb_realtime.isChecked())
            self.page_batch.setVisible(self.rb_batch.isChecked())
            self.page_replay.setVisible(self.rb_replay.isChecked())

        def params(self) -> LaunchParams:
            """当前表单内容（不校验）。"""
            mode = (
                "batch" if self.rb_batch.isChecked()
                else "replay" if self.rb_replay.isChecked()
                else "realtime"
            )
            return LaunchParams(
                mode=mode,
                source_lang=self.src_combo.currentData() or "",
                target_lang=self.tgt_combo.currentData() or "",
                source=self.source_combo.currentData() or "mic",
                loopback_device=self.loopback_edit.currentText().strip(),
                grammar=self.grammar_check.isChecked(),
                debug=self.debug_check.isChecked(),
                files=[
                    self.file_list.item(i).text()
                    for i in range(self.file_list.count())
                ],
                session_dir=self.session_edit.text().strip(),
                media=self.media_edit.text().strip(),
                srt=self.srt_edit.text().strip(),
            )

        def validate(self, p: LaunchParams) -> str:
            """返回错误文案；空串 = 合法。"""
            if p.mode == "batch" and not p.files:
                return tr("launcher.error.no_files")
            if p.mode == "replay" and not (p.session_dir or p.srt):
                return tr("launcher.error.no_replay_input")
            return ""

        def _on_start(self) -> None:
            p = self.params()
            error = self.validate(p)
            if error:
                self.error_label.setText(error)
                return
            self.result_params = p
            self.accept()

    return LauncherWindow()
