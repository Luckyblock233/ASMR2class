"""桌面悬浮字幕窗（M3 阶段 3 完整 A–D 栏）。

布局：A 控制栏 / B 模型与档案栏 / C 运行状态栏 / D 字幕区。
- OverlayWindow 为顶层可测试类；create_overlay() 保留旧信号兼容。
- 槽位状态来自 SettingsController，控件不自行猜测；换绑在 worker 线程执行。
- C 栏每秒读取 telemetry 快照与 Registry 状态快照，不做网络健康请求。
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass

from ..config import UIConfig

log = logging.getLogger(__name__)

DISPLAY_ORDERS = ("source_first", "translation_first", "source_only", "translation_only")

VRAM_WARN_FREE_MB = 2000  # 空闲显存低于该值时 C 栏橙色告警

# 阈值色与 health_color 统一由 appearance 提供（句内延迟小字同源）
from .appearance import HEALTH_CRIT, HEALTH_GOOD, HEALTH_WARN, health_color


def _provider_detail_text(p) -> str:
    """转发到 settings.provider_detail_text（唯一实现，避免两处漂移）。"""
    from .settings import provider_detail_text

    return provider_detail_text(p)


# 服务状态点统一语义：●运行中 ●降级运行 ●错误 ○停止
_SERVICE_STATUS = {
    "running": ("●", HEALTH_GOOD, "overlay.status.running"),
    "degraded": ("●", HEALTH_WARN, "overlay.status.degraded"),
    "error": ("●", HEALTH_CRIT, "overlay.status.error"),
    "stopped": ("○", "#808080", "overlay.status.stopped"),
}


@dataclass
class OverlayLine:
    start: float
    end: float
    source: str
    translation: str | None = None


class SubtitleBuffer:
    """保留给既有调用方的纯文本数据层；新窗口使用 SubtitleView。"""

    def __init__(self, display_order: str = "source_first", capacity: int = 50):
        self.display_order = display_order if display_order in DISPLAY_ORDERS else "source_first"
        self.lines: deque[OverlayLine] = deque(maxlen=capacity)

    def add(self, line: OverlayLine) -> None:
        self.lines.append(line)

    def render(self) -> str:
        out = []
        for ln in self.lines:
            src, tgt = ln.source, ln.translation or ""
            if self.display_order == "source_only":
                rows = [src]
            elif self.display_order == "translation_only":
                rows = [tgt] if tgt else [src]
            elif self.display_order == "translation_first":
                rows = [tgt, src] if tgt else [src]
            else:
                rows = [src, tgt] if tgt else [src]
            out.append("\n".join(rows))
        return "\n\n".join(out)


class OverlayUnavailableError(RuntimeError):
    pass


def fix_app_font(app) -> None:
    """Windows + QSS px 字号的已知告警（QFont::setPointSize(-1)）防御。

    默认应用字体若只有 pixelSize（pointSize=-1），样式表引擎换算时会告警。
    给一个带有效 pointSize 的默认字体即可消除；对显示效果无影响。
    """
    font = app.font()
    if font.pointSize() <= 0:
        from PyQt6.QtGui import QFont

        app.setFont(QFont(font.family(), 9))


def _import_qt():
    try:
        from PyQt6.QtCore import Qt, QTimer, pyqtSignal
        from PyQt6.QtGui import QAction, QIcon
        from PyQt6.QtWidgets import (
            QApplication,
            QComboBox,
            QHBoxLayout,
            QLabel,
            QMenu,
            QMessageBox,
            QProgressBar,
            QPushButton,
            QSizeGrip,
            QStyle,
            QSystemTrayIcon,
            QToolButton,
            QVBoxLayout,
            QWidget,
        )
        from .grammar_card import GrammarCardPopup
        from .settings import ApplyResult
        from .subtitle_view import SubtitleView
    except ImportError as e:
        raise OverlayUnavailableError(f"PyQt6 不可用：{e}") from e
    return locals()


def make_overlay_window_class():
    """构建 OverlayWindow 类（延迟导入 PyQt6，确保 QApplication 已存在）。"""
    qt = _import_qt()
    Qt = qt["Qt"]
    QTimer = qt["QTimer"]
    pyqtSignal = qt["pyqtSignal"]
    QAction = qt["QAction"]
    QIcon = qt["QIcon"]
    QApplication = qt["QApplication"]
    QComboBox = qt["QComboBox"]
    QHBoxLayout = qt["QHBoxLayout"]
    QLabel = qt["QLabel"]
    QMenu = qt["QMenu"]
    QMessageBox = qt["QMessageBox"]
    QPushButton = qt["QPushButton"]
    QProgressBar = qt["QProgressBar"]
    QSizeGrip = qt["QSizeGrip"]
    QStyle = qt["QStyle"]
    QSystemTrayIcon = qt["QSystemTrayIcon"]
    QToolButton = qt["QToolButton"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]
    GrammarCardPopup = qt["GrammarCardPopup"]
    SubtitleView = qt["SubtitleView"]

    from ..subtitle.document import Segment
    from .appearance import AppearanceSpec
    from .i18n import tr

    class OverlayWindow(QWidget):
        """完整悬浮窗；read_only=replay 预览，replay_mode=回放（单击 seek）。"""

        add_sentence = pyqtSignal(float, float, str, object)  # 旧接口兼容
        upsert_segment = pyqtSignal(object)
        analysis_ready = pyqtSignal(object)
        analysis_requested = pyqtSignal(int)
        set_status = pyqtSignal(str)
        pause_toggled = pyqtSignal(bool)      # A 栏 运行/暂停（True = 已暂停）
        seek_requested = pyqtSignal(int)      # 回放句块 seek
        grammar_requested = pyqtSignal(int)   # 回放双击语法卡
        pair_change_requested = pyqtSignal(str, str)  # B 栏语言对切换（源, 目标；源可为 auto）
        _op_done = pyqtSignal(object)         # worker → GUI 回调

        def __init__(
            self,
            display_order: str = "source_first",
            opacity: float = 0.85,
            profile=None,
            read_only: bool = False,
            replay_mode: bool = False,
            ui: UIConfig | None = None,
            controller=None,
            metrics=None,
            sampler=None,
            confirm_exit: bool = False,
            profiles=None,
        ):
            super().__init__()
            self.profile = profile
            self.controller = controller
            self.metrics = metrics
            self.sampler = sampler
            self.confirm_exit = confirm_exit
            self._drag_pos = None
            self._click_through = False
            self._opacity = max(0.2, min(1.0, opacity))
            self._legacy_id = 0
            self._paused = False
            self._settings_dialog = None
            self._preview_active = False
            self._appearance_saved = None
            self.grammar_card = None
            self._ui = ui or UIConfig(opacity=opacity)
            self._appearance = AppearanceSpec.from_ui(self._ui)

            self.setWindowTitle("ASR2class 字幕")
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.resize(self._ui.window_width, self._ui.window_height)
            self.setMinimumSize(320, 160)

            # ---- D 区字幕 ----
            self.subtitle_view = SubtitleView(
                display_order=display_order,
                source_lang=profile.source_lang if profile is not None else "",
                read_only=read_only,
                replay_mode=replay_mode,
                auto_scroll=self._ui.auto_scroll,
                capacity=None if replay_mode else self._ui.view_capacity,
                appearance=self._appearance,
                parent=self,
            )
            self.text = self.subtitle_view  # 旧属性名保留
            # 主程序只在语法槽成功绑定后开启手动分析，避免发出无人接收的请求。
            self.subtitle_view.set_grammar_enabled(False)
            self.subtitle_view.sentence_activated.connect(self._show_grammar)
            self.subtitle_view.analysis_requested.connect(self.analysis_requested.emit)
            self.subtitle_view.seek_requested.connect(self.seek_requested.emit)
            self.subtitle_view.grammar_requested.connect(self.grammar_requested.emit)
            self.subtitle_view.appearance_requested.connect(self.open_settings)

            # ---- A 栏：控制条 ----
            self.barA = QWidget(self)
            self.barA.setObjectName("barA")
            self.barA.setStyleSheet("QWidget#barA { background: transparent; }")
            bar = QHBoxLayout(self.barA)
            bar.setContentsMargins(2, 2, 2, 2)
            # 按钮/下拉文字颜色由 apply_appearance 的主题样式表统一着色，
            # 此处不再逐控件硬编码（否则主题切换时文字不换色）

            self.btn_menu = QToolButton(self.barA)
            self.btn_menu.setText("☰")
            self.btn_menu.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            self.main_menu = QMenu(self.btn_menu)
            self.act_through = QAction(tr("overlay.menu.click_through"), self)
            self.act_through.setCheckable(True)
            self.act_through.triggered.connect(
                lambda checked: self.set_click_through(checked)
            )
            self.act_ontop = QAction(tr("overlay.menu.always_on_top"), self)
            self.act_ontop.setCheckable(True)
            self.act_ontop.setChecked(True)
            self.act_ontop.triggered.connect(self._toggle_on_top)
            self.act_autoscroll = QAction(tr("overlay.menu.auto_scroll"), self)
            self.act_autoscroll.setCheckable(True)
            self.act_autoscroll.setChecked(self._ui.auto_scroll)
            self.act_autoscroll.triggered.connect(
                lambda checked: self.subtitle_view.set_auto_scroll(checked)
            )
            self.act_compact = QAction(tr("overlay.menu.compact"), self)
            self.act_compact.setCheckable(True)
            self.act_compact.setChecked(self._ui.compact_mode)
            self.act_compact.triggered.connect(self._toggle_compact)
            self.act_subtitles = QAction(tr("overlay.menu.subtitles"), self)
            self.act_subtitles.setCheckable(True)
            self.act_subtitles.setChecked(True)
            self.act_subtitles.triggered.connect(
                lambda checked: self.subtitle_view.setVisible(checked)
            )
            for a in (
                self.act_through,
                self.act_ontop,
                self.act_autoscroll,
                self.act_compact,
                self.act_subtitles,
            ):
                self.main_menu.addAction(a)
            self.btn_menu.setMenu(self.main_menu)

            self.status_label = QPushButton("", self.barA)
            self.status_label.setFlat(True)
            self.status_label.setStyleSheet("color: #9f9; font-size: 10px;")

            self.btn_pause = QPushButton(tr("overlay.pause"), self.barA)
            self.btn_pause.setCheckable(True)
            self.btn_pause.toggled.connect(self._on_pause_toggled)
            self.btn_clear = QPushButton(tr("overlay.clear"), self.barA)
            self.btn_clear.clicked.connect(self._on_clear_clicked)
            self.btn_settings = QPushButton(tr("overlay.settings"), self.barA)
            self.btn_settings.clicked.connect(self.open_settings)
            self.btn_close = QPushButton("×", self.barA)
            self.btn_close.clicked.connect(self.close)
            if replay_mode:
                self.btn_pause.setVisible(False)  # 回放暂停由播放器控制

            bar.addWidget(self.btn_menu)
            bar.addWidget(self.status_label)
            bar.addStretch(1)
            bar.addWidget(self.btn_pause)
            bar.addWidget(self.btn_clear)
            bar.addWidget(self.btn_settings)
            bar.addWidget(self.btn_close)

            # ---- B 栏：语言对与槽位（每项信息独立一行）----
            self.barB = QWidget(self)
            self.barB.setObjectName("barB")
            self.barB.setStyleSheet("QWidget#barB { background: transparent; }")
            bar_b = QVBoxLayout(self.barB)
            bar_b.setContentsMargins(2, 0, 2, 0)
            bar_b.setSpacing(2)
            from ..profiles.loader import AUTO_LANG

            self._profiles = list(profiles or [])
            # 源语言：自动识别 + 各语言包；目标语言：各语言包
            src_row = QHBoxLayout()
            src_row.setContentsMargins(0, 0, 0, 0)
            src_label = QLabel(tr("overlay.source_lang"), self.barB)
            src_label.setFixedWidth(64)
            src_row.addWidget(src_label)
            self.src_combo = QComboBox(self.barB)
            self.src_combo.setObjectName("srcCombo")
            self.src_combo.setMinimumWidth(160)
            self.src_combo.addItem(tr("overlay.lang.auto"), AUTO_LANG)
            for p in self._profiles:
                self.src_combo.addItem(p.display_name, p.language)
            src_row.addWidget(self.src_combo, 1)
            bar_b.addLayout(src_row)

            tgt_row = QHBoxLayout()
            tgt_row.setContentsMargins(0, 0, 0, 0)
            tgt_label = QLabel(tr("overlay.target_lang"), self.barB)
            tgt_label.setFixedWidth(64)
            tgt_row.addWidget(tgt_label)
            self.tgt_combo = QComboBox(self.barB)
            self.tgt_combo.setObjectName("tgtCombo")
            self.tgt_combo.setMinimumWidth(160)
            for p in self._profiles:
                self.tgt_combo.addItem(p.display_name, p.language)
            tgt_row.addWidget(self.tgt_combo, 1)
            bar_b.addLayout(tgt_row)

            if profile is not None:
                idx = self.src_combo.findData(profile.source_lang)
                self.src_combo.setCurrentIndex(max(0, idx))
                idx = self.tgt_combo.findData(profile.target_lang)
                self.tgt_combo.setCurrentIndex(max(0, idx))
            self.src_combo.activated.connect(self._on_pair_combo)
            self.tgt_combo.activated.connect(self._on_pair_combo)
            enabled = bool(self._profiles) and not replay_mode
            self.src_combo.setEnabled(enabled)
            self.tgt_combo.setEnabled(enabled)
            self.slot_combos: dict = {}
            if self.controller is not None:
                from ..nlp.providers import SLOT_GRAMMAR, SLOT_NAMES, SLOT_TRANSLATION

                for slot in (SLOT_TRANSLATION, SLOT_GRAMMAR):
                    row = QHBoxLayout()
                    row.setContentsMargins(0, 0, 0, 0)
                    label = QLabel(SLOT_NAMES.get(slot, slot), self.barB)
                    label.setFixedWidth(64)  # 标签固定分区，不被下拉框遮挡
                    row.addWidget(label)
                    combo = QComboBox(self.barB)
                    combo.setObjectName(f"slotCombo_{slot}")
                    combo.setMinimumWidth(160)
                    row.addWidget(combo, 1)
                    combo.activated.connect(
                        lambda _i, s=slot: self._apply_slot_binding(s)
                    )
                    self.slot_combos[slot] = combo
                    bar_b.addLayout(row)
                self.refresh_slots()
            else:
                self.slot_status_label = QLabel("", self.barB)
                bar_b.addWidget(self.slot_status_label)

            # ---- C 栏：运行状态（参考 LiveTrans 布局）----
            self.barC = QWidget(self)
            self.barC.setObjectName("barC")
            self.barC.setStyleSheet("QWidget#barC { background: transparent; }")
            bar_c = QVBoxLayout(self.barC)
            bar_c.setContentsMargins(2, 0, 2, 0)
            bar_c.setSpacing(1)
            c_style = "color: #9a9a9a; font-size: 12px; background: transparent;"

            # 行 1：RMS/VAD 横向进度条（各占半行，宽度自适应）
            meters_row = QHBoxLayout()
            meters_row.setContentsMargins(0, 0, 0, 0)
            meters_row.setSpacing(6)
            self.rms_bar = QProgressBar(self.barC)
            self.rms_bar.setObjectName("rmsBar")
            self.vad_bar = QProgressBar(self.barC)
            self.vad_bar.setObjectName("vadBar")
            for name, bar_w in (("RMS", self.rms_bar), ("VAD", self.vad_bar)):
                lbl = QLabel(name, self.barC)
                lbl.setStyleSheet(c_style)
                meters_row.addWidget(lbl)
                bar_w.setRange(0, 100)
                bar_w.setTextVisible(True)
                bar_w.setFormat("%p%")
                bar_w.setFixedHeight(14)
                meters_row.addWidget(bar_w, 1)
            bar_c.addLayout(meters_row)

            # 行 2：监控信息整行平铺（space-between，占满整行宽度而非左对齐）
            self.monitor_row = QWidget(self.barC)
            self.monitor_row.setObjectName("monitorRow")
            self.monitor_layout = QHBoxLayout(self.monitor_row)
            self.monitor_layout.setContentsMargins(0, 0, 0, 0)
            self.monitor_layout.setSpacing(4)
            bar_c.addWidget(self.monitor_row)
            self._monitor_parts: list = []
            self._asr_info_text = ""

            # 行 3：服务状态点：●运行中 ●降级运行 ●错误 ○停止；点击查看详情/最近错误
            self.services_row = QWidget(self.barC)
            services_layout = QHBoxLayout(self.services_row)
            services_layout.setContentsMargins(0, 0, 0, 0)
            services_layout.setSpacing(8)
            legend = QLabel(tr("overlay.services"), self.services_row)
            legend.setStyleSheet(c_style)
            legend.setToolTip(tr("overlay.status.legend"))
            services_layout.addWidget(legend)
            self._service_buttons: dict = {}
            services_layout.addStretch(1)
            bar_c.addWidget(self.services_row)

            # ---- 组装 ----
            self.central = QWidget(self)
            self.central.setObjectName("centralHost")
            layout = QVBoxLayout(self.central)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(2)
            layout.addWidget(self.barA)
            layout.addWidget(self.barB)
            layout.addWidget(self.barC)
            layout.addWidget(self.subtitle_view, 1)
            bottom_row = QHBoxLayout()
            bottom_row.setContentsMargins(0, 0, 0, 0)
            bottom_row.addStretch(1)
            self.size_grip = QSizeGrip(self.central)
            bottom_row.addWidget(self.size_grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
            layout.addLayout(bottom_row)

            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.addWidget(self.central)

            self.apply_appearance(self._ui, resize=False)

            # ---- 托盘 ----
            style = QApplication.style()
            icon = (
                style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
                if style is not None
                else QIcon()
            )
            self.tray = QSystemTrayIcon(icon, self)
            menu = QMenu(self)
            self.tray_menu = menu
            act_restore = QAction(tr("overlay.tray.restore"), self)
            act_restore.triggered.connect(lambda: self.set_click_through(False))
            act_quit = QAction(tr("overlay.tray.quit"), self)
            act_quit.triggered.connect(self.close)
            menu.addAction(act_restore)
            menu.addAction(act_quit)
            menu.setStyleSheet(self._appearance.menu_stylesheet)
            self.tray.setContextMenu(menu)
            self.tray.setToolTip("ASR2class 字幕悬浮窗")
            self.tray.show()

            # ---- 信号接线 ----
            self.add_sentence.connect(self._on_legacy_sentence)
            self.upsert_segment.connect(self.subtitle_view.upsert_segment)
            self.analysis_ready.connect(self.subtitle_view.upsert_segment)
            self.set_status.connect(self._on_status)
            self._op_done.connect(lambda cb: cb())

            # ---- C 栏每秒刷新 ----
            self.metrics_timer = QTimer(self)
            self.metrics_timer.setInterval(1000)
            self.metrics_timer.timeout.connect(self._refresh_metrics)
            if self.metrics is not None:
                self.metrics_timer.start()

        # ==================== A 栏 ====================

        def _on_pause_toggled(self, paused: bool) -> None:
            self._paused = paused
            self.btn_pause.setText(tr("overlay.resume") if paused else tr("overlay.pause"))
            self.pause_toggled.emit(paused)

        def _on_clear_clicked(self) -> None:
            """清空显示：只清当前控件，不动文档/记忆/落盘文件。"""
            if self.confirm_exit:
                answer = QMessageBox.question(
                    self, tr("overlay.clear"), tr("overlay.clear_confirm")
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            self.subtitle_view.clear()

        def open_settings(self) -> None:
            """打开设置页（A 栏按钮与句块右键「外观设置…」同一入口）。

            外观控件变化即时预览；关闭时若有未保存的预览则还原；
            重新打开时表单从已保存配置重载（不残留上次未应用的预览值）。
            """
            if self.controller is None:
                return
            from .settings import create_settings_dialog

            if self._settings_dialog is None:
                self._appearance_saved = self._ui
                self._preview_active = False
                self._settings_dialog = create_settings_dialog(
                    self.controller,
                    on_appearance_applied=self._on_appearance_applied,
                    on_appearance_preview=self._preview_appearance,
                )
                self._settings_dialog.finished.connect(self._on_settings_closed)
            else:
                # 重开：表单回到已保存配置；槽位状态同步最新
                saved = self._appearance_saved or self._ui
                self._settings_dialog._reload_appearance_form(saved)
                self._settings_dialog._refresh_slot_combos()
                self._settings_dialog._reload_memory_editor()
            self._settings_dialog.show()
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()

        def _preview_appearance(self, ui: UIConfig) -> None:
            self._preview_active = True
            self.apply_appearance(ui, resize=False)

        def _on_appearance_applied(self, ui: UIConfig) -> None:
            self._preview_active = False
            self._appearance_saved = ui
            self.apply_appearance(ui, resize=False)

        def _on_settings_closed(self, _result: int) -> None:
            if self._preview_active and self._appearance_saved is not None:
                self._preview_active = False
                self.apply_appearance(self._appearance_saved, resize=False)

        def set_click_through(self, on: bool) -> None:
            self._click_through = on
            self.act_through.setChecked(on)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, on)
            for b in (self.btn_menu, self.btn_pause, self.btn_clear, self.btn_settings, self.btn_close):
                b.setVisible(not on)
            self.show()
            self.tray.showMessage(
                "ASR2class",
                tr("overlay.through_on") if on else tr("overlay.through_off"),
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )

        def _toggle_on_top(self, on: bool) -> None:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
            self.show()

        def _toggle_compact(self, on: bool) -> None:
            self.barB.setVisible(not on and self._ui.show_bar_b)
            self.barC.setVisible(not on and self._ui.show_bar_c)

        # ==================== 语言对切换（§4.2 会话滚动）====================

        def _on_pair_combo(self, _index: int) -> None:
            source = self.src_combo.currentData()
            target = self.tgt_combo.currentData()
            if (
                not source
                or not target
                or (
                    self.profile is not None
                    and source == self.profile.source_lang
                    and target == self.profile.target_lang
                )
            ):
                return
            # 切换期间禁点，防止并发滚动；完成后由 apply_profile/notify_profile_error 恢复
            self.src_combo.setEnabled(False)
            self.tgt_combo.setEnabled(False)
            self.pair_change_requested.emit(source, target)

        def apply_profile(self, profile, merged_ui: UIConfig | None = None, notice: str = "") -> None:
            """切换完成后的 GUI 侧更新：新档案、新视图、系统提示行、槽位刷新。"""
            self.profile = profile
            for combo, lang in ((self.src_combo, profile.source_lang),
                                (self.tgt_combo, profile.target_lang)):
                index = combo.findData(lang)
                combo.blockSignals(True)
                combo.setCurrentIndex(max(0, index))
                combo.blockSignals(False)
                combo.setEnabled(True)
            if merged_ui is not None:
                self.apply_appearance(merged_ui, resize=False)
            self.subtitle_view.source_lang = profile.source_lang
            self.subtitle_view.set_display_order(
                profile.subtitle.get("display_order", "source_first")
            )
            self.subtitle_view.clear()  # 新会话新视图；旧会话已落盘
            if notice:
                self.subtitle_view.add_system_message(notice)
            if self.controller is not None:
                self.controller.set_profile(profile)
                self.refresh_slots()

        def notify_profile_error(self, _what: str, error: str) -> None:
            """切换失败：恢复下拉状态并提示，不改变当前会话。"""
            if self.profile is not None:
                for combo, lang in ((self.src_combo, self.profile.source_lang),
                                    (self.tgt_combo, self.profile.target_lang)):
                    index = combo.findData(lang)
                    if index >= 0:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(index)
                        combo.blockSignals(False)
            self.src_combo.setEnabled(True)
            self.tgt_combo.setEnabled(True)
            self.tray.showMessage(
                "ASR2class",
                tr("overlay.profile_switch_failed") + f"：{error}",
                QSystemTrayIcon.MessageIcon.Warning,
                3000,
            )

        def post_to_gui(self, fn) -> None:
            """worker 线程安全地把操作排入 GUI 线程。"""
            self._op_done.emit(fn)

        # ==================== B 栏 ====================

        def refresh_slots(self) -> None:
            """从控制器刷新槽位下拉（能力过滤 + 档案覆盖锁定 + applying 禁点）。

            主行只显示模型名；id/端点/量化等细节放条目 tooltip。
            """
            if self.controller is None:
                return
            for slot, combo in self.slot_combos.items():
                view = self.controller.slot_view(slot)
                combo.blockSignals(True)
                combo.clear()
                combo.addItem(tr("settings.slots.unbound"), "")
                for p in self.controller.candidates(slot):
                    combo.addItem(p.name, p.id)
                    combo.setItemData(
                        combo.count() - 1,
                        _provider_detail_text(p),
                        Qt.ItemDataRole.ToolTipRole,
                    )
                index = combo.findData(view.effective_id)
                combo.setCurrentIndex(max(0, index))
                current = self.controller.registry.config(view.effective_id) if view.effective_id else None
                combo.setToolTip(
                    _provider_detail_text(current) if current is not None else view.status
                )
                combo.setEnabled(view.editable)
                if view.overridden_by_profile:
                    combo.setToolTip(
                        (combo.toolTip() + "\n" if combo.toolTip() else "")
                        + tr("settings.slots.profile_override")
                    )
                combo.blockSignals(False)

        def _apply_slot_binding(self, slot: str) -> None:
            """B 栏换绑：worker 线程健康检查 + 事务，applying 期间禁点。"""
            combo = self.slot_combos[slot]
            pid = combo.currentData() or ""
            for c in self.slot_combos.values():
                c.setEnabled(False)

            def work():
                return self.controller.apply_binding(
                    slot, pid, health_check=self.controller.registry.check_health
                )

            def done(result):
                self.refresh_slots()
                self.tray.showMessage("ASR2class", result.text(),
                                      QSystemTrayIcon.MessageIcon.Information, 2000)

            self._run_async(work, done)

        def _run_async(self, fn, on_done) -> None:
            def target():
                from .settings import ApplyResult

                try:
                    result = fn()
                except Exception as e:  # noqa: BLE001
                    log.debug("悬浮窗异步操作异常", exc_info=True)
                    result = ApplyResult(False, "settings.msg.internal_error", str(e))
                self._op_done.emit(lambda: on_done(result))

            threading.Thread(target=target, name="overlay-op", daemon=True).start()

        # ==================== C 栏 ====================

        def set_asr_info(self, text: str) -> None:
            """ASR 引擎/设备信息（监控行首段）。"""
            self._asr_info_text = text

        def _refresh_metrics(self) -> None:
            if not self.barC.isVisible():
                return
            system = self.sampler.sample() if self.sampler is not None else None
            snap = self.metrics.snapshot(system=system) if self.metrics is not None else None
            if snap is None:
                return
            # 行 1：RMS/VAD 横向进度条
            self.rms_bar.setValue(min(100, round(snap.rms * 250)))
            self.vad_bar.setValue(100 if snap.vad_active else 0)

            # 行 2：监控信息整行平铺（引擎 | 资源 | 计数；延迟已移到每句行尾）
            s = snap.system
            ui = self._ui
            vram_warn = s.vram_free_mb is not None and s.vram_free_mb < VRAM_WARN_FREE_MB
            meta = self._appearance.theme.meta_color
            parts: list[tuple[str, str]] = []
            if self._asr_info_text:
                parts.append((self._asr_info_text, meta))
            if s.cpu_percent is not None:
                parts.append((f"CPU {s.cpu_percent:.0f}%", meta))
            if s.ram_percent is not None:
                parts.append((
                    f"RAM {s.ram_percent:.0f}%",
                    HEALTH_WARN if s.ram_percent >= ui.ram_warn_percent else meta,
                ))
            if s.gpu_percent is not None:
                parts.append((f"GPU {s.gpu_percent:.0f}%", meta))
            if s.vram_free_mb is not None:
                parts.append((
                    f"VRAM {s.vram_free_mb:.0f}MB",
                    HEALTH_WARN if vram_warn else meta,
                ))
            parts.append((f"句 {snap.asr_count} · 译 {snap.translate_count}", meta))
            if snap.translate_failed:
                parts.append((f"失败 {snap.translate_failed}", HEALTH_CRIT))
            self._set_monitor_parts(parts)
            self.monitor_row.setToolTip(tr("overlay.vram_warn") if vram_warn else "")
            # 行 3：服务状态点：统一 ●运行中 ●降级运行 ●错误 ○停止；与设置页共用 Registry 快照
            if self.controller is not None:
                self._refresh_service_dots()

        def _set_monitor_parts(self, parts: list) -> None:
            """监控行整行平铺：部件标签之间插入 stretch，两端顶格、中间均分。"""
            while self.monitor_layout.count():
                item = self.monitor_layout.takeAt(0)
                if item.widget() is not None:
                    item.widget().deleteLater()
            self._monitor_parts = []
            for i, (text, color) in enumerate(parts):
                if i > 0:
                    self.monitor_layout.addStretch(1)
                lbl = QLabel(text, self.monitor_row)
                lbl.setStyleSheet(
                    f"color: {color}; font-size: 12px; background: transparent;"
                )
                self.monitor_layout.addWidget(lbl)
                self._monitor_parts.append(lbl)

        def monitor_text(self) -> str:
            """监控行纯文本（测试/诊断用）。"""
            return " ｜ ".join(lbl.text() for lbl in self._monitor_parts)

        def _refresh_service_dots(self) -> None:
            snapshot = self.controller.registry.snapshot()
            for p in self.controller.cfg.providers:
                info = snapshot.get(p.id)
                status = info["status"] if info else "stopped"
                dot, color, label_key = _SERVICE_STATUS[status]
                button = self._service_buttons.get(p.id)
                if button is None:
                    button = QToolButton(self.services_row)
                    button.setObjectName(f"serviceDot_{p.id}")
                    button.setCursor(Qt.CursorShape.PointingHandCursor)
                    button.clicked.connect(
                        lambda _checked=False, pid=p.id: self._show_service_detail(pid)
                    )
                    layout = self.services_row.layout()
                    layout.insertWidget(layout.count() - 1, button)
                    self._service_buttons[p.id] = button
                button.setText(f"{dot} {p.name}")
                button.setStyleSheet(
                    f"QToolButton {{ color: {color}; font-size: 11px; border: none;"
                    " background: transparent; padding: 0px; }"
                )
                tip = f"{tr(label_key)} · {p.id}"
                if info and info.get("last_error"):
                    tip += f"\n{tr('overlay.status.last_error')}: {info['last_error']}"
                tip += "\n" + tr("overlay.status.click_detail")
                button.setToolTip(tip)
            self.services_row.setVisible(bool(self.controller.cfg.providers))

        def _show_service_detail(self, provider_id: str) -> None:
            """状态点点击：查看状态详情与最近错误（日志口径只含 id/model，不含密钥）。"""
            p = next(
                (x for x in self.controller.cfg.providers if x.id == provider_id), None
            )
            if p is None:
                return
            info = self.controller.registry.snapshot().get(provider_id)
            status = info["status"] if info else "stopped"
            _dot, _color, label_key = _SERVICE_STATUS[status]
            lines = [
                f"{p.name}（{p.id}）",
                f"{tr('settings.providers.field.type')}: {p.type}",
                f"model: {p.model}",
                f"endpoint: {p.base_url}",
                f"{tr('overlay.status.state')}: {tr(label_key)}",
            ]
            if info:
                lines.append(
                    f"{tr('overlay.status.active_calls')}: {info['active_calls']}"
                    f"（generation {info['generation']}）"
                )
                if info.get("last_error"):
                    lines.append(f"{tr('overlay.status.last_error')}: {info['last_error']}")
            QMessageBox.information(self, tr("overlay.status.detail_title"), "\n".join(lines))

        # ==================== 外观 ====================

        def apply_appearance(self, ui: UIConfig, *, resize: bool = True) -> None:
            """集中外观入口：设置页与右键快捷入口共用。"""
            self._ui = ui
            self._appearance = AppearanceSpec.from_ui(ui)
            theme = self._appearance.theme
            self.central.setStyleSheet(
                f"QWidget#centralHost {{ background: {self._appearance.background_rgba};"
                " border-radius: 8px; }"
                + self._appearance.bars_stylesheet  # A/B/C 栏控件随主题着色
            )
            self.main_menu.setStyleSheet(self._appearance.menu_stylesheet)
            if getattr(self, "tray_menu", None) is not None:
                self.tray_menu.setStyleSheet(self._appearance.menu_stylesheet)
            # 次要信息（C 栏监控行/服务图例）随主题更新：重绘监控部件
            c_style = (
                f"color: {theme.meta_color}; font-size: 12px; background: transparent;"
            )
            legend = self.services_row.findChild(QLabel)
            if legend is not None:
                legend.setStyleSheet(c_style)
            if self.metrics is not None:
                self._refresh_metrics()  # 监控行部件按新主题重建
            self.subtitle_view.set_appearance(self._appearance)
            self.subtitle_view.set_auto_scroll(ui.auto_scroll)
            self.act_autoscroll.setChecked(ui.auto_scroll)
            self.act_compact.setChecked(ui.compact_mode)
            self.barA.setVisible(ui.show_bar_a)
            self.barB.setVisible(ui.show_bar_b and not ui.compact_mode)
            self.barC.setVisible(ui.show_bar_c and not ui.compact_mode)
            if resize:
                self.resize(ui.window_width, ui.window_height)

        # ==================== 既有行为 ====================

        def _on_legacy_sentence(self, start: float, end: float, source: str, translation) -> None:
            self._legacy_id += 1
            self.subtitle_view.upsert_segment(
                Segment(start, end, source, translation, id=self._legacy_id)
            )

        def _show_grammar(self, segment: Segment) -> None:
            if self.profile is None or not segment.has_analysis:
                return
            if self.grammar_card is None:
                self.grammar_card = GrammarCardPopup()
            self.grammar_card.show_segment(segment, self.profile, anchor=self.subtitle_view)

        def show_grammar(self, segment) -> None:
            """公开入口：回放双击手势按 ID 取到快照后调这里。"""
            if segment is not None:
                self._show_grammar(segment)

        def add_menu_action(self, text: str, fn):
            """向主菜单追加入口（如回放的「打开媒体…」）；返回 QAction。"""
            action = QAction(text, self)
            action.triggered.connect(fn)
            self.main_menu.addAction(action)
            return action

        def set_grammar_enabled(self, enabled: bool) -> None:
            self.subtitle_view.set_grammar_enabled(enabled)

        def _on_status(self, status: str) -> None:
            online = status == "ONLINE"
            theme = self._appearance.theme
            self.status_label.setText("● " + status)
            self.status_label.setStyleSheet(
                f"color: {theme.status_ok if online else theme.status_err}; font-size: 10px;"
            )

        def _adjust_opacity(self, delta: float) -> None:
            self._opacity = max(0.2, min(1.0, self._opacity + delta))
            self.setWindowOpacity(self._opacity)

        def mousePressEvent(self, event):
            """仅 A 栏空白处可拖动窗口。"""
            if event.button() == Qt.MouseButton.LeftButton:
                pos = event.position().toPoint()
                child = self.childAt(pos)
                on_bar = self.barA.geometry().contains(
                    self.barA.mapFrom(self, pos)
                ) or self.barA.rect().contains(self.barA.mapFrom(self, pos))
                if self.barA.isVisible() and on_bar and (
                    child is None or child is self.status_label or child is self.barA
                ):
                    self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                    event.accept()
                    return

        def mouseMoveEvent(self, event):
            if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self.move(event.globalPosition().toPoint() - self._drag_pos)
                event.accept()

        def mouseReleaseEvent(self, event):
            self._drag_pos = None

        def closeEvent(self, event):
            if self.confirm_exit:
                answer = QMessageBox.question(
                    self, "ASR2class", tr("overlay.exit_confirm")
                )
                if answer != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
            if self.grammar_card is not None:
                self.grammar_card.close()
            if self._settings_dialog is not None:
                self._settings_dialog.close()
            self.tray.hide()
            QApplication.quit()
            super().closeEvent(event)

    return OverlayWindow


_OVERLAY_CLASS = None
_APP_REF = None  # 保持 QApplication 引用，防止被 GC 后 Qt 崩溃


def create_overlay(
    display_order: str = "source_first",
    opacity: float = 0.85,
    profile=None,
    read_only: bool = False,
    replay_mode: bool = False,
    ui: UIConfig | None = None,
    controller=None,
    metrics=None,
    sampler=None,
    confirm_exit: bool = False,
    profiles=None,
):
    """创建悬浮窗（延迟导入 PyQt6，确保 QApplication 已存在）。"""
    global _OVERLAY_CLASS, _APP_REF

    import sys as _sys

    _import_qt()  # 提前失败 → OverlayUnavailableError

    from PyQt6.QtWidgets import QApplication

    if QApplication.instance() is None:
        _APP_REF = QApplication(_sys.argv)
    # 托盘常驻应用：窗口（设置页/悬浮窗）关闭不意味着退出，退出只能走托盘/关闭按钮
    app = QApplication.instance()
    app.setQuitOnLastWindowClosed(False)
    fix_app_font(app)

    if _OVERLAY_CLASS is None:
        _OVERLAY_CLASS = make_overlay_window_class()
    return _OVERLAY_CLASS(
        display_order=display_order,
        opacity=opacity,
        profile=profile,
        read_only=read_only,
        replay_mode=replay_mode,
        ui=ui,
        controller=controller,
        metrics=metrics,
        sampler=sampler,
        confirm_exit=confirm_exit,
        profiles=profiles,
    )
