"""完整 A–D 悬浮窗测试（offscreen）：A 栏控制、B 栏槽位、C 栏伪指标。"""

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from src.config import SlotsConfig, UIConfig, load_config
from src.nlp.providers import SLOT_GRAMMAR, SLOT_TRANSLATION, ProviderRegistry
from src.nlp.slots import ModelSlots, resolve_slots
from src.telemetry import RuntimeMetrics, SystemSample
from src.ui import i18n
from src.ui.overlay import create_overlay
from src.ui.settings import SettingsController

from .test_lease import _cloud

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _i18n():
    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    yield
    i18n._reset_for_testing()


def _profile(slots=None):
    return SimpleNamespace(
        name="test",
        display_name="测试语言",
        source_lang="ja",
        target_lang="zh",
        slots=slots or {},
    )


def _make_controller(tmp_path, providers=(), profile=None):
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.example.yaml").write_text("llm: {}\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    cfg.providers = list(providers)
    registry = ProviderRegistry(cfg.providers)
    profile = profile or _profile()
    slots = ModelSlots(registry, resolve_slots(cfg.slots, profile.slots, registry))
    return SettingsController(cfg, registry, slots, profile)


# ---- A 栏 ----


def test_a_bar_controls_and_compact(qapp, tmp_path):
    ui = UIConfig(show_bar_b=True, show_bar_c=True)
    w = create_overlay(ui=ui)
    w.show()
    qapp.processEvents()

    # 运行/暂停信号
    pauses = []
    w.pause_toggled.connect(pauses.append)
    w.btn_pause.click()
    assert pauses == [True]
    assert w.btn_pause.text() == "继续"
    w.btn_pause.click()
    assert pauses == [True, False]

    # 清空显示（默认无确认弹窗；confirm_exit=True 时才确认）
    w.upsert_segment.emit(__import__("src.subtitle.document", fromlist=["Segment"]).Segment(0, 1, "x", "y", id=1))
    qapp.processEvents()
    assert w.subtitle_view._blocks
    w.btn_clear.click()
    qapp.processEvents()
    assert not w.subtitle_view._blocks

    # 字幕区显隐
    w.act_subtitles.setChecked(False)
    w.act_subtitles.triggered.emit(False)
    assert not w.subtitle_view.isVisible()
    w.act_subtitles.triggered.emit(True)

    # 紧凑模式隐藏 B/C 栏
    w.act_compact.triggered.emit(True)
    assert not w.barB.isVisible() and not w.barC.isVisible()
    w.act_compact.triggered.emit(False)
    assert w.barB.isVisible() and w.barC.isVisible()

    # 穿透时控制按钮隐藏，托盘可用
    w.set_click_through(True)
    assert not w.btn_close.isVisible()
    assert w.tray.isVisible()
    w.set_click_through(False)
    w.close()


def test_bar_visibility_from_config(qapp):
    ui = UIConfig(show_bar_b=False, show_bar_c=False)
    w = create_overlay(ui=ui)
    w.show()
    qapp.processEvents()
    assert not w.barB.isVisible() and not w.barC.isVisible()
    assert w.barA.isVisible()
    w.close()


# ---- B 栏 ----


def test_b_bar_capability_filter_and_override_lock(qapp, tmp_path):
    t_only = _cloud("t-only")
    t_only.capabilities = ["translation"]
    both = _cloud("both")
    profile = _profile(slots={SLOT_GRAMMAR: "both"})
    controller = _make_controller(tmp_path, [t_only, both], profile)
    w = create_overlay(profile=profile, controller=controller)
    w.show()
    qapp.processEvents()

    t_combo = w.slot_combos[SLOT_TRANSLATION]
    g_combo = w.slot_combos[SLOT_GRAMMAR]
    assert t_combo.count() == 3   # 未绑定 + 2 个可译
    assert g_combo.count() == 2   # 未绑定 + 仅 both
    assert not g_combo.isEnabled()  # 档案覆盖锁定
    assert "档案覆盖" in g_combo.toolTip()
    assert t_combo.isEnabled()
    w.close()


def test_b_bar_rebind_via_controller(qapp, tmp_path):
    both = _cloud("both")
    controller = _make_controller(tmp_path, [both])
    w = create_overlay(controller=controller)
    w.show()
    qapp.processEvents()

    combo = w.slot_combos[SLOT_TRANSLATION]
    index = combo.findData("both")
    combo.setCurrentIndex(index)
    # 健康检查打桩为直接成功，避免真实网络
    controller.registry.check_health = lambda pid: None
    combo.activated.emit(index)
    # worker 线程异步完成
    import time

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        qapp.processEvents()
        if controller.slots.resolution.bindings.get(SLOT_TRANSLATION) == "both":
            break
        time.sleep(0.02)
    assert controller.slots.resolution.bindings[SLOT_TRANSLATION] == "both"
    w.close()


# ---- C 栏 ----


class _FakeSampler:
    def __init__(self, sample):
        self._sample = sample

    def sample(self):
        return self._sample


def test_c_bar_metrics_and_vram_warning(qapp):
    metrics = RuntimeMetrics()
    metrics.record_asr(200.0)
    metrics.record_asr(200.0)
    metrics.record_translation(900.0, True)
    metrics.record_translation(900.0, False)

    low_vram = SystemSample(cpu_percent=12.0, ram_percent=40.0,
                            gpu_percent=30.0, vram_used_mb=6000.0, vram_free_mb=1500.0)
    w = create_overlay(metrics=metrics, sampler=_FakeSampler(low_vram))
    w.show()
    w.set_asr_info("ASR small @ cuda（int8）")
    w._refresh_metrics()  # 直接触发，不依赖定时器节拍

    text = w.monitor_text()
    assert "ASR small @ cuda" in text        # 监控行整行平铺
    assert "CPU 12%" in text and "RAM 40%" in text and "GPU 30%" in text
    assert "VRAM" in text
    assert "句 2" in text and "译 2" in text
    assert "失败 1" in text
    # VRAM 告警色在对应部件的样式表上；tooltip 在行容器上
    vram_part = next(l for l in w._monitor_parts if l.text().startswith("VRAM"))
    assert "#FFC94D" in vram_part.styleSheet()
    assert w.monitor_row.toolTip()
    # 平铺布局：部件之间插入了 stretch（不是简单左对齐）
    assert w.monitor_layout.count() == 2 * len(w._monitor_parts) - 1
    # 平均延迟不再占用监控行（已移到每句行尾）
    assert "900ms" not in text

    w.sampler = _FakeSampler(SystemSample(cpu_percent=5.0, vram_free_mb=6000.0))
    w._refresh_metrics()
    vram_part = next(l for l in w._monitor_parts if l.text().startswith("VRAM"))
    assert "#FFC94D" not in vram_part.styleSheet()
    assert w.monitor_row.toolTip() == ""

    # 无 GPU：N/A 字段不显示
    w.sampler = _FakeSampler(SystemSample(cpu_percent=5.0))
    w._refresh_metrics()
    text = w.monitor_text()
    assert "GPU" not in text
    assert "VRAM" not in text
    w.close()


def test_rms_vad_progress_bars(qapp):
    """RMS/VAD 为横向进度条（百分比），不再是文字。"""
    metrics = RuntimeMetrics()
    metrics.record_audio_block(0.2)   # RMS 0.2 → 50%
    metrics.set_vad_active(True)
    w = create_overlay(metrics=metrics, sampler=_FakeSampler(SystemSample()))
    w.show()
    w._refresh_metrics()
    assert w.rms_bar.value() == 50
    assert w.vad_bar.value() == 100
    metrics.set_vad_active(False)
    w._refresh_metrics()
    assert w.vad_bar.value() == 0
    w.close()


def test_latency_thresholds_color_per_sentence(qapp):
    """延迟显示移到每句行尾，阈值着色（≤good 绿、good–warn 黄、>warn 红）。"""
    from src.subtitle.document import TRANSLATION_READY, Segment

    w = create_overlay()
    w.show()
    view = w.subtitle_view
    cases = [(200.0, "#7ED957"), (700.0, "#FFC94D"), (1500.0, "#FF6B6B")]
    for i, (ms, color) in enumerate(cases, 1):
        view.upsert_segment(Segment(
            id=i, start=i - 1, end=i, source=f"原{i}", translation=f"译{i}",
            translation_status=TRANSLATION_READY, asr_ms=ms, translate_ms=ms,
        ))
    qapp.processEvents()
    for i, (ms, color) in enumerate(cases, 1):
        block = view._blocks[i]
        assert f"ASR {ms:.0f}ms" in block.source_label.text()
        assert f"TL {ms:.0f}ms" in block.translation_label.text()
        assert color in block.source_label.text(), f"{ms}ms 应为 {color}"
        assert color in block.translation_label.text()

    # 自定义阈值生效（应用外观后句块重着色）
    w.apply_appearance(UIConfig(latency_good_ms=100, latency_warn_ms=300))
    qapp.processEvents()
    assert "#FFC94D" in view._blocks[1].source_label.text()   # 200ms → 黄
    assert "#FF6B6B" in view._blocks[2].source_label.text()   # 700ms → 红
    w.close()


def test_ram_threshold_color(qapp):
    metrics = RuntimeMetrics()
    w = create_overlay(metrics=metrics,
                       sampler=_FakeSampler(SystemSample(ram_percent=90.0)))
    w.show()
    w._refresh_metrics()
    ram_part = next(l for l in w._monitor_parts if l.text().startswith("RAM"))
    assert "#FFC94D" in ram_part.styleSheet()  # 90% ≥ 默认 85% 阈值
    w._ui = UIConfig(ram_warn_percent=95)
    w._refresh_metrics()
    ram_part = next(l for l in w._monitor_parts if l.text().startswith("RAM"))
    assert "#FFC94D" not in ram_part.styleSheet()
    w.close()


def test_service_status_dots_semantics(qapp, tmp_path):
    """状态点统一语义：○停止 / ●运行中 / ●错误（tooltip 带最近错误）。"""
    p = _cloud("p")
    controller = _make_controller(tmp_path, [p])
    w = create_overlay(controller=controller, metrics=RuntimeMetrics(),
                       sampler=_FakeSampler(SystemSample()))
    w.show()
    w._refresh_metrics()
    dot = w._service_buttons["p"]
    assert dot.text().startswith("○")  # 未实例化 = 停止
    assert "停止" in dot.toolTip()

    controller.registry.get("p")  # 实例化 → 运行中
    w._refresh_metrics()
    assert dot.text().startswith("●")
    assert "运行中" in dot.toolTip()

    inst = controller.registry.get("p")
    inst._client._set_online(False)
    inst._client._last_error = "APIConnectionError: 超时"
    w._refresh_metrics()
    assert "#FF6B6B" in dot.styleSheet()
    assert "错误" in dot.toolTip()
    assert "超时" in dot.toolTip()
    w.close()


def test_b_bar_hierarchy_and_label_not_occluded(qapp, tmp_path):
    """B 栏：主行只显示模型名（细节在 tooltip）；标签固定宽度不被下拉框遮挡。"""
    p = _cloud("p")
    p.name = "Index-Translate-2B"
    controller = _make_controller(tmp_path, [p])
    w = create_overlay(controller=controller)
    w.show()
    qapp.processEvents()
    combo = w.slot_combos[SLOT_TRANSLATION]
    assert combo.itemText(1) == "Index-Translate-2B"  # 无 （id） 后缀
    from PyQt6.QtCore import Qt

    tip = combo.itemData(1, Qt.ItemDataRole.ToolTipRole)
    assert "id: p" in tip and "endpoint:" in tip
    from PyQt6.QtWidgets import QLabel

    bar_b = w.barB.layout()
    row = bar_b.itemAt(1).layout()  # 第一行槽位
    label = row.itemAt(0).widget()
    assert isinstance(label, QLabel)
    assert label.minimumWidth() >= 56  # 标签有固定分区
    assert combo.minimumWidth() >= 160
    w.close()


def test_sentence_blocks_have_separator(qapp):
    """字幕条目之间有分隔线，便于扫读定位。"""
    w = create_overlay()
    w.show()
    w.upsert_segment.emit(__import__("src.subtitle.document", fromlist=["Segment"]).Segment(0, 1, "原文", "译文", id=1))
    qapp.processEvents()
    block = w.subtitle_view._blocks[1]
    assert "border-bottom" in block.styleSheet()
    w.close()


def test_appearance_apply_restyles(qapp):
    w = create_overlay()
    w.show()
    # 主题切换：白天模式整套预定义配色生效
    ui = UIConfig(theme="light", background_opacity=0.5, compact_mode=True)
    w.apply_appearance(ui)
    qapp.processEvents()
    assert "rgba(250, 250, 250" in w.central.styleSheet()
    assert "#1A1A1A" in w.subtitle_view._appearance.source_style  # 白天模式原文色
    assert not w.barB.isVisible()  # 紧凑模式
    # 未知主题回退深色
    w.apply_appearance(UIConfig(theme="nonexistent"))
    assert "#FFFFFF" in w.subtitle_view._appearance.source_style
    w.close()


# ---- 缺陷回归 ----


def test_background_unified_scroll_area_transparent(qapp):
    """字幕滚动区/视口/容器全链路透明：背景色调整对整个悬浮窗生效（不应有白色文字底板）。"""
    w = create_overlay()
    w.show()
    qapp.processEvents()
    view = w.subtitle_view
    assert not view.scroll_area.autoFillBackground()
    assert not view.scroll_area.viewport().autoFillBackground()
    assert not view._container.autoFillBackground()
    assert "transparent" in view.scroll_area.styleSheet()
    assert "transparent" in view.scroll_area.viewport().styleSheet()
    assert "transparent" in view._container.styleSheet()
    w.close()


def test_b_c_bars_are_multiline(qapp, tmp_path):
    """B 栏每项信息独立成行；C 栏为 进度条行 + 整行监控 + 服务状态行（参考 LiveTrans）。"""
    from PyQt6.QtWidgets import QVBoxLayout

    controller = _make_controller(tmp_path, [_cloud("p")])
    w = create_overlay(controller=controller, metrics=RuntimeMetrics())
    w.show()
    qapp.processEvents()
    assert isinstance(w.barB.layout(), QVBoxLayout)
    assert w.barB.layout().count() == 4  # 源语言行 + 目标语言行 + 翻译槽行 + 语法槽行
    assert isinstance(w.barC.layout(), QVBoxLayout)
    assert w.barC.layout().count() == 3  # RMS/VAD 进度条行 + 监控行 + 服务状态行
    w.close()


def test_closing_settings_does_not_quit_app(qapp, tmp_path):
    """关闭设置页只关闭自身：悬浮窗不关、应用不退出（WA_QuitOnClose=False）。"""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    controller = _make_controller(tmp_path, [_cloud("p")])
    w = create_overlay(controller=controller)
    w.show()
    w.open_settings()
    qapp.processEvents()
    dialog = w._settings_dialog
    assert dialog is not None and dialog.isVisible()
    assert not dialog.testAttribute(Qt.WidgetAttribute.WA_QuitOnClose)
    assert not QApplication.quitOnLastWindowClosed()

    dialog.reject()  # 等价于点「关闭」
    qapp.processEvents()
    assert not dialog.isVisible()
    assert w.isVisible()  # 悬浮窗仍在
    w.close()


# ---- 档案切换（§4.2 会话滚动的 UI 侧）----


def test_profile_combo_switch_flow(qapp, tmp_path):
    from src.profiles.loader import get_profile

    ja = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    en = get_profile(PROJECT_ROOT / "config" / "languages", "en")
    controller = _make_controller(tmp_path, [_cloud("p")], profile=ja)
    zh = get_profile(PROJECT_ROOT / "config" / "languages", "zh")
    w = create_overlay(profile=ja, controller=controller, profiles=[ja, en, zh])
    w.show()
    w.upsert_segment.emit(__import__("src.subtitle.document", fromlist=["Segment"]).Segment(0, 1, "旧句", "旧译", id=1))
    qapp.processEvents()

    requests = []
    w.pair_change_requested.connect(lambda s, t: requests.append((s, t)))
    index = w.src_combo.findData("en")
    w.src_combo.setCurrentIndex(index)
    w.src_combo.activated.emit(index)
    assert requests == [("en", "zh")]
    assert not w.src_combo.isEnabled()  # 切换期间禁点

    # 切换完成：视图清空、系统提示、槽位刷新、下拉恢复
    w.apply_profile(en, notice="已切换语言档案：英语")
    qapp.processEvents()
    assert w.src_combo.isEnabled()
    assert w.src_combo.currentData() == "en"
    assert not w.subtitle_view._blocks
    system_texts = [
        w.subtitle_view._layout.itemAt(i).widget().text()
        for i in range(w.subtitle_view._layout.count())
        if w.subtitle_view._layout.itemAt(i).widget() is not None
        and w.subtitle_view._layout.itemAt(i).widget().objectName() == "systemMessage"
    ]
    assert system_texts == ["— 已切换语言档案：英语 —"]
    assert controller.profile is en  # 控制器档案引用随动（槽位覆盖判断更新）
    w.close()


def test_profile_switch_error_restores_combo(qapp, tmp_path):
    from src.profiles.loader import get_profile

    ja = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    en = get_profile(PROJECT_ROOT / "config" / "languages", "en")
    controller = _make_controller(tmp_path, [_cloud("p")], profile=ja)
    w = create_overlay(profile=ja, controller=controller, profiles=[ja, en])
    w.show()
    index = w.src_combo.findData("en")
    w.src_combo.setCurrentIndex(index)
    w.src_combo.activated.emit(index)
    w.notify_profile_error("en→zh", "语言包不存在")
    qapp.processEvents()
    assert w.src_combo.isEnabled()
    assert w.src_combo.currentData() == "ja"  # 回退到当前档案
    w.close()


# ---- 外观即时预览 ----


def test_appearance_preview_does_not_save_and_reverts_on_close(qapp, tmp_path):
    import yaml

    from src.ui.settings import create_settings_dialog

    controller = _make_controller(tmp_path, [_cloud("p")])
    previews = []
    dialog = create_settings_dialog(controller, on_appearance_preview=previews.append)
    dialog.show()
    dialog.a_opacity.setValue(60)  # 控件变化 → 即时预览
    qapp.processEvents()
    assert previews and previews[-1].background_opacity == pytest.approx(0.6)
    raw = yaml.safe_load(controller.cfg.config_file.read_text(encoding="utf-8"))
    assert "ui" not in raw  # 预览不落盘
    dialog.close()

    # 悬浮窗侧：预览后未应用就关闭 → 还原
    w = create_overlay(controller=controller)
    w.show()
    w.open_settings()
    base_theme = w._ui.theme
    w._preview_appearance(UIConfig(theme="light"))
    assert w._ui.theme == "light"
    w._on_settings_closed(0)
    assert w._ui.theme == base_theme
    # 应用过则不再还原
    w._preview_appearance(UIConfig(theme="light"))
    w._on_appearance_applied(UIConfig(theme="light"))
    w._on_settings_closed(0)
    assert w._ui.theme == "light"

    # 回归：预览后未应用就关闭 → 悬浮窗还原；重开设置表单也必须是已保存值
    w._preview_appearance(UIConfig(theme="dark", background_opacity=0.4))
    w._on_settings_closed(0)
    assert w._ui.theme == "light"  # 悬浮窗还原
    w.open_settings()              # 重开设置
    qapp.processEvents()
    dialog = w._settings_dialog
    assert dialog.a_theme.currentData() == "light"  # 表单是已保存值，不是预览残留
    assert dialog.a_opacity.value() == round(
        w._ui.effective_background_opacity * 100
    )
    w.close()


def test_theme_switch_recolors_all_bar_widgets(qapp, tmp_path):
    """主题切换时 A/B/C 栏标签、按钮、下拉框、菜单文字颜色全部随动（不留系统默认色）。"""
    p = _cloud("p")
    controller = _make_controller(tmp_path, [p])
    metrics = RuntimeMetrics()
    w = create_overlay(controller=controller, metrics=metrics,
                       sampler=_FakeSampler(SystemSample()))
    w.show()
    qapp.processEvents()

    dark = w.central.styleSheet()
    assert "#FFFFFF" in dark  # 深色主题文字
    assert "QComboBox" in dark and "QLabel" in dark

    w.apply_appearance(UIConfig(theme="light"))
    qapp.processEvents()
    light = w.central.styleSheet()
    assert "#1A1A1A" in light   # 白天主题文字
    assert "#FFFFFF" not in light.split("QWidget#centralHost")[1]  # 栏位不再残留深色文字色
    assert "#FAFAFA" in w.main_menu.styleSheet()  # 主菜单随主题
    assert "#FAFAFA" in w.tray_menu.styleSheet()  # 托盘菜单随主题

    # 句块元信息/分隔线也随主题（C 栏 meta 色、句块分隔线）
    assert "#666666" in w.subtitle_view._appearance.block_stylesheet \
        or "rgba(0, 0, 0" in w.subtitle_view._appearance.block_stylesheet
    w.close()


def test_context_menu_themed(qapp):
    """句块右键菜单（QMenu）随主题着色：不再是默认黑字。"""
    from PyQt6.QtWidgets import QMenu

    from src.subtitle.document import Segment

    w = create_overlay()
    w.show()
    w.upsert_segment.emit(Segment(0, 1, "原文", "译文", id=1))
    qapp.processEvents()

    # 深色主题：菜单白字深底
    css = w.central.styleSheet()
    assert "#centralHost QMenu" in css and "#FFFFFF" in css

    # 白天主题：菜单黑字浅底
    w.apply_appearance(UIConfig(theme="light"))
    qapp.processEvents()
    css = w.central.styleSheet()
    assert "#centralHost QMenu" in css and "#1A1A1A" in css
    # 句块右键菜单是 centralHost 后代，继承该样式表
    block = w.subtitle_view._blocks[1]
    menu = QMenu(block)
    assert menu.styleSheet() == ""  # 无需自设：继承自 centralHost
    assert w.central.styleSheet() is css or "#centralHost QMenu" in w.central.styleSheet()
    menu.deleteLater()
    w.close()
