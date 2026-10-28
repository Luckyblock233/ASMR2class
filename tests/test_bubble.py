"""漫画气泡模式测试（offscreen）。"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from src.subtitle.document import Segment, SessionDocument
from src.ui import i18n
from src.ui.bubble import TAIL_DIRECTIONS, create_bubble

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _i18n():
    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    yield
    i18n._reset_for_testing()


def _segment(text="原文内容", translation="译文内容", sid=1):
    return Segment(id=sid, start=0, end=1, source=text, translation=translation)


def test_bubble_displays_current_sentence(qapp):
    b = create_bubble()
    b.show()
    b.update_segment(_segment())
    qapp.processEvents()
    assert b.source_text == "原文内容"
    assert b.translation_text == "译文内容"
    b.close()


def test_bubble_display_modes(qapp):
    b = create_bubble()
    b.update_segment(_segment())
    b.set_display_mode("source_only")
    b.update_segment(_segment())
    assert b.source_text == "原文内容"
    assert b.translation_text == ""
    b.set_display_mode("translation_only")
    b.update_segment(_segment())
    assert b.translation_text == "译文内容"
    assert b.source_text == ""
    b.close()


def test_bubble_tail_directions(qapp):
    b = create_bubble()
    b.resize(360, 160)
    body_s = b._body_rect()
    b.set_tail_direction("W")
    body_w = b._body_rect()
    assert body_w.left() > body_s.left()  # 本体让出左侧尾巴带
    b.set_tail_direction("N")
    assert b._body_rect().top() > body_s.top()
    b.set_tail_direction("not-a-direction")  # 非法方向忽略
    assert b.tail_direction() == "N"
    for d in TAIL_DIRECTIONS:  # 八方向都能设置
        b.set_tail_direction(d)
        assert b.tail_direction() == d
    b.close()


def test_bubble_exit_signal(qapp):
    b = create_bubble()
    received = []
    b.exit_requested.connect(lambda: received.append(True))
    b.exit_requested.emit()
    assert received == [True]
    b.close()


def test_overlay_bubble_mode_toggle(qapp):
    """开启：悬浮窗隐藏、气泡显示并带入最新句；退出：恢复正常悬浮窗。"""
    from src.ui.overlay import create_overlay

    w = create_overlay()
    w.show()
    w.upsert_segment.emit(_segment("最新句", "最新译文"))
    qapp.processEvents()

    w.set_bubble_mode(True)
    qapp.processEvents()
    assert not w.isVisible()
    assert w._bubble.isVisible()
    assert w._bubble.source_text == "最新句"  # 带入最新句
    assert w.act_bubble_exit.isVisible()  # 托盘出现退出项

    # 气泡模式下新句继续流入
    w.upsert_segment.emit(_segment("更新的句", "更新译文", sid=2))
    qapp.processEvents()
    assert w._bubble.source_text == "更新的句"

    # 托盘退出
    w.act_bubble_exit.trigger()
    qapp.processEvents()
    assert w.isVisible()
    assert not w._bubble.isVisible()
    assert not w.act_bubble_exit.isVisible()
    w.close()


def test_overlay_bubble_replay_follows_current(qapp):
    """回放模式：气泡跟随当前播放句。"""
    from src.ui.overlay import create_overlay

    w = create_overlay(replay_mode=True, read_only=True)
    w.show()
    w.upsert_segment.emit(_segment("第一句", "一", sid=1))
    w.upsert_segment.emit(_segment("第二句", "二", sid=2))
    qapp.processEvents()

    w.set_bubble_mode(True)
    w.notify_current_segment(2)
    qapp.processEvents()
    assert w._bubble.source_text == "第二句"
    w.notify_current_segment(None)  # 句间空白：气泡内容不变
    assert w._bubble.source_text == "第二句"
    w.close()


def test_bubble_tail_geometry_correct(qapp):
    """尾巴几何：本体在尾巴侧让出尾巴带，尖端伸入带内（不再偏出窗口/贴边）。"""
    b = create_bubble()
    b.resize(360, 160)
    for d in TAIL_DIRECTIONS:
        b.set_tail_direction(d)
        body = b._body_rect()
        tail = b._tail_polygon()
        pts = [tail.at(i) for i in range(tail.count())]
        # 尖端必须在窗口范围内
        assert all(0 <= p.x() <= 360 and 0 <= p.y() <= 160 for p in pts), d
        # 本体在尾巴侧缩进
        if d == "N":
            assert body.top() > 20
        elif d == "S":
            assert body.bottom() < 140
        elif d == "W":
            assert body.left() > 20
        elif d == "E":
            assert body.right() < 340
    b.close()


def test_bubble_full_text_no_marquee(qapp):
    """长文本完整显示（可换行），不再有跑马灯滚动。"""
    b = create_bubble()
    long_text = "这是一段很长很长的原文，超出气泡宽度也不应该被滚动切割，而是完整换行显示"
    b.update_segment(_segment(long_text, "译文"))
    assert b.source_text == long_text  # 完整文本，无跑马灯偏移
    b.close()


def test_bubble_fade_out_on_gap(qapp):
    """句间空白淡出：dim 模式（默认）降到低透明度保持。"""
    b = create_bubble()
    b.show()
    b.update_segment(_segment())
    qapp.processEvents()
    b.fade_out()
    assert b._fade.endValue() == 0.25
    from PyQt6.QtCore import QAbstractAnimation

    assert b._fade.state() == QAbstractAnimation.State.Running
    b.close()


def test_bubble_independent_appearance(qapp):
    """气泡独立外观：主题/不透明度/字号与主配置分离。"""
    from src.config import UIConfig
    from src.ui.bubble import bubble_spec

    ui = UIConfig(theme="dark", bubble_theme="light", bubble_opacity=0.5,
                  bubble_font_size=22)
    spec = bubble_spec(ui)
    assert spec.theme.name == "light"
    assert spec.background_opacity == 0.5
    assert spec.theme.source_font_size == 22

    # 跟随模式：全空回落主设置
    ui2 = UIConfig(theme="dark", background_opacity=0.7)
    spec2 = bubble_spec(ui2)
    assert spec2.theme.name == "dark"
    assert spec2.background_opacity == 0.7


def test_bubble_resize_by_edge_drag(qapp):
    """贴着气泡本体可见描边拖动实时调整大小（四边 + 四角）。"""
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    b = create_bubble()
    b.show()
    b.resize(360, 160)
    top_left = b.frameGeometry().topLeft()
    body = b._body_rect()
    corner = QPoint(int(body.right()) - 4, int(body.bottom()) - 4)  # 右下角描边内侧

    def ev(kind, local, global_):
        return QMouseEvent(kind, QPointF(local), QPointF(global_),
                           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier)

    b.mousePressEvent(ev(QMouseEvent.Type.MouseButtonPress, corner,
                         top_left + corner))
    b.mouseMoveEvent(ev(QMouseEvent.Type.MouseMove, corner,
                        top_left + corner + QPoint(60, 40)))
    b.mouseReleaseEvent(None)
    assert b.width() == 420 and b.height() == 200
    assert b.frameGeometry().topLeft() == top_left  # 右下角拖动：左上角不动
    b.close()


def test_bubble_resize_from_left_edge_moves_origin(qapp):
    """左边缘拖动：尺寸变化的同时窗口原点跟随平移。"""
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    b = create_bubble()
    b.show()
    b.resize(360, 160)
    top_left = b.frameGeometry().topLeft()
    body = b._body_rect()
    left_mid = QPoint(int(body.left()) + 2, int(body.center().y()))

    def ev(kind, local, global_):
        return QMouseEvent(kind, QPointF(local), QPointF(global_),
                           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier)

    assert b._edge_at(left_mid) == "left"
    b.mousePressEvent(ev(QMouseEvent.Type.MouseButtonPress, left_mid,
                         top_left + left_mid))
    b.mouseMoveEvent(ev(QMouseEvent.Type.MouseMove, left_mid,
                        top_left + left_mid + QPoint(-50, 30)))
    b.mouseReleaseEvent(None)
    assert b.width() == 410 and b.height() == 160  # 向左拉宽 50，高度不变
    assert b.frameGeometry().topLeft() == top_left + QPoint(-50, 0)
    b.close()


def test_bubble_resize_zone_on_visible_border_not_transparent_ring(qapp):
    """回归：判定区贴可见描边，而非窗口外缘（外缘透明像素在 Windows 上点击穿透）。"""
    from PyQt6.QtCore import QPoint

    b = create_bubble()
    b.resize(360, 160)
    body = b._body_rect()
    # 窗口右下角（尾巴带 + 透明留白内）：不判定为调整区（真机上收不到事件）
    assert b._edge_at(QPoint(359, 159)) == ""
    # 本体描边内侧：判定为调整区
    assert b._edge_at(QPoint(int(body.right()) - 2, int(body.bottom()) - 2)) == "bottomright"
    b.close()


def test_tail_base_overlaps_body_no_seam(qapp):
    """无缝隙：尾巴基边两个端点都深入本体内部（union 描边不出现共享边）。"""
    b = create_bubble()
    b.resize(360, 160)
    for d in TAIL_DIRECTIONS:
        b.set_tail_direction(d)
        body = b._body_rect()
        tail = b._tail_polygon()
        base = [tail.at(0), tail.at(1)]
        for p in base:
            assert body.contains(p), f"{d}: 基边端点 ({p.x():.0f},{p.y():.0f}) 不在本体内"
    b.close()


def test_bubble_edge_cursor_feedback(qapp):
    """悬停在本体描边附近显示对应方向的调整光标，中间区域恢复默认。"""
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    b = create_bubble()
    b.show()
    b.resize(360, 160)
    body = b._body_rect()
    bl, br = int(body.left()), int(body.right())
    bt, bb = int(body.top()), int(body.bottom())
    cy = int(body.center().y())
    cx = int(body.center().x())

    def move_to(x, y):
        b.mouseMoveEvent(QMouseEvent(
            QMouseEvent.Type.MouseMove, QPointF(x, y), QPointF(x, y),
            Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        ))

    move_to(br - 2, bb - 2)  # 右下角描边
    assert b.cursor().shape() == Qt.CursorShape.SizeFDiagCursor
    move_to(bl + 2, bt + 2)  # 左上角描边
    assert b.cursor().shape() == Qt.CursorShape.SizeFDiagCursor
    move_to(br - 2, bt + 2)  # 右上角描边
    assert b.cursor().shape() == Qt.CursorShape.SizeBDiagCursor
    move_to(bl + 2, bb - 2)  # 左下角描边
    assert b.cursor().shape() == Qt.CursorShape.SizeBDiagCursor
    move_to(br - 2, cy)      # 右缘
    assert b.cursor().shape() == Qt.CursorShape.SizeHorCursor
    move_to(bl + 2, cy)      # 左缘
    assert b.cursor().shape() == Qt.CursorShape.SizeHorCursor
    move_to(cx, bt + 2)      # 上缘
    assert b.cursor().shape() == Qt.CursorShape.SizeVerCursor
    move_to(cx, bb - 2)      # 下缘
    assert b.cursor().shape() == Qt.CursorShape.SizeVerCursor
    move_to(cx, cy)          # 中间
    assert b.cursor().shape() in (None, Qt.CursorShape.ArrowCursor)
    b.close()


def test_bubble_appearance_entry_wired(qapp):
    """右键菜单带外观设置入口（与主窗同一个设置页）。"""
    called = []
    b = create_bubble(on_open_settings=lambda: called.append(True))
    assert b._on_open_settings is not None
    b._on_open_settings()
    assert called == [True]
    b.close()


def test_bubble_labels_mouse_transparent(qapp):
    """无子控件：文本直接绘制，整个气泡都可拖动/边缘调整，无标签分界线。"""
    from PyQt6.QtWidgets import QLabel

    b = create_bubble()
    assert b.findChildren(QLabel) == []
    b.close()


def test_bubble_drag_from_text_area_moves(qapp):
    """文字区域按下拖动 = 移动窗口（标签不拦截）。"""
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    b = create_bubble()
    b.show()
    pos0 = b.frameGeometry().topLeft()

    def ev(kind, local, global_):
        return QMouseEvent(kind, QPointF(local), QPointF(global_),
                           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier)

    center = QPoint(180, 80)
    b.mousePressEvent(ev(QMouseEvent.Type.MouseButtonPress, center, pos0 + center))
    b.mouseMoveEvent(ev(QMouseEvent.Type.MouseMove, center,
                        pos0 + center + QPoint(30, 20)))
    b.mouseReleaseEvent(None)
    assert b.frameGeometry().topLeft() == pos0 + QPoint(30, 20)
    b.close()


def test_bubble_fade_uses_window_opacity(qapp):
    """淡出用 windowOpacity（不用会破坏半透明背景的 GraphicsEffect）。"""
    b = create_bubble()
    assert b.graphicsEffect() is None
    assert b._fade.propertyName() == b"windowOpacity"
    b.fade_out()
    assert b._fade.endValue() == 0.25  # 默认 dim 模式：降到低透明度保持
    b.close()


def test_bubble_appearance_dialog_bubble_only(qapp, tmp_path):
    """气泡外观小窗只含气泡三项，预览/应用链路工作。"""
    from src.config import UIConfig
    from src.ui.settings import create_bubble_appearance_dialog

    from .test_overlay_window import _make_controller

    controller = _make_controller(tmp_path, [])
    previews, applied = [], []
    dialog = create_bubble_appearance_dialog(
        controller, on_preview=previews.append, on_applied=applied.append)
    dialog.show()
    qapp.processEvents()
    assert not hasattr(dialog, "a_theme")  # 不含主悬浮窗设置
    dialog.b_font.setValue(22)
    qapp.processEvents()
    assert previews and previews[-1].bubble_font_size == 22
    assert applied == []  # 预览不落盘
    dialog.close()


def test_bubble_fill_alpha_follows_opacity(qapp):
    """不透明度设置生效：填充 alpha = 设置的不透明度。"""
    from src.config import UIConfig

    b = create_bubble(UIConfig(bubble_opacity=0.5))
    b.resize(360, 160)
    b.show()
    b.update_segment(_segment())
    qapp.processEvents()
    img = b.grab().toImage()
    c = img.pixelColor(180, 80)
    assert abs(c.alpha() - 127) < 10  # 0.5 * 255
    assert (c.red(), c.green(), c.blue()) == (30, 30, 30)  # 默认深色
    b.close()


def test_bubble_fade_hides_window(qapp):
    """淡出完成后窗口真正隐藏，淡入重新显示。"""
    from PyQt6.QtTest import QTest

    from src.config import UIConfig

    b = create_bubble(UIConfig(bubble_fade="hide"))
    b.show()
    b._fade.setDuration(1)
    b.fade_out()
    QTest.qWait(50)
    qapp.processEvents()
    assert not b.isVisible()
    assert b.windowOpacity() == 1.0  # 已复位
    b.update_segment(_segment())     # 新句 → 淡入并重新显示
    assert b.isVisible()
    b.close()


def test_bubble_fade_modes(qapp):
    """淡出三模式：off 不动 / dim 保持低透明度 / hide 淡出后隐藏。"""
    from PyQt6.QtTest import QTest

    from src.config import UIConfig

    b = create_bubble(UIConfig(bubble_fade="off"))
    b.show()
    b.fade_out()
    assert b.windowOpacity() == 1.0  # off：完全不动
    b.close()

    b = create_bubble(UIConfig(bubble_fade="dim", bubble_fade_min=0.4))
    b.show()
    b.fade_out()
    assert b._fade.endValue() == 0.4
    b._fade.setDuration(1)
    QTest.qWait(50)
    qapp.processEvents()
    assert b.isVisible()  # dim 不隐藏
    b.close()


def test_bubble_settings_reopen_shows_saved_values(qapp, tmp_path):
    """回归：预览后未应用就关闭 → 重开时表单显示已保存值，不是预览残留。"""
    from src.config import UIConfig
    from src.ui.settings import create_bubble_appearance_dialog

    from .test_overlay_window import _make_controller

    controller = _make_controller(tmp_path, [])
    dialog = create_bubble_appearance_dialog(controller)
    dialog.show()
    dialog.b_font.setValue(30)  # 预览但未应用
    dialog.close()
    dialog2 = create_bubble_appearance_dialog(controller)
    # 模拟重开：reload 后回到已保存值
    dialog2.reload()
    assert dialog2.b_font.value() == controller.cfg.ui.bubble_font_size
    dialog2.close()


def test_bubble_rms_vad_fade_and_idle(qapp):
    """RMS/VAD 低于阈值持续超过延时才触发（防抖）；恢复后复位计时。"""
    from PyQt6.QtCore import QAbstractAnimation

    from src.config import UIConfig
    from src.telemetry import RuntimeMetrics, SystemSample
    from src.ui.overlay import create_overlay

    metrics = RuntimeMetrics()

    class FakeSampler:
        def sample(self):
            return SystemSample()

    w = create_overlay(metrics=metrics, sampler=FakeSampler(),
                       ui=UIConfig(bubble_idle_text="♪", bubble_fade_delay_sec=1.5))
    w.show()
    w.set_bubble_mode(True)
    qapp.processEvents()

    # RMS 低于阈值但未满延时 → 不触发
    metrics.record_audio_block(0.005)  # < 默认 0.01
    w._refresh_metrics()
    assert w._bubble._fade.state() != QAbstractAnimation.State.Running

    # 持续超过延时 → 触发淡出
    w._bubble_below_since -= 2.0  # 模拟已持续 2s
    w._refresh_metrics()
    assert w._bubble._fade.state() == QAbstractAnimation.State.Running

    # 恢复（RMS 正常 + 有人声）→ 计时复位
    metrics.set_vad_active(True)
    metrics.record_audio_block(0.2)
    w._refresh_metrics()
    assert w._bubble_below_since is None

    # RMS 恢复即显示：VAD 无人声 → 立即显示待机文字（不等延时/不等语音确认）
    metrics.set_vad_active(False)
    w._refresh_metrics()
    assert w._bubble.source_text == "♪"

    # 新句到来 → 恢复真实内容
    w.upsert_segment.emit(_segment("有人声了", "译文"))
    qapp.processEvents()
    assert w._bubble.source_text == "有人声了"
    assert w._bubble.translation_text == "译文"
    w.close()


def test_bubble_reappears_on_voice_without_new_sentence(qapp):
    """回归：hide 模式隐藏后，VAD 检测到语音即恢复显示，不等新句产出。"""
    from PyQt6.QtTest import QTest

    from src.config import UIConfig
    from src.telemetry import RuntimeMetrics, SystemSample
    from src.ui.overlay import create_overlay

    metrics = RuntimeMetrics()

    class FakeSampler:
        def sample(self):
            return SystemSample()

    w = create_overlay(metrics=metrics, sampler=FakeSampler(),
                       ui=UIConfig(bubble_fade="hide", bubble_fade_delay_sec=0.0))
    w.show()
    w.set_bubble_mode(True)
    w.upsert_segment.emit(_segment("句子", "译文"))
    qapp.processEvents()

    # 静默超过延时 → 淡出隐藏
    w._bubble._fade.setDuration(1)
    metrics.record_audio_block(0.005)
    w._refresh_metrics()
    QTest.qWait(50)
    qapp.processEvents()
    assert not w._bubble.isVisible()

    # 语音恢复（VAD 活动、RMS 正常）→ 立即淡入，不等新句
    metrics.record_audio_block(0.2)
    metrics.set_vad_active(True)
    w._refresh_metrics()
    qapp.processEvents()
    assert w._bubble.isVisible()
    w.close()


def test_bubble_idle_hold_keeps_recent_subtitle(qapp):
    """有声无人声片段：刚识别的字幕保持 hold 时长后才切待机文字（不被立刻顶掉）。"""
    import time

    b = create_bubble()
    b.show()
    b.update_segment(_segment("刚识别的句子", "译文"))
    qapp.processEvents()

    # 保持期内：VAD 无人声也继续显示字幕
    b.show_idle("……", hold_sec=2.0)
    assert b.source_text == "刚识别的句子"

    # 超过保持期 → 切待机文字
    b._content_at = time.monotonic() - 3.0
    b.show_idle("……", hold_sec=2.0)
    assert b.source_text == "……"

    # 新句到来重新计时
    b.update_segment(_segment("下一句", "译文2"))
    b.show_idle("……", hold_sec=2.0)
    assert b.source_text == "下一句"

    # hold=0 → 立即切换
    b.show_idle("……", hold_sec=0.0)
    assert b.source_text == "……"
    b.close()


def test_bubble_idle_hold_via_overlay(qapp):
    """联动：有声无人声时字幕按配置时长保持，超时后才显示待机文字。"""
    from src.config import UIConfig
    from src.telemetry import RuntimeMetrics, SystemSample
    from src.ui.overlay import create_overlay

    metrics = RuntimeMetrics()

    class FakeSampler:
        def sample(self):
            return SystemSample()

    w = create_overlay(metrics=metrics, sampler=FakeSampler(),
                       ui=UIConfig(bubble_idle_text="♪", bubble_idle_hold_sec=2.0))
    w.show()
    w.set_bubble_mode(True)
    w.upsert_segment.emit(_segment("刚识别", "译文"))
    qapp.processEvents()

    # 有声（RMS 正常）但无人声：保持期内字幕不被顶掉
    metrics.record_audio_block(0.2)
    metrics.set_vad_active(False)
    w._refresh_metrics()
    assert w._bubble.source_text == "刚识别"

    # 超过保持期 → 显示待机文字
    w._bubble._content_at -= 3.0
    w._refresh_metrics()
    assert w._bubble.source_text == "♪"
    w.close()


def test_idle_text_updates_immediately(qapp):
    """回归：待机文字变更立即生效（不被 _idle 幂等短路挡住）。"""
    from src.config import UIConfig

    b = create_bubble(UIConfig(bubble_idle_text="♪"))
    b.show()
    b.show_idle("♪")
    assert b.source_text == "♪"
    b.show_idle("（￣▽￣）")  # 设置页修改后再次触发
    assert b.source_text == "（￣▽￣）"
    b.close()
