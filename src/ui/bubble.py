"""漫画对话框（气泡）模式：把悬浮窗变成人物旁边的对话框。

- 圆角气泡 + 八方向尾巴；窗口 = 气泡本体 + 尾巴带，尾巴向对应方向伸出
- 当前句完整显示（可换行，不滚动）；句子结束（回放进入句间空白）淡出，新句淡入
- 右键气泡：尾巴方向（八方向）/显示内容（双语/仅原文/仅译文）/退出气泡模式
- 外观独立：主题/不透明度/字号可在设置页单独调整（settings 的「气泡」组）
- 大小可用边缘拖动实时调整
实时与回放模式均可用：实时显示最新句，回放跟随当前播放句。
"""

from __future__ import annotations

import dataclasses
import time

from ..config import UIConfig

TAIL_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
DISPLAY_MODES = ("both", "source_only", "translation_only")

_TAIl_LEN = 22  # 尾巴长度（px）
_PAD = 6        # 气泡本体外缘留白
_EDGE = 14      # 调整大小判定宽度：贴着气泡本体可见描边的内/外距离
                 # （窗口外圈与尾巴带是透明像素，Windows 分层窗口透明像素
                 #  点击穿透收不到鼠标事件，判定必须落在不透明的本体描边附近）


def bubble_spec(ui: UIConfig):
    """气泡独立外观：主题/不透明度/字号可单独设置，留空跟随主悬浮窗。"""
    from .appearance import AppearanceSpec, get_theme

    theme = get_theme(ui.bubble_theme or ui.theme)
    if ui.bubble_font_size > 0:
        theme = dataclasses.replace(
            theme,
            source_font_size=ui.bubble_font_size,
            translation_font_size=ui.bubble_font_size + 2,
        )
    return AppearanceSpec(
        theme=theme,
        background_opacity=ui.bubble_opacity
        if ui.bubble_opacity is not None
        else ui.effective_background_opacity,
        line_spacing=ui.line_spacing,
    )


def create_bubble(ui: UIConfig | None = None, on_exit=None, on_open_settings=None):
    """构建气泡窗口（延迟导入 PyQt6）。

    on_open_settings：右键「外观设置…」入口（与主悬浮窗同一个设置页，
    预览/应用经 apply_appearance 即时同步到气泡）。
    """
    from PyQt6.QtCore import (
        QEasingCurve,
        QPointF,
        QPropertyAnimation,
        QRect,
        QRectF,
        Qt,
        pyqtSignal,
    )
    from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF
    from PyQt6.QtWidgets import QMenu, QWidget

    from .i18n import tr

    ui = ui or UIConfig()
    appearance = bubble_spec(ui)

    class BubbleWindow(QWidget):
        """圆角气泡窗；尾巴方向/显示模式可右键调整，边缘拖动调整大小。"""

        exit_requested = pyqtSignal()

        def __init__(self):
            super().__init__()
            self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.resize(360, 160)
            self.setMinimumSize(240, 120)
            self.setMouseTracking(True)  # 悬停显示边缘调整光标
            self._drag_pos = None
            self._resize_edge = ""
            self._tail = "S"  # 尾巴默认朝下（人物在下方）
            self._display_mode = "both"
            self._last_segment = None
            self._content_at = 0.0  # 最近一次真实内容（update_segment）的时刻
            self._appearance = appearance
            self._on_open_settings = on_open_settings
            self._ui_fade_mode = ui.bubble_fade if ui.bubble_fade in ("off", "dim", "hide") else "dim"
            self._ui_fade_min = max(0.0, min(1.0, ui.bubble_fade_min))
            self._idle = False

            # 淡出/淡入：用顶层窗口的 windowOpacity（QGraphicsOpacityEffect 会破坏
            # WA_TranslucentBackground，透明区域被栅格化成黑色——白天模式黑圈的根因）。
            # 淡出结束后真正隐藏窗口，消除分层窗口半透明中间态的边框伪影。
            self._fade = QPropertyAnimation(self, b"windowOpacity", self)
            self._fade.setDuration(400)
            self._fade.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self._fade.finished.connect(self._on_fade_finished)
            self._fading_out = False

            # 不用任何子控件：文本在 paintEvent 中与气泡同层直接绘制，
            # 消除子控件合成产生的分界线/色圈伪影；整个窗口天然可拖动
            self._source_text = ""
            self._translation_text = ""

        # ---- 外观 ----

        def set_appearance(self, spec) -> None:
            """外观即时更新（设置页预览/应用）。"""
            self._appearance = spec
            self.update()

        def set_fade_config(self, mode: str, fade_min: float) -> None:
            if mode in ("off", "dim", "hide"):
                self._ui_fade_mode = mode
            self._ui_fade_min = max(0.0, min(1.0, fade_min))

        # ---- 内容 ----

        @property
        def source_text(self) -> str:
            return self._source_text

        @property
        def translation_text(self) -> str:
            return self._translation_text

        def update_segment(self, segment) -> None:
            """更新当前句（实时=最新句，回放=当前播放句）；完整显示并淡入。"""
            self._last_segment = segment
            self._idle = False
            self._content_at = time.monotonic()  # 真实内容的时刻，待机切换前保持用
            source = segment.source or ""
            translation = segment.translation or ""
            if self._display_mode == "source_only":
                translation = ""
            elif self._display_mode == "translation_only":
                source = ""
            self._source_text = source
            self._translation_text = translation
            self.update()
            self.fade_in()

        def show_idle(self, text: str, hold_sec: float = 0.0) -> None:
            """待机：RMS 正常但 VAD 无人声 → 显示可配置的待机文字（不淡出）。
            文字变化（设置页预览/应用）时即时更新。
            hold_sec：刚识别的真实内容保持该时长后才切待机，避免字幕没看清就被顶掉。"""
            if self._idle and self._source_text == text:
                return
            if (
                not self._idle
                and self._last_segment is not None
                and hold_sec > 0
                and time.monotonic() - self._content_at < hold_sec
            ):
                return  # 保持期内：继续显示刚识别的字幕
            self._idle = True
            self._source_text = text
            self._translation_text = ""
            self.update()
            self.fade_in()

        def fade_out(self) -> None:
            """当前句播放完毕：按配置淡出（off 不动 / dim 降到低透明度保持 /
            hide 淡出后隐藏窗口——避免分数透明度中间态的合成伪影）。"""
            mode = self._fade_mode
            if mode == "off":
                return
            self._fade.stop()
            self._fading_out = True
            self._fade.setStartValue(self.windowOpacity())
            self._fade.setEndValue(0.0 if mode == "hide" else self._fade_min)
            self._fade.start()

        def fade_in(self) -> None:
            was_hidden = not self.isVisible()
            if was_hidden:
                self.setWindowOpacity(0.0)
                self.show()
            self._fade.stop()
            self._fading_out = False
            if not was_hidden and self.windowOpacity() >= 1.0:
                return
            self._fade.setStartValue(self.windowOpacity())
            self._fade.setEndValue(1.0)
            self._fade.start()

        def _on_fade_finished(self) -> None:
            # 只有淡出到隐藏才真正 hide；dim 模式停在低透明度
            if self._fading_out and self._fade_mode == "hide":
                self.hide()
                self.setWindowOpacity(1.0)  # 复位，下次淡入从 0 起
            self._fading_out = False

        @property
        def _fade_mode(self) -> str:
            return self._ui_fade_mode

        @property
        def _fade_min(self) -> float:
            return self._ui_fade_min

        # ---- 形状 ----

        def tail_direction(self) -> str:
            return self._tail

        def set_tail_direction(self, direction: str) -> None:
            if direction not in TAIL_DIRECTIONS:
                return
            self._tail = direction
            self.update()

        def display_mode(self) -> str:
            return self._display_mode

        def set_display_mode(self, mode: str) -> None:
            if mode in DISPLAY_MODES:
                self._display_mode = mode
                if self._last_segment is not None:
                    self.update_segment(self._last_segment)

        def _body_rect(self) -> QRectF:
            """气泡本体（不含尾巴带）：尾巴所在侧让出尾巴长度。"""
            w, h = self.width(), self.height()
            left = _PAD + (_TAIl_LEN if self._tail in ("W", "NW", "SW") else 0)
            right = w - _PAD - (_TAIl_LEN if self._tail in ("E", "NE", "SE") else 0)
            top = _PAD + (_TAIl_LEN if self._tail in ("N", "NW", "NE") else 0)
            bottom = h - _PAD - (_TAIl_LEN if self._tail in ("S", "SW", "SE") else 0)
            return QRectF(left, top, right - left, bottom - top)

        def _tail_polygon(self) -> QPolygonF:
            """尾巴三角形：底边深入本体 3px（消除描边缝隙），尖端伸入尾巴带。"""
            body = self._body_rect()
            cx, cy = body.center().x(), body.center().y()
            t = _PAD + _TAIl_LEN - 2  # 尖端位置（尾巴带外缘内侧）
            w, h = self.width(), self.height()
            d = self._tail
            if d == "N":
                pts = [(cx - 16, body.top() + 3), (cx + 16, body.top() + 3), (cx, t - _PAD)]
            elif d == "S":
                pts = [(cx - 16, body.bottom() - 3), (cx + 16, body.bottom() - 3), (cx, h - t + _PAD)]
            elif d == "W":
                pts = [(body.left() + 3, cy - 16), (body.left() + 3, cy + 16), (t - _PAD, cy)]
            elif d == "E":
                pts = [(body.right() - 3, cy - 16), (body.right() - 3, cy + 16), (w - t + _PAD, cy)]
            elif d == "NW":
                pts = [(body.left() + 14, body.top() + 3), (body.left() + 3, body.top() + 14), (t - _PAD, t - _PAD)]
            elif d == "NE":
                pts = [(body.right() - 14, body.top() + 3), (body.right() - 3, body.top() + 14), (w - t + _PAD, t - _PAD)]
            elif d == "SW":
                pts = [(body.left() + 14, body.bottom() - 3), (body.left() + 3, body.bottom() - 14), (t - _PAD, h - t + _PAD)]
            else:  # SE
                pts = [(body.right() - 14, body.bottom() - 3), (body.right() - 3, body.bottom() - 14), (w - t + _PAD, h - t + _PAD)]
            return QPolygonF([QPointF(x, y) for x, y in pts])

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            # 本体与尾巴做布尔 union 成单一路径：描边不出现内部共享边（无缝隙）
            body_path = QPainterPath()
            body_path.addRoundedRect(self._body_rect(), 16, 16)
            tail_path = QPainterPath()
            tail_path.addPolygon(self._tail_polygon())
            path = body_path.united(tail_path)
            theme = self._appearance.theme
            fill = QColor(theme.background_color)
            fill.setAlphaF(self._appearance.background_opacity)  # 不透明度独立生效
            painter.fillPath(path, fill)
            painter.setPen(QPen(QColor(theme.separator), 1.5))
            painter.drawPath(path)

            # 文本与气泡同层直接绘制：无任何子控件，绝无分界线/色圈
            content = self._body_rect().adjusted(14, 10, -14, -10)
            spacing = 8
            parts = []
            if self._source_text:
                parts.append((self._source_text, theme.source_font_size,
                              theme.source_color, False))
            if self._translation_text:
                parts.append((self._translation_text, theme.translation_font_size,
                              theme.translation_color, theme.translation_bold))
            if parts:
                flags = int(Qt.TextFlag.TextWordWrap)
                # 先量总高度，垂直居中（用各自字号的 QFontMetrics 量）
                heights = []
                for text, size, _color, bold in parts:
                    font = painter.font()
                    font.setPixelSize(size)
                    font.setBold(bold)
                    heights.append(
                        QFontMetrics(font)
                        .boundingRect(0, 0, int(content.width()), 10000, flags, text)
                        .height()
                    )
                total = sum(heights) + spacing * (len(parts) - 1)
                y = content.top() + max(0.0, (content.height() - total) / 2)
                for (text, size, color, bold), h in zip(parts, heights):
                    font = painter.font()
                    font.setPixelSize(size)
                    font.setBold(bold)
                    painter.setFont(font)
                    painter.setPen(QColor(color))
                    painter.drawText(
                        QRectF(content.left(), y, content.width(), h),
                        flags,
                        text,
                    )
                    y += h + spacing
            painter.end()

        # ---- 拖动移动 / 边缘调整大小 ----

        def _edge_at(self, pos) -> str:
            """判定调整方向：贴着气泡本体的可见描边（四边 + 四角）。

            不能按窗口外缘判定：透明留白与尾巴带是 alpha=0 像素，Windows
            分层窗口上鼠标事件会穿透到下层窗口，根本到不了这里。
            本体是实色填充，描边附近的事件必定可达。
            """
            body = self._body_rect()
            if not body.adjusted(-_EDGE, -_EDGE, _EDGE, _EDGE).contains(
                QPointF(pos.x(), pos.y())
            ):
                return ""
            on_left = abs(pos.x() - body.left()) <= _EDGE
            on_right = abs(pos.x() - body.right()) <= _EDGE
            on_top = abs(pos.y() - body.top()) <= _EDGE
            on_bottom = abs(pos.y() - body.bottom()) <= _EDGE
            edge = ""
            if on_top:
                edge += "top"
            if on_bottom:
                edge += "bottom"
            if on_left:
                edge += "left"
            if on_right:
                edge += "right"
            return edge

        @staticmethod
        def _cursor_for(edge: str):
            if edge in ("left", "right"):
                return Qt.CursorShape.SizeHorCursor
            if edge in ("top", "bottom"):
                return Qt.CursorShape.SizeVerCursor
            if edge in ("topleft", "bottomright"):
                return Qt.CursorShape.SizeFDiagCursor
            if edge in ("topright", "bottomleft"):
                return Qt.CursorShape.SizeBDiagCursor
            return None

        def _apply_resize(self, delta) -> None:
            """按按下时锁定的方向与起始几何调整（左/上边随拖动平移）。"""
            edge = self._resize_edge
            g = QRect(self._drag_geom)
            dw = delta.x() if "right" in edge else -delta.x() if "left" in edge else 0
            dh = delta.y() if "bottom" in edge else -delta.y() if "top" in edge else 0
            new_w = max(self.minimumWidth(), g.width() + dw)
            new_h = max(self.minimumHeight(), g.height() + dh)
            x = g.x() + (g.width() - new_w) if "left" in edge else g.x()
            y = g.y() + (g.height() - new_h) if "top" in edge else g.y()
            self.setGeometry(x, y, new_w, new_h)

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton:
                pos = event.position().toPoint()
                self._resize_edge = self._edge_at(pos)
                self._drag_pos = (
                    event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                )
                self._drag_geom = self.geometry()
                event.accept()

        def mouseMoveEvent(self, event) -> None:
            if self._drag_pos is None or not event.buttons() & Qt.MouseButton.LeftButton:
                # 未按下：悬停在描边附近时显示对应的调整光标
                cursor = self._cursor_for(self._edge_at(event.position().toPoint()))
                if cursor is not None:
                    self.setCursor(cursor)
                else:
                    self.unsetCursor()
                return
            if self._resize_edge:
                delta = event.globalPosition().toPoint() - (
                    self._drag_pos + self._drag_geom.topLeft()
                )
                self._apply_resize(delta)
            else:
                self.move(event.globalPosition().toPoint() - self._drag_pos)
            self.update()
            event.accept()

        def mouseReleaseEvent(self, event) -> None:
            self._drag_pos = None
            self._resize_edge = ""

        # ---- 右键菜单 ----

        def contextMenuEvent(self, event) -> None:
            menu = QMenu(self)
            menu.setStyleSheet(self._appearance.menu_stylesheet)
            tail_menu = menu.addMenu(tr("bubble.tail"))
            for d in TAIL_DIRECTIONS:
                action = tail_menu.addAction(tr(f"bubble.tail.{d}"))
                action.setCheckable(True)
                action.setChecked(d == self._tail)
                action.triggered.connect(
                    lambda _c=False, dd=d: self.set_tail_direction(dd)
                )
            display_menu = menu.addMenu(tr("bubble.display"))
            for m in DISPLAY_MODES:
                action = display_menu.addAction(tr(f"bubble.display.{m}"))
                action.setCheckable(True)
                action.setChecked(m == self._display_mode)
                action.triggered.connect(
                    lambda _c=False, mm=m: self.set_display_mode(mm)
                )
            menu.addSeparator()
            if self._on_open_settings is not None:
                appearance_action = menu.addAction(tr("bubble.appearance"))
                appearance_action.triggered.connect(self._on_open_settings)
            quit_action = menu.addAction(tr("bubble.exit"))
            quit_action.triggered.connect(self.exit_requested.emit)
            menu.exec(event.globalPos())

        def resizeEvent(self, event) -> None:
            self.update()
            super().resizeEvent(event)

    bubble = BubbleWindow()
    if on_exit is not None:
        bubble.exit_requested.connect(on_exit)
    return bubble
