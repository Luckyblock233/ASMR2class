"""外观应用入口（M3 阶段 3）：悬浮窗、右键快捷面板和设置页共用同一份状态。

外观个性化收敛为两项：透明度 + 白天/夜间主题。两套主题为预定义
（THEMES），UIConfig 只存 theme 名与不透明度；颜色/字号不再逐项可配。
背景用 RGBA 实现半透明，不再让文字随 setWindowOpacity() 一起变淡。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import UIConfig


def _hex_rgb(color: str) -> tuple[int, int, int]:
    """'#RRGGBB' → (r, g, b)；非法输入回退深灰。"""
    value = (color or "").lstrip("#")
    try:
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    except (ValueError, IndexError):
        return 30, 30, 30


@dataclass(frozen=True)
class Theme:
    """一套预定义主题的全部视觉参数。"""

    name: str
    background_color: str
    source_color: str
    translation_color: str
    meta_color: str          # 时间戳/状态/C 栏等次要信息
    hover_bg: str            # 句块悬停底色
    current_bg: str          # 回放当前句高亮底色
    separator: str           # 句块分隔线
    status_ok: str
    status_err: str
    source_font_size: int = 15
    translation_font_size: int = 17   # 译文是主要阅读对象：更大、加粗（参考 LiveTrans）
    translation_bold: bool = True


THEMES: dict[str, Theme] = {
    "dark": Theme(
        name="dark",
        background_color="#1E1E1E",
        source_color="#FFFFFF",
        translation_color="#B8D8FF",
        meta_color="#9A9A9A",
        hover_bg="rgba(255, 255, 255, 24)",
        current_bg="rgba(120, 170, 255, 38)",
        separator="rgba(255, 255, 255, 16)",
        status_ok="#9F9",
        status_err="#F66",
    ),
    "light": Theme(
        name="light",
        background_color="#FAFAFA",
        source_color="#1A1A1A",
        translation_color="#1B4F8A",
        meta_color="#666666",
        hover_bg="rgba(0, 0, 0, 14)",
        current_bg="rgba(30, 100, 200, 30)",
        separator="rgba(0, 0, 0, 22)",
        status_ok="#1B7A2E",
        status_err="#C62828",
    ),
}

THEME_NAMES = tuple(THEMES)
DEFAULT_THEME = "dark"


def get_theme(name: str) -> Theme:
    """按名取主题；未知名回退深色。"""
    return THEMES.get(name) or THEMES[DEFAULT_THEME]


# C 栏/句内延迟的阈值着色：≤good 绿、good–warn 黄、>warn 红
HEALTH_GOOD = "#7ED957"
HEALTH_WARN = "#FFC94D"
HEALTH_CRIT = "#FF6B6B"


def health_color(value: float, good: float, warn: float) -> str:
    if value <= good:
        return HEALTH_GOOD
    if value <= warn:
        return HEALTH_WARN
    return HEALTH_CRIT


@dataclass(frozen=True)
class AppearanceSpec:
    """UIConfig + 主题解析后的不可变外观快照（控件直接消费）。"""

    theme: Theme = field(default_factory=lambda: THEMES[DEFAULT_THEME])
    background_opacity: float = 0.85
    line_spacing: float = 1.15
    compact_mode: bool = False
    # 延迟阈值（ms）：句内 ASR/TL 小字与监控行共用同一语义
    latency_good_ms: int = 500
    latency_warn_ms: int = 1000

    @classmethod
    def from_ui(cls, ui: UIConfig) -> "AppearanceSpec":
        return cls(
            theme=get_theme(ui.theme),
            background_opacity=ui.effective_background_opacity,
            line_spacing=ui.line_spacing,
            compact_mode=ui.compact_mode,
            latency_good_ms=ui.latency_good_ms,
            latency_warn_ms=ui.latency_warn_ms,
        )

    @property
    def background_rgba(self) -> str:
        r, g, b = _hex_rgb(self.theme.background_color)
        alpha = round(max(0.2, min(1.0, self.background_opacity)) * 255)
        return f"rgba({r}, {g}, {b}, {alpha})"

    def _text_style(self, size: int, color: str, bold: bool = False) -> str:
        weight = " font-weight: 600;" if bold else ""
        return f"color: {color}; font-size: {size}px; background: transparent;{weight}"

    @property
    def source_style(self) -> str:
        return self._text_style(self.theme.source_font_size, self.theme.source_color)

    @property
    def translation_style(self) -> str:
        return self._text_style(
            self.theme.translation_font_size,
            self.theme.translation_color,
            bold=self.theme.translation_bold,
        )

    @property
    def meta_style(self) -> str:
        return f"color: {self.theme.meta_color}; background: transparent;"

    def latency_span_style(self, ms: float) -> str:
        """句内 ASR/TL 延迟小字样式：dim 小字 + 阈值着色。"""
        return (
            f"color: {health_color(ms, self.latency_good_ms, self.latency_warn_ms)};"
            " font-size: 12px;"
        )

    @property
    def block_stylesheet(self) -> str:
        t = self.theme
        return (
            "SentenceBlock {"
            f" border-bottom: 1px solid {t.separator};"
            " padding-bottom: 3px; }"
            'SentenceBlock[hovered="true"] {'
            f" background-color: {t.hover_bg}; border-radius: 4px; }}"
            'SentenceBlock[current="true"] {'
            f" background-color: {t.current_bg}; border-radius: 4px; }}"
        )

    @property
    def block_margins(self) -> tuple[int, int, int, int]:
        """句块内边距（紧凑模式收窄）。"""
        return (4, 2, 4, 2) if self.compact_mode else (6, 4, 6, 4)

    @property
    def block_spacing(self) -> int:
        return max(1, round(2 * self.line_spacing))

    @property
    def bars_stylesheet(self) -> str:
        """A/B/C 栏控件的随主题样式（标签/按钮/下拉框文字与边框）。

        挂在 centralHost 上：设置过自身 stylesheet 的控件（C 栏指标、句块文本等）
        以各自规则优先，不受此影响。
        """
        t = self.theme
        return (
            f"#centralHost QLabel {{ color: {t.source_color}; background: transparent; }}"
            f"#centralHost QPushButton, #centralHost QToolButton {{"
            f" color: {t.source_color}; background: transparent;"
            f" border: 1px solid {t.separator}; border-radius: 4px;"
            " padding: 2px 6px; font-size: 12px; }"
            f"#centralHost QPushButton:hover, #centralHost QToolButton:hover {{"
            f" background: {t.hover_bg}; }}"
            f"#centralHost QPushButton:disabled, #centralHost QToolButton:disabled {{"
            f" color: {t.meta_color}; }}"
            f"#centralHost QComboBox {{ color: {t.source_color}; background: transparent;"
            f" border: 1px solid {t.separator}; border-radius: 4px;"
            " padding: 1px 4px; font-size: 12px; }"
            f"#centralHost QComboBox:disabled {{ color: {t.meta_color}; }}"
            f"#centralHost QComboBox QAbstractItemView {{ background: {t.background_color};"
            f" color: {t.source_color}; selection-background-color: {t.current_bg}; }}"
            f"#centralHost QProgressBar {{ color: {t.meta_color}; background: transparent;"
            f" border: 1px solid {t.separator}; border-radius: 3px;"
            " text-align: center; }"  # 不写 font-size：Windows 上 QSS px 字号会触发 QFont::setPointSize(-1) 告警
            f"#centralHost QProgressBar::chunk {{ background: {t.status_ok}; border-radius: 2px; }}"
            # 句块右键菜单等弹窗（QMenu 是 centralHost 的后代，继承其样式表）
            f"#centralHost QMenu {{ background: {t.background_color};"
            f" color: {t.source_color}; border: 1px solid {t.separator}; }}"
            f"#centralHost QMenu::item:selected {{ background: {t.hover_bg}; }}"
            f"#centralHost QMenu::item:disabled {{ color: {t.meta_color}; }}"
        )

    @property
    def menu_stylesheet(self) -> str:
        """主菜单/托盘菜单（顶层弹窗，不继承 centralHost 规则）。"""
        t = self.theme
        return (
            f"QMenu {{ background: {t.background_color}; color: {t.source_color};"
            f" border: 1px solid {t.separator}; }}"
            f"QMenu::item:selected {{ background: {t.hover_bg}; }}"
        )
