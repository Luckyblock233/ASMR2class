"""分层配置加载（需求文档 §4.1 配置管理）。

优先级：环境变量 > 语言档案 llm 覆盖 > config/config.yaml。
API key 支持环境变量 ASR2CLASS_LLM_API_KEY 覆盖，且不写入文件、不打印日志。
"""

from __future__ import annotations

import copy
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

API_KEY_ENV = "ASR2CLASS_LLM_API_KEY"


def find_project_root(start: Path | None = None) -> Path:
    """向上查找项目根目录（以 config/config.yaml 或 config/config.example.yaml 为标记）。"""
    cur = (start or Path.cwd()).resolve()
    for d in (cur, *cur.parents):
        if (d / "config" / "config.yaml").is_file() or (
            d / "config" / "config.example.yaml"
        ).is_file():
            return d
    raise FileNotFoundError("未找到 config/config.example.yaml，请在项目根目录下运行")


@dataclass
class ProviderLocalConfig:
    """本地托管 Provider（llama.cpp server）配置。所有路径/参数均来自配置，代码零硬编码。"""

    managed: bool = True           # 应用负责拉起/回收子进程；false = 只连端点
    gguf_path: str = ""
    server_bin: str = "llama-server"  # 可填绝对路径；默认从 PATH 查找
    server_args: str = ""          # 透传给 llama-server 的额外参数
    vram_budget_mb: int = 1500     # 启动前显存预检预算（不足时警告，不静默抢占）
    download: dict[str, Any] = field(default_factory=dict)  # hf_repo/hf_file/modelscope_repo/modelscope_file


@dataclass
class ProviderConfig:
    """模型提供方声明（模型可插拔架构 §2）。id 唯一，槽位按 id 绑定。"""

    id: str = ""
    name: str = ""
    type: str = "openai_compatible"  # openai_compatible | llama_cpp_server | vllm
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    capabilities: list[str] = field(default_factory=list)  # translation | grammar | glossary_constraint
    local: ProviderLocalConfig | None = None
    params: dict[str, Any] = field(default_factory=dict)   # 请求参数透传（temperature/max_tokens/enable_thinking…）
    timeout_sec: float = 30.0
    max_retries: int = 3

    @property
    def is_complete(self) -> bool:
        return bool(self.id and self.base_url and self.model)

    def to_llm_config(self) -> "LLMConfig":
        return LLMConfig(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            timeout_sec=self.timeout_sec,
            max_retries=self.max_retries,
        )


@dataclass
class SlotsConfig:
    """模型槽位绑定：值均为 provider id，空 = 未绑定（对应功能停用）。"""

    translation: str = ""
    grammar: str = ""


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_sec: float = 30.0
    max_retries: int = 3

    @property
    def is_complete(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def merged(self, override: dict[str, Any] | None) -> "LLMConfig":
        """应用语言档案级覆盖（FR-1.7），未覆盖项回落到全局。"""
        if not override:
            return self
        return LLMConfig(
            base_url=override.get("base_url") or self.base_url,
            api_key=override.get("api_key") or self.api_key,
            model=override.get("model") or self.model,
            timeout_sec=float(override.get("timeout_sec") or self.timeout_sec),
            max_retries=int(override.get("max_retries") or self.max_retries),
        )


@dataclass
class ASRConfig:
    device: str = "auto"  # auto | cuda | cpu
    model_size: str = "small"
    compute_type: str = ""  # 空 = 按设备自动选择
    vram_budget_mb: int = 2500
    vram_min_free_mb: int = 2000
    beam_size: int = 5
    # 幻觉门控：静音/噪声段 Whisper 会幻觉出视频结尾高频字幕句（如「ご視聴ありがとう…」），
    # 同时满足 no_speech_prob > 阈值 且 avg_logprob < 阈值的段被丢弃
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4  # 重复性幻觉（压缩比异常高）直接丢弃
    condition_on_previous: bool = False  # True 会让幻觉句把后续段落带偏，流式场景应关闭
    # 温度回退链：低质量窗口（no_speech/logprob/compression 触发）按链升温重试
    temperature_fallback: list[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    # 幻觉短语黑名单：形态固定的视频结尾客套话（日/英），归一化后包含匹配即丢弃
    hallucination_phrases: list[str] = field(default_factory=lambda: [
        "視聴してくださって", "ご視聴ありがとう", "最後までご視聴", "最後まで見てくださって",
        "チャンネル登録", "字幕は自動生成", "thank you for watching", "thanks for watching",
        "please subscribe",
    ])


@dataclass
class AudioConfig:
    source: str = "mic"  # mic | loopback
    sample_rate: int = 16000
    block_ms: int = 30
    loopback_device: str = ""


@dataclass
class SegmentationConfig:
    silence_ms: int = 500
    max_segment_sec: float = 15.0


@dataclass
class BlockConfig:
    max_sentences: int = 8          # 块句数上限（离线模式放宽到 20，FR-C2.5）
    max_sentences_offline: int = 20
    silence_gap_sec: float = 1.5    # 静音间隙超过该阈值即成块


@dataclass
class MemoryConfig:
    enabled: bool = True
    glossary_max_entries: int = 100
    style_max_chars: int = 200


@dataclass
class UIConfig:
    """界面与外观配置（M3 后续 A–D 悬浮窗）。

    外观个性化收敛为两项：背景不透明度 + 白天/夜间主题（`theme`）。
    颜色/字号由主题预定义，不再逐项可配。
    opacity 为旧字段（整窗透明，文字一起变淡）；新实现用 RGBA 背景，
    background_opacity 留空时把 opacity 当作迁移输入。
    """

    overlay: bool = True     # 悬浮字幕窗开关
    opacity: float = 0.85    # 旧版整窗透明度（仅作 background_opacity 的迁移输入）
    theme: str = "dark"      # 主题：dark | light（预定义配色与字号）
    background_opacity: float | None = None  # 背景不透明度；None = 沿用 opacity
    line_spacing: float = 1.15             # 行距倍数
    # 窗口
    window_width: int = 720
    window_height: int = 480
    # A/B/C 栏显隐（A=控制栏 B=模型/档案栏 C=状态栏）
    show_bar_a: bool = True
    show_bar_b: bool = True
    show_bar_c: bool = True
    auto_scroll: bool = True               # 新句自动滚动到底（用户上翻时自动暂停跟随）
    compact_mode: bool = False             # 紧凑模式（隐藏 B/C 栏、缩小边距）
    ui_language: str = "zh_CN"             # UI 文案语言：zh_CN | en
    view_capacity: int = 200               # 实时视图句块控件容量上限（文档仍完整保存）
    # C 栏健康阈值（仪表盘颜色：≤good 绿、good–warn 黄、>warn 红）
    latency_good_ms: int = 500             # ASR/翻译平均耗时良好阈值
    latency_warn_ms: int = 1000            # 超过该值显示红色
    ram_warn_percent: int = 85             # RAM 占用超过该百分比显示橙色
    # 气泡模式（漫画对话框）独立外观：空/None/0 = 跟随主悬浮窗
    bubble_theme: str = ""                 # dark | light | 空（跟随主主题）
    bubble_opacity: float | None = None    # 气泡背景不透明度（None = 跟随）
    bubble_font_size: int = 0              # 气泡字号（0 = 主题默认）
    # 气泡淡出：off（不淡出）| dim（降到低透明度保持）| hide（淡出后隐藏）
    bubble_fade: str = "dim"
    bubble_fade_min: float = 0.25          # dim 模式的最低透明度
    # 气泡淡出触发（实时模式）：RMS 低于阈值 → 淡出；
    # RMS 正常但 VAD 无人声 → 显示待机文字；均需持续超过延时才触发（防抖）
    bubble_rms_threshold: float = 0.01
    bubble_idle_text: str = "……"           # 待机文字（可自定义，如 ♪）
    bubble_fade_delay_sec: float = 1.5     # 低于阈值的持续延时（秒）
    # 刚识别的字幕进入「有声无人声」片段时保持显示的时长（秒），
    # 避免字幕刚出来就被待机文字顶掉；0 = 立即切换
    bubble_idle_hold_sec: float = 3.0

    @property
    def effective_background_opacity(self) -> float:
        """背景不透明度：新字段优先，否则迁移旧 opacity。"""
        value = self.background_opacity if self.background_opacity is not None else self.opacity
        return max(0.2, min(1.0, value))

    def merged(self, override: dict[str, Any] | None) -> "UIConfig":
        """应用语言档案级外观覆盖，未覆盖项回落全局；空覆盖返回自身。"""
        if not override:
            return self
        merged = copy.copy(self)
        for key, value in override.items():
            if not hasattr(merged, key) or value is None:
                continue  # 未知/空字段忽略，保持向前兼容
            current = getattr(merged, key)
            if isinstance(current, bool):
                value = bool(value)
            elif isinstance(current, int) and not isinstance(current, bool):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            setattr(merged, key, value)
        if merged.background_opacity is not None:
            merged.background_opacity = max(0.2, min(1.0, merged.background_opacity))
        merged.opacity = max(0.2, min(1.0, merged.opacity))
        return merged


@dataclass
class ContextConfig:
    """上下文感知翻译配置（FR-C4.2：三项机制独立开关）。"""

    tail_sentences: int = 3         # 0 = 关闭滑动窗口
    history_max_tokens: int = 512
    block: BlockConfig = field(default_factory=BlockConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @property
    def window_enabled(self) -> bool:
        return self.tail_sentences > 0

    def merged(self, override: dict[str, Any] | None) -> "ContextConfig":
        """应用语言档案级 context 覆盖（FR-C1.2），未覆盖项回落全局。"""
        if not override:
            return self
        blk = override.get("block") or {}
        mem = override.get("memory") or {}
        return ContextConfig(
            tail_sentences=int(override.get("tail_sentences", self.tail_sentences)),
            history_max_tokens=int(override.get("history_max_tokens", self.history_max_tokens)),
            block=BlockConfig(
                max_sentences=int(blk.get("max_sentences", self.block.max_sentences)),
                max_sentences_offline=int(blk.get("max_sentences_offline", self.block.max_sentences_offline)),
                silence_gap_sec=float(blk.get("silence_gap_sec", self.block.silence_gap_sec)),
            ),
            memory=MemoryConfig(
                enabled=bool(mem.get("enabled", self.memory.enabled)),
                glossary_max_entries=int(mem.get("glossary_max_entries", self.memory.glossary_max_entries)),
                style_max_chars=int(mem.get("style_max_chars", self.memory.style_max_chars)),
            ),
        )


@dataclass
class GlobalConfig:
    root: Path
    llm: LLMConfig = field(default_factory=LLMConfig)  # 简易模式/向后兼容（providers 为空时合成 default provider）
    providers: list[ProviderConfig] = field(default_factory=list)
    slots: SlotsConfig = field(default_factory=SlotsConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    default_profile: str = "ja"

    @property
    def config_file(self) -> Path:
        return self.root / "config" / "config.yaml"

    @property
    def config_example_file(self) -> Path:
        return self.root / "config" / "config.example.yaml"

    @property
    def languages_dir(self) -> Path:
        return self.root / "config" / "languages"

    @property
    def prompts_dir(self) -> Path:
        return self.root / "config" / "prompts"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _float_or(value: Any, default: float) -> float:
    """可空 float 解析：None 用默认值，0 等合法 falsy 值原样保留。"""
    return default if value is None else float(value)


def ensure_config_file(root: Path) -> Path:
    """首次运行：config.yaml 不存在时从 config.example.yaml 复制创建。"""
    path = root / "config" / "config.yaml"
    if not path.is_file():
        example = root / "config" / "config.example.yaml"
        if not example.is_file():
            raise FileNotFoundError(f"配置模板缺失：{example}")
        shutil.copyfile(example, path)
        print(f"已根据模板创建本地配置：{path}（该文件已 gitignore，可安全存放密钥）")
    return path


def _parse_provider(raw: dict[str, Any]) -> ProviderConfig:
    local_raw = raw.get("local")
    local = None
    if isinstance(local_raw, dict):
        local = ProviderLocalConfig(
            managed=bool(local_raw.get("managed") if local_raw.get("managed") is not None else True),
            gguf_path=str(local_raw.get("gguf_path") or ""),
            server_bin=str(local_raw.get("server_bin") or "llama-server"),
            server_args=str(local_raw.get("server_args") or ""),
            vram_budget_mb=int(local_raw.get("vram_budget_mb") or 1500),
            download=local_raw.get("download") or {},
        )
    caps = raw.get("capabilities") or []
    return ProviderConfig(
        id=str(raw.get("id") or ""),
        name=str(raw.get("name") or raw.get("id") or ""),
        type=str(raw.get("type") or "openai_compatible"),
        base_url=str(raw.get("base_url") or ""),
        api_key=str(raw.get("api_key") or ""),
        model=str(raw.get("model") or ""),
        capabilities=[str(c) for c in caps],
        local=local,
        params=raw.get("params") or {},
        timeout_sec=float(raw.get("timeout_sec") or 30.0),
        max_retries=int(raw.get("max_retries") or 3),
    )


def load_config(root: Path | None = None) -> GlobalConfig:
    root = (root or find_project_root()).resolve()
    raw = _load_yaml(ensure_config_file(root))

    llm_raw = raw.get("llm") or {}
    llm = LLMConfig(
        base_url=str(llm_raw.get("base_url") or ""),
        api_key=str(llm_raw.get("api_key") or ""),
        model=str(llm_raw.get("model") or ""),
        timeout_sec=float(llm_raw.get("timeout_sec") or 30.0),
        max_retries=int(llm_raw.get("max_retries") or 3),
    )
    # 环境变量覆盖密钥（FR：环境变量可覆盖密钥；key 不入库）。
    # 同时填充 api_key 为空的 provider（向导选择「不写入文件」时的每次运行注入路径）。
    env_key = os.environ.get(API_KEY_ENV, "").strip()
    if env_key:
        llm.api_key = env_key

    asr_raw = raw.get("asr") or {}
    asr = ASRConfig(
        device=str(asr_raw.get("device") or "auto"),
        model_size=str(asr_raw.get("model_size") or "small"),
        compute_type=str(asr_raw.get("compute_type") or ""),
        vram_budget_mb=int(asr_raw.get("vram_budget_mb") or 2500),
        vram_min_free_mb=int(asr_raw.get("vram_min_free_mb") or 2000),
        beam_size=int(asr_raw.get("beam_size") or 5),
        no_speech_threshold=float(asr_raw.get("no_speech_threshold") or 0.6),
        log_prob_threshold=float(asr_raw.get("log_prob_threshold") or -1.0),
        compression_ratio_threshold=float(asr_raw.get("compression_ratio_threshold") or 2.4),
        condition_on_previous=bool(asr_raw.get("condition_on_previous") or False),
        temperature_fallback=[float(t) for t in asr_raw.get("temperature_fallback") or ASRConfig().temperature_fallback],
        hallucination_phrases=[str(p) for p in asr_raw.get("hallucination_phrases") or ASRConfig().hallucination_phrases],
    )

    audio_raw = raw.get("audio") or {}
    audio = AudioConfig(
        source=str(audio_raw.get("source") or "mic"),
        sample_rate=int(audio_raw.get("sample_rate") or 16000),
        block_ms=int(audio_raw.get("block_ms") or 30),
        loopback_device=str(audio_raw.get("loopback_device") or ""),
    )

    seg_raw = raw.get("segmentation") or {}
    seg = SegmentationConfig(
        silence_ms=int(seg_raw.get("silence_ms") or 500),
        max_segment_sec=float(seg_raw.get("max_segment_sec") or 15.0),
    )

    ctx_raw = raw.get("context") or {}
    blk_raw = ctx_raw.get("block") or {}
    mem_raw = ctx_raw.get("memory") or {}
    ctx = ContextConfig(
        tail_sentences=int(ctx_raw.get("tail_sentences") if ctx_raw.get("tail_sentences") is not None else 3),
        history_max_tokens=int(ctx_raw.get("history_max_tokens") or 512),
        block=BlockConfig(
            max_sentences=int(blk_raw.get("max_sentences") or 8),
            max_sentences_offline=int(blk_raw.get("max_sentences_offline") or 20),
            silence_gap_sec=float(blk_raw.get("silence_gap_sec") or 1.5),
        ),
        memory=MemoryConfig(
            enabled=bool(mem_raw.get("enabled") if mem_raw.get("enabled") is not None else True),
            glossary_max_entries=int(mem_raw.get("glossary_max_entries") or 100),
            style_max_chars=int(mem_raw.get("style_max_chars") or 200),
        ),
    )

    providers = [_parse_provider(p) for p in (raw.get("providers") or []) if isinstance(p, dict)]
    providers = [p for p in providers if p.id]
    # 相对 gguf_path 锚定到项目根，避免依赖运行时工作目录
    for p in providers:
        if p.local and p.local.gguf_path and not Path(p.local.gguf_path).is_absolute():
            p.local.gguf_path = str(root / p.local.gguf_path)
    slots_raw = raw.get("slots") or {}
    slots = SlotsConfig(
        translation=str(slots_raw.get("translation") or ""),
        grammar=str(slots_raw.get("grammar") or ""),
    )
    # 向后兼容：未声明 providers 且旧 llm: 节完整时，合成一个 default provider 并绑到两槽
    if not providers and llm.is_complete:
        providers = [
            ProviderConfig(
                id="default",
                name="全局 llm 配置（简易模式）",
                type="openai_compatible",
                base_url=llm.base_url,
                api_key=llm.api_key,
                model=llm.model,
                capabilities=["translation", "grammar"],
                timeout_sec=llm.timeout_sec,
                max_retries=llm.max_retries,
            )
        ]
        if not slots.translation:
            slots.translation = "default"
        if not slots.grammar:
            slots.grammar = "default"
    if env_key:
        for p in providers:
            if not p.api_key:
                p.api_key = env_key

    ui_raw = raw.get("ui") or {}
    bg_opacity_raw = ui_raw.get("background_opacity")
    # 旧配置里的颜色/字体字段（background_color/source_color 等）已被主题取代，
    # 解析时忽略；保存时不再写回。
    ui = UIConfig(
        overlay=bool(ui_raw.get("overlay") if ui_raw.get("overlay") is not None else True),
        opacity=float(ui_raw.get("opacity") or 0.85),
        theme=str(ui_raw.get("theme") or "dark"),
        background_opacity=float(bg_opacity_raw) if bg_opacity_raw is not None else None,
        line_spacing=float(ui_raw.get("line_spacing") or 1.15),
        window_width=int(ui_raw.get("window_width") or 720),
        window_height=int(ui_raw.get("window_height") or 480),
        show_bar_a=bool(ui_raw.get("show_bar_a") if ui_raw.get("show_bar_a") is not None else True),
        show_bar_b=bool(ui_raw.get("show_bar_b") if ui_raw.get("show_bar_b") is not None else True),
        show_bar_c=bool(ui_raw.get("show_bar_c") if ui_raw.get("show_bar_c") is not None else True),
        auto_scroll=bool(ui_raw.get("auto_scroll") if ui_raw.get("auto_scroll") is not None else True),
        compact_mode=bool(ui_raw.get("compact_mode") or False),
        ui_language=str(ui_raw.get("ui_language") or "zh_CN"),
        view_capacity=int(ui_raw.get("view_capacity") or 200),
        latency_good_ms=int(ui_raw.get("latency_good_ms") or 500),
        latency_warn_ms=int(ui_raw.get("latency_warn_ms") or 1000),
        ram_warn_percent=int(ui_raw.get("ram_warn_percent") or 85),
        bubble_theme=str(ui_raw.get("bubble_theme") or ""),
        bubble_opacity=(
            float(ui_raw["bubble_opacity"])
            if ui_raw.get("bubble_opacity") is not None
            else None
        ),
        bubble_font_size=int(ui_raw.get("bubble_font_size") or 0),
        bubble_fade=str(ui_raw.get("bubble_fade") or "dim"),
        # 0 是合法配置值（如延时 0 = 立即触发），不能用 `or 默认值` 兜底
        bubble_fade_min=_float_or(ui_raw.get("bubble_fade_min"), 0.25),
        bubble_rms_threshold=_float_or(ui_raw.get("bubble_rms_threshold"), 0.01),
        bubble_idle_text=str(ui_raw.get("bubble_idle_text") or "……"),
        bubble_fade_delay_sec=_float_or(ui_raw.get("bubble_fade_delay_sec"), 1.5),
        bubble_idle_hold_sec=_float_or(ui_raw.get("bubble_idle_hold_sec"), 3.0),
    )

    session_raw = raw.get("session") or {}
    return GlobalConfig(
        root=root,
        llm=llm,
        providers=providers,
        slots=slots,
        asr=asr,
        audio=audio,
        segmentation=seg,
        context=ctx,
        ui=ui,
        default_profile=str(session_raw.get("default_profile") or "ja"),
    )


def _atomic_write_yaml(path: Path, raw: dict[str, Any]) -> None:
    """同目录临时文件 + 原子替换：保存中途崩溃不破坏现有配置。"""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _persistable_api_key(value: str) -> str:
    """环境变量注入的 API key 不落盘：值与环境变量一致时写回空串。"""
    env_key = os.environ.get(API_KEY_ENV, "").strip()
    if env_key and value == env_key:
        return ""
    return value


def save_llm_config(cfg: GlobalConfig, llm: LLMConfig) -> None:
    """设置向导写回 LLM 三项到 config.yaml（原子写入，保留其余字段）。

    注意：yaml 往返会丢失注释；向导只在首次配置时运行，可接受。
    环境变量注入的密钥不会写入文件。
    """
    path = cfg.config_file
    raw = _load_yaml(path)
    raw["llm"] = {
        "base_url": llm.base_url,
        "api_key": _persistable_api_key(llm.api_key),
        "model": llm.model,
        "timeout_sec": llm.timeout_sec,
        "max_retries": llm.max_retries,
    }
    _atomic_write_yaml(path, raw)


def save_ui_config(cfg: GlobalConfig, ui: UIConfig) -> None:
    """外观设置原子写回 config.yaml 的 ui 节；保留其余配置与旧 opacity 字段。"""
    path = cfg.config_file
    raw = _load_yaml(path)
    ui_raw = raw.get("ui")
    if not isinstance(ui_raw, dict):
        ui_raw = {}
    ui_raw.update(
        {
            "overlay": ui.overlay,
            "theme": ui.theme,
            "background_opacity": round(ui.effective_background_opacity, 3),
            "line_spacing": ui.line_spacing,
            "window_width": ui.window_width,
            "window_height": ui.window_height,
            "show_bar_a": ui.show_bar_a,
            "show_bar_b": ui.show_bar_b,
            "show_bar_c": ui.show_bar_c,
            "auto_scroll": ui.auto_scroll,
            "compact_mode": ui.compact_mode,
            "ui_language": ui.ui_language,
            "view_capacity": ui.view_capacity,
            "latency_good_ms": ui.latency_good_ms,
            "latency_warn_ms": ui.latency_warn_ms,
            "ram_warn_percent": ui.ram_warn_percent,
            "bubble_theme": ui.bubble_theme,
            "bubble_opacity": ui.bubble_opacity,
            "bubble_font_size": ui.bubble_font_size,
            "bubble_fade": ui.bubble_fade,
            "bubble_fade_min": ui.bubble_fade_min,
            "bubble_rms_threshold": ui.bubble_rms_threshold,
            "bubble_idle_text": ui.bubble_idle_text,
            "bubble_fade_delay_sec": ui.bubble_fade_delay_sec,
            "bubble_idle_hold_sec": ui.bubble_idle_hold_sec,
        }
    )
    # 主题取代逐项颜色/字体配置：保存时清掉旧字段
    for legacy in (
        "background_color", "source_font_family", "source_font_size", "source_color",
        "translation_font_family", "translation_font_size", "translation_color",
    ):
        ui_raw.pop(legacy, None)
    raw["ui"] = ui_raw
    _atomic_write_yaml(path, raw)


def save_slots_config(cfg: GlobalConfig, slots: SlotsConfig) -> None:
    """只写回槽位绑定（原子写入），不触碰 providers 节。"""
    path = cfg.config_file
    raw = _load_yaml(path)
    raw["slots"] = {"translation": slots.translation, "grammar": slots.grammar}
    _atomic_write_yaml(path, raw)


def delete_provider_config(cfg: GlobalConfig, provider_id: str) -> None:
    """从 config.yaml 删除 provider 声明（原子写入）；槽位绑定由调用方先行解除。"""
    path = cfg.config_file
    raw = _load_yaml(path)
    providers_raw = [p for p in (raw.get("providers") or []) if isinstance(p, dict)]
    raw["providers"] = [p for p in providers_raw if p.get("id") != provider_id]
    _atomic_write_yaml(path, raw)


def save_provider_config(
    cfg: GlobalConfig, provider: ProviderConfig, slots: SlotsConfig
) -> None:
    """设置向导写回：新增/替换一个 provider 声明并更新槽位绑定（原子写入）。

    已存在的同 id 条目保留未知字段与 timeout/max_retries 等，只覆盖本函数负责的键；
    环境变量注入的密钥不会写入文件。
    """
    path = cfg.config_file
    raw = _load_yaml(path)
    providers_raw = [p for p in (raw.get("providers") or []) if isinstance(p, dict)]
    existing = next((p for p in providers_raw if p.get("id") == provider.id), None)
    entry: dict[str, Any] = dict(existing) if existing else {}
    entry.update(
        {
            "id": provider.id,
            "name": provider.name,
            "type": provider.type,
            "base_url": provider.base_url,
            "api_key": _persistable_api_key(provider.api_key),
            "model": provider.model,
            "capabilities": provider.capabilities,
            "timeout_sec": provider.timeout_sec,
            "max_retries": provider.max_retries,
        }
    )
    if provider.params:
        entry["params"] = provider.params
    if provider.local is not None:
        entry["local"] = {
            "managed": provider.local.managed,
            "gguf_path": provider.local.gguf_path,
            "server_bin": provider.local.server_bin,
            "server_args": provider.local.server_args,
            "vram_budget_mb": provider.local.vram_budget_mb,
            "download": provider.local.download,
        }
    providers_raw = [p for p in providers_raw if p.get("id") != provider.id]
    providers_raw.append(entry)
    raw["providers"] = providers_raw
    raw["slots"] = {"translation": slots.translation, "grammar": slots.grammar}
    _atomic_write_yaml(path, raw)
