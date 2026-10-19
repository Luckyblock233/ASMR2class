"""分层配置加载（需求文档 §4.1 配置管理）。

优先级：环境变量 > 语言档案 llm 覆盖 > config/config.yaml。
API key 支持环境变量 ASR2CLASS_LLM_API_KEY 覆盖，且不写入文件、不打印日志。
"""

from __future__ import annotations

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
class VADConfig:
    """VAD 后端配置（可插拔：silero | whisper_asmr）。"""

    backend: str = "silero"
    threshold: float = 0.5         # whisper_asmr 语音帧判定阈值（负阈值自动 -0.15 滞后）
    model_path: str = ""           # whisper_asmr 的 ONNX 路径（相对项目根自动锚定）
    device: str = "auto"           # whisper_asmr 推理设备：auto | cuda | cpu（auto = 有 CUDA EP 则用 GPU）
    recompute_ms: int = 1000       # whisper_asmr 增量推理：新增音频不足该间隔复用缓存（切句确认不受限）
    download: dict[str, Any] = field(default_factory=dict)  # hf_repo/hf_file/modelscope_repo/modelscope_file


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
    overlay: bool = True     # 悬浮字幕窗开关
    opacity: float = 0.85    # 初始透明度


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
    vad: VADConfig = field(default_factory=VADConfig)
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

    vad_raw = raw.get("vad") or {}
    vad = VADConfig(
        backend=str(vad_raw.get("backend") or "silero"),
        threshold=float(vad_raw.get("threshold") or 0.5),
        model_path=str(vad_raw.get("model_path") or ""),
        device=str(vad_raw.get("device") or "auto"),
        recompute_ms=int(vad_raw.get("recompute_ms") or 1000),
        download=vad_raw.get("download") or {},
    )
    if vad.model_path and not Path(vad.model_path).is_absolute():
        vad.model_path = str(root / vad.model_path)

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
    ui = UIConfig(
        overlay=bool(ui_raw.get("overlay") if ui_raw.get("overlay") is not None else True),
        opacity=float(ui_raw.get("opacity") or 0.85),
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
        vad=vad,
        context=ctx,
        ui=ui,
        default_profile=str(session_raw.get("default_profile") or "ja"),
    )


def save_llm_config(cfg: GlobalConfig, llm: LLMConfig) -> None:
    """设置向导写回 LLM 三项到 config.yaml（保留其余字段与注释以外的内容）。

    注意：yaml 往返会丢失注释；向导只在首次配置时运行，可接受。
    """
    path = cfg.config_file
    raw = _load_yaml(path)
    raw["llm"] = {
        "base_url": llm.base_url,
        "api_key": llm.api_key,
        "model": llm.model,
        "timeout_sec": llm.timeout_sec,
        "max_retries": llm.max_retries,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)


def save_provider_config(
    cfg: GlobalConfig, provider: ProviderConfig, slots: SlotsConfig
) -> None:
    """设置向导写回：新增/替换一个 provider 声明并更新槽位绑定。"""
    path = cfg.config_file
    raw = _load_yaml(path)
    providers_raw = [p for p in (raw.get("providers") or []) if isinstance(p, dict)]
    entry: dict[str, Any] = {
        "id": provider.id,
        "name": provider.name,
        "type": provider.type,
        "base_url": provider.base_url,
        "api_key": provider.api_key,
        "model": provider.model,
        "capabilities": provider.capabilities,
    }
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
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
