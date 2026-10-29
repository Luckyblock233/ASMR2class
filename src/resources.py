"""启动前资源检查与按需下载（启动器流程：确认 → 下载 → 健康检查 → 正式启动）。

资源 = ASR 模型（faster-whisper，下载到 HF 缓存）+ 本地托管 Provider 的 GGUF
（下载到项目 models/ 目录，路径来自 provider 的 local.gguf_path）。
云端 Provider 无可下载资源，其可用性由健康检查覆盖（缺 API key 会在此暴露）。
回放模式不产字幕，无需任何资源。代码不含任何具体模型/服务商名，全部来自配置。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .asr.engine import MIRROR_ENDPOINT, _is_network_error, _model_cached
from .config import GlobalConfig, ProviderConfig

log = logging.getLogger(__name__)


@dataclass
class ResourceNeed:
    """一项缺失的待下载资源。"""

    kind: str  # "whisper" | "gguf"
    label: str  # 显示名（如 "ASR 模型 faster-whisper（small）"）
    detail: str  # 来源/目标说明
    provider_id: str = ""  # kind=gguf 时的 provider id


def whisper_repo_id(model_size: str) -> str:
    """model_size → HF 仓库 id（本地路径/已带命名空间的 id 原样返回）。"""
    if os.path.sep in model_size or "/" in model_size:
        return model_size
    return f"Systran/faster-whisper-{model_size}"


def whisper_present(model_size: str) -> bool:
    """ASR 模型是否已在本地（HF 缓存或本地路径）。"""
    return _model_cached(model_size)


def _needed_slots(grammar_enabled: bool) -> list[str]:
    from .nlp.providers import SLOT_GRAMMAR, SLOT_TRANSLATION

    slots = [SLOT_TRANSLATION]
    if grammar_enabled:
        slots.append(SLOT_GRAMMAR)
    return slots


def bound_providers(
    cfg: GlobalConfig, grammar_enabled: bool, slots_override=None
) -> list[ProviderConfig]:
    """启动所需槽位绑定的 provider 配置（去重，保持 翻译→语法 顺序）。

    启动器的自由语言对模式下档案由语言包合成、无 slots 覆盖，按全局 slots 解析；
    slots_override 传入启动器中用户刚选择的绑定（尚未落盘）时优先采用。
    """
    from .nlp.providers import ProviderRegistry
    from .nlp.slots import resolve_slots

    registry = ProviderRegistry.from_config(cfg)
    try:
        resolution = resolve_slots(slots_override or cfg.slots, None, registry)
        seen: set[str] = set()
        out: list[ProviderConfig] = []
        for slot in _needed_slots(grammar_enabled):
            pid = resolution.provider_id(slot)
            if not pid or pid in seen:
                continue
            seen.add(pid)
            pcfg = registry.config(pid)
            if pcfg:
                out.append(pcfg)
        return out
    finally:
        registry.close_all(timeout=5.0)


def local_server_missing(provider: ProviderConfig) -> str | None:
    """本地托管 provider 的服务端程序（llama-server）缺失检测。

    下载 GGUF 前的前置检查：服务端程序不在时下载模型文件也是徒劳。
    返回 None = 就绪，否则返回面向用户的指引文案。
    """
    local = provider.local
    if provider.type != "llama_cpp_server" or not local or not local.managed:
        return None
    import shlex
    import shutil

    parts = shlex.split(local.server_bin) if local.server_bin.strip() else []
    if not parts:
        return f"Provider '{provider.id}' 未配置 local.server_bin"
    if Path(parts[0]).is_file() or shutil.which(parts[0]):
        return None
    return (
        f"本地模型「{provider.name or provider.id}」需要 llama.cpp 服务端，"
        f"但未找到可执行文件「{local.server_bin}」。\n"
        "请先安装 llama.cpp（GitHub 搜索 llama.cpp，Releases 页下载 Windows 版压缩包并解压），"
        "然后把解压目录加入系统 PATH；或在 provider 配置的 local.server_bin 填 llama-server 的绝对路径。"
    )


def gguf_need(cfg: GlobalConfig, provider: ProviderConfig) -> ResourceNeed | None:
    """单个 provider 的 GGUF 缺失检查（本地托管且文件不存在 → 待下载项，否则 None）。

    collect_missing（启动器）与设置页换绑（选中了未下载的本地模型时即时提示下载）共用。
    """
    local = provider.local
    if provider.type != "llama_cpp_server" or not local or not local.managed:
        return None
    dest = Path(local.gguf_path)
    if not dest.is_absolute():
        dest = cfg.root / dest
    if dest.is_file():
        return None
    return ResourceNeed(
        kind="gguf",
        label=f"本地翻译模型 GGUF（{provider.name or provider.id}）",
        detail=f"→ {dest}",
        provider_id=provider.id,
    )


def collect_missing(
    cfg: GlobalConfig, *, mode: str, grammar_enabled: bool, slots_override=None
) -> list[ResourceNeed]:
    """所选启动模式下缺失的可下载资源（回放模式恒为空）。"""
    if mode == "replay":
        return []
    needs: list[ResourceNeed] = []
    if not whisper_present(cfg.asr.model_size):
        needs.append(
            ResourceNeed(
                kind="whisper",
                label=f"ASR 模型 faster-whisper（{cfg.asr.model_size}）",
                detail=f"HuggingFace {whisper_repo_id(cfg.asr.model_size)} → 本地模型缓存",
            )
        )
    for p in bound_providers(cfg, grammar_enabled, slots_override):
        need = gguf_need(cfg, p)
        if need is not None:
            needs.append(need)
    return needs


def download_whisper(model_size: str, log_fn=print) -> str:
    """下载 faster-whisper 模型到 HF 缓存（直连失败自动切镜像），返回本地路径。"""
    from huggingface_hub import snapshot_download

    repo = whisper_repo_id(model_size)
    try:
        return snapshot_download(repo_id=repo)
    except Exception as e:
        if _is_network_error(e) and not os.environ.get("HF_ENDPOINT"):
            log_fn(f"直连 HuggingFace 失败，切换镜像 {MIRROR_ENDPOINT}…")
            from huggingface_hub import HfApi

            return HfApi(endpoint=MIRROR_ENDPOINT).snapshot_download(repo)
        raise


def download_need(need: ResourceNeed, cfg: GlobalConfig, log_fn=print) -> None:
    """下载单项资源（GGUF 落到 provider 配置的 local.gguf_path，即项目 models/ 下）。"""
    if need.kind == "whisper":
        download_whisper(cfg.asr.model_size, log_fn=log_fn)
        return
    if need.kind == "gguf":
        from .nlp.providers.download import download_model

        pcfg = next((p for p in cfg.providers if p.id == need.provider_id), None)
        if pcfg is None:
            raise ValueError(f"Provider '{need.provider_id}' 不存在")
        download_model(pcfg, cfg.root)
        return
    raise ValueError(f"未知资源类型：{need.kind}")


def health_check(
    cfg: GlobalConfig, grammar_enabled: bool, *, slots_override=None, registry=None, log_fn=print
) -> list[str]:
    """启动健康检查：拉起本地托管 Provider（验证 GGUF 可加载）/ ping 云端。

    返回错误文案列表（空 = 全部通过）。检查结束后回收本地 Provider 子进程——
    正式运行时会重新拉起（两个 Registry 实例不能同时托管同一端口的 llama-server）。
    """
    from .nlp.providers import SLOT_NAMES, ProviderRegistry
    from .nlp.slots import resolve_slots

    own = registry is None
    registry = registry or ProviderRegistry.from_config(cfg)
    try:
        resolution = resolve_slots(slots_override or cfg.slots, None, registry)
        errors: list[str] = []
        for slot in _needed_slots(grammar_enabled):
            slot_name = SLOT_NAMES.get(slot, slot)
            pid = resolution.provider_id(slot)
            if not pid:
                errors.append(f"{slot_name}：{resolution.errors.get(slot, '未绑定')}")
                continue
            log_fn(f"健康检查：{slot_name}（{pid}）…")
            err = registry.check_health(pid)
            if err:
                errors.append(f"{slot_name}：{err}")
        return errors
    finally:
        if own:
            registry.close_all(timeout=10.0)
