"""本地 Provider 的 GGUF 模型下载（HF / ModelScope 双源，国内镜像友好）。

下载目标与仓库信息全部来自 provider 配置的 local.download 节，代码不含任何
具体模型/仓库名。用法：uv run python -m src.main --download-model <provider_id>
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ...asr.engine import MIRROR_ENDPOINT, _is_network_error
from ...config import ProviderConfig

log = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """所有下载源均失败，附带可操作指引。"""


def _dest_path(cfg: ProviderConfig, root: Path) -> Path:
    local = cfg.local
    assert local is not None
    p = Path(local.gguf_path)
    return p if p.is_absolute() else root / p


def _download_hf(repo: str, filename: str, dest: Path) -> None:
    """HF 下载（直连失败自动切镜像，与 ASR 模型下载同策略）。"""
    from huggingface_hub import hf_hub_download

    try:
        cached = hf_hub_download(repo_id=repo, filename=filename)
    except Exception as e:
        import os

        if _is_network_error(e) and not os.environ.get("HF_ENDPOINT"):
            log.info("直连 HuggingFace 失败，切换镜像 %s", MIRROR_ENDPOINT)
            from huggingface_hub import HfApi

            cached = HfApi(endpoint=MIRROR_ENDPOINT).hf_hub_download(
                repo_id=repo, filename=filename
            )
        else:
            raise
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)


def _download_modelscope(repo: str, filename: str, dest: Path) -> None:
    """ModelScope 直链流式下载（带进度输出）。"""
    import httpx

    url = f"https://modelscope.cn/api/v1/models/{repo}/repo?FilePath={filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r下载中 {done / 1e6:.0f}/{total / 1e6:.0f}MB", end="", flush=True)
    if total:
        print()
    tmp.rename(dest)


def download_model(cfg: ProviderConfig, root: Path, source: str = "auto") -> Path:
    """按 provider 的 local.download 元数据下载 GGUF 到 local.gguf_path。

    source: "hf" | "modelscope" | "auto"（hf 失败自动试 modelscope）。
    """
    local = cfg.local
    if local is None:
        raise DownloadError(f"Provider '{cfg.id}' 无 local 配置节，不是本地托管模型")
    dl = local.download or {}
    dest = _dest_path(cfg, root)
    if dest.is_file():
        print(f"模型文件已存在：{dest}（如需重下请先删除）")
        return dest

    sources: list[tuple[str, str, str]] = []  # (源名, repo, filename)
    if source in ("hf", "auto") and dl.get("hf_repo") and dl.get("hf_file"):
        sources.append(("hf", dl["hf_repo"], dl["hf_file"]))
    if source in ("modelscope", "auto") and dl.get("modelscope_repo") and dl.get("modelscope_file"):
        sources.append(("modelscope", dl["modelscope_repo"], dl["modelscope_file"]))
    if not sources:
        raise DownloadError(
            f"Provider '{cfg.id}' 的 local.download 未配置可用的 {source} 下载源"
            "（需要 hf_repo/hf_file 或 modelscope_repo/modelscope_file）"
        )

    errors: list[str] = []
    for name, repo, filename in sources:
        print(f"下载 {cfg.id}：{name} {repo}/{filename} → {dest}")
        try:
            if name == "hf":
                _download_hf(repo, filename, dest)
            else:
                _download_modelscope(repo, filename, dest)
            print(f"✓ 下载完成：{dest}")
            return dest
        except Exception as e:  # noqa: BLE001 —— 换下一个源
            log.warning("下载源 %s 失败：%s", name, e)
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise DownloadError(
        f"Provider '{cfg.id}' 所有下载源均失败：\n  " + "\n  ".join(errors)
    )
