"""本地 llama.cpp server 托管 Provider（模型可插拔架构 §3）。

应用以子进程托管 llama-server：Provider 被任一槽位绑定且首次使用时拉起，
换绑出所有槽位或退出时由 Registry 调 close() 回收。
显存预检：空闲显存低于 vram_budget_mb + 保留量时打印警告（不静默抢占、不阻断）。

测试可注入 popen 工厂与 sleep 函数，无需真实 llama-server。
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

from ...config import ProviderConfig
from ..llm_client import ChatResult
from .base import ProviderUnavailableError
from .openai_compatible import OpenAICompatibleProvider

log = logging.getLogger(__name__)

VRAM_RESERVE_MB = 2000  # 为系统与其他应用保留的空闲显存（与 ASR 策略一致）
HEALTH_TIMEOUT_SEC = 60.0


def _health_url(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1") + "/health"


def _wait_health(base_url: str, timeout: float, sleep=time.sleep, clock=time.monotonic) -> bool:
    deadline = clock() + timeout
    while clock() < deadline:
        try:
            with urllib.request.urlopen(_health_url(base_url), timeout=2) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            pass
        sleep(0.5)
    return False


class LlamaCppLocalProvider(OpenAICompatibleProvider):
    """managed=true 时托管 llama-server 子进程；否则仅连端点（等价 openai_compatible）。"""

    def __init__(self, cfg: ProviderConfig, popen=None):
        if cfg.type != "llama_cpp_server":
            raise ValueError(f"LlamaCppLocalProvider 仅支持 type=llama_cpp_server，实际 {cfg.type}")
        self._popen = popen or subprocess.Popen
        self._proc: subprocess.Popen | None = None
        super().__init__(cfg)

    # ---- 子进程托管 ----

    def _check_vram(self) -> str | None:
        """显存预检，返回警告文案（不阻断）。无 GPU/无法检测返回 None。"""
        from ...asr.engine import _cuda_free_vram_mb

        budget = self.cfg.local.vram_budget_mb if self.cfg.local else 0
        if budget <= 0:
            return None
        free = _cuda_free_vram_mb()
        if free is None:
            return None
        if free < budget + VRAM_RESERVE_MB:
            return (
                f"⚠ 空闲显存 {free}MB 低于预算 {budget}MB + 保留 {VRAM_RESERVE_MB}MB。"
                "建议：server_args 去掉 GPU 层卸载（如 -ngl）改用 CPU 推理，或改绑其他 Provider。"
            )
        return None

    def _build_cmd(self) -> list[str]:
        local = self.cfg.local
        # server_bin 支持带子命令：新版 llama.cpp 合并为单二进制（"llama serve"），
        # 旧版为独立 llama-server 可执行文件
        return [
            *shlex.split(local.server_bin),
            "--model", local.gguf_path,
            "--alias", self.cfg.model,
            *shlex.split(local.server_args),
        ]

    def start(self) -> None:
        local = self.cfg.local
        if self._proc is not None and self._proc.poll() is None:
            return  # 已在运行
        if not local or not local.managed:
            return
        gguf = Path(local.gguf_path)
        if not gguf.is_file():
            raise ProviderUnavailableError(
                f"Provider '{self.cfg.id}' 的 GGUF 模型文件不存在：{gguf}\n"
                f"请先下载：uv run python -m src.main --download-model {self.cfg.id}"
            )
        warn = self._check_vram()
        if warn:
            log.warning(warn)
            print(warn)
        try:
            self._proc = self._popen(
                self._build_cmd(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as e:
            raise ProviderUnavailableError(
                f"找不到 llama-server 可执行文件：{local.server_bin}。"
                "请安装 llama.cpp 并加入 PATH（新版为单二进制，可填 \"llama serve\"），"
                "或在 local.server_bin 填绝对路径"
            ) from e
        if not _wait_health(self.cfg.base_url, HEALTH_TIMEOUT_SEC):
            self.close()
            raise ProviderUnavailableError(
                f"llama-server 启动后 {HEALTH_TIMEOUT_SEC:.0f}s 内未就绪（{_health_url(self.cfg.base_url)}），"
                "请检查 local.server_args 与模型文件"
            )
        log.info("llama-server 已就绪：%s（pid=%s）", self.cfg.base_url, self._proc.pid)

    def ensure_available(self) -> None:
        if self.cfg.local and self.cfg.local.managed:
            self.start()
        super().ensure_available()

    def close(self) -> None:
        """回收子进程释放显存（换绑出所有槽位/退出时由 Registry 调用）。"""
        if self._proc is not None and self._proc.poll() is None:
            log.info("停止 llama-server（pid=%s）", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
