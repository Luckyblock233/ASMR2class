"""Provider Registry 生命周期与热切换测试（本地托管子进程用 fake popen）。"""

from pathlib import Path

import pytest

from src.config import ProviderConfig, ProviderLocalConfig
from src.nlp.providers.base import ProviderUnavailableError
from src.nlp.providers.llama_cpp_local import LlamaCppLocalProvider
from src.nlp.providers.registry import ProviderRegistry
from src.nlp.slots import ModelSlots, resolve_slots
from src.config import SlotsConfig


class FakeProc:
    def __init__(self):
        self.pid = 12345
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if not (self.terminated or self.killed) else 0

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


def make_local_provider_cfg(tmp_path: Path, pid: str = "local-tr") -> ProviderConfig:
    gguf = tmp_path / "models" / "x.gguf"
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(b"fake")
    return ProviderConfig(
        id=pid,
        name="本地翻译",
        type="llama_cpp_server",
        base_url="http://127.0.0.1:8080/v1",
        api_key="EMPTY",
        model="local-tr",
        capabilities=["translation"],
        local=ProviderLocalConfig(managed=True, gguf_path=str(gguf), server_args="--port 8080"),
    )


def test_local_provider_starts_and_stops(tmp_path, monkeypatch):
    import src.nlp.providers.llama_cpp_local as mod

    monkeypatch.setattr(mod, "_wait_health", lambda *a, **k: True)
    procs = []
    cfg = make_local_provider_cfg(tmp_path)
    p = LlamaCppLocalProvider(cfg, popen=lambda *a, **k: procs.append(FakeProc()) or procs[-1])
    p.start()
    assert len(procs) == 1
    cmd_args = p._build_cmd()
    assert "--model" in cmd_args and str(cfg.local.gguf_path) in cmd_args
    assert "--alias" in cmd_args and "local-tr" in cmd_args
    assert "--port" in cmd_args
    p.start()  # 幂等：不重复拉起
    assert len(procs) == 1
    p.close()
    assert procs[0].terminated


def test_missing_gguf_raises_with_download_hint(tmp_path):
    cfg = make_local_provider_cfg(tmp_path)
    cfg.local.gguf_path = str(tmp_path / "models" / "missing.gguf")
    p = LlamaCppLocalProvider(cfg, popen=lambda *a, **k: FakeProc())
    with pytest.raises(ProviderUnavailableError, match="--download-model local-tr"):
        p.start()


def test_missing_server_binary_raises(tmp_path, monkeypatch):
    import src.nlp.providers.llama_cpp_local as mod

    monkeypatch.setattr(mod, "_wait_health", lambda *a, **k: True)
    cfg = make_local_provider_cfg(tmp_path)

    def no_such_bin(*a, **k):
        raise FileNotFoundError("llama-server")

    p = LlamaCppLocalProvider(cfg, popen=no_such_bin)
    with pytest.raises(ProviderUnavailableError, match="llama-server"):
        p.start()


def test_health_timeout_closes_process(tmp_path, monkeypatch):
    import src.nlp.providers.llama_cpp_local as mod

    monkeypatch.setattr(mod, "_wait_health", lambda *a, **k: False)
    procs = []
    cfg = make_local_provider_cfg(tmp_path)
    p = LlamaCppLocalProvider(cfg, popen=lambda *a, **k: procs.append(FakeProc()) or procs[-1])
    with pytest.raises(ProviderUnavailableError, match="未就绪"):
        p.start()
    assert procs[0].terminated  # 超时要回收子进程


def test_registry_get_and_unknown_id(tmp_path):
    registry = ProviderRegistry([make_local_provider_cfg(tmp_path)])
    assert registry.get("no-such") is None
    assert registry.get("") is None
    assert registry.config("local-tr") is not None


def _stub_cloud_ping(monkeypatch):
    """健康检查中的云端 ping 走真实网络，测试打桩为直接可用。"""
    from src.nlp.providers.openai_compatible import OpenAICompatibleProvider

    monkeypatch.setattr(OpenAICompatibleProvider, "ensure_available", lambda self: None)


def test_rebind_releases_unbound_local_provider(tmp_path, monkeypatch):
    import src.nlp.providers.llama_cpp_local as mod

    monkeypatch.setattr(mod, "_wait_health", lambda *a, **k: True)
    _stub_cloud_ping(monkeypatch)
    procs = []
    popen = lambda *a, **k: procs.append(FakeProc()) or procs[-1]  # noqa: E731
    local_cfg = make_local_provider_cfg(tmp_path)
    cloud_cfg = ProviderConfig(
        id="cloud-a", type="openai_compatible",
        base_url="https://a.example/v1", api_key="k", model="m",
        capabilities=["translation", "grammar"],
    )
    registry = ProviderRegistry([local_cfg, cloud_cfg], popen=popen)
    slots = ModelSlots(registry, resolve_slots(SlotsConfig(translation="local-tr"), {}, registry))

    # 绑定本地 provider → 健康检查拉起子进程
    assert registry.check_health("local-tr") is None
    assert len(procs) == 1

    # 换绑到云端 → 本地子进程被回收
    slots.rebind(resolve_slots(SlotsConfig(translation="cloud-a"), {}, registry))
    assert procs[0].terminated
    assert slots.translation_provider().cfg.id == "cloud-a"
    # 新请求走新 provider（getter 每次取当前绑定）
    assert slots.provider("translation").cfg.id == "cloud-a"


def test_release_unused_keeps_still_bound(tmp_path, monkeypatch):
    import src.nlp.providers.llama_cpp_local as mod

    monkeypatch.setattr(mod, "_wait_health", lambda *a, **k: True)
    _stub_cloud_ping(monkeypatch)
    procs = []
    popen = lambda *a, **k: procs.append(FakeProc()) or procs[-1]  # noqa: E731
    registry = ProviderRegistry([make_local_provider_cfg(tmp_path)], popen=popen)
    assert registry.check_health("local-tr") is None
    registry.release_unused({"local-tr"})  # 仍在用 → 不回收
    assert not procs[0].terminated
    registry.release_unused(set())  # 不再被任何槽位引用 → 回收
    assert procs[0].terminated
