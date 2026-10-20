"""M3 阶段 2：Provider 租约与安全热换绑测试。"""

import threading
import time
from pathlib import Path

import pytest

from src.config import ProviderConfig, ProviderLocalConfig, SlotsConfig
from src.nlp.llm_client import ChatResult
from src.nlp.providers.base import ProviderBase
from src.nlp.providers.registry import ProviderRegistry
from src.nlp.slots import ModelSlots, resolve_slots
from src.profiles.loader import get_profile

from .test_registry import FakeProc, _stub_cloud_ping, make_local_provider_cfg

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StubProvider(ProviderBase):
    """可控 chat 阻塞的桩 Provider，用于验证租约语义。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.closed = 0
        self.chat_started = threading.Event()
        self.release_chat = threading.Event()
        self.chat_calls = 0

    def chat(self, prompt, *, system=None, json_mode=False):
        self.chat_calls += 1
        self.chat_started.set()
        self.release_chat.wait(timeout=10)
        return ChatResult(content="ok")

    @property
    def online(self):
        return True

    def ensure_available(self):
        return None

    def close(self):
        self.closed += 1


def _stub_registry(monkeypatch, *configs):
    """create_provider 打桩：返回按 id 记录的 StubProvider 实例。"""
    import src.nlp.providers.registry as registry_mod

    created: dict[str, list[StubProvider]] = {}

    def factory(cfg, popen=None):
        inst = StubProvider(cfg)
        created.setdefault(cfg.id, []).append(inst)
        return inst

    monkeypatch.setattr(registry_mod, "create_provider", factory)
    registry = registry_mod.ProviderRegistry(list(configs))
    return registry, created


def _cloud(pid):
    return ProviderConfig(
        id=pid, type="openai_compatible",
        base_url=f"https://{pid}.example/v1", api_key="k", model="m",
        capabilities=["translation", "grammar"],
    )


def _wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_in_flight_lease_survives_rebind_and_closes_once(monkeypatch, tmp_path):
    """换绑时旧 Provider 有在途请求：新请求走新实例；旧实例等租约释放后才关闭且恰好一次。"""
    import src.nlp.providers.llama_cpp_local as local_mod

    monkeypatch.setattr(local_mod, "_wait_health", lambda *a, **k: True)
    _stub_cloud_ping(monkeypatch)
    procs = []
    popen = lambda *a, **k: procs.append(FakeProc()) or procs[-1]  # noqa: E731
    local_cfg = make_local_provider_cfg(tmp_path)
    cloud_cfg = _cloud("cloud-b")
    registry = ProviderRegistry([local_cfg, cloud_cfg], popen=popen)
    slots = ModelSlots(registry, resolve_slots(SlotsConfig(translation="local-tr"), {}, registry))

    # 拉起本地 provider 并占住一个租约（模拟慢翻译请求）
    assert registry.check_health("local-tr") is None
    hold = threading.Event()

    def slow_request():
        with slots.acquire("translation") as p:
            assert p is not None
            hold.wait(timeout=10)

    worker = threading.Thread(target=slow_request)
    worker.start()
    assert _wait_until(lambda: registry.active_calls("local-tr") == 1)

    # 换绑到云端：旧代进入 retiring，进程还不能回收
    slots.rebind(resolve_slots(SlotsConfig(translation="cloud-b"), {}, registry))
    with slots.acquire("translation") as p:
        assert p is not None and p.cfg.id == "cloud-b"  # 新请求立即走新实例
    assert not procs[0].terminated

    # 租约释放后旧代被回收（worker 线程不阻塞，回收线程异步关闭）
    hold.set()
    worker.join(timeout=5)
    assert _wait_until(lambda: procs[0].terminated)
    assert registry.active_calls("local-tr") == 0


def test_lease_released_on_exception(monkeypatch):
    registry, _created = _stub_registry(monkeypatch, _cloud("a"))
    with pytest.raises(RuntimeError):
        with registry.acquire("a") as _p:
            raise RuntimeError("boom")
    assert registry.active_calls("a") == 0


def test_same_provider_shared_by_two_slots_not_retired_early(monkeypatch):
    registry, created = _stub_registry(monkeypatch, _cloud("shared"), _cloud("other"))
    slots = ModelSlots(
        registry,
        resolve_slots(SlotsConfig(translation="shared", grammar="shared"), {}, registry),
    )
    first = registry.get("shared")
    # 翻译槽换绑到 other，语法槽仍引用 shared：实例必须保留
    slots.rebind(resolve_slots(SlotsConfig(translation="other", grammar="shared"), {}, registry))
    assert registry.get("shared") is first
    assert created["shared"][0].closed == 0
    # 语法槽也换走后才允许回收
    slots.rebind(resolve_slots(SlotsConfig(translation="other", grammar="other"), {}, registry))


def test_rebind_a_to_b_to_a_creates_new_generation(monkeypatch, tmp_path):
    """A→B→A：再次绑定 A 必须是新代实例，不能复活正在退休的旧 A。"""
    import src.nlp.providers.llama_cpp_local as local_mod

    monkeypatch.setattr(local_mod, "_wait_health", lambda *a, **k: True)
    _stub_cloud_ping(monkeypatch)
    procs = []
    popen = lambda *a, **k: procs.append(FakeProc()) or procs[-1]  # noqa: E731
    cfg_a = make_local_provider_cfg(tmp_path, pid="a")
    cfg_b = make_local_provider_cfg(tmp_path, pid="b")
    cfg_b.local = ProviderLocalConfig(
        managed=True, gguf_path=cfg_a.local.gguf_path, server_args="--port 8081"
    )
    registry = ProviderRegistry([cfg_a, cfg_b], popen=popen)
    slots = ModelSlots(registry, resolve_slots(SlotsConfig(translation="a"), {}, registry))

    assert registry.check_health("a") is None
    first_gen = registry.get("a")
    slots.rebind(resolve_slots(SlotsConfig(translation="b"), {}, registry))
    assert procs[0].terminated

    assert registry.check_health("b") is None
    slots.rebind(resolve_slots(SlotsConfig(translation="a"), {}, registry))
    assert registry.check_health("a") is None
    second_gen = registry.get("a")
    assert second_gen is not first_gen  # 新代，不复用退休实例
    assert len(procs) == 3  # a, b, a'


def test_same_id_config_edit_creates_new_generation(monkeypatch):
    registry, created = _stub_registry(monkeypatch, _cloud("a"))
    first = registry.get("a")
    edited = _cloud("a")
    edited.model = "m2"
    registry.update_config(edited)
    assert created["a"][0].closed == 1  # 无在途租约：旧代立即关闭且恰好一次
    second = registry.get("a")
    assert second is not first
    assert second.cfg.model == "m2"


def test_close_all_waits_for_in_flight_lease(monkeypatch):
    registry, _created = _stub_registry(monkeypatch, _cloud("a"))
    provider = registry.get("a")
    provider.release_chat.clear()

    def slow():
        with registry.acquire("a") as p:
            p.chat("hi")

    worker = threading.Thread(target=slow)
    worker.start()
    assert provider.chat_started.wait(timeout=5)

    done = threading.Event()
    closer = threading.Thread(target=lambda: (registry.close_all(timeout=10), done.set()))
    closer.start()
    time.sleep(0.2)
    assert provider.closed == 0  # 在途请求未结束时不得关闭
    assert not done.is_set()
    provider.release_chat.set()
    worker.join(timeout=5)
    closer.join(timeout=5)
    assert done.is_set() and provider.closed == 1


def test_close_all_timeout_forces_close(monkeypatch):
    registry, _created = _stub_registry(monkeypatch, _cloud("a"))
    provider = registry.get("a")

    def stuck():
        with registry.acquire("a") as p:
            p.release_chat.wait(timeout=30)  # 永不主动释放

    worker = threading.Thread(target=stuck, daemon=True)
    worker.start()
    assert _wait_until(lambda: registry.active_calls("a") == 1)
    started = time.monotonic()
    registry.close_all(timeout=0.3)  # 超时强制关闭，不无限等待
    assert time.monotonic() - started < 5
    assert provider.closed == 1


def test_translator_chat_runs_inside_lease():
    """Translator 配置 lease 后 chat 在租约内执行，结束后租约归零。"""
    from src.nlp.prompt_engine import PromptEngine
    from src.nlp.translate import Translator

    class FakeLease:
        def __init__(self, provider):
            self.provider = provider
            self.active = 0
            self.max_active = 0

        def __call__(self):
            return self

        def __enter__(self):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            return self.provider

        def __exit__(self, *exc):
            self.active -= 1
            return False

    class EchoLLM:
        online = True
        cfg = None

        def chat(self, prompt, *, system=None, json_mode=False):
            return ChatResult(content='{"sentences":[{"id":1,"translation":"好"}]}')

    llm = EchoLLM()
    lease = FakeLease(llm)
    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    prompts = PromptEngine(PROJECT_ROOT / "config" / "prompts")
    t = Translator(lambda: llm, prompts, profile, lease=lease)
    results = t.translate_block(["你好"])
    assert results[0].translation == "好"
    assert lease.max_active == 1 and lease.active == 0
