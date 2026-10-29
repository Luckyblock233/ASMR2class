"""启动前资源检查/下载/健康检查测试。"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import yaml

from src.config import load_config
from src.resources import (
    ResourceNeed,
    collect_missing,
    download_need,
    health_check,
    whisper_repo_id,
)


def _write_config(tmp_path, providers=None, slots=None, model_size="small"):
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "asr": {"model_size": model_size},
                "providers": providers or [],
                "slots": slots or {},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return load_config(tmp_path)


_LOCAL_PROVIDER = {
    "id": "local-mt",
    "name": "本地翻译",
    "type": "llama_cpp_server",
    "base_url": "http://127.0.0.1:8080/v1",
    "model": "local-mt",
    "capabilities": ["translation"],
    "local": {
        "managed": True,
        "gguf_path": "models/mt.gguf",
        "server_bin": "llama-server",
        "download": {"hf_repo": "org/repo", "hf_file": "mt.gguf"},
    },
}

_CLOUD_PROVIDER = {
    "id": "cloud",
    "type": "openai_compatible",
    "base_url": "https://api.example.com/v1",
    "api_key": "k",
    "model": "m",
    "capabilities": ["translation", "grammar"],
}


def test_whisper_repo_id():
    assert whisper_repo_id("small") == "Systran/faster-whisper-small"
    assert whisper_repo_id("custom/repo") == "custom/repo"  # 已带命名空间原样返回
    assert whisper_repo_id("C:\\models\\fw") == "C:\\models\\fw"  # 本地路径原样返回


def test_collect_missing_replay_needs_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.resources.whisper_present", lambda _s: False)
    cfg = _write_config(tmp_path)
    assert collect_missing(cfg, mode="replay", grammar_enabled=True) == []


def test_collect_missing_whisper_and_gguf(tmp_path, monkeypatch):
    monkeypatch.setattr("src.resources.whisper_present", lambda _s: False)
    cfg = _write_config(
        tmp_path,
        providers=[_LOCAL_PROVIDER, _CLOUD_PROVIDER],
        slots={"translation": "local-mt", "grammar": "cloud"},
    )
    needs = collect_missing(cfg, mode="realtime", grammar_enabled=False)
    kinds = [n.kind for n in needs]
    assert kinds == ["whisper", "gguf"]  # ASR + 本地 GGUF
    gguf = needs[1]
    assert gguf.provider_id == "local-mt"
    assert str(tmp_path / "models" / "mt.gguf") in gguf.detail  # 目标在项目 models/ 下


def test_collect_missing_skips_present_resources(tmp_path, monkeypatch):
    monkeypatch.setattr("src.resources.whisper_present", lambda _s: True)
    cfg = _write_config(tmp_path, providers=[_LOCAL_PROVIDER],
                        slots={"translation": "local-mt"})
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "mt.gguf").write_bytes(b"gguf")  # GGUF 已存在
    assert collect_missing(cfg, mode="realtime", grammar_enabled=False) == []


def test_default_template_needs_only_whisper(tmp_path, monkeypatch):
    """入库模板默认配置：翻译/语法都走云端 API，唯一下载项是 ASR 模型；
    本地 GGUF provider 已声明但未绑定 → 不提示下载。"""
    import shutil

    monkeypatch.setattr("src.resources.whisper_present", lambda _s: False)
    (tmp_path / "config").mkdir()
    root = Path(__file__).resolve().parent.parent
    shutil.copy(root / "config" / "config.example.yaml",
                tmp_path / "config" / "config.example.yaml")
    cfg = load_config(tmp_path)
    needs = collect_missing(cfg, mode="realtime", grammar_enabled=True)
    assert [n.kind for n in needs] == ["whisper"]


def test_collect_missing_grammar_provider_excluded_when_disabled(tmp_path, monkeypatch):
    """语法未启用时，语法槽绑定的本地 provider 不纳入下载。"""
    monkeypatch.setattr("src.resources.whisper_present", lambda _s: True)
    grammar_local = dict(_LOCAL_PROVIDER, id="grammar-local",
                         capabilities=["grammar"])
    cfg = _write_config(
        tmp_path,
        providers=[_CLOUD_PROVIDER, grammar_local],
        slots={"translation": "cloud", "grammar": "grammar-local"},
    )
    assert collect_missing(cfg, mode="batch", grammar_enabled=False) == []
    needs = collect_missing(cfg, mode="batch", grammar_enabled=True)
    assert [n.provider_id for n in needs] == ["grammar-local"]


def test_collect_missing_slots_override(tmp_path, monkeypatch):
    """启动器中刚选择的绑定（未落盘）优先生效：改绑本地模型 → 提示下载其 GGUF。"""
    from src.config import SlotsConfig

    monkeypatch.setattr("src.resources.whisper_present", lambda _s: True)
    cfg = _write_config(
        tmp_path,
        providers=[_CLOUD_PROVIDER, _LOCAL_PROVIDER],
        slots={"translation": "cloud"},  # 已落盘的是云端
    )
    # 无 override：云端绑定，无缺失
    assert collect_missing(cfg, mode="realtime", grammar_enabled=False) == []
    # override 为本地模型：GGUF 缺失 → 提示下载
    needs = collect_missing(
        cfg, mode="realtime", grammar_enabled=False,
        slots_override=SlotsConfig(translation="local-mt"),
    )
    assert [n.provider_id for n in needs] == ["local-mt"]


def test_local_server_missing_detection(tmp_path, monkeypatch):
    """llama-server 前置检测：云端跳过；PATH 有则就绪；缺失则给安装指引。"""
    from src.config import ProviderConfig, ProviderLocalConfig
    from src.resources import local_server_missing

    local = ProviderConfig(
        id="l", type="llama_cpp_server", base_url="http://127.0.0.1:8080/v1",
        model="m", capabilities=["translation"],
        local=ProviderLocalConfig(managed=True, gguf_path="models/x.gguf",
                                  server_bin="llama-server"),
    )
    monkeypatch.setattr("shutil.which", lambda b: None)
    msg = local_server_missing(local)
    assert msg and "llama.cpp" in msg  # 缺失 → 指引文案

    monkeypatch.setattr("shutil.which", lambda b: f"/usr/bin/{b}")
    assert local_server_missing(local) is None  # PATH 中存在 → 就绪

    local.local.server_bin = str(tmp_path / "llama-server.exe")  # 绝对路径存在 → 就绪
    (tmp_path / "llama-server.exe").write_bytes(b"x")
    assert local_server_missing(local) is None

    cloud = ProviderConfig(
        id="c", type="openai_compatible", base_url="https://x/v1", model="m",
        capabilities=["translation"],
    )
    monkeypatch.setattr("shutil.which", lambda b: None)
    assert local_server_missing(cloud) is None  # 云端不检查


def test_download_need_dispatch(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, providers=[_LOCAL_PROVIDER],
                        slots={"translation": "local-mt"})
    calls = []
    monkeypatch.setattr("src.resources.download_whisper",
                        lambda size, log_fn=print: calls.append(("whisper", size)))
    import src.nlp.providers.download as dl

    monkeypatch.setattr(dl, "download_model",
                        lambda pcfg, root: calls.append(("gguf", pcfg.id, root)))
    download_need(ResourceNeed("whisper", "l", "d"), cfg)
    download_need(ResourceNeed("gguf", "l", "d", provider_id="local-mt"), cfg)
    assert calls == [("whisper", "small"), ("gguf", "local-mt", tmp_path)]
    with pytest.raises(ValueError, match="未知资源类型"):
        download_need(ResourceNeed("unknown", "l", "d"), cfg)


class _FakeRegistry:
    """健康检查用假注册表：按 id 返回预设错误（None = 健康）。"""

    def __init__(self, errors: dict):
        self._errors = errors
        self.closed = False

    def config(self, pid):
        from types import SimpleNamespace

        return SimpleNamespace(capabilities=["translation", "grammar"]) if pid else None

    def check_health(self, pid):
        return self._errors.get(pid)

    def close_all(self, timeout=0):
        self.closed = True


def test_health_check_reports_bound_slots(tmp_path):
    cfg = _write_config(tmp_path, providers=[_CLOUD_PROVIDER],
                        slots={"translation": "cloud", "grammar": "cloud"})
    reg = _FakeRegistry({"cloud": None})
    assert health_check(cfg, True, registry=reg) == []
    assert not reg.closed  # 外部注入的 registry 由调用方管理

    reg2 = _FakeRegistry({"cloud": "401 未授权：请配置 API key"})
    errors = health_check(cfg, True, registry=reg2)
    assert len(errors) == 2  # 同一 provider 绑了两个槽位 → 逐槽报告
    assert all("401" in e for e in errors)


def test_health_check_unbound_translation_is_error(tmp_path):
    cfg = _write_config(tmp_path, providers=[_CLOUD_PROVIDER], slots={})
    errors = health_check(cfg, False, registry=_FakeRegistry({}))
    assert any("翻译槽" in e for e in errors)


# ---- 下载确认对话框（offscreen，注入假下载/健康检查）----


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _i18n():
    from pathlib import Path

    from src.ui import i18n

    i18n.set_language("zh_CN", Path(__file__).resolve().parent.parent / "config" / "locales")
    yield
    i18n._reset_for_testing()


def _wait(predicate, timeout_ms=5000):
    from PyQt6.QtTest import QTest

    elapsed = 0
    while not predicate() and elapsed < timeout_ms:
        QTest.qWait(20)
        elapsed += 20
    return predicate()


def test_download_dialog_success_flow(qapp):
    """确认 → 逐项下载 → 健康检查通过 → ok=True 并关闭。"""
    from src.ui.download_dialog import create_download_dialog

    needs = [ResourceNeed("whisper", "ASR 模型", "repo"), ResourceNeed("gguf", "GGUF", "path")]
    downloaded, health_called = [], []

    dialog = create_download_dialog(
        needs,
        on_download=lambda need, log_fn: downloaded.append(need.kind),
        on_health=lambda: health_called.append(True) or [],
    )
    dialog.show()
    assert dialog.tree.topLevelItemCount() == 2
    dialog.btn_start.click()
    assert _wait(lambda: not dialog.isVisible() or dialog.ok)
    assert dialog.ok
    assert downloaded == ["whisper", "gguf"]
    assert health_called == [True]


def test_download_dialog_failure_allows_retry(qapp):
    """下载失败 → 显示错误、可重试；取消 → ok=False。"""
    from src.ui.download_dialog import create_download_dialog

    def fail(need, log_fn):
        raise RuntimeError("网络不可达")

    dialog = create_download_dialog(
        [ResourceNeed("gguf", "GGUF", "path")], on_download=fail, on_health=lambda: []
    )
    dialog.show()
    dialog.btn_start.click()
    assert _wait(lambda: "网络不可达" in dialog.result_label.text())
    assert not dialog.ok
    assert not dialog.btn_continue.isVisibleTo(dialog)  # 下载失败不给强行启动
    assert dialog.btn_start.isEnabled()  # 可重试
    dialog.btn_cancel.click()
    assert not dialog.ok


def test_download_dialog_health_warning_allows_continue(qapp):
    """下载成功但健康检查未过（如缺 API key）→ 可「仍要启动」放行，也可取消返回。"""
    from src.ui.download_dialog import create_download_dialog

    dialog = create_download_dialog(
        [ResourceNeed("whisper", "ASR", "repo")],
        on_download=lambda need, log_fn: None,
        on_health=lambda: ["翻译槽：401 未授权"],
    )
    dialog.show()
    dialog.btn_start.click()
    assert _wait(lambda: "401" in dialog.result_label.text())
    assert not dialog.ok
    assert _wait(lambda: dialog.btn_continue.isVisibleTo(dialog))
    dialog.btn_continue.click()  # 强行启动
    assert dialog.ok
    assert not dialog.isVisible()
