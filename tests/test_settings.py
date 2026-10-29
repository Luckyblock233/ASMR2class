"""设置控制器（Qt 无关事务）与 i18n 文案层测试。"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.config import SlotsConfig, UIConfig, load_config
from src.nlp.providers import SLOT_GRAMMAR, SLOT_TRANSLATION, ProviderRegistry
from src.nlp.slots import ModelSlots, resolve_slots
from src.ui import i18n
from src.ui.settings import SettingsController

from .test_lease import _cloud

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_env(tmp_path, providers=(), slots_cfg=None, profile_slots=None):
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.example.yaml").write_text("llm: {}\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    cfg.providers = list(providers)
    cfg.slots = slots_cfg or SlotsConfig()
    registry = ProviderRegistry(cfg.providers)
    profile = SimpleNamespace(slots=profile_slots or {}, name="test")
    slots = ModelSlots(registry, resolve_slots(cfg.slots, profile.slots, registry))
    return cfg, registry, slots, SettingsController(cfg, registry, slots, profile)


def _yaml_slots(cfg) -> dict:
    raw = yaml.safe_load(cfg.config_file.read_text(encoding="utf-8"))
    return raw.get("slots") or {}


def test_candidates_filtered_by_capability(tmp_path):
    t_only = _cloud("t-only")
    t_only.capabilities = ["translation"]
    both = _cloud("both")
    _cfg, _r, _s, ctl = _make_env(tmp_path, [t_only, both])
    assert [p.id for p in ctl.candidates(SLOT_TRANSLATION)] == ["t-only", "both"]
    assert [p.id for p in ctl.candidates(SLOT_GRAMMAR)] == ["both"]


def test_slot_view_profile_override_is_honest(tmp_path):
    _cfg, _r, _s, ctl = _make_env(
        tmp_path, [_cloud("p")], profile_slots={SLOT_GRAMMAR: "p"}
    )
    view = ctl.slot_view(SLOT_GRAMMAR)
    assert view.overridden_by_profile and not view.editable
    assert view.effective_id == "p"
    view_t = ctl.slot_view(SLOT_TRANSLATION)
    assert view_t.editable and not view_t.overridden_by_profile


def test_apply_binding_refused_when_profile_overrides(tmp_path):
    cfg, _r, slots, ctl = _make_env(
        tmp_path, [_cloud("p"), _cloud("q")], profile_slots={SLOT_GRAMMAR: "p"}
    )
    called = []
    result = ctl.apply_binding(SLOT_GRAMMAR, "q", health_check=lambda pid: called.append(pid) or None)
    assert not result.ok
    assert result.message_key == "settings.msg.profile_locked"
    assert called == []  # 未做健康检查
    assert slots.resolution.bindings.get(SLOT_GRAMMAR) == "p"  # 运行态不变
    assert "slots" not in (yaml.safe_load(cfg.config_file.read_text(encoding="utf-8")) or {})


def test_apply_binding_health_failure_keeps_old_state(tmp_path):
    cfg, _r, slots, ctl = _make_env(tmp_path, [_cloud("p")])
    result = ctl.apply_binding(SLOT_TRANSLATION, "p", health_check=lambda pid: "连接失败")
    assert not result.ok
    assert result.message_key == "settings.msg.health_failed"
    assert "连接失败" in result.detail
    assert slots.resolution.bindings.get(SLOT_TRANSLATION) is None
    assert _yaml_slots(cfg) == {}


def test_apply_binding_capability_mismatch_without_health_check(tmp_path):
    t_only = _cloud("t-only")
    t_only.capabilities = ["translation"]
    _cfg, _r, _s, ctl = _make_env(tmp_path, [t_only])
    called = []
    result = ctl.apply_binding(SLOT_GRAMMAR, "t-only", health_check=lambda pid: called.append(pid) and None)
    assert not result.ok
    assert result.message_key == "settings.msg.capability_mismatch"
    assert called == []


def test_apply_binding_success_persists_and_publishes(tmp_path):
    cfg, _r, slots, ctl = _make_env(tmp_path, [_cloud("p")])
    result = ctl.apply_binding(SLOT_TRANSLATION, "p", health_check=lambda pid: None)
    assert result.ok
    assert cfg.slots.translation == "p"
    assert slots.resolution.bindings[SLOT_TRANSLATION] == "p"
    assert _yaml_slots(cfg)["translation"] == "p"

    # 解绑：空 pid 不做健康检查
    result = ctl.apply_binding(SLOT_TRANSLATION, "", health_check=lambda pid: "不应调用")
    assert result.ok
    assert slots.resolution.bindings[SLOT_TRANSLATION] is None


def test_apply_binding_save_failure_keeps_old_runtime(tmp_path, monkeypatch):
    cfg, _r, slots, ctl = _make_env(tmp_path, [_cloud("p")])
    import src.ui.settings as settings_mod

    def boom(_cfg, _slots):
        raise OSError("磁盘只读")

    monkeypatch.setattr(settings_mod, "save_slots_config", boom)
    result = ctl.apply_binding(SLOT_TRANSLATION, "p", health_check=lambda pid: None)
    assert not result.ok
    assert result.message_key == "settings.msg.save_failed"
    assert slots.resolution.bindings.get(SLOT_TRANSLATION) is None  # 旧运行态保留


def test_apply_provider_add_edit_and_remove(tmp_path):
    cfg, registry, _s, ctl = _make_env(tmp_path)
    new = _cloud("p1")
    result = ctl.apply_provider(new, health_check=None)
    assert result.ok
    assert "p1" in ctl.provider_ids()
    raw = yaml.safe_load(cfg.config_file.read_text(encoding="utf-8"))
    assert [p["id"] for p in raw["providers"]] == ["p1"]
    assert registry.config("p1").model == "m"

    # 同 id 编辑：配置发布为新代
    edited = _cloud("p1")
    edited.model = "m2"
    assert ctl.apply_provider(edited, health_check=None).ok
    assert registry.config("p1").model == "m2"

    # 绑定中不可删除
    ctl.apply_binding(SLOT_TRANSLATION, "p1", health_check=lambda pid: None)
    result = ctl.remove_provider("p1")
    assert not result.ok
    assert result.message_key == "settings.msg.provider_in_use"

    # 解绑后可删除
    ctl.apply_binding(SLOT_TRANSLATION, "", health_check=lambda pid: None)
    assert ctl.remove_provider("p1").ok
    assert "p1" not in ctl.provider_ids()
    raw = yaml.safe_load(cfg.config_file.read_text(encoding="utf-8"))
    assert raw.get("providers") in (None, [])


def test_apply_provider_incomplete_rejected(tmp_path):
    _cfg, _r, _s, ctl = _make_env(tmp_path)
    bad = _cloud("bad")
    bad.base_url = ""
    result = ctl.apply_provider(bad)
    assert not result.ok
    assert result.message_key == "settings.msg.provider_incomplete"


def test_apply_appearance_persists(tmp_path):
    cfg, _r, _s, ctl = _make_env(tmp_path)
    ui = UIConfig(theme="light", background_opacity=0.5, view_capacity=100)
    assert ctl.apply_appearance(ui).ok
    assert cfg.ui.theme == "light"
    raw = yaml.safe_load(cfg.config_file.read_text(encoding="utf-8"))
    assert raw["ui"]["theme"] == "light"
    assert raw["ui"]["background_opacity"] == 0.5


# ---- i18n ----


@pytest.fixture(autouse=True)
def _reset_i18n():
    i18n._reset_for_testing()
    yield
    i18n._reset_for_testing()


def test_i18n_fallback_chain(tmp_path):
    (tmp_path / "zh_CN.yaml").write_text(
        "a.key: 中文值\nshared.key: 中文共享\n", encoding="utf-8"
    )
    (tmp_path / "en.yaml").write_text("a.key: English value\n", encoding="utf-8")
    i18n.set_language("en", tmp_path)
    assert i18n.tr("a.key") == "English value"
    assert i18n.tr("shared.key") == "中文共享"  # en 缺失回退 zh_CN
    assert i18n.tr("missing.key") == "missing.key"  # 完全缺失回退 key 本身


def test_i18n_real_locales_load():
    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    assert i18n.tr("settings.title") == "设置"
    i18n.set_language("en")
    assert i18n.tr("settings.title") == "Settings"
    # en 与 zh_CN 的 key 集合一致（防漏译）
    zh = yaml.safe_load((PROJECT_ROOT / "config" / "locales" / "zh_CN.yaml").read_text(encoding="utf-8"))
    en = yaml.safe_load((PROJECT_ROOT / "config" / "locales" / "en.yaml").read_text(encoding="utf-8"))
    assert set(zh) == set(en)


# ---- 设置对话框（offscreen 冒烟）----

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_settings_dialog_smoke_and_override_lock(tmp_path, qapp):
    from src.ui.settings import create_settings_dialog

    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    _cfg, _r, _s, ctl = _make_env(
        tmp_path, [_cloud("p")], profile_slots={SLOT_GRAMMAR: "p"}
    )
    dialog = create_settings_dialog(ctl)
    dialog.show()
    qapp.processEvents()

    # 档案覆盖的槽位：下拉与应用按钮禁用（不假报可改）
    assert not dialog._slot_combos[SLOT_GRAMMAR].isEnabled()
    assert not dialog._slot_apply[SLOT_GRAMMAR].isEnabled()
    assert dialog._slot_combos[SLOT_TRANSLATION].isEnabled()
    # 外观表单回填当前配置（主题/透明度）
    assert dialog.a_theme.currentData() == _cfg.ui.theme
    assert dialog.a_opacity.value() == round(_cfg.ui.effective_background_opacity * 100)
    dialog.close()


def _local_provider(pid="local-mt"):
    from src.config import ProviderConfig, ProviderLocalConfig

    return ProviderConfig(
        id=pid,
        name="本地翻译",
        type="llama_cpp_server",
        base_url="http://127.0.0.1:8080/v1",
        model=pid,
        capabilities=["translation"],
        local=ProviderLocalConfig(
            managed=True,
            gguf_path="models/mt.gguf",
            download={"hf_repo": "org/repo", "hf_file": "mt.gguf"},
        ),
    )


class _FakeDownloadDialog:
    """替换真实下载确认窗：记录 needs，exec 后按预设 ok 返回。"""

    ok_to_give = False
    instances: list = []

    def __init__(self, needs, on_download, on_health, parent=None):
        self.needs = needs
        self.ok = False
        type(self).instances.append(self)

    def exec(self):
        self.ok = type(self).ok_to_give


def test_binding_local_provider_missing_gguf_prompts_download(tmp_path, qapp, monkeypatch):
    """换绑到本地托管模型且 GGUF 缺失 → 先弹下载窗；取消则不换绑，下载成功才继续换绑。"""
    from PyQt6.QtTest import QTest

    from src.ui.settings import create_settings_dialog

    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    cfg, registry, slots, ctl = _make_env(
        tmp_path, [_cloud("p"), _local_provider()], slots_cfg=SlotsConfig(translation="p")
    )
    monkeypatch.setattr(
        "src.ui.download_dialog.create_download_dialog", _FakeDownloadDialog
    )
    monkeypatch.setattr("src.resources.local_server_missing", lambda _p: None)  # 服务端就绪
    dialog = create_settings_dialog(ctl)
    dialog.show()
    combo = dialog._slot_combos[SLOT_TRANSLATION]
    combo.setCurrentIndex(combo.findData("local-mt"))

    # 取消下载 → 不换绑
    _FakeDownloadDialog.ok_to_give = False
    dialog._apply_slot_binding(SLOT_TRANSLATION)
    qapp.processEvents()
    assert len(_FakeDownloadDialog.instances) == 1
    assert _FakeDownloadDialog.instances[0].needs[0].provider_id == "local-mt"
    assert slots.resolution.provider_id(SLOT_TRANSLATION) == "p"

    # 下载成功 → 继续换绑（健康检查打桩通过；_run_async 走 worker 线程，等待信号落地）
    _FakeDownloadDialog.ok_to_give = True
    monkeypatch.setattr(registry, "check_health", lambda _pid: None)
    dialog._apply_slot_binding(SLOT_TRANSLATION)
    for _ in range(100):
        QTest.qWait(20)
        if slots.resolution.provider_id(SLOT_TRANSLATION) == "local-mt":
            break
    assert slots.resolution.provider_id(SLOT_TRANSLATION) == "local-mt"
    assert _yaml_slots(cfg)["translation"] == "local-mt"  # 已落盘
    dialog.close()


def test_binding_blocked_when_llama_server_missing(tmp_path, qapp, monkeypatch):
    """llama-server 未安装：弹指引、不下载 GGUF、不换绑（下载也是徒劳）。"""
    from PyQt6.QtWidgets import QMessageBox

    from src.ui.settings import create_settings_dialog

    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    _cfg, _r, slots, ctl = _make_env(
        tmp_path, [_cloud("p"), _local_provider()], slots_cfg=SlotsConfig(translation="p")
    )
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warnings.append(a)))
    monkeypatch.setattr(
        "src.ui.download_dialog.create_download_dialog", _FakeDownloadDialog
    )
    _FakeDownloadDialog.instances.clear()
    monkeypatch.setattr("shutil.which", lambda _b: None)  # 模拟未安装 llama-server
    dialog = create_settings_dialog(ctl)
    dialog.show()
    combo = dialog._slot_combos[SLOT_TRANSLATION]
    combo.setCurrentIndex(combo.findData("local-mt"))
    dialog._apply_slot_binding(SLOT_TRANSLATION)
    qapp.processEvents()
    assert len(warnings) == 1  # 弹出安装指引
    assert _FakeDownloadDialog.instances == []  # 不进入下载流程
    assert slots.resolution.provider_id(SLOT_TRANSLATION) == "p"  # 不换绑
    dialog.close()


def test_binding_cloud_provider_no_download_prompt(tmp_path, qapp, monkeypatch):
    """换绑云端 provider 不弹下载窗。"""
    from src.ui.settings import create_settings_dialog

    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    _cfg, registry, _s, ctl = _make_env(
        tmp_path, [_cloud("p"), _cloud("q")], slots_cfg=SlotsConfig(translation="p")
    )
    monkeypatch.setattr(
        "src.ui.download_dialog.create_download_dialog", _FakeDownloadDialog
    )
    _FakeDownloadDialog.instances.clear()
    monkeypatch.setattr(registry, "check_health", lambda _pid: "401")  # 失败也无妨
    dialog = create_settings_dialog(ctl)
    dialog.show()
    combo = dialog._slot_combos[SLOT_TRANSLATION]
    combo.setCurrentIndex(combo.findData("q"))
    dialog._apply_slot_binding(SLOT_TRANSLATION)
    qapp.processEvents()
    assert _FakeDownloadDialog.instances == []
    dialog.close()


# ==================== 阶段 6：记忆编辑与 prompt 恢复 ====================


def test_live_memory_edit_visible_in_next_prompt(tmp_path):
    """实时编辑 Translator.memory 同一实例后，下一次 prompt 立即看到新术语。"""
    import json

    from src.nlp.llm_client import ChatResult
    from src.nlp.prompt_engine import PromptEngine
    from src.nlp.translate import Translator

    class EchoLLM:
        online = True
        cfg = None

        def __init__(self):
            self.prompts = []

        def chat(self, prompt, **kw):
            self.prompts.append(prompt)
            return ChatResult(json.dumps(
                {"sentences": [{"id": 1, "translation": "好"}]}, ensure_ascii=False
            ))

    from src.profiles.loader import get_profile

    profile = get_profile(PROJECT_ROOT / "config" / "languages", "ja")
    llm = EchoLLM()
    translator = Translator(llm, PromptEngine(PROJECT_ROOT / "config" / "prompts"), profile)
    _cfg, _r, _s, ctl = _make_env(tmp_path)
    ctl._memory_provider = lambda: translator.memory

    mem = ctl.live_memory()
    assert mem is translator.memory
    mem.set_term("ユキ", "小雪")
    mem.set_style("轻松日常")
    translator.translate_block(["ユキが来た"])
    assert "ユキ → 小雪" in llm.prompts[-1]
    assert "轻松日常" in llm.prompts[-1]


def test_memory_tab_edit_save_and_style_lock(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from src.nlp.context_memory import RollingMemory
    from src.ui.settings import create_settings_dialog

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")

    mem = RollingMemory()
    mem.set_term("旧", "旧译")
    _cfg, _r, _s, ctl = _make_env(tmp_path)
    ctl._memory_provider = lambda: mem
    dialog = create_settings_dialog(ctl)
    dialog.show()
    qapp.processEvents()

    assert dialog.m_table.rowCount() == 1
    dialog._memory_add_row("新术语", "新译", "person")
    assert dialog._memory_dirty
    dialog._memory_save()
    assert not dialog._memory_dirty
    assert mem.glossary["新术语"].translation == "新译"
    assert mem.glossary["新术语"].type == "person"

    # 删除行后保存 → 从实时记忆移除
    dialog.m_table.selectRow(0)
    dialog._memory_remove_selected()
    dialog._memory_save()
    assert "旧" not in mem.glossary

    # 空字段校验：保存被拒且不破坏记忆
    dialog._memory_add_row("只有术语", "", "")
    dialog._memory_save()
    assert "只有术语" not in mem.glossary

    # style_preset 锁定时风格区只读
    locked = RollingMemory(style_preset="正式书面语")
    ctl._memory_provider = lambda: locked
    dialog._reload_memory_editor()
    assert dialog.m_style.isReadOnly()
    assert "锁定" in dialog.m_style_label.text()
    dialog.close()


def test_restore_prompt_template_with_backup(tmp_path):
    """恢复默认：自定义模板被 .bak 备份后覆盖为 _default 内容；mtime 热重载自然生效。"""
    prompts_dir = tmp_path / "config" / "prompts"
    (prompts_dir / "_default").mkdir(parents=True)
    (prompts_dir / "ja").mkdir(parents=True)
    (prompts_dir / "_default" / "translate.txt").write_text("默认模板", encoding="utf-8")
    custom = prompts_dir / "ja" / "translate.txt"
    custom.write_text("用户自定义模板", encoding="utf-8")

    _cfg, _r, _s, ctl = _make_env(tmp_path)
    profile = SimpleNamespace(
        name="ja",
        slots={},
        prompt_path=lambda kind: {"translate": "ja/translate.txt"}.get(kind),
    )
    ctl.set_profile(profile)

    result = ctl.restore_prompt_template("translate")
    assert result.ok
    assert custom.read_text(encoding="utf-8") == "默认模板"
    backups = list(prompts_dir.glob("ja/*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "用户自定义模板"

    # 档案未指定自定义模板时不做危险操作
    profile2 = SimpleNamespace(name="ja", slots={}, prompt_path=lambda kind: None)
    ctl.set_profile(profile2)
    result = ctl.restore_prompt_template("translate")
    assert not result.ok
    assert result.message_key == "settings.msg.no_custom_template"
