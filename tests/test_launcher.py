"""启动入口窗口与批处理进度窗口测试（offscreen）。"""

import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from src.config import load_config
from src.offline.runner import OfflineRunner
from src.profiles.loader import load_profiles
from src.ui import i18n
from src.ui.launcher import LaunchParams, create_launcher

from .test_offline import FakeEngine, alternating_audio, fake_decode_factory, fake_vad

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _i18n():
    i18n.set_language("zh_CN", PROJECT_ROOT / "config" / "locales")
    yield
    i18n._reset_for_testing()


def _profiles():
    return list(load_profiles(PROJECT_ROOT / "config" / "languages").values())


def test_launcher_defaults_and_params(qapp):
    w = create_launcher(_profiles(), default_profile="ja", default_source="mic")
    w.show()
    qapp.processEvents()
    p = w.params()
    assert p.mode == "realtime"
    assert p.source_lang == "ja"   # 默认档案 ja 的源语言
    assert p.target_lang == "zh"   # 默认档案 ja 的目标语言
    assert p.source == "mic"
    assert p.grammar is False
    assert w.page_realtime.isVisible() and not w.page_batch.isVisible()
    w.close()


def test_launcher_batch_params_and_validation(qapp):
    w = create_launcher(_profiles())
    w.show()
    w.rb_batch.setChecked(True)
    qapp.processEvents()
    assert w.page_batch.isVisible() and not w.page_realtime.isVisible()

    p = w.params()
    assert p.mode == "batch"
    assert w.validate(p)  # 无文件 → 报错
    w.file_list.addItem("a.mp4")
    w.file_list.addItem("b.mkv")
    p = w.params()
    assert p.files == ["a.mp4", "b.mkv"]
    assert w.validate(p) == ""

    w._on_start()  # 合法 → accept 并产出参数
    assert w.result_params is not None and w.result_params.mode == "batch"
    w.close()


def test_launcher_replay_requires_input(qapp):
    w = create_launcher(_profiles())
    w.rb_replay.setChecked(True)
    p = w.params()
    assert p.mode == "replay"
    assert w.validate(p)  # 会话目录和 SRT 都没有 → 报错
    w.session_edit.setText("output/20261019-ja")
    assert w.validate(w.params()) == ""
    w.close()


def test_launcher_cancel_yields_none(qapp):
    w = create_launcher(_profiles())
    w.show()
    w.reject()
    assert w.result_params is None
    w.close()


def test_batch_window_progress_and_finish(qapp, tmp_path):
    """批处理窗口：假 runner 跑完两文件，状态/进度/日志正确，rc=0。"""
    from src.ui.batch_window import create_batch_shell

    cfg = load_config(PROJECT_ROOT)
    profile = _profiles()[0]
    runner = OfflineRunner(
        cfg, profile, output_root=tmp_path,
        engine=FakeEngine(), vad_fn=fake_vad,
        decoder=fake_decode_factory(alternating_audio(2)),
        slots=None,
    )
    w = create_batch_shell(["a.wav", "b.wav"])
    w.show()
    w.start_runner(runner)
    deadline = time.monotonic() + 10
    while w.return_code is None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert w.return_code == 0
    assert w.progress.value() == 2
    texts = [w.file_list.item(i).text() for i in range(w.file_list.count())]
    assert all(t.startswith("✓") for t in texts)
    w.close()


def test_batch_window_cancel(qapp, tmp_path):
    """取消按钮置 cancel_event：runner 保存已完成部分并停止。"""
    from src.ui.batch_window import create_batch_shell

    cfg = load_config(PROJECT_ROOT)
    profile = _profiles()[0]

    def slow_decoder(path):
        while True:
            yield alternating_audio(1)[0]

    runner = OfflineRunner(
        cfg, profile, output_root=tmp_path,
        engine=FakeEngine(), vad_fn=fake_vad, decoder=slow_decoder, slots=None,
    )
    w = create_batch_shell(["x.wav"])
    w.show()
    w.start_runner(runner)
    # 等跑起来再取消
    deadline = time.monotonic() + 10
    while not w._runner._cancel.is_set() and time.monotonic() < deadline:
        qapp.processEvents()
        if w.progress.format():  # 已有进度输出
            w.btn_cancel.click()
        time.sleep(0.01)
    deadline = time.monotonic() + 10
    while w.return_code is None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert w.return_code == 130
    w.close()


def test_launcher_modal_exec_returns_on_start(qapp):
    """回归：点启动后模态 exec 必须立即返回（不再卡住）。"""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QDialog

    w = create_launcher(_profiles(), default_profile="ja")
    QTimer.singleShot(0, w._on_start)
    rc = w.exec()
    assert rc == QDialog.DialogCode.Accepted
    assert w.result_params is not None and w.result_params.mode == "realtime"


def test_batch_window_close_cancels_runner(qapp, tmp_path):
    """回归：关闭批处理窗口会取消在途 runner（不再留下空转进程）。"""
    from src.ui.batch_window import create_batch_shell

    cfg = load_config(PROJECT_ROOT)
    profile = _profiles()[0]

    def slow_decoder(path):
        while True:
            yield alternating_audio(1)[0]

    runner = OfflineRunner(
        cfg, profile, output_root=tmp_path,
        engine=FakeEngine(), vad_fn=fake_vad, decoder=slow_decoder, slots=None,
    )
    w = create_batch_shell(["x.wav"])
    w.show()
    w.start_runner(runner)
    qapp.processEvents()
    w.close()
    assert runner._cancel.is_set()


def test_check_bound_providers_starts_llm(tmp_path):
    """回归：批处理路径必须对绑定槽位做健康检查（本地 Provider 在此拉起）。"""
    from src.main import _check_bound_providers

    calls = []

    class FakeRegistry:
        def check_health(self, pid):
            calls.append(pid)
            return "启动失败" if pid == "bad" else None

    class FakeRes:
        bindings = {"translation": "p1", "grammar": "bad"}

        def provider_id(self, slot):
            return self.bindings.get(slot)

    class FakeSlots:
        resolution = FakeRes()

    errors = []
    _check_bound_providers(
        FakeRegistry(), FakeSlots(), grammar_enabled=False,
        on_error=lambda slot, err: errors.append((slot, err)),
    )
    assert calls == ["p1"]  # 未启用语法时不碰语法槽

    _check_bound_providers(
        FakeRegistry(), FakeSlots(), grammar_enabled=True,
        on_error=lambda slot, err: errors.append((slot, err)),
    )
    assert calls == ["p1", "p1", "bad"]
    assert errors == [("grammar", "启动失败")]  # 失败只警告上报，不阻断


def test_batch_window_log_event(qapp, tmp_path):
    from src.ui.batch_window import create_batch_shell

    w = create_batch_shell([])
    w.show()
    w._emit("log", text="⚠ 测试日志")
    qapp.processEvents()
    assert "测试日志" in w.log_view.toPlainText()
    w.close()


def test_loopback_device_editable_combo(qapp):
    """内录设备：可编辑下拉框——可选枚举项，也可手填关键词。"""
    w = create_launcher(_profiles())
    w.show()
    qapp.processEvents()
    combo = w.loopback_edit
    assert combo.isEditable()
    assert combo.itemText(0) == ""  # 空 = 默认输出设备
    # 手填关键词
    combo.setCurrentText("扬声器")
    assert w.params().loopback_device == "扬声器"
    # 从下拉选择（枚举失败时至少有默认空项）
    if combo.count() > 1:
        combo.setCurrentIndex(1)
        assert w.params().loopback_device == combo.itemText(1)
    w.close()


def test_launcher_auto_lang_param(qapp):
    """源语言选「自动识别」→ LaunchParams.source_lang == 'auto'。"""
    w = create_launcher(_profiles(), default_profile="ja")
    w.show()
    assert w.params().source_lang == "ja"
    idx = w.src_combo.findData("auto")
    assert idx == 0  # 第一项即自动识别
    w.src_combo.setCurrentIndex(idx)
    assert w.params().source_lang == "auto"
    w.close()
