"""设置页与设置控制器（M3 阶段 2：Provider/槽位/外观）。

分层：
- SettingsController：Qt 无关的应用事务。所有网络/进程等待（健康检查、本地
  Provider 启动）都由调用方注入的 health_check 在 worker 线程执行；
  事务顺序：校验候选 → 健康检查 → 原子配置保存 → 发布运行态 → 回收旧代。
  健康或保存失败时保留旧运行态；启动失败的临时候选必须关闭。
- SettingsDialog：纯 GUI。自身不阻塞：耗时操作经 _run_async 放 worker 线程，
  结果用 Qt 信号回到 GUI 线程。

日志只记录 provider id/model，绝不记录 api_key。
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import threading
from dataclasses import dataclass
from typing import Callable

from ..config import (
    GlobalConfig,
    ProviderConfig,
    SlotsConfig,
    UIConfig,
    delete_provider_config,
    save_provider_config,
    save_slots_config,
    save_ui_config,
)
from ..nlp.providers import (
    SLOT_GRAMMAR,
    SLOT_NAMES,
    SLOT_TRANSLATION,
    ProviderRegistry,
    slot_error,
)
from ..nlp.providers.capabilities import filter_for_slot
from ..nlp.slots import ModelSlots, resolve_slots
from .i18n import tr

log = logging.getLogger(__name__)

ALL_SLOTS = (SLOT_TRANSLATION, SLOT_GRAMMAR)


def provider_detail_text(p: ProviderConfig) -> str:
    """Provider 详细信息：主行只显示名称，量化/端点/实例 ID 等放 tooltip。"""
    lines = [f"id: {p.id}", f"model: {p.model}", f"type: {p.type}", f"endpoint: {p.base_url}"]
    if p.local is not None and p.local.gguf_path:
        from pathlib import Path

        lines.append(f"gguf: {Path(p.local.gguf_path).name}")
        if p.local.server_args:
            lines.append(f"args: {p.local.server_args}")
    return "\n".join(lines)


@dataclass
class ApplyResult:
    ok: bool
    message_key: str
    detail: str = ""

    def text(self) -> str:
        base = tr(self.message_key)
        return f"{base}：{self.detail}" if self.detail else base


@dataclass
class SlotView:
    """B 栏/设置页共用的槽位显示状态（状态来自控制器，控件不自行猜测）。"""

    slot: str
    label: str
    effective_id: str            # 实际生效绑定（档案覆盖优先），空 = 未绑定
    overridden_by_profile: bool  # 当前档案固定了该槽位
    editable: bool               # 可在全局作用域改绑
    status: str                  # 未绑定/不可用原因


class SettingsController:
    """设置事务控制器（Qt 无关，可直接单测）。"""

    def __init__(
        self,
        cfg: GlobalConfig,
        registry: ProviderRegistry,
        slots: ModelSlots,
        profile=None,
        memory_provider: Callable[[], object] | None = None,
    ):
        self.cfg = cfg
        self.registry = registry
        self.slots = slots
        self.profile = profile
        # 返回实时 Translator.memory 的 callable（None = 当前无实时会话记忆）
        self._memory_provider = memory_provider

    # ---- 查询 ----

    def slot_view(self, slot: str) -> SlotView:
        override = ""
        if self.profile is not None:
            override = str(self.profile.slots.get(slot) or "")
        res = self.slots.resolution
        effective = res.bindings.get(slot) or ""
        return SlotView(
            slot=slot,
            label=SLOT_NAMES.get(slot, slot),
            effective_id=effective,
            overridden_by_profile=bool(override),
            editable=not override,
            status="" if effective else res.errors.get(slot, tr("settings.slots.unbound")),
        )

    def candidates(self, slot: str) -> list[ProviderConfig]:
        """按能力过滤的可绑 Provider。"""
        return filter_for_slot(self.cfg.providers, slot)

    def provider_ids(self) -> list[str]:
        return [p.id for p in self.cfg.providers]

    def available_profiles(self):
        """全部语言档案（B 栏档案切换下拉）。"""
        from ..profiles.loader import load_profiles

        return list(load_profiles(self.cfg.languages_dir).values())

    def set_profile(self, profile) -> None:
        """运行中切换档案后更新控制器引用（档案覆盖判断随动）。"""
        self.profile = profile

    # ---- 记忆与 prompt ----

    def live_memory(self):
        """实时会话的 RollingMemory 实例（编辑同一实例，下一 prompt 立即生效）。"""
        if self._memory_provider is None:
            return None
        return self._memory_provider()

    def restore_prompt_template(self, kind: str) -> ApplyResult:
        """把当前档案的 prompt 模板恢复为 _default 版本（确认后调用；保留 .bak 备份）。

        PromptEngine 按 mtime 热重载，恢复后下一次请求即生效。
        """
        import shutil
        import time

        if self.profile is None:
            return ApplyResult(False, "settings.msg.internal_error", "无当前档案")
        rel = self.profile.prompt_path(kind)
        if not rel:
            return ApplyResult(False, "settings.msg.no_custom_template")
        target = self.cfg.prompts_dir / rel
        default = self.cfg.prompts_dir / "_default" / f"{kind}.txt"
        if not default.is_file():
            return ApplyResult(False, "settings.msg.default_template_missing", str(default))
        try:
            if target.is_file():
                backup = target.with_name(
                    f"{target.name}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
                )
                shutil.copyfile(target, backup)
            shutil.copyfile(default, target)
        except OSError as e:
            return ApplyResult(False, "settings.msg.save_failed", str(e))
        log.info("prompt 模板已恢复默认：%s", rel)
        return ApplyResult(True, "settings.msg.template_restored")

    # ---- 槽位绑定事务 ----

    def apply_binding(
        self,
        slot: str,
        provider_id: str,
        health_check: Callable[[str], str | None],
    ) -> ApplyResult:
        """换绑事务。health_check 由调用方在 worker 线程提供（如 registry.check_health）。

        任一环节失败都保留旧运行态；成功路径：保存 → 发布 → 旧代由 Registry 排空回收。
        """
        if slot not in ALL_SLOTS:
            return ApplyResult(False, "settings.msg.unknown_slot", slot)
        if self.profile is not None and self.profile.slots.get(slot):
            # 档案覆盖作用域：不假报"已生效"，第一版禁止全局换绑（见交接文档 §4.4）
            return ApplyResult(False, "settings.msg.profile_locked")
        provider_id = provider_id.strip()
        if provider_id:
            pcfg = self.registry.config(provider_id)
            if pcfg is None:
                return ApplyResult(False, "settings.msg.provider_missing", provider_id)
            err = slot_error(pcfg, slot)
            if err:
                return ApplyResult(False, "settings.msg.capability_mismatch", err)
            err = health_check(provider_id)
            if err:
                return ApplyResult(False, "settings.msg.health_failed", err)

        current = self.slots.resolution.bindings
        new_slots = SlotsConfig(
            translation=provider_id if slot == SLOT_TRANSLATION else (current.get(SLOT_TRANSLATION) or ""),
            grammar=provider_id if slot == SLOT_GRAMMAR else (current.get(SLOT_GRAMMAR) or ""),
        )
        try:
            save_slots_config(self.cfg, new_slots)
        except OSError as e:
            return ApplyResult(False, "settings.msg.save_failed", str(e))
        self.cfg.slots = new_slots
        resolution = resolve_slots(new_slots, self.profile.slots if self.profile else {}, self.registry)
        self.slots.rebind(resolution)
        log.info("槽位换绑已发布：%s → %s", slot, provider_id or "(未绑定)")
        return ApplyResult(True, "settings.msg.applied")

    # ---- Provider 增删改事务 ----

    def apply_provider(
        self,
        provider: ProviderConfig,
        health_check: Callable[[str], str | None] | None = None,
    ) -> ApplyResult:
        """新增/编辑 Provider：健康检查（可选）→ 原子保存 → 发布配置（同 id 编辑作废旧代）。"""
        if not provider.id:
            return ApplyResult(False, "settings.msg.provider_id_required")
        if not provider.is_complete:
            return ApplyResult(False, "settings.msg.provider_incomplete")
        old_cfg = self.registry.config(provider.id)
        # 先把候选配置发布到 Registry（懒构造，不起进程），健康检查使用候选实例
        self.registry.update_config(provider)
        if health_check is not None:
            err = health_check(provider.id)
            if err:
                self.registry.retire(provider.id)  # 启动失败的临时候选必须关闭
                self._restore_config(old_cfg)
                return ApplyResult(False, "settings.msg.health_failed", err)
        try:
            save_provider_config(self.cfg, provider, self.cfg.slots)
        except OSError as e:
            self.registry.retire(provider.id)
            self._restore_config(old_cfg)
            return ApplyResult(False, "settings.msg.save_failed", str(e))
        self._upsert_cfg_provider(provider)
        log.info("Provider 已保存：%s（model=%s）", provider.id, provider.model)
        return ApplyResult(True, "settings.msg.provider_saved")

    def remove_provider(self, provider_id: str) -> ApplyResult:
        """删除 Provider；仍被全局或档案槽位绑定时拒绝。"""
        bound_in = []
        for slot in ALL_SLOTS:
            if (self.cfg.slots.translation if slot == SLOT_TRANSLATION else self.cfg.slots.grammar) == provider_id:
                bound_in.append(SLOT_NAMES.get(slot, slot))
            if self.profile is not None and self.profile.slots.get(slot) == provider_id:
                bound_in.append(f"{SLOT_NAMES.get(slot, slot)}（档案覆盖）")
        if bound_in:
            return ApplyResult(False, "settings.msg.provider_in_use", "、".join(bound_in))
        self.registry.remove_config(provider_id)
        try:
            delete_provider_config(self.cfg, provider_id)
        except OSError as e:
            return ApplyResult(False, "settings.msg.save_failed", str(e))
        self.cfg.providers = [p for p in self.cfg.providers if p.id != provider_id]
        log.info("Provider 已删除：%s", provider_id)
        return ApplyResult(True, "settings.msg.provider_deleted")

    # ---- 外观事务 ----

    def apply_appearance(self, ui: UIConfig) -> ApplyResult:
        """保存外观配置；GUI 收到成功后负责即时重建外观。"""
        try:
            save_ui_config(self.cfg, ui)
        except OSError as e:
            return ApplyResult(False, "settings.msg.save_failed", str(e))
        self.cfg.ui = ui
        return ApplyResult(True, "settings.msg.applied")

    # ---- 内部 ----

    def _restore_config(self, old_cfg: ProviderConfig | None) -> None:
        if old_cfg is not None:
            with self.registry._lock:
                self.registry._configs[old_cfg.id] = old_cfg
        # old_cfg 为 None 时候选是纯新增：配置留在 Registry 无害（无槽位引用、实例已退役）

    def _upsert_cfg_provider(self, provider: ProviderConfig) -> None:
        for i, p in enumerate(self.cfg.providers):
            if p.id == provider.id:
                self.cfg.providers[i] = provider
                return
        self.cfg.providers.append(provider)


# ==================== GUI ====================


def create_settings_dialog(
    controller: SettingsController,
    on_appearance_applied: Callable[[UIConfig], None] | None = None,
    on_appearance_preview: Callable[[UIConfig], None] | None = None,
    parent=None,
):
    """构建设置对话框（延迟导入 PyQt6）。

    on_appearance_preview：外观控件变化时即时预览（不保存）；保存走
    on_appearance_applied。关闭未保存的预览由调用方负责还原。
    """
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSpinBox,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    class SettingsDialog(QDialog):
        """设置对话框；耗时操作一律 worker 线程 + Qt 信号回 GUI。"""

        _op_done = pyqtSignal(object)  # 携带 GUI 线程回调

        def __init__(self):
            super().__init__(parent)
            self.controller = controller
            self._on_appearance_applied = on_appearance_applied
            self._on_appearance_preview = on_appearance_preview
            self._applying = False
            self.setWindowTitle(tr("settings.title"))
            self.resize(640, 520)
            # 关闭设置只关闭本页面，绝不退出应用（托盘常驻，退出走悬浮窗关闭按钮）
            self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
            self._op_done.connect(lambda cb: cb())

            tabs = QTabWidget(self)
            tabs.addTab(self._build_providers_tab(), tr("settings.tab.providers"))
            tabs.addTab(self._build_appearance_tab(), tr("settings.tab.appearance"))
            tabs.addTab(self._build_memory_tab(), tr("settings.tab.memory"))
            tabs.addTab(self._build_prompt_tab(), tr("settings.tab.prompt"))

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
            buttons.rejected.connect(self.reject)

            layout = QVBoxLayout(self)
            layout.addWidget(tabs)
            status_row = QHBoxLayout()
            self.status_label = QLabel("")
            self.status_label.setWordWrap(True)
            status_row.addWidget(self.status_label, 1)
            layout.addLayout(status_row)
            layout.addWidget(buttons)

        # ---- 异步执行 ----

        def _run_async(self, fn, on_done) -> None:
            self._set_applying(True)

            def target():
                try:
                    result = fn()
                except Exception as e:  # noqa: BLE001 —— worker 异常必须回 GUI 显示
                    log.debug("设置操作异常", exc_info=True)
                    result = ApplyResult(False, "settings.msg.internal_error", str(e))
                self._op_done.emit(lambda: self._finish_op(result, on_done))

            threading.Thread(target=target, name="settings-op", daemon=True).start()

        def _finish_op(self, result, on_done) -> None:
            self._set_applying(False)
            self._show_result(result)
            on_done(result)

        def _set_applying(self, applying: bool) -> None:
            self._applying = applying
            for btn in getattr(self, "_action_buttons", []):
                btn.setEnabled(not applying)

        def _show_result(self, result: ApplyResult) -> None:
            self.status_label.setText(("✓ " if result.ok else "✗ ") + result.text())
            self.status_label.setStyleSheet(
                f"color: {'#4CAF50' if result.ok else '#FF6B6B'};"
            )

        # ---- 模型与槽位页 ----

        def _build_providers_tab(self) -> QWidget:
            tab = QWidget(self)
            layout = QVBoxLayout(tab)

            # 槽位绑定区
            slots_box = QGroupBox(tr("settings.slots.group"), tab)
            slots_form = QFormLayout(slots_box)
            self._slot_combos: dict[str, QComboBox] = {}
            self._slot_apply: dict[str, QPushButton] = {}
            self._action_buttons: list[QPushButton] = []
            for slot in ALL_SLOTS:
                view = self.controller.slot_view(slot)
                row = QHBoxLayout()
                combo = QComboBox(slots_box)
                combo.addItem(tr("settings.slots.unbound"), "")
                for p in self.controller.candidates(slot):
                    combo.addItem(p.name, p.id)
                    combo.setItemData(
                        combo.count() - 1,
                        provider_detail_text(p),
                        Qt.ItemDataRole.ToolTipRole,
                    )
                index = combo.findData(view.effective_id)
                combo.setCurrentIndex(max(0, index))
                combo.setEnabled(view.editable)
                apply_btn = QPushButton(tr("settings.apply"), slots_box)
                apply_btn.setEnabled(view.editable)
                apply_btn.clicked.connect(
                    lambda _=False, s=slot: self._apply_slot_binding(s)
                )
                self._action_buttons.append(apply_btn)
                row.addWidget(combo, 1)
                row.addWidget(apply_btn)
                label = view.label
                if view.overridden_by_profile:
                    label += tr("settings.slots.profile_override")
                slots_form.addRow(label, row)
                self._slot_combos[slot] = combo
                self._slot_apply[slot] = apply_btn
            layout.addWidget(slots_box)

            # Provider 列表 + 编辑表单
            body = QHBoxLayout()
            self.provider_list = QListWidget(tab)
            self.provider_list.currentRowChanged.connect(self._load_provider_into_form)
            body.addWidget(self.provider_list, 1)

            form_widget = QWidget(tab)
            form = QFormLayout(form_widget)
            self.f_id = QLineEdit()
            self.f_name = QLineEdit()
            self.f_type = QComboBox()
            self.f_type.addItems(["openai_compatible", "llama_cpp_server", "vllm"])
            self.f_base_url = QLineEdit()
            self.f_api_key = QLineEdit()
            self.f_api_key.setEchoMode(QLineEdit.EchoMode.Password)  # 密钥默认掩码
            self.f_model = QLineEdit()
            self.f_timeout = QDoubleSpinBox()
            self.f_timeout.setRange(1, 600)
            self.f_timeout.setValue(30)
            self.f_retries = QSpinBox()
            self.f_retries.setRange(0, 10)
            self.f_retries.setValue(3)
            self.c_translation = QCheckBox(tr("settings.providers.cap.translation"))
            self.c_grammar = QCheckBox(tr("settings.providers.cap.grammar"))
            self.c_glossary = QCheckBox(tr("settings.providers.cap.glossary"))
            caps_row = QHBoxLayout()
            for cb in (self.c_translation, self.c_grammar, self.c_glossary):
                caps_row.addWidget(cb)
            caps_row.addStretch(1)
            form.addRow("id", self.f_id)
            form.addRow(tr("settings.providers.field.name"), self.f_name)
            form.addRow(tr("settings.providers.field.type"), self.f_type)
            form.addRow("base_url", self.f_base_url)
            form.addRow("api_key", self.f_api_key)
            form.addRow("model", self.f_model)
            form.addRow(tr("settings.providers.field.capabilities"), caps_row)
            form.addRow(tr("settings.providers.field.timeout"), self.f_timeout)
            form.addRow(tr("settings.providers.field.retries"), self.f_retries)
            self.local_notice = QLabel(tr("settings.providers.local_notice"))
            self.local_notice.setWordWrap(True)
            self.local_notice.setVisible(False)
            form.addRow(self.local_notice)
            body.addWidget(form_widget, 2)
            layout.addLayout(body)

            btn_row = QHBoxLayout()
            self.btn_new = QPushButton(tr("settings.providers.new"))
            self.btn_save = QPushButton(tr("settings.providers.save"))
            self.btn_delete = QPushButton(tr("settings.providers.delete"))
            self.btn_health = QPushButton(tr("settings.providers.health"))
            self._action_buttons += [self.btn_save, self.btn_delete, self.btn_health]
            self.btn_new.clicked.connect(self._new_provider)
            self.btn_save.clicked.connect(self._save_provider)
            self.btn_delete.clicked.connect(self._delete_provider)
            self.btn_health.clicked.connect(self._health_check_selected)
            for b in (self.btn_new, self.btn_save, self.btn_delete, self.btn_health):
                btn_row.addWidget(b)
            btn_row.addStretch(1)
            layout.addLayout(btn_row)

            self._reload_provider_list()
            return tab

        def _reload_provider_list(self, select_id: str = "") -> None:
            self.provider_list.clear()
            from PyQt6.QtCore import Qt as _Qt
            from PyQt6.QtWidgets import QListWidgetItem

            for p in self.controller.cfg.providers:
                item = QListWidgetItem(p.name)  # 主行只显示名称，细节放 tooltip
                item.setData(_Qt.ItemDataRole.UserRole, p.id)
                item.setToolTip(provider_detail_text(p))
                self.provider_list.addItem(item)
            if select_id:
                ids = self.controller.provider_ids()
                if select_id in ids:
                    self.provider_list.setCurrentRow(ids.index(select_id))

        def _selected_provider(self) -> ProviderConfig | None:
            row = self.provider_list.currentRow()
            ids = self.controller.provider_ids()
            if 0 <= row < len(ids):
                return next(
                    (p for p in self.controller.cfg.providers if p.id == ids[row]), None
                )
            return None

        def _load_provider_into_form(self, _row: int) -> None:
            p = self._selected_provider()
            if p is None:
                return
            self.f_id.setText(p.id)
            self.f_id.setReadOnly(True)  # id 是主键，编辑即同 id 新代
            self.f_name.setText(p.name)
            self.f_type.setCurrentText(p.type)
            self.f_base_url.setText(p.base_url)
            self.f_api_key.setText(p.api_key)
            self.f_model.setText(p.model)
            self.f_timeout.setValue(p.timeout_sec)
            self.f_retries.setValue(p.max_retries)
            self.c_translation.setChecked("translation" in p.capabilities)
            self.c_grammar.setChecked("grammar" in p.capabilities)
            self.c_glossary.setChecked("glossary_constraint" in p.capabilities)
            self.local_notice.setVisible(p.local is not None)

        def _new_provider(self) -> None:
            self.provider_list.clearSelection()
            self.f_id.setReadOnly(False)
            existing = set(self.controller.provider_ids())
            n = 1
            while f"cloud-{n}" in existing:
                n += 1
            for field in (self.f_id, self.f_name, self.f_base_url, self.f_api_key, self.f_model):
                field.clear()
            self.f_id.setText(f"cloud-{n}")
            self.f_type.setCurrentText("openai_compatible")
            self.c_translation.setChecked(True)
            self.c_grammar.setChecked(True)
            self.c_glossary.setChecked(False)
            self.local_notice.setVisible(False)

        def _provider_from_form(self) -> ProviderConfig:
            caps = []
            if self.c_translation.isChecked():
                caps.append("translation")
            if self.c_grammar.isChecked():
                caps.append("grammar")
            if self.c_glossary.isChecked():
                caps.append("glossary_constraint")
            existing = next(
                (p for p in self.controller.cfg.providers if p.id == self.f_id.text().strip()),
                None,
            )
            if existing is not None:
                # 同 id 编辑：保留表单外的字段（local/params 等），只覆盖表单字段
                provider = copy.deepcopy(existing)
                provider.name = self.f_name.text().strip() or provider.id
                provider.type = self.f_type.currentText()
                provider.base_url = self.f_base_url.text().strip()
                provider.api_key = self.f_api_key.text().strip()
                provider.model = self.f_model.text().strip()
                provider.capabilities = caps
                provider.timeout_sec = float(self.f_timeout.value())
                provider.max_retries = int(self.f_retries.value())
                return provider
            return ProviderConfig(
                id=self.f_id.text().strip(),
                name=self.f_name.text().strip() or self.f_id.text().strip(),
                type=self.f_type.currentText(),
                base_url=self.f_base_url.text().strip(),
                api_key=self.f_api_key.text().strip(),
                model=self.f_model.text().strip(),
                capabilities=caps,
                timeout_sec=float(self.f_timeout.value()),
                max_retries=int(self.f_retries.value()),
            )

        def _apply_slot_binding(self, slot: str) -> None:
            pid = self._slot_combos[slot].currentData() or ""

            def work():
                return self.controller.apply_binding(
                    slot, pid, health_check=self.controller.registry.check_health
                )

            self._run_async(work, lambda _r: None)

        def _save_provider(self) -> None:
            provider = self._provider_from_form()

            def work():
                return self.controller.apply_provider(
                    provider, health_check=self.controller.registry.check_health
                )

            def done(result):
                if result.ok:
                    self._reload_provider_list(select_id=provider.id)
                    self._refresh_slot_combos()

            self._run_async(work, done)

        def _delete_provider(self) -> None:
            p = self._selected_provider()
            if p is None:
                return
            answer = QMessageBox.question(
                self, tr("settings.providers.delete"),
                tr("settings.providers.delete_confirm") + f"\n{p.name}（{p.id}）",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            result = self.controller.remove_provider(p.id)  # 纯本地操作，无需 worker
            self._show_result(result)
            if result.ok:
                self._reload_provider_list()
                self._refresh_slot_combos()

        def _health_check_selected(self) -> None:
            p = self._selected_provider()
            if p is None:
                return
            self.status_label.setText(tr("settings.health.checking"))

            def work():
                err = self.controller.registry.check_health(p.id)
                return ApplyResult(err is None, "settings.msg.health_ok" if err is None else "settings.msg.health_failed", err or "")

            self._run_async(work, lambda _r: None)

        def _refresh_slot_combos(self) -> None:
            for slot in ALL_SLOTS:
                view = self.controller.slot_view(slot)
                combo = self._slot_combos[slot]
                current = combo.currentData()
                combo.clear()
                combo.addItem(tr("settings.slots.unbound"), "")
                for p in self.controller.candidates(slot):
                    combo.addItem(p.name, p.id)
                    combo.setItemData(
                        combo.count() - 1,
                        provider_detail_text(p),
                        Qt.ItemDataRole.ToolTipRole,
                    )
                index = combo.findData(view.effective_id or current)
                combo.setCurrentIndex(max(0, index))

        # ---- 外观页 ----

        def _build_appearance_tab(self) -> QWidget:
            tab = QWidget(self)
            form = QFormLayout(tab)
            ui = self.controller.cfg.ui

            from .appearance import THEME_NAMES

            self.a_theme = QComboBox()
            self.a_theme.setObjectName("themeCombo")
            for name in THEME_NAMES:
                self.a_theme.addItem(tr(f"settings.appearance.theme.{name}"), name)
            index = self.a_theme.findData(ui.theme)
            self.a_theme.setCurrentIndex(max(0, index))
            self.a_opacity = QSpinBox()
            self.a_opacity.setRange(20, 100)
            self.a_opacity.setSuffix(" %")
            self.a_opacity.setValue(round(ui.effective_background_opacity * 100))
            self.a_line_spacing = QDoubleSpinBox()
            self.a_line_spacing.setRange(1.0, 3.0)
            self.a_line_spacing.setSingleStep(0.05)
            self.a_line_spacing.setValue(ui.line_spacing)
            self.a_width = QSpinBox()
            self.a_width.setRange(320, 3840)
            self.a_width.setValue(ui.window_width)
            self.a_height = QSpinBox()
            self.a_height.setRange(200, 2160)
            self.a_height.setValue(ui.window_height)
            self.a_bar_a = QCheckBox()
            self.a_bar_a.setChecked(ui.show_bar_a)
            self.a_bar_b = QCheckBox()
            self.a_bar_b.setChecked(ui.show_bar_b)
            self.a_bar_c = QCheckBox()
            self.a_bar_c.setChecked(ui.show_bar_c)
            self.a_auto_scroll = QCheckBox()
            self.a_auto_scroll.setChecked(ui.auto_scroll)
            self.a_compact = QCheckBox()
            self.a_compact.setChecked(ui.compact_mode)
            self.a_ui_language = QComboBox()
            self.a_ui_language.addItems(["zh_CN", "en"])
            self.a_ui_language.setCurrentText(ui.ui_language)
            self.a_capacity = QSpinBox()
            self.a_capacity.setRange(20, 2000)
            self.a_capacity.setValue(ui.view_capacity)

            form.addRow(tr("settings.appearance.theme"), self.a_theme)
            form.addRow(tr("settings.appearance.background_opacity"), self.a_opacity)
            form.addRow(tr("settings.appearance.line_spacing"), self.a_line_spacing)
            size_row = QHBoxLayout()
            size_row.addWidget(self.a_width)
            size_row.addWidget(QLabel("×"))
            size_row.addWidget(self.a_height)
            size_row.addStretch(1)
            form.addRow(tr("settings.appearance.window_size"), size_row)
            bars_row = QHBoxLayout()
            for cb, key in (
                (self.a_bar_a, "settings.appearance.bar_a"),
                (self.a_bar_b, "settings.appearance.bar_b"),
                (self.a_bar_c, "settings.appearance.bar_c"),
            ):
                bars_row.addWidget(QLabel(tr(key)))
                bars_row.addWidget(cb)
            bars_row.addStretch(1)
            form.addRow(tr("settings.appearance.bars"), bars_row)
            form.addRow(tr("settings.appearance.auto_scroll"), self.a_auto_scroll)
            form.addRow(tr("settings.appearance.compact_mode"), self.a_compact)
            form.addRow(tr("settings.appearance.ui_language"), self.a_ui_language)
            form.addRow(tr("settings.appearance.view_capacity"), self.a_capacity)

            btn_row = QHBoxLayout()
            self.btn_apply_appearance = QPushButton(tr("settings.apply"))
            self.btn_restore_defaults = QPushButton(tr("settings.appearance.restore_defaults"))
            self._action_buttons.append(self.btn_apply_appearance)
            self.btn_apply_appearance.clicked.connect(self._apply_appearance)
            self.btn_restore_defaults.clicked.connect(self._restore_appearance_defaults)
            btn_row.addWidget(self.btn_apply_appearance)
            btn_row.addWidget(self.btn_restore_defaults)
            btn_row.addStretch(1)
            form.addRow(btn_row)

            # 即时预览：控件变化即回调预览（不保存），保存走「应用」按钮
            self.a_theme.currentIndexChanged.connect(lambda _i: self._preview_appearance())
            for w in (self.a_opacity, self.a_line_spacing,
                      self.a_width, self.a_height, self.a_capacity):
                w.valueChanged.connect(lambda _v: self._preview_appearance())
            for cb in (self.a_bar_a, self.a_bar_b, self.a_bar_c,
                       self.a_auto_scroll, self.a_compact):
                cb.toggled.connect(lambda _v: self._preview_appearance())
            return tab

        def _preview_appearance(self) -> None:
            if self._on_appearance_preview is not None:
                self._on_appearance_preview(self._appearance_from_form())

        def _appearance_from_form(self) -> UIConfig:
            ui = dataclasses.replace(self.controller.cfg.ui)  # 保留 overlay 等表单外字段
            ui.theme = self.a_theme.currentData() or "dark"
            ui.background_opacity = self.a_opacity.value() / 100.0
            ui.line_spacing = float(self.a_line_spacing.value())
            ui.window_width = int(self.a_width.value())
            ui.window_height = int(self.a_height.value())
            ui.show_bar_a = self.a_bar_a.isChecked()
            ui.show_bar_b = self.a_bar_b.isChecked()
            ui.show_bar_c = self.a_bar_c.isChecked()
            ui.auto_scroll = self.a_auto_scroll.isChecked()
            ui.compact_mode = self.a_compact.isChecked()
            ui.ui_language = self.a_ui_language.currentText()
            ui.view_capacity = int(self.a_capacity.value())
            return ui

        def _apply_appearance(self) -> None:
            ui = self._appearance_from_form()
            result = self.controller.apply_appearance(ui)  # 纯本地原子写，无需 worker
            self._show_result(result)
            if result.ok and self._on_appearance_applied is not None:
                self._on_appearance_applied(ui)

        def _restore_appearance_defaults(self) -> None:
            answer = QMessageBox.question(
                self,
                tr("settings.appearance.restore_defaults"),
                tr("settings.appearance.restore_confirm"),
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            defaults = UIConfig()
            ui = dataclasses.replace(defaults, overlay=self.controller.cfg.ui.overlay)
            result = self.controller.apply_appearance(ui)
            self._show_result(result)
            if result.ok:
                self._reload_appearance_form(ui)
                if self._on_appearance_applied is not None:
                    self._on_appearance_applied(ui)

        def _reload_appearance_form(self, ui: UIConfig) -> None:
            index = self.a_theme.findData(ui.theme)
            self.a_theme.setCurrentIndex(max(0, index))
            self.a_opacity.setValue(round(ui.effective_background_opacity * 100))
            self.a_line_spacing.setValue(ui.line_spacing)
            self.a_width.setValue(ui.window_width)
            self.a_height.setValue(ui.window_height)
            self.a_bar_a.setChecked(ui.show_bar_a)
            self.a_bar_b.setChecked(ui.show_bar_b)
            self.a_bar_c.setChecked(ui.show_bar_c)
            self.a_auto_scroll.setChecked(ui.auto_scroll)
            self.a_compact.setChecked(ui.compact_mode)
            self.a_ui_language.setCurrentText(ui.ui_language)
            self.a_capacity.setValue(ui.view_capacity)

        # ==================== 记忆页签 ====================

        def _build_memory_tab(self) -> QWidget:
            tab = QWidget(self)
            layout = QVBoxLayout(tab)
            self.m_hint = QLabel("", tab)
            self.m_hint.setWordWrap(True)
            layout.addWidget(self.m_hint)

            self.m_table = QTableWidget(0, 3, tab)
            self.m_table.setHorizontalHeaderLabels([
                tr("settings.memory.term"),
                tr("settings.memory.translation"),
                tr("settings.memory.type"),
            ])
            self.m_table.horizontalHeader().setStretchLastSection(True)
            layout.addWidget(self.m_table, 1)

            row_btns = QHBoxLayout()
            self.btn_m_add = QPushButton(tr("settings.memory.add"), tab)
            self.btn_m_remove = QPushButton(tr("settings.memory.remove"), tab)
            self.btn_m_add.clicked.connect(lambda: self._memory_add_row("", "", ""))
            self.btn_m_remove.clicked.connect(self._memory_remove_selected)
            row_btns.addWidget(self.btn_m_add)
            row_btns.addWidget(self.btn_m_remove)
            row_btns.addStretch(1)
            layout.addLayout(row_btns)

            self.m_style_label = QLabel(tr("settings.memory.style"), tab)
            layout.addWidget(self.m_style_label)
            self.m_style = QPlainTextEdit(tab)
            self.m_style.setMaximumHeight(80)
            layout.addWidget(self.m_style)

            action_row = QHBoxLayout()
            self.btn_m_save = QPushButton(tr("settings.apply"), tab)
            self.btn_m_reload = QPushButton(tr("settings.memory.reload"), tab)
            self.btn_m_clear = QPushButton(tr("settings.memory.clear"), tab)
            self.btn_m_load_file = QPushButton(tr("settings.memory.load_file"), tab)
            self.btn_m_save_file = QPushButton(tr("settings.memory.save_file"), tab)
            self.btn_m_save.clicked.connect(self._memory_save)
            self.btn_m_reload.clicked.connect(self._reload_memory_editor)
            self.btn_m_clear.clicked.connect(self._memory_clear)
            self.btn_m_load_file.clicked.connect(self._memory_load_file)
            self.btn_m_save_file.clicked.connect(self._memory_save_file)
            for b in (self.btn_m_save, self.btn_m_reload, self.btn_m_clear,
                      self.btn_m_load_file, self.btn_m_save_file):
                action_row.addWidget(b)
            action_row.addStretch(1)
            layout.addLayout(action_row)

            self._memory_dirty = False
            self._memory_loading = False
            self._memory_file = None  # 无实时记忆时编辑的文件路径
            self.m_table.itemChanged.connect(lambda _i: self._memory_mark_dirty())
            self.m_style.textChanged.connect(self._memory_mark_dirty)
            self._reload_memory_editor()
            return tab

        def _memory_source_memory(self):
            """编辑目标：优先实时 Translator.memory（同一实例，下一 prompt 生效）。"""
            return self.controller.live_memory()

        def _reload_memory_editor(self) -> None:
            from ..nlp.context_memory import RollingMemory

            mem = self._memory_source_memory()
            self._memory_loading = True
            try:
                self.m_table.setRowCount(0)
                if mem is not None:
                    for e in mem.glossary_snapshot():
                        self._memory_add_row(e.term, e.translation, e.type)
                    self.m_style.setPlainText(mem.style_prompt())
                    locked = mem.style_locked
                    self.m_style.setReadOnly(locked)
                    self.m_style_label.setText(
                        tr("settings.memory.style")
                        + (tr("settings.memory.style_locked") if locked else "")
                    )
                    self.m_hint.setText(tr("settings.memory.live_hint"))
                    self._memory_file = None
                else:
                    self.m_style.setPlainText("")
                    self.m_style.setReadOnly(False)
                    self.m_hint.setText(tr("settings.memory.no_live"))
                self._memory_set_dirty(False)
            finally:
                self._memory_loading = False

        def _memory_add_row(self, term: str, translation: str, type_: str) -> None:
            row = self.m_table.rowCount()
            self.m_table.insertRow(row)
            for col, value in enumerate((term, translation, type_)):
                self.m_table.setItem(row, col, QTableWidgetItem(value))

        def _memory_remove_selected(self) -> None:
            for index in sorted(
                {i.row() for i in self.m_table.selectedIndexes()}, reverse=True
            ):
                self.m_table.removeRow(index)

        def _memory_mark_dirty(self) -> None:
            if not self._memory_loading:
                self._memory_set_dirty(True)

        def _memory_set_dirty(self, dirty: bool) -> None:
            self._memory_dirty = dirty
            self.btn_m_save.setText(
                ("● " if dirty else "") + tr("settings.apply")
            )

        def _memory_staged(self) -> list[tuple[str, str, str]]:
            """表格暂存内容；空字段抛 MemoryValidationError。"""
            from ..nlp.context_memory import MemoryValidationError

            rows = []
            seen = set()
            for row in range(self.m_table.rowCount()):
                term = (self.m_table.item(row, 0) or QTableWidgetItem("")).text().strip()
                translation = (self.m_table.item(row, 1) or QTableWidgetItem("")).text().strip()
                type_ = (self.m_table.item(row, 2) or QTableWidgetItem("")).text().strip()
                if not term and not translation:
                    continue  # 完全空行忽略
                if not term or not translation:
                    raise MemoryValidationError(
                        tr("settings.memory.empty_field") + f"（第 {row + 1} 行）"
                    )
                if term in seen:
                    raise MemoryValidationError(
                        tr("settings.memory.duplicate") + f"：{term}"
                    )
                seen.add(term)
                rows.append((term, translation, type_))
            return rows

        def _memory_save(self) -> None:
            from ..nlp.context_memory import MemoryValidationError, RollingMemory

            try:
                rows = self._memory_staged()
                mem = self._memory_source_memory()
                if mem is not None:
                    existing = {e.term for e in mem.glossary_snapshot()}
                    staged = {t: (tr_, ty) for t, tr_, ty in rows}
                    for term in existing - staged.keys():
                        mem.remove_term(term)
                    for term, (translation, type_) in staged.items():
                        mem.set_term(term, translation, type_)
                    if not mem.style_locked:
                        mem.set_style(self.m_style.toPlainText())
                    # 实时记忆：同一实例即改即生效，会话结束随 recorder 落盘
                    if self._memory_file is not None:
                        mem.save(self._memory_file,
                                 language_profile=self.controller.profile.name
                                 if self.controller.profile else "")
                elif self._memory_file is not None:
                    export = RollingMemory()
                    for term, translation, type_ in rows:
                        export.set_term(term, translation, type_)
                    export.style_notes = self.m_style.toPlainText().strip()
                    export.save(self._memory_file)
                else:
                    self.status_label.setText("✗ " + tr("settings.memory.no_live"))
                    return
            except MemoryValidationError as e:
                QMessageBox.warning(self, tr("settings.tab.memory"), str(e))
                return
            except OSError as e:
                QMessageBox.warning(self, tr("settings.tab.memory"), str(e))
                return
            self._memory_set_dirty(False)
            self.status_label.setText("✓ " + tr("settings.msg.memory_saved"))
            self.status_label.setStyleSheet("color: #4CAF50;")

        def _memory_clear(self) -> None:
            answer = QMessageBox.question(
                self, tr("settings.memory.clear"), tr("settings.memory.clear_confirm")
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self.m_table.setRowCount(0)
            if not self.m_style.isReadOnly():
                self.m_style.setPlainText("")
            self._memory_set_dirty(True)  # 清空也是暂存修改，应用后生效

        def _memory_load_file(self) -> None:
            from ..nlp.context_memory import MemoryValidationError, RollingMemory

            path, _ = QFileDialog.getOpenFileName(
                self, tr("settings.memory.load_file"), "", "memory.json (memory*.json);;JSON (*.json)"
            )
            if not path:
                return
            try:
                mem = RollingMemory.load(
                    path,
                    strict=True,
                    expected_profile=self.controller.profile.name
                    if self.controller.profile
                    else "",
                )
            except MemoryValidationError as e:
                QMessageBox.warning(self, tr("settings.memory.load_file"), str(e))
                return
            self._memory_loading = True
            try:
                self.m_table.setRowCount(0)
                for e in mem.glossary_snapshot():
                    self._memory_add_row(e.term, e.translation, e.type)
                self.m_style.setPlainText(mem.style_prompt())
            finally:
                self._memory_loading = False
            self._memory_file = __import__("pathlib").Path(path)
            self.m_hint.setText(f"{tr('settings.memory.editing_file')}：{path}")
            self._memory_set_dirty(False)

        def _memory_save_file(self) -> None:
            from ..nlp.context_memory import MemoryValidationError, RollingMemory

            try:
                rows = self._memory_staged()
            except MemoryValidationError as e:
                QMessageBox.warning(self, tr("settings.tab.memory"), str(e))
                return
            default = str(self._memory_file) if self._memory_file else "memory.json"
            path, _ = QFileDialog.getSaveFileName(
                self, tr("settings.memory.save_file"), default, "JSON (*.json)"
            )
            if not path:
                return
            export = RollingMemory()
            for term, translation, type_ in rows:
                export.set_term(term, translation, type_)
            export.style_notes = self.m_style.toPlainText().strip()
            try:
                export.save(
                    __import__("pathlib").Path(path),
                    language_profile=self.controller.profile.name
                    if self.controller.profile
                    else "",
                )
            except OSError as e:
                QMessageBox.warning(self, tr("settings.memory.save_file"), str(e))
                return
            self.status_label.setText("✓ " + tr("settings.msg.memory_saved"))

        # ==================== Prompt 页签 ====================

        def _build_prompt_tab(self) -> QWidget:
            from ..nlp.prompt_engine import PromptEngine, PromptTemplateError

            tab = QWidget(self)
            layout = QVBoxLayout(tab)
            prompts_dir = self.controller.cfg.prompts_dir
            dir_row = QHBoxLayout()
            dir_row.addWidget(QLabel(f"{tr('settings.prompt.dir')}：{prompts_dir}", tab), 1)
            btn_open_dir = QPushButton(tr("settings.prompt.open_dir"), tab)
            btn_open_dir.clicked.connect(lambda: self._open_path(prompts_dir))
            dir_row.addWidget(btn_open_dir)
            layout.addLayout(dir_row)

            engine = PromptEngine(prompts_dir)
            profile = self.controller.profile
            prompt_path = getattr(profile, "prompt_path", None)
            for kind in ("translate", "translate_insttrans", "grammar"):
                row = QHBoxLayout()
                try:
                    path = engine.resolve(
                        prompt_path(kind) if callable(prompt_path) else None, kind
                    )
                    path_text = str(path)
                except PromptTemplateError:
                    path_text = tr("settings.prompt.missing")
                row.addWidget(QLabel(f"{kind}：{path_text}", tab), 1)
                btn_restore = QPushButton(tr("settings.prompt.restore"), tab)
                btn_restore.setObjectName(f"restorePrompt_{kind}")
                btn_restore.clicked.connect(
                    lambda _=False, k=kind: self._restore_prompt(k)
                )
                row.addWidget(btn_restore)
                layout.addLayout(row)
            layout.addStretch(1)
            return tab

        def _open_path(self, path) -> None:
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

        def _restore_prompt(self, kind: str) -> None:
            answer = QMessageBox.question(
                self,
                tr("settings.prompt.restore"),
                tr("settings.prompt.restore_confirm") + f"（{kind}）",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._show_result(self.controller.restore_prompt_template(kind))

    return SettingsDialog()
