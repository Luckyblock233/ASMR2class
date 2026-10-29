"""首次运行资源下载确认窗（启动器流程的一环）。

所列资源缺失时弹出：用户确认后逐项下载（worker 线程，GUI 不阻塞），
全部完成后做健康检查（拉起本地 Provider / ping 云端），通过才放行启动；
失败可重试或取消。on_download / on_health 可注入（测试用假实现）。
"""

from __future__ import annotations

import threading

from ..resources import ResourceNeed


def create_download_dialog(needs: list[ResourceNeed], *, on_download, on_health, parent=None):
    """构建下载确认对话框（延迟导入 PyQt6）。

    on_download(need, log_fn)：下载单项资源，异常即失败。
    on_health() -> list[str]：健康检查，返回错误列表（空 = 通过）。
    parent：父窗口（启动器/设置页），模态遮罩其上方。
    """
    from PyQt6.QtCore import QObject, Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QDialog,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
    )

    from .i18n import tr

    class _Bridge(QObject):
        status = pyqtSignal(int, str)   # 行号, 状态文案
        log_line = pyqtSignal(str)
        finished = pyqtSignal(list, list)  # (下载错误, 健康检查警告)

    class DownloadDialog(QDialog):
        def __init__(self):
            super().__init__(parent)
            self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
            self.setWindowTitle(tr("download.title"))
            self.resize(620, 420)
            self.ok = False  # True = 资源就绪，放行启动
            self._running = False

            layout = QVBoxLayout(self)
            layout.addWidget(QLabel(tr("download.intro")))

            self.tree = QTreeWidget(self)
            self.tree.setHeaderLabels([tr("download.col.resource"), tr("download.col.status")])
            self.tree.setRootIsDecorated(False)
            for need in needs:
                item = QTreeWidgetItem([f"{need.label}\n{need.detail}", tr("download.status.pending")])
                self.tree.addTopLevelItem(item)
            self.tree.expandAll()
            self.tree.resizeColumnToContents(0)
            layout.addWidget(self.tree, 1)

            self.log_view = QPlainTextEdit(self)
            self.log_view.setReadOnly(True)
            self.log_view.setMaximumBlockCount(1000)
            layout.addWidget(self.log_view, 1)

            self.result_label = QLabel("")
            self.result_label.setWordWrap(True)
            layout.addWidget(self.result_label)

            btn_row = QHBoxLayout()
            self.btn_start = QPushButton(tr("download.start"))
            self.btn_continue = QPushButton(tr("download.continue_anyway"))
            self.btn_continue.setVisible(False)  # 仅健康检查未过时可强行启动
            self.btn_cancel = QPushButton(tr("download.cancel"))
            self.btn_start.clicked.connect(self._start)
            self.btn_continue.clicked.connect(self._continue_anyway)
            self.btn_cancel.clicked.connect(self.reject)
            btn_row.addStretch(1)
            btn_row.addWidget(self.btn_continue)
            btn_row.addWidget(self.btn_start)
            btn_row.addWidget(self.btn_cancel)
            layout.addLayout(btn_row)

            self._bridge = _Bridge(self)
            self._bridge.status.connect(self._on_status)
            self._bridge.log_line.connect(self.log_view.appendPlainText)
            self._bridge.finished.connect(self._on_finished)

        def _log(self, msg: str) -> None:
            self._bridge.log_line.emit(msg)

        def _start(self) -> None:
            if self._running:
                return
            self._running = True
            self.btn_start.setEnabled(False)
            self.btn_cancel.setEnabled(False)  # 下载中不可取消（GGUF 落盘是原子的 .part→改名）
            self.result_label.setText("")

            def work() -> None:
                download_errors: list[str] = []
                health_warnings: list[str] = []
                for i, need in enumerate(needs):
                    self._bridge.status.emit(i, tr("download.status.downloading"))
                    try:
                        on_download(need, self._log)
                        self._bridge.status.emit(i, tr("download.status.ok"))
                    except Exception as e:  # noqa: BLE001 —— 逐项收集错误，继续下一项
                        download_errors.append(f"{need.label}：{type(e).__name__}: {e}")
                        self._bridge.status.emit(i, tr("download.status.failed"))
                if not download_errors:
                    self._bridge.log_line.emit(tr("download.health"))
                    try:
                        health_warnings.extend(on_health())
                    except Exception as e:  # noqa: BLE001
                        health_warnings.append(f"{type(e).__name__}: {e}")
                self._bridge.finished.emit(download_errors, health_warnings)

            threading.Thread(target=work, name="resource-download", daemon=True).start()

        def _on_status(self, row: int, text: str) -> None:
            item = self.tree.topLevelItem(row)
            if item is not None:
                item.setText(1, text)

        def _continue_anyway(self) -> None:
            """资源已下载但健康检查未过（多为云端密钥缺失）：用户选择强行启动。"""
            self.ok = True
            self.accept()

        def _on_finished(self, download_errors: list, health_warnings: list) -> None:
            self._running = False
            self.btn_cancel.setEnabled(True)
            if not download_errors and not health_warnings:
                self.result_label.setText(tr("download.done"))
                self.ok = True
                self.accept()
                return
            if download_errors:
                # 下载失败：死路，只能重试或取消
                self.result_label.setStyleSheet("color: #C62828;")
                self.result_label.setText(
                    tr("download.failed") + "\n" + "\n".join(str(e) for e in download_errors)
                )
                self.btn_start.setEnabled(True)  # 允许重试
                return
            # 下载成功但健康检查未过（典型：云端 API key 未填）：
            # 不阻断——可返回修改，也可强行启动（程序内降级为仅显示原文）
            self.result_label.setStyleSheet("color: #E65100;")
            self.result_label.setText(
                tr("download.health_failed") + "\n" + "\n".join(str(e) for e in health_warnings)
            )
            self.btn_continue.setVisible(True)

    return DownloadDialog()
