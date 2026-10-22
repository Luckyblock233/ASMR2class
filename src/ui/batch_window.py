"""离线批处理进度窗口（与实时悬浮窗不同的独立 UI）。

窗口先出框架（文件列表/进度/日志），ASR 引擎加载与跑批在 worker 线程：
ready 信号把建好的 runner 交回 GUI 线程启动；取消按钮置 cancel_event，
runner 保存已完成部分后退出。
"""

from __future__ import annotations

import threading


def create_batch_shell(files: list[str]):
    """构建批处理窗口框架；runner 就绪后由 __ready__ 事件启动。"""
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    from .i18n import tr

    class BatchWindow(QWidget):
        _event = pyqtSignal(str, object)  # (kind, data dict)

        def __init__(self):
            super().__init__()
            # 批处理窗口是该模式的主窗口：关闭即结束（run_batch_window 已设
            # quitOnLastWindowClosed=True），不设 WA_QuitOnClose=False。
            self.setWindowTitle(tr("batch.title"))
            self.resize(640, 480)
            self._rc: int | None = None
            self._runner = None
            self._worker: threading.Thread | None = None

            self.file_list = QListWidget(self)
            self._items: dict[str, QListWidgetItem] = {}
            for path in files:
                item = QListWidgetItem(f"⏳ {path}")
                self._items[path] = item
                self.file_list.addItem(item)

            self.progress = QProgressBar(self)
            self.progress.setRange(0, len(files))
            self.progress.setValue(0)
            self.progress.setFormat(tr("batch.progress") + " %v/%m")

            self.log_view = QPlainTextEdit(self)
            self.log_view.setReadOnly(True)

            self.btn_cancel = QPushButton(tr("batch.cancel"), self)
            self.btn_cancel.clicked.connect(self._on_cancel)
            self.btn_close = QPushButton(tr("batch.close"), self)
            self.btn_close.clicked.connect(self.close)
            self.btn_close.setEnabled(False)

            btn_row = QHBoxLayout()
            btn_row.addStretch(1)
            btn_row.addWidget(self.btn_cancel)
            btn_row.addWidget(self.btn_close)

            layout = QVBoxLayout(self)
            layout.addWidget(self.file_list, 1)
            layout.addWidget(self.progress)
            layout.addWidget(self.log_view, 1)
            layout.addLayout(btn_row)

            self._event.connect(self._on_event)

        # ---- runner 接线（GUI 线程）----

        def start_runner(self, runner) -> None:
            self._runner = runner
            runner._on_event = lambda kind, **kw: self._emit(kind, **kw)
            self._worker = threading.Thread(
                target=self._run, name="batch-runner", daemon=True
            )
            self._worker.start()

        def _run(self) -> None:
            rc = self._runner.run(files)
            self._event.emit("__done__", {"rc": rc})

        def _emit(self, kind: str, data: dict | None = None, **kw) -> None:
            self._event.emit(kind, data if data is not None else kw)

        def _on_event(self, kind: str, data: dict) -> None:
            if kind == "__ready__":
                self._log(tr("batch.file_start") + "…")
                self.start_runner(data["runner"])
                return
            if kind == "__done__":
                self._rc = data["rc"]
                self.btn_cancel.setEnabled(False)
                self.btn_close.setEnabled(True)
                return
            if kind == "log":
                self._log(data.get("text", ""))
                return
            path = data.get("path", "")
            item = self._items.get(path)
            if kind == "file_start":
                if item:
                    item.setText(f"▶ {path}")
                self._log(tr("batch.file_start") + f"：{path}")
            elif kind == "file_done":
                if item:
                    item.setText(f"✓ {path}")
                self.progress.setValue(self.progress.value() + 1)
                self._log(tr("batch.file_done") + f"：{data.get('out_dir') or path}")
            elif kind == "file_error":
                if item:
                    item.setText(f"✗ {path}")
                self.progress.setValue(self.progress.value() + 1)
                self._log(tr("batch.file_error") + f"：{path}（{data.get('error', '')}）")
            elif kind == "cancelled":
                if item:
                    item.setText(f"⊘ {path}")
                self._log(tr("batch.cancelled"))
            elif kind == "sentence":
                self.progress.setFormat(
                    tr("batch.progress") + f" %v/%m · {data.get('count', 0)} 句"
                )
            elif kind == "all_done":
                self._log(
                    tr("batch.all_done").format(
                        ok=data.get("total", 0) - data.get("failures", 0),
                        total=data.get("total", 0),
                    )
                )

        def _log(self, text: str) -> None:
            self.log_view.appendPlainText(text)

        def _on_cancel(self) -> None:
            if self._runner is not None:
                self._runner._cancel.set()
            self.btn_cancel.setEnabled(False)
            self._log(tr("batch.cancelling"))

        def closeEvent(self, event) -> None:
            # 关闭窗口 = 取消在途批处理（runner 保存已完成部分后退出）
            if self._runner is not None and self._rc is None:
                self._runner._cancel.set()
            super().closeEvent(event)

        @property
        def return_code(self) -> int | None:
            return self._rc

    return BatchWindow()
