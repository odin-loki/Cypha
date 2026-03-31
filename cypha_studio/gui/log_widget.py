"""
Append-only studio log: status and errors from ``SignalBus``.
"""
from __future__ import annotations

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..server.local_server import SignalBus

_MAX_LINES = 2000


class LogDockWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._bus = SignalBus.instance()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Studio log"))
        hdr.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(56)
        clear_btn.clicked.connect(self._clear)
        hdr.addWidget(clear_btn)
        layout.addLayout(hdr)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 10px; color: #ccc;"
        )
        layout.addWidget(self._text)

        self._bus.status_message.connect(self._on_status)
        self._bus.error_occurred.connect(self._on_error)
        self._bus.training_finished.connect(self._on_train_done)

    def _trim(self) -> None:
        doc = self._text.document()
        while doc.blockCount() > _MAX_LINES and doc.blockCount() > 1:
            cur = QTextCursor(doc.firstBlock())
            cur.select(QTextCursor.BlockUnderCursor)
            cur.removeSelectedText()
            cur.deleteChar()

    def _append(self, line: str) -> None:
        self._text.append(line.rstrip())
        self._trim()
        self._text.verticalScrollBar().setValue(
            self._text.verticalScrollBar().maximum()
        )

    def _clear(self) -> None:
        self._text.clear()

    def _on_status(self, msg: str) -> None:
        self._append(f"[status] {msg}")

    def _on_error(self, msg: str) -> None:
        self._append(f"[error] {msg}")

    def _on_train_done(self, metrics: dict) -> None:
        acc = metrics.get("accuracy", 0.0)
        f1 = metrics.get("macro_f1", 0.0)
        self._append(f"[training] finished  acc={acc:.4f}  macro_f1={f1:.4f}")
