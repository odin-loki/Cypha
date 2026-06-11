"""Simple confusion-matrix dialog for classification on a held-out set."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

Scalar = str | int | float


def _labels(y_true: Sequence[Scalar], y_pred: Sequence[Scalar]) -> list[str]:
    s = set(str(x) for x in y_true) | set(str(x) for x in y_pred)
    return sorted(s)


def confusion_counts(
    y_true: Iterable[Scalar], y_pred: Iterable[Scalar]
) -> tuple[list[str], np.ndarray]:
    yt = [str(x) for x in y_true]
    yp = [str(x) for x in y_pred]
    labels = _labels(yt, yp)
    idx = {lb: i for i, lb in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(yt, yp, strict=False):
        cm[idx[t], idx[p]] += 1
    return labels, cm


def show_confusion_dialog(parent, y_true: Sequence[Scalar], y_pred: Sequence[Scalar],
                          title: str = "Confusion matrix (test set)") -> None:
    labels, cm = confusion_counts(y_true, y_pred)
    n = len(labels)
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(min(80 + n * 72, 900), min(120 + n * 28, 700))
    lay = QVBoxLayout(dlg)
    tbl = QTableWidget(n, n)
    tbl.setHorizontalHeaderLabels([f"pred {lb}" for lb in labels])
    tbl.setVerticalHeaderLabels([f"true {lb}" for lb in labels])
    tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    tbl.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    for i in range(n):
        for j in range(n):
            tbl.setItem(i, j, QTableWidgetItem(str(cm[i, j])))
    lay.addWidget(tbl)
    box = QDialogButtonBox(QDialogButtonBox.Close)
    box.rejected.connect(dlg.reject)
    box.accepted.connect(dlg.accept)
    lay.addWidget(box)
    dlg.exec()
