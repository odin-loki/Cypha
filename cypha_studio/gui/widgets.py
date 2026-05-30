"""
cypha_studio.gui.training_widget
──────────────────────────────────
Live training monitor with loss/accuracy curves, per-class recall heatmap,
confidence histogram, and world-prior health indicator.
"""
from __future__ import annotations
from collections import deque
from functools import partial
from typing import List, Sequence, Tuple, Union

import numpy as np
from PySide6.QtCore    import Qt, QElapsedTimer, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QScrollArea,
    QPushButton, QComboBox, QSpinBox, QCheckBox, QGroupBox, QFormLayout,
    QLineEdit, QTextEdit, QDialogButtonBox, QDialog, QDoubleSpinBox,
)
from ..server.local_server import SignalBus

try:
    import pyqtgraph as pg
    pg.setConfigOption('background', '#1e1e1e')
    pg.setConfigOption('foreground', '#ccc')
    HAS_PG = True
except ImportError:
    HAS_PG = False


class TrainingWidget(QWidget):
    """Live training dashboard."""

    MAX_POINTS = 2000
    # Downsample only what we hand to pyqtgraph (series already capped by deques).
    PLOT_DISPLAY_MAX = 1400

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bus = SignalBus.instance()
        self._steps       : deque = deque(maxlen=self.MAX_POINTS)
        self._losses      : deque = deque(maxlen=self.MAX_POINTS)
        self._roll_accs   : deque = deque(maxlen=self.MAX_POINTS)
        self._val_steps   : deque = deque(maxlen=self.MAX_POINTS)
        self._val_accs    : deque = deque(maxlen=self.MAX_POINTS)
        self._val_f1s     : deque = deque(maxlen=self.MAX_POINTS)
        self._per_class   : dict  = {}
        self._n_correct   : deque = deque(maxlen=50)
        self._world_xs    : deque = deque(maxlen=self.MAX_POINTS)
        self._world_ys    : deque = deque(maxlen=self.MAX_POINTS)
        self._world_eval_i = 0

        self._plot_clock = QElapsedTimer()
        self._plot_clock.start()
        self._min_plot_ms = 80  # cap pyqtgraph line refresh (~12.5 Hz)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Summary bar
        self._summary = QLabel("Waiting for training…")
        self._summary.setStyleSheet("color: #aaa; padding: 2px;")
        layout.addWidget(self._summary)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # ── Tab 1: Loss + rolling accuracy ──────────────────────────────────
        if HAS_PG:
            loss_widget = QWidget()
            lw_layout   = QVBoxLayout(loss_widget)

            self._loss_plot = pg.PlotWidget(title="Training Loss")
            self._loss_plot.setLabel('left', 'Loss')
            self._loss_plot.setLabel('bottom', 'Step')
            self._loss_curve = self._loss_plot.plot(pen=pg.mkPen('#e07020', width=1))
            lw_layout.addWidget(self._loss_plot)

            self._acc_plot = pg.PlotWidget(title="Rolling Accuracy (window=50)")
            self._acc_plot.setLabel('left', 'Accuracy')
            self._acc_plot.setLabel('bottom', 'Step')
            self._acc_plot.setYRange(0, 1)
            self._acc_curve  = self._acc_plot.plot(pen=pg.mkPen('#20c060', width=1.5))
            self._val_curve  = self._acc_plot.plot(pen=pg.mkPen('#60a0ff', width=2),
                                                    symbol='o', symbolSize=5)
            lw_layout.addWidget(self._acc_plot)
            tabs.addTab(loss_widget, "Loss / Accuracy")
        else:
            no_pg = QLabel("Install pyqtgraph for live plots:\n  pip install pyqtgraph")
            no_pg.setAlignment(Qt.AlignCenter)
            no_pg.setStyleSheet("color: #888;")
            tabs.addTab(no_pg, "Loss / Accuracy")

        # ── Tab 2: Per-class recall table ────────────────────────────────────
        self._class_table = QTableWidget(0, 4)
        self._class_table.setHorizontalHeaderLabels(
            ["Class", "Precision", "Recall", "F1"]
        )
        self._class_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._class_table.setAlternatingRowColors(True)
        tabs.addTab(self._class_table, "Per-Class Metrics")

        # ── Tab 3: World prior health ────────────────────────────────────────
        health_w = QWidget()
        hw_layout = QVBoxLayout(health_w)
        self._world_lbl = QLabel(
            "World prior health: awaiting training data…"
        )
        self._world_lbl.setWordWrap(True)
        self._world_lbl.setStyleSheet("color: #aaa; padding: 6px;")
        hw_layout.addWidget(self._world_lbl)
        if HAS_PG:
            self._world_plot = pg.PlotWidget(title="World Prior ‖μ‖ over steps")
            self._world_plot.setLabel('left', '‖μ‖')
            self._world_plot.setLabel('bottom', 'Eval step')
            self._world_curve = self._world_plot.plot(pen=pg.mkPen('#c040c0', width=1.5))
            hw_layout.addWidget(self._world_plot)
        hw_layout.addStretch()
        tabs.addTab(health_w, "World Prior")

    def _connect_signals(self):
        bus = self._bus
        bus.training_step.connect(self._on_step)
        bus.training_evaluated.connect(self._on_evaluated)
        bus.training_finished.connect(self._on_finished)
        bus.training_stopped.connect(lambda: self._summary.setText("Training stopped."))

    @staticmethod
    def _compress_xy(xs: Sequence[float], ys: Sequence[float],
                     cap: int) -> Tuple[List[float], List[float]]:
        n = len(xs)
        if n <= cap:
            return list(xs), list(ys)
        idx = np.unique(
            np.linspace(0, n - 1, min(cap, n), dtype=np.int64)
        )
        return [float(xs[i]) for i in idx], [float(ys[i]) for i in idx]

    def _plot_train_curves(self) -> None:
        if not HAS_PG or not self._steps:
            return
        sx, sy = self._compress_xy(
            self._steps, self._losses, self.PLOT_DISPLAY_MAX
        )
        self._loss_curve.setData(sx, sy)
        ax, ay = self._compress_xy(
            self._steps, self._roll_accs, self.PLOT_DISPLAY_MAX
        )
        self._acc_curve.setData(ax, ay)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_step(self, step: int, loss: float, label: str, correct: bool):
        self._steps.append(step)
        self._losses.append(loss)
        self._n_correct.append(int(correct))
        roll = sum(self._n_correct) / max(len(self._n_correct), 1)
        self._roll_accs.append(roll)
        self._summary.setText(
            f"Step {step}  |  loss {loss:.4f}  |  rolling acc {roll:.3f}"
        )
        if HAS_PG and self._plot_clock.elapsed() >= self._min_plot_ms:
            self._plot_clock.restart()
            self._plot_train_curves()

    def _on_evaluated(self, metrics: dict):
        step = metrics.get('step', len(self._val_steps))
        acc  = metrics.get('accuracy', 0.0)
        f1   = metrics.get('macro_f1', 0.0)
        wn   = metrics.get('world_mu_norm', 0.0)

        self._val_steps.append(step)
        self._val_accs.append(acc)
        self._val_f1s.append(f1)

        if HAS_PG:
            vx, vy = self._compress_xy(
                self._val_steps, self._val_accs, self.PLOT_DISPLAY_MAX
            )
            self._val_curve.setData(vx, vy)
            if wn > 0:
                self._world_eval_i += 1
                self._world_xs.append(float(self._world_eval_i))
                self._world_ys.append(wn)
                wx, wy = self._compress_xy(
                    self._world_xs, self._world_ys, self.PLOT_DISPLAY_MAX
                )
                self._world_curve.setData(wx, wy)
                self._world_lbl.setText(
                    f"World ‖μ‖ = {wn:.4f}   Classes: {metrics.get('n_classes',0)}"
                )

        # Per-class table
        per_class = metrics.get('per_class', {})
        if per_class:
            self._class_table.setRowCount(len(per_class))
            for row, (lbl, vals) in enumerate(sorted(per_class.items())):
                self._class_table.setItem(row, 0, QTableWidgetItem(str(lbl)))
                self._class_table.setItem(row, 1,
                    QTableWidgetItem(f"{vals.get('precision',0):.4f}"))
                self._class_table.setItem(row, 2,
                    QTableWidgetItem(f"{vals.get('recall',0):.4f}"))
                self._class_table.setItem(row, 3,
                    QTableWidgetItem(f"{vals.get('f1',0):.4f}"))

    def _on_finished(self, metrics: dict):
        acc = metrics.get('accuracy', 0.0)
        r2  = metrics.get('r2_score', 0.0)
        self._summary.setText(
            f"Training complete  |  val acc={acc:.4f}  f1={metrics.get('macro_f1',0):.4f}  R²={r2:.4f}"
        )
        if HAS_PG and self._steps:
            self._plot_train_curves()


# ─────────────────────────────────────────────────────────────────────────────
# ModelWidget — class gallery, world prior panel, field state
# ─────────────────────────────────────────────────────────────────────────────

"""
cypha_studio.gui.model_widget
"""

class ModelWidget(QWidget):
    """Model inspector: class gallery, world prior, field state."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state
        self._bus   = SignalBus.instance()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._title = QLabel("No model loaded")
        self._title.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self._title)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # ── Class gallery ────────────────────────────────────────────────────
        self._class_table = QTableWidget(0, 3)
        self._class_table.setHorizontalHeaderLabels(["Class", "n_obs", "‖Δμ‖"])
        self._class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._class_table.setAlternatingRowColors(True)
        tabs.addTab(self._class_table, "Classes")

        # ── Model card summary ───────────────────────────────────────────────
        self._card_lbl = QLabel("Load a model to see its card.")
        self._card_lbl.setWordWrap(True)
        self._card_lbl.setStyleSheet("color: #aaa; padding: 8px;")
        self._card_lbl.setAlignment(Qt.AlignTop)
        tabs.addTab(self._card_lbl, "Model Card")

        # ── Capacity gauge ───────────────────────────────────────────────────
        cap_w = QWidget()
        cap_l = QVBoxLayout(cap_w)
        self._cap_lbl = QLabel("Capacity: —")
        self._cap_lbl.setStyleSheet("color: #aaa; padding: 4px;")
        self._cap_bar = QProgressBar()
        self._cap_bar.setRange(0, 100)
        cap_l.addWidget(self._cap_lbl)
        cap_l.addWidget(self._cap_bar)
        cap_l.addStretch()
        tabs.addTab(cap_w, "Capacity")

    def _connect_signals(self):
        self._bus.model_loaded.connect(self._on_model_loaded)
        self._bus.model_updated.connect(self._refresh)
        self._bus.training_evaluated.connect(lambda _: self._refresh())

    def _on_model_loaded(self, card):
        self._title.setText(f"{card.name} v{card.version}")
        card_text = (
            f"<b>Name:</b> {card.name}<br>"
            f"<b>Version:</b> {card.version}<br>"
            f"<b>Task:</b> {card.task}<br>"
            f"<b>Model:</b> {card.model_type} / {card.encoder_type}<br>"
            f"<b>Dataset:</b> {card.dataset_name}<br>"
            f"<b>Val accuracy:</b> {card.val_accuracy:.4f}<br>"
            f"<b>Val F1:</b> {card.val_f1:.4f}<br>"
            f"<b>Train steps:</b> {card.train_steps}<br>"
            f"<b>Stage:</b> {card.stage}<br>"
            f"<b>GH protected:</b> {card.gh_protected}<br>"
            f"<b>Description:</b> {card.description or '—'}<br>"
            f"<b>Intended use:</b> {card.intended_use or '—'}<br>"
            f"<b>Limitations:</b> {card.known_limitations or '—'}"
        )
        self._card_lbl.setText(card_text)
        self._refresh()

    def _refresh(self, *_):
        engine = self._state.engine
        if engine is None:
            return
        model = engine.model
        try:
            with model.memory._lock:
                classes = dict(model.memory._classes)
            self._class_table.setRowCount(len(classes))
            for row, (lbl, cd) in enumerate(sorted(classes.items())):
                self._class_table.setItem(row, 0, QTableWidgetItem(str(lbl)))
                self._class_table.setItem(row, 1,
                    QTableWidgetItem(f"{cd.n_obs:.0f}"))
                self._class_table.setItem(row, 2,
                    QTableWidgetItem(f"{np.linalg.norm(cd.delta_mu):.4f}"))
            # Capacity gauge
            n = len(classes)
            cap = min(100, int(n / 128 * 100))
            self._cap_bar.setValue(cap)
            self._cap_lbl.setText(
                f"Classes: {n} / 128  |  Buffer: {cap}% full"
            )
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# ConfidenceWidget — live confidence bars + OOD meter
# ─────────────────────────────────────────────────────────────────────────────

"""
cypha_studio.gui.confidence_widget
"""

class ConfidenceWidget(QWidget):
    """Live per-class confidence bars and OOD anomaly gauge."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bus = SignalBus.instance()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._title = QLabel("Confidence")
        self._title.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(self._title)

        # OOD meter
        ood_frame = QFrame()
        ood_frame.setFrameShape(QFrame.StyledPanel)
        ood_layout = QVBoxLayout(ood_frame)
        ood_layout.setContentsMargins(6, 4, 6, 4)
        self._ood_lbl = QLabel("OOD Score: 0.00")
        self._ood_lbl.setStyleSheet("font-size: 12px; font-weight: bold;")
        self._ood_bar = QProgressBar()
        self._ood_bar.setRange(0, 100)
        self._ood_bar.setTextVisible(False)
        self._ood_bar.setFixedHeight(14)
        ood_layout.addWidget(self._ood_lbl)
        ood_layout.addWidget(self._ood_bar)
        layout.addWidget(ood_frame)

        # Per-class confidence bars
        self._class_bars: dict = {}
        self._bars_container = QWidget()
        self._bars_layout    = QVBoxLayout(self._bars_container)
        self._bars_layout.setContentsMargins(0, 0, 0, 0)
        self._bars_layout.setSpacing(3)
        layout.addWidget(self._bars_container)
        layout.addStretch()

        # History summary
        self._history_lbl = QLabel(
            "Awaiting predictions…"
        )
        self._history_lbl.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._history_lbl)

    def _connect_signals(self):
        self._bus.prediction_made.connect(self._on_prediction)
        self._bus.anomaly_detected.connect(self._on_anomaly)

    def _on_prediction(self, pred):
        # Update OOD meter
        ood_pct = min(100, int(pred.anomaly_score * 20))
        self._ood_bar.setValue(ood_pct)
        ood_colour = "#e04040" if pred.is_ood else "#40a040"
        self._ood_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {ood_colour}; }}")
        self._ood_lbl.setText(f"OOD Score: {pred.anomaly_score:.2f}"
                               + (" ⚠ OOD" if pred.is_ood else ""))

        # Per-class bars
        scores = pred.all_scores
        if not scores:
            scores = {pred.label: pred.confidence}
        sorted_scores = sorted(scores.items(), key=lambda kv: -kv[1])

        # Create or update bars
        for lbl, score in sorted_scores[:8]:
            conf_pct = max(0, min(100, int(abs(score) * 5)))
            if lbl not in self._class_bars:
                row_w  = QWidget()
                row_l  = QHBoxLayout(row_w)
                row_l.setContentsMargins(0, 0, 0, 0)
                lbl_w  = QLabel(str(lbl))
                lbl_w.setFixedWidth(100)
                lbl_w.setStyleSheet("color: #ccc; font-size: 11px;")
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setTextVisible(False)
                bar.setFixedHeight(12)
                row_l.addWidget(lbl_w)
                row_l.addWidget(bar)
                self._bars_layout.addWidget(row_w)
                self._class_bars[lbl] = (bar, lbl_w)
            bar, lbl_w = self._class_bars[lbl]
            bar.setValue(conf_pct)
            colour = "#4090e0" if lbl == pred.label else "#406090"
            bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: {colour}; }}"
            )

        self._history_lbl.setText(
            f"Last: {pred.label}  conf: {pred.confidence:.3f}  "
            f"R_eff: {pred.r_eff:.2f}"
        )

    def _on_anomaly(self, score: float):
        self._ood_lbl.setText(f"OOD Score: {score:.2f} ⚠ ANOMALY DETECTED")
        self._ood_bar.setStyleSheet(
            "QProgressBar::chunk { background: #e04040; }"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DatasetWidget — browser with stats and split controls
# ─────────────────────────────────────────────────────────────────────────────

"""
cypha_studio.gui.dataset_widget
"""

class DatasetWidget(QWidget):
    """Dataset browser: load, preview, view stats, configure split."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state   = state
        self._bus     = SignalBus.instance()
        self._dataset = None
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 4, 4, 4)

        self._ds_lbl = QLabel("No dataset loaded")
        self._ds_lbl.setStyleSheet(
            "font-weight: bold; color: #e8e8e8; font-size: 12px; padding: 2px 0;"
        )
        layout.addWidget(self._ds_lbl)

        your_grp = QGroupBox("Your data")
        your_grp.setToolTip("Load real datasets from disk (CSV, NPZ, NPY).")
        your_lay = QVBoxLayout(your_grp)
        hint_you = QLabel(
            "Use <b>Browse for file…</b> for your own CSV, NPZ, or NPY — this is the path "
            "you will use for production-style work."
        )
        hint_you.setWordWrap(True)
        hint_you.setTextFormat(Qt.TextFormat.RichText)
        hint_you.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        your_lay.addWidget(hint_you)
        load_btn = QPushButton("Browse for file…")
        load_btn.setToolTip("Open a dataset from your computer.")
        load_btn.clicked.connect(self._on_load)
        your_lay.addWidget(load_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(your_grp)

        demo_grp = QGroupBox("Demo datasets (sklearn)")
        demo_grp.setToolTip(
            "Tiny built-in examples only — good for trying the UI, not representative of real data."
        )
        demo_lay = QVBoxLayout(demo_grp)
        hint_demo = QLabel(
            "Small toy sets bundled for exploration. For experiments and deployment, "
            "use <b>Your data</b> above."
        )
        hint_demo.setWordWrap(True)
        hint_demo.setTextFormat(Qt.TextFormat.RichText)
        hint_demo.setStyleSheet("color: #a8a8a8; font-size: 11px;")
        demo_lay.addWidget(hint_demo)
        demo_row = QHBoxLayout()
        demo_row.setSpacing(6)
        for name in ("iris", "wine", "breast_cancer", "digits", "diabetes"):
            btn = QPushButton(name)
            btn.setFixedHeight(26)
            btn.setAutoDefault(False)
            btn.setDefault(False)
            btn.setStyleSheet(
                "QPushButton { font-size: 11px; padding: 2px 8px; color: #d4d4d4; "
                "background: #353535; border: 1px solid #555; }"
                "QPushButton:hover { background: #404040; }"
            )
            # Use partial — PySide6's clicked(bool) slot arity can break naive lambdas.
            btn.clicked.connect(partial(self._load_sklearn, name))
            demo_row.addWidget(btn)
        demo_row.addStretch()
        demo_lay.addLayout(demo_row)
        layout.addWidget(demo_grp)

        # Stats
        self._stats_lbl = QLabel("")
        self._stats_lbl.setWordWrap(True)
        self._stats_lbl.setStyleSheet("color: #aaa; font-size: 11px; padding: 4px;")
        layout.addWidget(self._stats_lbl)

        # Class distribution table
        self._dist_table = QTableWidget(0, 2)
        self._dist_table.setHorizontalHeaderLabels(["Label", "Count"])
        self._dist_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._dist_table.setMaximumHeight(180)
        layout.addWidget(self._dist_table)

        split_grp = QGroupBox("Train / validation / test split")
        split_grp.setToolTip(
            "Fractions are percentages; test is the remainder. "
            "Stratify keeps class proportions per split (classification only)."
        )
        split_form = QFormLayout(split_grp)
        self._sp_train = QSpinBox()
        self._sp_train.setRange(5, 90)
        self._sp_train.setSuffix(" %")
        self._sp_train.setToolTip("Training portion of the dataset (percent).")
        self._sp_val = QSpinBox()
        self._sp_val.setRange(5, 90)
        self._sp_val.setSuffix(" %")
        self._sp_val.setToolTip("Validation portion (percent).")
        self._sp_test_lbl = QLabel("—")
        self._sp_shuffle = QCheckBox("Shuffle before split")
        self._sp_shuffle.setToolTip("Randomize row order before cutting indices.")
        self._sp_strat = QCheckBox("Stratify (classification)")
        self._sp_strat.setToolTip("Per-class splits; ignored automatically for regression.")
        self._sp_seed = QSpinBox()
        self._sp_seed.setRange(0, 2_000_000_000)
        self._sp_seed.setToolTip("RNG seed for shuffling and stratified index draws.")
        split_form.addRow("Train:", self._sp_train)
        split_form.addRow("Validation:", self._sp_val)
        split_form.addRow("Test (computed):", self._sp_test_lbl)
        split_form.addRow("", self._sp_shuffle)
        split_form.addRow("", self._sp_strat)
        split_form.addRow("Random seed:", self._sp_seed)
        layout.addWidget(split_grp)

        self._sp_train.valueChanged.connect(self._on_split_changed)
        self._sp_val.valueChanged.connect(self._on_split_changed)
        self._sp_shuffle.toggled.connect(self._on_split_toggle)
        self._sp_strat.toggled.connect(self._on_split_toggle)
        self._sp_seed.valueChanged.connect(self._on_split_changed)

        self._load_split_widgets_from_state()
        self._update_test_pct_label()

        layout.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll)

    def _raise_dataset_dock(self) -> None:
        win = self.window()
        if win is None:
            return
        dock = getattr(win, "_dataset_dock", None)
        if dock is not None:
            dock.show()
            dock.raise_()

    def _load_split_widgets_from_state(self):
        cfg = self._state.split_config
        self._sp_train.blockSignals(True)
        self._sp_val.blockSignals(True)
        self._sp_shuffle.blockSignals(True)
        self._sp_strat.blockSignals(True)
        self._sp_seed.blockSignals(True)
        self._sp_train.setValue(int(round(cfg.train_frac * 100)))
        self._sp_val.setValue(int(round(cfg.val_frac * 100)))
        self._sp_shuffle.setChecked(cfg.shuffle)
        self._sp_strat.setChecked(cfg.stratify)
        self._sp_seed.setValue(int(cfg.seed))
        self._sp_train.blockSignals(False)
        self._sp_val.blockSignals(False)
        self._sp_shuffle.blockSignals(False)
        self._sp_strat.blockSignals(False)
        self._sp_seed.blockSignals(False)

    def _update_test_pct_label(self):
        te = 100 - self._sp_train.value() - self._sp_val.value()
        self._sp_test_lbl.setText(f"{te} %")

    def _on_split_changed(self, *args):
        self._update_test_pct_label()
        self._apply_split_if_ready()

    def _on_split_toggle(self, checked: bool = False):
        self._apply_split_if_ready()

    def _apply_split_if_ready(self):
        if self._dataset is None:
            return
        tr = self._sp_train.value() / 100.0
        va = self._sp_val.value() / 100.0
        te = 1.0 - tr - va
        if tr <= 0 or va <= 0 or te < 0.05:
            self._bus.emit_status(
                "Split not applied: need positive train/val and at least 5% test."
            )
            return
        from ..core.dataset import SplitConfig

        st = self._dataset.task == "classification"
        cfg = SplitConfig(
            train_frac=tr,
            val_frac=va,
            test_frac=te,
            shuffle=self._sp_shuffle.isChecked(),
            stratify=self._sp_strat.isChecked() and st,
            seed=int(self._sp_seed.value()),
        )
        self._state.split_config = cfg
        train_ds, val_ds, test_ds = self._dataset.split(cfg)
        self._state._train_ds = train_ds
        self._state._val_ds = val_ds
        self._state._test_ds = test_ds
        self._bus.emit_status(
            f"Split updated: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
        )

    def _on_load(self):
        from PySide6.QtWidgets import QFileDialog

        from .path_history import dataset_dialog_start_dir_preferred

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Dataset",
            dataset_dialog_start_dir_preferred(getattr(self._state, "preferences", None)),
            "CSV files (*.csv);;NumPy (*.npy *.npz);;All (*)",
        )
        if path:
            self.load_file(path)

    def load_file(self, path: str):
        try:
            from ..core.dataset import CSVDataset, NumpyDataset

            lp = path.lower()
            if lp.endswith('.csv'):
                cr = self._state.preferences.effective_csv_chunk_rows()
                ds = (
                    CSVDataset.from_file(path, read_chunk_rows=cr)
                    if cr > 0
                    else CSVDataset.from_file(path)
                )
            else:
                ds = NumpyDataset.from_npz(path)
            self._set_dataset(ds)
            from .path_history import record_dataset_opened

            record_dataset_opened(path)
            self._bus.emit_dataset_opened(path)
        except Exception as e:
            self._bus.emit_error(f"Dataset load failed: {e}")

    def _load_sklearn(self, name: str, *_args) -> None:
        """Load a bundled sklearn toy set. Accepts extra args from ``clicked`` signal."""
        try:
            from ..core.dataset import SklearnDataset

            task = "regression" if name == "diabetes" else "classification"
            ds = SklearnDataset.load(name, task=task)
            self._set_dataset(ds)
            self._raise_dataset_dock()
        except ImportError as e:
            self._bus.emit_error(
                f"Demo datasets require scikit-learn. Install with:\n"
                f"  pip install scikit-learn\n\n{e}"
            )
        except Exception as e:
            self._bus.emit_error(f"Failed to load {name}: {e}")

    def _set_dataset(self, ds):
        self._dataset = ds
        self._ds_lbl.setText(f"{ds.name}  ({ds.n_samples} × {ds.n_features})")

        # Compute and show stats
        stats = ds.stats()
        self._stats_lbl.setText(
            f"Samples: {stats.n_samples}  |  Features: {stats.n_features}  |  "
            f"Classes: {stats.n_classes}  |  "
            f"Balance: {stats.class_balance:.2f}  |  "
            f"Missing: {stats.missing_values}"
        )

        # Class distribution (clear stale rows for regression / empty)
        if stats.class_counts:
            self._dist_table.setRowCount(len(stats.class_counts))
            for row, (lbl, cnt) in enumerate(
                sorted(stats.class_counts.items(), key=lambda kv: -kv[1])
            ):
                self._dist_table.setItem(row, 0, QTableWidgetItem(str(lbl)))
                self._dist_table.setItem(row, 1, QTableWidgetItem(str(cnt)))
        else:
            self._dist_table.setRowCount(0)

        self._sp_strat.setEnabled(ds.task == "classification")
        if ds.task != "classification":
            self._sp_strat.setChecked(False)

        train_ds, val_ds, test_ds = ds.split(self._state.split_config)
        self._state._train_ds = train_ds
        self._state._val_ds   = val_ds
        self._state._test_ds  = test_ds
        self._bus.emit_status(
            f"Dataset ready: {ds.name}  train={len(train_ds)} "
            f"val={len(val_ds)} test={len(test_ds)}"
        )

    @property
    def dataset(self):
        return self._dataset


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentWidget — runs table, leaderboard, comparison
# ─────────────────────────────────────────────────────────────────────────────

"""
cypha_studio.gui.experiment_widget
"""

class ExperimentWidget(QWidget):
    """Experiment tracker: sortable runs table and registry comparison."""

    _RUN_PAGE = 50

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state
        self._bus   = SignalBus.instance()
        self._compare_mode = False
        self._run_offset = 0
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        hdr = QHBoxLayout()
        self._title = QLabel("Experiments")
        self._title.setStyleSheet("font-weight: bold;")
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(30)
        refresh_btn.clicked.connect(self._refresh)
        hdr.addWidget(self._title)
        hdr.addStretch()
        hdr.addWidget(refresh_btn)
        layout.addLayout(hdr)

        self._runs_table = QTableWidget(0, 6)
        self._runs_table.setHorizontalHeaderLabels(
            ["Name", "Status", "Accuracy", "F1", "Steps", "Duration"]
        )
        self._runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._runs_table.setAlternatingRowColors(True)
        self._runs_table.setSortingEnabled(True)
        layout.addWidget(self._runs_table)

        self._load_more = QPushButton("Load more runs…")
        self._load_more.clicked.connect(self._on_load_more_runs)
        layout.addWidget(self._load_more)

    def _connect_signals(self):
        self._bus.training_finished.connect(lambda _: self._refresh())
        self._bus.registry_changed.connect(self._refresh)

    def _fill_run_rows(self, start_row: int, runs) -> None:
        for i, run in enumerate(runs):
            row = start_row + i
            self._runs_table.setItem(row, 0, QTableWidgetItem(run.name))
            self._runs_table.setItem(row, 1, QTableWidgetItem(run.status))
            self._runs_table.setItem(row, 2,
                QTableWidgetItem(f"{run.accuracy:.4f}"))
            self._runs_table.setItem(row, 3,
                QTableWidgetItem(f"{run.macro_f1:.4f}"))
            self._runs_table.setItem(row, 4,
                QTableWidgetItem(str(run.n_steps)))
            self._runs_table.setItem(row, 5,
                QTableWidgetItem(f"{run.duration_s:.1f}s"))

    def _refresh(self, *_):
        self._compare_mode = False
        self._run_offset = 0
        self._load_more.show()
        self._title.setText("Experiments")
        self._runs_table.setSortingEnabled(False)
        self._runs_table.setColumnCount(6)
        self._runs_table.setHorizontalHeaderLabels(
            ["Name", "Status", "Accuracy", "F1", "Steps", "Duration"]
        )
        runs = self._state.db.list_runs(limit=self._RUN_PAGE, offset=0)
        self._runs_table.setRowCount(len(runs))
        self._fill_run_rows(0, runs)
        self._run_offset = len(runs)
        self._load_more.setEnabled(len(runs) >= self._RUN_PAGE)
        self._runs_table.setSortingEnabled(True)

    def _on_load_more_runs(self):
        if self._compare_mode:
            return
        self._runs_table.setSortingEnabled(False)
        runs = self._state.db.list_runs(
            limit=self._RUN_PAGE, offset=self._run_offset,
        )
        if not runs:
            self._load_more.setEnabled(False)
            self._runs_table.setSortingEnabled(True)
            return
        n0 = self._runs_table.rowCount()
        self._runs_table.setRowCount(n0 + len(runs))
        self._fill_run_rows(n0, runs)
        self._run_offset += len(runs)
        self._load_more.setEnabled(len(runs) >= self._RUN_PAGE)
        self._runs_table.setSortingEnabled(True)

    def show_comparison(self, models: Sequence[Union[Tuple[str, str], object]]):
        test_ds = getattr(self._state, '_test_ds', None)
        pairs = []
        for m in models:
            if isinstance(m, tuple) and len(m) >= 2:
                pairs.append((str(m[0]), str(m[1])))
            else:
                pairs.append((m.name, m.version))
        rows = self._state.registry.compare(pairs, test_ds=test_ds)
        with_test = test_ds is not None and any(
            'test_accuracy' in r for r in rows
        )

        self._compare_mode = True
        self._load_more.hide()
        self._title.setText(
            "Registry compare"
            + (" (+ test set)" if with_test else "")
        )
        ncols = 12 if with_test else 10
        headers = [
            "Name", "Stage", "Task", "Type",
            "Val Acc", "Val F1", "Val R²",
            "Train N", "Steps", "Version",
        ]
        if with_test:
            headers.extend(["Test Acc", "Test F1"])

        self._runs_table.setSortingEnabled(False)
        self._runs_table.setColumnCount(ncols)
        self._runs_table.setHorizontalHeaderLabels(headers)
        self._runs_table.setRowCount(len(rows))

        for row, r in enumerate(rows):
            if r.get('error'):
                self._runs_table.setItem(row, 0, QTableWidgetItem(r.get('name', '')))
                self._runs_table.setItem(row, 1, QTableWidgetItem(str(r.get('error', ''))))
                continue
            self._runs_table.setItem(row, 0, QTableWidgetItem(r.get('name', '')))
            self._runs_table.setItem(row, 1, QTableWidgetItem(str(r.get('stage', ''))))
            self._runs_table.setItem(row, 2, QTableWidgetItem(str(r.get('task', ''))))
            self._runs_table.setItem(row, 3, QTableWidgetItem(str(r.get('model_type', ''))))
            self._runs_table.setItem(row, 4,
                QTableWidgetItem(f"{r.get('val_accuracy', 0):.4f}"))
            self._runs_table.setItem(row, 5,
                QTableWidgetItem(f"{r.get('val_f1', 0):.4f}"))
            self._runs_table.setItem(row, 6,
                QTableWidgetItem(f"{r.get('val_r2', 0):.4f}"))
            self._runs_table.setItem(row, 7,
                QTableWidgetItem(str(r.get('n_train', 0))))
            self._runs_table.setItem(row, 8,
                QTableWidgetItem(str(r.get('train_steps', 0))))
            self._runs_table.setItem(row, 9, QTableWidgetItem(r.get('version', '')))
            if with_test:
                self._runs_table.setItem(row, 10,
                    QTableWidgetItem(f"{r.get('test_accuracy', 0):.4f}"))
                self._runs_table.setItem(row, 11,
                    QTableWidgetItem(f"{r.get('test_f1', 0):.4f}"))

        self._runs_table.setSortingEnabled(True)


# ─────────────────────────────────────────────────────────────────────────────
# Dialogs
# ─────────────────────────────────────────────────────────────────────────────

"""
cypha_studio.gui.dialogs
"""

from PySide6.QtWidgets import (
    QDialog, QFormLayout, QDialogButtonBox, QLineEdit,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QGroupBox,
)


class NewExperimentDialog(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state
        self.setWindowTitle("New Experiment")
        self.resize(400, 300)
        layout = QFormLayout(self)

        self._name = QLineEdit("my-experiment")
        self._desc = QTextEdit()
        self._desc.setMaximumHeight(80)
        self._dataset = QLineEdit()
        self._task = QComboBox()
        self._task.addItems(['classification', 'regression'])

        layout.addRow("Name:", self._name)
        layout.addRow("Description:", self._desc)
        layout.addRow("Dataset:", self._dataset)
        layout.addRow("Task:", self._task)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def _on_ok(self):
        exp = self._state.db.create_experiment(
            name=self._name.text(),
            description=self._desc.toPlainText(),
            dataset_name=self._dataset.text(),
            task=self._task.currentText(),
        )
        self._state._active_exp_id = exp.experiment_id
        SignalBus.instance().emit_status(
            f"Experiment created: {exp.name}"
        )
        self.accept()


class LoadModelDialog(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state
        self.setWindowTitle("Load Model")
        self.resize(350, 200)
        layout = QFormLayout(self)

        self._name_combo = QComboBox()
        self._ver_combo  = QComboBox()
        for name in state.registry.list_model_names():
            self._name_combo.addItem(name)
        self._name_combo.currentTextChanged.connect(self._on_name_changed)
        self._on_name_changed(self._name_combo.currentText())

        layout.addRow("Model:", self._name_combo)
        layout.addRow("Version:", self._ver_combo)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def _on_name_changed(self, name: str):
        self._ver_combo.clear()
        if name:
            for v in self._state.registry.registered_versions(name):
                self._ver_combo.addItem(v)

    def _on_ok(self):
        name    = self._name_combo.currentText()
        version = self._ver_combo.currentText()
        if not name or not version:
            return
        try:
            from ..core.inference import InferenceEngine, InferenceSession

            model, pre, card = self._state.registry.load(name, version)
            p = self._state.preferences
            self._state.engine = InferenceEngine(
                model, pre, ood_threshold=p.inference_ood_threshold
            )
            self._state.session = InferenceSession(self._state.engine)
            self._state.session.set_gh_params(p.inference_chi, p.inference_psi)
            SignalBus.instance().emit_model_loaded(card)
            self.accept()
        except Exception as e:
            SignalBus.instance().emit_error(str(e))


class TrainConfigDialog(QDialog):
    """Configure all ``TrainerConfig`` fields (tabbed)."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state
        self.setWindowTitle("Training Configuration")
        self.resize(520, 640)
        from ..core.trainer import TrainerConfig

        self._cfg = getattr(state, "_train_config", TrainerConfig())

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # ── Tab: Model & encoder ──────────────────────────────────────────
        t1 = QWidget()
        f1 = QFormLayout(t1)
        self._model_type = QComboBox()
        self._model_type.addItems(["CyphaDIF", "RFFRegressor", "TwoStageDIF", "MKE"])
        self._model_type.setCurrentText(self._cfg.model_type)
        self._model_type.setToolTip("CyphaDIF / RFF / two-stage / mixture-of-experts routing.")
        self._encoder = QComboBox()
        self._encoder.addItems(["VectorEncoder", "RFFEncoder"])
        self._encoder.setCurrentText(self._cfg.encoder_type)
        self._encoder.setToolTip("Dense encoder vs random Fourier features.")
        self._feat_dim = QSpinBox()
        self._feat_dim.setRange(16, 2048)
        self._feat_dim.setValue(self._cfg.feat_dim)
        self._field_dim = QSpinBox()
        self._field_dim.setRange(16, 2048)
        self._field_dim.setValue(self._cfg.field_dim)
        self._rff_D = QSpinBox()
        self._rff_D.setRange(32, 4096)
        self._rff_D.setValue(self._cfg.rff_D)
        self._rff_D.setToolTip("RFF projection dimension (RFFEncoder).")
        self._rff_gamma = QDoubleSpinBox()
        self._rff_gamma.setRange(0.0001, 100.0)
        self._rff_gamma.setDecimals(4)
        self._rff_gamma.setValue(self._cfg.rff_gamma)
        self._rff_gamma.setToolTip("RFF kernel bandwidth scale.")
        self._auto_gamma = QCheckBox("Auto γ (CV)")
        self._auto_gamma.setChecked(self._cfg.auto_gamma_cv)
        self._auto_ard = QCheckBox("Auto ARD")
        self._auto_ard.setChecked(self._cfg.auto_ard)
        self._n_experts = QSpinBox()
        self._n_experts.setRange(2, 64)
        self._n_experts.setValue(self._cfg.n_experts)
        self._n_experts.setToolTip("Expert count for MKE routing.")
        f1.addRow("Model type:", self._model_type)
        f1.addRow("Encoder:", self._encoder)
        f1.addRow("Feat dim:", self._feat_dim)
        f1.addRow("Field dim:", self._field_dim)
        f1.addRow("RFF D:", self._rff_D)
        f1.addRow("RFF γ:", self._rff_gamma)
        f1.addRow("", self._auto_gamma)
        f1.addRow("", self._auto_ard)
        f1.addRow("N experts (MKE):", self._n_experts)
        tabs.addTab(t1, "Model & encoder")

        # ── Tab: Optimization ───────────────────────────────────────────────
        t2 = QWidget()
        f2 = QFormLayout(t2)
        self._world_lr = QDoubleSpinBox()
        self._world_lr.setRange(0.0001, 1.0)
        self._world_lr.setDecimals(4)
        self._world_lr.setValue(self._cfg.world_lr)
        self._delta_lr = QDoubleSpinBox()
        self._delta_lr.setRange(0.001, 1.0)
        self._delta_lr.setDecimals(4)
        self._delta_lr.setValue(self._cfg.delta_lr)
        self._enc_lr = QDoubleSpinBox()
        self._enc_lr.setRange(0.0001, 1.0)
        self._enc_lr.setDecimals(4)
        self._enc_lr.setValue(self._cfg.enc_lr)
        self._mdl_lambda = QDoubleSpinBox()
        self._mdl_lambda.setRange(0.0, 1.0)
        self._mdl_lambda.setDecimals(4)
        self._mdl_lambda.setValue(self._cfg.mdl_lambda)
        self._forget = QDoubleSpinBox()
        self._forget.setRange(0.0, 1.0)
        self._forget.setDecimals(4)
        self._forget.setValue(self._cfg.forgetting_factor)
        self._forget.setToolTip("1.0 = no forgetting; lower down-weights old evidence.")
        self._temperature = QDoubleSpinBox()
        self._temperature.setRange(0.05, 8.0)
        self._temperature.setDecimals(3)
        self._temperature.setValue(self._cfg.temperature)
        self._context_win = QSpinBox()
        self._context_win.setRange(4, 512)
        self._context_win.setValue(self._cfg.context_win)
        self._gh_chk = QCheckBox("GH adversarial protection (training)")
        self._gh_chk.setChecked(self._cfg.gh_protect)
        self._gh_chk.setToolTip("Regularizer during training; separate from GH inference in chat.")
        f2.addRow("World LR:", self._world_lr)
        f2.addRow("Delta LR:", self._delta_lr)
        f2.addRow("Encoder LR:", self._enc_lr)
        f2.addRow("MDL λ:", self._mdl_lambda)
        f2.addRow("Forgetting factor:", self._forget)
        f2.addRow("Temperature:", self._temperature)
        f2.addRow("Context window:", self._context_win)
        f2.addRow("", self._gh_chk)
        tabs.addTab(t2, "Optimization")

        # ── Tab: Schedule & preprocessing ───────────────────────────────────
        t3 = QWidget()
        f3 = QFormLayout(t3)
        self._mode = QComboBox()
        self._mode.addItems(["online", "batch"])
        self._mode.setCurrentText(self._cfg.mode)
        self._mode.setToolTip("Sample streaming vs batched epochs.")
        self._n_epochs = QSpinBox()
        self._n_epochs.setRange(1, 500)
        self._n_epochs.setValue(self._cfg.n_epochs)
        self._batch_sz = QSpinBox()
        self._batch_sz.setRange(0, 65536)
        self._batch_sz.setMinimum(0)
        self._batch_sz.setSpecialValueText("default (None)")
        bs = self._cfg.batch_size
        self._batch_sz.setValue(0 if bs is None else int(bs))
        self._batch_sz.setToolTip("0 lets the trainer pick batching automatically.")
        self._eval_every = QSpinBox()
        self._eval_every.setRange(10, 100000)
        self._eval_every.setValue(self._cfg.eval_every_n)
        self._save_every = QSpinBox()
        self._save_every.setRange(10, 100000)
        self._save_every.setValue(self._cfg.save_every_n)
        self._early_stop = QCheckBox("Early stopping")
        self._early_stop.setChecked(self._cfg.early_stopping)
        self._patience = QSpinBox()
        self._patience.setRange(1, 100)
        self._patience.setValue(self._cfg.early_stopping_patience)
        self._pre_scale = QCheckBox("Scale features (preprocessor)")
        self._pre_scale.setChecked(self._cfg.preprocess_scale)
        self._pre_pca = QSpinBox()
        self._pre_pca.setRange(0, 512)
        self._pre_pca.setMinimum(0)
        self._pre_pca.setSpecialValueText("off")
        pc = self._cfg.preprocess_pca
        self._pre_pca.setValue(0 if pc is None else int(pc))
        self._pre_pca.setToolTip("0 = no PCA; else number of PCA components.")
        self._seed = QSpinBox()
        self._seed.setRange(0, 2_000_000_000)
        self._seed.setValue(self._cfg.seed)
        f3.addRow("Mode:", self._mode)
        f3.addRow("Epochs:", self._n_epochs)
        f3.addRow("Batch size:", self._batch_sz)
        f3.addRow("Eval every N steps:", self._eval_every)
        f3.addRow("Save checkpoint every N:", self._save_every)
        f3.addRow("", self._early_stop)
        f3.addRow("Early-stop patience:", self._patience)
        f3.addRow("", self._pre_scale)
        f3.addRow("PCA components:", self._pre_pca)
        f3.addRow("Random seed:", self._seed)
        tabs.addTab(t3, "Schedule & preprocessing")

        layout.addWidget(tabs)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_ok(self):
        from dataclasses import replace

        from ..core.trainer import TrainerConfig

        bs = self._batch_sz.value()
        bs_v = None if bs <= 0 else int(bs)
        pc = self._pre_pca.value()
        pc_v = None if pc <= 0 else int(pc)

        cfg = replace(
            self._cfg,
            model_type=self._model_type.currentText(),
            encoder_type=self._encoder.currentText(),
            feat_dim=self._feat_dim.value(),
            field_dim=self._field_dim.value(),
            rff_D=self._rff_D.value(),
            rff_gamma=self._rff_gamma.value(),
            auto_gamma_cv=self._auto_gamma.isChecked(),
            auto_ard=self._auto_ard.isChecked(),
            n_experts=self._n_experts.value(),
            world_lr=self._world_lr.value(),
            delta_lr=self._delta_lr.value(),
            enc_lr=self._enc_lr.value(),
            mdl_lambda=self._mdl_lambda.value(),
            forgetting_factor=self._forget.value(),
            temperature=self._temperature.value(),
            context_win=self._context_win.value(),
            gh_protect=self._gh_chk.isChecked(),
            mode=self._mode.currentText(),
            n_epochs=self._n_epochs.value(),
            batch_size=bs_v,
            eval_every_n=self._eval_every.value(),
            save_every_n=self._save_every.value(),
            early_stopping=self._early_stop.isChecked(),
            early_stopping_patience=self._patience.value(),
            preprocess_scale=self._pre_scale.isChecked(),
            preprocess_pca=pc_v,
            seed=self._seed.value(),
        )
        self._state._train_config = cfg
        SignalBus.instance().emit_status("Training config updated")
        self.accept()


class ExportModelDialog(QDialog):
    """Save current model to registry with a card."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state
        self.setWindowTitle("Export Model")
        self.resize(400, 360)
        layout = QFormLayout(self)

        self._name    = QLineEdit("my-model")
        self._version = QLineEdit(
            state.registry.next_version("my-model")
            if state.registry.exists("my-model") else "1.0.0"
        )
        self._desc    = QLineEdit()
        self._author  = QLineEdit()
        self._intended = QLineEdit()
        self._stage   = QComboBox()
        self._stage.addItems(['dev', 'staging', 'production'])

        layout.addRow("Name:",        self._name)
        layout.addRow("Version:",     self._version)
        layout.addRow("Description:", self._desc)
        layout.addRow("Author:",      self._author)
        layout.addRow("Intended use:",self._intended)
        layout.addRow("Stage:",       self._stage)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def _on_ok(self):
        eng  = self._state.engine
        if eng is None:
            return
        from ..core.registry import ModelCard
        card = ModelCard(
            name=self._name.text(),
            version=self._version.text(),
            description=self._desc.text(),
            author=self._author.text(),
            intended_use=self._intended.text(),
            stage=self._stage.currentText(),
            task=getattr(self._state, '_train_config',
                         type('_', (), {'task': 'classification'})()).model_type,
            model_type=type(eng.model).__name__,
        )
        # Fill in metrics from current card if available
        if self._state.current_card:
            card.val_accuracy = self._state.current_card.val_accuracy
            card.val_f1       = self._state.current_card.val_f1

        try:
            self._state.registry.register(eng.model, card, eng._preprocessor)
            SignalBus.instance().emit_status(
                f"Model saved: {card.name} v{card.version}"
            )
            SignalBus.instance().registry_changed.emit()
            self.accept()
        except Exception as e:
            SignalBus.instance().emit_error(str(e))
