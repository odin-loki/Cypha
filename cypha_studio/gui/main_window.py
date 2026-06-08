"""
cypha_studio.gui.main_window
─────────────────────────────
Top-level Qt6 application window.

Dock layout:
  Left   — Model browser (registry tree)
  Centre — Chat widget (primary interface)
  Right  — Confidence panel (live scores + OOD meter)
  Bottom — Training monitor (tabbed: loss, accuracy, per-class recall)

Toolbar: model selector, Train, Evaluate, Export buttons.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from PySide6.QtCore    import Qt, QThread, Signal, QTimer, QSettings
from PySide6.QtGui     import (
    QAction, QIcon, QFont, QPalette, QColor, QShortcut, QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QStatusBar, QToolBar, QMessageBox,
    QFileDialog, QSplitter, QTabWidget,
)

from ..server.local_server import SignalBus
from .chat_widget      import ChatWidget
from .training_widget  import TrainingWidget
from .model_widget     import ModelWidget
from .confidence_widget import ConfidenceWidget
from .dataset_widget   import DatasetWidget
from .experiment_widget import ExperimentWidget
from .log_widget       import LogDockWidget
from .dialogs          import (NewExperimentDialog, LoadModelDialog,
                                TrainConfigDialog, ExportModelDialog)
from .settings_dialog  import SettingsDialog
from .help_widget      import HelpWidget
from .studio_preferences import apply_preferences_to_inference_state


# ─────────────────────────────────────────────────────────────────────────────
# Application state (shared across widgets via bus)
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    """Global mutable state — one instance, passed to widgets that need it."""
    def __init__(self):
        from ..core.registry  import ModelRegistry
        from ..core.experiment import ExperimentDB
        from ..core.dataset import SplitConfig
        from .studio_preferences import load_studio_preferences

        self.preferences = load_studio_preferences()
        self.split_config = SplitConfig()
        self.registry  = ModelRegistry(self.preferences.effective_registry_root())
        self.db        = ExperimentDB()
        self.engine    = None   # InferenceEngine, set when model loads
        self.lm_engine = None   # LMEngine (CyphaLM), optional
        self.branch_a_router = None  # BranchARouter, lazy
        self.session   = None   # InferenceSession
        self.trainer   = None   # active Trainer
        self.train_thread = None
        self.current_card = None  # ModelCard


# ─────────────────────────────────────────────────────────────────────────────
# Training worker (QThread)
# ─────────────────────────────────────────────────────────────────────────────

class TrainingWorker(QThread):
    """Runs Trainer.fit() in a background thread, emits progress via bus."""

    def __init__(self, trainer, train_ds, val_ds, config, run_id, db):
        super().__init__()
        self._trainer  = trainer
        self._train_ds = train_ds
        self._val_ds   = val_ds
        self._config   = config
        self._run_id   = run_id
        self._db       = db

    def run(self):
        from ..core.trainer import TrainerCallback, EvalMetrics
        bus = SignalBus.instance()

        class BusCallback(TrainerCallback):
            def on_step(self, step, loss, label, correct):
                bus.emit_training_step(step, loss, label, correct)
            def on_evaluate(self, metrics):
                bus.emit_training_evaluated(metrics)
                if self._db and self._run_id:
                    self._db.log_metrics(self._run_id, metrics)
            def on_train_end(self, metrics):
                bus.emit_training_finished(metrics)
            def on_stop_requested(self):
                return self._trainer._stop_flag.is_set()

        cb = BusCallback()
        cb._db = self._db
        cb._run_id = self._run_id
        cb._trainer = self._trainer
        self._trainer.add_callback(cb)

        try:
            self._db.update_run(self._run_id, status='running')
            self._trainer.fit(self._train_ds, self._val_ds, self._config)
            final = self._trainer.evaluate(self._val_ds, self._config) \
                if self._val_ds else EvalMetrics()
            self._db.finish_run(self._run_id, final)
        except Exception as e:
            self._db.fail_run(self._run_id, str(e))
            bus.emit_error(f"Training failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.bus   = SignalBus.instance()

        self.setWindowTitle("CyphaStudio")
        self.resize(1400, 900)
        self._apply_dark_theme()

        self._build_toolbar()
        self._build_menu()
        self._build_dock_layout()
        self._build_status_bar()
        self._restore_saved_window_state()
        self._connect_signals()
        self._build_shortcuts()
        self._refresh_recent_dataset_menu()

    # ── Theme ────────────────────────────────────────────────────────────────

    def _apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window,          QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText,      QColor(220, 220, 220))
        palette.setColor(QPalette.Base,            QColor(22, 22, 22))
        palette.setColor(QPalette.AlternateBase,   QColor(38, 38, 38))
        palette.setColor(QPalette.Text,            QColor(220, 220, 220))
        palette.setColor(QPalette.Button,          QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
        palette.setColor(QPalette.Highlight,       QColor(0, 120, 200))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        palette.setColor(QPalette.Link,            QColor(80, 160, 255))
        app_inst = QApplication.instance()
        if app_inst is not None:
            app_inst.setPalette(palette)
            # Windows often leaves widgets on default (black) text; force readable contrast.
            app_inst.setStyleSheet(
                """
                QWidget { color: #e6e6e6; }
                QMainWindow { background: #1e1e1e; }
                QMenuBar {
                    background: #2d2d2d;
                    color: #ececec;
                    border-bottom: 1px solid #404040;
                }
                QMenuBar::item:selected { background: #3d5a80; }
                QMenu {
                    background: #2d2d2d;
                    color: #ececec;
                    border: 1px solid #505050;
                }
                QMenu::item:selected { background: #3d5a80; }
                QToolBar {
                    background: #252526;
                    border: none;
                    spacing: 4px;
                }
                QToolBar QLabel { color: #c8c8c8; }
                QDockWidget::title {
                    background: #333333;
                    color: #ececec;
                    padding: 5px;
                }
                QStatusBar { background: #252526; color: #c0c0c0; }
                QStatusBar QLabel { color: #c0c0c0; }
                QGroupBox {
                    color: #e6e6e6;
                    border: 1px solid #505050;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding: 8px 4px 4px 4px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: 10px;
                    padding: 0 6px;
                    color: #b8d4f0;
                }
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    background: #3c3c3c;
                    color: #f2f2f2;
                    border: 1px solid #555;
                    border-radius: 2px;
                    padding: 2px 6px;
                    min-height: 18px;
                }
                QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                    border: 1px solid #569cd6;
                }
                QComboBox QAbstractItemView {
                    background: #3c3c3c;
                    color: #f2f2f2;
                    selection-background-color: #3d5a80;
                }
                QCheckBox { color: #e6e6e6; }
                QCheckBox::indicator { width: 16px; height: 16px; }
                QPushButton {
                    background: #404040;
                    color: #f5f5f5;
                    border: 1px solid #5a5a5a;
                    border-radius: 3px;
                    padding: 4px 12px;
                    min-height: 20px;
                }
                QPushButton:hover { background: #4a4a4a; }
                QPushButton:pressed { background: #353535; }
                QPushButton:disabled { color: #888; background: #353535; }
                QTableWidget {
                    background: #2a2a2a;
                    alternate-background-color: #323232;
                    color: #ececec;
                    gridline-color: #454545;
                    border: 1px solid #505050;
                }
                QTableWidget::item:selected {
                    background: #3d5a80;
                    color: #ffffff;
                }
                QHeaderView::section {
                    background: #3a3a3a;
                    color: #e8e8e8;
                    padding: 4px;
                    border: 1px solid #505050;
                }
                QTabWidget::pane { border: 1px solid #505050; background: #252526; }
                QTabBar::tab {
                    background: #333;
                    color: #ccc;
                    padding: 6px 12px;
                    border: 1px solid #505050;
                }
                QTabBar::tab:selected { background: #3d5a80; color: #fff; }
                QTextBrowser, QTextEdit {
                    background: #1e1e1e;
                    color: #e6e6e6;
                    border: 1px solid #505050;
                }
                QScrollArea { border: none; }
                QMessageBox { background: #2d2d2d; }
                QMessageBox QLabel { color: #ececec; min-width: 240px; }
                QDialogButtonBox QPushButton { min-width: 72px; }
                """
            )

    # ── Window geometry (QSettings) ─────────────────────────────────────────

    def _studio_settings(self) -> QSettings:
        return QSettings("Cypha", "CyphaStudio")

    def _restore_saved_window_state(self) -> None:
        """Qt order: outer geometry, then toolbars/docks (`restoreState` may resize to fit docks)."""
        s = self._studio_settings()
        st, geo = s.value("windowState"), s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        elif st is None:
            self.resize(1400, 900)
        if st is not None:
            self.restoreState(st, 0)

    def closeEvent(self, event):
        s = self._studio_settings()
        s.setValue("windowState", self.saveState(0))
        s.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._shortcut_send_chat)
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self._shortcut_focus_chat)

    def _shortcut_send_chat(self):
        self.chat_widget._on_send()

    def _shortcut_focus_chat(self):
        self.chat_widget._input.setFocus(Qt.FocusReason.ShortcutFocusReason)

    # ── Toolbar ──────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        tb.setObjectName("toolbar_main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        # Model selector
        tb.addWidget(QLabel("  Model: "))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(220)
        self.model_combo.addItem("(no model loaded)")
        tb.addWidget(self.model_combo)
        tb.addSeparator()

        # Buttons
        self.btn_train    = QPushButton("▶  Train")
        self.btn_evaluate = QPushButton("📊  Evaluate")
        self.btn_export   = QPushButton("💾  Export")
        self.btn_stop     = QPushButton("⏹  Stop")
        self.btn_stop.setEnabled(False)

        for btn in (self.btn_train, self.btn_evaluate,
                    self.btn_export, self.btn_stop):
            btn.setFixedHeight(32)
            tb.addWidget(btn)

        tb.addSeparator()
        self.lbl_step = QLabel("  Steps: 0")
        tb.addWidget(self.lbl_step)

    # ── Menu ─────────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("File")
        file_menu.addAction("New Experiment…",  self._on_new_experiment)
        file_menu.addAction("Open Model…",      self._on_load_model)
        file_menu.addAction("Load CyphaLM…",     self._on_load_cyphalm)
        file_menu.addAction("Import Dataset…",  self._on_import_dataset)
        self._recent_menu = file_menu.addMenu("Recent Datasets")
        file_menu.addSeparator()
        file_menu.addAction("Settings…",        self._on_settings)
        file_menu.addSeparator()
        file_menu.addAction("Export test predictions (CSV)…", self._on_export_test_predictions)
        file_menu.addAction("Export Model…",    self._on_export_model)
        file_menu.addSeparator()
        file_menu.addAction("Quit",             self.close)

        # Train
        train_menu = mb.addMenu("Train")
        train_menu.addAction("Configure…",      self._on_train_config)
        train_menu.addAction("Start Training",  self._on_start_train)
        train_menu.addAction("Stop Training",   self._on_stop_train)
        train_menu.addSeparator()
        train_menu.addAction("Evaluate on Test Set", self._on_evaluate)

        # Model
        model_menu = mb.addMenu("Model")
        model_menu.addAction("Model Inspector",   self._toggle_model_panel)
        model_menu.addAction("Compare Models…",   self._on_compare_models)
        model_menu.addAction("Confusion matrix (test set)…", self._on_confusion_matrix)
        model_menu.addAction("Promote to Production…", self._on_promote)

        # View
        view_menu = mb.addMenu("View")
        view_menu.addAction("Chat",       lambda: self._toggle_dock('chat'))
        view_menu.addAction("Training",   lambda: self._toggle_dock('training'))
        view_menu.addAction("Confidence", lambda: self._toggle_dock('confidence'))
        view_menu.addAction("Dataset",    lambda: self._toggle_dock('dataset'))
        view_menu.addAction("Experiments",lambda: self._toggle_dock('experiment'))
        view_menu.addAction("Studio Log", lambda: self._toggle_dock('log'))
        view_menu.addAction("User Guide", lambda: self._toggle_dock('help'))
        view_menu.addSeparator()
        view_menu.addAction("Reset Layout", self._reset_layout)

        # Help
        help_menu = mb.addMenu("Help")
        help_menu.addAction("User Guide",       self._on_user_guide)
        help_menu.addAction("Keyboard Shortcuts…", self._on_keyboard_shortcuts)
        help_menu.addAction("About CyphaStudio", self._on_about)

    # ── Dock layout ──────────────────────────────────────────────────────────

    def _build_dock_layout(self):
        # ── Centre: Chat ────────────────────────────────────────────────────
        self.chat_widget = ChatWidget(self.state, self)
        self.setCentralWidget(self.chat_widget)

        # ── Left: Model browser ──────────────────────────────────────────────
        self.model_widget = ModelWidget(self.state, self)
        self._model_dock = self._make_dock(
            "Model Inspector", self.model_widget, Qt.LeftDockWidgetArea,
            "dock_model_inspector",
        )

        # ── Right: Confidence ─────────────────────────────────────────────
        self.conf_widget = ConfidenceWidget(self)
        self._conf_dock  = self._make_dock(
            "Confidence & OOD", self.conf_widget, Qt.RightDockWidgetArea,
            "dock_confidence_ood",
        )

        # ── Bottom tabs ───────────────────────────────────────────────────
        self.train_widget = TrainingWidget(self)
        self._train_dock  = self._make_dock(
            "Training Monitor", self.train_widget, Qt.BottomDockWidgetArea,
            "dock_training_monitor",
        )

        self.dataset_widget  = DatasetWidget(self.state, self)
        self._dataset_dock   = self._make_dock(
            "Dataset", self.dataset_widget, Qt.BottomDockWidgetArea,
            "dock_dataset",
        )
        self.tabifyDockWidget(self._train_dock, self._dataset_dock)

        self.exp_widget  = ExperimentWidget(self.state, self)
        self._exp_dock   = self._make_dock(
            "Experiments", self.exp_widget, Qt.BottomDockWidgetArea,
            "dock_experiments",
        )
        self.tabifyDockWidget(self._train_dock, self._exp_dock)

        self.log_widget = LogDockWidget(self)
        self._log_dock = self._make_dock(
            "Studio Log", self.log_widget, Qt.BottomDockWidgetArea,
            "dock_studio_log",
        )
        self.tabifyDockWidget(self._train_dock, self._log_dock)

        self.help_widget = HelpWidget(self)
        self._help_dock = self._make_dock(
            "User Guide", self.help_widget, Qt.BottomDockWidgetArea,
            "dock_user_guide",
        )
        self.tabifyDockWidget(self._train_dock, self._help_dock)

        self._train_dock.raise_()

        self._docks = {
            'chat'      : None,  # central widget, no dock
            'training'  : self._train_dock,
            'confidence': self._conf_dock,
            'dataset'   : self._dataset_dock,
            'experiment': self._exp_dock,
            'model'     : self._model_dock,
            'log'       : self._log_dock,
            'help'      : self._help_dock,
        }

    def _make_dock(self, title: str, widget: QWidget,
                   area: Qt.DockWidgetArea, object_name: str) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(area, dock)
        return dock

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self):
        sb = self.statusBar()
        self.lbl_status  = QLabel("Ready")
        self.lbl_model   = QLabel("No model")
        self.lbl_acc     = QLabel("acc: —")
        self.lbl_infer   = QLabel("")
        self.lbl_infer.setStyleSheet("color: #888;")
        sb.addWidget(self.lbl_status, 1)
        sb.addPermanentWidget(self.lbl_infer)
        sb.addPermanentWidget(QLabel("  |  "))
        sb.addPermanentWidget(self.lbl_model)
        sb.addPermanentWidget(QLabel("  |  "))
        sb.addPermanentWidget(self.lbl_acc)

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self):
        # Toolbar buttons
        self.btn_train.clicked.connect(self._on_start_train)
        self.btn_stop.clicked.connect(self._on_stop_train)
        self.btn_evaluate.clicked.connect(self._on_evaluate)
        self.btn_export.clicked.connect(self._on_export_model)
        self.model_combo.currentTextChanged.connect(self._on_model_selected)

        # Signal bus
        bus = self.bus
        bus.model_loaded.connect(self._on_model_loaded_signal)
        bus.training_step.connect(self._on_training_step)
        bus.training_evaluated.connect(self._on_training_evaluated)
        bus.training_finished.connect(self._on_training_finished)
        bus.status_message.connect(self.lbl_status.setText)
        bus.error_occurred.connect(self._on_error)

        # Refresh model combo on registry change
        bus.registry_changed.connect(self._refresh_model_combo)
        bus.dataset_opened.connect(lambda _: self._refresh_recent_dataset_menu())
        bus.preferences_changed.connect(self._on_preferences_changed)
        self._refresh_model_combo()
        self._update_inference_status_label()

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_model_loaded_signal(self, card):
        self.state.current_card = card
        self.lbl_model.setText(f"{card.name} v{card.version}")
        self.bus.emit_status(f"Model ready: {card.name}")

    def _on_training_step(self, step, loss, label, correct):
        self.lbl_step.setText(f"  Steps: {step}")

    def _on_training_evaluated(self, metrics_dict):
        acc = metrics_dict.get('accuracy', 0.0)
        self.lbl_acc.setText(f"acc: {acc:.4f}")

    def _on_training_finished(self, metrics_dict):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        acc = metrics_dict.get('accuracy', 0.0)
        r2  = metrics_dict.get('r2_score', 0.0)
        self.lbl_acc.setText(f"acc: {acc:.4f}" if acc else f"R²: {r2:.4f}")
        self.bus.emit_status("Training complete")
        self._refresh_model_combo()

    def _on_error(self, msg: str):
        QMessageBox.warning(self, "Error", msg)
        self.lbl_status.setText(f"Error: {msg[:60]}")

    def _on_model_selected(self, text: str):
        if text and text != "(no model loaded)":
            # Parse "name v1.0.0" format
            try:
                parts = text.rsplit(' v', 1)
                if len(parts) == 2:
                    name, version = parts
                    self._load_model_by_name(name, version)
            except Exception as e:
                self.bus.emit_error(str(e))

    def _load_model_by_name(self, name: str, version: str):
        try:
            from ..core.inference import InferenceEngine, InferenceSession

            model, pre, card = self.state.registry.load(name, version)
            p = self.state.preferences
            self.state.engine = InferenceEngine(
                model, pre, ood_threshold=p.inference_ood_threshold
            )
            self.state.session = InferenceSession(self.state.engine)
            self.state.session.set_gh_params(p.inference_chi, p.inference_psi)
            self.bus.emit_model_loaded(card)
            self._update_inference_status_label()
        except Exception as e:
            self.bus.emit_error(f"Failed to load {name}: {e}")

    def _refresh_model_combo(self):
        self.model_combo.blockSignals(True)
        current = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItem("(no model loaded)")
        for name in self.state.registry.list_model_names():
            for ver in self.state.registry.registered_versions(name):
                self.model_combo.addItem(f"{name} v{ver}")
        # Restore selection
        idx = self.model_combo.findText(current)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)

    # ── Menu actions ──────────────────────────────────────────────────────────

    def _on_new_experiment(self):
        dlg = NewExperimentDialog(self.state, self)
        dlg.exec()

    def _on_load_model(self):
        dlg = LoadModelDialog(self.state, self)
        if dlg.exec():
            self._refresh_model_combo()

    def _on_load_cyphalm(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load CyphaLM checkpoint",
            "",
            "CyphaLM JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            from ..core.lm_engine import LMEngine

            self.state.lm_engine = LMEngine.from_checkpoint(path)
            summary = self.state.lm_engine.summary()
            SignalBus.instance().lm_loaded.emit(summary)
            SignalBus.instance().emit_status(
                f"CyphaLM loaded ({summary.get('n_experts', 0)} experts)"
            )
            self.chat_widget.on_lm_loaded(summary)
        except ImportError as exc:
            QMessageBox.warning(
                self,
                "CyphaLM",
                f"cypha_lm is not installed:\n{exc}\n\nRun: pip install -e cypha_lm/",
            )
        except Exception as exc:
            QMessageBox.critical(self, "CyphaLM", f"Failed to load checkpoint:\n{exc}")

    def _on_import_dataset(self):
        from .path_history import dataset_dialog_start_dir_preferred

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Dataset",
            dataset_dialog_start_dir_preferred(self.state.preferences),
            "Data files (*.csv *.npy *.npz);;All files (*)",
        )
        if path:
            self.dataset_widget.load_file(path)
            self._dataset_dock.show()
            self._dataset_dock.raise_()

    def _refresh_recent_dataset_menu(self):
        import os

        from .path_history import recent_dataset_paths

        self._recent_menu.clear()
        paths = [p for p in recent_dataset_paths() if os.path.isfile(p)]
        if not paths:
            a = self._recent_menu.addAction("(none)")
            a.setEnabled(False)
            return
        for p in paths:
            disp = p if len(p) < 80 else "…" + p[-76:]
            act = self._recent_menu.addAction(disp)
            act.triggered.connect(lambda *_, pth=p: self._open_recent_dataset(pth))

    def _open_recent_dataset(self, path: str):
        import os

        if not os.path.isfile(path):
            QMessageBox.warning(
                self,
                "Recent dataset",
                f"File no longer exists:\n{path}",
            )
            self._refresh_recent_dataset_menu()
            return
        self.dataset_widget.load_file(path)
        self._dataset_dock.show()
        self._dataset_dock.raise_()

    def _on_keyboard_shortcuts(self):
        QMessageBox.information(
            self,
            "Keyboard shortcuts",
            "Ctrl+Enter — Send chat message\n"
            "Ctrl+L — Focus chat input\n"
            "Enter — Send (when chat input is focused)\n\n"
            "File → Load CyphaLM… — char-level LM generation in Chat (streamed).\n"
            "File → Recent Datasets lists recently opened CSV/NPZ/NPY paths.\n"
            "View → Studio Log shows status and error history.\n"
            "View → User Guide opens the in-app documentation.",
        )

    def _on_settings(self):
        SettingsDialog(self.state, self).exec()

    def _on_user_guide(self):
        self._help_dock.show()
        self._help_dock.raise_()

    def _apply_ui_font_from_preferences(self):
        pt = int(self.state.preferences.ui_font_pt)
        app_inst = QApplication.instance()
        if app_inst is None:
            return
        if pt <= 0:
            app_inst.setFont(QFont())
            return
        f = app_inst.font()
        f.setPointSize(pt)
        app_inst.setFont(f)

    def _update_inference_status_label(self):
        p = self.state.preferences
        gh = "GH on" if p.inference_use_gh else "GH off"
        self.lbl_infer.setText(
            f"Infer: {gh}  ·  OOD≤{p.inference_ood_threshold:g}  ·  χ={p.inference_chi:g} ψ={p.inference_psi:g}"
        )

    def _on_preferences_changed(self):
        from pathlib import Path

        from ..core.registry import ModelRegistry

        apply_preferences_to_inference_state(self.state)
        self._apply_ui_font_from_preferences()
        self._update_inference_status_label()
        self.chat_widget.refresh_inference_banner()

        want = Path(self.state.preferences.effective_registry_root()).expanduser().resolve()
        have = self.state.registry.root.resolve()
        if want != have:
            self.state.registry = ModelRegistry(str(want))
            self.state.engine = None
            self.state.session = None
            self.state.current_card = None
            self.lbl_model.setText("No model")
            self._refresh_model_combo()
            self.bus.emit_status("Registry root updated — load a model from the new location.")

    def _on_export_test_predictions(self):
        import csv

        if self.state.engine is None:
            QMessageBox.information(self, "No Model", "Load a model first.")
            return
        te = getattr(self.state, "_test_ds", None)
        if te is None:
            QMessageBox.information(
                self,
                "No dataset",
                "Load a dataset so train/val/test splits exist.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export test predictions",
            "",
            "CSV (*.csv);;All files (*)",
        )
        if not path:
            return
        card = self.state.current_card
        is_reg = bool(
            card and getattr(card, "task", "") == "regression"
        ) or getattr(self.state.engine, "_task", "") == "regression"
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if is_reg:
                    w.writerow(["index", "y_true", "y_pred", "uncertainty"])
                    for i in range(len(te)):
                        pred = self.state.session.predict(
                            te.X[i], use_gh=self.state.preferences.inference_use_gh
                        )
                        yt = float(te.y[i])
                        yp = (
                            pred.regression_val
                            if pred.regression_val is not None
                            else float("nan")
                        )
                        w.writerow([i, yt, yp, pred.uncertainty])
                else:
                    w.writerow(
                        [
                            "index",
                            "y_true",
                            "y_pred",
                            "confidence",
                            "is_ood",
                            "anomaly_score",
                        ]
                    )
                    for i in range(len(te)):
                        pred = self.state.session.predict(
                            te.X[i], use_gh=self.state.preferences.inference_use_gh
                        )
                        w.writerow(
                            [
                                i,
                                te.y[i],
                                pred.label,
                                f"{pred.confidence:.6f}",
                                int(pred.is_ood),
                                f"{pred.anomaly_score:.6f}",
                            ]
                        )
            self.bus.emit_status(f"Exported {len(te)} rows → {path}")
        except Exception as e:
            self.bus.emit_error(f"Export failed: {e}")

    def _on_export_model(self):
        if self.state.engine is None:
            QMessageBox.information(self, "No Model", "Load or train a model first.")
            return
        dlg = ExportModelDialog(self.state, self)
        dlg.exec()

    def _on_train_config(self):
        dlg = TrainConfigDialog(self.state, self)
        dlg.exec()

    def _on_start_train(self):
        if self.state.engine is None:
            QMessageBox.information(self, "No Model",
                "Load a dataset and configure training first.")
            return
        from ..core.trainer import Trainer, TrainerConfig
        config = getattr(self.state, '_train_config', TrainerConfig())
        train_ds = getattr(self.state, '_train_ds', None)
        val_ds   = getattr(self.state, '_val_ds',   None)
        if train_ds is None:
            QMessageBox.information(self, "No Dataset",
                "Import a dataset in the Dataset panel first.")
            return

        trainer = Trainer()
        self.state.trainer = trainer

        # Create experiment run
        exp_id = getattr(self.state, '_active_exp_id', None)
        if exp_id is None:
            exp = self.state.db.create_experiment('default')
            exp_id = exp.experiment_id
        run = self.state.db.create_run(exp_id, f"run-{int(__import__('time').time())}", config)

        worker = TrainingWorker(trainer, train_ds, val_ds, config,
                                run.run_id, self.state.db)
        self.state.train_thread = worker
        worker.finished.connect(lambda: self.btn_stop.setEnabled(False))
        worker.start()

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.bus.emit_status("Training started…")
        self._train_dock.show(); self._train_dock.raise_()

    def _on_stop_train(self):
        if self.state.trainer:
            self.state.trainer.stop()
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.bus.emit_status("Training stopped")
        self.bus.training_stopped.emit()

    def _on_evaluate(self):
        if self.state.engine is None:
            QMessageBox.information(self, "No Model", "No model loaded.")
            return
        val_ds = getattr(self.state, '_val_ds', None)
        if val_ds is None:
            QMessageBox.information(self, "No Dataset", "No validation set available.")
            return
        from ..core.trainer import Trainer, TrainerConfig
        config = getattr(self.state, '_train_config', TrainerConfig())
        trainer = Trainer()
        trainer._model = self.state.engine.model
        trainer._preprocessor = self.state.engine._preprocessor
        metrics = trainer.evaluate(val_ds, config)
        self.bus.emit_training_evaluated(
            {'accuracy': metrics.accuracy, 'macro_f1': metrics.macro_f1,
             'r2_score': metrics.r2_score, 'step': 0}
        )

    def _toggle_model_panel(self):
        self._model_dock.setVisible(not self._model_dock.isVisible())

    def _toggle_dock(self, name: str):
        dock = self._docks.get(name)
        if dock:
            dock.setVisible(not dock.isVisible())
            if dock.isVisible():
                dock.raise_()

    def _reset_layout(self):
        for dock in self._docks.values():
            if dock:
                dock.setVisible(True)

    def _on_compare_models(self):
        pairs = []
        for name in self.state.registry.list_model_names():
            for ver in self.state.registry.registered_versions(name):
                pairs.append((name, ver))
        if len(pairs) < 2:
            QMessageBox.information(self, "Compare", "Need at least 2 models to compare.")
            return
        self._exp_dock.show(); self._exp_dock.raise_()
        self.exp_widget.show_comparison(pairs)

    def _on_confusion_matrix(self):
        from .confusion_dialog import show_confusion_dialog

        if self.state.engine is None:
            QMessageBox.information(self, "No Model", "Load a model first.")
            return
        te = getattr(self.state, "_test_ds", None)
        if te is None:
            QMessageBox.information(
                self,
                "No dataset",
                "Load a dataset so a test split exists.",
            )
            return
        card = self.state.current_card
        if (card and getattr(card, "task", "") == "regression") or getattr(
            self.state.engine, "_task", ""
        ) == "regression":
            QMessageBox.information(
                self,
                "Classification only",
                "Confusion matrix applies to classification tasks.",
            )
            return
        y_true, y_pred = [], []
        ugh = self.state.preferences.inference_use_gh
        for i in range(len(te)):
            pred = self.state.session.predict(te.X[i], use_gh=ugh)
            y_true.append(str(te.y[i]))
            y_pred.append(pred.label)
        show_confusion_dialog(self, y_true, y_pred)

    def _on_promote(self):
        card = self.state.current_card
        if card is None:
            QMessageBox.information(self, "Promote", "No model loaded.")
            return
        self.state.registry.promote(card.name, card.version, to='production')
        self.bus.emit_status(f"Promoted {card.name} v{card.version} to production")
        self.bus.registry_changed.emit()

    def _on_about(self):
        QMessageBox.about(self, "CyphaStudio",
            "<b>CyphaStudio</b><br>"
            "Training, inference, and monitoring for Cypha DIF models.<br><br>"
            "Built on CyphaDIF — online learning, GH adversarial protection,<br>"
            "RFF universal approximation, and NIG posterior uncertainty.<br><br>"
            "Binary format: <code>CYPHA\\x00</code> v3 (C++/native compatible)")
