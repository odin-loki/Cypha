"""
Studio Settings — inference, paths, appearance, environment reference.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .studio_preferences import StudioPreferences, save_studio_preferences
from ..env_config import api_default_host, api_default_port, cors_allow_origins
from ..env_config import csv_read_chunk_rows as env_csv_chunk_rows
from ..env_config import registry_root as env_registry_root
from ..server.local_server import SignalBus


class SettingsDialog(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state
        self.setWindowTitle("Studio Settings")
        self.resize(520, 440)

        p: StudioPreferences = state.preferences
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        root.addWidget(tabs)

        tabs.addTab(self._tab_inference(p), "Inference")
        tabs.addTab(self._tab_paths(p), "Paths & data")
        tabs.addTab(self._tab_appearance(p), "Appearance")
        tabs.addTab(self._tab_environment(), "Environment")

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._on_restore_defaults)
        root.addWidget(btns)

    def _tab_inference(self, p: StudioPreferences) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Classification inference pipeline")
        grp.setToolTip(
            "GH (Griffiths–Hinton) path uses gh_infer with session χ/ψ state. "
            "OOD uses anomaly score vs threshold on the engine."
        )
        form = QFormLayout(grp)
        self._use_gh = QCheckBox("Use GH inference path (when model supports it)")
        self._use_gh.setChecked(p.inference_use_gh)
        self._use_gh.setToolTip(
            "If enabled, chat and batch tools call gh_infer with χ, ψ. "
            "Regression models ignore this for the forward pass."
        )
        form.addRow(self._use_gh)

        self._ood = QDoubleSpinBox()
        self._ood.setRange(0.1, 100.0)
        self._ood.setDecimals(2)
        self._ood.setValue(p.inference_ood_threshold)
        self._ood.setToolTip(
            "anomaly_score greater than this marks the input as out-of-distribution."
        )
        form.addRow("OOD threshold:", self._ood)

        self._chi = QDoubleSpinBox()
        self._chi.setRange(0.01, 100.0)
        self._chi.setDecimals(4)
        self._chi.setValue(p.inference_chi)
        self._chi.setToolTip("GH prior strength χ passed to gh_infer.")
        form.addRow("GH χ:", self._chi)

        self._psi = QDoubleSpinBox()
        self._psi.setRange(0.01, 100.0)
        self._psi.setDecimals(4)
        self._psi.setValue(p.inference_psi)
        self._psi.setToolTip("GH prior strength ψ passed to gh_infer.")
        form.addRow("GH ψ:", self._psi)

        lay.addWidget(grp)
        note = QLabel(
            "Changes apply to the next prediction and to the loaded model immediately "
            "(OOD threshold and χ/ψ)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888; font-size: 11px;")
        lay.addWidget(note)
        lay.addStretch()
        return w

    def _tab_paths(self, p: StudioPreferences) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        reg_row = QHBoxLayout()
        self._reg_edit = QLineEdit(p.registry_root_override)
        self._reg_edit.setPlaceholderText(f"Default: {env_registry_root()}")
        self._reg_edit.setToolTip(
            "Leave empty to use CYPHA_REGISTRY_ROOT or ~/.cypha/models. "
            "Non-empty overrides for this Studio session after Save."
        )
        br = QPushButton("Browse…")
        br.clicked.connect(self._browse_registry)
        reg_row.addWidget(self._reg_edit)
        reg_row.addWidget(br)
        lay.addWidget(QLabel("<b>Model registry root</b>"))
        lay.addLayout(reg_row)

        self._csv_chunk = QSpinBox()
        self._csv_chunk.setRange(0, 10_000_000)
        self._csv_chunk.setMinimum(0)
        self._csv_chunk.setSpecialValueText("From env / full file")
        self._csv_chunk.setValue(
            p.csv_chunk_rows_override if p.csv_chunk_rows_override > 0 else 0
        )
        self._csv_chunk.setToolTip(
            "0 = use CYPHA_CSV_CHUNK_ROWS if set, else load full CSV. "
            "Positive = stream CSV in chunks of this many rows."
        )
        form = QFormLayout()
        form.addRow("CSV chunk rows (0 = auto):", self._csv_chunk)
        lay.addLayout(form)

        ds_row = QHBoxLayout()
        self._ds_dir = QLineEdit(p.dataset_dialog_start_dir)
        self._ds_dir.setPlaceholderText("Use recent-path history")
        self._ds_dir.setToolTip("Optional starting folder for File → Import Dataset / Dataset Load.")
        db = QPushButton("Browse…")
        db.clicked.connect(self._browse_dataset_dir)
        ds_row.addWidget(self._ds_dir)
        ds_row.addWidget(db)
        lay.addWidget(QLabel("<b>Dataset file dialog start folder</b>"))
        lay.addLayout(ds_row)

        env_note = QLabel(
            f"<span style='color:#888'>Env: CYPHA_REGISTRY_ROOT, CYPHA_CSV_CHUNK_ROWS</span>"
        )
        env_note.setTextFormat(Qt.RichText)
        lay.addWidget(env_note)
        lay.addStretch()
        return w

    def _tab_appearance(self, p: StudioPreferences) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self._font_pt = QSpinBox()
        self._font_pt.setRange(0, 24)
        self._font_pt.setValue(p.ui_font_pt)
        self._font_pt.setToolTip(
            "0 = do not change application font. 9–14 recommended for readability."
        )
        form = QFormLayout()
        form.addRow("UI font size (pt, 0 = default):", self._font_pt)
        lay.addLayout(form)
        lay.addStretch()
        return w

    def _tab_environment(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        cors = cors_allow_origins()
        cors_s = ", ".join(cors) if cors != ["*"] else "*"
        txt = (
            "<p><b>Variables</b> (see <code>docs/studio/CYPHA_ENV.md</code> in the repo)</p>"
            "<table cellspacing='8' style='font-size:12px'>"
            "<tr><td><code>CYPHA_REGISTRY_ROOT</code></td><td>Model tree root</td></tr>"
            "<tr><td><code>CYPHA_API_HOST</code></td><td>REST bind host</td></tr>"
            "<tr><td><code>CYPHA_API_PORT</code></td><td>REST port</td></tr>"
            "<tr><td><code>CYPHA_CORS_ORIGINS</code></td><td>CORS allow list</td></tr>"
            "<tr><td><code>CYPHA_CSV_CHUNK_ROWS</code></td><td>CSV streaming chunk size</td></tr>"
            "<tr><td><code>CYPHA_REGRESSION_HEAD</code></td><td>Optional regression head JSON</td></tr>"
            "</table>"
            "<p><b>Effective REST defaults</b> (read-only):<br>"
            f"{api_default_host()}:{api_default_port()} &nbsp; CORS: {cors_s}<br>"
            f"Registry: <code>{env_registry_root()}</code><br>"
            f"CSV chunk (env): {env_csv_chunk_rows() or 'full buffer'}</p>"
        )
        lbl = QLabel(txt)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.RichText)
        lbl.setOpenExternalLinks(False)
        lay.addWidget(lbl)
        lay.addStretch()
        return w

    def _browse_registry(self):
        path = QFileDialog.getExistingDirectory(self, "Registry root", self._reg_edit.text())
        if path:
            self._reg_edit.setText(path)

    def _browse_dataset_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Dataset dialog folder", self._ds_dir.text())
        if path:
            self._ds_dir.setText(path)

    def _on_ok(self):
        p = self._state.preferences
        p.inference_use_gh = self._use_gh.isChecked()
        p.inference_ood_threshold = self._ood.value()
        p.inference_chi = self._chi.value()
        p.inference_psi = self._psi.value()
        p.registry_root_override = self._reg_edit.text().strip()
        cr = self._csv_chunk.value()
        p.csv_chunk_rows_override = cr if cr > 0 else 0
        p.dataset_dialog_start_dir = self._ds_dir.text().strip()
        p.ui_font_pt = self._font_pt.value()
        save_studio_preferences(p)
        SignalBus.instance().emit_preferences_changed()
        self.accept()

    def _on_restore_defaults(self):
        d = StudioPreferences()
        self._use_gh.setChecked(d.inference_use_gh)
        self._ood.setValue(d.inference_ood_threshold)
        self._chi.setValue(d.inference_chi)
        self._psi.setValue(d.inference_psi)
        self._reg_edit.clear()
        self._csv_chunk.setValue(0)
        self._ds_dir.clear()
        self._font_pt.setValue(0)
