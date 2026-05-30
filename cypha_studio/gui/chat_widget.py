"""
cypha_studio.gui.chat_widget
─────────────────────────────
Chat interface: type input → model responds with label + confidence.
Right-click to correct. Expandable LLR breakdown. OOD highlighting.
"""
from __future__ import annotations
import time
from PySide6.QtCore    import Qt, Signal, QTimer
from PySide6.QtGui     import QColor, QTextCursor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QFrame, QScrollArea, QSizePolicy,
    QMenu, QDialog, QListWidget, QDialogButtonBox,
)
from ..server.local_server import SignalBus


class MessageBubble(QFrame):
    """One message in the chat — user input or model response."""

    correction_requested = Signal(object, str)  # (prediction, correct_label)

    def __init__(self, role: str, content: str,
                 prediction=None, parent=None):
        super().__init__(parent)
        self._prediction = prediction
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        # Role label
        role_lbl = QLabel(role.upper())
        role_lbl.setStyleSheet(
            "color: #888; font-size: 10px; font-weight: bold;"
        )
        layout.addWidget(role_lbl)

        # Content
        self._content_lbl = QLabel(content)
        self._content_lbl.setWordWrap(True)
        self._content_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._content_lbl)

        # Colour by role
        colours = {
            'user'  : '#1a2a3a',
            'model' : '#1e2e1e',
            'system': '#2a2a1a',
            'error' : '#3a1a1a',
            'ood'   : '#3a1e1e',
        }
        self.setStyleSheet(
            f"background-color: {colours.get(role, '#222')}; "
            f"border-radius: 6px; border: 1px solid #333;"
        )

        # OOD warning
        if prediction is not None and prediction.is_ood:
            ood_lbl = QLabel("⚠  Out-of-distribution input flagged")
            ood_lbl.setStyleSheet("color: #ff8080; font-size: 11px;")
            layout.addWidget(ood_lbl)

        # Confidence detail (collapsible)
        if prediction is not None and prediction.all_scores:
            self._detail = QLabel()
            self._detail.setVisible(False)
            self._detail.setStyleSheet("color: #aaa; font-size: 11px;")
            self._update_detail(prediction)
            layout.addWidget(self._detail)

            toggle = QLabel('<a href="#">Show scores ▾</a>')
            toggle.setOpenExternalLinks(False)
            toggle.linkActivated.connect(self._toggle_detail)
            layout.addWidget(toggle)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def _update_detail(self, pred):
        lines = [f"  {lbl}: {score:+.3f}"
                 for lbl, score in sorted(pred.all_scores.items(),
                                          key=lambda kv: -kv[1])]
        self._detail.setText("\n".join(lines))

    def _toggle_detail(self):
        if hasattr(self, '_detail'):
            self._detail.setVisible(not self._detail.isVisible())

    def _context_menu(self, pos):
        if self._prediction is None:
            return
        menu = QMenu(self)
        menu.addAction("Correct this prediction…",
                       self._show_correction_dialog)
        menu.addAction("Show explanation",
                       lambda: self._toggle_detail())
        menu.exec(self.mapToGlobal(pos))

    def _show_correction_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Correct Prediction")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(
            f"Current prediction: <b>{self._prediction.label}</b><br>"
            "Choose correct label:"
        ))
        lst = QListWidget()
        if self._prediction.all_scores:
            for lbl in sorted(self._prediction.all_scores.keys()):
                lst.addItem(lbl)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(lst)
        layout.addWidget(buttons)
        if dlg.exec() and lst.currentItem():
            correct = lst.currentItem().text()
            self.correction_requested.emit(self._prediction, correct)


class ChatWidget(QWidget):
    """Main chat interface widget."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state
        self._bus   = SignalBus.instance()
        self._setup_ui()
        self._connect_signals()
        self.refresh_inference_banner()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        self._header = QLabel("CyphaStudio Chat  —  no model loaded")
        self._header.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #aaa; padding: 4px;"
        )
        layout.addWidget(self._header)

        self._infer_lbl = QLabel("")
        self._infer_lbl.setStyleSheet("color: #777; font-size: 11px; padding: 0 4px;")
        self._infer_lbl.setWordWrap(True)
        layout.addWidget(self._infer_lbl)

        self._empty_tip = QLabel(
            "Load a model: <b>File → Open Model…</b> or train from <b>Dataset</b> + <b>Train</b>. "
            "Shortcuts: <b>Ctrl+L</b> focus input · <b>Ctrl+Enter</b> send."
        )
        self._empty_tip.setWordWrap(True)
        self._empty_tip.setTextFormat(Qt.RichText)
        self._empty_tip.setOpenExternalLinks(False)
        self._empty_tip.setStyleSheet(
            "color: #777; font-size: 11px; padding: 0 4px 6px 4px;"
        )
        layout.addWidget(self._empty_tip)

        # Scroll area for messages
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._msg_container = QWidget()
        self._msg_layout    = QVBoxLayout(self._msg_container)
        self._msg_layout.setAlignment(Qt.AlignTop)
        self._msg_layout.setSpacing(6)
        self._msg_layout.addStretch()
        self._scroll.setWidget(self._msg_container)
        layout.addWidget(self._scroll, 1)

        # Session stats bar
        self._stats_lbl = QLabel(
            "Predictions: 0   Corrections: 0   Avg conf: —   OOD: 0"
        )
        self._stats_lbl.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
        layout.addWidget(self._stats_lbl)

        # Input area
        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText(
            "Enter a comma-separated feature vector or text (if text model)…"
        )
        self._input.returnPressed.connect(self._on_send)
        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(70)
        self._send_btn.clicked.connect(self._on_send)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedWidth(60)
        self._clear_btn.clicked.connect(self._on_clear)
        input_row.addWidget(self._input)
        input_row.addWidget(self._send_btn)
        input_row.addWidget(self._clear_btn)
        layout.addLayout(input_row)

    def _connect_signals(self):
        self._bus.model_loaded.connect(self._on_model_loaded)
        self._bus.preferences_changed.connect(self.refresh_inference_banner)

    def refresh_inference_banner(self):
        p = self._state.preferences
        try:
            ood = float(p.inference_ood_threshold)
        except Exception:
            ood = 0.0
        try:
            chi = float(p.inference_chi)
        except Exception:
            chi = 1.0
        try:
            psi = float(p.inference_psi)
        except Exception:
            psi = 1.0
        self._infer_lbl.setText(
            f"Inference: {'GH path' if p.inference_use_gh else 'plain infer'}  ·  "
            f"OOD threshold {ood:g}  ·  χ={chi:g} ψ={psi:g}"
        )

    def _on_model_loaded(self, card):
        self._empty_tip.hide()
        self.refresh_inference_banner()
        self._header.setText(
            f"CyphaStudio Chat  —  {card.name} v{card.version}  "
            f"[{card.task}]"
        )
        self._add_system(f"Model loaded: {card.name} v{card.version}. "
                         f"Task: {card.task}. "
                         f"Classes: {', '.join(card.class_labels) or 'unknown'}")

    def _on_send(self):
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()

        # Display user message
        self._add_message('user', text)

        # Parse input
        engine = self._state.engine
        if engine is None:
            self._add_message('error', "No model loaded. Load a model first.")
            return

        try:
            # Try numeric vector first
            parts = [p.strip() for p in text.split(',')]
            if all(self._is_number(p) for p in parts):
                import numpy as np
                x = np.array([float(p) for p in parts], dtype=np.float64)
            else:
                # Text input
                x = text

            pred = self._state.session.predict(
                x, use_gh=self._state.preferences.inference_use_gh
            )
            self._add_prediction(pred)
            self._bus.emit_prediction(pred)
            self._update_stats()

        except Exception as e:
            self._add_message('error', f"Inference error: {e}")

    def _is_number(self, s: str) -> bool:
        try: float(s); return True
        except ValueError: return False

    def _add_message(self, role: str, content: str, prediction=None):
        bubble = MessageBubble(role, content, prediction, self._msg_container)
        if prediction is not None:
            bubble.correction_requested.connect(self._on_correction)
        # Insert before the stretch
        count = self._msg_layout.count()
        self._msg_layout.insertWidget(count - 1, bubble)
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _add_system(self, content: str):
        self._add_message('system', content)

    def _add_prediction(self, pred):
        if pred.regression_val is not None:
            content = (f"<b>{pred.regression_val:.4f}</b>  "
                       f"± {pred.uncertainty:.4f}  "
                       f"(confidence {pred.confidence:.3f})")
        else:
            conf_bar = "█" * int(pred.confidence * 20) + "░" * (20 - int(pred.confidence * 20))
            content  = (f"<b>{pred.label}</b>  "
                        f"<span style='font-family:monospace;color:#888'>{conf_bar}</span>  "
                        f"{pred.confidence*100:.1f}%")
        role = 'ood' if pred.is_ood else 'model'
        self._add_message(role, content, prediction=pred)

    def _on_correction(self, prediction, correct_label):
        if self._state.session is None:
            return
        loss = self._state.session.correct(prediction, correct_label)
        self._add_message('system',
            f"Correction applied: {prediction.label} → {correct_label}  "
            f"(training loss: {loss:.4f})")
        self._bus.model_updated.emit(len(self._state.session.corrections))
        self._update_stats()

    def _on_clear(self):
        if self._state.session:
            self._state.session.clear()
        # Remove all message bubbles
        for i in reversed(range(self._msg_layout.count() - 1)):
            item = self._msg_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        self._update_stats()

    def _update_stats(self):
        if self._state.session is None:
            return
        s = self._state.session.summary()
        self._stats_lbl.setText(
            f"Predictions: {s.get('n_predictions',0)}   "
            f"Corrections: {s.get('n_corrections',0)}   "
            f"Avg conf: {s.get('mean_confidence',0):.3f}   "
            f"OOD: {s.get('n_ood_flagged',0)}"
        )

    def _scroll_to_bottom(self):
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())
