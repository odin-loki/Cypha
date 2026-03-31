"""
cypha_studio.server.local_server
─────────────────────────────────
In-process pub/sub server using Qt signals.

Every widget subscribes to the signals it needs. The training loop,
inference engine, and GUI never hold direct references to each other —
they all talk through this bus.

Usage:
    bus = SignalBus.instance()
    bus.prediction_made.connect(my_widget.on_prediction)
    bus.emit_prediction(pred)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from PySide6.QtCore import QObject, Signal
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

if TYPE_CHECKING:
    from ..core.inference import Prediction
    from ..core.trainer import EvalMetrics, TrainerConfig
    from ..core.registry import ModelCard


if QT_AVAILABLE:
    class SignalBus(QObject):
        """
        Singleton Qt signal bus. All inter-component communication goes
        through here so components stay fully decoupled.

        Signals
        ───────
        model_loaded(ModelCard)                 — new model is active
        model_unloaded()                        — model removed
        prediction_made(dict)                   — inference result
        session_updated(dict)                   — session summary changed
        training_started(dict)                  — config dict
        training_step(int, float, str, bool)    — step, loss, label, correct
        training_evaluated(dict)                — EvalMetrics as dict
        training_epoch_done(int, dict)          — epoch, metrics dict
        training_finished(dict)                 — final EvalMetrics as dict
        training_stopped()                      — user requested stop
        model_updated(int)                      — n_corrections after online update
        drift_detected(float)                   — drift score
        anomaly_detected(float)                 — anomaly score for OOD input
        registry_changed()                      — model added/removed/promoted
        error_occurred(str)                     — error message for status bar
        status_message(str)                     — informational status bar update
        """

        # ── Model lifecycle ──────────────────────────────────────────────────
        model_loaded        = Signal(object)   # ModelCard
        model_unloaded      = Signal()

        # ── Inference ────────────────────────────────────────────────────────
        prediction_made     = Signal(object)   # Prediction
        session_updated     = Signal(dict)

        # ── Training ─────────────────────────────────────────────────────────
        training_started    = Signal(dict)     # TrainerConfig as dict
        training_step       = Signal(int, float, str, bool)
        training_evaluated  = Signal(dict)     # EvalMetrics as dict
        training_epoch_done = Signal(int, dict)
        training_finished   = Signal(dict)
        training_stopped    = Signal()

        # ── Online updates ───────────────────────────────────────────────────
        model_updated       = Signal(int)      # n_corrections

        # ── Monitoring ───────────────────────────────────────────────────────
        drift_detected      = Signal(float)
        anomaly_detected    = Signal(float)

        # ── Registry ─────────────────────────────────────────────────────────
        registry_changed    = Signal()

        # ── UI ───────────────────────────────────────────────────────────────
        error_occurred      = Signal(str)
        status_message      = Signal(str)
        dataset_opened      = Signal(str)      # filesystem path (CSV / npy / npz)

        _instance = None

        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

        @classmethod
        def instance(cls) -> 'SignalBus':
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        # ── Emit helpers (type-checked wrappers) ─────────────────────────────

        def emit_model_loaded(self, card):
            self.model_loaded.emit(card)
            self.status_message.emit(f"Model loaded: {card.name} v{card.version}")

        def emit_prediction(self, pred):
            self.prediction_made.emit(pred)
            if pred.is_ood:
                self.anomaly_detected.emit(pred.anomaly_score)

        def emit_training_step(self, step: int, loss: float,
                                label: str, correct: bool):
            self.training_step.emit(step, loss, label, correct)

        def emit_training_evaluated(self, metrics):
            try:
                from dataclasses import asdict as _asdict
                d = _asdict(metrics)
            except Exception:
                d = {'accuracy': getattr(metrics, 'accuracy', 0.0)}
            self.training_evaluated.emit(d)

        def emit_training_finished(self, metrics):
            try:
                from dataclasses import asdict as _asdict
                d = _asdict(metrics)
            except Exception:
                d = {'accuracy': getattr(metrics, 'accuracy', 0.0)}
            self.training_finished.emit(d)
            self.status_message.emit(
                f"Training complete — acc={d.get('accuracy', 0):.4f}"
            )

        def emit_error(self, msg: str):
            self.error_occurred.emit(msg)

        def emit_status(self, msg: str):
            self.status_message.emit(msg)

        def emit_dataset_opened(self, path: str):
            self.dataset_opened.emit(path)

else:
    # Headless fallback: no Qt, use plain callable lists
    class SignalBus:  # type: ignore
        """Headless fallback when PySide6 is not available."""

        _instance = None

        def __new__(cls):
            if cls._instance is None:
                obj = object.__new__(cls)
                obj._handlers = {}
                cls._instance = obj
            return cls._instance

        @classmethod
        def instance(cls) -> 'SignalBus':
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        def _get(self, name):
            return self._handlers.setdefault(name, [])

        def _emit(self, name, *args):
            for fn in self._handlers.get(name, []):
                try: fn(*args)
                except Exception: pass

        # Minimal shim matching the Qt API
        def emit_model_loaded(self, card):    self._emit('model_loaded', card)
        def emit_prediction(self, pred):      self._emit('prediction_made', pred)
        def emit_training_step(self, *a):     self._emit('training_step', *a)
        def emit_training_evaluated(self, m): self._emit('training_evaluated', m)
        def emit_training_finished(self, m):  self._emit('training_finished', m)
        def emit_error(self, msg):            self._emit('error_occurred', msg)
        def emit_status(self, msg):           self._emit('status_message', msg)
        def emit_dataset_opened(self, path: str): self._emit('dataset_opened', path)

        def connect(self, signal_name: str, handler):
            self._get(signal_name).append(handler)
