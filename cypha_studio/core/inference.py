"""
cypha_studio.core.inference
────────────────────────────
Inference engine with session tracking, online correction,
explanation, uncertainty, and an optional text-to-vector front-end.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dataset import Preprocessor

# ─────────────────────────────────────────────────────────────────────────────
# Prediction result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Prediction:
    """Result of a single inference call."""
    label          : str
    confidence     : float
    all_scores     : dict[str, float]    = field(default_factory=dict)
    anomaly_score  : float               = 0.0   # 0=normal, 1=OOD
    is_ood         : bool                = False
    r_eff          : float               = 0.0
    uncertainty    : float               = 0.0   # regression: posterior std
    regression_val : float | None     = None  # for regression models
    timestamp      : float               = field(default_factory=time.time)
    input_vector   : np.ndarray | None= None
    raw_input      : Any | None       = None  # original text / dict


@dataclass
class CorrectionRecord:
    """User-provided correction to a prediction."""
    prediction     : Prediction
    correct_label  : str
    timestamp      : float = field(default_factory=time.time)
    applied        : bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# InferenceEngine
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Wraps a trained (CyphaDIF, Preprocessor) pair for inference.

    Handles:
      - Single and batch prediction
      - Explanation (per-class LLR breakdown)
      - OOD detection via GH gate
      - Online correction (updates model in-place)
      - Uncertainty for regression models
    """

    OOD_THRESHOLD = 3.0   # anomaly_score > this → is_ood=True

    def __init__(self, model, preprocessor: Preprocessor | None = None,
                 ood_threshold: float = OOD_THRESHOLD):
        self._model       = model
        self._preprocessor = preprocessor
        self._ood_threshold = ood_threshold
        self._n_predictions = 0
        self._n_corrections = 0
        self._task = self._detect_task()

    def set_ood_threshold(self, t: float) -> None:
        """Update OOD decision boundary (anomaly_score > t → is_ood)."""
        self._ood_threshold = float(t)

    def _detect_task(self) -> str:
        """Infer task type from model class name."""
        name = type(self._model).__name__
        if 'Regressor' in name or 'Regression' in name or 'TwoStage' in name:
            return 'regression'
        return 'classification'

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        if self._preprocessor is not None:
            return self._preprocessor.transform_one(x)
        return np.asarray(x, dtype=np.float64)

    def _preprocess_batch(self, X: np.ndarray) -> np.ndarray:
        if self._preprocessor is not None:
            return self._preprocessor.transform(X)
        return np.asarray(X, dtype=np.float64)

    # ── Single prediction ────────────────────────────────────────────────────

    def predict(self, raw_input: np.ndarray | list[float] | str,
                use_gh: bool = False,
                chi: float = 1.0, psi: float = 1.0) -> Prediction:
        """
        Predict for one input.

        raw_input: numeric vector, list of floats, or text (if TextPreprocessor attached)
        """
        if isinstance(raw_input, str):
            vec = self._text_to_vec(raw_input)
        else:
            vec = np.asarray(raw_input, dtype=np.float64).ravel()

        x_pp = self._preprocess(vec)
        self._n_predictions += 1

        if self._task == 'regression':
            return self._predict_regression(x_pp, vec)

        return self._predict_classification(x_pp, vec, raw_input, use_gh, chi, psi)

    def _predict_classification(self, x_pp, vec, raw_input, use_gh, chi, psi) -> Prediction:
        if use_gh:
            pred, conf, r_eff, chi_new, psi_new = self._model.gh_infer(x_pp, chi, psi)
        else:
            pred, conf = self._model.infer(x_pp)
            r_eff = 0.0

        # Full LLR breakdown for explanation
        try:
            H = self._model.batch_encode(x_pp.reshape(1, -1))
            # Match CyphaDIF.infer(use_field=True) so all_scores align with classification
            llr_matrix, labels = self._model.score_matrix(H, use_field=True)
            all_scores = {str(lbl): float(llr_matrix[0, i])
                          for i, lbl in enumerate(labels)}
        except Exception:
            all_scores = {pred: float(conf)}

        # Anomaly score: R_eff / R_base normalised to [0, ∞)
        anomaly = float(r_eff) if r_eff > 0 else 0.0
        r_base = getattr(self._model, '_mahal_ema', 1.0) or 1.0
        if r_base > 0:
            anomaly = max(0.0, (anomaly - r_base) / r_base)

        return Prediction(
            label=str(pred),
            confidence=float(conf),
            all_scores=all_scores,
            anomaly_score=anomaly,
            is_ood=(anomaly > float(self._ood_threshold)),
            r_eff=float(r_eff),
            input_vector=x_pp,
            raw_input=raw_input,
        )

    def _predict_regression(self, x_pp, vec) -> Prediction:
        if hasattr(self._model, 'predict_with_uncertainty'):
            y_pred, var = self._model.predict_with_uncertainty(x_pp.reshape(1, -1))
            val = float(y_pred[0])
            unc = float(np.sqrt(max(var[0], 0.0)))
        elif hasattr(self._model, 'predict'):
            val = float(self._model.predict(x_pp.reshape(1, -1))[0])
            unc = 0.0
        else:
            y_pred, unc_v = self._model.predict(x_pp)
            val = float(y_pred)
            unc = float(unc_v)

        return Prediction(
            label=f"{val:.4f}",
            confidence=max(0.0, 1.0 - unc / (abs(val) + 1.0)),
            all_scores={'regression': val},
            regression_val=val,
            uncertainty=unc,
            input_vector=x_pp,
            raw_input=vec,
        )

    # ── Batch prediction ─────────────────────────────────────────────────────

    def predict_batch(self, inputs: np.ndarray | list) -> list[Prediction]:
        X = np.array(inputs, dtype=np.float64)
        X_pp = self._preprocess_batch(X)
        results = []
        for i in range(len(X_pp)):
            results.append(self.predict(X[i]))
        return results

    # ── Explanation ──────────────────────────────────────────────────────────

    def explain(self, raw_input: np.ndarray | str) -> dict:
        """
        Return a detailed explanation of a prediction.

        Returns:
          label, confidence, all_scores (LLR per class),
          is_ood, anomaly_score, r_eff, world_mu_distance,
          nearest_class_details (n_obs, delta_mu_norm per class)
        """
        pred = self.predict(raw_input, use_gh=True)

        explanation = {
            'label'        : pred.label,
            'confidence'   : pred.confidence,
            'all_scores'   : pred.all_scores,
            'anomaly_score': pred.anomaly_score,
            'is_ood'       : pred.is_ood,
            'r_eff'        : pred.r_eff,
        }

        # Per-class details from memory
        try:
            with self._model.memory._lock:
                class_details = {}
                for lbl, cd in self._model.memory._classes.items():
                    class_details[lbl] = {
                        'n_obs'       : float(cd.n_obs),
                        'delta_mu_norm': float(np.linalg.norm(cd.delta_mu)),
                    }
            explanation['class_details'] = class_details

            # World prior
            with self._model.memory._lock:
                mu = self._model.memory.world.mu.copy()
            if pred.input_vector is not None:
                dist = float(np.linalg.norm(pred.input_vector - mu))
                explanation['world_mu_distance'] = dist
        except Exception:
            pass

        return explanation

    # ── Online correction ────────────────────────────────────────────────────

    def update(self, raw_input: np.ndarray | str,
               correct_label: str,
               use_gh: bool = True,
               replay_u01: list[float] | None = None) -> float:
        """
        Online update: train model on (input, correct_label).

        Optional ``replay_u01`` fixes priority-replay U(0,1) draws (native ``cypha_rest`` parity).

        Returns training loss.
        """
        if isinstance(raw_input, str):
            vec = self._text_to_vec(raw_input)
        else:
            vec = np.asarray(raw_input, dtype=np.float64).ravel()

        x_pp = self._preprocess(vec)
        self._n_corrections += 1

        old_rng = None
        if replay_u01 is not None:
            from .replay_rng import ListReplayRng
            old_rng = getattr(self._model, "_replay_rng", None)
            lr = ListReplayRng([float(v) for v in replay_u01])
            self._model._replay_rng = lr
            if hasattr(self._model, "replay") and hasattr(self._model.replay, "_rng"):
                self._model.replay._rng = lr

        try:
            if use_gh and hasattr(self._model, 'gh_train_step'):
                loss, _, _, _ = self._model.gh_train_step(x_pp, str(correct_label),
                                                            1.0, 1.0)
            else:
                loss = self._model.train_step(x_pp, str(correct_label))
        finally:
            if old_rng is not None:
                self._model._replay_rng = old_rng
                if hasattr(self._model, "replay") and hasattr(self._model.replay, "_rng"):
                    self._model.replay._rng = old_rng
        return float(loss)

    # ── Anomaly score ────────────────────────────────────────────────────────

    def anomaly_score(self, raw_input: np.ndarray | str) -> float:
        """Return anomaly score in [0, ∞). > OOD_THRESHOLD = likely OOD."""
        pred = self.predict(raw_input, use_gh=True)
        return pred.anomaly_score

    # ── Text preprocessing hook ──────────────────────────────────────────────

    def _text_to_vec(self, text: str) -> np.ndarray:
        """Convert text to vector if a TextPreprocessor is attached."""
        if hasattr(self, '_text_preprocessor') and self._text_preprocessor:
            return self._text_preprocessor.transform(text)
        raise ValueError(
            "Cannot process text input: no TextPreprocessor attached. "
            "Call attach_text_preprocessor() first."
        )

    def attach_text_preprocessor(self, tp: TextPreprocessor):
        self._text_preprocessor = tp

    # ── Stats ────────────────────────────────────────────────────────────────

    @property
    def n_predictions(self) -> int:
        return self._n_predictions

    @property
    def n_corrections(self) -> int:
        return self._n_corrections

    @property
    def model(self):
        return self._model


# ─────────────────────────────────────────────────────────────────────────────
# InferenceSession
# ─────────────────────────────────────────────────────────────────────────────

class InferenceSession:
    """
    Maintains conversation history and per-session GH gate state.

    session = InferenceSession(engine)
    pred = session.predict("some input text")
    session.correct(pred, 'true_label')
    session.summary()
    """

    def __init__(self, engine: InferenceEngine):
        self._engine   = engine
        self._history  : list[Prediction]       = []
        self._corrections : list[CorrectionRecord] = []
        self._chi      : float = 1.0
        self._psi      : float = 1.0
        self._started  : float = time.time()

    def set_gh_params(self, chi: float, psi: float) -> None:
        """Set GH gate prior strengths for subsequent ``predict`` calls."""
        self._chi = float(chi)
        self._psi = float(psi)

    def predict(self, raw_input: Any, use_gh: bool = True) -> Prediction:
        """Predict and update session GH state."""
        pred = self._engine.predict(raw_input, use_gh=use_gh,
                                    chi=self._chi, psi=self._psi)
        self._history.append(pred)

        # Update session chi/psi if gh_infer returned them
        # (stored on prediction via r_eff — no direct access, that's fine)
        return pred

    def correct(self, prediction: Prediction, correct_label: str) -> float:
        """Apply a correction and log it."""
        rec = CorrectionRecord(prediction=prediction, correct_label=correct_label)
        loss = self._engine.update(prediction.input_vector
                                    if prediction.input_vector is not None
                                    else prediction.raw_input,
                                   correct_label)
        rec.applied = True
        self._corrections.append(rec)
        return loss

    def undo_last(self):
        """Remove last prediction from history (no model rollback)."""
        if self._history:
            self._history.pop()

    def clear(self):
        """Clear session history (model state preserved)."""
        self._history.clear()
        self._corrections.clear()
        self._chi = 1.0
        self._psi = 1.0

    def summary(self) -> dict:
        n = len(self._history)
        n_corr = len(self._corrections)
        if n == 0:
            return {
                'n_predictions': 0,
                'n_corrections': 0,
                'correction_accuracy': 0.0,
                'mean_confidence': 0.0,
                'mean_anomaly': 0.0,
                'n_ood_flagged': 0,
                'label_distribution': {},
                'session_duration_s': time.time() - self._started,
            }

        labels = [p.label for p in self._history]
        confs  = [p.confidence for p in self._history]
        anomalies = [p.anomaly_score for p in self._history]

        # Correction accuracy: how many user corrections led to a now-correct prediction?
        corr_acc = 0.0
        if n_corr > 0:
            # Re-predict on corrected inputs and check
            applied = [c for c in self._corrections if c.applied]
            if applied:
                correct_now = sum(
                    1 for c in applied
                    if self._engine.predict(c.prediction.input_vector
                                             if c.prediction.input_vector is not None
                                             else c.prediction.raw_input).label
                    == c.correct_label
                )
                corr_acc = correct_now / len(applied)

        return {
            'n_predictions'     : n,
            'n_corrections'     : n_corr,
            'correction_accuracy': corr_acc,
            'mean_confidence'   : float(np.mean(confs)),
            'mean_anomaly'      : float(np.mean(anomalies)),
            'n_ood_flagged'     : sum(1 for p in self._history if p.is_ood),
            'label_distribution': {lbl: labels.count(lbl)
                                   for lbl in set(labels)},
            'session_duration_s': time.time() - self._started,
        }

    @property
    def history(self) -> list[Prediction]:
        return list(self._history)

    @property
    def corrections(self) -> list[CorrectionRecord]:
        return list(self._corrections)


# ─────────────────────────────────────────────────────────────────────────────
# TextPreprocessor — TF-IDF vectoriser front-end
# ─────────────────────────────────────────────────────────────────────────────

class TextPreprocessor:
    """
    Converts raw text to a fixed-length float vector for CyphaDIF.

    Backed by TF-IDF (always available, no heavy dependencies).
    Optional: sentence embeddings via sentence-transformers if installed.

    tp = TextPreprocessor(mode='tfidf', max_features=512)
    tp.fit(list_of_texts)
    vec = tp.transform("classify this text")
    """

    def __init__(self, mode: str = 'tfidf',
                 max_features: int = 512,
                 model_name: str | None = None):
        self.mode         = mode
        self.max_features = max_features
        self.model_name   = model_name
        self._vectorizer  = None
        self._model       = None
        self._fitted      = False
        self._out_dim     : int | None = None

    def fit(self, texts: list[str]) -> TextPreprocessor:
        if self.mode == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                sublinear_tf=True, strip_accents='unicode',
                analyzer='word', min_df=1,
            )
            self._vectorizer.fit(texts)
            self._out_dim = self.max_features
            self._fitted  = True

        elif self.mode == 'sentence':
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_name or 'all-MiniLM-L6-v2'
                )
                sample = self._model.encode([texts[0]])
                self._out_dim = sample.shape[1]
                self._fitted  = True
            except ImportError:
                raise ImportError(
                    "Install sentence-transformers for sentence embedding mode: "
                    "pip install sentence-transformers"
                ) from None
        else:
            raise ValueError(f"Unknown mode {self.mode!r}")

        return self

    def transform(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        if self.mode == 'tfidf':
            vec = self._vectorizer.transform([text]).toarray()[0]
            return vec.astype(np.float64)
        else:
            return self._model.encode([text])[0].astype(np.float64)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return np.array([self.transform(t) for t in texts], dtype=np.float64)

    @property
    def output_dim(self) -> int | None:
        return self._out_dim

    def save_state(self) -> dict:
        import base64
        import pickle
        state: dict[str, Any] = {
            'mode': self.mode,
            'max_features': self.max_features,
            'model_name': self.model_name,
            'fitted': self._fitted,
            'out_dim': self._out_dim,
        }
        if self.mode == 'tfidf' and self._vectorizer:
            state['vectorizer'] = base64.b64encode(
                pickle.dumps(self._vectorizer)
            ).decode('ascii')
        return state

    def load_state(self, state: dict) -> None:
        import base64
        import pickle
        self.mode         = state['mode']
        self.max_features = state['max_features']
        self.model_name   = state.get('model_name')
        self._fitted      = state['fitted']
        self._out_dim     = state.get('out_dim')
        if state.get('vectorizer'):
            self._vectorizer = pickle.loads(
                base64.b64decode(state['vectorizer'].encode('ascii'))
            )
