"""
cypha_studio.core.trainer
─────────────────────────
Training loop with callbacks, evaluation, online/batch modes,
and hyperparameter search.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time

_CYPHA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _CYPHA_ROOT not in sys.path:
    sys.path.insert(0, _CYPHA_ROOT)
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dataset import CyphaDataset, Preprocessor

# ─────────────────────────────────────────────────────────────────────────────
# TrainerConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    """All hyperparameters for a Cypha training run."""

    # Model type
    model_type    : str   = 'CyphaDIF'      # 'CyphaDIF', 'RFFRegressor', 'TwoStageDIF', 'MKE'

    # Encoder
    encoder_type  : str   = 'VectorEncoder' # 'VectorEncoder', 'RFFEncoder'
    feat_dim      : int   = 128
    field_dim     : int   = 128
    rff_D         : int   = 256
    rff_gamma     : float = 1.0
    auto_gamma_cv : bool  = True
    auto_ard      : bool  = False

    # Learning rates (defaults match scripts/tune_quality_performance.py medium-grid classification best)
    world_lr      : float = 0.008
    delta_lr      : float = 0.05
    enc_lr        : float = 0.002
    mdl_lambda    : float = 0.001
    forgetting_factor : float = 1.0

    # Routing (MKE)
    n_experts     : int   = 8

    # Classification
    temperature   : float = 1.15
    context_win   : int   = 32

    # GH protection
    gh_protect    : bool  = False

    # Training mode
    mode          : str   = 'online'    # 'online' or 'batch'
    n_epochs      : int   = 3
    batch_size    : int | None = None

    # Evaluation
    eval_every_n  : int   = 100         # steps between validation evaluations
    save_every_n  : int   = 500

    # Early stopping
    early_stopping       : bool  = True
    early_stopping_patience : int = 5   # evaluations without improvement

    # Preprocessing
    preprocess_scale : bool = True
    preprocess_pca   : int | None = None

    # Reproducibility
    seed : int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    """Evaluation results from one validation pass."""
    step           : int   = 0
    epoch          : int   = 0
    timestamp      : float = 0.0

    # Classification
    accuracy       : float = 0.0
    macro_f1       : float = 0.0
    macro_recall   : float = 0.0
    macro_precision: float = 0.0
    per_class      : dict[str, dict[str, float]] = field(default_factory=dict)
    calibration_error : float = 0.0
    ood_auroc      : float = 0.0

    # Regression
    r2_score       : float = 0.0
    rmse           : float = 0.0
    mae            : float = 0.0

    # General
    loss           : float = 0.0
    conf_mean      : float = 0.0
    conf_std       : float = 0.0

    # World prior health
    world_mu_norm  : float = 0.0
    n_classes      : int   = 0


def _compute_classification_metrics(y_true, y_pred, y_conf) -> EvalMetrics:
    m = EvalMetrics(timestamp=time.time())
    unique_labels = sorted(set(list(y_true) + list(y_pred)))
    n = len(y_true)
    if n == 0:
        return m

    m.accuracy = sum(p == t for p, t in zip(y_pred, y_true, strict=False)) / n
    m.conf_mean = float(np.mean(y_conf)) if len(y_conf) > 0 else 0.0
    m.conf_std  = float(np.std(y_conf))  if len(y_conf) > 0 else 0.0

    for lbl in unique_labels:
        tp = sum(1 for p, t in zip(y_pred, y_true, strict=False) if p == lbl and t == lbl)
        fp = sum(1 for p, t in zip(y_pred, y_true, strict=False) if p == lbl and t != lbl)
        fn = sum(1 for p, t in zip(y_pred, y_true, strict=False) if p != lbl and t == lbl)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        m.per_class[lbl] = {'precision': prec, 'recall': rec, 'f1': f1,
                            'support': sum(1 for t in y_true if t == lbl)}

    m.macro_precision = float(np.mean([v['precision'] for v in m.per_class.values()]))
    m.macro_recall    = float(np.mean([v['recall']    for v in m.per_class.values()]))
    m.macro_f1        = float(np.mean([v['f1']        for v in m.per_class.values()]))

    # Calibration error (ECE)
    if len(y_conf) > 0:
        confs = np.array(y_conf)
        correct = np.array([int(p == t) for p, t in zip(y_pred, y_true, strict=False)], dtype=float)
        ece = 0.0
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() > 0:
                ece += mask.sum() * abs(confs[mask].mean() - correct[mask].mean()) / n
        m.calibration_error = float(ece)

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class TrainerCallback:
    """Base callback. Override any method to hook into training."""

    def on_train_begin(self, config: TrainerConfig) -> None: pass
    def on_train_end(self, metrics: EvalMetrics)    -> None: pass
    def on_step(self, step: int, loss: float, label: str, correct: bool) -> None: pass
    def on_evaluate(self, metrics: EvalMetrics)     -> None: pass
    def on_save(self, path: str)                    -> None: pass
    def on_epoch_begin(self, epoch: int)            -> None: pass
    def on_epoch_end(self, epoch: int, metrics: EvalMetrics) -> None: pass
    def on_stop_requested(self)                     -> bool: return False


class MetricsCallback(TrainerCallback):
    """Collects training metrics for plotting."""

    def __init__(self):
        self.step_losses     : list[tuple[int, float]] = []
        self.step_accuracies : list[tuple[int, float]] = []
        self.eval_history    : list[EvalMetrics]       = []
        self._window_correct : list[int]   = []
        self._window_size    : int         = 50

    def on_step(self, step, loss, label, correct):
        self.step_losses.append((step, loss))
        self._window_correct.append(int(correct))
        if len(self._window_correct) > self._window_size:
            self._window_correct.pop(0)
        roll_acc = sum(self._window_correct) / max(len(self._window_correct), 1)
        self.step_accuracies.append((step, roll_acc))

    def on_evaluate(self, metrics: EvalMetrics):
        self.eval_history.append(metrics)

    def recent_loss(self, n: int = 100) -> float:
        losses = [l for _, l in self.step_losses[-n:]]
        return float(np.mean(losses)) if losses else 0.0

    def recent_accuracy(self, n: int = 50) -> float:
        accs = [a for _, a in self.step_accuracies[-n:]]
        return float(np.mean(accs)) if accs else 0.0


class ProgressCallback(TrainerCallback):
    """Prints progress to stdout."""

    def __init__(self, print_every: int = 200):
        self._print_every = print_every
        self._t0 = time.time()

    def on_train_begin(self, config):
        self._t0 = time.time()
        print(f"[Trainer] Starting {config.mode} training — {config.model_type}")

    def on_step(self, step, loss, label, correct):
        if step % self._print_every == 0:
            elapsed = time.time() - self._t0
            print(f"  step={step:6d}  loss={loss:.4f}  elapsed={elapsed:.1f}s")

    def on_evaluate(self, metrics):
        print(f"  [Eval] step={metrics.step}  acc={metrics.accuracy:.4f}  "
              f"f1={metrics.macro_f1:.4f}  ece={metrics.calibration_error:.4f}")

    def on_train_end(self, metrics):
        elapsed = time.time() - self._t0
        print(f"[Trainer] Done. acc={metrics.accuracy:.4f}  "
              f"f1={metrics.macro_f1:.4f}  time={elapsed:.1f}s")


class CheckpointCallback(TrainerCallback):
    """Saves model checkpoints."""

    def __init__(self, save_dir: str, monitor: str = 'accuracy', mode: str = 'max'):
        import os

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir  = save_dir
        self.monitor   = monitor
        self.mode      = mode
        self.best_val  = -math.inf if mode == 'max' else math.inf
        self.best_path : str | None = None

    def on_evaluate(self, metrics: EvalMetrics):
        val = getattr(metrics, self.monitor, 0.0)
        improved = (self.mode == 'max' and val > self.best_val) or \
                   (self.mode == 'min' and val < self.best_val)
        if improved:
            self.best_val = val
            import os
            path = os.path.join(self.save_dir, f"best_{self.monitor}.json")
            # Notify via on_save — actual saving handled by Trainer
            self.best_path = path


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Main training engine. Wraps CyphaDIF / RFFRegressor train_step calls
    with evaluation, callbacks, and early stopping.

    Usage:
        trainer = Trainer()
        trainer.add_callback(MetricsCallback())
        trainer.add_callback(ProgressCallback())
        trainer.fit(train_ds, val_ds, config)
    """

    def __init__(self):
        self._callbacks   : list[TrainerCallback] = []
        self._stop_flag   = threading.Event()
        self._model       = None
        self._preprocessor : Preprocessor | None = None
        self._step        = 0
        self._epoch       = 0
        self._best_metric = -math.inf
        self._patience_count = 0

    def add_callback(self, cb: TrainerCallback) -> Trainer:
        self._callbacks.append(cb)
        return self

    def stop(self):
        """Request early stop (thread-safe)."""
        self._stop_flag.set()

    def _emit(self, event: str, *args, **kwargs):
        for cb in self._callbacks:
            getattr(cb, event)(*args, **kwargs)

    # ── Model construction ───────────────────────────────────────────────────

    def _build_model(self, config: TrainerConfig, d_in: int):
        """Build a fresh model from config."""
        import Cypha

        if config.model_type == 'CyphaDIF':
            if config.encoder_type == 'RFFEncoder':
                enc = Cypha.RFFEncoder(d_in, D=config.rff_D, gamma=config.rff_gamma,
                                       seed=config.seed)
            else:
                enc = Cypha.VectorEncoder(d_in)
            clf = Cypha.CyphaDIF(
                encoder=enc,
                field_dim=config.field_dim,
                enc_lr=config.enc_lr,
                delta_lr=config.delta_lr,
                world_lr=config.world_lr,
                mdl_lambda=config.mdl_lambda,
                context_win=config.context_win,
                rng=np.random.default_rng(config.seed),
            )
            clf.temperature = config.temperature
            return clf

        elif config.model_type == 'RFFRegressor':
            reg = Cypha.RFFRegressor(D=config.rff_D, seed=config.seed)
            return reg

        elif config.model_type == 'TwoStageDIF':
            reg = Cypha.TwoStageDIFRegressor(K=config.n_experts, D=config.rff_D,
                                              seed=config.seed)
            return reg

        elif config.model_type == 'MKE':
            # Real model is built in ``_fit_mke`` via ``MKERegressor.from_data``.
            return None

        else:
            raise ValueError(f"Unknown model_type {config.model_type!r}")

    # ── Preprocessing ────────────────────────────────────────────────────────

    def _build_preprocessor(self, config: TrainerConfig,
                              train_ds: CyphaDataset) -> Preprocessor:
        pre = Preprocessor(
            scale=config.preprocess_scale,
            pca_dim=config.preprocess_pca,
            seed=config.seed,
        )
        pre.fit(train_ds.X)
        train_ds.preprocessor = pre
        return pre

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self,
            train_ds   : CyphaDataset,
            val_ds     : CyphaDataset | None = None,
            config     : TrainerConfig | None = None) -> Trainer:
        """
        Train on train_ds, evaluate on val_ds periodically.
        Returns self for chaining.
        """
        if config is None:
            config = TrainerConfig()
        self._stop_flag.clear()
        self._step = 0
        self._epoch = 0
        self._best_metric = -math.inf
        self._patience_count = 0

        # Preprocessor
        self._preprocessor = self._build_preprocessor(config, train_ds)
        X_tr_pp = self._preprocessor.transform(train_ds.X)
        d_in = X_tr_pp.shape[1]

        # Model
        self._model = self._build_model(config, d_in)

        # Special case: batch-mode regressors
        if config.model_type in ('RFFRegressor', 'TwoStageDIF'):
            return self._fit_batch_regressor(X_tr_pp, train_ds, val_ds, config)

        if config.model_type == 'MKE':
            return self._fit_mke(X_tr_pp, train_ds, val_ds, config)

        self._emit('on_train_begin', config)

        # Online training loop
        for epoch in range(config.n_epochs):
            self._epoch = epoch
            self._emit('on_epoch_begin', epoch)

            for x, label in zip(X_tr_pp[np.random.default_rng(config.seed + epoch)
                                         .permutation(len(X_tr_pp))],
                                 train_ds.y[np.random.default_rng(config.seed + epoch)
                                            .permutation(len(train_ds.y))], strict=False):
                if self._stop_flag.is_set():
                    break

                if config.gh_protect:
                    loss, _, _, _ = self._model.gh_train_step(x, str(label), 1.0, 1.0)
                    correct = (self._model.infer(x)[0] == str(label))
                else:
                    loss = self._model.train_step(x, str(label))
                    correct = (self._model.infer(x)[0] == str(label))

                self._emit('on_step', self._step, float(loss), str(label), correct)
                self._step += 1

                # Periodic evaluation
                if val_ds is not None and self._step % config.eval_every_n == 0:
                    metrics = self.evaluate(val_ds, config)
                    metrics.step = self._step
                    metrics.epoch = epoch
                    self._emit('on_evaluate', metrics)

                    # Early stopping
                    if config.early_stopping:
                        if metrics.accuracy > self._best_metric:
                            self._best_metric = metrics.accuracy
                            self._patience_count = 0
                        else:
                            self._patience_count += 1
                            if self._patience_count >= config.early_stopping_patience:
                                break

            if self._stop_flag.is_set():
                break

            # End-of-epoch evaluation
            if val_ds is not None:
                metrics = self.evaluate(val_ds, config)
                metrics.step = self._step
                metrics.epoch = epoch
                self._emit('on_epoch_end', epoch, metrics)

        # Final metrics
        final_metrics = EvalMetrics(step=self._step, epoch=self._epoch,
                                     timestamp=time.time())
        if val_ds is not None:
            final_metrics = self.evaluate(val_ds, config)
        self._emit('on_train_end', final_metrics)
        return self

    def _fit_batch_regressor(self, X_pp, train_ds, val_ds, config):
        self._emit('on_train_begin', config)
        y_tr = train_ds.y.astype(float)

        if config.model_type == 'RFFRegressor':
            self._model.fit(X_pp, y_tr)
        else:  # TwoStageDIF
            self._model.fit(X_pp, y_tr)

        self._emit('on_step', 1, 0.0, 'regression', True)
        final_metrics = EvalMetrics(step=1, epoch=0, timestamp=time.time())

        if val_ds is not None:
            X_val = self._preprocessor.transform(val_ds.X)
            y_val = val_ds.y.astype(float)
            y_pred = self._model.predict(X_val)
            ss_res = float(np.sum((y_val - y_pred)**2))
            ss_tot = float(np.sum((y_val - y_val.mean())**2))
            final_metrics.r2_score = 1.0 - ss_res / max(ss_tot, 1e-8)
            final_metrics.rmse = float(np.sqrt(np.mean((y_val - y_pred)**2)))
            final_metrics.mae  = float(np.mean(np.abs(y_val - y_pred)))
            self._emit('on_evaluate', final_metrics)

        self._emit('on_train_end', final_metrics)
        return self

    def _fit_mke(self, X_pp, train_ds, val_ds, config):
        import Cypha
        self._emit('on_train_begin', config)
        y_tr = train_ds.y.astype(float)
        self._model = Cypha.MKERegressor.from_data(
            X_pp, y_seed=y_tr, K=config.n_experts, D=config.rff_D,
            rng_seed=config.seed, auto_ard=config.auto_ard
        )
        self._model.forgetting_factor = config.forgetting_factor

        rng = np.random.default_rng(config.seed)
        for i in rng.permutation(len(X_pp)):
            if self._stop_flag.is_set(): break
            loss = self._model.train_step(X_pp[i], float(y_tr[i]))
            self._emit('on_step', self._step, float(loss), 'regression', True)
            self._step += 1

        final_metrics = EvalMetrics(step=self._step, epoch=0, timestamp=time.time())
        if val_ds is not None:
            X_val = self._preprocessor.transform(val_ds.X)
            y_val = val_ds.y.astype(float)
            y_pred, _ = self._model.predict_batch(X_val)
            ss_res = float(np.sum((y_val - y_pred)**2))
            ss_tot = float(np.sum((y_val - y_val.mean())**2))
            final_metrics.r2_score = 1.0 - ss_res / max(ss_tot, 1e-8)
            self._emit('on_evaluate', final_metrics)
        self._emit('on_train_end', final_metrics)
        return self

    # ── Evaluate ─────────────────────────────────────────────────────────────

    def evaluate(self, ds: CyphaDataset,
                 config: TrainerConfig | None = None) -> EvalMetrics:
        """Evaluate model on dataset. Returns EvalMetrics."""
        if self._model is None:
            return EvalMetrics()

        X_pp = self._preprocessor.transform(ds.X) if self._preprocessor else ds.X

        if ds.task == 'regression':
            y_val = ds.y.astype(float)
            if hasattr(self._model, 'predict_batch'):
                y_pred, _ = self._model.predict_batch(X_pp)
            else:
                y_pred = self._model.predict(X_pp)
            ss_res = float(np.sum((y_val - y_pred)**2))
            ss_tot = float(np.sum((y_val - y_val.mean())**2))
            m = EvalMetrics(timestamp=time.time())
            m.r2_score = 1.0 - ss_res / max(ss_tot, 1e-8)
            m.rmse = float(np.sqrt(np.mean((y_val - y_pred)**2)))
            m.mae  = float(np.mean(np.abs(y_val - y_pred)))
            return m

        # Classification
        y_true, y_pred, y_conf = [], [], []
        for x, label in zip(X_pp, ds.y, strict=False):
            pred, conf = self._model.infer(x)
            y_true.append(str(label))
            y_pred.append(pred)
            y_conf.append(conf)

        m = _compute_classification_metrics(y_true, y_pred, y_conf)

        # World prior health
        try:
            with self._model.memory._lock:
                m.world_mu_norm = float(np.linalg.norm(self._model.memory.world.mu))
                m.n_classes = len(self._model.memory._classes)
        except Exception:
            pass

        return m

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def model(self):
        return self._model

    @property
    def preprocessor(self) -> Preprocessor | None:
        return self._preprocessor

    @property
    def step(self) -> int:
        return self._step


# ─────────────────────────────────────────────────────────────────────────────
# HyperparameterSearch
# ─────────────────────────────────────────────────────────────────────────────

class HyperparameterSearch:
    """
    Grid or random search over TrainerConfig fields.

    search = GridSearch({'world_lr': [0.01, 0.02, 0.05],
                          'delta_lr': [0.05, 0.08, 0.1]})
    results = search.run(train_ds, val_ds, base_config)
    """

    def __init__(self, param_grid: dict[str, list[Any]],
                 n_jobs: int = 1,
                 verbose: bool = True):
        self.param_grid = param_grid
        self.n_jobs     = n_jobs
        self.verbose    = verbose
        self.results_   : list[dict] = []

    def _generate_configs(self, base: TrainerConfig) -> list[TrainerConfig]:
        import copy
        import itertools
        keys = list(self.param_grid.keys())
        vals = list(self.param_grid.values())
        configs = []
        for combo in itertools.product(*vals):
            cfg = copy.deepcopy(base)
            for k, v in zip(keys, combo, strict=False):
                setattr(cfg, k, v)
            configs.append((cfg, dict(zip(keys, combo, strict=False))))
        return configs

    def run(self, train_ds: CyphaDataset, val_ds: CyphaDataset,
            base_config: TrainerConfig | None = None) -> list[dict]:
        if base_config is None:
            base_config = TrainerConfig()

        configs_with_params = self._generate_configs(base_config)
        self.results_ = []

        for i, (cfg, params) in enumerate(configs_with_params):
            if self.verbose:
                print(f"[Search] Run {i+1}/{len(configs_with_params)}: {params}")
            try:
                metrics_cb = MetricsCallback()
                trainer = Trainer()
                trainer.add_callback(metrics_cb)
                trainer.fit(train_ds, val_ds, cfg)
                val_metrics = trainer.evaluate(val_ds, cfg)
                result = {
                    'params'   : params,
                    'config'   : cfg,
                    'metrics'  : val_metrics,
                    'accuracy' : val_metrics.accuracy,
                    'r2'       : val_metrics.r2_score,
                }
                self.results_.append(result)
                if self.verbose:
                    print(f"  acc={val_metrics.accuracy:.4f}  f1={val_metrics.macro_f1:.4f}")
            except Exception as e:
                if self.verbose:
                    print(f"  FAILED: {e}")
                self.results_.append({'params': params, 'error': str(e)})

        # Sort by accuracy (desc)
        self.results_.sort(key=lambda r: r.get('accuracy', -1), reverse=True)
        return self.results_

    @property
    def best_config(self) -> TrainerConfig | None:
        if not self.results_:
            return None
        for r in self.results_:
            if 'config' in r:
                return r['config']
        return None

    @property
    def best_params(self) -> dict | None:
        if not self.results_:
            return None
        for r in self.results_:
            if 'params' in r and 'error' not in r:
                return r['params']
        return None


class RandomSearch(HyperparameterSearch):
    """
    Random search over continuous/discrete distributions.

    search = RandomSearch({'world_lr': ('loguniform', 0.001, 0.1),
                            'delta_lr': ('uniform', 0.02, 0.15),
                            'n_experts': ('randint', 4, 16)},
                           n_iter=20)
    """

    def __init__(self, param_distributions: dict[str, tuple],
                 n_iter: int = 20,
                 seed: int = 42,
                 verbose: bool = True):
        super().__init__(param_grid={}, verbose=verbose)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.seed   = seed

    def _sample(self, rng: np.random.Generator) -> dict[str, Any]:
        sample = {}
        for k, spec in self.param_distributions.items():
            dist = spec[0]
            if dist == 'uniform':
                sample[k] = float(rng.uniform(spec[1], spec[2]))
            elif dist == 'loguniform':
                lo, hi = math.log(spec[1]), math.log(spec[2])
                sample[k] = float(math.exp(rng.uniform(lo, hi)))
            elif dist == 'randint':
                sample[k] = int(rng.integers(spec[1], spec[2]))
            elif dist == 'choice':
                idx = rng.integers(0, len(spec[1]))
                sample[k] = spec[1][idx]
            else:
                raise ValueError(f"Unknown distribution {dist!r}")
        return sample

    def _generate_configs(self, base: TrainerConfig):
        import copy
        rng = np.random.default_rng(self.seed)
        results = []
        for _ in range(self.n_iter):
            params = self._sample(rng)
            cfg = copy.deepcopy(base)
            for k, v in params.items():
                setattr(cfg, k, v)
            results.append((cfg, params))
        return results

# Alias for convenience
GridSearch = HyperparameterSearch
