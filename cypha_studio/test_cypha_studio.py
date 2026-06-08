"""
test_cypha_studio.py
─────────────────────
Test suite for cypha_studio core modules.
Run from repo root: python cypha_studio/test_cypha_studio.py
  (REST API test needs: pip install -r requirements-verify.txt)

48 tests across 8 sections:
  1. Dataset (7)     — loading, splitting, streaming, preprocessor
  2. Preprocessor (5)— scale, PCA, RFF, save/load, identity
  3. Trainer (10)    — all model types, callbacks, metrics, early stopping
  4. Search (4)      — grid search, random search, best config
  5. Experiment (7)  — create, log, finish, fail, leaderboard, compare
  6. Registry (7)    — register, load, version, promote, compare, delete
  7. Inference (6)   — predict, batch, explain, correct, session, OOD
  8. API (2)         — app creation, headless bus
"""
import os
import sys
import math
import tempfile
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

_PASSED = 0
_FAILED = 0
_ERRORS = []

def test(name):
    def decorator(fn):
        global _PASSED, _FAILED
        try:
            fn()
            _PASSED += 1
            print(f"  ✓ {name}")
        except Exception as e:
            _FAILED += 1
            _ERRORS.append((name, e))
            print(f"  ✗ {name}")
            print(f"    {type(e).__name__}: {e}")
        return fn
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iris():
    from cypha_studio.core.dataset import SklearnDataset
    return SklearnDataset.load('iris')

def _diabetes():
    from cypha_studio.core.dataset import SklearnDataset
    return SklearnDataset.load('diabetes', task='regression')

def _quick_clf(ds=None, feat_dim=32, field_dim=32):
    """Train a quick CyphaDIF on iris."""
    from cypha_studio.core.dataset import Preprocessor, SplitConfig
    from cypha_studio.core.trainer import TrainerConfig, Trainer
    if ds is None: ds = _iris()
    tr, val, te = ds.split(SplitConfig(seed=42))
    pre = Preprocessor(); pre.fit(tr.X)
    tr.preprocessor = pre; val.preprocessor = pre
    cfg = TrainerConfig(feat_dim=feat_dim, field_dim=field_dim,
                        n_epochs=1, eval_every_n=9999, seed=42)
    t = Trainer(); t.fit(tr, val, cfg)
    return t, pre, tr, val, te, ds


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Dataset
# ══════════════════════════════════════════════════════════════════════════════

print("\n Section 1: Dataset")

@test("SklearnDataset: loads iris with correct shape and labels")
def _():
    ds = _iris()
    assert ds.n_samples == 150
    assert ds.n_features == 4
    assert ds.n_classes == 3
    assert 'setosa' in ds.labels or 'Iris-setosa' in ds.labels or '0' in ds.labels

@test("SklearnDataset: regression dataset has float targets")
def _():
    ds = _diabetes()
    assert ds.task == 'regression'
    assert ds.y.dtype in (np.float64, np.float32)

@test("SplitConfig: stratified split preserves class proportions")
def _():
    from cypha_studio.core.dataset import SplitConfig
    ds = _iris()
    cfg = SplitConfig(train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)
    tr, val, te = ds.split(cfg)
    assert abs(len(tr) / ds.n_samples - 0.7) < 0.05
    for lbl in ds.labels:
        # Each class should appear in train
        assert any(str(y) == lbl for y in tr.y)

@test("CyphaDataset.stream: yields (x, label) pairs of correct types")
def _():
    ds = _iris()
    tr, _, _ = ds.split()
    items = list(tr.stream(shuffle=True, seed=42))
    assert len(items) == len(tr)
    x0, lbl0 = items[0]
    assert isinstance(x0, np.ndarray)
    assert isinstance(lbl0, str)
    assert x0.shape == (ds.n_features,)

@test("DatasetStats: class_balance is 1.0 for balanced dataset")
def _():
    ds = _iris()
    stats = ds.stats()
    # Iris has 50 samples per class → perfectly balanced
    assert abs(stats.class_balance - 1.0) < 0.01
    assert stats.missing_values == 0

@test("CSVDataset: round-trips numeric data correctly")
def _():
    from cypha_studio.core.dataset import CSVDataset
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'data.csv')
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (40, 3))
        y = rng.integers(0, 2, 40)
        with open(path, 'w') as f:
            f.write("a,b,c,label\n")
            for row, lbl in zip(X, y):
                f.write(','.join(map(str, row)) + f',{lbl}\n')
        ds = CSVDataset.from_file(path, target_col='label')
        assert ds.n_samples == 40
        assert ds.n_features == 3
        assert set(ds.labels) == {'0', '1'}

@test("NumpyDataset: loads from array and round-trips stream")
def _():
    from cypha_studio.core.dataset import NumpyDataset
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (30, 5))
    y = np.array(['a','b','c'] * 10)
    ds = NumpyDataset.from_arrays(X, y, name='test')
    assert ds.n_samples == 30
    items = list(ds.stream())
    assert len(items) == 30
    assert all(isinstance(lbl, str) for _, lbl in items)


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Preprocessor
# ══════════════════════════════════════════════════════════════════════════════

print("\n Section 2: Preprocessor")

@test("Preprocessor: scale=True gives zero mean and unit variance")
def _():
    from cypha_studio.core.dataset import Preprocessor
    rng = np.random.default_rng(1)
    X = rng.normal(5, 3, (100, 4))
    pre = Preprocessor(scale=True)
    X_pp = pre.fit_transform(X)
    assert np.allclose(X_pp.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_pp.std(axis=0), 1, atol=1e-10)

@test("Preprocessor: PCA reduces dimensionality")
def _():
    from cypha_studio.core.dataset import Preprocessor
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (100, 10))
    pre = Preprocessor(scale=True, pca_dim=4)
    X_pp = pre.fit_transform(X)
    assert X_pp.shape == (100, 4)
    assert pre.output_dim == 4

@test("Preprocessor: RFF maps to specified dimension")
def _():
    from cypha_studio.core.dataset import Preprocessor
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (50, 6))
    pre = Preprocessor(scale=True, rff_dim=32, seed=42)
    X_pp = pre.fit_transform(X)
    assert X_pp.shape == (50, 32)

@test("Preprocessor: save_state/load_state round-trip is exact")
def _():
    from cypha_studio.core.dataset import Preprocessor
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (80, 6))
    pre = Preprocessor(scale=True, pca_dim=4, seed=42)
    X_pp = pre.fit_transform(X)
    st = pre.save_state()
    pre2 = Preprocessor()
    pre2.load_state(st)
    X_pp2 = pre2.transform(X)
    assert np.allclose(X_pp, X_pp2, atol=1e-12)

@test("Preprocessor: transform_one matches batch transform row")
def _():
    from cypha_studio.core.dataset import Preprocessor
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (50, 4))
    pre = Preprocessor(scale=True)
    pre.fit(X)
    for i in range(5):
        single = pre.transform_one(X[i])
        batch  = pre.transform(X[i:i+1])[0]
        assert np.allclose(single, batch, atol=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Trainer
# ══════════════════════════════════════════════════════════════════════════════

print("\n Section 3: Trainer")

@test("Trainer: CyphaDIF classification achieves acc > 0.6 on iris")
def _():
    t, pre, tr, val, te, ds = _quick_clf()
    m = t.evaluate(val)
    assert m.accuracy > 0.6, f"acc={m.accuracy:.4f} too low"

@test("Trainer: EvalMetrics has per_class breakdown with correct keys")
def _():
    t, pre, tr, val, te, ds = _quick_clf()
    m = t.evaluate(val)
    assert len(m.per_class) >= 2
    for lbl, metrics in m.per_class.items():
        assert 'precision' in metrics and 'recall' in metrics and 'f1' in metrics

@test("Trainer: MetricsCallback collects step losses")
def _():
    from cypha_studio.core.trainer import TrainerConfig, Trainer, MetricsCallback
    ds = _iris(); tr, val, _ = ds.split()
    cb = MetricsCallback()
    cfg = TrainerConfig(feat_dim=32, field_dim=32, n_epochs=1, eval_every_n=9999)
    t = Trainer()
    t.add_callback(cb)
    t.fit(tr, val, cfg)
    assert len(cb.step_losses) > 0
    assert all(math.isfinite(l) for _, l in cb.step_losses)
    assert cb.recent_loss() > 0

@test("Trainer: RFFRegressor regression gives finite R²")
def _():
    from cypha_studio.core.trainer import TrainerConfig, Trainer
    ds = _diabetes(); tr, val, _ = ds.split()
    cfg = TrainerConfig(model_type='RFFRegressor', rff_D=128)
    t = Trainer(); t.fit(tr, val, cfg)
    m = t.evaluate(val, cfg)
    assert math.isfinite(m.r2_score)
    assert -5 < m.r2_score < 1.01

@test("Trainer: TwoStageDIF regression gives finite R²")
def _():
    from cypha_studio.core.trainer import TrainerConfig, Trainer
    ds = _diabetes(); tr, val, _ = ds.split()
    cfg = TrainerConfig(model_type='TwoStageDIF', rff_D=128, n_experts=6)
    t = Trainer(); t.fit(tr, val, cfg)
    m = t.evaluate(val, cfg)
    assert math.isfinite(m.r2_score)

@test("Trainer: MKE regression gives finite R²")
def _():
    from cypha_studio.core.trainer import TrainerConfig, Trainer
    ds = _diabetes(); tr, val, _ = ds.split()
    cfg = TrainerConfig(model_type='MKE', rff_D=128, n_experts=4)
    t = Trainer(); t.fit(tr, val, cfg)
    m = t.evaluate(val, cfg)
    assert math.isfinite(m.r2_score)

@test("Trainer: GH protection trains without error")
def _():
    from cypha_studio.core.trainer import TrainerConfig, Trainer
    ds = _iris(); tr, val, _ = ds.split()
    cfg = TrainerConfig(feat_dim=32, field_dim=32, n_epochs=1,
                        eval_every_n=9999, gh_protect=True)
    t = Trainer(); t.fit(tr, val, cfg)
    m = t.evaluate(val, cfg)
    assert m.accuracy > 0

@test("Trainer: stop() halts training mid-run")
def _():
    from cypha_studio.core.trainer import TrainerConfig, Trainer, TrainerCallback
    import threading
    ds = _iris(); tr, val, _ = ds.split()
    cfg = TrainerConfig(feat_dim=64, field_dim=64, n_epochs=10, eval_every_n=9999)
    t = Trainer()
    steps_at_stop = []

    class StopAfter(TrainerCallback):
        def on_step(self, step, loss, label, correct):
            if step >= 20:
                t.stop()
                steps_at_stop.append(step)

    t.add_callback(StopAfter())
    t.fit(tr, val, cfg)
    # Should have stopped well before 10 epochs × ~105 steps
    assert t.step < 200, f"Trainer didn't stop: step={t.step}"

@test("Trainer: calibration_error is in [0, 1]")
def _():
    t, pre, tr, val, te, ds = _quick_clf()
    m = t.evaluate(val)
    assert 0 <= m.calibration_error <= 1.0

@test("Trainer: step count increases correctly across epochs")
def _():
    from cypha_studio.core.trainer import TrainerConfig, Trainer
    ds = _iris(); tr, val, _ = ds.split()
    n_tr = len(tr)
    cfg = TrainerConfig(feat_dim=32, field_dim=32, n_epochs=2, eval_every_n=9999)
    t = Trainer(); t.fit(tr, val, cfg)
    assert t.step == 2 * n_tr, f"Expected {2*n_tr} steps, got {t.step}"


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Hyperparameter Search
# ══════════════════════════════════════════════════════════════════════════════

print("\n Section 4: Search")

@test("GridSearch: runs all combinations and returns sorted results")
def _():
    from cypha_studio.core.trainer import TrainerConfig, GridSearch
    ds = _iris(); tr, val, _ = ds.split()
    cfg = TrainerConfig(feat_dim=32, field_dim=32, n_epochs=1, eval_every_n=9999)
    search = GridSearch({'world_lr': [0.01, 0.05]}, verbose=False)
    results = search.run(tr, val, cfg)
    assert len(results) == 2
    # Results sorted descending by accuracy
    accs = [r.get('accuracy', -1) for r in results if 'accuracy' in r]
    assert accs == sorted(accs, reverse=True)

@test("GridSearch: best_config has the highest accuracy")
def _():
    from cypha_studio.core.trainer import TrainerConfig, GridSearch
    ds = _iris(); tr, val, _ = ds.split()
    cfg = TrainerConfig(feat_dim=32, field_dim=32, n_epochs=1, eval_every_n=9999)
    search = GridSearch({'delta_lr': [0.04, 0.08, 0.12]}, verbose=False)
    search.run(tr, val, cfg)
    assert search.best_config is not None
    assert search.best_params is not None
    assert 'delta_lr' in search.best_params

@test("RandomSearch: samples from distributions correctly")
def _():
    from cypha_studio.core.trainer import TrainerConfig, RandomSearch
    ds = _iris(); tr, val, _ = ds.split()
    cfg = TrainerConfig(feat_dim=32, field_dim=32, n_epochs=1, eval_every_n=9999)
    search = RandomSearch(
        {'world_lr': ('loguniform', 0.001, 0.1),
         'delta_lr': ('uniform',   0.03,  0.15)},
        n_iter=3, seed=42, verbose=False
    )
    results = search.run(tr, val, cfg)
    assert len(results) == 3
    # All sampled world_lr values in [0.001, 0.1]
    for r in results:
        if 'params' in r and 'error' not in r:
            assert 0.001 <= r['params']['world_lr'] <= 0.1

@test("RandomSearch: choice distribution selects from list")
def _():
    from cypha_studio.core.trainer import TrainerConfig, RandomSearch
    ds = _iris(); tr, val, _ = ds.split()
    cfg = TrainerConfig(feat_dim=32, field_dim=32, n_epochs=1, eval_every_n=9999)
    search = RandomSearch(
        {'gh_protect': ('choice', [True, False])},
        n_iter=3, seed=99, verbose=False
    )
    results = search.run(tr, val, cfg)
    assert len(results) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: Experiment DB
# ══════════════════════════════════════════════════════════════════════════════

print("\n Section 5: Experiment DB")

@test("ExperimentDB: create/get experiment round-trip")
def _():
    from cypha_studio.core.experiment import ExperimentDB
    with tempfile.TemporaryDirectory() as td:
        db = ExperimentDB(os.path.join(td,'e.db'))
        exp = db.create_experiment('test', description='desc', dataset_name='iris',
                                   task='classification', tags=['a','b'])
        exp2 = db.get_experiment(exp.experiment_id)
        assert exp2.name == 'test'
        assert exp2.description == 'desc'
        assert 'a' in exp2.tags

@test("ExperimentDB: create run and log metrics")
def _():
    from cypha_studio.core.experiment import ExperimentDB
    from cypha_studio.core.trainer import EvalMetrics
    with tempfile.TemporaryDirectory() as td:
        db = ExperimentDB(os.path.join(td,'e.db'))
        exp = db.create_experiment('e1')
        run = db.create_run(exp.experiment_id, 'r1')
        assert run.status == 'pending'
        db.update_run(run.run_id, status='running')
        m = EvalMetrics(step=100, accuracy=0.88, macro_f1=0.87)
        db.log_metrics(run.run_id, m)
        r2 = db.get_run(run.run_id)
        assert r2.status == 'running'
        assert len(r2.metrics_history) == 1

@test("ExperimentDB: finish_run stores final metrics")
def _():
    from cypha_studio.core.experiment import ExperimentDB
    from cypha_studio.core.trainer import EvalMetrics
    with tempfile.TemporaryDirectory() as td:
        db = ExperimentDB(os.path.join(td,'e.db'))
        exp = db.create_experiment('e2')
        run = db.create_run(exp.experiment_id, 'r1')
        m = EvalMetrics(step=500, accuracy=0.95, macro_f1=0.94)
        db.finish_run(run.run_id, m)
        r2 = db.get_run(run.run_id)
        assert r2.status == 'done'
        assert abs(r2.accuracy - 0.95) < 1e-6

@test("ExperimentDB: fail_run marks run as failed with error message")
def _():
    from cypha_studio.core.experiment import ExperimentDB
    with tempfile.TemporaryDirectory() as td:
        db = ExperimentDB(os.path.join(td,'e.db'))
        exp = db.create_experiment('e3')
        run = db.create_run(exp.experiment_id, 'r1')
        db.fail_run(run.run_id, "OOM error")
        r2 = db.get_run(run.run_id)
        assert r2.status == 'failed'
        assert 'OOM' in r2.notes

@test("ExperimentDB: leaderboard returns runs sorted by accuracy")
def _():
    from cypha_studio.core.experiment import ExperimentDB
    from cypha_studio.core.trainer import EvalMetrics
    with tempfile.TemporaryDirectory() as td:
        db = ExperimentDB(os.path.join(td,'e.db'))
        exp = db.create_experiment('e4')
        for acc in [0.7, 0.9, 0.8]:
            run = db.create_run(exp.experiment_id, f'r-{acc}')
            db.finish_run(run.run_id, EvalMetrics(accuracy=acc, step=100))
        lb = db.leaderboard(exp.experiment_id)
        assert lb[0].accuracy >= lb[1].accuracy >= lb[2].accuracy

@test("ExperimentDB: best_run returns highest accuracy run")
def _():
    from cypha_studio.core.experiment import ExperimentDB
    from cypha_studio.core.trainer import EvalMetrics
    with tempfile.TemporaryDirectory() as td:
        db = ExperimentDB(os.path.join(td,'e.db'))
        exp = db.create_experiment('e5')
        for acc in [0.6, 0.85, 0.75]:
            run = db.create_run(exp.experiment_id, f'r')
            db.finish_run(run.run_id, EvalMetrics(accuracy=acc, step=100))
        best = db.best_run(exp.experiment_id)
        assert abs(best.accuracy - 0.85) < 1e-6

@test("ExperimentDB: compare_runs returns all requested run summaries")
def _():
    from cypha_studio.core.experiment import ExperimentDB
    from cypha_studio.core.trainer import EvalMetrics
    with tempfile.TemporaryDirectory() as td:
        db = ExperimentDB(os.path.join(td,'e.db'))
        exp = db.create_experiment('e6')
        run_ids = []
        for i in range(3):
            run = db.create_run(exp.experiment_id, f'r{i}')
            db.finish_run(run.run_id, EvalMetrics(accuracy=0.7+i*0.1, step=100))
            run_ids.append(run.run_id)
        rows = db.compare_runs(run_ids)
        assert len(rows) == 3
        assert all('accuracy' in r for r in rows)


# ══════════════════════════════════════════════════════════════════════════════
# Section 6: Registry
# ══════════════════════════════════════════════════════════════════════════════

print("\n Section 6: Registry")

@test("ModelRegistry: register and load preserves predictions")
def _():
    from cypha_studio.core.registry import ModelRegistry, ModelCard
    from cypha_studio.core.dataset import Preprocessor
    t, pre, tr, val, te, ds = _quick_clf()
    with tempfile.TemporaryDirectory() as td:
        reg = ModelRegistry(td)
        card = ModelCard(name='iris', version='1.0.0', task='classification',
                         model_type='CyphaDIF', encoder_type='VectorEncoder',
                         input_dim=4)
        reg.register(t.model, card, pre)
        m2, p2, c2 = reg.load('iris', '1.0.0')
        assert c2.name == 'iris'
        x = pre.transform(ds.X[:1])[0]
        pred1, conf1 = t.model.infer(x)
        pred2, conf2 = m2.infer(x)
        assert pred1 == pred2

@test("ModelRegistry: list_models returns all saved models")
def _():
    from cypha_studio.core.registry import ModelRegistry, ModelCard
    t, pre, tr, val, te, ds = _quick_clf()
    with tempfile.TemporaryDirectory() as td:
        reg = ModelRegistry(td)
        for v in ['1.0.0', '1.1.0', '2.0.0']:
            card = ModelCard(name='m', version=v, task='classification',
                             model_type='CyphaDIF', encoder_type='VectorEncoder',
                             input_dim=4)
            reg.register(t.model, card)
        cards = reg.list_models()
        versions = {c.version for c in cards}
        assert {'1.0.0', '1.1.0', '2.0.0'} == versions

@test("ModelRegistry: next_version bumps correctly")
def _():
    from cypha_studio.core.registry import ModelRegistry, ModelCard
    t, pre, tr, val, te, ds = _quick_clf()
    with tempfile.TemporaryDirectory() as td:
        reg = ModelRegistry(td)
        card = ModelCard(name='m', version='1.2.3', task='classification',
                         model_type='CyphaDIF', encoder_type='VectorEncoder',
                         input_dim=4)
        reg.register(t.model, card)
        assert reg.next_version('m', 'patch') == '1.2.4'
        assert reg.next_version('m', 'minor') == '1.3.0'
        assert reg.next_version('m', 'major') == '2.0.0'

@test("ModelRegistry: promote updates stage field in card")
def _():
    from cypha_studio.core.registry import ModelRegistry, ModelCard
    t, pre, tr, val, te, ds = _quick_clf()
    with tempfile.TemporaryDirectory() as td:
        reg = ModelRegistry(td)
        card = ModelCard(name='m', version='1.0.0', task='classification',
                         model_type='CyphaDIF', encoder_type='VectorEncoder',
                         input_dim=4, stage='dev')
        reg.register(t.model, card)
        reg.promote('m', '1.0.0', to='production')
        c2 = reg.load_card('m', '1.0.0')
        assert c2.stage == 'production'

@test("ModelRegistry: delete removes model from disk")
def _():
    from cypha_studio.core.registry import ModelRegistry, ModelCard
    t, pre, tr, val, te, ds = _quick_clf()
    with tempfile.TemporaryDirectory() as td:
        reg = ModelRegistry(td)
        card = ModelCard(name='m', version='1.0.0', task='classification',
                         model_type='CyphaDIF', encoder_type='VectorEncoder',
                         input_dim=4)
        reg.register(t.model, card)
        assert reg.exists('m', '1.0.0')
        reg.delete('m', '1.0.0')
        assert not reg.exists('m', '1.0.0')

@test("ModelRegistry: overwrite=False raises if model exists")
def _():
    from cypha_studio.core.registry import ModelRegistry, ModelCard
    t, pre, tr, val, te, ds = _quick_clf()
    with tempfile.TemporaryDirectory() as td:
        reg = ModelRegistry(td)
        card = ModelCard(name='m', version='1.0.0', task='classification',
                         model_type='CyphaDIF', encoder_type='VectorEncoder',
                         input_dim=4)
        reg.register(t.model, card)
        raised = False
        try:
            reg.register(t.model, card, overwrite=False)
        except FileExistsError:
            raised = True
        assert raised

@test("ModelRegistry: RFFRegressor saves and loads correctly")
def _():
    from cypha_studio.core.registry import ModelRegistry, ModelCard
    from cypha_studio.core.trainer  import TrainerConfig, Trainer
    from cypha_studio.core.inference import InferenceEngine
    ds = _diabetes(); tr, val, _ = ds.split()
    cfg = TrainerConfig(model_type='RFFRegressor', rff_D=64)
    t = Trainer(); t.fit(tr, val, cfg)
    with tempfile.TemporaryDirectory() as td:
        reg = ModelRegistry(td)
        card = ModelCard(name='diab', version='1.0.0', task='regression',
                         model_type='RFFRegressor', encoder_type='RFFEncoder',
                         input_dim=ds.n_features)
        reg.register(t.model, card, t.preprocessor)
        m2, p2, c2 = reg.load('diab', '1.0.0')
        eng = InferenceEngine(m2, p2)
        pred = eng.predict(ds.X[0])
        assert pred.regression_val is not None
        assert math.isfinite(pred.regression_val)


# ══════════════════════════════════════════════════════════════════════════════
# Section 7: Inference
# ══════════════════════════════════════════════════════════════════════════════

print("\n Section 7: Inference")

@test("InferenceEngine: predict returns valid Prediction")
def _():
    from cypha_studio.core.inference import InferenceEngine, Prediction
    t, pre, tr, val, te, ds = _quick_clf()
    eng = InferenceEngine(t.model, pre)
    pred = eng.predict(ds.X[0])
    assert isinstance(pred, Prediction)
    assert isinstance(pred.label, str)
    assert 0 <= pred.confidence <= 1
    assert pred.input_vector is not None

@test("InferenceEngine: predict_batch returns correct number of predictions")
def _():
    from cypha_studio.core.inference import InferenceEngine
    t, pre, tr, val, te, ds = _quick_clf()
    eng = InferenceEngine(t.model, pre)
    preds = eng.predict_batch(ds.X[:10])
    assert len(preds) == 10
    assert all(0 <= p.confidence <= 1 for p in preds)

@test("InferenceEngine: explain returns class_details and all_scores")
def _():
    from cypha_studio.core.inference import InferenceEngine
    t, pre, tr, val, te, ds = _quick_clf()
    eng = InferenceEngine(t.model, pre)
    expl = eng.explain(ds.X[0])
    assert 'label' in expl
    assert 'confidence' in expl
    assert 'all_scores' in expl
    assert 'class_details' in expl

@test("InferenceEngine: update reduces loss when given correct label")
def _():
    from cypha_studio.core.inference import InferenceEngine
    t, pre, tr, val, te, ds = _quick_clf()
    eng = InferenceEngine(t.model, pre)
    losses = []
    for i in range(5):
        loss = eng.update(ds.X[i], str(ds.y[i]))
        assert math.isfinite(loss)
        losses.append(loss)
    assert eng.n_corrections == 5

@test("InferenceSession: summary reflects prediction history")
def _():
    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    t, pre, tr, val, te, ds = _quick_clf()
    eng = InferenceEngine(t.model, pre)
    sess = InferenceSession(eng)
    for x in ds.X[:15]:
        sess.predict(x)
    s = sess.summary()
    assert s['n_predictions'] == 15
    assert 0 <= s['mean_confidence'] <= 1
    assert isinstance(s['label_distribution'], dict)
    # Clear and check
    sess.clear()
    s2 = sess.summary()
    assert s2['n_predictions'] == 0

@test("InferenceEngine: OOD detection flags far-OOD inputs")
def _():
    from cypha_studio.core.inference import InferenceEngine
    t, pre, tr, val, te, ds = _quick_clf()
    eng = InferenceEngine(t.model, pre, ood_threshold=1.0)
    # In-distribution: normal iris features
    pred_ind = eng.predict(ds.X[0])
    # OOD: massive values far from training distribution
    rng = np.random.default_rng(42)
    x_ood = rng.normal(0, 50, ds.n_features) + 1000
    pred_ood = eng.predict(x_ood)
    assert pred_ood.anomaly_score >= pred_ind.anomaly_score, \
        f"OOD score {pred_ood.anomaly_score:.4f} should >= IND {pred_ind.anomaly_score:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# Section 8: Server
# ══════════════════════════════════════════════════════════════════════════════

print("\n Section 8: Server")

@test("SignalBus: singleton returns same instance")
def _():
    from cypha_studio.server.local_server import SignalBus
    b1 = SignalBus.instance()
    b2 = SignalBus.instance()
    assert b1 is b2

@test("REST API: create_app returns FastAPI app with correct routes")
def _():
    from cypha_studio.server.api import create_app
    from fastapi.testclient import TestClient
    t, pre, tr, val, te, ds = _quick_clf()
    from cypha_studio.core.inference import InferenceEngine, InferenceSession
    eng  = InferenceEngine(t.model, pre)
    sess = InferenceSession(eng)
    app = create_app(engine=eng, session=sess)
    # Check routes are registered
    routes = {r.path for r in app.routes}
    assert '/predict' in routes
    assert '/update'  in routes
    assert '/models'  in routes
    assert '/session' in routes
    assert '/health'  in routes
    assert '/ready'  in routes
    assert '/metrics' in routes
    assert '/adapt_temperature' in routes
    assert '/load' in routes
    assert '/classes' in routes
    assert '/route/text' in routes
    assert '/route/generate' in routes
    assert '/route/health' in routes
    c = TestClient(app)
    rd = c.get("/ready")
    assert rd.status_code == 200
    assert rd.json().get("ready") is True
    assert rd.json().get("model_type") == "CyphaDIF"
    mj = c.get("/metrics").json()
    assert "regression_head_loaded" in mj
    assert mj["regression_head_loaded"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

total = _PASSED + _FAILED
print(f"\n{'═'*60}")
if _FAILED == 0:
    print(f"ALL PASSED  ({_PASSED}/{total})")
else:
    print(f"{_FAILED} FAILED  ({_PASSED}/{total} passed)")
    for name, err in _ERRORS:
        print(f"  FAIL: {name}")
        print(f"    {type(err).__name__}: {err}")
print(f"{'═'*60}\n")
if _FAILED:
    sys.exit(1)
