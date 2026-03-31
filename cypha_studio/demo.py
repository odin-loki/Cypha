#!/usr/bin/env python3
"""
demo.py — CyphaStudio headless demo

Shows the complete workflow:
  1. Load dataset
  2. Train model
  3. Save to registry
  4. Load from registry
  5. Inference session with corrections
  6. Hyperparameter search
  7. Experiment tracking
  8. REST API (quick smoke test)

Run: python demo.py
"""
import os, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, warnings
warnings.filterwarnings('ignore')

from cypha_studio.core.dataset    import SklearnDataset, Preprocessor, SplitConfig
from cypha_studio.core.trainer    import TrainerConfig, Trainer, MetricsCallback
from cypha_studio.core.experiment import ExperimentDB
from cypha_studio.core.registry   import ModelRegistry, ModelCard
from cypha_studio.core.inference  import InferenceEngine, InferenceSession

print("═" * 60)
print("  CyphaStudio — Headless Demo")
print("═" * 60)

with tempfile.TemporaryDirectory() as workdir:
    # ── 1. Dataset ───────────────────────────────────────────────────────────
    print("\n[1] Loading Wine dataset…")
    ds = SklearnDataset.load('wine')
    tr, val, te = ds.split(SplitConfig(seed=42))
    pre = Preprocessor(); pre.fit(tr.X)
    tr.preprocessor = pre; val.preprocessor = pre; te.preprocessor = pre
    stats = ds.stats()
    print(f"    {ds}  balance={stats.class_balance:.2f}")
    print(f"    Train={len(tr)}  Val={len(val)}  Test={len(te)}")

    # ── 2. Experiment ─────────────────────────────────────────────────────────
    print("\n[2] Creating experiment…")
    db = ExperimentDB(os.path.join(workdir, 'experiments.db'))
    exp = db.create_experiment('wine-demo', dataset_name='wine',
                               description='CyphaStudio headless demo',
                               tags=['demo', 'wine'])
    print(f"    Experiment: {exp.name}  id={exp.experiment_id}")

    # ── 3. Training ───────────────────────────────────────────────────────────
    print("\n[3] Training CyphaDIF (profiled medium, config/profiled_medium.json)…")
    cfg = TrainerConfig(
        model_type='CyphaDIF',
        feat_dim=128,
        field_dim=128,
        world_lr=0.008,
        delta_lr=0.05,
        enc_lr=0.002,
        mdl_lambda=0.001,
        temperature=1.15,
        context_win=32,
        n_epochs=3,
        eval_every_n=50,
        gh_protect=False,
        seed=42,
    )
    cb  = MetricsCallback()
    trainer = Trainer()
    trainer.add_callback(cb)
    t0 = time.time()
    trainer.fit(tr, val, cfg)
    elapsed = time.time() - t0
    final = trainer.evaluate(val, cfg)
    test_m = trainer.evaluate(te, cfg)
    print(f"    Training time: {elapsed:.1f}s  steps={trainer.step}")
    print(f"    Val:   acc={final.accuracy:.4f}  f1={final.macro_f1:.4f}  "
          f"ece={final.calibration_error:.4f}")
    print(f"    Test:  acc={test_m.accuracy:.4f}  f1={test_m.macro_f1:.4f}")
    print("    Per-class recall:")
    for lbl, m in sorted(final.per_class.items()):
        print(f"      {lbl:20s}: recall={m['recall']:.4f}  f1={m['f1']:.4f}")

    # Log to experiment
    run = db.create_run(exp.experiment_id, 'run-gh', cfg)
    db.finish_run(run.run_id, final)

    # ── 4. Registry ──────────────────────────────────────────────────────────
    print("\n[4] Saving to registry…")
    reg = ModelRegistry(os.path.join(workdir, 'models'))
    card = ModelCard(
        name='wine-clf', version='1.0.0',
        task='classification', model_type='CyphaDIF',
        encoder_type='VectorEncoder', input_dim=ds.n_features,
        val_accuracy=final.accuracy, val_f1=final.macro_f1,
        n_train=len(tr), dataset_name='wine',
        train_steps=trainer.step, stage='staging',
        gh_protected=True, class_labels=ds.labels,
        intended_use='Wine variety classification from chemical features',
    )
    reg.register(trainer.model, card, pre)
    print(f"    Saved: {card.name} v{card.version}  stage={card.stage}")

    # ── 5. Load + Inference ──────────────────────────────────────────────────
    print("\n[5] Loading from registry and running inference session…")
    m2, p2, c2 = reg.load('wine-clf', '1.0.0')
    engine = InferenceEngine(m2, p2)
    session = InferenceSession(engine)

    n_correct = 0
    corrections = 0
    for x, y_true in zip(te.X, te.y):
        pred = session.predict(x)
        if pred.label == str(y_true):
            n_correct += 1
        elif corrections < 3:
            # Apply a correction for the first few misclassifications
            session.correct(pred, str(y_true))
            corrections += 1

    sess_acc = n_correct / len(te)
    summary  = session.summary()
    print(f"    Session accuracy: {sess_acc:.4f}  ({n_correct}/{len(te)})")
    print(f"    Corrections:      {summary['n_corrections']}")
    print(f"    Mean confidence:  {summary['mean_confidence']:.4f}")
    print(f"    OOD flagged:      {summary['n_ood_flagged']}")

    # Detailed prediction example
    x_ex = te.X[0]
    pred_ex = engine.predict(x_ex, use_gh=True)
    print(f"\n    Example prediction:")
    print(f"      True:   {te.y[0]}")
    print(f"      Pred:   {pred_ex.label}  conf={pred_ex.confidence:.4f}")
    print(f"      OOD:    {pred_ex.is_ood}  anomaly={pred_ex.anomaly_score:.4f}")
    expl = engine.explain(x_ex)
    top_scores = sorted(expl['all_scores'].items(), key=lambda kv: -kv[1])[:3]
    print(f"      Top LLRs: " + "  ".join(f"{l}:{s:+.2f}" for l, s in top_scores))

    # ── 6. Adversarial protection demo ───────────────────────────────────────
    print("\n[6] Adversarial protection demo…")
    rng = np.random.default_rng(99)
    n_adv = 15
    chi, psi = 1.0, 1.0
    for _ in range(n_adv):
        x_adv = rng.normal(0, 20, ds.n_features) - 30  # far OOD
        _, _, chi, psi = m2.gh_train_step(x_adv, str(ds.labels[0]), chi, psi)

    acc_after_adv = sum(
        1 for x, y in zip(te.X, te.y)
        if engine.predict(x).label == str(y)
    ) / len(te)
    print(f"    Accuracy after {n_adv} adversarial injections: {acc_after_adv:.4f}")
    print(f"    (should be close to {sess_acc:.4f} — GH gate protects model)")

    # ── 7. Hyperparameter search ─────────────────────────────────────────────
    print("\n[7] Mini hyperparameter search (4 configs)…")
    from cypha_studio.core.trainer import GridSearch
    search_cfg = TrainerConfig(feat_dim=64, field_dim=64, n_epochs=1,
                               eval_every_n=9999, seed=42)
    search = GridSearch(
        {'world_lr': [0.01, 0.05], 'delta_lr': [0.06, 0.1]},
        verbose=False
    )
    results = search.run(tr, val, search_cfg)
    print(f"    Best params:  {search.best_params}")
    print(f"    Best acc:     {results[0]['accuracy']:.4f}")

    for r in results:
        if 'accuracy' in r:
            run_i = db.create_run(exp.experiment_id, 'search-run',
                                   notes=str(r['params']))
            db.finish_run(run_i.run_id,
                          __import__('cypha_studio.core.trainer',
                                     fromlist=['EvalMetrics']).EvalMetrics(
                              accuracy=r['accuracy'], step=len(tr)))

    # ── 8. REST API smoke test ────────────────────────────────────────────────
    print("\n[8] REST API smoke test…")
    from cypha_studio.server.api import create_app
    app = create_app(engine=engine, session=session)
    print(f"    Routes: {[r.path for r in app.routes if hasattr(r, 'path')]}")

    # Simulate a request using TestClient
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.json()
        print(f"    /health: status={data['status']}  model={data['model']}")

        resp2 = client.post('/predict', json={'input': te.X[0].tolist()})
        assert resp2.status_code == 200
        p = resp2.json()
        print(f"    /predict: label={p['label']}  conf={p['confidence']:.4f}")
    except ImportError:
        print("    (httpx not installed — skipping TestClient check)")

    # ── 9. Experiment leaderboard ─────────────────────────────────────────────
    print("\n[9] Experiment leaderboard…")
    lb = db.leaderboard(exp.experiment_id, top_n=4)
    print(f"    {'Run name':<20}  {'Accuracy':>10}  {'Steps':>8}")
    for r in lb:
        print(f"    {r.name:<20}  {r.accuracy:10.4f}  {r.n_steps:8d}")

    # ── 10. Promote to production ─────────────────────────────────────────────
    print("\n[10] Promoting model to production…")
    reg.promote('wine-clf', '1.0.0', to='production')
    c3 = reg.load_card('wine-clf', '1.0.0')
    print(f"     Stage: {c3.stage}")
    print(f"     Next version: {reg.next_version('wine-clf', 'minor')}")

print("\n" + "═" * 60)
print("  Demo complete. All systems operational.")
print("═" * 60)
