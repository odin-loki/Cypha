"""
cypha_studio.server.api
────────────────────────
Optional FastAPI REST layer. Exposes the active model for external tools,
the C++ inference runtime, web clients, and mobile apps.

Start (module exposes default ``app`` with no model loaded; registry root from ``CYPHA_REGISTRY_ROOT``):

    uvicorn cypha_studio.server.api:app --host 0.0.0.0 --port 7749

Endpoints include ``/health``, ``/ready``, ``/metrics``, ``/models`` (optional ``?summary=true``), ``/register``
(bundle copy into ``ModelRegistry.root`` when a registry is attached — same semantics as native ``cypha_rest``;
**503** if no registry) — see ``docs/studio/CYPHA_ENV.md``.

CORS defaults: ``CYPHA_CORS_ORIGINS`` (see ``docs/studio/CYPHA_ENV.md``).

Optional scalar MoE on ``POST /predict``: ``CYPHA_REGRESSION_HEAD`` or ``create_app(..., regression_head_path=...)`` (``regression_head.json`` — same as native ``cypha_rest --regression-json``).

Or from code:
    from cypha_studio.server.api import start_server
    start_server(engine, registry, port=7749)
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    JSONResponse = None  # type: ignore
    FASTAPI_AVAILABLE = False

import numpy as np


def _parse_regression_head_file(path: str) -> dict[str, tuple[float, float]]:
    """Load native-style ``regression_head.json`` → ``label -> (mu, var_ema)`` (scalar ``mu``)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    ex = raw.get("experts")
    if not isinstance(ex, dict):
        raise ValueError("regression head JSON must contain an object 'experts'")
    out: dict[str, tuple[float, float]] = {}
    for lbl, row in ex.items():
        if not isinstance(row, dict):
            continue
        mu_raw = row.get("mu", 0.0)
        if isinstance(mu_raw, list):
            mu = float(mu_raw[0]) if mu_raw else 0.0
        else:
            mu = float(mu_raw)
        var_e = float(row.get("var_ema", 0.0))
        out[str(lbl)] = (mu, var_e)
    return out


def _softmax_llr_native_style(llr: np.ndarray, temperature: float, eps: float = 1e-8) -> np.ndarray:
    """Row softmax matching native ``softmax_row_like_python`` (small K path)."""
    z = np.asarray(llr, dtype=np.float64) / (float(temperature) + eps)
    mx = float(np.max(z))
    e = np.exp(z - mx)
    s = float(np.sum(e)) + eps
    return (e / s).astype(np.float64)


def _scalar_moe_from_pred(
    eng: Any,
    pred: Any,
    experts: dict[str, tuple[float, float]],
) -> tuple[float, float] | None:
    """
    If ``pred`` is classification (no ``regression_val``), blend expert μ/σ² with routing
    softmax — same contract as ``cypha_rest`` + ``regression_head.json``.
    """
    if pred.regression_val is not None:
        return None
    if not experts:
        return None
    model = getattr(eng, "model", None)
    mem = getattr(model, "memory", None) if model is not None else None
    labels = getattr(mem, "_label_order", None) if mem is not None else None
    if not labels:
        return None
    llr = np.array([float(pred.all_scores.get(lbl, 0.0)) for lbl in labels], dtype=np.float64)
    temp = float(getattr(model, "temperature", 1.15))
    probs = _softmax_llr_native_style(llr, temp)
    mu = np.array([experts.get(lbl, (0.0, 0.0))[0] for lbl in labels], dtype=np.float64)
    var = np.array([experts.get(lbl, (0.0, 0.0))[1] for lbl in labels], dtype=np.float64)
    y = float(np.dot(probs, mu))
    mix_var = float(np.dot(probs, var))
    u = float(np.sqrt(max(mix_var, 0.0)))
    return y, u


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    def _maybe_raise_input_dim_mismatch(exc: BaseException) -> None:
        """Raise HTTP 400 with the same ``detail`` string as native ``cypha_rest`` for bad lengths."""
        if isinstance(exc, ValueError | TypeError):
            s = str(exc).lower()
            if "got length" in s or ("shape" in s and "mismatch" in s):
                raise HTTPException(400, "input dim mismatch after preprocessor") from exc

    class PredictRequest(BaseModel):
        input       : list[float]
        use_gh      : bool = False
        return_explanation : bool = False

    class PredictResponse(BaseModel):
        label        : str
        confidence   : float
        all_scores   : dict[str, float] = {}
        anomaly_score: float = 0.0
        is_ood       : bool  = False
        regression_val: float | None = None
        uncertainty  : float = 0.0
        explanation  : dict | None = None
        latency_ms   : float = 0.0

    class UpdateRequest(BaseModel):
        input        : list[float]
        correct_label: str
        use_gh       : bool = True
        # Optional native ``cypha_rest`` parity keys (see PORT_CONTRACT §3).
        regression_y: float | None = None
        router_train_label: str | None = None
        replay_u01: list[float] | None = None

    class UpdateResponse(BaseModel):
        loss         : float
        n_corrections: int

    class AdaptTemperatureCalibrationRow(BaseModel):
        input         : list[float]
        correct_label : str

    class AdaptTemperatureRequest(BaseModel):
        calibration: list[AdaptTemperatureCalibrationRow]
        n_grid     : int = 20
        T_min      : float = 0.3
        T_max      : float = 8.0
        n_bins     : int = 10

    class AdaptTemperatureResponse(BaseModel):
        temperature: float
        n_used     : int

    class LoadRequest(BaseModel):
        name    : str
        version : str = 'latest'

    class SessionResponse(BaseModel):
        n_predictions     : int
        n_corrections     : int
        correction_accuracy: float = 0.0
        mean_confidence   : float
        mean_anomaly      : float
        n_ood_flagged     : int
        label_distribution: dict[str, int]
        session_duration_s: float

    class RegisterRequest(BaseModel):
        """Same JSON body as native ``cypha_rest`` ``POST /register`` (paths on the server host)."""

        name: str
        version: str
        model_cypha: str
        card_json: str
        preprocessor_json: str | None = None
        overwrite: bool = False

    class RngStateResponse(BaseModel):
        """Snapshot of the active replay-RNG MT19937 state (numpy ``bit_generator.state`` shape)."""
        bit_generator: str
        state: list[int]
        pos: int

    class RngSeedRequest(BaseModel):
        """Seed or restore the replay-RNG.
        Provide either ``seed`` (re-seed from scratch) or ``state`` + ``pos`` (full restore).
        """
        seed: int | None = None
        state: list[int] | None = None
        pos: int = 0

    class LMLoadRequest(BaseModel):
        """Load CyphaLM from a checkpoint base path (writes ``.json`` + ``.npz``)."""
        checkpoint_path: str

    class LMPredictNextRequest(BaseModel):
        token_id: int

    class LMGenerateRequest(BaseModel):
        prompt_ids: list[int]
        max_tokens: int = 64
        temperature: float = 0.9
        strategy: str = "temperature"  # greedy | temperature | top_k | top_p | uncertainty_gated
        top_k: int = 40
        top_p: float = 0.9
        uncertainty_threshold: float | None = None
        stream: bool = False

    class RouteTextRequest(BaseModel):
        """Branch A: embed text → CyphaDIF classify + epistemic gate."""
        text: str
        epistemic_threshold: float | None = None

    class RouteGenerateRequest(BaseModel):
        """Branch A: route then CyphaLM (in-domain) or Ollama fallback (OOD)."""
        text: str
        epistemic_threshold: float | None = None
        max_tokens: int = 128
        ollama_model: str | None = None
        ollama_system: str | None = None
        cypha_lm_strategy: str = "top_p"
        cypha_lm_temperature: float = 0.9

    class DIFGenerateRequest(BaseModel):
        """CyphaDIF latent generation (``POST /dif/generate`` — not CyphaLM ``POST /generate``)."""
        input: list[float]
        mode: str  # langevin | from_observation | retrieval_augmented
        database: list[list[float]] | None = None
        label: str | None = None
        k_neighbors: int = 5
        n_samples: int = 10
        n_steps: int = 30
        temperature: float = 1.0
        seed: int | None = None

    class DIFRetrieveRequest(BaseModel):
        input: list[float]
        database: list[list[float]]
        top_k: int = 5
        label: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(
    engine=None,
    registry=None,
    session=None,
    cors_allow_origins: Sequence[str] | None = None,
    regression_head_path: str | None = None,
    lm_engine=None,
) -> FastAPI:
    """
    Create the FastAPI app with the given inference engine and registry.
    These can be replaced at runtime via app.state.

    ``cors_allow_origins``: list of allowed origins, or ``["*"]`` for all.
    If ``None``, uses ``CYPHA_CORS_ORIGINS`` (see ``cypha_studio.env_config``).

    ``regression_head_path``: optional path to ``regression_head.json`` (same schema as native
    ``cypha_rest --regression-json``). If ``None``, reads ``CYPHA_REGRESSION_HEAD`` when set.
    Fills ``regression_val`` / ``uncertainty`` on ``POST /predict`` for classification models.
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn")

    if cors_allow_origins is None:
        from ..env_config import cors_allow_origins as _cors

        origins: list[str] = list(_cors())
    else:
        origins = list(cors_allow_origins)

    app = FastAPI(
        title="CyphaStudio API",
        description="REST interface for Cypha model inference and management",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine   = engine
    app.state.registry = registry
    app.state.session  = session
    app.state.lm_engine = lm_engine
    app.state.branch_a_router = None
    app.state.started  = time.time()

    # Optional CyphaLM checkpoint on startup (FastAPI-only; not in native cypha_rest).
    if lm_engine is None:
        lm_ckpt = os.environ.get("CYPHA_LM_CHECKPOINT", "").strip()
        if lm_ckpt:
            try:
                from ..core.lm_engine import LMEngine
                app.state.lm_engine = LMEngine.from_checkpoint(lm_ckpt)
            except ImportError:
                pass

    reg_path = regression_head_path
    if reg_path is None:
        reg_path = os.environ.get("CYPHA_REGRESSION_HEAD", "").strip() or None
    app.state.regression_experts: dict[str, tuple[float, float]] = {}
    app.state.mke_state = None
    if reg_path:
        app.state.regression_experts = _parse_regression_head_file(reg_path)
        try:
            from .mke_update import parse_mke_from_regression_file
            app.state.mke_state = parse_mke_from_regression_file(reg_path)
        except (ValueError, KeyError, TypeError):
            app.state.mke_state = None

    # ── Endpoints ────────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        eng = app.state.engine
        model_name = "none"
        if eng is not None:
            model_name = type(eng.model).__name__
        return {
            "status" : "ok",
            "model"  : model_name,
            "lm_loaded": app.state.lm_engine is not None,
            "uptime" : time.time() - app.state.started,
            "n_predictions": eng.n_predictions if eng else 0,
        }

    @app.get("/ready")
    def ready():
        """
        Readiness for orchestrators that require a loaded model.

        Returns **503** with ``{"ready": false, "reason": "no_model_loaded"}`` when
        no engine is attached. Use ``/health`` for process liveness regardless of model.
        """
        eng = app.state.engine
        if eng is None:
            return JSONResponse(
                status_code=503,
                content={"ready": False, "reason": "no_model_loaded"},
            )
        return {"ready": True, "model_type": type(eng.model).__name__}

    @app.get("/metrics")
    def metrics():
        """
        JSON snapshot for monitoring / orchestration (not Prometheus text format).

        Includes uptime, model counters, registry size, and a subset of session stats.
        """
        now = time.time()
        eng = app.state.engine
        reg = app.state.registry
        sess = app.state.session
        payload: dict[str, Any] = {
            "uptime_seconds": round(now - app.state.started, 3),
            "model_loaded": eng is not None,
            "model_type": type(eng.model).__name__ if eng else None,
            "n_predictions": eng.n_predictions if eng else 0,
            "n_corrections": eng.n_corrections if eng else 0,
            "registry_model_count": reg.registered_entry_count() if reg else 0,
        }
        if eng is not None and hasattr(eng.model, "_gh_chi_session"):
            payload["gh_chi_session"] = float(eng.model._gh_chi_session)
            payload["gh_psi_session"] = float(eng.model._gh_psi_session)
        if sess is not None:
            payload["session"] = sess.summary()
        else:
            payload["session"] = None
        experts = getattr(app.state, "regression_experts", {}) or {}
        payload["regression_head_loaded"] = eng is not None and (bool(experts) or getattr(app.state, "mke_state", None) is not None)
        lm = app.state.lm_engine
        payload["lm_loaded"] = lm is not None
        if lm is not None:
            payload["lm"] = lm.summary()
        router = getattr(app.state, "branch_a_router", None)
        payload["branch_a_router"] = router.summary() if router is not None else {"trained": False}
        return payload

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        eng = app.state.engine
        if eng is None:
            raise HTTPException(503, "No model loaded")
        t0 = time.perf_counter()
        try:
            pred = eng.predict(np.array(req.input, dtype=np.float64),
                               use_gh=req.use_gh)
        except (ValueError, TypeError) as e:
            _maybe_raise_input_dim_mismatch(e)
            raise
        latency = (time.perf_counter() - t0) * 1000

        reg_val = pred.regression_val
        unc_out = float(pred.uncertainty)
        experts = getattr(app.state, "regression_experts", {}) or {}
        blended = _scalar_moe_from_pred(eng, pred, experts)
        if blended is not None:
            reg_val, unc_out = blended[0], blended[1]

        explanation = None
        if req.return_explanation:
            try:
                explanation = eng.explain(np.array(req.input, dtype=np.float64))
                # Remove numpy arrays from explanation for JSON serialisation
                explanation = {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                               for k, v in explanation.items()}
            except Exception as e:
                explanation = {"error": str(e)}

        return PredictResponse(
            label=pred.label,
            confidence=pred.confidence,
            all_scores=pred.all_scores,
            anomaly_score=pred.anomaly_score,
            is_ood=pred.is_ood,
            regression_val=reg_val,
            uncertainty=unc_out,
            explanation=explanation,
            latency_ms=latency,
        )

    @app.post("/update", response_model=UpdateResponse)
    def update(req: UpdateRequest):
        eng = app.state.engine
        if eng is None:
            raise HTTPException(503, "No model loaded")
        x = np.array(req.input, dtype=np.float64)
        if req.regression_y is not None:
            mke = getattr(app.state, "mke_state", None)
            if mke is None:
                raise HTTPException(
                    400,
                    "regression_y requires mke block in regression_head.json",
                )
            try:
                from .mke_update import mke_rest_update
                x_pp = eng._preprocess(x)
                _, router_loss = mke_rest_update(
                    eng.model,
                    mke,
                    x_pp,
                    float(req.regression_y),
                    router_train_label=req.router_train_label,
                    replay_u01=req.replay_u01,
                    use_gh=req.use_gh,
                )
                eng._n_corrections += 1
                loss = router_loss
            except (ValueError, TypeError) as e:
                _maybe_raise_input_dim_mismatch(e)
                raise
            except RuntimeError as e:
                if "ListReplayRng exhausted" in str(e):
                    raise HTTPException(400, str(e)) from e
                raise
        else:
            try:
                loss = eng.update(
                    x,
                    req.correct_label,
                    use_gh=req.use_gh,
                    replay_u01=req.replay_u01,
                )
            except (ValueError, TypeError) as e:
                _maybe_raise_input_dim_mismatch(e)
                raise
            except RuntimeError as e:
                if "ListReplayRng exhausted" in str(e):
                    raise HTTPException(400, str(e)) from e
                raise
        return UpdateResponse(loss=loss, n_corrections=eng.n_corrections)

    @app.post("/adapt_temperature", response_model=AdaptTemperatureResponse)
    def adapt_temperature(req: AdaptTemperatureRequest):
        """
        Grid-search temperature to minimise ECE on a labelled calibration set.

        Wraps ``CyphaDIF.adapt_temperature`` including ``n_bins`` for ECE histograms
        (aligned with native ``cypha_rest``).
        """
        eng = app.state.engine
        if eng is None:
            raise HTTPException(503, "No model loaded")
        model = eng.model
        adapt = getattr(model, "adapt_temperature", None)
        if adapt is None:
            raise HTTPException(400, "Model does not support adapt_temperature")
        cal = [(np.array(row.input, dtype=np.float64), row.correct_label) for row in req.calibration]
        n_grid = max(1, int(req.n_grid))
        with model.memory._lock:
            kset = set(model.memory._classes.keys())
        n_used = sum(1 for _x, y in cal if y in kset)
        try:
            T_star = adapt(
                cal,
                n_grid=n_grid,
                T_min=float(req.T_min),
                T_max=float(req.T_max),
                n_bins=max(2, int(req.n_bins)),
            )
        except (ValueError, TypeError) as e:
            _maybe_raise_input_dim_mismatch(e)
            raise
        return AdaptTemperatureResponse(temperature=float(T_star), n_used=n_used)

    @app.get("/models")
    def list_models(summary: bool = False):
        """
        List registered models. With ``summary=true``, returns only ``name`` and
        ``version`` per row (directory scan + ``card.json`` presence — no full card parse).
        Default is full ``ModelCard`` dicts via ``list_models()``.
        """
        reg = app.state.registry
        if reg is None:
            return {"models": []}
        if summary:
            return {
                "models": [
                    {"name": n, "version": v}
                    for n, v in reg.iter_registered_pairs()
                ],
            }
        from dataclasses import asdict
        return {"models": [asdict(c) for c in reg.list_models()]}

    @app.post("/register")
    def register_bundle(req: RegisterRequest):
        """
        Copy ``model.cypha`` + ``card.json`` (+ optional ``preprocessor.json``) into
        ``<registry_root>/<name>/<version>/``, matching native ``registry_register_bundle`` / ``cypha_rest``.
        """
        reg = app.state.registry
        if reg is None:
            raise HTTPException(503, "No registry configured")

        root: Path = reg.root
        dest = root / req.name / req.version
        cypha_src = Path(req.model_cypha).expanduser()
        card_src = Path(req.card_json).expanduser()
        pre_src: Path | None = None
        if req.preprocessor_json:
            pre_src = Path(req.preprocessor_json).expanduser()

        if dest.exists():
            if not req.overwrite:
                raise HTTPException(400, "destination exists (use overwrite)")
            shutil.rmtree(dest)

        if not cypha_src.is_file():
            raise HTTPException(400, "cypha source missing")
        if not card_src.is_file():
            raise HTTPException(400, "card source missing")
        if pre_src is not None and not pre_src.is_file():
            raise HTTPException(400, "preprocessor source missing")

        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cypha_src, dest / "model.cypha")
        shutil.copy2(card_src, dest / "card.json")
        if pre_src is not None:
            shutil.copy2(pre_src, dest / "preprocessor.json")

        return {"registered": True, "model_dir": str(dest.resolve())}

    @app.post("/load")
    def load_model(req: LoadRequest):
        reg = app.state.registry
        if reg is None:
            raise HTTPException(503, "No registry configured")
        try:
            model, pre, card = reg.load(req.name, req.version)
            from ..core.inference import InferenceEngine, InferenceSession
            app.state.engine  = InferenceEngine(model, pre)
            app.state.session = InferenceSession(app.state.engine)
            from dataclasses import asdict
            return {"loaded": asdict(card)}
        except Exception as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/session", response_model=SessionResponse)
    def session_info():
        sess = app.state.session
        if sess is None:
            return SessionResponse(
                n_predictions=0, n_corrections=0, correction_accuracy=0.0,
                mean_confidence=0.0, mean_anomaly=0.0,
                n_ood_flagged=0, label_distribution={},
                session_duration_s=0.0,
            )
        s = sess.summary()
        return SessionResponse(
            n_predictions=s.get('n_predictions', 0),
            n_corrections=s.get('n_corrections', 0),
            correction_accuracy=s.get('correction_accuracy', 0.0),
            mean_confidence=s.get('mean_confidence', 0.0),
            mean_anomaly=s.get('mean_anomaly', 0.0),
            n_ood_flagged=s.get('n_ood_flagged', 0),
            label_distribution=s.get('label_distribution', {}),
            session_duration_s=s.get('session_duration_s', 0.0),
        )

    @app.delete("/session")
    def clear_session():
        sess = app.state.session
        if sess:
            sess.clear()
        return {"cleared": True}

    @app.get("/session/rng", response_model=RngStateResponse)
    def get_rng_state():
        """
        Export the current MT19937 state of the model's replay-RNG.

        Response fields match numpy ``bit_generator.state`` structure:
        ``bit_generator`` is always ``"MT19937"``, ``state`` is a list of 624 uint32
        words, and ``pos`` is the current position in the state array (0–623).

        Use ``POST /session/rng`` to restore this snapshot or to re-seed.

        Returns **503** if no model is loaded, **404** if the model lacks ``_replay_rng``.
        """
        eng = app.state.engine
        if eng is None:
            raise HTTPException(503, "No model loaded")
        model = eng.model
        rng = getattr(model, "_replay_rng", None)
        if rng is None:
            raise HTTPException(404, "model has no _replay_rng")
        bg = rng.bit_generator.state
        return RngStateResponse(
            bit_generator="MT19937",
            state=bg["state"]["key"].tolist(),
            pos=int(bg["state"]["pos"]),
        )

    @app.post("/session/rng", response_model=RngStateResponse)
    def set_rng_state(req: RngSeedRequest):
        """
        Seed or restore the model's replay-RNG for deterministic replay.

        * Provide **``seed``** to re-initialise from scratch (discards current state).
        * Provide **``state``** (list of 624 uint32 words) and **``pos``** to do a
          full state restore (e.g. from a previously captured snapshot).

        Returns the new state in the same shape as ``GET /session/rng``.
        """
        eng = app.state.engine
        if eng is None:
            raise HTTPException(503, "No model loaded")
        model = eng.model
        old_rng = getattr(model, "_replay_rng", None)
        if old_rng is None:
            raise HTTPException(404, "model has no _replay_rng")
        if req.seed is not None:
            new_rng = np.random.Generator(np.random.MT19937(int(req.seed)))
            model._replay_rng = new_rng
            if hasattr(model, "replay") and hasattr(model.replay, "_rng"):
                model.replay._rng = new_rng
        elif req.state is not None:
            if len(req.state) != 624:
                raise HTTPException(400, "state must have exactly 624 uint32 values")
            bg = old_rng.bit_generator.state
            bg["state"]["key"] = np.array(req.state, dtype=np.uint32)
            bg["state"]["pos"] = int(req.pos)
            old_rng.bit_generator.state = bg
            new_rng = old_rng
        else:
            raise HTTPException(400, "provide seed or state")
        bg_out = new_rng.bit_generator.state
        return RngStateResponse(
            bit_generator="MT19937",
            state=bg_out["state"]["key"].tolist(),
            pos=int(bg_out["state"]["pos"]),
        )

    @app.get("/classes")
    def get_classes():
        eng = app.state.engine
        if eng is None:
            raise HTTPException(503, "No model loaded")
        try:
            with eng.model.memory._lock:
                classes = {
                    lbl: {'n_obs': float(cd.n_obs)}
                    for lbl, cd in eng.model.memory._classes.items()
                }
            return {"classes": classes}
        except Exception as e:
            raise HTTPException(500, str(e)) from e

    # ── CyphaLM language-model routes (FastAPI-only) ─────────────────────

    @app.post("/lm/load")
    def lm_load(req: LMLoadRequest):
        """Load CyphaLM from checkpoint (``.json`` + ``.npz`` pair)."""
        try:
            from ..core.lm_engine import LMEngine
            app.state.lm_engine = LMEngine.from_checkpoint(req.checkpoint_path)
        except ImportError as exc:
            raise HTTPException(501, f"cypha_lm not installed: {exc}") from exc
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"loaded": True, "summary": app.state.lm_engine.summary()}

    @app.get("/lm/metrics")
    def lm_metrics():
        lm = app.state.lm_engine
        if lm is None:
            raise HTTPException(503, "No CyphaLM loaded")
        return lm.summary()

    @app.post("/lm/predict_next")
    def lm_predict_next(req: LMPredictNextRequest):
        """Single-token forward with CyphaDIF routing probabilities."""
        lm = app.state.lm_engine
        if lm is None:
            raise HTTPException(503, "No CyphaLM loaded")
        return lm.predict_next(req.token_id)

    @app.post("/generate")
    def lm_generate(req: LMGenerateRequest):
        """
        Autoregressive generation with CyphaLM.

        Set ``stream=true`` for Server-Sent Events (one JSON object per token).
        Strategies: ``greedy``, ``temperature``, ``top_k``, ``top_p``,
        ``uncertainty_gated`` (use ``uncertainty_threshold``).
        """
        lm = app.state.lm_engine
        if lm is None:
            raise HTTPException(503, "No CyphaLM loaded")

        if req.stream:
            def _sse():
                for chunk in lm.stream_generate(
                    req.prompt_ids,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    strategy=req.strategy,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    uncertainty_threshold=req.uncertainty_threshold,
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: {\"done\": true}\n\n"

            return StreamingResponse(_sse(), media_type="text/event-stream")

        out = lm.generate(
            req.prompt_ids,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            strategy=req.strategy,
            top_k=req.top_k,
            top_p=req.top_p,
            uncertainty_threshold=req.uncertainty_threshold,
        )
        # JSON-safe: convert numpy arrays in per_step_metrics if present
        if "per_step_metrics" in out:
            pm = out["per_step_metrics"]
            out["per_step_metrics"] = {
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in pm.items()
            }
        return out

    @app.post("/generate/stream")
    def lm_generate_stream(req: LMGenerateRequest):
        """Streaming generation (SSE). Same body as ``POST /generate``."""
        lm = app.state.lm_engine
        if lm is None:
            raise HTTPException(503, "No CyphaLM loaded")

        def _sse():
            for chunk in lm.stream_generate(
                req.prompt_ids,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                strategy=req.strategy,
                top_k=req.top_k,
                top_p=req.top_p,
                uncertainty_threshold=req.uncertainty_threshold,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: {\"done\": true}\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    # ── Branch A text routing (FastAPI-only) ─────────────────────────────

    def _get_branch_a_router():
        router = app.state.branch_a_router
        if router is None:
            from ..core.branch_a_router import BranchARouter

            router = BranchARouter()
            router.try_load_checkpoint()
            app.state.branch_a_router = router
        return router

    @app.get("/route/health")
    def route_health():
        """Branch A router status and optional Ollama reachability."""
        from ..core.ollama_client import ollama_available, ollama_base_url, ollama_model

        router = getattr(app.state, "branch_a_router", None)
        return {
            "router_trained": router.is_trained if router is not None else False,
            "router_summary": router.summary() if router is not None else None,
            "ollama_url": ollama_base_url(),
            "ollama_model": ollama_model(),
            "ollama_reachable": ollama_available(),
            "lm_loaded": app.state.lm_engine is not None,
        }

    @app.post("/route/text")
    def route_text(req: RouteTextRequest):
        """Embed user text → CyphaDIF label + epistemic gate (no generation)."""
        if not req.text.strip():
            raise HTTPException(400, "text must be non-empty")
        t0 = time.perf_counter()
        router = _get_branch_a_router()
        try:
            result = router.route(req.text, epistemic_threshold=req.epistemic_threshold)
        except Exception as exc:
            raise HTTPException(500, f"Branch A route failed: {exc}") from exc
        result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return result

    @app.post("/route/generate")
    def route_generate(req: RouteGenerateRequest):
        """
        Route then generate: in-domain → CyphaLM if loaded; OOD abstain → Ollama.
        """
        if not req.text.strip():
            raise HTTPException(400, "text must be non-empty")
        t0 = time.perf_counter()
        router = _get_branch_a_router()
        try:
            out = router.dispatch_generate(
                req.text,
                app.state.lm_engine,
                epistemic_threshold=req.epistemic_threshold,
                max_tokens=req.max_tokens,
                ollama_model=req.ollama_model,
                ollama_system=req.ollama_system,
                cypha_lm_strategy=req.cypha_lm_strategy,
                cypha_lm_temperature=req.cypha_lm_temperature,
            )
        except Exception as exc:
            raise HTTPException(500, f"Branch A dispatch failed: {exc}") from exc
        out["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    @app.post("/route/save")
    def route_save():
        """Persist trained Branch A router to ``CYPHA_BRANCH_A_CHECKPOINT`` (``.json`` + ``.npz``)."""
        router = _get_branch_a_router()
        if not router.is_trained:
            raise HTTPException(400, "Router not trained — call /route/text or /route/generate first")
        try:
            path = router.save_checkpoint()
        except OSError as exc:
            raise HTTPException(500, f"Failed to save checkpoint: {exc}") from exc
        return {"saved": True, "checkpoint": str(path), "summary": router.summary()}

    # ── CyphaDIF generation / retrieval (FastAPI parity with native ``/dif/*``) ──

    def _dif_model_rng(eng: Any, seed: int | None):
        import numpy as np
        model = getattr(eng, "model", None) or getattr(eng, "_model", None)
        if model is None:
            raise HTTPException(503, "No model loaded")
        rng = getattr(model, "_rng", None)
        if rng is None:
            rng = np.random.default_rng(424242)
        if seed is not None:
            rng = np.random.default_rng(int(seed))
        return model, rng

    def _dif_preprocess_vector(eng: Any, inp: list[float]) -> np.ndarray:
        vec = np.array(inp, dtype=np.float64)
        try:
            return eng._preprocess(vec)
        except (ValueError, TypeError) as e:
            _maybe_raise_input_dim_mismatch(e)
            raise

    def _dif_latent_from_observation(model: Any, h_obs: np.ndarray, label: str, n: int,
                                     temperature: float, n_steps: int, rng: Any) -> list[list[float]]:
        """Latent-space ``generate_from_observation`` (matches native REST ``samples``)."""
        import math
        with model.memory._lock:
            if label not in model.memory._classes:
                return [np.asarray(h_obs, dtype=np.float64).tolist()] * n
            cd = model.memory._classes[label]
            mu0 = model.memory.world.mu.copy()
            inv_v = model.memory.world.inv_v.copy()
        mu_k = mu0 + cd.delta_mu
        lr = float(temperature) * 0.05
        noise_scale = math.sqrt(2.0 * lr)
        out: list[list[float]] = []
        for _ in range(n):
            h = np.asarray(h_obs, dtype=np.float64).copy()
            for _ in range(n_steps):
                grad = -(h - mu_k) * inv_v
                h = h + lr * grad + rng.standard_normal(len(h)) * noise_scale
            out.append(h.tolist())
        return out

    @app.post("/dif/generate")
    def dif_generate(req: DIFGenerateRequest):
        """
        CyphaDIF latent generation (Langevin / observation-anchored / retrieval-augmented).

        Returns latent vectors in ``samples`` (``space: latent``). This is **not** CyphaLM
        ``POST /generate`` (token autoregression) — use ``/lm/*`` and ``POST /generate`` for LM.
        """
        eng = app.state.engine
        if eng is None:
            raise HTTPException(503, "No model loaded")
        mode = req.mode.strip().lower()
        if mode not in ("langevin", "from_observation", "retrieval_augmented"):
            raise HTTPException(
                400,
                "mode must be langevin, from_observation, or retrieval_augmented",
            )
        x = _dif_preprocess_vector(eng, req.input)
        model, rng = _dif_model_rng(eng, req.seed)
        _, h_query = model._encode(x)
        label = req.label
        if not label:
            label, _ = model.classify_latent(np.asarray(h_query, dtype=np.float64))

        if mode == "langevin":
            H = model.generate_langevin(
                label, n=req.n_samples, n_steps=req.n_steps, step_size=0.05,
                temperature=req.temperature, rng=rng,
            )
            samples = [np.asarray(h, dtype=np.float64).tolist() for h in H]
        elif mode == "from_observation":
            samples = _dif_latent_from_observation(
                model, np.asarray(h_query, dtype=np.float64), label,
                req.n_samples, req.temperature, req.n_steps, rng,
            )
        else:
            if not req.database:
                raise HTTPException(400, "database required for retrieval_augmented")
            db_rows = [_dif_preprocess_vector(eng, row) for row in req.database]
            hits = model.retrieve(x, db_rows, top_k=req.k_neighbors)
            if hits:
                h_db = [model._encode(row)[1] for row in db_rows]
                dists2 = []
                h_hits = []
                for idx, _ll, _lbl in hits:
                    h_i = np.asarray(h_db[idx], dtype=np.float64)
                    h_hits.append(h_i)
                    dists2.append(float(np.sum((h_i - h_query) ** 2)))
                dists2_arr = np.asarray(dists2, dtype=np.float64)
                d_max = float(dists2_arr.max()) if len(dists2_arr) else 0.0
                weights = np.exp(-dists2_arr / max(d_max * 0.1, 1e-8))
                weights = weights / (weights.sum() + 1e-8)
                h_anchor = sum(float(weights[j]) * h_hits[j] for j in range(len(hits)))
                nearest_j = int(np.argmin(dists2_arr))
                label = hits[nearest_j][2]
                samples = _dif_latent_from_observation(
                    model, np.asarray(h_anchor, dtype=np.float64), label,
                    req.n_samples, req.temperature, req.n_steps, rng,
                )
            else:
                H = model.generate_langevin(
                    label, n=req.n_samples, n_steps=req.n_steps, step_size=0.05,
                    temperature=req.temperature, rng=rng,
                )
                samples = [np.asarray(h, dtype=np.float64).tolist() for h in H]

        return {
            "mode": mode,
            "label": label,
            "n_samples": req.n_samples,
            "space": "latent",
            "samples": samples,
        }

    @app.post("/dif/retrieve")
    def dif_retrieve(req: DIFRetrieveRequest):
        """Log-likelihood ranked retrieval over a caller-supplied database."""
        eng = app.state.engine
        if eng is None:
            raise HTTPException(503, "No model loaded")
        x = _dif_preprocess_vector(eng, req.input)
        db_rows = [_dif_preprocess_vector(eng, row) for row in req.database]
        model, _rng = _dif_model_rng(eng, None)
        hits = model.retrieve(x, db_rows, top_k=req.top_k, label=req.label)
        return {
            "top_k": req.top_k,
            "hits": [
                {"index": idx, "log_likelihood": ll, "predicted_label": lbl}
                for idx, ll, lbl in hits
            ],
        }

    return app


if FASTAPI_AVAILABLE:
    # Default ASGI app for ``uvicorn cypha_studio.server.api:app`` (no model pre-loaded).
    # Registry root follows ``CYPHA_REGISTRY_ROOT`` (see ``cypha_studio.env_config``) so ``/register``,
    # ``/load``, and ``/models`` work without embedding ``create_app``.
    from ..core.registry import ModelRegistry
    from ..env_config import registry_root

    app = create_app(registry=ModelRegistry(registry_root()))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience launcher
# ─────────────────────────────────────────────────────────────────────────────

def start_server(engine=None, registry=None, session=None,
                  host: str = '127.0.0.1', port: int = 7749,
                  log_level: str = 'info',
                  regression_head_path: str | None = None):
    """
    Start the REST API server in the current thread (blocking).

    Typical usage: call this in a QThread from the main application.
    ``regression_head_path``: optional; if ``None``, ``CYPHA_REGRESSION_HEAD`` is used when set.
    """
    import uvicorn
    app = create_app(
        engine=engine, registry=registry, session=session,
        regression_head_path=regression_head_path,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level)


def start_server_async(engine=None, registry=None, session=None,
                        host: str = '127.0.0.1', port: int = 7749,
                        regression_head_path: str | None = None):
    """
    Start the REST API server in a background thread (non-blocking).
    Returns the thread.
    """
    import threading
    t = threading.Thread(
        target=start_server,
        kwargs=dict(
            engine=engine, registry=registry, session=session,
            host=host, port=port, log_level='warning',
            regression_head_path=regression_head_path,
        ),
        daemon=True,
    )
    t.start()
    return t
