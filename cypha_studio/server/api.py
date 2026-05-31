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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


def _parse_regression_head_file(path: str) -> Dict[str, Tuple[float, float]]:
    """Load native-style ``regression_head.json`` → ``label -> (mu, var_ema)`` (scalar ``mu``)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    ex = raw.get("experts")
    if not isinstance(ex, dict):
        raise ValueError("regression head JSON must contain an object 'experts'")
    out: Dict[str, Tuple[float, float]] = {}
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
    experts: Dict[str, Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
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
        if isinstance(exc, (ValueError, TypeError)):
            s = str(exc).lower()
            if "got length" in s or ("shape" in s and "mismatch" in s):
                raise HTTPException(400, "input dim mismatch after preprocessor") from exc

    class PredictRequest(BaseModel):
        input       : List[float]
        use_gh      : bool = False
        return_explanation : bool = False

    class PredictResponse(BaseModel):
        label        : str
        confidence   : float
        all_scores   : Dict[str, float] = {}
        anomaly_score: float = 0.0
        is_ood       : bool  = False
        regression_val: Optional[float] = None
        uncertainty  : float = 0.0
        explanation  : Optional[Dict] = None
        latency_ms   : float = 0.0

    class UpdateRequest(BaseModel):
        input        : List[float]
        correct_label: str
        use_gh       : bool = True
        # Native cypha_rest only (PORT_CONTRACT §3); FastAPI classification path ignores / rejects below.
        regression_y: Optional[float] = None
        router_train_label: Optional[str] = None
        replay_u01: Optional[List[float]] = None

    class UpdateResponse(BaseModel):
        loss         : float
        n_corrections: int

    class AdaptTemperatureCalibrationRow(BaseModel):
        input         : List[float]
        correct_label : str

    class AdaptTemperatureRequest(BaseModel):
        calibration: List[AdaptTemperatureCalibrationRow]
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
        label_distribution: Dict[str, int]
        session_duration_s: float

    class RegisterRequest(BaseModel):
        """Same JSON body as native ``cypha_rest`` ``POST /register`` (paths on the server host)."""

        name: str
        version: str
        model_cypha: str
        card_json: str
        preprocessor_json: Optional[str] = None
        overwrite: bool = False

    class RngStateResponse(BaseModel):
        """Snapshot of the active replay-RNG MT19937 state (numpy ``bit_generator.state`` shape)."""
        bit_generator: str
        state: List[int]
        pos: int

    class RngSeedRequest(BaseModel):
        """Seed or restore the replay-RNG.
        Provide either ``seed`` (re-seed from scratch) or ``state`` + ``pos`` (full restore).
        """
        seed: Optional[int] = None
        state: Optional[List[int]] = None
        pos: int = 0

    class LMLoadRequest(BaseModel):
        """Load CyphaLM from a checkpoint base path (writes ``.json`` + ``.npz``)."""
        checkpoint_path: str

    class LMPredictNextRequest(BaseModel):
        token_id: int

    class LMGenerateRequest(BaseModel):
        prompt_ids: List[int]
        max_tokens: int = 64
        temperature: float = 0.9
        strategy: str = "temperature"  # greedy | temperature | top_k | top_p | uncertainty_gated
        top_k: int = 40
        top_p: float = 0.9
        uncertainty_threshold: Optional[float] = None
        stream: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(
    engine=None,
    registry=None,
    session=None,
    cors_allow_origins: Optional[Sequence[str]] = None,
    regression_head_path: Optional[str] = None,
    lm_engine=None,
) -> 'FastAPI':
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

        origins: List[str] = list(_cors())
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
    app.state.regression_experts: Dict[str, Tuple[float, float]] = {}
    if reg_path:
        app.state.regression_experts = _parse_regression_head_file(reg_path)

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
        payload: Dict[str, Any] = {
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
        payload["regression_head_loaded"] = eng is not None and bool(experts)
        lm = app.state.lm_engine
        payload["lm_loaded"] = lm is not None
        if lm is not None:
            payload["lm"] = lm.summary()
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
        if req.regression_y is not None:
            raise HTTPException(
                501,
                "regression_y (MKERegressor-style train step) is implemented in native cypha_rest "
                "with regression_head.json mke block; FastAPI update is classification-only",
            )
        if req.replay_u01 is not None:
            raise HTTPException(
                501,
                "replay_u01 on /update is native cypha_rest only; FastAPI uses in-process RNG for replay",
            )
        if req.router_train_label is not None:
            raise HTTPException(
                501,
                "router_train_label on /update is native cypha_rest only (mke router override)",
            )
        try:
            loss = eng.update(np.array(req.input, dtype=np.float64),
                              req.correct_label, use_gh=req.use_gh)
        except (ValueError, TypeError) as e:
            _maybe_raise_input_dim_mismatch(e)
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
        pre_src: Optional[Path] = None
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
            raise HTTPException(404, str(e))

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
            raise HTTPException(500, str(e))

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
                  regression_head_path: Optional[str] = None):
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
                        regression_head_path: Optional[str] = None):
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
