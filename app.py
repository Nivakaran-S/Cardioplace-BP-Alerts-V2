"""Cardioplace BP Alerts -- provider-visible decision support API.

Composition root only: routes, mounts and error handlers. Every frame is built in
`src/serving/mapping.py`, every dashboard block in `src/serving/enrich.py`. `app.py` is what
a reviewer opens first, and the risky code is deliberately not here.

The design property worth stating up front: **this API is useful with no model on disk.**
The deterministic 56-rule engine is the safety-critical layer and needs no ML. An SBP of 195
produces a red emergency banner on a fresh checkout with an empty `final_model/`. Only the
forecast, the personalised threshold and the early-warning score require a bundle, and each
degrades to an explicit "not issued" rather than a 503.

Run:  uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1

One worker, deliberately. Two would mean two independently hot-reloading predictors and two
training managers racing for the same single-flight lock; the state here is process-local by
construction.
"""

import os
import time
import uuid

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.constants.training_pipeline import EMERGENCY_FLOOR_MMHG, POPULATION_THRESHOLD_MMHG
from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.serving import settings as S
from src.serving.enrich import (
    Timer,
    anomaly_block,
    backtest_block,
    build_panel_frame,
    history_echo,
    predicted_alert_block,
    rule_engine_block,
    symptom_block,
)
from src.serving.jsonify import to_jsonable
from src.serving.mapping import to_engine_panel, to_history
from src.serving.model_registry import ModelRegistry
from src.serving.schemas import PredictRequest
from src.serving.training import TrainingManager
from src.serving.vocabulary import build_vocabulary
from src.utils.ml_utils.rule_engine.registry import build_registry

app = FastAPI(title="Cardioplace BP Alerts", version="2.0",
              description="Provider-visible decision support. Not a medical device.")

os.makedirs(S.STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=S.STATIC_DIR), name="static")
templates = Jinja2Templates(directory=S.TEMPLATES_DIR)

REGISTRY = ModelRegistry()
TRAINER = TrainingManager(on_exit=lambda: REGISTRY.refresh(force=True))
# build_registry() probes vip.csv for a pulse column and does file I/O; cache it once.
RULES = build_registry()
VOCAB = build_vocabulary(RULES)
BLOCKED_NOTE = (VOCAB.get("rules") or {}).get("note", "")


@app.on_event("startup")
def _startup():
    REGISTRY.refresh(force=True)
    h = REGISTRY.health()
    logging.info("API ready | model_loaded=%s | %s", h["model_loaded"],
                 h.get("path") or h.get("detail"))


# ------------------------------------------------------------------------- pages

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/schema")
def api_schema():
    return to_jsonable(VOCAB)


@app.get("/api/health")
def api_health():
    REGISTRY.refresh()
    return to_jsonable({**REGISTRY.health(), "training": TRAINER.status()})


@app.get("/api/model")
def api_model():
    REGISTRY.refresh()
    if not REGISTRY.loaded:
        return JSONResponse(status_code=404,
                            content={"detail": REGISTRY.health()["detail"]})
    return to_jsonable({**REGISTRY.describe(),
                        "sklearn_runtime": REGISTRY.health()["sklearn_runtime"],
                        "warnings": REGISTRY.warnings})


# ------------------------------------------------------------------------ predict

@app.post("/api/predict", response_model=None)
def api_predict(req: PredictRequest):
    """The advisory, plus every dashboard block.

    No `response_model`: `BPPredictor.predict()` returns a deliberately open dict -- `note`
    is conditional, `forecast` is keyed by signal, `personalisation` is passed straight
    through from OffsetModel. A declared response model would silently DROP any key it did
    not know about, which is the worst available failure mode for a clinical payload.
    """
    T = Timer()
    REGISTRY.refresh()
    predictor = REGISTRY.predictor
    as_of = pd.Timestamp(req.as_of) if req.as_of else None

    history = to_history(req)
    # Enrichments run over a bounded tail; predict() still sees everything, because its cost
    # is one vectorised build while the blocks below are O(n) in model calls.
    trunc = None
    hist_enrich = history
    if len(history) > S.ENRICH_READING_CAP:
        hist_enrich = history.tail(S.ENRICH_READING_CAP).reset_index(drop=True)
        trunc = {"submitted": len(history), "enrichment_readings": len(hist_enrich)}

    # ---- core advisory --------------------------------------------------------
    t0 = time.perf_counter()
    if predictor is not None:
        advisory = predictor.predict(history, as_of=as_of)
    else:
        last_ts = history.ts.max()
        now = as_of or last_ts
        advisory = {
            "patient_id": req.patient_id, "as_of": str(now), "model_version": None,
            "n_observations": int(history.sbp.notna().sum()),
            "confidence_tier": "no_model", "personalisation": None, "forecast": {},
            "early_warning": None,
            # From the constants, not from a bundle: the floor is a governance value and
            # does not stop existing because no model is loaded.
            "emergency_floor_mmHg": EMERGENCY_FLOOR_MMHG,
            "staleness": {"last_reading": str(last_ts),
                          "days_since_last_reading": round(
                              float((now - last_ts).total_seconds() / 86400.0), 1)},
            "note": "no trained model is loaded; the rule engine below is unaffected",
        }
    T.mark("predict", t0)

    threshold = (advisory.get("personalisation") or {}).get("threshold")
    if threshold is None and req.profile.provider_target is not None:
        threshold = float(req.profile.provider_target)

    out = dict(advisory)
    out["history"] = history_echo(req)

    # ---- rule engine ----------------------------------------------------------
    t0 = time.perf_counter()
    panel = to_engine_panel(req, hist_enrich, threshold)
    out["rule_engine"] = (rule_engine_block(RULES, panel, threshold, BLOCKED_NOTE)
                          if req.enrich.rule_engine else None)
    T.mark("engine", t0)

    # ---- model-dependent blocks ------------------------------------------------
    if predictor is not None:
        if req.enrich.predicted_alert:
            t0 = time.perf_counter()
            out["predicted_alert"] = predicted_alert_block(RULES, panel, advisory, threshold)
            T.mark("predicted_alert", t0)
        if req.enrich.anomaly or req.enrich.backtest:
            t0 = time.perf_counter()
            try:
                F = build_panel_frame(predictor, hist_enrich)
            except Exception as exc:                                 # noqa: BLE001
                logging.warning("feature frame unavailable: %s", exc)
                F = None
            T.mark("frame", t0)
            if F is not None and req.enrich.anomaly:
                t0 = time.perf_counter()
                out["anomaly"] = anomaly_block(predictor, F, hist_enrich, advisory)
                T.mark("anomaly", t0)
            if F is not None and req.enrich.backtest:
                t0 = time.perf_counter()
                out["backtest"] = backtest_block(predictor, F)
                T.mark("backtest", t0)
        if req.enrich.symptom_risk:
            t0 = time.perf_counter()
            out["symptom_risk"] = symptom_block(predictor, history, as_of)
            T.mark("symptom", t0)
    else:
        out["predicted_alert"] = {"horizons": [], "basis": "no model loaded",
                                  "symptom_note": ""}
        out["anomaly"] = None
        out["backtest"] = None
        out["symptom_risk"] = symptom_block(None, history)
        out["degraded"] = {
            "model_loaded": False,
            "reason": REGISTRY.health()["detail"],
            "remedy": "run `python main.py`, or POST /api/train",
            "still_available": ("the deterministic rule engine, which is the "
                                "safety-critical layer and needs no model"),
        }

    if trunc:
        out["truncated"] = trunc
    timings = T.total()
    budget = float(getattr(getattr(predictor, "config", None), "latency_budget_ms",
                           200.0) or 200.0)
    out["timings"] = timings
    # The budget was written for the serving path -- BPPredictor.predict -- not for the
    # dashboard extras. Judging the total against it would be measuring against a number
    # that was never about the total.
    out["budget"] = {"latency_budget_ms": budget,
                     "core_within_budget": timings.get("predict_ms", 0) <= budget,
                     "scope": "predict() only; the dashboard blocks are extra"}
    out["governance"] = {"population_threshold_mmHg": POPULATION_THRESHOLD_MMHG,
                         "emergency_floor_mmHg": EMERGENCY_FLOOR_MMHG}
    return to_jsonable(out)


# ------------------------------------------------------------------------ training

@app.post("/api/train")
def api_train():
    ok, detail = TRAINER.start()
    return JSONResponse(status_code=200 if ok else 409,
                        content={"started": ok, "detail": detail,
                                 "warning": S.TRAIN_WARNING})


@app.get("/api/train/status")
def api_train_status():
    return to_jsonable(TRAINER.status())


@app.post("/api/train/cancel")
def api_train_cancel():
    ok, detail = TRAINER.cancel()
    return JSONResponse(status_code=200 if ok else 409,
                        content={"cancelled": ok, "detail": detail})


# ------------------------------------------------------------------- error handling

@app.exception_handler(RequestValidationError)
def _validation(request: Request, exc: RequestValidationError):
    # Flattened to one string: the client shows `detail` verbatim in a single banner.
    parts = []
    for e in exc.errors():
        loc = ".".join(str(x) for x in e.get("loc", []) if x != "body")
        parts.append(f"{loc}: {e.get('msg')}" if loc else str(e.get("msg")))
    return JSONResponse(status_code=422, content={"detail": " · ".join(parts[:4])})


@app.exception_handler(CustomException)
def _custom(request: Request, exc: CustomException):
    rid = uuid.uuid4().hex[:8]
    # CustomException.__str__ embeds absolute source paths. Logged, never returned.
    logging.error("[%s] %s", rid, exc)
    return JSONResponse(status_code=500,
                        content={"detail": "internal error while scoring; see server logs",
                                 "request_id": rid})


@app.exception_handler(Exception)
def _unhandled(request: Request, exc: Exception):
    rid = uuid.uuid4().hex[:8]
    logging.exception("[%s] unhandled", rid)
    return JSONResponse(status_code=500,
                        content={"detail": f"{type(exc).__name__}", "request_id": rid})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")), workers=1)
