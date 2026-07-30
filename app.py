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
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.serving import settings as S
from src.serving.advisory import build_advisory
from src.serving.jsonify import to_jsonable
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
    # Request first. Starlette's old `TemplateResponse(name, context)` signature is gone in
    # current releases -- passing the name first makes it the `request` argument and the
    # context dict becomes the template name, which surfaces as an unhashable-dict TypeError
    # from deep inside Jinja's cache. Older releases accept this form too, so it is the one
    # that works across the range `fastapi>=0.141.1` actually resolves to.
    return templates.TemplateResponse(request, "index.html")


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

    The assembly lives in `src/serving/advisory.py` because the Hugging Face Space
    (`gradio_app.py`) needs the identical sequence, and two front ends each assembling their
    own advisory is how a banner and a chart start disagreeing about the same patient.
    """
    REGISTRY.refresh()
    return to_jsonable(build_advisory(req, REGISTRY.predictor, RULES, BLOCKED_NOTE,
                                      degraded_reason=REGISTRY.health()["detail"]))


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
