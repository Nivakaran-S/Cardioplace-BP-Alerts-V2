"""Serving configuration: paths, caps, feature flags.

`src/serving/**` and `app.py` may import from `src/utils/**`, `src/constants/**`,
`src/entity/config_entity.py`, `src/logging/` and `src/exception/`. They may NOT import
`src/components/**` or `src/pipeline/**`.

That is not a style preference. The training graph pulls in mlflow, the whole component
chain and every heavy dependency; if the API imported it, one broken training module would
take the API down with it -- and the API's most important job is the one it does with no
model at all, running the deterministic rule engine. The training subprocess is the only
thing allowed to touch that side. CI enforces the rule with a grep.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _p(*parts) -> str:
    return os.path.join(REPO_ROOT, *parts)


# --- model discovery, in priority order -------------------------------------------
MODEL_PATH_ENV = "CARDIOPLACE_MODEL_PATH"
FINAL_MODEL_PATH = _p("final_model", "model.pkl")
ARTIFACTS_GLOB = _p("Artifacts", "*", "model_trainer", "trained_model", "predictor.joblib")

STATIC_DIR = _p("static")
TEMPLATES_DIR = _p("templates")
LOG_DIR = _p("logs")

# --- request limits ----------------------------------------------------------------
#: Hard cap on submitted readings. Above this the request is refused with 413 rather than
#: quietly truncated -- a silently shortened history produces a different forecast.
MAX_READINGS = 400
#: Enrichments (charts, backtest, engine timeline) use only the most recent N readings.
#: `predict()` still sees the whole history: its cost is one vectorised feature build, while
#: the enrichments are O(n) in model calls.
ENRICH_READING_CAP = 240

# --- hot reload ---------------------------------------------------------------------
#: Seconds between stat() calls on the model path during a request burst.
RELOAD_DEBOUNCE_S = 2.0
#: A promotion writes the pickle; a reader that catches it mid-write sees a truncated file.
#: `save_object` writes atomically, but a model dropped in by hand may not, so the loader
#: also requires (mtime, size) to be unchanged across two stats this far apart.
STABILITY_WINDOW_S = 0.25

# --- training subprocess -------------------------------------------------------------
TRAIN_TAIL_LINES = 500
TRAIN_WARNING = (
    "A full run streams data/vip.csv (251 MB, 4.37M readings), rebuilds the feature panel "
    "and bakes off the whole architecture registry. Expect tens of minutes at the default "
    "tier and hours at MODEL_TIER=all. Training locally is almost always the better choice; "
    "a free-tier hosted container will not finish.")
