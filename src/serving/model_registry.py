"""Find, load, validate and hot-reload the serving bundle.

The loader has to SNIFF, because the pipeline writes two different on-disk shapes:

  final_model/model.pkl        `save_object(path, predictor)` -> a pickled BPPredictor
  .../predictor.joblib         `predictor.save(path)`         -> the bundle dict

`BPPredictor.load` assumes the second. Point it at the first and you get
`BPPredictor(bundle=<BPPredictor>)`, which fails later with `TypeError: 'BPPredictor' object
is not subscriptable` -- a confusing error a long way from its cause. Both shapes are valid
artifacts, so the loader accepts both rather than making one of the writers wrong.
"""

import glob
import os
import sys
import threading
import time
import warnings

import pandas as pd

from src.logging.logger import logging
from src.serving import settings as S


class ModelRegistry:
    """Holds at most one good predictor, and never replaces it with a worse one."""

    def __init__(self):
        self._lock = threading.Lock()
        self.predictor = None
        self.path = None
        self.error = None
        self.source = None
        self.loaded_at = None
        self.warnings = []
        self._stamp = None
        self._last_stat = 0.0

    # ---------------------------------------------------------------- discovery
    @staticmethod
    def candidate_paths() -> list:
        out = []
        env = os.getenv(S.MODEL_PATH_ENV)
        if env:
            out.append(env)
        out.append(S.FINAL_MODEL_PATH)
        # Newest run's own artifact: a dev convenience so the API is usable straight after a
        # training run that was BLOCKED from promotion and therefore never wrote final_model.
        out.extend(sorted(glob.glob(S.ARTIFACTS_GLOB), key=os.path.getmtime, reverse=True))
        return [p for p in out if p]

    @staticmethod
    def _stamp_of(path: str):
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)

    # ---------------------------------------------------------------- loading
    def _load(self, path: str):
        """Sniff the artifact shape, then prove it works before publishing it."""
        import joblib

        from src.utils.ml_utils.model.estimator import BPPredictor

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw = joblib.load(path)          # reads a plain pickle.dump file too
            if isinstance(raw, BPPredictor):
                predictor, shape = raw, "pickled BPPredictor (save_object)"
            elif isinstance(raw, dict) and "forecasters" in raw:
                predictor, shape = BPPredictor(raw), "bundle dict (predictor.save)"
            else:
                raise ValueError(f"unrecognised artifact: {type(raw).__name__}")
            notes = [str(w.message)[:200] for w in caught
                     if "InconsistentVersion" in type(w.message).__name__]

        required = ("model_version", "feature_names", "forecasters", "detector", "config")
        missing = [k for k in required if k not in predictor.b]
        if missing:
            raise ValueError(f"bundle is missing {missing}")
        self._smoke(predictor)
        return predictor, shape, notes

    @staticmethod
    def _smoke(predictor):
        """Score a synthetic 12-reading history before trusting the bundle.

        A model that unpickles but cannot predict is worse than no model: the API would
        report healthy and 500 on every request. This is the difference between publishing
        a broken artifact and keeping the last good one.
        """
        n = 12
        hist = pd.DataFrame({
            "patient_id": ["_smoke"] * n,
            "ts": pd.date_range("2024-01-01", periods=n, freq="2D"),
            "sbp": [138.0, 142, 145, 140, 150, 147, 139, 144, 152, 148, 141, 146],
            "dbp": [80.0, 82, 84, 79, 86, 85, 78, 83, 88, 84, 80, 82],
            "idwg": [2.2] * n, "weight": [70.0] * n,
            "sbp_drop": [float("nan")] * n, "uf_total": [float("nan")] * n,
            "age": [65.0] * n, "is_male": [1] * n, "is_dm": [0] * n, "DM": [0.0] * n,
            "n_meas": [2] * n})
        out = predictor.predict(hist)
        if not isinstance(out, dict) or "confidence_tier" not in out:
            raise ValueError("smoke prediction did not return an advisory")

    def refresh(self, force: bool = False) -> bool:
        """Reload if the artifact changed. Returns True when the predictor was replaced."""
        now = time.monotonic()
        if not force and (now - self._last_stat) < S.RELOAD_DEBOUNCE_S:
            return False
        self._last_stat = now

        path = next((p for p in self.candidate_paths() if os.path.exists(p)), None)
        if path is None:
            with self._lock:
                if self.predictor is None:
                    self.error = ("no model artifact found. Run `python main.py`, or POST "
                                  "/api/train. The rule engine works without one.")
            return False
        try:
            stamp = self._stamp_of(path)
            if not force and path == self.path and stamp == self._stamp:
                return False
            # Require the file to be stable across two stats before reading it: a promotion
            # in flight would otherwise be read half-written.
            time.sleep(S.STABILITY_WINDOW_S)
            if self._stamp_of(path) != stamp:
                logging.info("model artifact still being written; will retry")
                return False

            predictor, shape, notes = self._load(path)
        except Exception as exc:                                     # noqa: BLE001
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
            # Keep serving the previous good model rather than going dark on a bad write.
            logging.error("model load failed (%s); keeping the previous model (%s)",
                          self.error, "none" if self.predictor is None else self.path)
            return False

        with self._lock:
            self.predictor, self.path, self._stamp = predictor, path, stamp
            self.source, self.error, self.warnings = shape, None, notes
            self.loaded_at = pd.Timestamp.utcnow().isoformat()
        logging.info("loaded %s from %s (%s)", predictor.b.get("model_version"), path, shape)
        return True

    # ---------------------------------------------------------------- reporting
    @property
    def loaded(self) -> bool:
        return self.predictor is not None

    def health(self) -> dict:
        import sklearn
        return {
            "model_loaded": self.loaded,
            "detail": self.error,
            "path": self.path,
            "path_searched": self.candidate_paths(),
            "artifact_shape": self.source,
            "loaded_at": self.loaded_at,
            "sklearn_runtime": sklearn.__version__,
            "python": sys.version.split()[0],
            # A 1.7.x runtime against a 1.8.0 pickle warns about invalid results. Silent
            # numeric drift on a clinical model is not acceptable, so it is surfaced.
            "warnings": self.warnings,
        }

    def describe(self) -> dict:
        b = self.predictor.b
        c = self.predictor.config
        return {
            "model_version": b.get("model_version"),
            "run_id": b.get("run_id"),
            "n_features": len(b.get("feature_names", [])),
            "selected_family": b.get("selected_family", {}),
            "shipped": {k: f"{v[0]}:{v[1]}" for k, v in (b.get("shipped") or {}).items()},
            "selected_but_unservable": b.get("selected_but_unservable") or {},
            "detector": (b.get("detector") or {}).get("name"),
            "symptom_labels_synthetic": b.get("symptom_labels_synthetic"),
            "governance": {
                "population_threshold_mmHg": c.population_threshold_mmHg,
                "emergency_floor_mmHg": c.emergency_floor_mmHg,
                "offset_cap_loosen": c.offset_cap_loosen,
                "offset_cap_tighten": c.offset_cap_tighten,
                "alert_budget_pct": c.alert_budget_pct,
                "warn_window": c.warn_window,
                "event_quantile": c.event_quantile,
                "stale_forecast_max_days": getattr(c, "stale_forecast_max_days", 14),
                "cold_start_min_readings": c.cold_start_min_readings,
                "steady_state_readings": c.steady_state_readings,
                "latency_budget_ms": c.latency_budget_ms,
            },
        }
