"""Build one advisory. The single orchestration both front ends call.

`app.py` (FastAPI + the SPA) and `gradio_app.py` (the Hugging Face Space) each need exactly
this sequence: predict, then rule engine, then the dashboard blocks. Putting it here means
there is one implementation to reason about. Two front ends each assembling their own
advisory is how the number on a chart and the number in a banner start disagreeing.
"""

import time

import pandas as pd

from src.constants.training_pipeline import EMERGENCY_FLOOR_MMHG, POPULATION_THRESHOLD_MMHG
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
from src.serving.mapping import to_engine_panel, to_history


def build_advisory(req, predictor, rules, blocked_note: str = "",
                   degraded_reason: str = "") -> dict:
    """The full advisory payload for one request.

    `predictor` may be None. That is the primary path on a fresh checkout, and the rule
    engine below is unaffected by it -- an SBP of 195 still produces an emergency.
    """
    T = Timer()
    as_of = pd.Timestamp(req.as_of) if req.as_of else None
    history = to_history(req)

    # Enrichments run over a bounded tail; predict() still sees everything, because its cost
    # is one vectorised build while the blocks below are O(n) in model calls.
    trunc, hist_enrich = None, history
    if len(history) > S.ENRICH_READING_CAP:
        hist_enrich = history.tail(S.ENRICH_READING_CAP).reset_index(drop=True)
        trunc = {"submitted": len(history), "enrichment_readings": len(hist_enrich)}

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

    t0 = time.perf_counter()
    panel = to_engine_panel(req, hist_enrich, threshold)
    out["rule_engine"] = (rule_engine_block(rules, panel, threshold, blocked_note)
                          if req.enrich.rule_engine else None)
    T.mark("engine", t0)

    if predictor is not None:
        if req.enrich.predicted_alert:
            t0 = time.perf_counter()
            out["predicted_alert"] = predicted_alert_block(rules, panel, advisory, threshold)
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
            out["symptom_risk"] = symptom_block(predictor, history, as_of,
                                                full=req.enrich.symptom_risk_full)
            T.mark("symptom", t0)
    else:
        out["predicted_alert"] = {"horizons": [], "basis": "no model loaded",
                                  "symptom_note": ""}
        out["anomaly"] = None
        out["backtest"] = None
        out["symptom_risk"] = symptom_block(None, history)
        out["degraded"] = {
            "model_loaded": False,
            "reason": degraded_reason or "no bundle on disk",
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
    # dashboard extras. Judging the total against it would measure against a number that was
    # never about the total.
    out["budget"] = {"latency_budget_ms": budget,
                     "core_within_budget": timings.get("predict_ms", 0) <= budget,
                     "scope": "predict() only; the dashboard blocks are extra"}
    out["governance"] = {"population_threshold_mmHg": POPULATION_THRESHOLD_MMHG,
                         "emergency_floor_mmHg": EMERGENCY_FLOOR_MMHG}
    return out


def banner_for(d: dict) -> tuple:
    """(severity, title, detail) using the clinical priority order.

    Emergency before detector before forecast before engine-fired. Shared so the Space and
    the SPA cannot rank the same advisory differently.
    """
    pers = d.get("personalisation") or {}
    ew = d.get("early_warning") or {}
    eng = (d.get("rule_engine") or {}).get("current") or {}
    horizons = (d.get("predicted_alert") or {}).get("horizons") or []
    first_fire = next((h for h in horizons if h.get("fired")), None)

    def pretty(s):
        return str(s or "").replace("RULE_", "").replace("_", " ").capitalize()

    if eng.get("is_emergency"):
        return ("critical", "Rule engine fired an emergency on the latest reading",
                f"{pretty(eng.get('rule_id'))}. The emergency floor is never personalised "
                f"and the engine is authoritative here.")
    if ew.get("flagged"):
        return ("critical", "Early-warning detector flagged this patient",
                f"Score {ew.get('score')} at or above the {ew.get('budget_pct')}% cut of "
                f"{ew.get('cut')}, roughly {ew.get('est_lead_days')} days of lead time.")
    if first_fire:
        return ("watch",
                f"{pretty(first_fire.get('tier'))} forecast in about "
                f"{first_fire.get('days_ahead')} days",
                f"Predicted SBP {first_fire.get('sbp')} mmHg would fire "
                f"{pretty(first_fire.get('rule_id'))} if the trend holds.")
    if eng.get("fired"):
        return ("watch", f"{pretty(eng.get('tier'))} on the latest reading",
                f"{pretty(eng.get('rule_id'))}. No further breach forecast.")
    tier = d.get("confidence_tier")
    if tier == "stale":
        return ("info", "History too old for a forecast", d.get("note") or "")
    if tier == "cold_start":
        return ("info", "Cold start — no forecast issued", d.get("note") or "")
    if tier == "no_model":
        return ("info", "Rule engine only — no model loaded",
                "Nothing fired on the latest reading.")
    return ("good", "Nothing due",
            f"The engine fired nothing, the forecast stays below the personalised threshold "
            f"of {pers.get('threshold')} mmHg, and the detector is below its cut.")
