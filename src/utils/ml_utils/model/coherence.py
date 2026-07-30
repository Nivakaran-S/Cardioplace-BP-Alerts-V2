"""Is the predicted (systolic, diastolic) pair a blood pressure a body could have?

Each signal's winning architecture is selected independently -- `select_and_decide` ranks per
signal and nothing ever looks at the joint (`regression_metric.py:196-218`). So SBP might ship
`hgb_delta` and DBP `ridge`, each excellent on its own marginal MAE, and nothing prevents them
from jointly emitting a diastolic above the systolic, or a pulse pressure of 8.

That is not a hypothetical nuisance. Pulse pressure is a rule axis: `engine.py:91-97` fires
RULE_PULSE_PRESSURE_WIDE and RULE_PULSE_PRESSURE_NARROW on `sbp - dbp`, and
`predicted_alert_block` pushes the predicted pair through the engine. An incoherent pair can
therefore raise a clinical watch banner that is an artefact of two models disagreeing.

The bound is not invented here. `schemas.Reading` already rejects a *submitted* reading with
`sbp - dbp < INGEST_RANGES["min_pulse_pressure"]`, and `data_ingestion` applies the same filter
to the corpus, so no such pair was ever trained on and none can be sent in. This module applies
that existing input contract to the output, which is why the constants are imported rather than
restated -- a second copy would drift.

Two design constraints, both load-bearing:

  * **Nothing here mutates a forecast.** These functions return a verdict; the caller attaches
    it. The point estimate is reported exactly as the model produced it, flagged. Clipping a
    forecast to look sensible would hide the disagreement that is the actual finding.

  * **This must never run inside `FinalForecaster.predict`.** `explain_prediction` perturbs one
    feature at a time and reads the change in the prediction; a clip would flatten those
    perturbations to zero and hand back an all-zero attribution -- reproducing exactly the
    failure critical gate 6 exists to catch (`gates.py:344-354`). Keep it at the advisory
    boundary, downstream of every `.predict` call.
"""

import numpy as np

from src.constants.training_pipeline import INGEST_RANGES, SCHEMA_RANGES

#: How far the predicted pulse pressure may sit from the patient's own recent mean, in units of
#: the cohort's absolute-deviation p99. Loose on purpose: this check exists to catch a joint
#: that is obviously wrong, not to second-guess a forecast that is merely surprising.
PP_DEVIATION_K: float = 1.0

_SBP_LO, _SBP_HI = SCHEMA_RANGES.get("sbp", INGEST_RANGES["sbp"])
_DBP_LO, _DBP_HI = SCHEMA_RANGES.get("dbp", INGEST_RANGES["dbp"])
_MIN_PP = float(INGEST_RANGES["min_pulse_pressure"])


def check_bp_pair(sbp, dbp, *, pp_ref=None, bounds=None) -> dict:
    """Verdict on one predicted pair. Returns; never mutates, never clips.

    `pp_ref` is the patient's own recent pulse pressure (`pp_mean7`, already a feature) and
    `bounds` is the cohort reference from `bundle["coherence"]`. Both are optional: a bundle
    trained before this existed simply gets the three absolute checks, and the check that needs
    a reference records itself as `skipped` rather than silently passing. Absent evidence is not
    evidence of coherence -- the same rule the safety gates use.

    Returns `{ok, pp, violations, checked, skipped, basis}`. `ok` is False iff at least one
    check that actually ran failed.
    """
    # `violations` is prose for a human reading one advisory; `codes` is the stable vocabulary
    # the batch report aggregates on. Grouping the prose would bucket by the mmHg value in the
    # message and produce one "category" per row, which is not a summary of anything.
    out = {"ok": True, "pp": None, "violations": [], "codes": [], "checked": [], "skipped": [],
           "basis": "input contract applied to the output; see coherence.py"}

    def fail(code, msg):
        out["codes"].append(code)
        out["violations"].append(msg)

    if sbp is None or dbp is None or not np.isfinite(sbp) or not np.isfinite(dbp):
        out["ok"] = False
        fail("leg_missing", "a leg of the pair is missing or non-finite")
        return out

    sbp, dbp = float(sbp), float(dbp)
    pp = sbp - dbp
    out["pp"] = round(pp, 1)

    # 1. Ordering. Degenerate, and the failure two independent regressors actually produce.
    out["checked"].append("dbp_below_sbp")
    if pp <= 0:
        fail("inverted",
             f"diastolic {dbp:.1f} is not below systolic {sbp:.1f} (pulse pressure {pp:+.1f})")

    # 2. The same floor `schemas.Reading` enforces on submitted readings. A predicted pair the
    #    API would have refused as an input is incoherent by the project's own definition.
    out["checked"].append("min_pulse_pressure")
    if 0 < pp < _MIN_PP:
        fail("below_pp_floor",
             f"pulse pressure {pp:.1f} is below the {_MIN_PP:.0f} mmHg floor that "
             f"schemas.Reading applies to submitted readings")

    # 3. Each leg inside the admission range the corpus was filtered to.
    out["checked"].append("legs_in_range")
    if not (_SBP_LO <= sbp <= _SBP_HI):
        fail("sbp_out_of_range", f"systolic {sbp:.1f} outside [{_SBP_LO}, {_SBP_HI}]")
    if not (_DBP_LO <= dbp <= _DBP_HI):
        fail("dbp_out_of_range", f"diastolic {dbp:.1f} outside [{_DBP_LO}, {_DBP_HI}]")

    # 4. Against the patient's own pulse pressure. The only check that can catch two
    #    individually plausible marginals forming an implausible joint: 150/95 and 150/70 are
    #    both unremarkable pairs, but not for the same patient in the same week.
    tol = None if bounds is None else bounds.get("pp_abs_dev_p99")
    if pp_ref is None or not np.isfinite(pp_ref) or tol is None or not np.isfinite(tol):
        out["skipped"].append(
            "pp_near_patient_baseline: no pp_mean7 for this patient"
            if pp_ref is None or not np.isfinite(pp_ref)
            else "pp_near_patient_baseline: bundle carries no coherence reference")
    else:
        out["checked"].append("pp_near_patient_baseline")
        dev, lim = abs(pp - float(pp_ref)), PP_DEVIATION_K * float(tol)
        if dev > lim:
            fail("pp_far_from_baseline",
                 f"pulse pressure {pp:.1f} is {dev:.1f} mmHg from this patient's recent "
                 f"{float(pp_ref):.1f}, beyond the {lim:.1f} mmHg cohort bound")

    out["ok"] = not out["violations"]
    return out


def attach_pair_coherence(forecast: dict, *, pp_ref=None, bounds=None) -> dict | None:
    """Check every (sbp, dbp) horizon pair in an advisory's `forecast` and attach the verdict.

    Mutates the nodes it is given -- that is the point, the caller wants the verdict on the
    payload -- but only ever ADDS a `coherence` key. It never touches `point`, so a flagged
    forecast is still reported exactly as the model produced it.

    Returns the roll-up for `out["forecast_coherence"]`, or None when there is no pair to check
    (dbp not forecast, or a cold-start/stale advisory with no forecast at all). None means "not
    applicable", which is different from `{"ok": True}` and is why it is not a bare bool.
    """
    sbp_fc = (forecast or {}).get("sbp") or {}
    dbp_fc = (forecast or {}).get("dbp") or {}
    if not sbp_fc or not dbp_fc:
        return None

    flagged = 0
    for key, snode in sbp_fc.items():
        dnode = dbp_fc.get(key)
        if not isinstance(snode, dict) or not isinstance(dnode, dict):
            continue
        v = check_bp_pair(snode.get("point"), dnode.get("point"),
                          pp_ref=pp_ref, bounds=bounds)
        snode["coherence"] = v
        dnode["coherence"] = v
        flagged += 0 if v["ok"] else 1
    return {
        "ok": flagged == 0, "n_flagged": flagged, "n_pairs": len(sbp_fc),
        "reference": ("bundle" if bounds else
                      "absolute checks only; this bundle predates the cohort reference"),
    }


def coherence_reference(panel, *, sbp_col: str = "sbp", dbp_col: str = "dbp") -> dict:
    """Cohort pulse-pressure reference for `bundle["coherence"]`.

    Six floats off the train+val rows the caller already holds. Quantiles rather than min/max so
    a single mis-keyed session cannot set the bound.
    """
    s = panel[sbp_col].astype(float)
    d = panel[dbp_col].astype(float)
    pp = (s - d).replace([np.inf, -np.inf], np.nan).dropna()
    if pp.empty:
        return {"basis": "no finite pulse pressure in the reference panel"}
    # Dispersion of |pp - rolling baseline| is what check 4 compares against, but a per-patient
    # rolling pass here would cost a groupby over the whole panel for six numbers. The spread of
    # pp about its own median is a sound stand-in and is deliberately generous.
    dev = (pp - pp.median()).abs()
    return {
        "pp_floor": _MIN_PP,
        "pp_p001": round(float(pp.quantile(0.001)), 2),
        "pp_p50": round(float(pp.median()), 2),
        "pp_p999": round(float(pp.quantile(0.999)), 2),
        "pp_abs_dev_p99": round(float(dev.quantile(0.99)), 2),
        "n": int(pp.size),
        "basis": "train+val pulse-pressure quantiles; floor from "
                 "data_schema/schema.yaml min_pulse_pressure",
    }


def pair_coherence(pred_sbp, pred_dbp, *, pp_ref=None, bounds=None, horizon=None) -> dict:
    """Batch verdict over a whole evaluation split, for the offline report.

    Nothing in training has ever inspected the joint, so this is the first measurement of
    whether the shipped pair is coherent on real rows. Report it before writing a gate on it --
    a rate of zero makes the guard cheap insurance, a rate above zero makes it a bug fix.
    """
    a = np.asarray(pred_sbp, dtype=float)
    b = np.asarray(pred_dbp, dtype=float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    ref = (np.full(n, np.nan) if pp_ref is None
           else np.asarray(pp_ref, dtype=float)[:n])

    rows = [check_bp_pair(a[i], b[i], pp_ref=ref[i], bounds=bounds) for i in range(n)]
    bad = [r for r in rows if not r["ok"]]
    pp = a - b
    finite = pp[np.isfinite(pp)]
    # Aggregate on `codes`, not on the prose. The messages embed mmHg values, so bucketing them
    # produced one "category" per row -- a list of individual failures wearing the name of a
    # summary.
    reasons: dict = {}
    for r in bad:
        for c in r["codes"]:
            reasons[c] = reasons.get(c, 0) + 1
    return {
        "horizon": horizon, "n": n, "n_violations": len(bad),
        "violation_rate": round(len(bad) / n, 6) if n else None,
        "pp_min": round(float(finite.min()), 1) if finite.size else None,
        "pp_median": round(float(np.median(finite)), 1) if finite.size else None,
        "pp_max": round(float(finite.max()), 1) if finite.size else None,
        "n_inverted": int((pp <= 0).sum()),
        "n_below_floor": int(((pp > 0) & (pp < _MIN_PP)).sum()),
        "top_reasons": "; ".join(f"{k} x{v}" for k, v in
                                 sorted(reasons.items(), key=lambda kv: -kv[1])[:3]),
    }
