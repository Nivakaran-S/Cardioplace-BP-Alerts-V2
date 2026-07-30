"""Symptom risk conditioned on the FORECAST rather than on today.

`symptom_block` scores the heads from `transform_for_inference(history)` -- the observed
present. So the dashboard says "your systolic will be 158 next session" and, separately, "your
dizziness risk *today* is 4%". Those are two answers to two different questions presented as
one panel. `predicted_alert_block` already closed the equivalent gap for the rule engine, and
its own docstring names what is still missing:

    "Today's symptoms are carried forward onto every forecast row. A symptom-driven tier on a
     horizon card reflects today's symptoms combined with a predicted blood pressure -- it is
     not a predicted symptom."

This module produces the predicted symptom. It forecasts the vitals, extends the history with
them, rebuilds the causal features, and scores the heads on that row.

## Why a chain rather than one joint model

The symptom labels are generated (`symptom_layer.py`) as

    sym[t] ~ Bernoulli(sigmoid(b0 + B_sbp*relu(sbp[t]-140) + B_drop*drop[t] + ... ))

so the symptom target at t+h is a noisy monotone function of the SBP target at t+h -- the
quantity Model 1 already forecasts. A direct classifier from features<=t-1 must implicitly
learn BOTH the forecast AND the link, from a binary signal at a 0.1-12% base rate. Forecasting
the continuous covariate (real-valued target, 100% label density) and then applying the link is
the better-specified factorisation of the same quantity.

It also keeps the provenance separation intact. The dependency runs one way, at inference time
only: nothing here can change the forecaster's weights, so "the BP forecast is uncontaminated
by generated labels" stays structurally true rather than becoming an argument.

## Why the distribution and not the point estimate

`relu(sbp - 140)` is convex, so by Jensen `E[relu(SBP-140)] > relu(E[SBP]-140)` for any
forecast with spread near the threshold -- and the gap is largest exactly at the threshold.
With this cohort's noise (~11 mmHg), a patient forecast at 138 gets `relu(-2) = 0` from a point
estimate: **zero excess risk**, while the correct marginal is about 3.6 mmHg of excess, roughly
22% higher odds. Plugging in the point estimate does not merely lose precision, it reports "no
elevated risk" for every patient sitting just below the threshold. That is a systematic bias
with a known sign, at the operating point that matters most.

So where a predictive interval exists, the head is integrated over it by 3-point
Gauss-Hermite. Where one does not, that is reported per node rather than papered over.

## What this cannot reach

Chaining only helps where the driver is forecastable AND present at serving:

  hypertensive (6 symptoms)  driver sbp_ex   -> forecast, present     -> helped
  volume (2)                 driver idwg/dw  -> forecast, needs dryweight
  hypotensive (4)            driver sbp_drop -> NOT a forecast signal -> unreachable
  drug (3)                   driver on_ace   -> static                -> unaffected

Four of fifteen symptoms cannot benefit from this path at all, because `sbp_drop` is an
intradialytic measurement that is neither forecast nor available at serving. Stated here and
in the payload, because "coherent joint BP and symptom prediction" implies otherwise.
"""

import numpy as np
import pandas as pd

#: z at the 90th percentile of the standard normal -- an 80% central interval spans +/- this.
_Z80 = 1.2815515655446004

#: 3-point Gauss-Hermite over a Gaussian: nodes at mu, mu +/- sqrt(3)*sigma with weights
#: 2/3, 1/6, 1/6. Exact for polynomials to degree 5, which is ample for a smooth sigmoid, and
#: it costs three feature builds instead of the dozens a Monte-Carlo pass would need.
_GH_NODES = ((0.0, 2.0 / 3.0), (np.sqrt(3.0), 1.0 / 6.0), (-np.sqrt(3.0), 1.0 / 6.0))

#: Columns a forecast does not produce. Carrying today's value forward would make every
#: rolling moment describe a series that never happened; NaN says "unknown", which is true.
_UNKNOWN_ON_FORECAST = ("weight", "dryweight", "sbp_drop", "uf_total", "uf_rate", "idwg_rel",
                        "heart_rate", "pulse", "temperature", "n_meas",
                        "took_all_meds", "missed_antihypertensive")


def _sigma_from_node(node: dict):
    """Forecast SD implied by an 80% conformal band, or None if the node carries no band."""
    lo, hi = node.get("lo80"), node.get("hi80")
    if lo is None or hi is None:
        return None
    w = float(hi) - float(lo)
    return w / (2.0 * _Z80) if w > 0 else None


def chained_history(history: pd.DataFrame, forecast: dict, upto_h: int,
                    sbp_shift: float = 0.0) -> pd.DataFrame:
    """`history` extended by the forecast readings for horizons 0..upto_h.

    Every non-forecast column on the appended rows is NaN, including the symptom flags: the
    symptoms at a future session are precisely what is being predicted, so carrying today's
    forward would feed the head its own answer for the horizon it is being asked about.

    `sbp_shift` displaces the systolic on the appended rows, which is how the quadrature nodes
    are applied. It is applied at the READING level rather than to a feature, so every
    derived moment -- lag1, mean3, std3, the z-score -- moves consistently. Perturbing
    `sbp_lag1` while `sbp_mean3` still reflected the point estimate would be incoherent in
    exactly the way this module exists to avoid.
    """
    if history is None or not len(history):
        return history
    sbp_fc = (forecast or {}).get("sbp") or {}
    dbp_fc = (forecast or {}).get("dbp") or {}
    idwg_fc = (forecast or {}).get("idwg") or {}
    if not sbp_fc:
        return history

    last = history.iloc[-1]
    gap = history.ts.diff().dt.days.median()
    gap = float(gap) if pd.notna(gap) and gap > 0 else 2.0

    rows = []
    for h in range(0, int(upto_h) + 1):
        node = sbp_fc.get(f"h{h}")
        if not isinstance(node, dict) or node.get("point") is None:
            break
        r = {c: np.nan for c in history.columns}
        # Identity and demographics persist; they are properties of the patient, not readings.
        for c in ("patient_id", "age", "is_male", "is_dm", "DM"):
            if c in history.columns:
                r[c] = last[c]
        r["ts"] = pd.Timestamp(last.ts) + pd.Timedelta(
            days=float(node.get("days_ahead_est") or (h + 1) * gap))
        r["sbp"] = float(node["point"]) + float(sbp_shift)
        dn = dbp_fc.get(f"h{h}")
        if isinstance(dn, dict) and dn.get("point") is not None:
            r["dbp"] = float(dn["point"])
        inode = idwg_fc.get(f"h{h}")
        if isinstance(inode, dict) and inode.get("point") is not None:
            r["idwg"] = float(inode["point"])
        for c in _UNKNOWN_ON_FORECAST:
            if c in history.columns:
                r[c] = np.nan
        rows.append(r)

    if not rows:
        return history
    return pd.concat([history, pd.DataFrame(rows)], ignore_index=True)


def _score_heads(predictor, hist: pd.DataFrame, keys, as_of=None) -> dict:
    """`{head_key: probability}` for one feature row built from `hist`."""
    row = predictor.fb.transform_for_inference(hist, as_of=as_of)
    X = row.reindex(columns=predictor.b["feature_names"])
    out = {}
    for k in keys:
        try:
            out[k] = float(predictor.b["symptom_models"][k].predict_proba(X)[0, 1])
        except Exception:                                             # noqa: BLE001
            continue
    return out


def chained_symptom_risk(predictor, history, advisory, as_of=None, *,
                         marginalise: bool = True, red_flags_only: bool = False) -> dict:
    """Symptom probabilities at each forecast horizon, conditioned on the predicted vitals.

    Uses the `_h0` head at EVERY horizon. `_h0` answers "what is the risk at the next step";
    extending the history by h+1 predicted readings moves "next step" to t+h+1. That matches
    the `steps_ahead = h+1` convention `predict()` already uses, and it means this path needs
    ~15 heads rather than 45.

    No `flagged` boolean is returned. The operating cut was chosen as a percentile of the
    probabilities the head produced on OBSERVED rows (`classifier_head.py:207-209`), and a
    chained row is a different input distribution, so that cut does not carry its alert-budget
    meaning across. Reusing it would silently break the budget the patient-disjoint
    calibration/threshold split exists to protect. The cut is reported for reference and
    explicitly marked as not applicable until a chained cut is fitted at training time.
    """
    models = (predictor.b.get("symptom_models") if predictor is not None else None) or {}
    if not models:
        return {"available": False, "reason": "this bundle carries no symptom heads",
                "items": []}

    fc = (advisory or {}).get("forecast") or {}
    if not (fc.get("sbp") or {}):
        return {"available": False, "reason": "no forecast was issued, so there is nothing "
                                              "to condition on", "items": []}

    reds = set(predictor.b.get("symptom_red_flags") or [])
    mech = predictor.b.get("symptom_mechanism") or {}
    cuts = predictor.b.get("symptom_cuts") or {}
    h0_keys = [k for k in models if k.endswith("_h0")]
    if red_flags_only:
        h0_keys = [k for k in h0_keys if k.rsplit("_h", 1)[0] in reds]
    if not h0_keys:
        return {"available": False, "reason": "this bundle has no next-session heads",
                "items": []}

    horizons = sorted(int(k[1:]) for k in (fc.get("sbp") or {}) if k.startswith("h"))
    items, bases = [], {}
    for h in horizons:
        node = (fc.get("sbp") or {}).get(f"h{h}") or {}
        sigma = _sigma_from_node(node) if marginalise else None

        if sigma:
            acc, wsum = {}, 0.0
            for z, w in _GH_NODES:
                probs = _score_heads(predictor,
                                     chained_history(history, fc, h, sbp_shift=z * sigma),
                                     h0_keys, as_of)
                for k, p in probs.items():
                    acc[k] = acc.get(k, 0.0) + w * p
                wsum += w
            marg = {k: v / wsum for k, v in acc.items()}
            point = _score_heads(predictor, chained_history(history, fc, h), h0_keys, as_of)
            basis = f"marginalised over an 80% conformal band (sigma {sigma:.1f} mmHg)"
        else:
            point = _score_heads(predictor, chained_history(history, fc, h), h0_keys, as_of)
            marg = point
            basis = ("point forecast only -- no interval is fitted at this horizon, so the "
                     "Jensen correction could not be applied and this probability is "
                     "understated for a patient near the threshold")

        for k, p in marg.items():
            base = k.rsplit("_h", 1)[0]
            items.append({
                "key": base, "horizon": h + 1,
                "days_ahead": node.get("days_ahead_est"),
                "predicted_sbp": node.get("point"),
                "prob": round(float(p), 4),
                "prob_point": round(float(point.get(k, p)), 4),
                "jensen_gap": round(float(p) - float(point.get(k, p)), 4),
                "cut": (round(float(cuts[k]), 4) if k in cuts
                        and np.isfinite(cuts[k]) else None),
                "cut_applies": False,
                "mechanism": mech.get(k),
                "red_flag": base in reds,
                "uncertainty_basis": basis,
            })
            bases[h] = basis

    # Red flags first, then by probability. Never averaged into a mechanism score: a red flag
    # next to fatigue is not half a red flag.
    items.sort(key=lambda d: (d["horizon"], not d["red_flag"], -d["prob"]))
    gaps = [abs(i["jensen_gap"]) for i in items]
    return {
        "available": True, "items": items, "n_heads": len(h0_keys),
        "horizons": [h + 1 for h in horizons],
        "marginalised": bool(marginalise and any("marginalised" in b for b in bases.values())),
        "max_jensen_gap": round(max(gaps), 4) if gaps else 0.0,
        "uncertainty_basis": bases,
        "cut_note": ("The operating cut was chosen on OBSERVED rows and does not carry its "
                     "alert-budget meaning to a chained row, so no flag is raised here. It "
                     "is shown for reference only until a chained cut is fitted."),
        "reach_note": ("Only the systolic and interdialytic-weight-gain drivers are forecast. "
                       "The intradialytic pressure drop is not a forecast signal and is absent "
                       "at serving, so the four hypotensive-mechanism symptoms -- dizziness, "
                       "syncope, palpitations, fatigue -- gain nothing from this path. "
                       "Diastolic pressure does not enter the symptom model at all."),
    }
