"""Request -> the two DataFrames the model and the engine each need.

They are different frames on purpose. `BPPredictor` wants the raw session history it was
trained on; `ExtendedRuleEngine` wants a panel carrying every clinical axis the 56 rules
read. Building one frame that satisfies both would mean feeding the feature builder columns
it must never see.
"""

import numpy as np
import pandas as pd

from src.utils.ml_utils.feature.cadence import attach_cadence
from src.utils.ml_utils.rule_engine.synthetic import CONDITION_KEYS, MED_KEYS, SYMPTOM_KEYS

#: Columns `CausalFeatureBuilder._one_series` touches unconditionally. Every one has to
#: exist even when the user cannot supply it, or the feature build raises KeyError inside a
#: CustomException and the API returns a 500 for a perfectly valid request.
#: `transform_for_inference` only NaNs these if they are ALREADY present, so absence is not
#: the same as NaN here.
REQUIRED_HISTORY_COLS = ("patient_id", "ts", "sbp", "dbp", "idwg", "weight",
                         "sbp_drop", "uf_total", "age", "is_male", "is_dm", "DM", "n_meas")


def to_history(req) -> pd.DataFrame:
    """One row per submitted reading, in the shape the trained model expects.

    `sbp_drop` and `uf_total` are intradialytic quantities: a journaling app cannot supply
    them, so they are NaN here while they were populated in training. That is a real
    train/serve distribution shift, it is disclosed in the governance panel, and it is what
    the missingness sweep quantifies -- it is not silently zero-filled, because zero UF is a
    clinically meaningful and wrong statement.
    """
    p = req.profile
    rows = []
    for r in req.readings:
        rows.append({
            "patient_id": req.patient_id,
            "ts": pd.Timestamp(r.date),
            "sbp": float(r.sbp), "dbp": float(r.dbp),
            "idwg": float(r.idwg) if r.idwg is not None else np.nan,
            "weight": float(r.weight) if r.weight is not None else np.nan,
            "sbp_drop": np.nan, "uf_total": np.nan,
            "age": float(p.age), "is_male": int(p.is_male), "is_dm": int(p.is_dm),
            "DM": float(p.is_dm), "n_meas": int(r.n_meas),
            "pulse": float(r.pulse) if r.pulse is not None else np.nan,
            "position": r.position,
            "_symptoms": tuple(r.symptoms),
        })
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    for c in REQUIRED_HISTORY_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def to_engine_panel(req, history: pd.DataFrame, ml_threshold: float = None) -> pd.DataFrame:
    """The panel `ExtendedRuleEngine.run` evaluates.

    `evaluate_row` reaches values three ways and the difference decides what may be omitted:
    `r.sbp`/`r.dbp` are direct attributes and must exist; `row.step`/`ts`/`series_id`/
    `days_since_last` are read directly by the gate; everything else goes through a getattr
    with a default, so an absent column reads as "not present" and cannot fire a rule.
    """
    p = req.profile
    d = history.copy()
    d["series_id"] = req.patient_id
    # step_offset keeps a returning patient from looking like a new one. Without it every
    # submitted history starts at step 0 and the first-month adherence nudge fires for all.
    d["step"] = np.arange(len(d)) + int(p.step_offset)
    d = attach_cadence(d, by="series_id")

    as_of = pd.Timestamp(req.as_of) if req.as_of else d.ts.max()
    # The gate's staleness question is "how old is this READING", not "how long was the gap
    # before it". Supplying reading_age_days means HISTORICAL_ENTRY means what it says.
    d["reading_age_days"] = (as_of - d.ts).dt.days.clip(lower=0).astype(float)

    d["weight_delta_24h"] = d.weight.diff().fillna(0.0)
    d["hf_type"] = p.hf_type
    d["is_pregnant"] = int(p.is_pregnant)
    d["history_hdp"] = 0
    d["provider_target"] = (float(p.provider_target) if p.provider_target is not None
                            else np.nan)
    d["missed_3d"] = int(p.missed_3d)
    d["adherence_7d"] = float(p.adherence_7d)

    conds, meds = set(p.conditions), set(p.medications)
    for k in CONDITION_KEYS:
        d[k] = int(k in conds)
    for k in MED_KEYS:
        d[k] = int(k in meds)
    d["missed_bb_today"] = int(bool(p.missed_3d) and "on_bb" in meds)

    # Symptoms are per reading, not per patient: the user attaches them to the session they
    # happened in, and a symptom-driven emergency is about today's symptoms with today's BP.
    sym_lists = d["_symptoms"] if "_symptoms" in d else [()] * len(d)
    for k in SYMPTOM_KEYS:
        d[k] = [int(k in s) for s in sym_lists]

    pulse = d.pulse if "pulse" in d else pd.Series(np.nan, index=d.index)
    d["brady_run_len"] = _brady_run(pulse, d)
    d["hr_high_recent"] = (pulse.shift(1) > 100).fillna(False).astype(int)

    # prev_emergency is read as a COLUMN by the confirmed-normal rule, not passed as the
    # third argument to evaluate_row (ExtendedRuleEngine ignores that argument). Omit the
    # column and that rule is silently dead.
    d["prev_emergency"] = (d.sbp.shift(1) >= 180).fillna(False)

    if ml_threshold is not None and np.isfinite(ml_threshold):
        d["ml_threshold"] = float(ml_threshold)

    return d.drop(columns=["_symptoms"], errors="ignore")


def _brady_run(pulse: pd.Series, d: pd.DataFrame) -> np.ndarray:
    """Consecutive sessions at 40-49 bpm with no dizziness -- the surveillance rule's input."""
    out, run = [], 0
    dizzy = d["dizziness"] if "dizziness" in d else pd.Series(0, index=d.index)
    for hr, dz in zip(pulse.to_numpy(dtype=float), np.asarray(dizzy)):
        run = run + 1 if (np.isfinite(hr) and 40 <= hr <= 49 and not dz) else 0
        out.append(run)
    return np.asarray(out, dtype=int)
