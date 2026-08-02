"""Request -> the two DataFrames the model and the engine each need.

They are different frames on purpose. `BPPredictor` wants the raw session history it was
trained on; `ExtendedRuleEngine` wants a panel carrying every clinical axis the 56 rules
read. Building one frame that satisfies both would mean feeding the feature builder columns
it must never see.
"""

import numpy as np
import pandas as pd

from src.utils.ml_utils.feature.cadence import attach_cadence
from src.utils.ml_utils.rule_engine.symptom_layer import (
    SYMPTOM_KEY_ALIAS,
    SYMPTOM_MECHANISM,
    SYMPTOM_RED_FLAG,
    SYMPTOMS,
)
from src.utils.ml_utils.rule_engine.synthetic import CONDITION_KEYS, MED_KEYS, SYMPTOM_KEYS

#: Columns `CausalFeatureBuilder._one_series` touches unconditionally. Every one has to
#: exist even when the user cannot supply it, or the feature build raises KeyError inside a
#: CustomException and the API returns a 500 for a perfectly valid request.
#: `transform_for_inference` only NaNs these if they are ALREADY present, so absence is not
#: the same as NaN here.
REQUIRED_HISTORY_COLS = ("patient_id", "ts", "sbp", "dbp", "idwg", "weight",
                         "sbp_drop", "uf_total", "age", "is_male", "is_dm", "DM", "n_meas")

#: serving key -> the canonical name the training layer used. `SYMPTOM_KEY_ALIAS` runs the
#: other way (canonical -> serving), and five of the fifteen differ: `chest_pain` is submitted
#: as `chest_pain_dyspnea`, `shortness_of_breath` as `sob`, and so on. Inverting it here rather
#: than restating the pairs means the two directions cannot drift.
_SERVING_TO_CANONICAL = {SYMPTOM_KEY_ALIAS.get(c, c): c for c in SYMPTOMS}

#: The four serving keys with no training counterpart -- `new_onset_headache`, `ruq_pain`,
#: `edema`, `nsaid_use`. They are rule-engine axes only. Emitting `sym_*` columns for them
#: would hand the feature builder inputs the model never saw, so they are deliberately dropped
#: on this path while still reaching the engine through `to_engine_panel`.
_ENGINE_ONLY_SYMPTOMS = tuple(sorted(set(SYMPTOM_KEYS) - set(_SERVING_TO_CANONICAL)))

#: Static clinical flags the FORECASTER was fitted on, as the training layer names them
#: (`symptom_layer.py:192`). Only these reached feature selection -- the serving vocabulary
#: offers more conditions than the model uses, and emitting the extras would hand the feature
#: builder columns it never saw.
_MODEL_CONDITIONS = ("has_hf", "has_ihd", "has_afib")
_MODEL_MEDS = ("on_ace", "on_arb", "on_bb", "on_loop", "on_nsaid")

#: serving condition key -> the name training used. Ischaemic heart disease and coronary
#: artery disease are the same condition under two labels; the vocabulary offers `has_cad`
#: and the bundle was fitted on `has_ihd`. Without this the flag is silently NaN for every
#: patient who has it -- the checkbox is ticked, the engine reads it, and the model does not.
_CONDITION_ALIAS = {"has_cad": "has_ihd"}

#: The antihypertensive classes whose omission training counted as a missed dose
#: (`symptom_layer.py:154`). Restated here so the serving derivation cannot drift from the
#: definition the column was built under.
_ANTIHYPERTENSIVES = ("on_ace", "on_arb", "on_bb")


def to_history(req) -> pd.DataFrame:
    """One row per submitted reading, in the shape the trained model expects.

    Two kinds of absence, and keeping them apart is the whole job of this function.

    **Accidentally absent** is a bug, and an invisible one: `_one_series` skips columns it
    cannot find, the serving reindex fills the gap with NaN, and HistGradientBoosting consumes
    NaN natively -- no exception, no log line, no failing test. It has happened twice here.
    First `sym_*` and `heart_rate` were collected and then dropped, disabling 40 of the 175
    selected features on every request. Then static conditions and medications turned out to
    reach `to_engine_panel` and not this frame, medication adherence reached neither, and the
    total came to 65 of 175. Both are fixed: what the user submits, the model sees.

    **Deliberately absent** is a product decision and is stated rather than hidden. This is a
    blood-pressure service, so it does not ask for interdialytic weight gain, ultrafiltration
    volume or rate, session length, intradialytic pressure drop, dry weight or dialysis
    vintage -- a field nobody can fill implies the answer degrades for a reason the user could
    fix, which is worse than an honest gap. The forecaster was fitted on a haemodialysis
    corpus and those features were populated then, so this is a real train/serve shift with a
    known cost; `feature_coverage` reports it per request.

    Since the dialysis features were dropped from the model, the second kind of absence no
    longer costs anything: `CausalFeatureBuilder` neither derives nor lags these columns, and
    `idwg` is no longer a forecast target, so a bundle trained after that change has no
    feature that reads them. They are still emitted as NaN -- never zero, because zero
    ultrafiltration is a clinically meaningful and wrong statement -- for two reasons: the
    rule-engine advisory path still probes `history.idwg` before deciding it has nothing to
    say, and a bundle frozen BEFORE the change still lists those features, so the coverage
    block has to be able to report them as absent rather than raise.
    """

    p = req.profile

    # Time-invariant, so computed once. Aliased to the training names -- see _CONDITION_ALIAS.
    conds = {_CONDITION_ALIAS.get(k, k) for k in p.conditions}
    meds = set(p.medications)
    static = {k: int(k in conds) for k in _MODEL_CONDITIONS}
    static.update({k: int(k in meds) for k in _MODEL_MEDS})
    on_antihypertensive = any(k in meds for k in _ANTIHYPERTENSIVES)

    rows = []
    for r in req.readings:
        # Canonical names, because that is what the feature builder was trained on. The four
        # engine-only keys are dropped here on purpose -- see _ENGINE_ONLY_SYMPTOMS.
        present = {_SERVING_TO_CANONICAL[k] for k in r.symptoms if k in _SERVING_TO_CANONICAL}
        row = {
            "patient_id": req.patient_id,
            "ts": pd.Timestamp(r.date),
            "sbp": float(r.sbp), "dbp": float(r.dbp),
            "weight": float(r.weight) if r.weight is not None else np.nan,
            # The dialysis columns are carried as NaN rather than omitted. Nothing in the
            # feature builder reads them any more, but the engine panel is built from this
            # frame and an older bundle still names the features they fed, so the coverage
            # block needs them present to report them missing.
            "idwg": np.nan, "dryweight": np.nan, "sbp_drop": np.nan,
            "uf_total": np.nan, "session_hours": np.nan,
            "age": float(p.age), "is_male": int(p.is_male), "is_dm": int(p.is_dm),
            "DM": float(p.is_dm), "n_meas": int(r.n_meas),
            **static,
            "pulse": float(r.pulse) if r.pulse is not None else np.nan,
            # `_one_series` reads `heart_rate`, not `pulse`. Same measurement, two names, and
            # the rename is why hr_z / hr_d1 / shock_index_lag1 were all NaN at serving.
            "heart_rate": float(r.pulse) if r.pulse is not None else np.nan,
            "position": r.position,
            "_symptoms": tuple(r.symptoms),
        }
        # idwg_rel and uf_rate were derived here while the fields were collected. The feature
        # builder no longer produces either, so these exist only for an older bundle's
        # coverage report.
        row["idwg_rel"] = np.nan
        row["uf_rate"] = np.nan

        # THE ONE SAME-DAY EXCEPTION. `_one_series` emits this as `adh_took_all_meds_today`
        # and lags it for the rest; see the field docstring on `Reading.took_all_meds`.
        # `missed_antihypertensive` is DERIVED, never taken from the client, so it cannot
        # disagree with the adherence answer it is a function of.
        if r.took_all_meds is None:
            row["took_all_meds"] = np.nan
            row["missed_antihypertensive"] = np.nan
        else:
            took = int(bool(r.took_all_meds))
            row["took_all_meds"] = took
            row["missed_antihypertensive"] = int(took == 0 and on_antihypertensive)

        # Dialysis vintage: not a quantity a cardiovascular BP service has.
        row["vintage_years"] = np.nan

        for s in SYMPTOMS:
            row[f"sym_{s}"] = int(s in present)
        row["sym_any"] = int(bool(present))
        row["sym_count"] = len(present)
        # SYMPTOM_RED_FLAG is the TRAINING layer's set (6 symptoms), which is deliberately
        # narrower than the 8 the UI marks with a flag -- the vocabulary also flags
        # severe_headache and severe_epigastric_pain for clinical emphasis. The feature
        # `sym_red_flag_lag1` was built from the training definition, so serving must use the
        # training definition or the column means something different than it did at fit time.
        # This is not the two sets drifting; they answer different questions.
        row["sym_red_flag"] = int(bool(present & set(SYMPTOM_RED_FLAG)))
        for mech in sorted(set(SYMPTOM_MECHANISM.values())):
            row[f"sym_mech_{mech}"] = int(any(SYMPTOM_MECHANISM[s] == mech for s in present))
        rows.append(row)

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
