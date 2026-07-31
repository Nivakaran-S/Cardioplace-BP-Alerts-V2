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

    `sbp_drop` and `uf_total` are intradialytic quantities: a journaling app cannot supply
    them, so they are NaN here while they were populated in training. That is a real
    train/serve distribution shift, it is disclosed in the governance panel, and it is what
    the missingness sweep quantifies -- it is not silently zero-filled, because zero UF is a
    clinically meaningful and wrong statement.

    Symptoms and heart rate are a different case, and for a while this function got it wrong.
    Both ARE submitted -- `Reading.symptoms` and `Reading.pulse` -- and both were collected
    here and then dropped, so `_one_series` found no `sym_*` and no `heart_rate` column, took
    its `if c not in g.columns: continue` branch, and the reindex in the serving path quietly
    filled the resulting gap with NaN. That silently disabled **40 of the 175 selected
    features** (36 symptom-history, 4 heart-rate) on every request: not a shift in their
    values, an absence of them. HistGradientBoosting handles NaN natively, so nothing raised
    and nothing logged.

    So the rule is: NaN when the user genuinely cannot supply it, real when they can.

    Applying that rule again turned up three more groups that were NaN for no better reason
    than that nothing carried them across, costing **65 of the 175 selected features** on a
    request that supplied everything the API accepted:

    * **Static conditions and medications.** `to_engine_panel` emits them, this function did
      not, so `has_hf` / `on_ace` / ... reached the rule engine and never reached the model.
      The checkbox was ticked and the forecaster saw NaN.
    * **Medication adherence.** `took_all_meds` and `missed_antihypertensive` were emitted
      nowhere, which left `took_all_meds_lag1/_mean7/_mean30`, the matching
      `missed_antihypertensive_*`, and `adh_took_all_meds_today` all NaN.
    * **Dialysis session values.** `sbp_drop` and `uf_total` were hardcoded NaN even when the
      caller could supply them.

    `uf_rate` and `vintage_years` get the same treatment `idwg_rel` already had: derived here
    with the training expression, because they are built in the panel-prep stage
    (`causal_features.py:55-71`) that the serving path never runs.
    """
    p = req.profile
    dry = getattr(p, "dryweight", None)

    # Time-invariant, so computed once. Aliased to the training names -- see _CONDITION_ALIAS.
    conds = {_CONDITION_ALIAS.get(k, k) for k in p.conditions}
    meds = set(p.medications)
    static = {k: int(k in conds) for k in _MODEL_CONDITIONS}
    static.update({k: int(k in meds) for k in _MODEL_MEDS})
    on_antihypertensive = any(k in meds for k in _ANTIHYPERTENSIVES)

    first_dx = getattr(p, "first_dialysis", None)
    first_ts = pd.Timestamp(first_dx) if first_dx else None

    rows = []
    for r in req.readings:
        # Canonical names, because that is what the feature builder was trained on. The four
        # engine-only keys are dropped here on purpose -- see _ENGINE_ONLY_SYMPTOMS.
        present = {_SERVING_TO_CANONICAL[k] for k in r.symptoms if k in _SERVING_TO_CANONICAL}
        row = {
            "patient_id": req.patient_id,
            "ts": pd.Timestamp(r.date),
            "sbp": float(r.sbp), "dbp": float(r.dbp),
            "idwg": float(r.idwg) if r.idwg is not None else np.nan,
            "weight": float(r.weight) if r.weight is not None else np.nan,
            "dryweight": float(dry) if dry is not None else np.nan,
            # Real when the caller can supply them, NaN when they cannot -- never zero-filled.
            "sbp_drop": float(r.sbp_drop) if r.sbp_drop is not None else np.nan,
            "uf_total": float(r.uf_total) if r.uf_total is not None else np.nan,
            "session_hours": (float(r.session_hours) if r.session_hours is not None
                              else np.nan),
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
        # `idwg_rel` is derived in the PANEL-prep stage (`causal_features.py:55-58`), which the
        # serving path never runs -- `transform_for_inference` rebuilds only the scaffolding.
        # So idwg_rel_lag1/mean7/mean30, all three of which survived selection, stayed NaN even
        # once dry weight was supplied. Same expression and the same clip as the training side;
        # a divergence here would mean the column means one thing at fit time and another at
        # serve time, which is worse than it being absent.
        if dry and r.idwg is not None and float(dry) > 0:
            row["idwg_rel"] = float(np.clip(100.0 * float(r.idwg) / float(dry), 0, 12))
        else:
            row["idwg_rel"] = np.nan

        # Same story as idwg_rel: built in panel-prep (`causal_features.py:59-64`), which the
        # serving path never runs, so uf_rate_lag1/mean7/mean30 stayed NaN even with uf_total
        # supplied. Same expression, same x1000 (uf_total is litres), same clip -- a
        # divergence here would mean the column means one thing at fit time and another now.
        if (dry and r.uf_total is not None and r.session_hours is not None
                and float(dry) > 0 and float(r.session_hours) > 0):
            row["uf_rate"] = float(np.clip(
                (float(r.uf_total) * 1000.0) / (float(r.session_hours) * float(dry)), 0, 30))
        else:
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

        # Years on dialysis at THIS session, so a long history does not carry one flat value.
        row["vintage_years"] = (float(np.clip((row["ts"] - first_ts).days / 365.25, 0, 40))
                                if first_ts is not None else np.nan)

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
