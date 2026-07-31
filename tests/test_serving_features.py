"""Do the features the model was trained on actually arrive at serving time?

A feature can be missing at serving in two very different ways:

  * **Honestly NaN** -- the user did not supply it. `sbp_drop` and `uf_total` are intradialytic
    measurements a journaling app has no access to, so they are absent on most requests. They
    are now ACCEPTED when a caller does have them, which makes the distinction sharper rather
    than softer: NaN must mean "not submitted", never "submitted and dropped".

  * **Accidentally absent** -- the user DID supply it and the mapping dropped it. This is a bug,
    and it is invisible: `_one_series` skips columns it cannot find, `reindex` fills the gap with
    NaN, and HistGradientBoosting consumes NaN natively. No exception, no log line, no failing
    test. It disabled 40 of 175 selected features on every request.

This file separates the two. It builds a request the way a real client would, runs it through
the actual serving path, and asserts that what the user submitted is present in the feature row
-- and that what they cannot submit is absent for the documented reason, not by accident.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from src.entity.config_entity import ModelTrainerConfig, TrainingPipelineConfig  # noqa: E402
from src.serving.mapping import to_history  # noqa: E402
from src.serving.schemas import PredictRequest  # noqa: E402
from src.utils.ml_utils.feature.causal_features import CausalFeatureBuilder  # noqa: E402

FAILS = []


def chk(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"  <- {extra}"))
    if not cond:
        FAILS.append(name)


def build_request(*, symptoms=True, pulse=True, session=True,
                  clinical=True, vitals=True, n=60):
    rows, d = [], datetime.date(2026, 2, 2)
    for i in range(n):
        r = {"date": d.isoformat(), "sbp": 138 + (i % 7) * 3 - (i % 3), "dbp": 78 + i % 5}
        if vitals:
            r["weight"] = round(71.0 + (i % 5) * 0.3, 2)
        if pulse:
            r["pulse"] = 68 + (i % 11)
        if symptoms and i % 4 == 0:
            r["symptoms"] = ["dizziness", "sob"]
        if symptoms and i % 13 == 0:
            r["symptoms"] = ["syncope"]                     # a training red flag
        if session:
            r["took_all_meds"] = i % 4 != 0
        rows.append(r)
        d += datetime.timedelta(days=2)
    prof = {"age": 68.0, "is_male": 1}
    if clinical:
        prof.update({"conditions": ["has_hf", "has_cad"], "medications": ["on_ace", "on_bb"]})
    return PredictRequest(patient_id="feat-1", readings=rows, profile=prof)


def feature_row(req):
    """The row the forecaster is actually handed, via the real serving path."""
    fb = CausalFeatureBuilder(ModelTrainerConfig(TrainingPipelineConfig()))
    return fb.transform_for_inference(to_history(req))


# Features that exist ONLY because the user submitted symptoms / pulse / dry weight. Each was
# NaN on every request before to_history emitted the underlying columns.
SYMPTOM_FEATURES = ["sym_any_lag1", "sym_any_mean7", "sym_count_lag1", "sym_red_flag_lag1",
                    "sym_dizziness_lag1", "sym_dizziness_rate30",
                    "sym_shortness_of_breath_lag1", "sym_syncope_rate30"]
HR_FEATURES = ["heart_rate_lag1", "heart_rate_mean7", "heart_rate_mean30",
               "shock_index_lag1", "hr_z", "hr_d1"]
#: Dialysis-derived features the product no longer collects the inputs for. NaN by
#: construction now, and asserted so: the distinction this file exists to keep is between
#: "not submitted" and "submitted and dropped", and a deliberate product decision belongs in
#: the first group with a reason attached.
NOT_COLLECTED = ["idwg_rel_lag1", "idwg_lag1", "uf_rate_lag1", "uf_lag1", "vintage_years",
                 "sbp_drop_lag1"]

#: Genuinely unavailable at serving. Their absence is the documented shift, not a defect.
HONESTLY_ABSENT = ["uf_total_lag1"]


def _submitted_features_arrive():
    row = feature_row(build_request())
    for group, cols in (("symptom", SYMPTOM_FEATURES), ("heart-rate", HR_FEATURES)):
        for c in cols:
            if c not in row.columns:
                chk(f"{group}: {c} exists in the feature row", False, "column not built")
                continue
            v = row[c].iloc[0]
            chk(f"{group}: {c} is a real value, not NaN",
                v is not None and np.isfinite(v), repr(v))


def _values_are_right():
    req = build_request()
    row = feature_row(req)
    hist = to_history(req)

    # hr features read the PREVIOUS reading, per the causal contract.
    chk("heart_rate_lag1 equals the last submitted pulse",
        np.isclose(row["heart_rate_lag1"].iloc[0], hist.heart_rate.iloc[-1]),
        (row["heart_rate_lag1"].iloc[0], hist.heart_rate.iloc[-1]))

    # sym_any_lag1 must reflect the newest reading's symptoms, not today's placeholder.
    chk("sym_any_lag1 equals the last submitted sym_any",
        np.isclose(row["sym_any_lag1"].iloc[0], hist.sym_any.iloc[-1]),
        (row["sym_any_lag1"].iloc[0], hist.sym_any.iloc[-1]))

    # The rate features must be a fraction, not a count.
    r30 = row["sym_dizziness_rate30"].iloc[0]
    chk("sym_dizziness_rate30 is a rate in [0, 1]", 0.0 <= r30 <= 1.0, r30)
    chk("    and it is non-zero (dizziness was submitted)", r30 > 0, r30)


def _control_no_symptoms():
    """The checks must be able to fail: a request with no symptoms and no pulse."""
    row = feature_row(build_request(symptoms=False, pulse=False))
    hr_nan = [c for c in HR_FEATURES
              if c in row.columns and not np.isfinite(row[c].iloc[0])]
    chk("(control) with no pulse submitted, the HR features ARE NaN",
        len(hr_nan) == len([c for c in HR_FEATURES if c in row.columns]), hr_nan)
    # With no symptoms the flags are legitimately 0, not NaN -- "reported nothing" is data.
    if "sym_any_lag1" in row.columns:
        chk("(control) with no symptoms, sym_any_lag1 is 0 rather than NaN",
            row["sym_any_lag1"].iloc[0] == 0, row["sym_any_lag1"].iloc[0])


def _honestly_absent_stay_absent():
    # session=False on purpose: these columns are now ACCEPTED, so the question is no longer
    # "can they exist" but "does NaN still mean not-submitted". A fixture that supplied them
    # would make this assert the opposite of what it says.
    row = feature_row(build_request(session=False))
    for c in HONESTLY_ABSENT:
        if c not in row.columns:
            chk(f"{c} is not fabricated", True)
            continue
        v = row[c].iloc[0]
        chk(f"{c} stays NaN -- a journaling client cannot supply it",
            v is None or not np.isfinite(v), repr(v))


def _not_collected():
    """Dialysis features must be NaN because nothing collects them -- and stay unreachable.

    This is the deliberate half of the file's distinction. `sbp_drop`, `idwg`, `uf_total`,
    `session_hours`, dry weight and the first-dialysis date were accepted until this became a
    blood-pressure product; now the schema forbids them. Both halves are asserted, because a
    field that is quietly re-accepted would populate features the forecaster was fitted on and
    change every number here without anything looking wrong.
    """
    from pydantic import ValidationError

    row = feature_row(build_request())
    for c in NOT_COLLECTED:
        if c not in row.columns:
            chk(f"{c} is not fabricated", True)
            continue
        v = row[c].iloc[0]
        chk(f"{c} is NaN -- this product does not collect its input",
            v is None or not np.isfinite(v), repr(v))

    # The schema is the enforcement point, so prove it rejects rather than ignores.
    base = {"date": "2026-01-01", "sbp": 140, "dbp": 80}
    for field, val in (("idwg", 2.1), ("uf_total", 2.4), ("session_hours", 4.0),
                       ("sbp_drop", 18.0)):
        try:
            PredictRequest(patient_id="x", readings=[{**base, field: val}])
            chk(f"the schema rejects a withdrawn reading field {field!r}", False,
                "accepted it -- extra=forbid is not doing its job")
        except ValidationError:
            chk(f"the schema rejects a withdrawn reading field {field!r}", True)
    for field, val in (("dryweight", 71.0), ("first_dialysis", "2019-03-01")):
        try:
            PredictRequest(patient_id="x", readings=[base], profile={field: val})
            chk(f"the schema rejects a withdrawn profile field {field!r}", False, "accepted")
        except ValidationError:
            chk(f"the schema rejects a withdrawn profile field {field!r}", True)

    # ...and the control: a field the product DOES collect must still be accepted, or the
    # checks above would pass on a schema that rejects everything.
    try:
        PredictRequest(patient_id="x", readings=[{**base, "weight": 71.0}])
        chk("    (control) a collected field is still accepted", True)
    except ValidationError as exc:
        chk("    (control) a collected field is still accepted", False, str(exc)[:120])


def _shared_row():
    """predict() and symptom_block must build the feature row ONCE between them.

    Counted by instrumenting the builder rather than by timing, so the check is exact and not
    machine-dependent. Also asserts the internal handoff key never reaches the payload -- it
    holds a DataFrame, which is neither JSON-serialisable nor anyone's business.
    """
    from src.serving.advisory import build_advisory
    from src.serving.model_registry import ModelRegistry
    from src.serving.vocabulary import build_vocabulary
    from src.utils.ml_utils.feature.causal_features import CausalFeatureBuilder as CFB
    from src.utils.ml_utils.rule_engine.registry import build_registry

    reg = ModelRegistry()
    reg.refresh(force=True)
    rules = build_registry()
    note = (build_vocabulary(rules).get("rules") or {}).get("note", "")
    req = build_request()

    calls = {"n": 0}
    real = CFB.transform_for_inference

    def counted(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    CFB.transform_for_inference = counted
    try:
        d = build_advisory(req, reg.predictor, rules, note)
    finally:
        CFB.transform_for_inference = real

    chk("the internal feature-row handoff never reaches the payload",
        "_feature_row" not in d, sorted(d)[:12])
    if reg.predictor is None:
        print(f"  INFO    no bundle on disk; {calls['n']} builds, sharing exercised "
              f"with a model only")
    else:
        # predict() builds one. symptom_block reuses it. The backtest and anomaly blocks use
        # build_panel_frame, which is a different (multi-row) build and is not counted here.
        chk("predict() and symptom_block share one row, not two",
            calls["n"] <= 1, f"{calls['n']} transform_for_inference calls")


def _coverage():
    """How much of the shipped feature list actually arrives? Reported, not asserted.

    Numeric columns only: the row also carries `ts`, `patient_id`, `position` and the
    `_symptoms` tuple, none of which `np.isfinite` accepts.
    """
    import pandas as pd
    row = feature_row(build_request())
    built = [c for c in row.columns
             if not c.startswith("y_") and pd.api.types.is_numeric_dtype(row[c])]
    finite = [c for c in built if np.isfinite(row[c].iloc[0])]
    print(f"  INFO    {len(finite)} of {len(built)} numeric columns are finite for a request "
          f"carrying symptoms, pulse, weight and same-day adherence")
    empty = sorted(set(built) - set(finite))
    print(f"  INFO    still NaN ({len(empty)}): {empty[:12]}"
          + (" ..." if len(empty) > 12 else ""))


def _bundle_coverage():
    """The claim this whole change rests on: a complete request feeds the WHOLE model.

    Measured against the shipped bundle's own `feature_names`, not against the columns the
    builder happens to emit -- those are a superset and would make the number look better than
    it is. Skipped, loudly, when no bundle is on disk.
    """
    from src.serving.model_registry import ModelRegistry
    reg = ModelRegistry()
    reg.refresh(force=True)
    P = reg.predictor
    if P is None:
        print("  SKIP    no bundle on disk; cannot measure against a fitted feature list")
        return
    feats = list(P.b["feature_names"])

    def missing(req):
        row = feature_row(req).reindex(columns=feats)
        return [c for c in feats if not np.isfinite(row[c].iloc[0])]

    full = missing(build_request())
    # Sorted into the two groups the API itself reports, so this test and the coverage block
    # cannot disagree about which absences are the caller's to fix.
    from src.serving.enrich import _NOT_COLLECTED, _stem
    product = [c for c in full if _stem(c) in _NOT_COLLECTED]
    fixable = [c for c in full if _stem(c) not in _NOT_COLLECTED]
    chk("*** a complete request resolves every feature this product can supply ***",
        not fixable, f"{len(fixable)} still NaN: {sorted(fixable)[:10]}")
    chk("    the rest are dialysis measurements, absent by product decision",
        len(product) == len(full) and len(product) > 20,
        f"{len(product)} of {len(full)}")

    # The control is the whole point: if the bare request ALSO resolved everything, the check
    # above would be passing for free and telling us nothing about the mapping.
    bare = missing(build_request(symptoms=False, pulse=False,
                                 session=False, clinical=False, vitals=False))
    chk("    (control) a bare request leaves many unresolved",
        len(bare) > 40, f"only {len(bare)} NaN -- the completeness check may be vacuous")
    print(f"  INFO    bare request: {len(feats) - len(bare)}/{len(feats)} resolved; "
          f"complete request: {len(feats) - len(full)}/{len(feats)}")

    # Each input group must be individually load-bearing, or the API is asking for a field
    # that changes nothing.
    for label, kw in (("per-session adherence", {"session": False}),
                      ("pulse", {"pulse": False}),
                      ("per-reading weight", {"vitals": False})):
        got = missing(build_request(**kw))
        chk(f"    withholding {label} costs features", len(got) > len(full),
            f"{len(got)} vs {len(full)}")

    # Symptoms and the clinical flags are the exception, and the difference matters: both are
    # emitted as 0/1 for every reading, so withholding them changes the VALUES rather than the
    # NaN count. An unticked condition means "does not have it", which is information, not
    # missingness. Counting NaN here would ask the wrong question and pass only if the
    # mapping were broken.
    full_row = feature_row(build_request()).reindex(columns=feats).iloc[0]
    for label, kw, prefix in (
            ("symptoms", {"symptoms": False}, ("sym_",)),
            ("conditions and medications", {"clinical": False}, ("has_", "on_"))):
        other = feature_row(build_request(**kw)).reindex(columns=feats).iloc[0]
        cols = [c for c in feats if c.startswith(prefix)]
        moved = [c for c in cols if full_row[c] != other[c]]
        chk(f"    withholding {label} changes their features (0/1, never NaN)",
            len(moved) >= 3, f"{len(moved)} of {len(cols)} moved")
        chk("    and they stay finite either way -- not-reported is 0, not missing",
            all(np.isfinite(other[c]) for c in cols),
            [c for c in cols if not np.isfinite(other[c])][:5])


def run():
    for title, fn in (("submitted features arrive", _submitted_features_arrive),
                      ("values are correct", _values_are_right),
                      ("control: nothing submitted", _control_no_symptoms),
                      ("honestly absent stay absent", _honestly_absent_stay_absent),
                      ("deliberately not collected", _not_collected),
                      ("shared feature row", _shared_row),
                      ("coverage", _coverage),
                      ("bundle coverage", _bundle_coverage)):
        print(f"\n--- {title} ---")
        fn()
    print("\n" + ("ALL SERVING FEATURE TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


def test_serving_features():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
