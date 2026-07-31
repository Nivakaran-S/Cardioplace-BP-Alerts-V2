"""Smoke test for src/utils/ml_utils/safety/ -- gates must both pass AND be able to fail."""

import sys
from pathlib import Path

# `python tests/test_x.py` puts tests/ on sys.path, not the repo root, so `import src` fails.
# The README documents running these files directly, so the fix belongs here rather than in a
# PYTHONPATH the reader has to know to set. CI hit exactly this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.entity.config_entity import ModelTrainerConfig, TrainingPipelineConfig  # noqa: E402
from src.utils.ml_utils.safety.gates import (  # noqa: E402
    AbstentionPolicy,
    provenance_guard,
    run_safety_gates,
)


def test_safety_gates():
    cfg = ModelTrainerConfig(TrainingPipelineConfig())
    rng = np.random.default_rng(0)

    N_PT, N_STEP = 40, 30
    sid = np.repeat([f"p{i}" for i in range(N_PT)], N_STEP)
    step = np.tile(np.arange(N_STEP), N_PT)
    ts = pd.to_datetime("2024-01-01") + pd.to_timedelta(step * 2, "D")
    sbp = rng.normal(138, 14, len(sid))

    alerts_pop = pd.DataFrame(dict(series_id=sid, step=step, ts=ts, sbp=sbp,
                                   dbp=sbp * 0.6, rule_id=None, tier=None))
    # ~3% fire, 0.5% emergency -> inside the 5% budget
    fire = rng.random(len(sid)) < 0.03
    alerts_pop.loc[fire, "tier"] = "BP_LEVEL_1_HIGH"
    emerg = rng.random(len(sid)) < 0.005
    alerts_pop.loc[emerg, "tier"] = "BP_LEVEL_2"
    alerts_pop["fired"] = alerts_pop.tier.notna()
    alerts_pop["is_emergency"] = alerts_pop.tier == "BP_LEVEL_2"
    alerts_pers = alerts_pop.copy()

    OFF = pd.DataFrame(dict(series_id=[f"p{i}" for i in range(N_PT)],
                            threshold=rng.uniform(130, 152, N_PT),
                            offset=rng.uniform(-10, 12, N_PT),
                            capped=rng.random(N_PT) < 0.2))
    fairness = pd.DataFrame([dict(metric="forecaster MAE (mmHg)", axis="is_male", level="1",
                                  n=200, value=8.1, overall=8.0, gap=0.1, passes=True)])
    cold = pd.DataFrame(dict(n_readings=[3, 7, 25, 60],
                             band_width=[22.0, 19.0, 16.0, 15.0]))
    missing = pd.DataFrame(dict(drop_frac=[0.0, 0.1, 0.3, 0.5], MAE=[8.0, 8.2, 9.1, 10.4]))
    train = pd.DataFrame(rng.normal(0, 1, (900, 6)), columns=[f"f{i}" for i in range(6)])
    test = pd.DataFrame(rng.normal(0, 1, (300, 6)), columns=train.columns)

    abstain = AbstentionPolicy(train, list(train.columns))
    ood_rate = float(np.mean(abstain._dist(test) > abstain.cut))
    print(f"abstention: cut={abstain.cut:.3f} n_fit={abstain.n_fit} ood_rate={ood_rate:.4f}")
    assert np.isfinite(abstain.cut) and abstain.cut > 0
    # _dist must tolerate BOTH a full frame and a pre-sliced one (the two real call sites).
    assert abstain._dist(test[list(train.columns)]).shape == (300,)
    assert abstain._dist(test.assign(extra=1.0)).shape == (300,)

    panel = pd.DataFrame(dict(series_id=sid))
    synth = pd.DataFrame(dict(series_id=np.repeat([f"s{i}" for i in range(10)], 5)))
    syn_alerts = synth.copy()
    prov = provenance_guard(panel, synth, alerts_pop, syn_alerts)
    assert prov["holds"], prov
    print("provenance guard: holds")

    kw = dict(alerts_pop=alerts_pop, alerts_pers=alerts_pers, OFF=OFF, advisories=[],
              cold=cold, fairness=fairness, explanation={"sbp_lag1": -3.2, "sbp_z": 0.9},
              abstain=abstain, ood_rate=ood_rate, config=cfg, missingness=missing,
              offset_learned_max=151.0, shipped_detector="d_isoforest",
              cold_start_penalty=1.4, symptom_labels_synthetic=True,
              detector_alert_rate=0.048, pair_violation_rate=0.0,
              shipped_forecaster={"sbp": "ridge", "dbp": "hgb", "idwg": "ridge"},
              selected_forecaster={"sbp": "ridge", "dbp": "hgb", "idwg": "ridge"},
              forecaster_features=["sbp_lag1", "sbp_z", "pp_mean7"],
              shipped_families={"sbp": "learned", "dbp": "learned", "idwg": "learned"},
              symptom_fairness=pd.DataFrame({"symptom": ["dizziness", "dizziness"],
                                             "axis": ["sex", "sex"],
                                             "level": ["male", "female"],
                                             "pr_auc": [0.21, 0.19], "spread": [0.02, 0.02],
                                             "passes": [True, True]}))

    good = run_safety_gates(**kw)
    print("\n--- healthy run ---")
    print(good.frame()[["gate", "status", "observed", "bound"]].to_string(index=False))
    assert good.promotable, good.critical_failures
    assert not good.critical_failures

    # Gate 5 (subgroup parity) was REMOVED as a gate by instruction on 2026-07-31. It is
    # still measured -- run_fairness writes fairness.csv with every slice, and slices outside
    # the margin are logged at WARNING -- but it no longer blocks promotion, so it is
    # deliberately absent from the must-be-able-to-fail list below. The metric itself stays
    # under test in test_slice_gate.py. Asserted rather than assumed, so a future edit that
    # quietly reinstates or fully deletes it has to come past this line.
    assert "subgroup parity" not in set(good.frame().gate), \
        "subgroup parity should no longer appear as a gate"
    assert good.promotable, "removing the fairness gate must not have broken promotability"
    print("\n  subgroup parity: removed as a gate by instruction; fairness.csv still written")

    print("\n--- each critical gate must be able to FAIL ---")
    cases = {
        "emergency floor": dict(OFF=OFF.assign(threshold=OFF.threshold.where(
            OFF.index != 0, 185.0))),
        "offset caps": dict(OFF=OFF.assign(offset=OFF.offset.where(OFF.index != 0, 40.0))),
        # The budget binds the DETECTOR, not the rule engine: the engine is the clinical
        # floor and fires on ~87% of sessions in this cohort by design.
        "detector alert budget": dict(detector_alert_rate=0.40),
        "emergency parity": dict(alerts_pers=alerts_pers.assign(is_emergency=False)),
        "zero explanation": dict(explanation={"sbp_lag1": 0.0, "sbp_z": 0.0}),
        "ood rate": dict(ood_rate=0.9),
        "learned offset floor": dict(offset_learned_max=195.0),
    }
    for name, override in cases.items():
        r = run_safety_gates(**{**kw, **override})
        assert not r.promotable, f"{name}: gate did not bite"
        print(f"  {name:24s} -> BLOCKED ({r.critical_failures})")

    print("\n--- a deliberate BASELINE ship is not a freeze failure ---")
    # ship_decision can rule that no learned model earned its place, and then a baseline
    # serves. That is the rule working; gate 15 exists for the winner failing to FREEZE.
    # Generalising the gate per signal conflated the two and reported dbp's legitimate
    # baseline as an unselected model. Both directions asserted here, because "does not
    # warn" is only the right behaviour if the real failure still does.
    baseline_ship = run_safety_gates(**{
        **kw,
        "shipped_forecaster": {"sbp": "ridge", "dbp": "ewma", "idwg": "ridge"},
        "selected_forecaster": {"sbp": "ridge", "dbp": "huber", "idwg": "ridge"},
        "shipped_families": {"sbp": "learned", "dbp": "baseline", "idwg": "learned"}})
    f15 = baseline_ship.frame()
    row = f15[f15.gate == "shipped forecaster is the selected forecaster"].iloc[0]
    assert row.status == "PASS", f"a baseline ship must not warn: {row.to_dict()}"
    print("  dbp ships the baseline by decision -> PASS (not called unselected)")

    froze_badly = run_safety_gates(**{
        **kw,
        "shipped_forecaster": {"sbp": "ridge", "dbp": "ridge", "idwg": "ridge"},
        "selected_forecaster": {"sbp": "ridge", "dbp": "local_ar", "idwg": "ridge"},
        "shipped_families": {"sbp": "learned", "dbp": "learned", "idwg": "learned"}})
    f15b = froze_badly.frame()
    row_b = f15b[f15b.gate == "shipped forecaster is the selected forecaster"].iloc[0]
    assert row_b.status == "WARN", f"a freeze failure must still warn: {row_b.to_dict()}"
    print("  learned winner could not freeze    -> WARN (still caught)")

    print("\n--- non-critical gates WARN but still promote ---")
    for name, override in {
        # Per SIGNAL now: a dbp winner that failed to freeze used to ship silently.
        "shipped != selected (dbp only)": dict(
            shipped_forecaster={"sbp": "ridge", "dbp": "ridge", "idwg": "ridge"},
            selected_forecaster={"sbp": "ridge", "dbp": "local_ar", "idwg": "ridge"}),
        "a target column reached the forecaster":
            dict(forecaster_features=["sbp_lag1", "y_sym_dizziness_h0"]),
        "symptom subgroup disparity": dict(
            symptom_fairness=pd.DataFrame({"symptom": ["dizziness"], "axis": ["sex"],
                                           "level": ["male"], "pr_auc": [0.4],
                                           "spread": [0.4], "passes": [False]})),
        "detector is reference": dict(shipped_detector="d_fixed_threshold"),
        "synthetic flag lost": dict(symptom_labels_synthetic=False),
        "cold-start penalty": dict(cold_start_penalty=9.0),
        # Gate 17. Non-critical on purpose: a JOINT property must not veto a forecaster whose
        # own marginal MAE cleared every other bar in the table.
        "incoherent forecast pairs": dict(pair_violation_rate=0.03),
        "missingness blowup": dict(missingness=missing.assign(MAE=[8.0, 12.0, 18.0, 25.0])),
    }.items():
        r = run_safety_gates(**{**kw, **override})
        assert r.promotable, f"{name} should warn, not block"
        assert r.warnings, f"{name} produced no warning"
        print(f"  {name:24s} -> WARN, still promotable")

    print("\n--- absent evidence must NOT count as PASS ---")
    r = run_safety_gates(**{**kw, "fairness": pd.DataFrame(), "cold": pd.DataFrame(),
                            "missingness": None, "offset_learned_max": None,
                            "shipped_detector": None, "cold_start_penalty": None,
                            "symptom_labels_synthetic": None,
                            "pair_violation_rate": None,
                            "forecaster_features": None,
                            "symptom_fairness": None})
    f = r.frame()
    skipped = f[f.detail.str.startswith("not evaluated")]
    assert len(skipped) >= 6, f.to_string()
    assert (skipped.status == "WARN").all(), "a skipped gate recorded as PASS"
    print(f"  {len(skipped)} gates recorded 'not evaluated' as WARN, none as PASS")

    print("\nALL SAFETY SMOKE TESTS PASSED")


if __name__ == "__main__":
    test_safety_gates()
