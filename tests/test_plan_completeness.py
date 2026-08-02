"""Audit the delivered pipeline against the agreed scope.

Structural, not behavioural: it asserts that each capability EXISTS and is wired, which the
other two suites then exercise. It exists because "did we actually build all of it" otherwise
gets answered from memory -- and memory is what let a referenced-but-absent `safety` package
sit in this repo unnoticed while `main.py` could not import.
"""


import sys
from pathlib import Path

# `python tests/test_x.py` puts tests/ on sys.path, not the repo root, so `import src` fails.
# The README documents running these files directly, so the fix belongs here rather than in a
# PYTHONPATH the reader has to know to set. CI hit exactly this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib  # noqa: E402
import inspect  # noqa: E402
import os  # noqa: E402

import app as A  # noqa: E402
import src.components.data_ingestion as DI  # noqa: E402
import src.components.data_validation as DV  # noqa: E402
import src.utils.main_utils.utils as U  # noqa: E402
import src.utils.ml_utils.model.detector as D  # noqa: E402
import src.utils.ml_utils.model.estimator as E  # noqa: E402
import src.utils.ml_utils.model.offset_learned as OL  # noqa: E402
import src.utils.ml_utils.rule_engine.symptom_layer as SL  # noqa: E402
from src.constants import training_pipeline as TP  # noqa: E402
from src.utils.ml_utils.feature.causal_features import (  # noqa: E402
    CausalFeatureBuilder,
    feature_group,
    leakage_audit,
)
from src.utils.ml_utils.feature.selection import MUST_KEEP, REDUNDANCY_R  # noqa: E402
from src.utils.ml_utils.metric.regression_metric import select_and_decide, tie_set  # noqa: E402
from src.utils.ml_utils.model.architectures import MODEL_SPEC  # noqa: E402
from src.utils.ml_utils.model.classifier_head import eval_symptom, train_symptom_heads  # noqa: E402

FAILS = []


def chk(item, ok, extra=""):
    print(("  OK   " if ok else "  MISS ") + item + ("" if ok else f"   <- {extra}"))
    if not ok:
        FAILS.append(item)


def has(mod, *names):
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        return False, f"import failed: {e}"
    miss = [n for n in names if not hasattr(m, n)]
    return (not miss), f"missing {miss}"


def run():
    print("\nP0 safety package (the import blocker)")
    chk("run_safety_gates / AbstentionPolicy / cold_start_curve / provenance_guard",
        *has("src.utils.ml_utils.safety.gates", "run_safety_gates", "AbstentionPolicy",
             "cold_start_curve", "provenance_guard", "GateResult"))
    chk("missingness_sweep", *has("src.utils.ml_utils.safety.missingness",
                                  "missingness_sweep"))
    chk("save_object is atomic", "os.replace" in inspect.getsource(U.save_object))

    print("\nP1 session reconciliation + schema single-source")
    si = inspect.getsource(DI)
    chk("gap-based sessionisation", "session_gap_hours" in si)
    chk("dialysis-day boundary", "dialysis_day_start_hour" in si)
    chk("session decomposition diagnostic",
        hasattr(DI.HemobpLoader, "_session_decomposition"))
    chk("same-day collapse", hasattr(DI.HemobpLoader, "_collapse_same_day"))
    chk("reconciliation gate + self-test",
        hasattr(DV.DataContract, "_reconciliation")
        and "reconciliation" in inspect.getsource(DV.DataContract.self_test))
    chk("schema.yaml is authoritative for horizons",
        TP.HORIZONS == tuple(TP.SCHEMA["horizons"]) == (0, 1, 2))
    chk("ingest ranges read from schema", TP.INGEST_RANGES["sbp"] == TP.SCHEMA_RANGES["sbp"])

    print("\nP2 / P3 feature families + selection")
    # Six, not the original seven: `idwg_per_kg` and `idwg_x_sbp_slope7` were the
    # "size-normalised" and "interaction" exemplars, and both were dropped with the dialysis
    # block. size-normalised survives through sbp_dbp_ratio; interaction has no members left,
    # because its only instance multiplied a fluid quantity by a pressure trend.
    fams = {feature_group(f) for f in ["pp_z", "sbp_mom_3_14", "sbp_excess_base",
                                       "sbp_vol_ratio", "sbp_ewm_resid", "sbp_dbp_ratio"]}
    chk("6 restored families classified distinctly", len(fams) == 6, fams)
    chk("no dialysis feature survives the builder's contract",
        {"nadir_sbp", "map_drop", "idwg_rel", "uf_rate", "vintage_years"}
        <= CausalFeatureBuilder.DROP)
    chk("idwg is no longer a forecast target", "idwg" not in TP.SIGNALS, TP.SIGNALS)
    chk("feature frame keeps the raw signals the audit rebuilds from",
        {"weight", "uf_total"} <= CausalFeatureBuilder.KEEP_NON_FEATURE)
    chk("MUST_KEEP includes sbp_ewm0.3 (the explanation gate reads it)",
        "sbp_ewm0.3" in MUST_KEEP)
    chk("REDUNDANCY_R = 0.95", REDUNDANCY_R == 0.95)

    print("\nP4 / P5 Model 1 bake-off + tuning")
    chk("24 architectures registered", len([k for k in MODEL_SPEC if k != "hgb"]) == 24,
        len(MODEL_SPEC))
    want = ["huber", "hgb_mse", "random_forest", "extra_trees", "knn", "mlp", "ridge_delta",
            "hgb_delta", "window_linear", "window_delta", "local_ar", "local_ridge",
            "local_ridge_delta", "local_hgb", "global_plus_intercept", "holt_damped",
            "theta", "ens_mean", "ens_median", "ens_inv_mae", "ens_stack_ridge"]
    chk("every added architecture present", all(w in MODEL_SPEC for w in want),
        [w for w in want if w not in MODEL_SPEC])
    chk("per-patient models flagged unfreezable",
        all(not MODEL_SPEC[k]["freezable"] for k in MODEL_SPEC
            if MODEL_SPEC[k]["scope"] == "local"))
    sw = inspect.getsource(E.run_sweep)
    chk("holdout_pt cold-start arm", "holdout_pt" in sw)
    # Baselines must reach `preds`, and for EVERY signal -- the store is keyed by signal now,
    # because collecting sbp alone sent dbp down the marginal-CI fallback.
    chk("baselines enter preds so the ship test can pair",
        "preds.setdefault(signal, {})[name]" in sw)
    chk("preds are collected for every signal, not sbp only",
        'signal == "sbp"' not in sw, "run_sweep still gates preds on sbp")
    chk("FinalForecaster + fit_final freeze layer",
        *has("src.utils.ml_utils.model.architectures", "FinalForecaster", "fit_final"))
    sd = inspect.getsource(select_and_decide)
    chk("cheapest-in-tie-set selection", "COST_RANK" in sd and "tie_set" in sd)
    chk("patient_win_rate reported", "patient_win_rate" in inspect.getsource(tie_set))
    chk("paired_delta wired (was dead code)", "paired_delta" in sd)
    chk("cold_start_penalty reported", "cold_start_penalty" in sd)
    chk("optuna / grid / random tuners + dispatcher",
        *has("src.utils.ml_utils.model.estimator", "optuna_search", "grid_search",
             "random_search", "tune", "optuna_params"))
    chk("never ships a config that lost to defaults",
        "best >= base" in inspect.getsource(E.optuna_search))

    print("\nP6 Model 2 learned offset")
    chk("13 candidates", len(OL.offset_candidates()) == 13, len(OL.offset_candidates()))
    chk("37 features including the clinical block",
        len(OL.OFFSET_FEATURES) == 37
        and any(f.startswith("clin_") for f in OL.OFFSET_FEATURES),
        len(OL.OFFSET_FEATURES))
    tl = inspect.getsource(OL.train_learned_offset)
    chk("pinball + coverage metrics", "pinball" in tl and "coverage" in tl)
    chk("conformal shift only for mean-native learners", 'native == "mean"' in tl)
    chk("4-way patient-disjoint split",
        all(x in inspect.getsource(OL.add_offset_split)
            for x in ["calib", "select", "holdout"]))
    chk("EmpiricalBayesBlend / QuantileForest / KNNQuantile",
        *has("src.utils.ml_utils.model.offset_learned", "EmpiricalBayesBlend",
             "QuantileForest", "KNNQuantile"))

    print("\nP7 Model 3 detector hygiene")
    chk("STATIC_IN_PATIENT restored",
        hasattr(D, "static_in_patient")
        and "static_in_patient" in inspect.getsource(D.build_score_matrix))
    chk("kernel fit sampling", hasattr(D, "_fit_sample"))
    ts = inspect.getsource(D.tune_detectors)
    chk("7 tuning families", all(f in ts for f in ["isolation_forest", "lof", "gmm", "pca",
                                                   "cusum", "page_hinkley", "statistical"]))
    chk("tuner seeded at the default, not -inf",
        "default_score" in ts and "beat_default" in ts)
    chk("rule-engine reference excluded from serving",
        "d_fixed_threshold" in inspect.getsource(D.choose_serving_detector))
    chk("+/-inf sanitised before ranking", "np.inf" in inspect.getsource(D.ap_lift))

    print("\nP8 Model 4 symptom heads")
    chk("15 symptoms across 4 mechanisms",
        len(SL.SYMPTOMS) == 15 and len(set(SL.SYMPTOM_MECHANISM.values())) == 4)
    th = inspect.getsource(train_symptom_heads)
    chk("calibration/threshold split BY PATIENT (notebook bug fixed)",
        "symptom-calibration" in th
        and "assert not (set(va_cal.series_id) & set(va_thr.series_id))" in th)
    es = inspect.getsource(eval_symptom)
    chk("rare-event metrics, not accuracy", "PR_lift" in es and "Brier_skill" in es)
    chk("latent generator variables dropped",
        {"adherence_trait", "frailty"} <= CausalFeatureBuilder.DROP)

    print("\nP9 evaluation machinery")
    chk("block importance / parity / latency / round-trip / concentration",
        *has("src.utils.ml_utils.metric.interpretability", "block_importance",
             "walk_forward_parity", "batch_latency", "bundle_round_trip",
             "error_concentration"))
    la = inspect.getsource(leakage_audit)
    probes = ["corrupting the future", "steps later", "shuffled-target",
              "symptom feature reproduces", "reads today, not tomorrow", "latent generator"]
    chk("all 6 notebook leakage probes present", all(p in la for p in probes),
        [p for p in probes if p not in la])

    print("\nP10 serving layer")
    for f in ["app.py", "templates/index.html", "static/app.js", "static/charts.js",
              "static/style.css"]:
        chk(f, os.path.exists(f) and os.path.getsize(f) > 0)
    routes = {r.path for r in A.app.routes}
    need = {"/", "/api/schema", "/api/health", "/api/model", "/api/predict", "/api/train",
            "/api/train/status", "/api/train/cancel"}
    chk("all 8 routes registered", need <= routes, sorted(need - routes))

    print("\nP11 project hygiene")
    for f in ["README.md", "Dockerfile", ".github/workflows/deploy.yaml", "data/README.md",
              ".gitignore"]:
        chk(f, os.path.exists(f) and os.path.getsize(f) > 0)
    with open(".gitignore") as fh:
        gi = fh.read()
    chk(".gitignore covers data / Artifacts / logs / final_model",
        all(x in gi for x in ["data/", "Artifacts/", "logs/", "final_model/*"]))
    # The promoted bundle is the one exception: tracked through LFS so a fresh checkout can
    # serve without a training run first. The rest of final_model/ stays ignored.
    chk("final_model/model.pkl is un-ignored and LFS-tracked",
        "!final_model/model.pkl" in gi
        and "final_model/*.pkl filter=lfs" in open(".gitattributes", encoding="utf-8").read())
    chk("old SPA deleted",
        not any(os.path.exists(p) for p in ["templates/app.js", "templates/style.css"]))

    print("\n" + ("EVERY AGREED CAPABILITY IS PRESENT" if not FAILS
                  else f"{len(FAILS)} MISSING:\n  - " + "\n  - ".join(FAILS)))
    return 1 if FAILS else 0


def test_plan_completeness():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
