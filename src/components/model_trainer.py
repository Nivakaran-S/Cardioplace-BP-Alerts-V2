import os
import sys
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DetectorMetricArtifact,
    ModelTrainerArtifact,
    OffsetMetricArtifact,
    RuleEngineArtifact,
    SafetyGateArtifact,
)
from src.entity.config_entity import ModelTrainerConfig
from src.exception.custom_exception import CustomException
from src.logging.logger import logging, timer
from src.utils.main_utils.utils import (
    load_dataframe,
    read_yaml_file,
    save_object,
    save_report,
    write_yaml_file,
)
from src.utils.ml_utils.metric.interpretability import (
    batch_latency,
    block_importance,
    bundle_round_trip,
    error_concentration,
    walk_forward_parity,
)
from src.utils.ml_utils.metric.regression_metric import (
    forecast_baselines,
    get_forecast_score,
    select_and_decide,
    ship_decision,
    shipped_forecasters,
)
from src.utils.ml_utils.metric.timeseries_metric import (
    FAIR_MARGIN_PP,
    DriftMonitor,
    decision_curve,
    detector_scorecard,
    elicitation_table,
    offset_scorecard,
    slice_gate,
    tier_metrics,
)
from src.utils.ml_utils.model.chain_eval import run_chain_evaluation
from src.utils.ml_utils.model.classifier_head import (
    build_classifier_frame,
    train_symptom_heads,
    train_tier_head,
)
from src.utils.ml_utils.model.coherence import pair_coherence
from src.utils.ml_utils.model.detector import (
    BASE_DETECTORS,
    DETECTOR_FAMILY,
    EarlyWarningDataset,
    build_score_matrix,
    build_serving_detector,
    choose_serving_detector,
    fit_default_detectors,
    fuse_detectors,
    tune_detectors,
    widen_detector_set,
)
from src.utils.ml_utils.model.estimator import (
    BaselineForecaster,
    BPPredictor,
    champion_challenger,
    enabled_kinds,
    explain_prediction,
    fit_quantile_interval,
    make_model,
    run_sweep,
    tune,
)
from src.utils.ml_utils.model.offset import observed_band, search_offset
from src.utils.ml_utils.model.offset_learned import train_learned_offset
from src.utils.ml_utils.rule_engine.advisory import arbitrate, build_advisories
from src.utils.ml_utils.rule_engine.engine import RuleEngine
from src.utils.ml_utils.rule_engine.labels import (
    disposition_coverage,
    label_quality,
    simulate_dispositions,
)
from src.utils.ml_utils.rule_engine.registry import TIERS, build_registry
from src.utils.ml_utils.rule_engine.synthetic import (
    ExtendedRuleEngine,
    generate_synthetic_cohort,
    prepare_synthetic_run,
    rule_coverage,
    synth_param_table,
)
from src.utils.ml_utils.safety.gates import (
    AbstentionPolicy,
    cold_start_curve,
    provenance_guard,
    run_safety_gates,
)
from src.utils.ml_utils.safety.missingness import missingness_sweep

load_dotenv()


# Only assign when the variable is actually set: os.environ[...] = None raises TypeError,
# which would take the whole module down at import time against an empty .env.
for _var in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
    _val = os.getenv(_var)
    if _val:
        os.environ[_var] = _val


def _log_candidates_enabled() -> bool:
    """Whether to log the ~136 non-final candidate runs.

    Off by default. Against a remote tracking server each nested run was measured at roughly
    75 seconds, so the full catalogue adds hours of network latency after the modelling has
    already finished. Read in exactly one place: this switch previously existed twice with
    opposite defaults, so the log said "finals only" while candidates were still uploading.
    """
    return os.getenv("MLFLOW_LOG_CANDIDATES", "0").strip().lower() in {"1", "true", "yes", "on"}


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.best_params = {}
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _mlflow_ready() -> bool:
        # Registry follows tracking. Hardcoding it lets the two drift onto different
        # DagsHub repos, so runs land in one project and registered models in another.
        _registry = os.getenv("MLFLOW_TRACKING_URI")
        if _registry:
            mlflow.set_registry_uri(_registry)
        return urlparse(mlflow.get_tracking_uri()).scheme != "file"

    @staticmethod
    def _clean_metrics(metrics: dict, prefix: str = "") -> dict:
        """Finite floats only, under names MLflow accepts."""
        out = {}
        for k, v in (metrics or {}).items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fv):
                continue
            name = f"{prefix}{k}".replace("%", "pct").replace("(", "").replace(")", "")
            name = "".join(c if (c.isalnum() or c in "_-./ ") else "_" for c in name)
            out[name] = fv
        return out

    def track_mlflow(self, best_model, metrics: dict, tag: str = "forecast"):
        try:
            with mlflow.start_run(nested=True):
                mlflow.set_tag("stage", tag)
                for k, v in self._clean_metrics(metrics).items():
                    mlflow.log_metric(k, v)
                if best_model is not None and self._mlflow_ready():
                    mlflow.sklearn.log_model(best_model, "model")
        except Exception as exc:
            # Tracking is observability, not a training dependency. A missing DagsHub
            # credential must not lose a completed run.
            logging.warning("mlflow tracking skipped (%s): %s", tag, exc)

    def _log_candidate(self, name: str, category: str, selection: str,
                       tags: dict, metrics: dict, params: dict = None,
                       model=None, registered_name: str = None) -> bool:
        """One MLflow run for one trained model.

        `selection` is the point of this: every candidate the pipeline fitted is recorded,
        and the one that actually serves is tagged `final`. Logging only the winner made
        the tracking store agree with itself no matter what was selected -- there was no
        way to see what a choice was made against, which is most of what a run is for.
        """
        if selection != "final" and not _log_candidates_enabled():
            return False
        try:
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tag("category", category)
                mlflow.set_tag("selection", selection)         # final | candidate
                mlflow.set_tag("is_final", str(selection == "final").lower())
                for k, v in (tags or {}).items():
                    if v is not None:
                        mlflow.set_tag(k, str(v))
                for k, v in (params or {}).items():
                    if v is not None:
                        mlflow.log_param(k, v)
                for k, v in self._clean_metrics(metrics).items():
                    mlflow.log_metric(k, v)
                # Only finals carry a serialised estimator: logging every candidate would
                # multiply artifact storage by ~100 for models nothing will ever load.
                if model is not None and selection == "final" and self._mlflow_ready():
                    try:
                        mlflow.sklearn.log_model(model, "model",
                                                 registered_model_name=registered_name)
                    except Exception as exc:
                        logging.warning("mlflow model artifact skipped for %s: %s", name, exc)
            return True
        except Exception as exc:
            logging.warning("mlflow candidate run skipped (%s): %s", name, exc)
            return False

    def track_model_catalogue(self, config, R, shipped, off_report, offset_search,
                              offset_model, det_report, det_report_val, det_name,
                              tier_metrics_df, predictor) -> int:
        """Log every model this run fitted, with the shipped ones tagged `final`.

        Four categories are trained and selected from -- forecasters, the personalisation
        offset, early-warning detectors and the classifier heads -- and previously exactly
        one headline run reached MLflow. Each candidate now gets its own nested run so the
        selection is auditable from the tracking store alone.
        """
        # Every candidate is one HTTP round trip to the tracking server. On this cohort
        # that is ~136 nested runs, and against a REMOTE tracking server (the .env here
        # points at DagsHub) each one was measured at roughly 75 seconds -- nearly three
        # hours of network latency bolted onto a training run, after all the modelling had
        # already finished.
        #
        # So the default is now finals-only. The shipped models are what anyone actually
        # opens; the full catalogue is a deliberate opt-in via MLFLOW_LOG_CANDIDATES=1,
        # which is the right way round for something that can triple a run's wall clock.
        finals_only = not _log_candidates_enabled()
        if finals_only:
            logging.info("mlflow: logging shipped models only. Set MLFLOW_LOG_CANDIDATES=1 "
                         "for the full candidate catalogue (~136 nested runs; slow against "
                         "a remote tracking server).")

        n = 0
        run_tags = dict(run_id=config.run_id, model_version=f"hemobp-bp-{config.run_id}")

        # ---- Model 1: forecasters (learned kinds and baselines, per signal x horizon) ----
        if R is not None and len(R):
            keys = ["family", "signal", "model", "horizon"]
            for (family, signal, model_name, h), grp in R.groupby(keys, dropna=False):
                metrics = {}
                for split, sg in grp.groupby("split"):
                    row = sg.iloc[0]
                    metrics.update(self._clean_metrics(
                        {k: row.get(k) for k in ("MAE", "MAE_lo", "MAE_hi", "RMSE", "R2",
                                                 "bias", "DirAcc", "target_sd", "n")},
                        prefix=f"{split}_"))
                ship_family, ship_name = shipped.get(signal, (None, None))
                is_final = family == ship_family and model_name == ship_name
                n += self._log_candidate(
                    name=f"forecaster/{signal}/h{int(h)}/{model_name}",
                    category="forecaster", selection="final" if is_final else "candidate",
                    tags=dict(signal=signal, horizon=int(h), family=family,
                              model=model_name, **run_tags),
                    metrics=metrics,
                    params=dict(**self.best_params.get((model_name, signal), {}),
                                horizon=int(h)),
                    model=(predictor.b["forecasters"].get((signal, int(h)))
                           if is_final else None),
                    registered_name=f"hemobp-forecaster-{signal}-h{int(h)}")

        # ---- Model 2: the offset, plus every warm/k/q the search fitted ----------------
        if off_report is not None and len(off_report):
            for model_name, grp in off_report.groupby("model"):
                metrics = {}
                for _, row in grp.iterrows():
                    p = str(row["split"]).replace(" ", "_").replace("-", "_")
                    metrics.update(self._clean_metrics(
                        {k: row.get(k) for k in ("MAE", "lo", "hi", "R2", "within_10mmHg")},
                        prefix=f"{p}_"))
                is_final = model_name == "learned (capped blend)"
                n += self._log_candidate(
                    name=f"offset/{model_name}", category="offset",
                    selection="final" if is_final else "candidate",
                    tags=dict(model=model_name, **run_tags), metrics=metrics,
                    params=(dict(warm=offset_model.warm, k=offset_model.k, q=offset_model.q)
                            if is_final else None),
                    registered_name="hemobp-offset")
        if offset_search is not None and len(offset_search):
            for i, row in offset_search.reset_index(drop=True).iterrows():
                sel = ("final" if (int(row.warm) == int(offset_model.warm)
                                   and float(row.k) == float(offset_model.k)
                                   and float(row.q) == float(offset_model.q))
                       else "candidate")
                n += self._log_candidate(
                    name=f"offset-search/warm{int(row.warm)}-k{row.k:g}-q{row.q:g}",
                    category="offset-search", selection=sel,
                    tags=dict(**run_tags),
                    metrics=self._clean_metrics({"mae_fit": row.mae_fit, "n": row.n}),
                    params=dict(warm=int(row.warm), k=float(row.k), q=float(row.q)))

        # ---- Model 3: every detector scored, at the configured alert budget -------------
        for label, rep in (("val", det_report_val), ("test", det_report)):
            if rep is None or not len(rep):
                continue
            at = rep[rep.budget_pct == config.alert_budget_pct]
            for _, row in at.iterrows():
                if label == "test":
                    continue        # metrics merged into the val run below
                te = (det_report[(det_report.budget_pct == config.alert_budget_pct)
                                 & (det_report.detector == row.detector)]
                      if det_report is not None and len(det_report) else None)
                metrics = self._clean_metrics(
                    {k: row.get(k) for k in ("precision", "recall", "lift", "auc_roc",
                                             "auc_pr", "auc_pr_lift", "base_rate",
                                             "lead_days_median", "alerts_per_patient_week")},
                    prefix="val_")
                if te is not None and len(te):
                    metrics.update(self._clean_metrics(
                        {k: te.iloc[0].get(k) for k in
                         ("precision", "recall", "lift", "auc_roc", "auc_pr",
                          "auc_pr_lift", "lead_days_median", "alerts_per_patient_week")},
                        prefix="test_"))
                is_final = row.detector == det_name
                n += self._log_candidate(
                    name=f"detector/{row.detector}", category="detector",
                    selection="final" if is_final else "candidate",
                    tags=dict(detector=row.detector, family=row.get("family"), **run_tags),
                    metrics=metrics,
                    params=dict(budget_pct=config.alert_budget_pct,
                                warn_window=config.warn_window,
                                event_quantile=config.event_quantile),
                    registered_name="hemobp-detector")

        # ---- classifier heads -----------------------------------------------------------
        if tier_metrics_df is not None and len(tier_metrics_df):
            for _, row in tier_metrics_df.iterrows():
                d = row.to_dict()
                tier = d.pop("tier", "?")
                n += self._log_candidate(
                    name=f"classifier/tier-head/{tier}", category="classifier",
                    selection="final", tags=dict(tier=tier, **run_tags),
                    metrics=self._clean_metrics(d))

        logging.info("mlflow: logged %d model runs (forecasters, offset, detectors, heads); "
                     "shipped ones tagged selection=final", n)
        return n

    # ------------------------------------------------------------------ forecaster
    def train_forecasters(self, F, features, config):
        """Bake-off across the registry, tuning, then the paired ship decision."""
        kinds = enabled_kinds(config.model_tier)
        logging.info("architecture bake-off: %d candidates at tier '%s' -- %s",
                     len(kinds), config.model_tier, ", ".join(kinds))
        with timer(f"default sweep ({len(kinds)} architectures)"):
            R_default, preds_default, _ = run_sweep(F, features, config)
        if R_default.empty:
            raise ValueError("the sweep produced no scorable rows; the cohort is too small")

        R, preds = R_default, preds_default
        tune_log, comparison = pd.DataFrame(), pd.DataFrame()
        if config.tune:
            with timer(f"hyperparameter search ({config.tuner})"):
                tune_log = tune(F, features, config, self.best_params)
            if self.best_params:
                # Re-run the identical sweep with tuned parameters so every number below
                # describes the tuned model. R_default survives as the null hypothesis.
                with timer("tuned sweep"):
                    R, preds, _ = run_sweep(F, features, config, params=self.best_params)
                comparison = self._tuning_comparison(R_default, R)

        # `preds` carries per-row errors on the same test rows for every candidate, which is
        # what makes the tie set and the ship rule PAIRED rather than a comparison of two
        # marginal confidence intervals.
        winner, decision, tie_cmp = select_and_decide(R, config, preds=preds)
        logging.info("selection: %s", winner)
        if len(tie_cmp):
            tied = tie_cmp[tie_cmp.ties_leader]
            logging.info("tie set: %d of %d candidates indistinguishable from the leader "
                         "(%s)", len(tied), len(tie_cmp), ", ".join(tied.model.head(8)))
        if len(decision):
            logging.info("ship decision:\n%s", decision.to_string(index=False))
        return R, tune_log, comparison, winner, decision, tie_cmp

    @staticmethod
    def _tuning_comparison(R_default: pd.DataFrame, R_tuned: pd.DataFrame) -> pd.DataFrame:
        """A tuning gain smaller than the bootstrap CI is not a gain."""
        d = R_default[R_default.split == "test"]
        t = R_tuned[R_tuned.split == "test"]
        cmp = (d.groupby(["signal", "model"]).MAE.mean().rename("default").to_frame()
               .join(t.groupby(["signal", "model"]).MAE.mean().rename("tuned"))
               .join(d.assign(w=lambda x: x.MAE_hi - x.MAE_lo)
                     .groupby(["signal", "model"]).w.mean().rename("CI_width")))
        cmp["delta"] = (cmp["default"] - cmp["tuned"]).round(3)
        cmp["beyond_noise"] = cmp.delta.abs() > cmp.CI_width
        logging.info("tuning: %d of %d cells moved beyond the bootstrap CI",
                     int(cmp.beyond_noise.sum()), len(cmp))
        return cmp.round(3).reset_index()

    # ------------------------------------------------------------------ detector
    def train_detectors(self, F, features, config, winner, shipped=None):
        with timer("early-warning dataset"):
            ew = EarlyWarningDataset(config)
            D = ew.build(F, features)

        imputer, dense_cols, X_train, X_all = build_score_matrix(D, features)
        D = fit_default_detectors(D, X_train, X_all)
        best_detector = tune_detectors(D, imputer, dense_cols, X_train, X_all)

        # The forecast-based detectors reuse Part 6's forecaster. It has to be the one
        # that SHIPS, not the learned winner: if the ship decision sent a baseline to
        # production, scoring these detectors against a learned forecaster would rank
        # them on a model no request will ever see.
        forecaster, fc_target = None, f"y_sbp_h{config.horizons[0]}"
        # "ridge" as the fallback, not MODEL_KINDS[-1]: that index is whatever the enabled
        # tier happens to end with (an ensemble at medium) and can be absent entirely at low.
        family, name = (shipped or {}).get("sbp", ("learned", winner.get("sbp", "ridge")))
        try:
            if family == "baseline":
                forecaster = BaselineForecaster(name, "sbp", config.horizons[0])
            else:
                fit_rows = D[(D.split == "train") & D[fc_target].notna()]
                if len(fit_rows) > 200:
                    if len(fit_rows) > config.max_train_rows:
                        fit_rows = fit_rows.sample(config.max_train_rows,
                                                   random_state=config.seed)
                    forecaster = make_model(
                        name, **self.best_params.get((name, "sbp"), {})).fit(
                        fit_rows[features], fit_rows[fc_target])
        except Exception as exc:
            logging.warning("forecast-residual detector unavailable: %s", exc)

        detectors = list(BASE_DETECTORS)
        detectors += [d for d in widen_detector_set(D, X_train, X_all, features,
                                                    forecaster, fc_target)
                      if d not in detectors]
        detectors += [d for d in fuse_detectors(D, detectors) if d not in detectors]
        logging.info("detector set: %d scorers across %d families", len(detectors),
                     len({DETECTOR_FAMILY.get(d, "?") for d in detectors}))

        D_test = D[(D.split == "test") & D.event_next.notna()]
        D_val = D[(D.split == "val") & D.event_next.notna()]
        sessions_per_week = 7.0 / max(float(D.days_since_last.median()), 1e-9)
        det_report = detector_scorecard(D_test, detectors, sessions_per_week)
        # Selection reads the VAL scorecard; test is kept for reporting only. Choosing
        # which detector ships by its test score would make the reported test number an
        # estimate of the winner's luck rather than of its performance.
        det_report_val = detector_scorecard(D_val, detectors, sessions_per_week)
        for rep in (det_report, det_report_val):
            if len(rep):
                rep["family"] = rep.detector.map(DETECTOR_FAMILY).fillna("?")
        return (D, D_val, D_test, detectors, det_report, det_report_val, imputer,
                dense_cols, best_detector, forecaster, sessions_per_week)

    # ------------------------------------------------------------------ orchestration
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            config = self.model_trainer_config
            os.makedirs(config.report_dir, exist_ok=True)

            meta = read_yaml_file(self.data_transformation_artifact.feature_list_file_path)
            features = list(meta["features"])
            F = load_dataframe(self.data_transformation_artifact.feature_file_path)
            panel = load_dataframe(self.data_transformation_artifact.panel_file_path)
            for df in (F, panel):
                if "ts" in df.columns:
                    df["ts"] = pd.to_datetime(df.ts, errors="coerce")
                df["series_id"] = df.series_id.astype(str)
            logging.info("loaded %d feature rows, %d panel rows, %d features",
                         len(F), len(panel), len(features))

            # ---- 1. forecaster ------------------------------------------------
            R, tune_log, tune_cmp, winner, decision, tie_cmp = self.train_forecasters(
                F, features, config)
            save_report(config.report_path("forecast_scorecard.csv"), R)
            if len(tie_cmp):
                save_report(config.report_path("paired_comparison.csv"), tie_cmp)
            if len(tune_log):
                save_report(config.report_path("tuning_log.csv"), tune_log)
            if len(tune_cmp):
                save_report(config.report_path("tuning_default_vs_tuned.csv"), tune_cmp)
            if len(decision):
                save_report(config.report_path("ship_decision.csv"), decision)
            # The verdict decides what serves, here and downstream. A signal whose
            # learned model did not clear the bar ships the baseline that beat it.
            shipped = shipped_forecasters(decision, winner)
            logging.info("ship decision applied: %s",
                         {s: f"{f}:{n}" for s, (f, n) in shipped.items()})

            # ---- 2. prediction intervals --------------------------------------
            interval_signal = "sbp"
            interval_horizon = config.horizons[min(1, len(config.horizons) - 1)]
            qmodels, qhat = fit_quantile_interval(F, features, config, self.best_params,
                                                  interval_signal, interval_horizon)

            # ---- 3. personalisation offset ------------------------------------
            with timer("offset search"):
                offset_model, offset_search = search_offset(F, config)
            OFF = offset_model.transform(F)
            assert OFF.empty or OFF.threshold.max() < config.emergency_floor_mmHg, \
                "offset breached the emergency floor"
            band = observed_band(F, offset_model.warm)
            M_all = OFF.merge(band, on="series_id").dropna(subset=["actual"])
            M_hold = M_all[M_all.patient_split == "holdout"]
            off_report = pd.DataFrame()
            offset_artifact = None
            if len(M_hold) >= 10:
                off_report = pd.concat([
                    offset_scorecard(M_hold, "held-out patients",
                                     config.population_threshold_mmHg, config),
                    offset_scorecard(M_all, "all patients",
                                     config.population_threshold_mmHg, config)]).sort_values(
                    ["split", "MAE"])
                save_report(config.report_path("offset_scorecard.csv"), off_report)
                h_ = off_report[off_report.split == "held-out patients"]
                learned = float(h_[h_.model == "learned (capped blend)"].MAE.iloc[0])
                alts = h_[h_.model != "learned (capped blend)"]
                base = float(alts.MAE.min())
                alt_name = str(alts.loc[alts.MAE.idxmin()].model)
                ci_w = float(h_[h_.model == "learned (capped blend)"].eval("hi - lo").iloc[0])
                verdict, gain = ship_decision(learned, base, ci_w)
                logging.info("offset decision -> %s (gain %+.2f mmHg vs %s, CI width %.2f)",
                             verdict, gain, alt_name, ci_w)
                # Unlike the forecaster, the blend is NOT auto-replaced on a BASELINE
                # verdict, because on this target the alternatives are not a simpler way
                # to do the same job:
                #   cohort-only  collapses to a constant under the caps (its 160-174
                #                band clips to 155 for every patient), which is not
                #                personalisation at all.
                #   personal-only does vary, but it has no shrinkage, so a patient with
                #                five readings gets a threshold set by five readings --
                #                the failure the blend's n/(n+k) weight exists to stop.
                # The separation is also inside the noise: the held-out spread is under
                # half a mmHg against a bootstrap CI several mmHg wide, and the blend is
                # ahead on the larger all-patients sample. "BASELINE" here means "not
                # proven better", and that is what gets recorded.
                if verdict != "learned model":
                    logging.info("offset retained: %s wins by %.2f mmHg on %d held-out "
                                 "patients against a CI %.2f wide -- inside the noise, "
                                 "and it carries no shrinkage for short histories",
                                 alt_name, -gain, len(M_hold), ci_w)
                offset_artifact = OffsetMetricArtifact(
                    label="capped shrinkage blend", warm=offset_model.warm,
                    k=offset_model.k, q=offset_model.q, n_patients=len(M_hold),
                    mae=round(learned, 3),
                    max_threshold=float(OFF.threshold.max()),
                    emergency_floor=config.emergency_floor_mmHg,
                    n_capped=int(OFF.capped.sum()),
                    ship=verdict, best_legal_alternative=alt_name,
                    gain_mmHg=round(gain, 3))
            if len(offset_search):
                save_report(config.report_path("offset_search.csv"), offset_search)

            # ---- 3b. learned offset ---------------------------------------------
            # The blend keys on age, sex and the patient's own band. This learns the
            # threshold from 37 features including the clinical context the blend cannot
            # see. It ships only on a paired win over the blend on held-out patients, so
            # the default outcome is that nothing changes.
            offset_board, offset_learned, offset_promo = pd.DataFrame(), None, {}
            if getattr(config, "learned_offset", False):
                with timer("learned offset (13 candidates)"):
                    offset_board, offset_learned, offset_promo = train_learned_offset(
                        F, config, offset_model, seed=config.seed)
                if len(offset_board):
                    save_report(config.report_path("offset_learned_board.csv"), offset_board)
                if offset_promo:
                    write_yaml_file(config.report_path("offset_promotion.yaml"),
                                    offset_promo, replace=True)

            # ---- 4. early-warning detectors -----------------------------------
            with timer("detector training"):
                (D, D_val, D_test, detectors, det_report, det_report_val, imputer,
                 dense_cols, best_detector, det_forecaster,
                 sessions_per_week) = self.train_detectors(
                    F, features, config, winner, shipped)
            detector_artifacts = []
            if len(det_report_val):
                save_report(config.report_path("detector_scorecard_val.csv"), det_report_val)
            if len(det_report):
                save_report(config.report_path("detector_scorecard.csv"), det_report)
                at_budget = det_report[det_report.budget_pct == config.alert_budget_pct]
                for r in at_budget.sort_values("auc_pr", ascending=False).head(5).itertuples():
                    detector_artifacts.append(DetectorMetricArtifact(
                        detector=r.detector, budget_pct=float(r.budget_pct),
                        base_rate=float(r.base_rate), precision=float(r.precision),
                        recall=float(r.recall), lift=float(r.lift),
                        auc_roc=float(r.auc_roc), auc_pr=float(r.auc_pr),
                        auc_pr_lift=float(r.auc_pr_lift),
                        lead_days_median=float(r.lead_days_median),
                        alerts_per_patient_week=float(r.alerts_per_patient_week)))

            # ---- 5. rule engine ------------------------------------------------
            registry = build_registry()
            save_report(config.report_path("rule_registry.csv"),
                        registry.assign(needs=registry.needs.astype(str),
                                        blocked_by=registry.blocked_by.astype(str)))
            personal_thr = dict(zip(OFF.series_id, OFF.threshold)) if len(OFF) else {}
            engine = RuleEngine(registry, personal_thr)
            with timer("rule engine (population mode)"):
                alerts_pop = engine.run(panel, use_personalisation=False)
            with timer("rule engine (personalised mode)"):
                alerts_pers = engine.run(panel, use_personalisation=True)
            save_report(config.report_path("alerts_population.csv"), alerts_pop)
            logging.info("%d readings evaluated | %d alerts fired (%.1f%%)",
                         len(alerts_pop), int(alerts_pop.fired.sum()),
                         100 * alerts_pop.fired.mean())

            rule_artifact = RuleEngineArtifact(
                alerts_file_path=config.report_path("alerts_population.csv"),
                registry_file_path=config.report_path("rule_registry.csv"),
                rows_evaluated=int(len(alerts_pop)),
                alerts_fired=int(alerts_pop.fired.sum()),
                emergency_fired=int(alerts_pop.is_emergency.sum()),
                evaluable_rules=int((registry.status == "EVALUABLE").sum()),
                blocked_rules=int((registry.status == "BLOCKED_ON_INPUTS").sum()))

            # ---- 6. label engineering -----------------------------------------
            save_report(config.report_path("disposition_coverage.csv"),
                        disposition_coverage(registry, TIERS))
            dispo = simulate_dispositions(alerts_pop, panel, seed=config.seed)
            if len(dispo):
                write_yaml_file(config.report_path("label_quality.yaml"),
                                label_quality(dispo, panel), replace=True)

            # ---- 7. classifier heads -------------------------------------------
            with timer("tier + rule heads"):
                CLS = build_classifier_frame(F, alerts_pop, config.horizons[0])
                tier_clf, rule_clf, rule_report, te_c, proba_te = train_tier_head(
                    CLS, features, config, seed=config.seed)
            if len(rule_report):
                save_report(config.report_path("rule_head_report.csv"), rule_report)

            # ---- 7b. Model 4: symptom heads (SYNTHETIC LABELS) --------------------
            sym_models, sym_cuts = {}, {}
            with timer("symptom heads"):
                sym_models, sym_cuts, sym_board, sym_skipped = train_symptom_heads(
                    F, features, config, seed=config.seed)
            if len(sym_board):
                save_report(config.report_path("symptom_board.csv"), sym_board)
            if len(sym_skipped):
                save_report(config.report_path("symptom_skipped.csv"), sym_skipped)

            # ---- 8. clinical-utility metrics ------------------------------------
            tier_rows = []
            if tier_clf is not None and proba_te is not None and len(te_c):
                classes = list(tier_clf.classes_)
                for tier in [t for t in classes if t != "NO_ALERT"]:
                    j = classes.index(tier)
                    y_bin = (te_c.tier_future.values == tier).astype(int)
                    if y_bin.sum() < 20:
                        continue
                    p = proba_te[:, j]
                    thr = float(np.percentile(p, 100 - config.alert_budget_pct))
                    tier_rows.append(dict(tier=tier,
                                          **tier_metrics(y_bin, p, thr, float(y_bin.mean()))))
            tier_metrics_df = pd.DataFrame(tier_rows)
            if len(tier_metrics_df):
                save_report(config.report_path("tier_metrics.csv"), tier_metrics_df)
                focus = tier_metrics_df.sort_values("AUC_PR").iloc[-1].tier
                j = list(tier_clf.classes_).index(focus)
                y_f = (te_c.tier_future.values == focus).astype(int)
                p_f = proba_te[:, j]
                save_report(config.report_path("decision_curve.csv"), decision_curve(y_f, p_f))
                save_report(config.report_path("elicitation.csv"),
                            elicitation_table(y_f, p_f, sessions_per_week))

            # ---- 9. serving bundle ----------------------------------------------
            # Previously an IsolationForest was hardcoded here whatever the scorecard
            # said -- on this cohort it ranks 18th of 20, below the rule-engine
            # reference it is supposed to improve on. The scorecard now decides.
            det_name = choose_serving_detector(det_report_val, config.alert_budget_pct)
            det = build_serving_detector(det_name, D, imputer, dense_cols, features,
                                         det_forecaster, config)
            logging.info("serving detector: %s (selected on val at the %.0f%% budget)",
                         det_name, config.alert_budget_pct)
            # No sbp_history argument: D_val holds many patients, so the reference band
            # must be taken per series_id. Passing the pooled column would calibrate the
            # cut against a population band the serving path never uses.
            val_scores = (det.score(D_val) if len(D_val) else np.array([0.0]))
            val_scores = np.asarray(val_scores, float)
            val_scores = val_scores[np.isfinite(val_scores)]
            if not len(val_scores):
                val_scores = np.array([0.0])
            det_cut = float(np.percentile(val_scores, 100 - config.alert_budget_pct))

            with timer("freeze serving artifacts"):
                predictor = BPPredictor.build(
                    F, features, config, winner, self.best_params, offset_model,
                    qmodels, qhat, interval_signal, interval_horizon,
                    det, imputer, dense_cols, det_cut, shipped,
                    symptom_models=sym_models, symptom_cuts=sym_cuts)
            os.makedirs(os.path.dirname(config.bundle_file_path), exist_ok=True)
            predictor.save(config.bundle_file_path)
            logging.info("bundle %s (%.1f MB) | model_version %s",
                         os.path.basename(config.bundle_file_path),
                         os.path.getsize(config.bundle_file_path) / 1e6,
                         predictor.b["model_version"])

            # ---- 9b. is the SHIPPED (sbp, dbp) pair coherent? ---------------------
            # Nothing upstream has ever inspected the joint: select_and_decide ranks per
            # signal, so sbp and dbp can ship different architectures that are each excellent
            # on their own marginal MAE and jointly emit a diastolic above the systolic. This
            # is the first measurement of whether that actually happens, and it is scored on
            # the frozen forecasters -- the ones that will serve -- rather than on sweep
            # predictions, for the same reason run_fairness audits what ships.
            pair_rows = []
            try:
                te_pair = F[(F.split == "test") & (F.patient_split == "fit")] \
                    if "patient_split" in F else F[F.split == "test"]
                if len(te_pair) > config.max_eval_rows:
                    te_pair = te_pair.sample(config.max_eval_rows, random_state=config.seed)
                Xp = te_pair[features]
                pp_ref = (te_pair["pp_mean7"].to_numpy(dtype=float)
                          if "pp_mean7" in te_pair else None)
                for h in config.horizons:
                    ms = predictor.b["forecasters"].get(("sbp", h))
                    md = predictor.b["forecasters"].get(("dbp", h))
                    if ms is None or md is None:
                        continue
                    pair_rows.append(pair_coherence(
                        np.asarray(ms.predict(Xp), dtype=float),
                        np.asarray(md.predict(Xp), dtype=float),
                        pp_ref=pp_ref, bounds=predictor.b.get("coherence"), horizon=h))
            except Exception as exc:                                  # noqa: BLE001
                logging.warning("pair coherence report unavailable: %s", exc)
            pair_df = pd.DataFrame(pair_rows)
            if len(pair_df):
                save_report(config.report_path("pair_coherence.csv"), pair_df)
                worst = float(pair_df.violation_rate.max())
                logging.info("forecast pair coherence: worst violation rate %.4f across %d "
                             "horizons (%s)", worst, len(pair_df),
                             "clean" if worst == 0 else
                             "; ".join(r for r in pair_df.top_reasons if r))

            # ---- 9c. the chained symptom path: cuts, its shift, and a board --------
            # Runs HERE, after the freeze, not inside train_symptom_heads. The heads are
            # trained at 7b and the forecasters do not exist until 9, so there is nothing to
            # chain from at head-training time -- and measuring the frozen forecaster is the
            # more honest choice anyway.
            chain_board, chain_shift_df, chain_cuts = (pd.DataFrame(), pd.DataFrame(), {})
            try:
                with timer("chained symptom evaluation"):
                    chain_board, chain_shift_df, chain_cuts = run_chain_evaluation(
                        panel, F, predictor, features, config, seed=config.seed)
            except Exception as exc:                                  # noqa: BLE001
                logging.warning("chained symptom evaluation skipped: %s", exc)
            if len(chain_board):
                save_report(config.report_path("symptom_chain_board.csv"), chain_board)
                wins = int(chain_board.chained_wins.sum())
                logging.info("[SYNTHETIC LABELS] chained vs direct: %d of %d heads improve on "
                             "a paired test; %d beat the patient's own 30-session rate",
                             wins, len(chain_board),
                             int(chain_board.beats_personal_rate.sum()))
            if len(chain_shift_df):
                save_report(config.report_path("symptom_chain_shift.csv"), chain_shift_df)
            if chain_cuts:
                # Stored beside the observed-row cuts, never replacing them: the two apply to
                # different input distributions and the serving path picks by which it used.
                predictor.b["symptom_cuts_chained"] = chain_cuts
                predictor.b["symptom_chain"] = {
                    "n_rows": int(len(chain_board)),
                    "shift_max_psi": (float(chain_shift_df.psi.max())
                                      if len(chain_shift_df) and "psi" in chain_shift_df
                                      else None),
                    "note": "cuts refitted on chained rows; the observed-row cut does not "
                            "carry its alert-budget meaning to this distribution",
                }
                predictor.save(config.bundle_file_path)

            # ---- 10. fairness ----------------------------------------------------
            fairness = self.run_fairness(F, features, config, winner, M_hold,
                                         D_test, det_report, shipped)
            if len(fairness):
                save_report(config.report_path("fairness.csv"), fairness)

            # ---- 10b. subgroup parity of the symptom heads ------------------------
            # Deliberately NOT folded into `fairness` above. Gate 5 is critical, and a
            # subgroup disparity in a GENERATED hazard model must never block promotion of a
            # forecaster validated on real blood pressure. The generator has structural
            # asymmetries by construction -- frailty is drawn Beta(2,5), on_bb shifts heart
            # rate -- so this is a live risk, not a hypothetical one. Reported, gated
            # non-critically, kept in its own file.
            sym_fair = pd.DataFrame()
            try:
                sym_fair = self.symptom_fairness(F, predictor, config)
            except Exception as exc:                                  # noqa: BLE001
                logging.warning("symptom fairness unavailable: %s", exc)
            if len(sym_fair):
                save_report(config.report_path("symptom_fairness.csv"), sym_fair)
                logging.info("[SYNTHETIC LABELS] symptom subgroup parity: %d of %d slices "
                             "within margin", int(sym_fair.passes.sum()), len(sym_fair))

            # ---- 11. advisories, abstention, explainability -----------------------
            probe_id = panel.series_id.iloc[0]
            hist = panel[panel.series_id == probe_id].sort_values("ts")
            reloaded = BPPredictor.load(config.bundle_file_path)
            advisories = build_advisories(reloaded, hist, tier_clf, rule_clf, features, config)
            engine_today = alerts_pop[(alerts_pop.series_id == probe_id) & alerts_pop.fired]
            shown, arb_log = arbitrate(advisories, set(engine_today.tail(1).tier.dropna()),
                                       tau=config.confidence_tau)
            logging.info("advisories generated: %d -> shown after arbitration: %d | %s",
                         len(advisories), len(shown), arb_log)
            if advisories:
                save_report(config.report_path("advisories.csv"),
                            pd.DataFrame([a.to_row() for a in advisories]))

            # The trainer already reloaded the bundle; it never checked that the reload
            # scores the same. A pickle that loads but predicts differently is invisible
            # everywhere else in this pipeline.
            rt = bundle_round_trip(predictor, reloaded, hist)
            write_yaml_file(config.report_path("round_trip.yaml"), rt, replace=True)
            if rt["status"] != "PASS":
                raise AssertionError(f"bundle round-trip mismatch: {rt}")

            row_x = reloaded.fb.transform_for_inference(hist).reindex(columns=features)
            train_medians = F[F.split == "train"][features].median()
            fc_model = (list(reloaded.b["forecasters"].values())[0]
                        if reloaded.b["forecasters"] else None)
            # Perturb every feature, not an arbitrary first-N slice. The slice happened to
            # contain the lags an HGB leans on; a shipped EWMA baseline reads one column
            # (sbp_ewm0.3) that fell outside it, so every perturbation was a no-op and the
            # explanation came back all zeros -- which gate 6 correctly refused to promote.
            # explain_prediction already returns only the top N by absolute contribution.
            explanation = (explain_prediction(fc_model, row_x, features, train_medians)
                           if fc_model is not None else {})

            ood_cols = [c for c in features if F[c].notna().mean() > .8][:25]
            abstain = AbstentionPolicy(F[F.split == "train"], ood_cols)
            te_ood = abstain._dist(F[F.split == "test"][ood_cols])
            ood_rate = float(np.mean(te_ood > abstain.cut))

            cold = cold_start_curve(reloaded, hist, config)
            if len(cold):
                save_report(config.report_path("cold_start.csv"), cold)

            # ---- 12. synthetic cohort and rule coverage ---------------------------
            with timer("synthetic daily cohort"):
                synth = generate_synthetic_cohort()
            synth_run, syn_personal = prepare_synthetic_run(synth)
            with timer("extended engine, synthetic cohort"):
                syn_alerts = ExtendedRuleEngine(registry, syn_personal).run(
                    synth_run, use_personalisation=False)
            coverage = rule_coverage(registry, alerts_pop, syn_alerts)
            save_report(config.report_path("rule_coverage.csv"), coverage)
            save_report(config.report_path("synthetic_parameters.csv"), synth_param_table())
            logging.info("[SYNTHETIC] %d readings evaluated | %d alerts fired | "
                         "rule coverage %d/%d",
                         len(syn_alerts), int(syn_alerts.fired.sum()),
                         int(coverage.covered.sum()), len(coverage))

            prov = provenance_guard(panel, synth, alerts_pop, syn_alerts)
            write_yaml_file(config.report_path("provenance.yaml"), prov, replace=True)
            if not prov["holds"]:
                raise AssertionError("provenance guard failed: synthetic rows reached the "
                                     "real-data record")

            # ---- 12b. missingness robustness ---------------------------------------
            # The pipeline never imputes a skipped session, which is correct; this is the
            # evidence that the choice survives contact with users who skip them.
            miss = pd.DataFrame()
            try:
                with timer("missingness sweep"):
                    miss = missingness_sweep(panel, features, config, shipped,
                                             self.best_params)
                if len(miss):
                    save_report(config.report_path("missingness_robustness.csv"), miss)
                    logging.info("missingness sweep:\n%s", miss.to_string(index=False))
            except Exception as exc:
                logging.warning("missingness sweep unavailable: %s", exc)

            # ---- 13. safety gates --------------------------------------------------
            # Evidence produced elsewhere in this run is passed explicitly rather than
            # rediscovered inside the gate module: a gate that goes looking for its own
            # inputs is a gate that can quietly find nothing and pass.
            sel_sbp = winner.get("sbp")
            ship_sbp = (shipped.get("sbp") or ("learned", sel_sbp))[1]
            cold_pen = float(decision.cold_start_penalty.dropna().max()) \
                if len(decision) and "cold_start_penalty" in decision else None
            # The realised rate the detector achieves on TEST at the cut calibrated on
            # val. This, not the rule engine's fire rate, is what the alert budget binds.
            det_rate = np.nan
            if len(D_test):
                ts_ = np.asarray(det.score(D_test), float)
                ts_ = ts_[np.isfinite(ts_)]
                if len(ts_):
                    det_rate = float(np.mean(ts_ >= det_cut))
            gates = run_safety_gates(alerts_pop, alerts_pers, OFF, advisories, cold,
                                     fairness, explanation, abstain, ood_rate, config,
                                     missingness=miss,
                                     shipped_forecaster={
                                         k: (v[1] if isinstance(v, (tuple, list)) else v)
                                         for k, v in (shipped or {}).items()} or ship_sbp,
                                     selected_forecaster=dict(winner) or sel_sbp,
                                     shipped_detector=det_name,
                                     cold_start_penalty=cold_pen,
                                     detector_alert_rate=det_rate,
                                     offset_learned_max=offset_promo.get(
                                         "max_prediction"),
                                     symptom_labels_synthetic=predictor.b.get(
                                         "symptom_labels_synthetic"),
                                     pair_violation_rate=(
                                         float(pair_df.violation_rate.max())
                                         if len(pair_df) else None),
                                     forecaster_features=predictor.b.get("feature_names"),
                                     symptom_fairness=sym_fair)
            save_report(config.report_path("safety_gates.csv"), gates.frame())
            gate_artifact = SafetyGateArtifact(
                gate_report_file_path=config.report_path("safety_gates.csv"),
                gates_passed=int((gates.frame().status == "PASS").sum()),
                gates_total=int(len(gates.frame())),
                critical_failures=int(len(gates.critical_failures)),
                promotable=bool(gates.promotable))

            # ---- 13b. interpretability and serving parity ---------------------------
            # Block permutation, not single-column: `meta["clusters"]` records which
            # features collapsed into which representative during selection, and permuting
            # a representative while its twin is still in the matrix measures nothing.
            try:
                tgt0 = f"y_sbp_h{config.horizons[0]}"
                te_i = F[(F.split == "test") & F[tgt0].notna()]
                if len(te_i) > 200 and shipped.get("sbp", ("learned", None))[0] == "learned":
                    te_i = te_i.sample(min(config.max_eval_rows, len(te_i)),
                                       random_state=config.seed)
                    mdl_i = predictor.b["forecasters"].get(("sbp", config.horizons[0]))
                    clusters = meta.get("clusters") or {}
                    blocks = {}
                    for f in features:
                        k = clusters.get(f)
                        blocks[f] = ([c for c, kk in clusters.items() if kk == k and c in F]
                                     if k is not None else [f])
                    imp = block_importance(mdl_i, te_i[features],
                                           te_i[tgt0].to_numpy(float), blocks,
                                           seed=config.seed)
                    if len(imp):
                        save_report(config.report_path("block_importance.csv"), imp)
                    conc = error_concentration(te_i[tgt0], mdl_i.predict(te_i[features]),
                                               te_i.series_id)
                    if len(conc):
                        save_report(config.report_path("error_concentration.csv"), conc)
            except Exception as exc:                                  # noqa: BLE001
                logging.warning("interpretability block skipped: %s", exc)

            # The only check that catches train/serve feature drift: everything else scores
            # the offline path, this one drives predict() exactly as the API does.
            try:
                off_mae = float(R[(R.split == "test") & (R.signal == "sbp")
                                  & (R.model == winner.get("sbp"))].MAE.mean())
                par = walk_forward_parity(reloaded, panel, off_mae, seed=config.seed)
                write_yaml_file(config.report_path("serving_parity.yaml"), par, replace=True)
                lat = batch_latency(reloaded, panel, config, seed=config.seed)
                write_yaml_file(config.report_path("batch_latency.yaml"), lat, replace=True)
            except Exception as exc:                                  # noqa: BLE001
                logging.warning("parity / latency block skipped: %s", exc)

            # ---- 14. monitoring and champion/challenger ----------------------------
            self.run_monitoring(F, features, config, R, winner, det_report, shipped)
            chall = champion_challenger(F, features, config,
                                        winner.get("sbp", "ridge"), "hgb")
            if len(chall):
                save_report(config.report_path("champion_challenger.csv"), chall)

            # ---- 15. persist the model and track ------------------------------------
            # The run always keeps its own artifact: a blocked run still has to be
            # inspectable, and this path is inside Artifacts/<run>, which nothing serves.
            model_dir_path = os.path.dirname(config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            save_object(config.trained_model_file_path, predictor)

            # The push to final_model/ is the actual promotion: that file is what the API
            # loads and what the deploy workflow ships. It is gated on `promotable`, which
            # until now was computed, logged as "PROMOTION BLOCKED", and then ignored --
            # the push happened above the check, so a model that failed a critical safety
            # gate reached production anyway and the previous good model was overwritten.
            final_path = os.path.join(config.final_model_dir, "model.pkl")
            if gates.promotable:
                save_object(final_path, predictor)
                logging.info("promoted to %s (%d/%d gates pass, 0 critical failures)",
                             final_path, gate_artifact.gates_passed,
                             gate_artifact.gates_total)
            else:
                logging.error("PROMOTION BLOCKED: %d critical safety-gate failures -- %s "
                              "left untouched, the run's own artifact is at %s",
                              gate_artifact.critical_failures, final_path,
                              config.trained_model_file_path)

            test_rows = R[(R.split == "test") & (R.family == "learned")]
            val_rows = R[(R.split == "val") & (R.family == "learned")]
            train_metrics = [get_forecast_score(r) for r in val_rows.to_dict("records")]
            test_metrics = [get_forecast_score(r) for r in test_rows.to_dict("records")]

            headline = {}
            sbp_test = test_rows[test_rows.signal == "sbp"]
            if len(sbp_test):
                headline = dict(sbp_test_mae=float(sbp_test.MAE.mean()),
                                sbp_test_rmse=float(sbp_test.RMSE.mean()),
                                sbp_test_r2=float(sbp_test.R2.mean()))
            if detector_artifacts:
                headline["detector_auc_pr"] = detector_artifacts[0].auc_pr
                headline["detector_lead_days"] = detector_artifacts[0].lead_days_median
            if offset_artifact:
                headline["offset_mae"] = offset_artifact.mae
            headline["gates_passed"] = gate_artifact.gates_passed
            headline["critical_gate_failures"] = gate_artifact.critical_failures
            headline["promotable"] = int(gates.promotable)

            # One parent run per pipeline run, with a child run for every model fitted.
            # The parent carries the headline and the governance outcome; the children make
            # the selection auditable -- previously only the winner was logged, so the
            # tracking store could not show what it had won against.
            try:
                self._mlflow_ready()
                with mlflow.start_run(run_name=f"hemobp-bp-{config.run_id}"):
                    mlflow.set_tag("stage", "hemobp-bp")
                    mlflow.set_tag("run_id", config.run_id)
                    mlflow.set_tag("promotable", str(gates.promotable).lower())
                    for s, (fam, nm) in shipped.items():
                        mlflow.set_tag(f"shipped_{s}", f"{fam}:{nm}")
                    mlflow.set_tag("shipped_detector", det_name)
                    for k, v in self._clean_metrics(headline).items():
                        mlflow.log_metric(k, v)
                    self.track_model_catalogue(
                        config, R, shipped, off_report, offset_search, offset_model,
                        det_report, det_report_val, det_name, tier_metrics_df, predictor)
            except Exception as exc:
                logging.warning("mlflow tracking skipped: %s", exc)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=config.trained_model_file_path,
                bundle_file_path=config.bundle_file_path,
                report_dir=config.report_dir,
                selected_family=winner,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
                detector_metric_artifact=detector_artifacts,
                offset_metric_artifact=offset_artifact,
                rule_engine_artifact=rule_artifact,
                safety_gate_artifact=gate_artifact,
            )
            logging.info("Model trainer artifact: %s", model_trainer_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ fairness
    def symptom_fairness(self, F, predictor, config) -> pd.DataFrame:
        """PR-AUC of each symptom head by demographic slice, on the test split.

        Separate from `run_fairness` on purpose -- see the call site. `slice_gate` is reused
        unchanged so the margin semantics match the forecaster's, but the metric is PR-AUC
        rather than mmHg, and a failing row here is a finding about `symptom_layer.py`, not
        about a patient.
        """
        from sklearn.metrics import average_precision_score
        models = predictor.b.get("symptom_models") or {}
        te = F[F.split == "test"] if "split" in F else F
        if not models or not len(te):
            return pd.DataFrame()
        X = te.reindex(columns=predictor.b["feature_names"])
        rows = []
        slices = [("sex", te.is_male.map({1: "male", 0: "female"}))] if "is_male" in te else []
        if "age" in te:
            slices.append(("age", pd.cut(te.age, [0, 55, 70, 200],
                                         labels=["<55", "55-70", "70+"]).astype(str)))
        for key in sorted(k for k in models if k.endswith("_h0")):
            base = key.rsplit("_h", 1)[0]
            tgt = f"y_sym_{base}_h0"
            if tgt not in te.columns:
                continue
            try:
                p = np.asarray(models[key].predict_proba(X)[:, 1], dtype=float)
            except Exception:                                         # noqa: BLE001
                continue
            for axis, lab in slices:
                for level in sorted(set(lab.dropna())):
                    m = (lab == level).to_numpy() & te[tgt].notna().to_numpy() & np.isfinite(p)
                    y = te[tgt].to_numpy(dtype=float)[m]
                    if m.sum() < 100 or y.sum() < 5:
                        continue
                    rows.append(dict(symptom=base, axis=axis, level=level, n=int(m.sum()),
                                     base_rate=round(float(y.mean()), 5),
                                     pr_auc=round(float(average_precision_score(y, p[m])), 4),
                                     labels="SYNTHETIC"))
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        # Within-symptom, within-axis spread against the same margin the forecaster uses.
        out["spread"] = out.groupby(["symptom", "axis"]).pr_auc.transform(
            lambda v: float(v.max() - v.min()))
        out["passes"] = out.spread <= float(getattr(config, "fair_margin_pp", 0.15))
        return out

    def run_fairness(self, F, features, config, winner, M_hold, D_test, det_report,
                     shipped=None):
        """Subgroup performance for the forecaster, the offset and the best detector.

        Audits whatever SHIPS. Auditing a learned model that lost the ship decision
        would clear a subgroup gate for something no patient is ever scored by.
        """
        try:
            frames = []
            target = f"y_sbp_h{config.horizons[0]}"
            df_f = F[F[target].notna()]
            tr_f = df_f[df_f.split == "train"]
            if len(tr_f) > 200:
                tr_f = tr_f.sample(min(config.max_train_rows, len(tr_f)),
                                   random_state=config.seed)
                family, kind = (shipped or {}).get("sbp", ("learned",
                                                           winner.get("sbp", "hgb")))
                if family == "baseline":
                    m = BaselineForecaster(kind, "sbp", config.horizons[0])
                else:
                    m = make_model(kind, **self.best_params.get((kind, "sbp"), {})).fit(
                        tr_f[features], tr_f[target].values)
                te_f = df_f[df_f.split == "test"].copy()
                if len(te_f) > 40:
                    te_f["pred"] = m.predict(te_f[features])
                    # The reference is the EWMA baseline scored on the SAME rows, so each
                    # subgroup is judged on how much the model improves on what is achievable
                    # for it rather than on the raw difficulty of its data. Measured here:
                    # the baseline alone is 2.50 mmHg worse for under-50s because their
                    # systolic variance is 24% higher, and the raw-MAE comparison failed the
                    # model for that.
                    te_f["_bl"] = np.asarray(
                        forecast_baselines(te_f, "sbp", config.horizons[0])["ewma"],
                        dtype=float)
                    frames.append(slice_gate(
                        te_f, ["is_male", "age_band", "is_dm"],
                        lambda x: mean_absolute_error(x[target], x["pred"]),
                        config.fair_margin_mmHg, "forecaster MAE (mmHg)",
                        reference=lambda x: float(np.nanmean(np.abs(x["_bl"] - x[target]))),
                        higher_is_better=False))

            if len(M_hold) >= 30:
                M_fair = M_hold.assign(age_band=pd.cut(
                    M_hold.age, [0, 50, 65, 75, 200],
                    labels=["<50", "50-64", "65-74", "75+"], right=False))
                frames.append(slice_gate(
                    M_fair, ["is_male", "age_band", "is_dm"],
                    lambda x: mean_absolute_error(x["actual"], x["threshold"]),
                    config.fair_margin_mmHg, "offset MAE (mmHg)"))

            if len(det_report) and len(D_test):
                learned = det_report[(det_report.budget_pct == config.alert_budget_pct)
                                     & (det_report.detector != "d_fixed_threshold")]
                if len(learned):
                    best_col = learned.sort_values("auc_pr").iloc[-1].detector
                    cut = float(np.percentile(D_test[best_col].dropna(),
                                              100 - config.alert_budget_pct))
                    dt = D_test.copy()
                    dt["flag"] = (dt[best_col] >= cut).astype(int)
                    dt["age_band"] = pd.cut(dt.age, [0, 50, 65, 75, 200],
                                            labels=["<50", "50-64", "65-74", "75+"],
                                            right=False)
                    # RECALL, not precision. At a single global score cut, precision differs
                    # across subgroups by Bayes alone: the same likelihood ratio against a
                    # different prior gives a different posterior. Measured on a controlled
                    # case, precision spread 0.464 between a 2% and a 10% base-rate group
                    # while the detector was equally good for both. Normalising by the base
                    # rate does not fix it either, because the achievable ceiling scales too.
                    #
                    # Recall does not have that problem -- P(flagged | event) does not depend
                    # on the prior -- and it is the better equity question for an alerting
                    # system regardless: of the patients in this group who deteriorate, what
                    # fraction do we actually alert on? Same case, recall spread 0.077.
                    #
                    # Precision is still reported per slice as a diagnostic; only the
                    # pass/fail basis changes.
                    frames.append(slice_gate(
                        dt, ["is_male", "age_band", "is_dm"],
                        lambda x: (float(x[x.event_next == 1].flag.mean())
                                   if (x.event_next == 1).any() else np.nan),
                        FAIR_MARGIN_PP,
                        f"{best_col} recall @{config.alert_budget_pct}%",
                        higher_is_better=True))
                    frames.append(slice_gate(
                        dt, ["is_male", "age_band", "is_dm"],
                        lambda x: (float(x[x.flag == 1].event_next.mean())
                                   if (x.flag == 1).any() else np.nan),
                        # Reported only: a margin of 1.0 cannot fail, because precision is not
                        # comparable across subgroups and must not gate promotion.
                        1.0,
                        f"{best_col} precision @{config.alert_budget_pct}% (reported only)",
                        higher_is_better=True))

            fairness = pd.concat([f for f in frames if len(f)], ignore_index=True) \
                if any(len(f) for f in frames) else pd.DataFrame()
            if len(fairness):
                fail = fairness[~fairness.passes]
                logging.info("slice gate: %d/%d pass", len(fairness) - len(fail), len(fairness))
            logging.info("Not auditable on this corpus: race / ethnicity (absent from HEMOBP).")
            return fairness
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ monitoring
    def run_monitoring(self, F, features, config, R, winner, det_report, shipped=None):
        """PSI on features plus error and alert-rate drift, with the response ladder.

        Tracks the model that ships, so the val->test error drift is measured on the
        thing in production rather than on a learned model that lost the ship decision.
        """
        try:
            ref, cur = F[F.split == "train"], F[F.split == "test"]
            mon = DriftMonitor(ref, features)
            psi = mon.feature_drift(cur)
            if len(psi):
                save_report(config.report_path("feature_drift_psi.csv"), psi)

            sel = (shipped or {}).get("sbp", ("learned", winner.get("sbp")))[1]
            ref_mae = float(R[(R.split == "val") & (R.signal == "sbp")
                              & (R.model == sel)].MAE.mean())
            cur_mae = float(R[(R.split == "test") & (R.signal == "sbp")
                              & (R.model == sel)].MAE.mean())
            observed_rate = config.alert_budget_pct / 100.0
            if len(det_report):
                at_budget = det_report[det_report.budget_pct == config.alert_budget_pct]
                if len(at_budget):
                    observed_rate = float(at_budget.iloc[0].budget_pct) / 100.0
            rows = [mon.error_drift(ref_mae, cur_mae),
                    mon.alert_rate_drift(observed_rate, config.alert_budget_pct / 100.0)]
            if "days_since_last_raw" in F.columns:
                rows.append(mon.cadence_drift(ref.days_since_last_raw,
                                              cur.days_since_last_raw))
            signals = pd.DataFrame(rows)
            save_report(config.report_path("drift_signals.csv"), signals)

            n_alarm = int((psi.status == "ALARM").sum() if len(psi) else 0) \
                + int((signals.status == "ALARM").sum())
            logging.info("drift alarms: %d | %s", n_alarm,
                         mon.response_ladder("ALARM" if n_alarm else "ok"))
            logging.info("Train->test PSI is a within-cohort temporal shift, not deployment "
                         "drift: a smoke test of the monitor and a floor on what to expect.")
        except Exception as e:
            raise CustomException(e, sys)
