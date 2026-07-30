import os
import sys

import pandas as pd

from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.exception.custom_exception import CustomException
from src.logging.logger import logging, timer
from src.utils.main_utils.utils import (
    save_dataframe,
    save_object,
    save_report,
    summarise,
    write_yaml_file,
)
from src.utils.ml_utils.feature.cadence import cadence_audit
from src.utils.ml_utils.feature.causal_features import (
    CausalFeatureBuilder,
    add_splits,
    build_panel,
    feature_dictionary,
    leakage_audit,
)
from src.utils.ml_utils.feature.selection import select_features
from src.utils.ml_utils.rule_engine.symptom_layer import symptom_param_table, synth_session_layer


class DataTransformation:
    """Panel construction, splits and strictly-causal feature engineering.

    There is no fitted preprocessor here by design: imputation lives inside `make_model`
    (median SimpleImputer for the linear pipelines; HistGradientBoosting handles NaN
    natively), so exactly the same transform runs at training and at serving time.
    """

    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df.ts, errors="coerce")
            if "patient_id" in df.columns:
                df["patient_id"] = df.patient_id.astype(str)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self) -> CausalFeatureBuilder:
        """The causal feature builder is the transformer for this pipeline."""
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            return CausalFeatureBuilder(self.data_transformation_config)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            cfg = self.data_transformation_config
            # Train and test were split temporally at ingest; features must be built over the
            # whole timeline per patient, then labelled by split. Building them separately
            # would leave the first rows of the test tail with no history to look back at.
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path)
            sessions = (pd.concat([train_df, test_df], ignore_index=True)
                        .drop(columns=["step"], errors="ignore")
                        .sort_values(["patient_id", "ts"]))

            static_cols = ["patient_id", "gender", "age", "DM", "first_dialysis_ts"]
            static = (sessions[[c for c in static_cols if c in sessions.columns]]
                      .drop_duplicates("patient_id"))
            sessions = sessions.drop(
                columns=[c for c in ("gender", "age", "DM", "first_dialysis_ts")
                         if c in sessions.columns])

            with timer("panel construction"):
                panel = build_panel(sessions, static, cfg)
                panel = add_splits(panel, cfg)
            # Model 4's label source. SYNTHETIC -- HEMOBP records no symptoms, medications
            # or conditions. Merged onto the real panel by (series_id, step) so each
            # generated label is conditioned on that patient's actual blood pressure that
            # day. See rule_engine/symptom_layer.py for why this exists and what it is not.
            if getattr(cfg, "symptom_layer", True):
                with timer("synthetic symptom layer"):
                    S, _rates = synth_session_layer(panel, seed=cfg.seed)
                panel["series_id"] = panel.series_id.astype(str)
                panel = panel.merge(S, on=["series_id", "step"], how="left")
                save_report(cfg.report_path("symptom_parameters.csv")
                            if hasattr(cfg, "report_path")
                            else os.path.join(os.path.dirname(cfg.leakage_report_file_path),
                                              "symptom_parameters.csv"),
                            symptom_param_table())
            summarise(panel, "panel")

            fb = self.get_data_transformer_object()
            with timer("feature construction"):
                F = fb.transform(panel)
            features = fb.feature_names_
            summarise(F, "features")
            logging.info("%d features across %d patients", len(features), F.series_id.nunique())

            # The causal contract is the single most consequential thing in this pipeline,
            # so it is tested rather than assumed.
            audit = leakage_audit(F, features, cfg, fb=fb)
            save_report(cfg.leakage_report_file_path, audit)
            failed = audit[audit.status == "FAIL"]
            leakage_clean = failed.empty
            if not leakage_clean:
                raise AssertionError("leakage audit failed:\n"
                                     + failed.to_string(index=False))
            warned = audit[audit.status == "WARN"]
            if len(warned):
                logging.warning("leakage audit: %d probe(s) warned\n%s",
                                len(warned), warned.to_string(index=False))
            logging.info("leakage audit clean: %d probes passed", int((audit.status == "PASS").sum()))

            # Cadence probes sit beside the leakage ones for the same reason: the gap
            # column is reconstructed in six places and silently becoming a model feature
            # would change every estimator's input matrix without any other signal.
            cad = cadence_audit(panel, features)
            save_report(cfg.report_path("cadence_audit.csv")
                        if hasattr(cfg, "report_path") else
                        os.path.join(os.path.dirname(cfg.leakage_report_file_path),
                                     "cadence_audit.csv"), cad)
            cad_failed = cad[cad.status == "FAIL"]
            if not cad_failed.empty:
                raise AssertionError("cadence audit failed:\n"
                                     + cad_failed.to_string(index=False))
            logging.info("cadence audit clean: %d probes passed | %s",
                         int((cad.status == "PASS").sum()),
                         cad.loc[cad.probe.str.startswith("the clip"), "detail"].iloc[0]
                         if (cad.probe.str.startswith("the clip")).any() else "")

            # Selection runs AFTER the audits, deliberately. It reads the targets, so it can
            # only be trusted once the causal contract has been shown to hold; and it is
            # fitted on train rows only, so it cannot see the split it will be judged on.
            all_features = list(features)
            cluster_of = {}
            if getattr(cfg, "feature_selection", True):
                with timer("feature selection"):
                    features, sel_audit, cluster_of = select_features(F, all_features, cfg,
                                                                      seed=cfg.seed)
                save_report(cfg.feature_selection_report_file_path, sel_audit)
            else:
                logging.info("feature selection disabled; shipping all %d features",
                             len(all_features))

            save_dataframe(cfg.panel_file_path, panel)
            save_dataframe(cfg.feature_file_path, F)
            save_dataframe(cfg.transformed_train_file_path, F[F.split.isin(["train", "val"])])
            save_dataframe(cfg.transformed_test_file_path, F[F.split == "test"])
            save_report(cfg.feature_dictionary_file_path, feature_dictionary(F, features))
            write_yaml_file(cfg.feature_list_file_path,
                            {"features": list(features),
                             "all_features": all_features,
                             # Cluster ids let the trainer permute whole redundancy blocks
                             # instead of single columns, which is the difference between
                             # measuring a feature's importance and measuring whether it has
                             # a surviving twin.
                             "clusters": {k: int(v) for k, v in cluster_of.items()},
                             "horizons": list(cfg.horizons),
                             "signals": list(cfg.signals)},
                            replace=True)
            # The feature builder holds no fitted state, but the pipeline contract expects an
            # object at this path and the serving bundle reconstructs one from the config.
            save_object(cfg.transformed_object_file_path, fb)

            return DataTransformationArtifact(
                transformed_object_file_path=cfg.transformed_object_file_path,
                transformed_train_file_path=cfg.transformed_train_file_path,
                transformed_test_file_path=cfg.transformed_test_file_path,
                panel_file_path=cfg.panel_file_path,
                feature_file_path=cfg.feature_file_path,
                feature_list_file_path=cfg.feature_list_file_path,
                leakage_report_file_path=cfg.leakage_report_file_path,
                feature_selection_report_file_path=cfg.feature_selection_report_file_path,
                n_features=len(features),
                n_features_built=len(all_features),
                n_patients=int(F.series_id.nunique()),
                leakage_clean=bool(leakage_clean),
            )
        except Exception as e:
            raise CustomException(e, sys)
