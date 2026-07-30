import os
from datetime import datetime

from src.constants import training_pipeline


class TrainingPipelineConfig:
    def __init__(self, timestamp=None):
        # Resolved inside the body: a `datetime.now()` default is evaluated once at import,
        # so every run in a long-lived process would otherwise reuse the first timestamp.
        timestamp = timestamp or datetime.now()
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.model_dir = os.path.join("final_model")
        self.timestamp: str = timestamp
        self.run_id: str = timestamp
        self.seed: int = training_pipeline.SEED


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME,
        )
        self.static_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.DATA_INGESTION_STATIC_FILE_NAME,
        )
        self.reconciliation_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.DATA_INGESTION_RECON_FILE_NAME,
        )
        self.decomposition_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.DATA_INGESTION_DECOMPOSITION_FILE_NAME,
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME,
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME,
        )

        # Raw HEMOBP corpus, read from the local repository rather than downloaded.
        self.raw_data_dir: str = training_pipeline.RAW_DATA_DIR
        self.vip_file_path: str = os.path.join(
            self.raw_data_dir, training_pipeline.VIP_FILE_NAME
        )
        self.d1_file_path: str = os.path.join(
            self.raw_data_dir, training_pipeline.D1_FILE_NAME
        )
        self.idp_file_path: str = os.path.join(
            self.raw_data_dir, training_pipeline.IDP_FILE_NAME
        )
        self.chunksize: int = training_pipeline.INGEST_CHUNKSIZE
        self.ingest_ranges: dict = training_pipeline.INGEST_RANGES
        self.age_reference_year: int = training_pipeline.AGE_REFERENCE_YEAR
        self.paper_counts: dict = training_pipeline.PAPER_COUNTS

        # Sessionisation: a session is a contiguous run of readings, dated by the
        # dialysis day rather than the calendar date.
        self.session_gap_hours: float = training_pipeline.SESSION_GAP_HOURS
        self.dialysis_day_start_hour: int = training_pipeline.DIALYSIS_DAY_START_HOUR
        self.session_recon_band: tuple = training_pipeline.SESSION_RECON_BAND
        self.max_sessions_without_d1: float = training_pipeline.MAX_SESSIONS_WITHOUT_D1

        # Cohort admission and splits
        self.min_sessions: int = training_pipeline.MIN_SESSIONS
        self.min_span_days: int = training_pipeline.MIN_SPAN_DAYS
        self.max_patients = training_pipeline.MAX_PATIENTS
        self.val_frac: float = training_pipeline.VAL_FRAC
        self.test_frac: float = training_pipeline.TEST_FRAC
        self.holdout_patient_frac: float = training_pipeline.HOLDOUT_PATIENT_FRAC
        self.patient_split_salt: str = training_pipeline.PATIENT_SPLIT_SALT


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline.TEST_FILE_NAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline.TEST_FILE_NAME
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )
        self.contract_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_CONTRACT_REPORT_FILE_NAME,
        )
        self.drift_threshold: float = training_pipeline.DATA_VALIDATION_DRIFT_THRESHOLD
        self.max_out_of_range: float = training_pipeline.MAX_OUT_OF_RANGE
        self.min_sessions: int = training_pipeline.MIN_SESSIONS


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME,
        )
        transformed = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        )
        self.transformed_train_file_path: str = os.path.join(
            transformed, training_pipeline.TRAIN_FILE_NAME.replace("csv", "parquet")
        )
        self.transformed_test_file_path: str = os.path.join(
            transformed, training_pipeline.TEST_FILE_NAME.replace("csv", "parquet")
        )
        self.panel_file_path: str = os.path.join(
            transformed, training_pipeline.DATA_TRANSFORMATION_PANEL_FILE_NAME
        )
        self.feature_file_path: str = os.path.join(
            transformed, training_pipeline.DATA_TRANSFORMATION_FEATURE_FILE_NAME
        )
        self.feature_list_file_path: str = os.path.join(
            transformed, training_pipeline.DATA_TRANSFORMATION_FEATURE_LIST_FILE_NAME
        )
        self.feature_dictionary_file_path: str = os.path.join(
            transformed, training_pipeline.DATA_TRANSFORMATION_FEATURE_DICT_FILE_NAME
        )
        self.leakage_report_file_path: str = os.path.join(
            transformed, training_pipeline.DATA_TRANSFORMATION_LEAKAGE_REPORT_FILE_NAME
        )
        self.feature_selection_report_file_path: str = os.path.join(
            transformed, training_pipeline.DATA_TRANSFORMATION_SELECTION_FILE_NAME
        )
        self.feature_selection: bool = training_pipeline.FEATURE_SELECTION
        self.symptom_layer: bool = training_pipeline.SYMPTOM_LAYER
        self.seed: int = training_pipeline.SEED
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,
        )

        self.horizons: tuple = training_pipeline.HORIZONS
        self.lags: tuple = training_pipeline.LAGS
        self.windows: tuple = training_pipeline.WINDOWS
        self.signals: tuple = training_pipeline.SIGNALS
        self.ewm_alphas: tuple = training_pipeline.EWM_ALPHAS
        self.min_sessions: int = training_pipeline.MIN_SESSIONS
        self.min_span_days: int = training_pipeline.MIN_SPAN_DAYS
        self.max_patients = training_pipeline.MAX_PATIENTS
        self.val_frac: float = training_pipeline.VAL_FRAC
        self.test_frac: float = training_pipeline.TEST_FRAC
        self.holdout_patient_frac: float = training_pipeline.HOLDOUT_PATIENT_FRAC
        self.patient_split_salt: str = training_pipeline.PATIENT_SPLIT_SALT


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_FILE_NAME,
        )
        self.bundle_file_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_TRAINER_BUNDLE_FILE_NAME,
        )
        self.report_dir: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_REPORT_DIR
        )
        self.final_model_dir: str = training_pipeline_config.model_dir
        self.run_id: str = training_pipeline_config.run_id
        self.seed: int = training_pipeline.SEED


        # The serving bundle rebuilds a CausalFeatureBuilder from this config, so it must
        # carry the full feature specification, not just the modelling knobs.
        self.horizons: tuple = training_pipeline.HORIZONS
        self.signals: tuple = training_pipeline.SIGNALS
        self.lags: tuple = training_pipeline.LAGS
        self.windows: tuple = training_pipeline.WINDOWS
        self.ewm_alphas: tuple = training_pipeline.EWM_ALPHAS
        self.max_train_rows: int = training_pipeline.MAX_TRAIN_ROWS
        self.max_eval_rows: int = training_pipeline.MAX_EVAL_ROWS
        self.tune: bool = training_pipeline.TUNE
        self.tuner: str = training_pipeline.TUNER
        self.tune_draws: int = training_pipeline.TUNE_DRAWS
        self.tune_trials: int = training_pipeline.TUNE_TRIALS
        self.tune_sample_rows: int = training_pipeline.TUNE_SAMPLE_ROWS
        self.tune_folds: int = training_pipeline.TUNE_FOLDS
        self.tpe_startup_trials: int = training_pipeline.TPE_STARTUP_TRIALS
        self.pruner_startup_trials: int = training_pipeline.PRUNER_STARTUP_TRIALS
        self.pruner_warmup_steps: int = training_pipeline.PRUNER_WARMUP_STEPS
        self.n_boot: int = training_pipeline.N_BOOT

        # Architecture registry scope
        self.model_tier: str = training_pipeline.MODEL_TIER
        self.fast_mode: bool = training_pipeline.FAST_MODE
        self.run_local_scope: bool = training_pipeline.RUN_LOCAL_SCOPE

        # Serving
        self.latency_budget_ms: float = training_pipeline.LATENCY_BUDGET_MS
        self.cold_start_min_readings: int = training_pipeline.COLD_START_MIN_READINGS
        self.steady_state_readings: int = training_pipeline.STEADY_STATE_READINGS

        # Governance -- fixed inputs, excluded from every search.
        self.population_threshold_mmHg: float = training_pipeline.POPULATION_THRESHOLD_MMHG
        self.emergency_floor_mmHg: float = training_pipeline.EMERGENCY_FLOOR_MMHG
        self.offset_cap_loosen: float = training_pipeline.OFFSET_CAP_LOOSEN
        self.offset_cap_tighten: float = training_pipeline.OFFSET_CAP_TIGHTEN
        self.alert_budget_pct: float = training_pipeline.ALERT_BUDGET_PCT
        self.warn_window: int = training_pipeline.WARN_WINDOW
        self.event_quantile: float = training_pipeline.EVENT_QUANTILE

        self.offset_warm: int = training_pipeline.OFFSET_WARM
        self.offset_k: float = training_pipeline.OFFSET_K
        self.offset_q: float = training_pipeline.OFFSET_Q
        self.offset_search_warm: tuple = training_pipeline.OFFSET_SEARCH_WARM
        self.offset_search_k: tuple = training_pipeline.OFFSET_SEARCH_K
        self.offset_search_q: tuple = training_pipeline.OFFSET_SEARCH_Q

        self.offset_target_q: float = training_pipeline.OFFSET_TARGET_Q
        self.offset_future: int = training_pipeline.OFFSET_FUTURE
        self.offset_cuts: tuple = training_pipeline.OFFSET_CUTS
        self.learned_offset: bool = training_pipeline.LEARNED_OFFSET

        self.confidence_tau: float = training_pipeline.CONFIDENCE_TAU
        self.fair_margin_mmHg: float = training_pipeline.FAIR_MARGIN_MMHG
        self.stale_forecast_max_days: int = training_pipeline.STALE_FORECAST_MAX_DAYS

        # Safety-gate tolerances. Read only by src/utils/ml_utils/safety/.
        self.ood_rate_max: float = training_pipeline.OOD_RATE_MAX
        self.pair_violation_max: float = training_pipeline.PAIR_VIOLATION_MAX
        self.cold_start_probe_n: tuple = training_pipeline.COLD_START_PROBE_N
        self.cold_start_penalty_max_mmHg: float = (
            training_pipeline.COLD_START_PENALTY_MAX_MMHG
        )
        self.missingness_fracs: tuple = training_pipeline.MISSINGNESS_FRACS
        self.missingness_max_degradation_mmHg: float = (
            training_pipeline.MISSINGNESS_MAX_DEGRADATION_MMHG
        )

    def report_path(self, name: str) -> str:
        """Path for a named scorecard inside the model-trainer report directory."""
        return os.path.join(self.report_dir, name)
