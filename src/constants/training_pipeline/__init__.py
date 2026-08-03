import os

import yaml

"""
Defining common constant variable for training pipeline.

Values mirror the reference notebook (notebooks/Experiments.ipynb). Anything under
GOVERNANCE is a clinician / ops input: it is never searched, never tuned, and never derived
from the data.

The session contract -- columns, ranges, target signals and horizons -- lives in
data_schema/schema.yaml and is LOADED here rather than restated. The duplicated copies that
used to sit in this file were never read from the YAML, so the two drifted: schema.yaml
claimed horizons [1, 2, 3] while the code used (0, 1, 2). Loading and asserting means that
class of disagreement fails at import instead of at scoring time.
"""

PIPELINE_NAME: str = "CardioplaceBPAlerts"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "sessions.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

# Resolved against this file's location, not the process working directory: the API, the
# training subprocess and pytest are all started from different places.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
with open(os.path.join(_REPO_ROOT, SCHEMA_FILE_PATH), encoding="utf-8") as _fh:
    SCHEMA: dict = yaml.safe_load(_fh)

SCHEMA_COLUMNS: list = [next(iter(d)) for d in SCHEMA["columns"]]
SCHEMA_DTYPES: dict = {k: v for d in SCHEMA["columns"] for k, v in d.items()}
NUMERICAL_COLUMNS: list = list(SCHEMA["numerical_columns"])
SCHEMA_RANGES: dict = {k: (tuple(v) if isinstance(v, list) else v)
                       for k, v in SCHEMA["ranges"].items()}
MAX_OUT_OF_RANGE: float = float(SCHEMA["max_out_of_range"])

MODEL_FILE_NAME = "model.pkl"

SEED: int = 42

"""
Raw corpus (HEMOBP). The three CSVs ship with the repository; vip.csv is tracked with Git LFS.
"""

RAW_DATA_DIR: str = "data"
VIP_FILE_NAME: str = "vip.csv"
D1_FILE_NAME: str = "d1.csv"
IDP_FILE_NAME: str = "idp.csv"

# vip.csv is ~250 MB / ~4.4M readings; it is streamed rather than loaded whole.
INGEST_CHUNKSIZE: int = 1_000_000

# Physiological admission ranges, applied at ingest. Read from schema.yaml so the ingest
# filter and the validation contract cannot disagree about what is admissible.
INGEST_RANGES: dict = {"sbp": SCHEMA_RANGES["sbp"], "dbp": SCHEMA_RANGES["dbp"],
                       "min_pulse_pressure": SCHEMA_RANGES["min_pulse_pressure"]}

# Published figures from the HEMOBP data descriptor, used for the ingest reconciliation.
PAPER_COUNTS: dict = {"bp_recordings": 4_366_298, "sessions": 165_986, "patients": 1_075}

"""
SESSIONISATION. A dialysis session is a contiguous run of readings, not a calendar date.

Keying sessions on (patient, calendar date) split every session that ran past midnight into
two, which inflated the derived session count by 43.8% and -- far worse -- left 53.6% of rows
with no d1 match, so `weight`, `dryweight` and `idwg` were silently NaN on half the corpus.

Measured on the corpus: within-patient gaps between consecutive readings are strongly
bimodal, p95 = 1.0h against p97 = 44h, with only 0.003% of gaps landing between 4h and 12h.
The threshold is therefore insensitive by construction -- anything from 3h to 12h yields the
same 197,260 sessions. Median session duration comes out at 3.98h, which is a 4-hour
hemodialysis session.
"""
SESSION_GAP_HOURS: float = 6.0

# The dialysis day starts at 08:00, not midnight. This is d1's own convention, not a fitted
# parameter: every `keyindate` in d1.csv is stamped exactly 08:00:00, and the corpus runs
# three shifts a day, the last starting ~19:00-01:00 and therefore crossing into the next
# calendar date. Dating a session by (start - 8h) lifts the d1 match from 66.8% to 98.3%
# and drops Sunday sessions from 16,959 to 22 (d1 records 2).
DIALYSIS_DAY_START_HOUR: int = 8

# Derived-to-published session ratio the run will accept. Centred on 1.19, not 1.00: after
# the sessionisation fix, 98.3% of d1's sessions are matched and the residual is vip holding
# BP readings for ~31k dialysis days that d1 never recorded. The published 165,986 is d1's
# row count, so it is a floor on the true session count, not the true count. The band exists
# to catch a regression in the keying, not to force agreement with the wrong denominator.
SESSION_RECON_BAND: tuple = (1.10, 1.30)

# Share of sessions allowed to carry no d1 record (and therefore no weight/dryweight/idwg).
MAX_SESSIONS_WITHOUT_D1: float = 0.25

# Reference year used to turn birth year into age, per the notebook.
AGE_REFERENCE_YEAR: int = 2016

# Header names that would indicate a pulse column; HEMOBP's published schema has none.
PULSE_CANDIDATES: tuple = ("pulse", "hr", "heart_rate", "heartrate", "pulse_rate", "pr", "bpm")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_STATIC_FILE_NAME: str = "static.csv"
DATA_INGESTION_RECON_FILE_NAME: str = "ingest_reconciliation.csv"
DATA_INGESTION_DECOMPOSITION_FILE_NAME: str = "session_decomposition.csv"

# Cohort admission
MIN_SESSIONS: int = 60
MIN_SPAN_DAYS: int = 120
MAX_PATIENTS = 150  # runtime cap; set to None to use every eligible patient

# Splits: temporal tail per patient, plus a hash-based patient-level holdout.
VAL_FRAC: float = 0.20
TEST_FRAC: float = 0.20
HOLDOUT_PATIENT_FRAC: float = 0.30
PATIENT_SPLIT_SALT: str = "patient-split"

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_CONTRACT_REPORT_FILE_NAME: str = "data_contract.csv"
DATA_VALIDATION_DRIFT_THRESHOLD: float = 0.05
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

# MAX_OUT_OF_RANGE is defined at the top of this file, loaded from schema.yaml. It used to be
# restated here as a second literal, which is the shape of bug this refactor removes.

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
DATA_TRANSFORMATION_PANEL_FILE_NAME: str = "panel.parquet"
DATA_TRANSFORMATION_FEATURE_FILE_NAME: str = "features.parquet"
DATA_TRANSFORMATION_FEATURE_LIST_FILE_NAME: str = "feature_names.yaml"
DATA_TRANSFORMATION_FEATURE_DICT_FILE_NAME: str = "feature_dictionary.csv"
DATA_TRANSFORMATION_LEAKAGE_REPORT_FILE_NAME: str = "leakage_audit.csv"
DATA_TRANSFORMATION_SELECTION_FILE_NAME: str = "feature_selection.csv"

# Collapse redundant features to one representative per correlation cluster. Off means the
# estimators see every column the builder emits, which is a valid ablation but a worse model:
# on this corpus most columns are the same signal at a different window.
FEATURE_SELECTION: bool = True

# Generate Model 4's synthetic symptom/clinical overlay. Off means no symptom heads and no
# clinical-history features -- a valid, more conservative configuration.
SYMPTOM_LAYER: bool = True

# Causal feature construction.
#
# `h` is the shift applied to the target: y_<signal>_h<h>[t] = signal[t+h], paired with
# features at t, which read only <= t-1. So the last reading the model sees is t-1 and the
# reading it predicts is t+h -- h+1 readings later, not h. At serving the placeholder row
# occupies slot t, which is the patient's NEXT reading, so h=0 is that next reading.
#
# This used to be (1, 2, 3) with everything labelled "h sessions ahead", which put every
# forecast one reading further out than advertised: h1 was presented as ~2 days when it was
# ~4. The set now starts at 0 so the shortest horizon really is the next reading, and
# `steps_ahead`/`days_ahead_est` report h+1 rather than h.
HORIZONS: tuple = tuple(SCHEMA["horizons"])   # (0, 1, 2) -> 1, 2, 3 readings ahead
SIGNALS: tuple = tuple(SCHEMA["target_signals"])
LAGS: tuple = (1, 2, 3, 5, 7, 14)
WINDOWS: tuple = (3, 7, 14, 30)
EWM_ALPHAS: tuple = (0.1, 0.3)

# The contract the rest of the pipeline is entitled to assume. Asserted at import so a
# schema edit that breaks an invariant fails immediately and names itself, rather than
# surfacing as an off-by-one in the advisory or a KeyError three components later.
assert HORIZONS[0] == 0, (
    "horizons are target shifts and must start at 0: h0 is the NEXT reading. A set starting "
    "at 1 puts every forecast one reading further out than the UI advertises.")
assert list(HORIZONS) == sorted(set(HORIZONS)), "horizons must be sorted and unique"
assert "sbp" in SIGNALS, "sbp is the primary signal and every model keys off it"
assert set(SIGNALS) <= set(SCHEMA_COLUMNS), "a target signal is not a schema column"
assert set(NUMERICAL_COLUMNS) <= set(SCHEMA_COLUMNS), (
    "numerical_columns names a column the schema does not declare")
assert 0.0 < MAX_OUT_OF_RANGE < 1.0, "max_out_of_range is a fraction"

"""
Model Trainer related content start with MODEL_TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_REPORT_DIR: str = "reports"
MODEL_TRAINER_BUNDLE_FILE_NAME: str = "predictor.joblib"

# Row caps so a full run stays tractable on CPU.
MAX_TRAIN_ROWS: int = 20_000
MAX_EVAL_ROWS: int = 5_000

"""
MODEL REGISTRY SCOPE.

The bake-off registers 24 architectures across 9 families. Running all of them across three
signals and three horizons is hours of CPU, so `MODEL_TIER` gates them by cost and FAST_MODE
shrinks every budget at once. Neither changes WHICH model can win -- only how many get to
compete -- so a tier change is a compute decision, not a modelling one.

  low     linear, classical smoothers, window/delta reparameterisations   seconds
  medium  + gradient boosting, per-patient local models, ensembles        minutes
  high    + random forest, extra trees, kNN, local HGB, stacked ensemble  hours
"""
MODEL_TIER: str = "high"            # low | medium | high | all
FAST_MODE: bool = False             # shrinks row caps, tuning budget and bootstrap draws

# Per-patient models: a patient contributing ~0.6 * 60 = 36 training rows cannot support 80+
# features, so the local scope gets a compact feature set and a pooled fallback.
RUN_LOCAL_SCOPE: bool = True
LOCAL_MIN_TRAIN_ROWS: int = 30
LOCAL_MAX_FEATURES: int = 10

# Distance-based learners are O(n^2)-ish at fit time; cap their training sample.
KERNEL_FIT_ROWS: int = 20_000

# Legs every ensemble combines. Deliberately one linear and one tree model: averaging two
# gradient boosters mostly averages the same errors.
ENS_BASE: tuple = ("ridge", "hgb")

# Tuning
TUNE: bool = True
TUNER: str = "optuna"               # optuna | grid | random (falls back to random if absent)
TUNE_DRAWS: int = 12                # random search draws
TUNE_TRIALS: int = 40               # optuna trials
TUNE_SAMPLE_ROWS: int = 30_000
TUNE_FOLDS: int = 3
TPE_STARTUP_TRIALS: int = 8
PRUNER_STARTUP_TRIALS: int = 5
PRUNER_WARMUP_STEPS: int = 1

# Bootstrap draws for every confidence interval. Configurable because it dominates the
# runtime of the scorecard and FAST_MODE needs to be able to cut it.
N_BOOT: int = 200

if FAST_MODE:
    MODEL_TIER = "low"
    TUNE_TRIALS = 10
    TUNE_DRAWS = 4
    TUNE_SAMPLE_ROWS = 8_000
    N_BOOT = 60

# Serving
LATENCY_BUDGET_MS: float = 200.0
COLD_START_MIN_READINGS: int = 7
STEADY_STATE_READINGS: int = 48

"""
GOVERNANCE: clinician / ops inputs. Never searched, never tuned, asserted throughout.
"""

POPULATION_THRESHOLD_MMHG: float = 140.0
EMERGENCY_FLOOR_MMHG: float = 180.0  # never personalised
OFFSET_CAP_LOOSEN: float = 15.0      # asymmetric on purpose: loosening is the hazardous way
OFFSET_CAP_TIGHTEN: float = 25.0
ALERT_BUDGET_PCT: float = 5.0        # staffing capacity, not a statistic
WARN_WINDOW: int = 3                 # sessions of lead time the detector must deliver
EVENT_QUANTILE: float = 0.95         # personal quantile defining a high-BP event

# How old the newest reading may be before a forecast is refused. Deliberately the same
# number as RuleEngine.STALE_GAP_DAYS: if the engine has already stopped treating a reading
# as current, the forecaster must not still be projecting from it. One threshold means the
# two layers cannot disagree about what "stale" means.
STALE_FORECAST_MAX_DAYS: int = 14

GOVERNANCE_KEYS: list = [
    "population_threshold_mmHg",
    "emergency_floor_mmHg",
    "offset_cap_loosen",
    "offset_cap_tighten",
    "alert_budget_pct",
    "warn_window",
    "event_quantile",
    "stale_forecast_max_days",
]

# Offset blend defaults (searched over warm/k/q; the caps above are deliberately excluded).
OFFSET_WARM: int = 48
OFFSET_K: float = 30.0
OFFSET_Q: float = 0.90
OFFSET_SEARCH_WARM: tuple = (24, 36, 48, 72)
OFFSET_SEARCH_K: tuple = (10, 30, 60)
OFFSET_SEARCH_Q: tuple = (0.85, 0.90, 0.95)

# Learned Model 2 (src/utils/ml_utils/model/offset_learned.py). The threshold is the q0.90
# of the patient's NEXT 30 sessions, evaluated at several history lengths so the model has
# to work at the short histories it will actually meet at serving time.
OFFSET_TARGET_Q: float = 0.90
OFFSET_FUTURE: int = 30
OFFSET_CUTS: tuple = (20, 32, 48, 72, 100, 140)
LEARNED_OFFSET: bool = True

# Advisory arbitration
CONFIDENCE_TAU: float = 0.55

# Fairness gate: a subgroup may not be worse than overall by more than this margin.
FAIR_MARGIN_MMHG: float = 1.5

"""
SAFETY GATES: the promotion contract. A run that fails a critical gate leaves final_model/
untouched, so these bounds decide what reaches production. Governance values above are the
inputs; these are the tolerances on the evidence that the governance held.
"""

# Share of test rows allowed to sit outside the training distribution before the model is
# judged to be answering questions it was never fitted for.
OOD_RATE_MAX: float = 0.20

# Share of forecast rows whose (sbp, dbp) pair may be physiologically incoherent -- diastolic
# at or above systolic, or a pulse pressure below the floor `schemas.Reading` enforces on
# submitted readings. Zero, because the two legs come from independently selected
# architectures and there is no benign reason for them to contradict each other: any non-zero
# rate is a real defect, not a tolerance to be spent. Reported by gate 17, which is
# non-critical until a full run establishes that zero is actually attainable on this cohort.
PAIR_VIOLATION_MAX: float = 0.0

# Reading counts the cold-start curve probes. 7 and 48 are the tier boundaries
# (COLD_START_MIN_READINGS / STEADY_STATE_READINGS); the rest bracket them.
COLD_START_PROBE_N: tuple = (3, 7, 10, 25, 48, 60)

# How much worse the forecaster may be on an unseen patient than on an unseen time window
# for the same patients. This is the cold-start penalty, and it is the number that decides
# whether a new user gets a usable model or a confident guess.
COLD_START_PENALTY_MAX_MMHG: float = 4.0

# Session dropout fractions for the missingness sweep. The pipeline never imputes a skipped
# session, which is correct; this is the evidence that the choice survives users who skip.
MISSINGNESS_FRACS: tuple = (0.0, 0.1, 0.2, 0.3, 0.5)

# Extra MAE the shipped forecaster may absorb at the largest dropout fraction.
MISSINGNESS_MAX_DEGRADATION_MMHG: float = 6.0
