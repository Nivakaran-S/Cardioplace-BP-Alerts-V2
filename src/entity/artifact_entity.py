from dataclasses import dataclass, field


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str
    feature_store_file_path: str
    static_file_path: str
    reconciliation_file_path: str
    decomposition_file_path: str | None = None


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str | None
    invalid_test_file_path: str | None
    drift_report_file_path: str
    contract_report_file_path: str
    n_pass: int = 0
    n_warn: int = 0
    n_fail: int = 0


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    panel_file_path: str
    feature_file_path: str
    feature_list_file_path: str
    leakage_report_file_path: str
    feature_selection_report_file_path: str | None = None
    n_features: int = 0
    n_features_built: int = 0
    n_patients: int = 0
    leakage_clean: bool = True


@dataclass
class ForecastMetricArtifact:
    """Point-forecast metrics for one signal x horizon, per the notebook's `evaluate`."""
    signal: str
    horizon: int
    model: str
    n: int
    mae: float
    mae_lo: float
    mae_hi: float
    rmse: float
    r2: float
    bias: float
    target_sd: float
    dir_acc: float | None = None
    split: str = "test"


@dataclass
class DetectorMetricArtifact:
    """Early-warning detector performance at the configured alert budget."""
    detector: str
    budget_pct: float
    base_rate: float
    precision: float
    recall: float
    lift: float
    auc_roc: float
    auc_pr: float
    auc_pr_lift: float
    lead_days_median: float
    alerts_per_patient_week: float


@dataclass
class OffsetMetricArtifact:
    """Personalisation offset: selected blend and the invariant it must satisfy."""
    label: str
    warm: int
    k: float
    q: float
    n_patients: int
    mae: float
    max_threshold: float
    emergency_floor: float
    n_capped: int
    ship: str = ""
    best_legal_alternative: str = ""
    gain_mmHg: float = float("nan")

    @property
    def floor_holds(self) -> bool:
        return self.max_threshold < self.emergency_floor


@dataclass
class RuleEngineArtifact:
    """Deterministic rule engine output over the panel."""
    alerts_file_path: str
    registry_file_path: str
    rows_evaluated: int
    alerts_fired: int
    emergency_fired: int
    evaluable_rules: int
    blocked_rules: int


@dataclass
class SafetyGateArtifact:
    gate_report_file_path: str
    gates_passed: int
    gates_total: int
    critical_failures: int
    promotable: bool


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    bundle_file_path: str
    report_dir: str
    selected_family: dict = field(default_factory=dict)
    train_metric_artifact: list = field(default_factory=list)
    test_metric_artifact: list = field(default_factory=list)
    detector_metric_artifact: list = field(default_factory=list)
    offset_metric_artifact: OffsetMetricArtifact | None = None
    rule_engine_artifact: RuleEngineArtifact | None = None
    safety_gate_artifact: SafetyGateArtifact | None = None
