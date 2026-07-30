import os
import sys
from dataclasses import dataclass

import pandas as pd
from scipy.stats import ks_2samp

from src.constants.training_pipeline import (
    INGEST_RANGES,
    MAX_OUT_OF_RANGE,
    MAX_SESSIONS_WITHOUT_D1,
    PAPER_COUNTS,
    SCHEMA_FILE_PATH,
    SCHEMA_RANGES,
    SESSION_RECON_BAND,
    STALE_FORECAST_MAX_DAYS,
)
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.utils.main_utils.utils import read_yaml_file, save_report, write_yaml_file
from src.utils.ml_utils.feature.cadence import GAP_CLIP_HI_DAYS, attach_cadence

# The contract's notion of "stale" must be the engine's, by construction rather than by
# two constants that happen to agree today.
from src.utils.ml_utils.rule_engine.engine import RuleEngine

STALE_GAP_DAYS = RuleEngine.STALE_GAP_DAYS

# The engine stops treating a reading as current at STALE_GAP_DAYS; the predictor stops
# forecasting from it at STALE_FORECAST_MAX_DAYS. If those ever diverge there is a window
# where the forecaster projects from a reading the engine has already written off, so the
# agreement is asserted rather than left to whoever edits one of them next.
assert RuleEngine.STALE_GAP_DAYS == STALE_FORECAST_MAX_DAYS, (
    f"staleness thresholds disagree: engine gates at {RuleEngine.STALE_GAP_DAYS} d but the "
    f"forecaster refuses at {STALE_FORECAST_MAX_DAYS} d")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    critical: bool = False

    @property
    def status(self) -> str:
        return "PASS" if self.passed else ("FAIL" if self.critical else "WARN")


class DataContract:
    """Schema, range, structural and referential checks over the raw and panel frames."""

    REQUIRED = {"patient_id": "object", "ts": "datetime", "sbp": "number", "dbp": "number"}
    OPTIONAL = ["weight", "idwg", "sbp_drop", "uf_total", "temperature", "dryweight",
                "session_hours", "session_source"]
    # Ranges and the tolerance come from data_schema/schema.yaml, which is the single source
    # of truth for the session contract. Restating them here is how they drifted before.
    RANGES = {k: v for k, v in SCHEMA_RANGES.items() if isinstance(v, tuple)}
    MAX_OUT_OF_RANGE = MAX_OUT_OF_RANGE

    def __init__(self, max_out_of_range: float = None):
        self.checks = []
        if max_out_of_range is not None:
            self.MAX_OUT_OF_RANGE = max_out_of_range

    def _add(self, name, passed, detail="", critical=False):
        self.checks.append(Check(name, bool(passed), detail, critical))

    def _schema(self, raw):
        for col, kind in self.REQUIRED.items():
            here = col in raw.columns
            self._add(f"schema: '{col}' present", here, critical=True)
            if not here:
                continue
            s = raw[col]
            ok = (pd.api.types.is_numeric_dtype(s) if kind == "number"
                  else pd.api.types.is_datetime64_any_dtype(s) if kind == "datetime" else True)
            self._add(f"schema: '{col}' usable as {kind}", ok, f"got {s.dtype}", critical=True)
        missing = [c for c in self.OPTIONAL if c not in raw.columns]
        self._add("schema: optional columns present", not missing, f"missing {missing}")

    def _ranges(self, raw, static):
        for col, (lo, hi) in self.RANGES.items():
            src = raw if col in raw.columns else (static if col in static.columns else None)
            if src is None:
                continue
            s = pd.to_numeric(src[col], errors="coerce").dropna()
            if s.empty:
                continue
            bad = float(((s < lo) | (s > hi)).mean())
            self._add(f"range: {col} in [{lo}, {hi}]", bad < self.MAX_OUT_OF_RANGE,
                      f"{bad:.3%} outside; observed [{s.min():.1f}, {s.max():.1f}]")
        pp_bad = float(((raw.sbp - raw.dbp) < INGEST_RANGES["min_pulse_pressure"]).mean())
        self._add("range: pulse pressure >= 10 mmHg", pp_bad < self.MAX_OUT_OF_RANGE,
                  f"{pp_bad:.3%} violate")

    MAX_CLIPPED_GAP_FRAC = 0.01
    MAX_FLAT_REPEAT_FRAC = 0.05

    def _structure(self, raw, panel, min_sessions):
        """Structural checks, all of which can actually fail.

        Two of these previously could not. "gaps positive and bounded" asserted
        1 <= days_since_last <= 30 on a column that had just been clipped to [1, 30], and
        "step index contiguous from 0" asserted contiguity of a cumcount() output. Both
        passed by construction and told us nothing -- the first hid the fact that this
        corpus contains gaps up to 124 days. They now assert on `days_since_last_raw`,
        the unclipped truth, and on properties the pipeline does not itself guarantee.
        """
        dup = int(raw.duplicated(["patient_id", "ts"]).sum())
        self._add("structure: unique (patient, session)", dup == 0, f"{dup} duplicates",
                  critical=True)
        self._add("structure: timestamps increasing within patient",
                  panel.groupby("series_id").ts.apply(lambda s: s.is_monotonic_increasing).all(),
                  critical=True)

        raw_gap = panel.days_since_last_raw.dropna()
        # C1 -- a non-positive gap means two readings share a timestamp. While `ts` is
        # date-only this overlaps the duplicate check; the moment timestamps carry a time
        # component it becomes the only check that catches it.
        n_bad = int((raw_gap <= 0).sum())
        self._add("structure: session gaps strictly positive", n_bad == 0,
                  f"{n_bad} of {len(raw_gap):,} non-positive", critical=True)

        # C3 -- the clip is a guard against a pathological value, not a transform applied
        # to a material share of the data. Past 1% the feature stops meaning "days since
        # the last reading" for enough rows that the model is training on a censored
        # variable without saying so, and the fix is to raise the ceiling and retrain.
        over = float((raw_gap > GAP_CLIP_HI_DAYS).mean()) if len(raw_gap) else 0.0
        self._add("structure: the cadence clip is a guard, not a transform",
                  over < self.MAX_CLIPPED_GAP_FRAC,
                  f"{over:.3%} of gaps exceed the {GAP_CLIP_HI_DAYS:.0f}-day ceiling "
                  f"(max {raw_gap.max():.0f}d, budget {self.MAX_CLIPPED_GAP_FRAC:.0%})",
                  critical=True)

        # C2 -- how often a reading arrives after the engine's staleness threshold. Not
        # critical: a rising stale rate is a fact about patient behaviour, not a data
        # defect, and blocking a training run on it would be wrong. It has to be visible.
        stale = float((raw_gap > STALE_GAP_DAYS).mean()) if len(raw_gap) else 0.0
        self._add("structure: sessions following a stale gap are rare",
                  stale < self.MAX_OUT_OF_RANGE,
                  f"{stale:.3%} of gaps exceed the {STALE_GAP_DAYS}-day staleness threshold")

        # C4 -- replaces the contiguity tautology. Forward-filling a daily grid is the one
        # thing panel construction forbids; its signature is consecutive rows carrying
        # identical vitals exactly one day apart. A real ffill drives this toward the fill
        # fraction (~57% on this cadence), so the threshold sits far above noise and far
        # below the failure mode.
        prev_same = (panel.sbp.eq(panel.sbp.shift(1)) & panel.dbp.eq(panel.dbp.shift(1))
                     & panel.series_id.eq(panel.series_id.shift(1))
                     & panel.days_since_last_raw.eq(1))
        flat = float(prev_same.mean())
        self._add("structure: no forward-filled runs", flat < self.MAX_FLAT_REPEAT_FRAC,
                  f"{flat:.3%} of rows repeat the previous vitals one day apart "
                  f"(a daily-grid ffill would approach the fill fraction)", critical=True)

        short = int((panel.groupby("series_id").size() < min_sessions).sum())
        self._add(f"structure: every patient has >= {min_sessions} sessions", short == 0,
                  f"{short} below floor", critical=True)

    def _reconciliation(self, sessions, recon=None):
        """Did sessionisation produce a defensible number of sessions?

        The published 165,986 is d1's row count, and vip carries BP readings for dialysis
        days d1 never recorded, so the honest target is a band around 1.19 rather than 1.00.
        The band exists to catch a REGRESSION in the session key, which is a failure mode
        this corpus has already exhibited: keying on the calendar date split every
        past-midnight session in two, pushed the ratio to 1.44, and left over half the rows
        with no dialysis record and therefore no weight, dryweight or idwg.

        Both bounds are critical. Too high means sessions are being split again; too low
        means sessions are being merged, which silently drops readings.
        """
        lo, hi = SESSION_RECON_BAND
        derived, published = len(sessions), PAPER_COUNTS["sessions"]
        ratio = derived / max(published, 1)
        self._add(f"reconciliation: session ratio in [{lo}, {hi}]", lo <= ratio <= hi,
                  f"{derived:,} derived vs {published:,} published (ratio {ratio:.3f}); "
                  f"above the band means sessions are being split, below means merged",
                  critical=True)

        if "session_source" in sessions.columns:
            # Every vip_only session is a row with no weight, dryweight or idwg. This is the
            # check that would have caught the 53.6% hole the calendar keying opened up.
            no_d1 = float((sessions.session_source != "d1").mean())
            self._add("reconciliation: sessions carry a dialysis record",
                      no_d1 <= MAX_SESSIONS_WITHOUT_D1,
                      f"{no_d1:.1%} of sessions have no d1 row, so weight/dryweight/idwg "
                      f"are unavailable on them (budget {MAX_SESSIONS_WITHOUT_D1:.0%})",
                      critical=True)

        if "session_hours" in sessions.columns:
            # A 4-hour median is what a hemodialysis session looks like. A median far from
            # that means the gap threshold merged or split sessions, and it moves before the
            # count ratio does, so it is the earlier warning of the two.
            med = float(pd.to_numeric(sessions.session_hours, errors="coerce").median())
            self._add("reconciliation: median session duration is dialysis-shaped",
                      2.0 <= med <= 6.0, f"median {med:.2f}h against a 4h standard session")

    def _referential(self, panel):
        unmatched = int(panel.age.isna().sum())
        self._add("join: demographics matched", unmatched == 0,
                  f"{unmatched} sessions ({unmatched / max(len(panel), 1):.2%}) unmatched")
        self._add("join: merge did not duplicate rows",
                  len(panel) == panel.drop_duplicates(["series_id", "ts"]).shape[0],
                  critical=True)
        self._add("join: demographics constant within patient",
                  int(panel.groupby("series_id")[["gender", "age"]].nunique().max().max()) <= 1,
                  "gender/age must not vary across a patient's sessions")

    def validate(self, raw, static, panel, min_sessions) -> pd.DataFrame:
        self.checks.clear()
        self._schema(raw)
        self._ranges(raw, static)
        self._reconciliation(raw)
        self._structure(raw, panel, min_sessions)
        self._referential(panel)
        return pd.DataFrame([{"check": c.name, "status": c.status, "detail": c.detail}
                             for c in self.checks])

    def enforce(self) -> None:
        fails = [c for c in self.checks if c.status == "FAIL"]
        if fails:
            raise AssertionError("data contract violated:\n"
                                 + "\n".join(f"  - {c.name}: {c.detail}" for c in fails))

    @classmethod
    def self_test(cls) -> pd.DataFrame:
        """Prove each structural check bites, by feeding it data that should trip it.

        A check that cannot fail is worse than no check: it reports PASS forever and is
        read as evidence. Two of these were exactly that before -- they asserted bounds on
        a column the pipeline had just clipped to those bounds. This runs every structural
        predicate against a frame built to violate it and asserts the violation is caught.
        """
        base = pd.DataFrame({
            "patient_id": ["p"] * 6, "series_id": ["p"] * 6,
            "ts": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-05",
                                  "2026-01-07", "2026-01-09", "2026-01-11"]),
            "sbp": [140.0, 142, 144, 146, 148, 150], "dbp": [80.0, 81, 82, 83, 84, 85],
            "age": [68.0] * 6, "gender": ["M"] * 6,
        })
        base = attach_cadence(base, by="series_id")
        base["step"] = range(len(base))

        def corrupt_duplicate(d):
            d = pd.concat([d, d.tail(1)], ignore_index=True)
            return attach_cadence(d, by="series_id")

        def corrupt_backwards(d):
            d = d.copy()
            d.loc[3, "ts"] = pd.Timestamp("2025-12-01")
            return attach_cadence(d, by="series_id")

        def corrupt_huge_gap(d):
            d = d.copy()
            d.loc[4, "ts"] = pd.Timestamp("2027-01-01")   # ~1 year, far past the ceiling
            return attach_cadence(d, by="series_id")

        def corrupt_ffill(d):
            d = d.copy()
            d["ts"] = pd.date_range("2026-01-01", periods=len(d), freq="D")
            d["sbp"] = 140.0
            d["dbp"] = 80.0
            return attach_cadence(d, by="series_id")

        cases = [
            ("structure: unique (patient, session)", corrupt_duplicate),
            ("structure: timestamps increasing within patient", corrupt_backwards),
            ("structure: the cadence clip is a guard, not a transform", corrupt_huge_gap),
            ("structure: no forward-filled runs", corrupt_ffill),
            ("structure: every patient has >= 60 sessions", lambda d: d),
        ]
        rows = []
        for name, fn in cases:
            bad = fn(base)
            c = cls()
            c._structure(bad, bad, min_sessions=60)
            hit = next((x for x in c.checks if x.name == name), None)
            rows.append(dict(check=name,
                             caught=bool(hit is not None and hit.status == "FAIL"),
                             detail=hit.detail if hit else "check not emitted"))

        # The reconciliation checks take the session frame directly rather than a panel, so
        # they are exercised on purpose-built frames instead of the shared `base`.
        lo, hi = SESSION_RECON_BAND
        n_pub = PAPER_COUNTS["sessions"]

        def _sessions(n, source="d1", hours=4.0):
            return pd.DataFrame({"session_source": [source] * n,
                                 "session_hours": [hours] * n})

        # Ratio above the band is the exact regression the calendar-date key produced.
        recon_cases = [
            (f"reconciliation: session ratio in [{lo}, {hi}]",
             _sessions(int(n_pub * (hi + 0.25)))),
            (f"reconciliation: session ratio in [{lo}, {hi}]",
             _sessions(int(n_pub * (lo - 0.25)))),
            ("reconciliation: sessions carry a dialysis record",
             _sessions(int(n_pub * 1.19), source="vip_only")),
        ]
        for name, frame in recon_cases:
            c = cls()
            c._reconciliation(frame)
            hit = next((x for x in c.checks if x.name == name), None)
            rows.append(dict(check=f"{name} (n={len(frame):,})",
                             caught=bool(hit is not None and hit.status == "FAIL"),
                             detail=hit.detail if hit else "check not emitted"))

        out = pd.DataFrame(rows)
        n_ok = int(out.caught.sum())
        logging.info("contract self-test: %d/%d structural checks provably bite",
                     n_ok, len(out))
        return out


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
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

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """Compare against the declared column list, not the number of top-level YAML keys."""
        try:
            expected = [list(c.keys())[0] if isinstance(c, dict) else c
                        for c in self._schema_config.get("columns", [])]
            missing = [c for c in expected if c not in dataframe.columns]
            logging.info("Required number of columns:%s", len(expected))
            logging.info("Data frame has columns:%s", len(dataframe.columns))
            if missing:
                logging.warning("missing columns: %s", missing)
                return False
            return True
        except Exception as e:
            raise CustomException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=None) -> bool:
        """Per-column KS test. Returns the status instead of silently discarding it."""
        try:
            threshold = threshold or self.data_validation_config.drift_threshold
            status = True
            report = {}
            numeric = [c for c in base_df.columns
                       if pd.api.types.is_numeric_dtype(base_df[c])
                       and c in current_df.columns]
            for column in numeric:
                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()
                if len(d1) < 2 or len(d2) < 2:
                    continue
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found,
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report, replace=True)
            return status
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            for label, df in (("train", train_dataframe), ("test", test_dataframe)):
                if not self.validate_number_of_columns(dataframe=df):
                    logging.warning("%s dataframe does not contain all schema columns", label)

            # The contract runs on the full session frame plus a minimal panel view, so the
            # structural checks (contiguous step, bounded gaps) have something to check.
            sessions = DataValidation.read_data(
                self.data_ingestion_artifact.feature_store_file_path)
            static = pd.read_csv(self.data_ingestion_artifact.static_file_path)
            static["patient_id"] = static.patient_id.astype(str)

            # Demographics are already on every session row (merged at ingest), so the panel
            # is just the session frame indexed by step -- re-merging would suffix the columns.
            panel = sessions.sort_values(["patient_id", "ts"]).copy()
            panel["series_id"] = panel.patient_id
            # Carries `days_since_last_raw` as well; the structural checks assert on the raw
            # column, because asserting on the clipped one cannot fail.
            panel = attach_cadence(panel, by="series_id")
            panel["step"] = panel.groupby("series_id").cumcount()
            # The contract's session floor applies to the admitted cohort; at this stage the
            # frame is still every patient, so check against the cohort that will be admitted.
            admitted = panel.groupby("series_id").size()
            keep = admitted[admitted >= self.data_validation_config.min_sessions].index
            panel_admitted = panel[panel.series_id.isin(keep)]

            # Run the self-test BEFORE the contract: a suite that cannot fail must not be
            # allowed to report PASS on real data, so a blunt check is itself a failure.
            st = DataContract.self_test()
            if not bool(st.caught.all()):
                raise AssertionError(
                    "data contract self-test failed -- these checks cannot detect the "
                    "corruption they exist for:\n"
                    + "\n".join(f"  - {r.check}" for r in st[~st.caught].itertuples()))

            contract = DataContract(self.data_validation_config.max_out_of_range)
            report = contract.validate(sessions, static, panel_admitted,
                                       self.data_validation_config.min_sessions)
            save_report(self.data_validation_config.contract_report_file_path, report)
            n_pass = int((report.status == "PASS").sum())
            n_warn = int((report.status == "WARN").sum())
            n_fail = int((report.status == "FAIL").sum())
            logging.info("data contract: %d pass | %d warn | %d fail | %d/%d checks "
                         "self-tested", n_pass, n_warn, n_fail,
                         int(st.caught.sum()), len(st))
            contract.enforce()
            logging.info("data contract satisfied")

            status = self.detect_dataset_drift(base_df=train_dataframe,
                                               current_df=test_dataframe)

            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path,
                                   index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path,
                                  index=False, header=True)

            return DataValidationArtifact(
                validation_status=bool(status and n_fail == 0),
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                contract_report_file_path=self.data_validation_config.contract_report_file_path,
                n_pass=n_pass, n_warn=n_warn, n_fail=n_fail,
            )
        except Exception as e:
            raise CustomException(e, sys)
