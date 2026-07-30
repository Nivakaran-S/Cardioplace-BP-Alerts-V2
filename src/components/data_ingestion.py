import os
import sys

import numpy as np
import pandas as pd

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception.custom_exception import CustomException
from src.logging.logger import logging, timer
from src.utils.main_utils.utils import summarise


class HemobpLoader:
    """Streams the local HEMOBP CSVs and reduces them to one row per dialysis session.

    vip.csv is ~250 MB / ~4.4M readings, so it is parsed in chunks. The chunks are then
    concatenated before sessions are cut, because a session is defined by the gap to the
    previous reading and a chunk boundary is not a session boundary. Only the six columns
    the reduction needs are retained, so the resident frame is ~110 MB.

    A session is a contiguous run of readings, NOT a calendar date. See the sessionisation
    note in `src/constants/training_pipeline/__init__.py`: dating on the calendar split every
    session that ran past midnight in two, and the resulting phantom sessions had no d1
    record, so weight/dryweight/idwg were NaN on 53.6% of the corpus.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.meta = {}
        self.decomposition = pd.DataFrame()

    def _resolve(self, path: str, label: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at {path}. The HEMOBP CSVs ship with this repository; "
                f"vip.csv is tracked with Git LFS, so run `git lfs pull` if it is a pointer file."
            )
        return path

    def _read_vip(self) -> pd.DataFrame:
        """Parse vip.csv in chunks, keep the admissible readings, return one frame."""
        dtypes = {"pid": "int64", "measuretime": "int16",
                  "sbp": "int16", "dbp": "int16", "uf": "float32"}
        ranges = self.config.ingest_ranges
        lo_s, hi_s = ranges["sbp"]
        lo_d, hi_d = ranges["dbp"]
        parts, n_raw, n_dropped, n_undated = [], 0, 0, 0

        path = self._resolve(self.config.vip_file_path, "vip.csv")
        for chunk in pd.read_csv(path, chunksize=self.config.chunksize,
                                 usecols=["pid", "datatime", "measuretime", "sbp", "dbp", "uf"],
                                 dtype=dtypes):
            n_raw += len(chunk)
            chunk["dt"] = pd.to_datetime(chunk.datatime, errors="coerce")
            n_undated += int(chunk.dt.isna().sum())
            chunk = chunk.dropna(subset=["dt"])
            keep = (chunk.sbp.between(lo_s, hi_s) & chunk.dbp.between(lo_d, hi_d)
                    & ((chunk.sbp - chunk.dbp) >= ranges["min_pulse_pressure"]))
            n_dropped += int((~keep).sum())
            chunk = chunk[keep]
            if not chunk.empty:
                parts.append(chunk[["pid", "dt", "measuretime", "sbp", "dbp", "uf"]])

        if not parts:
            raise ValueError("no admissible BP readings survived the ingest filters")
        v = pd.concat(parts, ignore_index=True)
        del parts
        self.meta.update(raw_bp_rows=n_raw, rows_out_of_range=n_dropped,
                         rows_undated=n_undated, readings_admissible=len(v))
        return v

    def _reduce_vip(self) -> pd.DataFrame:
        """Cut the reading stream into sessions and reduce each to one row.

        Two decisions carry this function, both measured rather than assumed:

        1. A session boundary is a gap of more than `session_gap_hours`. Gaps within a
           patient are bimodal -- p95 is 1.0h and p97 is 44h, with 0.003% of gaps in the
           4-12h band -- so any threshold in that valley gives the same answer.
        2. A session belongs to the dialysis day beginning at `dialysis_day_start_hour`,
           which is 08:00 because that is the hour every d1 `keyindate` is stamped with.
           The corpus runs three shifts and the last crosses midnight; dating on the
           calendar date instead put those sessions on a day with no d1 record.
        """
        v = self._read_vip()
        # mergesort keeps the sort stable, so readings sharing a timestamp (the corpus has
        # repeated transmissions of the same measurement) retain their file order and the
        # first/last reduction below is deterministic across runs.
        v = v.sort_values(["pid", "dt"], kind="mergesort").reset_index(drop=True)

        gap_h = v.groupby("pid", sort=False).dt.diff().dt.total_seconds() / 3600.0
        starts = gap_h.isna() | (gap_h > self.config.session_gap_hours)
        v["sid"] = np.cumsum(starts.to_numpy())

        g = v.groupby("sid", sort=True)
        s = g.agg(pid=("pid", "first"), session_start=("dt", "first"),
                  session_end=("dt", "last"), sbp=("sbp", "first"), dbp=("dbp", "first"),
                  sbp_post=("sbp", "last"), sbp_min=("sbp", "min"),
                  uf_total=("uf", "max"), n_meas=("sbp", "size")).reset_index(drop=True)

        s["session_hours"] = ((s.session_end - s.session_start)
                              .dt.total_seconds() / 3600.0).round(2)
        s["date"] = (s.session_start
                     - pd.Timedelta(hours=self.config.dialysis_day_start_hour)).dt.normalize()
        s["crosses_midnight"] = s.session_end.dt.normalize() != s.session_start.dt.normalize()

        n_cross = int(s.crosses_midnight.sum())
        n_runs, n_days = len(s), int(s.drop_duplicates(["pid", "date"]).shape[0])
        s = self._collapse_same_day(s)
        s["sbp_drop"] = s.sbp - s.sbp_min

        self.meta.update(
            sessions_derived=len(s),
            reading_runs=n_runs,
            sessions_crossing_midnight=n_cross,
            second_session_same_day=n_runs - n_days,
            calendar_day_sessions=int(v.assign(
                d=v.dt.dt.normalize()).drop_duplicates(["pid", "d"]).shape[0]),
            session_hours_median=round(float(s.session_hours.median()), 2))
        logging.info("sessionised %d readings into %d runs -> %d sessions (gap > %.0fh); "
                     "%d (%.1f%%) cross midnight and would have been double-counted by "
                     "calendar-date keying; median duration %.2fh",
                     len(v), n_runs, len(s), self.config.session_gap_hours, n_cross,
                     100 * n_cross / max(n_runs, 1), s.session_hours.median())
        return s

    @staticmethod
    def _collapse_same_day(s: pd.DataFrame) -> pd.DataFrame:
        """One row per (patient, dialysis day), even when the day held two treatment runs.

        `ts` means "the dialysis day", because that is what d1 records and what every
        downstream component keys on: the panel is one row per step, `days_since_last` is a
        day count, and the rule engine walks a patient's days in order. Two rows sharing a
        `ts` breaks all three -- the gap between them is 0 days, which the cadence clip would
        silently rewrite to 1 and report as a normal interval.

        So a second run on the same dialysis day is folded into the day rather than kept as
        a second row or dropped: pre-dialysis vitals come from the earliest run, the post
        reading from the latest, and the durations add. This is rare (46 of 197,260 runs on
        this corpus) but it is the difference between a contract that holds and one that has
        an exception carved into it.
        """
        dupes = s.duplicated(["pid", "date"], keep=False)
        if not dupes.any():
            return s.drop(columns=["crosses_midnight"])

        singles = s[~dupes].drop(columns=["crosses_midnight"])
        multi = s[dupes].sort_values(["pid", "date", "session_start"])
        merged = multi.groupby(["pid", "date"], as_index=False).agg(
            session_start=("session_start", "first"), session_end=("session_end", "last"),
            sbp=("sbp", "first"), dbp=("dbp", "first"), sbp_post=("sbp_post", "last"),
            sbp_min=("sbp_min", "min"), uf_total=("uf_total", "sum"),
            n_meas=("n_meas", "sum"),
            # Sum of the runs' durations, not end-minus-start: the latter would include the
            # hours between treatments and report a 20-hour dialysis session.
            session_hours=("session_hours", "sum"))
        out = pd.concat([singles, merged], ignore_index=True)
        logging.info("collapsed %d treatment runs into %d dialysis days (%d days held a "
                     "second run)", int(dupes.sum()), len(merged),
                     int(dupes.sum()) - len(merged))
        return out.sort_values(["pid", "date"]).reset_index(drop=True)

    def load(self):
        """Return (sessions, static): one row per (patient, session) plus demographics."""
        s = self._reduce_vip()
        s["pid"] = s.pid.astype(str)

        d1 = pd.read_csv(self._resolve(self.config.d1_file_path, "d1.csv"))
        d1["date"] = pd.to_datetime(d1.keyindate, errors="coerce").dt.normalize()
        n_d1_rows, n_d1_undated = len(d1), int(d1.date.isna().sum())
        d1 = d1.dropna(subset=["date"])
        d1["pid"] = d1.pid.astype(str)
        d1u = d1.drop_duplicates(["pid", "date"])

        self.decomposition = self._session_decomposition(s, d1, d1u)

        s = s.merge(d1u[["pid", "date", "weightstart", "weightend", "dryweight", "temperature"]],
                    on=["pid", "date"], how="left")
        s["weight"] = s.weightstart
        s["idwg"] = s.weightstart - s.dryweight
        # Which sessions the dialysis record actually covers. Carried onto every row rather
        # than inferred downstream from a NaN, because "d1 has no record of this session" and
        # "d1 recorded it but the scale reading is missing" are different facts and only the
        # first one is a coverage statement.
        s["session_source"] = np.where(s.weightstart.notna(), "d1", "vip_only")

        idp = pd.read_csv(self._resolve(self.config.idp_file_path, "idp.csv"))
        idp["pid"] = idp.pid.astype(str)
        idp["age"] = self.config.age_reference_year - idp.birthday
        # `first_dialysis` is a YYYY-MM string. Vintage -- years already spent on dialysis --
        # is a strong clinical covariate for blood-pressure control and costs one column.
        idp["first_dialysis_ts"] = pd.to_datetime(idp.first_dialysis, format="%Y-%m",
                                                  errors="coerce")
        static = idp[["pid", "gender", "age", "DM", "first_dialysis_ts"]].rename(
            columns={"pid": "patient_id"})

        no_d1 = float((s.session_source == "vip_only").mean())
        self.meta.update(patients_in_vip=int(s.pid.nunique()),
                         patients_in_idp=int(idp.pid.nunique()),
                         d1_rows=n_d1_rows, d1_undated=n_d1_undated,
                         sessions_without_d1=int((s.session_source == "vip_only").sum()),
                         frac_sessions_without_d1=round(no_d1, 4))
        logging.info("d1 merge: %.1f%% of sessions carry a dialysis record "
                     "(weight/dryweight/idwg available); %.1f%% are BP-only days d1 never "
                     "recorded", 100 * (1 - no_d1), 100 * no_d1)
        return s.rename(columns={"pid": "patient_id", "date": "ts"}), static

    def _session_decomposition(self, s: pd.DataFrame, d1: pd.DataFrame,
                               d1u: pd.DataFrame) -> pd.DataFrame:
        """Attribute the derived-vs-published session gap to named causes.

        The published figure (165,986) is d1's row count. It is a floor on the number of
        dialysis sessions, not the number of sessions -- vip carries BP readings for
        dialysis days d1 has no row for. This table exists so that statement is auditable
        instead of asserted, and so a future regression in the keying shows up as a shifted
        component rather than as a single ratio that someone talks themselves out of.
        """
        try:
            vip_pairs = set(map(tuple, s[["pid", "date"]].drop_duplicates().values))
            d1_pairs = set(map(tuple, d1u[["pid", "date"]].values))
            matched = vip_pairs & d1_pairs
            vip_only, d1_only = vip_pairs - d1_pairs, d1_pairs - vip_pairs

            span = s.groupby("pid").date.agg(["min", "max"])
            vo = pd.DataFrame(sorted(vip_only), columns=["pid", "date"])
            d1_span = d1u.groupby("pid").date.agg(["min", "max"])
            vo = vo.join(d1_span, on="pid")
            vo_outside = int((~vo.date.between(vo["min"], vo["max"])).sum()) if len(vo) else 0

            do = pd.DataFrame(sorted(d1_only), columns=["pid", "date"]).join(span, on="pid")
            do_inside = int(do.date.between(do["min"], do["max"]).sum()) if len(do) else 0

            published = self.config.paper_counts["sessions"]
            derived = len(s)
            delta = derived - published
            dupes = int(len(d1) - len(d1u))
            same_day = int(derived - len(vip_pairs))
            rows = [
                ("published sessions (d1 rows)", published, np.nan),
                ("derived sessions", derived, np.nan),
                ("delta", delta, 1.0),
                ("  sessions matching a d1 record", len(matched), np.nan),
                ("  BP-only days d1 never recorded, inside d1's span for that patient",
                 len(vip_only) - vo_outside, (len(vip_only) - vo_outside) / max(delta, 1)),
                ("  BP-only days outside d1's span for that patient", vo_outside,
                 vo_outside / max(delta, 1)),
                ("  second treatment run on the same dialysis day (collapsed into it)",
                 self.meta.get("second_session_same_day", same_day), np.nan),
                ("  d1 rows collapsed by (pid, date) dedup", dupes, np.nan),
                ("  d1 sessions with no admissible BP reading (inside vip's span)",
                 do_inside, np.nan),
                ("  d1 sessions with no admissible BP reading (outside vip's span)",
                 len(d1_only) - do_inside, np.nan),
                ("  d1 rows with an unparseable keyindate", self.meta.get("d1_undated", 0),
                 np.nan),
                ("sessions crossing midnight (recovered by gap keying)",
                 self.meta.get("sessions_crossing_midnight"), np.nan),
                ("sessions under the old calendar-date keying",
                 self.meta.get("calendar_day_sessions"), np.nan),
            ]
            out = pd.DataFrame(rows, columns=["component", "n", "share_of_delta"])
            out["share_of_delta"] = out.share_of_delta.round(4)
            self.meta.update(d1_match_rate=round(len(matched) / max(len(d1_pairs), 1), 4))
            logging.info("session decomposition: %d derived vs %d published (ratio %.3f); "
                         "%.1f%% of d1 sessions matched",
                         derived, published, derived / published,
                         100 * len(matched) / max(len(d1_pairs), 1))
            return out
        except Exception as e:
            raise CustomException(e, sys)

    def reconciliation(self, sessions: pd.DataFrame = None) -> pd.DataFrame:
        """Reconcile against the published data descriptor rather than restating it.

        A session here is a contiguous run of readings assigned to a dialysis day, so the
        sessions ratio is expected to sit above 1.0: the published 165,986 is d1's row count,
        and vip carries BP readings for dialysis days d1 never recorded. The companion
        `session_decomposition.csv` attributes that difference line by line. A large gap on
        the recordings or patients rows would mean an incomplete read instead.
        """
        paper = self.config.paper_counts
        recon = pd.DataFrame([
            dict(quantity="BP recordings (raw rows)", published=paper["bp_recordings"],
                 derived=self.meta.get("raw_bp_rows")),
            dict(quantity="Sessions", published=paper["sessions"],
                 derived=self.meta.get("sessions_derived")),
            dict(quantity="Patients", published=paper["patients"],
                 derived=self.meta.get("patients_in_idp")),
            dict(quantity="Rows dropped out of range", published=np.nan,
                 derived=self.meta.get("rows_out_of_range")),
            dict(quantity="d1 sessions matched (rate)", published=np.nan,
                 derived=self.meta.get("d1_match_rate")),
            dict(quantity="Sessions with no d1 record (share)", published=np.nan,
                 derived=self.meta.get("frac_sessions_without_d1")),
            dict(quantity="Sessions crossing midnight (recovered)", published=np.nan,
                 derived=self.meta.get("sessions_crossing_midnight")),
        ])
        recon["ratio"] = (recon.derived / recon.published).round(3)

        if sessions is not None and len(sessions):
            # DGP caveat, made visible rather than asserted: the upper tail is operationally
            # censored. Collection re-measured any SBP > 200 and kept the LATER value, so no
            # extreme-value claim is supportable from this corpus, and none is made.
            censored = float((sessions.sbp >= 200).mean())
            logging.info("pre-dialysis SBP >= 200 mmHg: %.3f%% of sessions -- the upper tail is "
                         "operationally censored (re-measured, later value kept)", 100 * censored)
            recon = pd.concat([recon, pd.DataFrame([dict(
                quantity="Sessions with SBP >= 200 (censored tail)", published=np.nan,
                derived=round(censored, 5), ratio=np.nan)])], ignore_index=True)
        return recon


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def export_sessions_as_dataframe(self):
        """Read the HEMOBP corpus from the local data directory."""
        try:
            loader = HemobpLoader(self.data_ingestion_config)
            with timer("HEMOBP ingest (vip -> sessions, merge d1 + idp)"):
                sessions, static = loader.load()
            logging.info("ingest meta: %s", loader.meta)
            return (sessions, static, loader.reconciliation(sessions),
                    loader.decomposition)
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_feature_store(self, sessions: pd.DataFrame, static: pd.DataFrame,
                                       recon: pd.DataFrame,
                                       decomposition: pd.DataFrame = None):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            sessions.to_csv(feature_store_file_path, index=False, header=True)
            static.to_csv(self.data_ingestion_config.static_file_path, index=False, header=True)
            recon.to_csv(self.data_ingestion_config.reconciliation_file_path, index=False)
            if decomposition is not None and len(decomposition):
                decomposition.to_csv(
                    self.data_ingestion_config.decomposition_file_path, index=False)
            return sessions
        except Exception as e:
            raise CustomException(e, sys)

    def split_data_as_train_test(self, sessions: pd.DataFrame):
        """Per-patient temporal tail, not a random row split.

        A random split puts a patient's future readings in train and their past in test.
        Every metric downstream would be measuring a model that has seen the answer.
        """
        try:
            cfg = self.data_ingestion_config
            df = sessions.sort_values(["patient_id", "ts"]).copy()
            step = df.groupby("patient_id").cumcount()
            last = step.groupby(df.patient_id).transform("max")
            is_test = step > (last * (1 - cfg.test_frac)).round()

            # `step` itself is not written: it is a split helper, and leaving it in would
            # dominate the drift report (test steps are higher than train steps by
            # construction). Data transformation recomputes it on the full timeline.
            train_set = df[~is_test]
            test_set = df[is_test]
            logging.info("Performed per-patient temporal train/test split "
                         "(%d train rows, %d test rows)", len(train_set), len(test_set))

            dir_path = os.path.dirname(cfg.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_set.to_csv(cfg.training_file_path, index=False, header=True)
            test_set.to_csv(cfg.testing_file_path, index=False, header=True)
            logging.info("Exported train and test file path.")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            sessions, static, recon, decomposition = self.export_sessions_as_dataframe()
            summarise(sessions, "sessions")
            summarise(static, "static")
            # Demographics ride along on every session row so the train/test CSVs are
            # self-describing. `static` is deduplicated first: a repeated patient_id would
            # multiply session rows and trip the contract's join check.
            sessions = sessions.merge(static.drop_duplicates("patient_id"),
                                      on="patient_id", how="left")
            sessions = self.export_data_into_feature_store(sessions, static, recon,
                                                           decomposition)
            self.split_data_as_train_test(sessions)
            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                static_file_path=self.data_ingestion_config.static_file_path,
                reconciliation_file_path=self.data_ingestion_config.reconciliation_file_path,
                decomposition_file_path=self.data_ingestion_config.decomposition_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
