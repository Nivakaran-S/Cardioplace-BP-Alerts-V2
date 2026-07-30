"""Panel construction, splits and strictly-causal feature engineering.

Port of notebook sections 1.3, 1.4, 3 and 3.2. The causal contract enforced here is the
single most consequential thing in the pipeline: a feature at step t may read only steps
<= t-1. `leakage_audit` checks it directly rather than assuming it.
"""

import numpy as np
import pandas as pd

from src.logging.logger import logging
from src.utils.main_utils.utils import deterministic_patient_split
from src.utils.ml_utils.feature.cadence import attach_cadence


def build_panel(sessions: pd.DataFrame, static: pd.DataFrame, config) -> pd.DataFrame:
    """Admit eligible patients, index them by session, and attach demographics."""
    g = sessions.groupby("patient_id")
    span = (g.ts.max() - g.ts.min()).dt.days
    sizes = g.size()
    eligible = [p for p in sizes[sizes >= config.min_sessions].index
                if span[p] >= config.min_span_days]
    df = sessions[sessions.patient_id.isin(eligible)]
    logging.info("%d patients meet >=%d sessions and >=%d days span",
                 len(eligible), config.min_sessions, config.min_span_days)

    if config.max_patients and len(eligible) > config.max_patients:
        # deterministic in the identifier, so the cohort does not change with row order
        ranked = sorted(eligible, key=lambda p: (-int(sizes[p]), str(p)))[:config.max_patients]
        df = df[df.patient_id.isin(ranked)]
        logging.info("capped to %d patients (set MAX_PATIENTS=None for all %d)",
                     config.max_patients, len(eligible))

    p = df.sort_values(["patient_id", "ts"]).copy()
    p["series_id"] = p.patient_id
    p = attach_cadence(p, by="series_id")
    p["step"] = p.groupby("series_id").cumcount()
    p = p.merge(static, on="patient_id", how="left")

    # derived clinical quantities used by both the rule engine and the features
    p["pulse_pressure"] = p.sbp - p.dbp
    p["map_est"] = p.dbp + (p.sbp - p.dbp) / 3.0
    p["dow"] = p.ts.dt.dayofweek
    p["is_weekend"] = (p.dow >= 5).astype(int)
    p["is_male"] = p.gender.astype(str).str.upper().str.startswith("M").astype(int)
    p["is_dm"] = p.DM.fillna(0).astype(int)
    p["age_band"] = pd.cut(p.age, [0, 50, 65, 75, 200],
                           labels=["<50", "50-64", "65-74", "75+"], right=False)

    # Intradialytic quantities. All of these are CONTEMPORANEOUS -- they describe the session
    # being predicted, not the history before it -- so every one of them is in
    # CausalFeatureBuilder.DROP and reaches a model only through its lagged form.
    p["nadir_sbp"] = p.sbp_min
    p["map_drop"] = p.map_est - (p.sbp_min + 2 * p.dbp) / 3.0
    if "dryweight" in p.columns:
        # % of dry weight gained between sessions. Clipped: the raw idwg column carries
        # scale errors out to +/-24 kg, which the data contract already reports as a WARN.
        p["idwg_rel"] = (100.0 * p.idwg / p.dryweight).clip(0, 12)
    if {"session_hours", "dryweight"} <= set(p.columns):
        # Ultrafiltration rate in mL/h/kg. 10 mL/h/kg is the cited threshold above which
        # intradialytic hypotension risk rises sharply, so the scale here is clinical, not
        # arbitrary. uf_total is litres, hence the x1000.
        hrs = p.session_hours.where(p.session_hours > 0)
        p["uf_rate"] = ((p.uf_total * 1000.0) / (hrs * p.dryweight)).clip(0, 30)
    if "first_dialysis_ts" in p.columns:
        # Re-parsed rather than assumed: the panel is rebuilt from the session CSV, and a
        # datetime column written to CSV comes back as a string. Subtracting it from `ts`
        # then raises deep inside pandas with a message that names neither column.
        fd = pd.to_datetime(p.first_dialysis_ts, errors="coerce")
        p["first_dialysis_ts"] = fd
        p["vintage_years"] = ((p.ts - fd).dt.days / 365.25).clip(0, 40)
    return p


def add_splits(df: pd.DataFrame, config) -> pd.DataFrame:
    """Temporal tail per patient -> val/test, plus a hash-based patient-level holdout.

    A random row split would put a patient's future readings in train and their past in
    test, which inflates every metric downstream.
    """
    out = df.copy()
    last = out.groupby("series_id")["step"].transform("max")
    test_edge = (last * (1 - config.test_frac)).round()
    val_edge = (last * (1 - config.test_frac - config.val_frac)).round()
    out["split"] = np.where(out.step > test_edge, "test",
                            np.where(out.step > val_edge, "val", "train"))
    assign = deterministic_patient_split(sorted(out.series_id.unique()),
                                         config.holdout_patient_frac,
                                         salt=config.patient_split_salt)
    out["patient_split"] = out.series_id.map(assign)
    return out


class CausalFeatureBuilder:
    """Builds strictly-causal per-session features and the forecast targets.

    Contract: for a row at step t, every feature reads only steps <= t-1, and the target
    y_<signal>_h<h> is the observed value at step t+h. Violating this contract is the single
    most consequential bug available in this pipeline, so `leakage_audit` tests it directly.
    """

    STATIC_PASSTHROUGH = ["age", "is_male", "is_dm", "is_weekend", "days_since_last"]
    # `days_since_last_raw` is the unclipped truth and must never become a model input --
    # feature_names_ below admits every numeric column not listed here, so omitting it would
    # silently grow the feature list and change every estimator's input matrix.
    # `cadence_audit` asserts this rather than trusting the comment.
    # Everything a model may not see raw. Two distinct reasons live in here:
    #   1. CONTEMPORANEOUS measurements -- they describe the session being predicted, so
    #      reading them is reading the answer. They reach a model only lagged.
    #   2. Bookkeeping and unclipped diagnostics that are not features at all.
    # `feature_names_` admits every numeric column not listed here, so a new panel column
    # that belongs in neither category silently becomes a model input. `leakage_audit` and
    # `cadence_audit` assert the consequences rather than trusting this comment.
    DROP = {"ts", "series_id", "patient_id", "gender", "step", "age_band", "split",
            "patient_split", "sbp", "dbp", "idwg", "weight", "sbp_post", "sbp_min",
            "sbp_drop", "uf_total", "temperature", "weightstart", "weightend", "dryweight",
            "n_meas", "pulse_pressure", "map_est", "DM", "keyindate", "datatime", "dow",
            "days_since_last_raw", "reading_age_days",
            # session identity and provenance, added by the sessionisation fix
            "session_start", "session_end", "session_hours", "session_source",
            "first_dialysis_ts",
            # contemporaneous intradialytic derivations from build_panel
            "nadir_sbp", "map_drop", "idwg_rel", "uf_rate",
            # Model 4's clinical overlay: contemporaneous, so lagged forms only.
            "heart_rate", "took_all_meds", "missed_antihypertensive", "sym_count",
            "sym_any", "sym_red_flag", "idh_kdoqi", "n_medications",
            # LATENT generator variables. Unobservable in any deployment, so a head that
            # reads them is scoring its own generator. Leakage probe P8 asserts this.
            "adherence_trait", "frailty"}

    #: Contemporaneous symptom indicators are dropped by prefix, because the set is defined
    #: in symptom_layer.py and listing it here would be a second copy that could drift.
    DROP_PREFIXES = ("sym_mech_",)

    def __init__(self, config):
        self.config = config
        self.feature_names_ = None

    @staticmethod
    def _rolling_slope(v: pd.Series, w: int) -> pd.Series:
        # Vectorised rolling OLS slope. x = 0..w-1 is fixed, so
        #   slope = [mean(x*y) - mean(x)*mean(y)] / var(x)
        # A .rolling().apply(np.polyfit) equivalent is ~200x slower and dominates runtime.
        y = v.astype(float)
        x_mean, x_var = (w - 1) / 2.0, (w * w - 1) / 12.0
        yv = y.values
        dot = np.full(len(yv), np.nan)
        if len(yv) >= w:
            dot[w - 1:] = np.convolve(np.nan_to_num(yv), np.arange(w, dtype=float)[::-1],
                                      mode="valid")
        return (pd.Series(dot, index=y.index) / w
                - x_mean * y.rolling(w, min_periods=w).mean()) / x_var

    def _one_series(self, g: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        g = g.sort_values("step").reset_index(drop=True).copy()
        for s in cfg.signals:
            v = g[s]
            for lag in cfg.lags:
                g[f"{s}_lag{lag}"] = v.shift(lag)
            for w in cfg.windows:
                r = v.shift(1).rolling(w, min_periods=max(2, w // 3))
                g[f"{s}_mean{w}"] = r.mean()
                g[f"{s}_std{w}"] = r.std()
                g[f"{s}_min{w}"] = r.min()
                g[f"{s}_max{w}"] = r.max()
                g[f"{s}_range{w}"] = g[f"{s}_max{w}"] - g[f"{s}_min{w}"]
                g[f"{s}_slope{w}"] = self._rolling_slope(v.shift(1), w)
            for a in cfg.ewm_alphas:
                g[f"{s}_ewm{a}"] = v.shift(1).ewm(alpha=a, adjust=False).mean()
            exp = v.shift(1).expanding(min_periods=5)
            g[f"{s}_base_mean"] = exp.mean()
            g[f"{s}_base_std"] = exp.std()
            g[f"{s}_z"] = (v.shift(1) - g[f"{s}_base_mean"]) / g[f"{s}_base_std"].replace(0, np.nan)
            g[f"{s}_d1"] = v.shift(1) - v.shift(2)

        if g.weight.notna().sum() > 3:
            g["weight_lag1"] = g.weight.shift(1)
            g["weight_slope24"] = self._rolling_slope(g.weight.shift(1), 24)
            g["weight_delta_base"] = (g.weight.shift(1)
                                      - g.weight.shift(1).expanding(min_periods=5).median())
        else:
            for c in ("weight_lag1", "weight_slope24", "weight_delta_base"):
                g[c] = np.nan

        g["gap_mean24"] = g.days_since_last.shift(1).rolling(24, min_periods=4).mean()
        g["sbp_drop_lag1"] = g.sbp_drop.shift(1)
        g["uf_lag1"] = g.uf_total.shift(1)

        # ---- pulse-pressure dynamics -------------------------------------------------
        # Widening pulse pressure is an arterial-stiffness signal that neither SBP nor DBP
        # carries on its own, and the level/trend/deviation triple mirrors what each signal
        # already gets.
        pp = (g.sbp - g.dbp).shift(1)
        g["pp_lag1"] = pp
        g["pp_mean7"] = pp.rolling(7, min_periods=3).mean()
        _pp_exp = pp.expanding(min_periods=5)
        g["pp_z"] = (pp - _pp_exp.mean()) / _pp_exp.std().replace(0, np.nan)

        # ---- contrast / ratio / residual features ------------------------------------
        # Everything above is a LEVEL, and levels on one patient's blood pressure are
        # mutually correlated at |r| > 0.95 -- sbp_mean7 and sbp_mean14 are close to the same
        # column. These are the differences and ratios between those levels: they carry the
        # part of the signal the levels share out, which is the part that moves. Feature
        # selection (feature/selection.py) collapses redundant levels into one representative
        # per cluster, so without these the reduced matrix would be almost all lags.
        for s in cfg.signals:
            m3, m7, m14 = g.get(f"{s}_mean3"), g.get(f"{s}_mean7"), g.get(f"{s}_mean14")
            s3, s14 = g.get(f"{s}_std3"), g.get(f"{s}_std14")
            if m3 is not None and m14 is not None:
                g[f"{s}_mom_3_14"] = m3 - m14            # short trend against long trend
            if m7 is not None:
                g[f"{s}_excess_base"] = m7 - g[f"{s}_base_mean"]   # recent vs personal norm
            if s3 is not None and s14 is not None:
                g[f"{s}_vol_ratio"] = s3 / s14.replace(0, np.nan)  # volatility regime change
            a_hi = max(cfg.ewm_alphas)
            if f"{s}_ewm{a_hi}" in g:
                # The reading with its own smoothed level removed: what is left is the
                # deviation, with the patient's level divided out.
                g[f"{s}_ewm_resid"] = g[s].shift(1) - g[f"{s}_ewm{a_hi}"]

        # ---- size-normalised ratios ---------------------------------------------------
        # A 2 kg gain means something different in a 50 kg patient and a 100 kg one.
        w1 = g.weight.shift(1).replace(0, np.nan)
        g["idwg_per_kg"] = g.idwg.shift(1) / w1
        g["uf_per_kg"] = g.uf_total.shift(1) / w1
        g["sbp_dbp_ratio"] = g.sbp.shift(1) / g.dbp.shift(1).replace(0, np.nan)

        # ---- interaction ---------------------------------------------------------------
        # The only explicit interaction in the design. Fluid loading and a rising pressure
        # trend are individually unremarkable and jointly the volume-overload picture; a
        # linear model cannot express that product and the trees would have to spend depth
        # rediscovering it.
        if "sbp_slope7" in g:
            g["idwg_x_sbp_slope7"] = g["idwg_lag1"] * g["sbp_slope7"]

        # Snapshot the RAW symptom columns before the blocks below start adding derived
        # ones. Reading `g.columns` afterwards would sweep up sym_any_lag1 and friends and
        # derive lags of lags -- sym_count_mean30_rate30, which is noise with a long name.
        raw_sym_cols = [c for c in g.columns
                        if c.startswith("sym_") and not c.endswith(
                            ("_lag1", "_rate30", "_mean7", "_mean30"))]

        # ---- clinical-layer history (Model 4) -------------------------------------
        # Present only when the synthetic symptom overlay has been merged onto the panel.
        # Every one of these reads shift(1) or later: the contemporaneous columns are in
        # DROP and reach a model only through these lagged forms.
        for c in ("heart_rate", "took_all_meds", "missed_antihypertensive", "uf_rate",
                  "idwg_rel", "sym_any", "sym_count"):
            if c not in g.columns:
                continue
            v = g[c].shift(1)
            g[f"{c}_lag1"] = v
            g[f"{c}_mean7"] = v.rolling(7, min_periods=2).mean()
            g[f"{c}_mean30"] = v.rolling(30, min_periods=5).mean()

        if "heart_rate" in g.columns:
            hr = g.heart_rate.shift(1)
            hx = hr.expanding(min_periods=5)
            g["hr_z"] = (hr - hx.mean()) / hx.std().replace(0, np.nan)
            # Shock index: HR over SBP. Rises when the heart is compensating for a falling
            # pressure, which neither signal shows on its own.
            g["shock_index_lag1"] = hr / g.sbp.shift(1).replace(0, np.nan)
            g["hr_d1"] = hr - g.heart_rate.shift(2)

        # THE ONE DELIBERATE EXCEPTION to "features read <= t-1". Same-day adherence really
        # is known at the moment the forecast is made, because the forecast is for the NEXT
        # session. Leakage probe P7 asserts it equals today and is not a proxy for tomorrow.
        for c in ("took_all_meds", "missed_antihypertensive"):
            if c in g.columns:
                g[f"adh_{c}_today"] = g[c]

        # ---- symptom history and targets ------------------------------------------
        sym_cols = raw_sym_cols
        for c in sym_cols:
            v = g[c].shift(1)
            g[f"{c}_lag1"] = v
            g[f"{c}_rate30"] = v.rolling(30, min_periods=5).mean()
        for h in cfg.horizons:
            for c in sym_cols:
                g[f"y_{c}_h{h}"] = g[c].shift(-h)

        for h in cfg.horizons:
            for s in cfg.signals:
                g[f"y_{s}_h{h}"] = g[s].shift(-h)
        # ~180 successive column insertions leave the block manager badly fragmented, and
        # this runs once per patient. One copy consolidates it and pandas stops warning.
        return g.copy()

    #: Non-feature columns the feature frame must still carry, because something downstream
    #: keys, splits, groups or audits on them. Everything else that is not a feature or a
    #: target is dropped: `run_sweep` alone takes a dozen boolean-mask copies of this frame,
    #: and an object-dtype column costs a pointer per row on every one of them. Carrying
    #: `session_start`, `session_end`, `session_source` and `first_dialysis_ts` through was
    #: enough to exhaust memory on a 116k-row panel.
    KEEP_NON_FEATURE = {
        "ts", "series_id", "patient_id", "step", "split", "patient_split",
        "sbp", "dbp", "idwg", "sbp_drop", "age", "age_band", "is_male", "is_dm",
        "days_since_last", "days_since_last_raw", "n_meas",
        # Raw signals `_one_series` reads. They are not features (they are contemporaneous
        # and sit in DROP), but keeping them makes the frame SELF-DESCRIBING: the leakage
        # audit rebuilds features from it to run the future-corruption probe, and without
        # these that probe degrades to a warning instead of a check.
        "weight", "uf_total", "dryweight",
    }

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        out = pd.concat([self._one_series(g) for _, g in panel.groupby("series_id", sort=False)],
                        ignore_index=True)
        for c in out.select_dtypes("float64").columns:
            out[c] = out[c].astype("float32")   # halves memory; mmHg/kg lose no usable precision
        def _is_raw_symptom(c):
            # `sym_x` is the contemporaneous label; `sym_x_lag1` / `sym_x_rate30` are its
            # legal lagged forms. Dropping by prefix alone would remove both.
            return (c.startswith("sym_") and not c.endswith(("_lag1", "_rate30", "_mean7",
                                                             "_mean30")))
        self.feature_names_ = [c for c in out.columns
                               if c not in self.DROP and not c.startswith("y_")
                               and not c.startswith(self.DROP_PREFIXES)
                               and not _is_raw_symptom(c)
                               and pd.api.types.is_numeric_dtype(out[c])]
        keep = [c for c in out.columns
                if c in self.feature_names_ or c.startswith("y_")
                or c in self.KEEP_NON_FEATURE]
        # Deliberately NOT converted to `category`, tempting as it is on 116k repeated
        # strings: pandas 2.x groupby defaults to observed=False, so a categorical
        # `series_id` would yield an empty group for every patient absent from a subset,
        # and the per-patient models iterate exactly those subsets.
        return out[keep].copy()

    def transform_for_inference(self, history: pd.DataFrame,
                                as_of: pd.Timestamp = None) -> pd.DataFrame:
        """Feature row describing 'now' for one patient.

        Training pairs features at step t (which see <= t-1) with the reading at t+h. Applied
        naively at serving time that discards the newest reading. Appending one placeholder step
        puts the newest reading inside the .shift(1) window -- the exact training distribution --
        and makes horizon h mean h steps ahead of now. Getting this wrong shifts every forecast
        by one step and is invisible in offline metrics.

        `as_of` is the moment the advisory is being asked for. Without it the placeholder is
        parked one MEDIAN gap after the last reading, which silently asserts the patient is
        up to date: someone who last logged 30 days ago got a forecast built as though they
        had just logged. With it, the placeholder sits at the real "now", so the cadence
        features carry the true staleness. `as_of=None` reproduces the old behaviour exactly,
        which is what every offline caller relies on.
        """
        g = history.sort_values("ts").copy()
        med_gap = float(g.ts.diff().dt.days.median() or 2)
        ph = g.iloc[[-1]].copy()
        nominal = g.ts.max() + pd.Timedelta(days=med_gap)
        ph["ts"] = nominal if as_of is None else max(pd.Timestamp(as_of),
                                                     g.ts.max() + pd.Timedelta(days=1))
        for c in ("sbp", "dbp", "idwg", "weight", "sbp_drop", "uf_total"):
            if c in ph:
                ph[c] = np.nan
        gg = pd.concat([g, ph], ignore_index=True)
        gg["series_id"] = str(g.patient_id.iloc[0])
        gg = attach_cadence(gg, by=None)
        gg["step"] = np.arange(len(gg))
        gg["is_weekend"] = (gg.ts.dt.dayofweek >= 5).astype(int)
        for c, d in (("DM", 0), ("is_dm", 0), ("is_male", 0), ("age", 65.0)):
            if c not in gg:
                gg[c] = d
        return self._one_series(gg).iloc[[-1]]


def feature_group(name: str) -> str:
    """Human-readable family for a feature name, used by the feature dictionary.

    Also the fallback block for permutation importance when no redundancy clustering is
    available, so a feature landing in "other" is not cosmetic -- it gets permuted with an
    unrelated set and its importance is attributed to the wrong thing.
    """
    if name.startswith(("sbp_", "dbp_", "idwg_")):
        base = name.split("_", 1)[1]
        # Order matters: the contrast families are suffixes on the same prefixes as the
        # level families, so they must be tested before the level prefixes below.
        if base.startswith("mom_"):
            return "momentum contrast"
        if base == "excess_base":
            return "baseline excess"
        if base == "vol_ratio":
            return "volatility ratio"
        if base == "ewm_resid":
            return "ewma residual"
        if base.startswith("x_"):
            return "interaction"
        if base.startswith("per_kg"):
            return "size-normalised"
        if base.startswith("lag"):
            return "lag"
        if base.startswith(("mean", "std", "min", "max", "range")):
            return "rolling moment"
        if base.startswith("slope"):
            return "trend"
        if base.startswith("ewm"):
            return "smoother"
        if base.startswith("base"):
            return "personal baseline"
        if base == "z":
            return "personal z-score"
        if base == "d1":
            return "first difference"
        if base == "dbp_ratio":
            return "size-normalised"
    if name.startswith("pp_"):
        return "pulse-pressure dynamics"
    if name in ("idwg_per_kg", "uf_per_kg", "sbp_dbp_ratio"):
        return "size-normalised"
    if name.startswith("idwg_x_") or "_x_" in name:
        return "interaction"
    if name.startswith("weight"):
        return "fluid / weight"
    if name.startswith("uf_") or name.startswith("uf_rate"):
        return "ultrafiltration"
    if name.startswith("sbp_drop") or name.startswith("map_drop") or name.startswith("nadir"):
        return "intradialytic"
    if name.startswith("hr_") or name.startswith("heart_rate") or name == "shock_index_lag1":
        return "heart rate"
    if name.startswith("sym_"):
        return "symptom history"
    if name.startswith("med_") or name.startswith("on_"):
        return "medication class"
    if name.startswith("adh_") or "adherence" in name or "took_all_meds" in name \
            or "missed_antihypertensive" in name:
        return "adherence"
    if name.startswith("has_") or name.startswith("dx_") or name.startswith("clin_"):
        return "diagnosed condition"
    if name in ("age", "is_male", "is_dm", "vintage_years"):
        return "static"
    if name in ("days_since_last", "gap_mean24", "is_weekend"):
        return "cadence"
    return "other"


def feature_dictionary(F: pd.DataFrame, features: list) -> pd.DataFrame:
    """Feature name -> group, signal and observed density."""
    d = pd.DataFrame({"feature": features})
    d["group"] = d.feature.map(feature_group)
    d["signal"] = d.feature.str.split("_").str[0].where(
        d.feature.str.startswith(("sbp", "dbp", "idwg", "weight")), "-")
    d["density"] = d.feature.map(F[features].notna().mean())
    return d


def leakage_audit(F: pd.DataFrame, features: list, config, fb=None) -> pd.DataFrame:
    """Structural probes over the feature matrix and the split layout.

    Deliberately not a correlation screen: on a 3x/week dialysis cadence a lag-2 feature
    lands on the same weekday as the current reading, so correlation flags the cadence
    artefact and calls it leakage. These probes test the contract directly instead.
    """
    rows = []

    def add(name, ok, detail=""):
        rows.append(dict(probe=name, status="PASS" if ok else "FAIL", detail=detail))

    corr = F[features].corrwith(F.sbp).abs().sort_values(ascending=False)
    add("no feature is a copy of the current reading", corr.max() < 0.999,
        f"max |corr| {corr.max():.4f} ({corr.index[0]})")

    probe = F.sort_values(["series_id", "step"]).groupby("series_id").head(80)
    exp = probe.groupby("series_id").sbp.shift(1)
    m = exp.notna()
    add("sbp_lag1 == previous row's SBP, exactly",
        bool(np.allclose(exp[m].values, probe.loc[m, "sbp_lag1"].values, atol=1e-3)))

    add("no rolling feature precedes its first lag",
        int((F.sbp_mean3.notna() & F.sbp_lag1.isna()).sum()) == 0)

    bad = []
    for sid, g in F.groupby("series_id"):
        mx, mn = g.groupby("split").step.max(), g.groupby("split").step.min()
        if {"train", "val"} <= set(mx.index) and mx["train"] >= mn["val"]:
            bad.append(sid)
        elif {"val", "test"} <= set(mx.index) and mx["val"] >= mn["test"]:
            bad.append(sid)
    add("train precedes val precedes test within every patient", not bad,
        f"{len(bad)} patients out of order")

    add("patient split is disjoint",
        int((F.groupby("series_id").patient_split.nunique() > 1).sum()) == 0)

    inf_cols = [c for c in features if np.isinf(F[c].fillna(0)).any()]
    add("no infinite values in the feature matrix", not inf_cols, f"{inf_cols[:5]}")

    for h in config.horizons:
        cov = float(F[f"y_sbp_h{h}"].notna().mean())
        add(f"target y_sbp_h{h} present", cov > 0.5, f"{cov:.1%} of rows")

    sparse = [c for c in features if F[c].notna().mean() < 0.30]
    rows.append(dict(probe="every feature non-null on >30% of rows",
                     status="PASS" if not sparse else "WARN", detail=f"{len(sparse)} sparse"))

    # ---- semantic probes ------------------------------------------------------------
    # The probes above are STRUCTURAL: they check shapes, orderings and coverage. These are
    # semantic -- they corrupt the data in a way that would only matter if the causal
    # contract were broken, and check that nothing moves. A structural probe cannot catch a
    # feature that reads the future; only this kind can.

    # The builder that PRODUCED F, when the caller has it. The probe rebuilds features and
    # compares, so using a fresh canonical builder here would test the canonical builder
    # rather than the one in use -- and a subclass that reads forward would sail through.
    fb = fb if fb is not None else CausalFeatureBuilder(config)
    probe_sid = F.series_id.iloc[0]

    # P2 -- the strongest causality test available, and the one with no structural analogue.
    # Rebuild features for a patient whose FUTURE readings have been corrupted by +50 mmHg.
    # Every feature row before the corruption point must come back bit-identical. If any
    # feature peeks forward -- a centred window, a fillna(method="bfill"), an expanding stat
    # applied without a shift -- this is what catches it, and nothing else will.
    try:
        g = F[F.series_id == probe_sid].sort_values("step")
        base_cols = [c for c in ("patient_id", "series_id", "ts", "step", "sbp", "dbp",
                                 "idwg", "weight", "sbp_drop", "uf_total", "age", "is_male",
                                 "is_dm", "DM", "days_since_last", "is_weekend", "n_meas")
                     if c in g.columns]
        g0 = g[base_cols].reset_index(drop=True)
        mid = len(g0) // 2
        if mid >= 10:
            g1 = g0.copy()
            g1.loc[g1.index >= mid, "sbp"] = g1.loc[g1.index >= mid, "sbp"] + 50.0
            cols = [c for c in features if c in fb.transform(g0).columns]
            a = fb.transform(g0).iloc[:mid][cols].to_numpy(dtype=float)
            b = fb.transform(g1).iloc[:mid][cols].to_numpy(dtype=float)
            same = bool(np.allclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-6))
            bad_cols = []
            if not same:
                d = ~np.isclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-6)
                bad_cols = [cols[j] for j in np.where(d.any(axis=0))[0]][:8]
            add("corrupting the future leaves past features unchanged", same,
                f"+50 mmHg after step {mid}; {len(bad_cols)} features moved {bad_cols}")
        else:
            rows.append(dict(probe="corrupting the future leaves past features unchanged",
                             status="WARN", detail="probe patient too short"))
    except Exception as exc:                                          # noqa: BLE001
        rows.append(dict(probe="corrupting the future leaves past features unchanged",
                         status="WARN", detail=f"probe unavailable: {exc}"))

    # P3 -- the target really is the reading h steps later, not h-1 or h+1. An off-by-one
    # here shifts every reported horizon and is invisible in the metrics, because a model
    # trained and scored against the same wrong target looks perfectly consistent.
    for h in config.horizons:
        tgt = f"y_sbp_h{h}"
        if tgt not in F.columns:
            continue
        left = F[["series_id", "step", tgt]].copy()
        left["step"] = left.step + h
        j = left.merge(F[["series_id", "step", "sbp"]], on=["series_id", "step"], how="inner")
        j = j.dropna(subset=[tgt, "sbp"])
        ok = bool(len(j) and np.allclose(j[tgt].to_numpy(float), j.sbp.to_numpy(float),
                                         atol=1e-3))
        add(f"target y_sbp_h{h} equals the reading {h} steps later", ok,
            f"checked {len(j):,} aligned rows")

    # P4 -- a model fitted on SHUFFLED targets must not beat persistence. If it does, the
    # feature matrix carries the answer through a route the other probes do not cover.
    try:
        from sklearn.metrics import mean_absolute_error

        from src.utils.ml_utils.model.architectures import make_model
        tgt = f"y_sbp_h{config.horizons[0]}"
        d = F[(F.split == "train") & F[tgt].notna()]
        if len(d) > 4000:
            d = d.sample(4000, random_state=42)
        te = F[(F.split == "test") & F[tgt].notna()]
        if len(te) > 2000:
            te = te.sample(2000, random_state=42)
        if len(d) <= 500 or len(te) <= 200:
            # Not enough rows to fit and score a probe model. Recorded rather than skipped:
            # a probe that quietly does nothing is indistinguishable from a probe that passed.
            rows.append(dict(probe="a shuffled-target model does not beat persistence",
                             status="WARN",
                             detail=f"not evaluated: {len(d)} train / {len(te)} test rows"))
        else:
            y = d[tgt].to_numpy(float).copy()
            np.random.default_rng(42).shuffle(y)
            m = make_model("ridge").fit(d[features], y)
            mae_shuf = mean_absolute_error(te[tgt], m.predict(te[features]))
            pers = te["sbp_lag1"].to_numpy(float)
            msk = np.isfinite(pers)
            mae_pers = mean_absolute_error(te[tgt].to_numpy(float)[msk], pers[msk])
            add("a shuffled-target model does not beat persistence", mae_shuf > mae_pers,
                f"shuffled {mae_shuf:.2f} vs persistence {mae_pers:.2f} mmHg")
    except Exception as exc:                                          # noqa: BLE001
        rows.append(dict(probe="a shuffled-target model does not beat persistence",
                         status="WARN", detail=f"probe unavailable: {exc}"))

    # P6 -- no symptom feature reproduces its own label. Only meaningful once Model 4's
    # columns exist; recorded as not-applicable rather than silently skipped.
    sym_feats = [c for c in features if c.startswith("sym_")]
    if not sym_feats:
        rows.append(dict(probe="no symptom feature reproduces its own label",
                         status="PASS",
                         detail="not applicable: this bundle has no symptom features"))
    else:
        worst, worst_c = 0.0, ""
        for c in sym_feats:
            raw = c
            for suf in ("_rate30", "_mean30", "_mean7", "_lag1"):
                if raw.endswith(suf):
                    raw = raw[: -len(suf)]
                    break
            if raw != c and raw in F.columns:
                r = abs(float(F[c].corr(F[raw])))
                if np.isfinite(r) and r > worst:
                    worst, worst_c = r, c
        add("no symptom feature reproduces its own label", worst < 0.95,
            f"max |corr| {worst:.3f} ({worst_c})")

    # P7 -- the one deliberate exception. Same-day adherence IS known when the forecast is
    # made, because the prediction is for the NEXT session. It must equal today and must NOT
    # be a proxy for tomorrow.
    adh_feats = [c for c in features if c.startswith("adh_") and c.endswith("_today")]
    if not adh_feats:
        rows.append(dict(probe="adh_*_today reads today, not tomorrow", status="PASS",
                         detail="not applicable: no same-day adherence features"))
    for c in adh_feats:
        raw = c[len("adh_"):-len("_today")]
        if raw not in F.columns:
            continue
        eq = bool(np.allclose(F[c].fillna(-1), F[raw].fillna(-1), atol=1e-6))
        fwd = abs(float(F[c].corr(F.groupby("series_id")[raw].shift(-1))))
        add(f"{c} reads today, not tomorrow", eq and (not np.isfinite(fwd) or fwd < 0.95),
            f"equals today={eq}, |corr| with tomorrow={fwd:.3f}")

    # P8 -- the latent generator variables must never reach a model. They are unobservable
    # in deployment, so a symptom head that reads them scores its own generator.
    latent = {"adherence_trait", "frailty"} & set(features)
    add("latent generator variables are not features", not latent, f"{sorted(latent)}")

    return pd.DataFrame(rows)
