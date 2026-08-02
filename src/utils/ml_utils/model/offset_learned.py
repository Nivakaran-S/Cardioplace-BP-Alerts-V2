"""Model 2, learned: a supervised personalised alert threshold.

`offset.py` holds the hand-built shrinkage blend -- a patient's own q0.90 pulled toward an
age x sex cohort prior. It works, and it is the reference every candidate here has to beat.
But it reads exactly three inputs: SBP history, age, sex. The notebook this is ported from
names that as a bug it fixed:

    "Whether a patient carries heart failure, takes a rate-limiting agent, or routinely
     misses antihypertensives is exactly the information that should move that threshold -
     and the first version of this model ignored all of it, keying only on age, sex and
     cohort."

So this module learns the threshold instead, from 37 features spanning the patient's own
band, their recent trajectory, and their clinical context.

Four things make this different from an ordinary regression, and all four are load-bearing:

1. THE TARGET IS A QUANTILE, not a mean. `y` is the q0.90 of the patient's NEXT 30 sessions:
   "where will this patient's high readings sit". Half the candidates optimise pinball loss
   directly; the mean-native ones are corrected in (3).

2. MULTI-CUT EXPANSION. Each patient contributes a row at each of several history lengths
   (20, 32, 48, 72, 100, 140 sessions). One row per patient would give ~150 training rows
   for a 37-feature model. It also forces the model to work at every history length it will
   actually meet at serving time, including short ones.

3. CONFORMAL CALIBRATION on a disjoint pool. A mean-native learner predicts the middle of
   the next 30 sessions, not the q0.90, so it is shifted by the q0.90 of its own residuals
   measured on patients it never trained on. Quantile-native learners already target q0.90
   and get no shift.

4. A FOUR-WAY PATIENT-DISJOINT SPLIT: train / calib / select / holdout. Calibrating the
   conformal shift and selecting the winner on the same rows would make both optimistic;
   reporting on either would make the reported number the one that was optimised.

Every prediction passes through `apply_caps` before it is scored, so the governance caps
bind the learned model exactly as they bind the blend, and the emergency floor is asserted
over the learned predictions rather than inherited from the blend's assertion.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    LinearRegression,
    QuantileRegressor,
    Ridge,
)
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants.training_pipeline import OFFSET_CUTS, OFFSET_FUTURE, OFFSET_TARGET_Q, SEED
from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.utils.main_utils.utils import stable_bucket
from src.utils.ml_utils.metric.regression_metric import paired_delta
from src.utils.ml_utils.model.offset import cohort_key

# Clinical context columns read from history. Absent on the HEMOBP corpus -- it carries no
# medication, condition or adherence data -- so they are zero-filled and the model learns to
# ignore them here. They are present at SERVING time, where the user supplies them, which is
# the whole reason the feature block exists: the same fitted model then has somewhere to put
# "this patient is on a rate-limiting agent".
CONDITION_COLS = ("has_htn", "has_ihd", "has_hf", "has_afib", "has_tachy", "has_brady",
                  "has_cad")
MED_COLS = ("on_ace", "on_arb", "on_bb", "on_ndhp_ccb", "on_loop", "on_thiazide",
            "on_mra", "on_nitrate", "on_antiarr", "on_nsaid")


def apply_caps(pred, config) -> np.ndarray:
    """Clamp any threshold into the governance band around the population threshold.

    Extracted as a free function rather than left inlined in `OffsetModel.threshold_for`,
    because every learned candidate has to be scored under the same caps the blend obeys.
    Scoring an uncapped prediction and capping it only at serving time would rank the
    candidates on numbers no patient ever sees.
    """
    p = np.asarray(pred, dtype=float)
    off = np.clip(p - config.population_threshold_mmHg,
                  -config.offset_cap_tighten, config.offset_cap_loosen)
    return config.population_threshold_mmHg + off


def pinball(y, p, q: float = OFFSET_TARGET_Q) -> float:
    """Quantile (pinball) loss -- the correct loss for a q0.90 target.

    MAE would reward a model that predicts the MEDIAN of the next 30 sessions, which is a
    different and much easier quantity than the one the threshold needs.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return np.nan
    d = y[m] - p[m]
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


# --------------------------------------------------------------------------- features

def clinical_context(hist: pd.DataFrame) -> dict:
    """The clinical block, read only from history strictly before the cut."""
    out = {}
    for c in CONDITION_COLS + MED_COLS:
        out[f"clin_{c}"] = (float(hist[c].max()) if c in hist.columns and len(hist)
                            else 0.0)
    out["clin_n_medications"] = sum(out[f"clin_{c}"] for c in MED_COLS)
    out["clin_n_antihypertensive"] = sum(
        out[f"clin_{c}"] for c in ("on_ace", "on_arb", "on_bb", "on_ndhp_ccb",
                                   "on_loop", "on_thiazide", "on_mra"))
    # Last-30-session adherence. Defaults to 1.0 (fully adherent) rather than 0.0, because
    # absent adherence data means "not recorded", and assuming total non-adherence would
    # loosen every threshold for every patient on a corpus that simply has no such column.
    out["clin_adherence"] = (float(hist["took_all_meds"].dropna().tail(30).mean())
                             if "took_all_meds" in hist.columns
                             and hist["took_all_meds"].notna().any() else 1.0)
    return out


def offset_features(sbp_history, age: float, is_male: int, cohort: float,
                    hist: pd.DataFrame = None) -> dict:
    """The 37-column feature vector. ORDER IS PART OF THE CONTRACT.

    `EmpiricalBayesBlend` addresses `hist_q90` and `cohort_prior` positionally, so reordering
    this dict silently rewires that estimator to read the wrong columns. `OFFSET_FEATURES`
    is derived from this function and asserted against the two indices at import.
    """
    s = pd.Series(sbp_history, dtype=float).dropna()
    n = int(len(s))
    q = s.quantile([.25, .5, .75, .90]) if n else pd.Series(
        {.25: np.nan, .5: np.nan, .75: np.nan, .90: np.nan})
    recent = s.tail(10)
    t20 = s.tail(20)
    slope = np.nan
    if len(t20) >= 5:
        slope = float(np.polyfit(np.arange(len(t20)), t20.to_numpy(float), 1)[0])
    out = {
        "hist_n": float(n),
        "hist_mean": float(s.mean()) if n else np.nan,
        "hist_sd": float(s.std()) if n > 1 else np.nan,
        "hist_q25": float(q[.25]), "hist_q50": float(q[.5]),
        "hist_q75": float(q[.75]), "hist_q90": float(q[.90]),
        "hist_iqr": float(q[.75] - q[.25]),
        "hist_min": float(s.min()) if n else np.nan,
        "hist_max": float(s.max()) if n else np.nan,
        "recent_mean10": float(recent.mean()) if len(recent) else np.nan,
        "recent_sd10": float(recent.std()) if len(recent) > 1 else np.nan,
        "recent_slope20": slope,
        "level_vs_cohort": float(q[.90] - cohort) if n else np.nan,
        "age": float(age), "is_male": float(is_male), "cohort_prior": float(cohort),
    }
    out.update(clinical_context(hist if hist is not None else pd.DataFrame()))
    return out


OFFSET_FEATURES = list(offset_features([120, 130, 125], 65, 1, 140.0).keys())
_IX = {c: i for i, c in enumerate(OFFSET_FEATURES)}
assert _IX["hist_q90"] == 6 and _IX["cohort_prior"] == 16, (
    "EmpiricalBayesBlend reads hist_q90 and cohort_prior positionally; offset_features "
    "changed shape and would silently feed it the wrong columns")


def build_offset_frame(F: pd.DataFrame, config, cohort_prior: dict,
                       global_prior: float) -> pd.DataFrame:
    """One row per (patient, history cut): features from the past, target from the future."""
    rows = []
    for sid, g in F.sort_values(["series_id", "step"]).groupby("series_id", sort=False):
        s = g.sbp.reset_index(drop=True)
        age = float(g.age.fillna(65).iloc[0])
        male = int(g.is_male.iloc[0])
        ck = cohort_key(age, male)
        cohort = float(cohort_prior.get(ck, global_prior))
        for c in OFFSET_CUTS:
            if len(s) < c + 10:
                continue
            past, fut = s.iloc[:c], s.iloc[c:c + OFFSET_FUTURE]
            if past.notna().sum() < 5 or fut.notna().sum() < 8:
                continue
            feats = offset_features(past, age, male, cohort, hist=g.iloc[:c])
            rows.append(dict(series_id=sid, cut=c,
                             patient_split=g.patient_split.iloc[0],
                             is_dm=int(g.is_dm.iloc[0]), cohort_key=ck,
                             y=float(fut.quantile(OFFSET_TARGET_Q)), **feats))
    return pd.DataFrame(rows)


def add_offset_split(OFR: pd.DataFrame, salt: str = "offset-split") -> pd.DataFrame:
    """Patient-disjoint train / calib / select, with the global holdout preserved.

    Salted independently of the main patient split so the two are uncorrelated: reusing the
    same salt would make `calib` a deterministic subset of the main `fit` pool and couple
    two decisions that are supposed to be independent.
    """
    out = OFR.copy()
    h = out.series_id.map(lambda s: stable_bucket(str(s), salt))
    out["offset_split"] = np.where(
        out.patient_split == "holdout", "holdout",
        np.where(h < 20, "calib", np.where(h < 40, "select", "train")))
    return out


# ------------------------------------------------------------------ custom estimators

def _lin(m):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("model", m)])


class QuantileForest(BaseEstimator, RegressorMixin):
    """A random forest read as a distribution rather than as a mean.

    Every tree gives one estimate; the standard forest averages them. Taking the q0.90
    ACROSS the trees instead turns the same fitted object into a quantile estimator, which
    is what a threshold needs. Costs nothing extra to fit.
    """

    def __init__(self, base=None, q: float = OFFSET_TARGET_Q):
        self.base = base
        self.q = q

    def fit(self, X, y):
        self.base_ = clone(self.base or RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED))
        Xi = np.nan_to_num(np.asarray(X, float), nan=0.0)
        self.base_.fit(Xi, y)
        return self

    def predict(self, X):
        Xi = np.nan_to_num(np.asarray(X, float), nan=0.0)
        P = np.column_stack([t.predict(Xi) for t in self.base_.estimators_])
        return np.quantile(P, self.q, axis=1)


class KNNQuantile(BaseEstimator, RegressorMixin):
    """q0.90 of the targets of the k most similar patients.

    Non-parametric and local: it makes no assumption that the threshold is a smooth function
    of the features, only that similar patients have similar high readings.
    """

    def __init__(self, k: int = 25, q: float = OFFSET_TARGET_Q):
        self.k = k
        self.q = q

    def _z(self, X):
        return (np.nan_to_num(np.asarray(X, float), nan=0.0) - self.mu_) / (self.sd_ + 1e-9)

    def fit(self, X, y):
        Xa = np.nan_to_num(np.asarray(X, float), nan=0.0)
        self.mu_, self.sd_ = Xa.mean(axis=0), Xa.std(axis=0)
        self.y_ = np.asarray(y, float)
        self.nn_ = NearestNeighbors(n_neighbors=min(self.k, len(self.y_))).fit(self._z(Xa))
        return self

    def predict(self, X):
        _, idx = self.nn_.kneighbors(self._z(X))
        return np.array([np.quantile(self.y_[i], self.q) for i in idx])


class EmpiricalBayesBlend(BaseEstimator, RegressorMixin):
    """The shrinkage blend with `k` LEARNED instead of grid-picked.

    This is the direct upgrade of `offset.OffsetModel`. That model shrinks a patient's own
    q0.90 toward the cohort prior with weight n/(n+k), and picks `k` from a three-point grid.
    Here `k` is estimated as the variance ratio it is supposed to be -- sigma^2 (noise in a
    patient's own estimate) over tau^2 (real spread between patients) -- which is the
    textbook empirical-Bayes shrinkage constant, plus a learned bias term.
    """

    def __init__(self, n_col=0, personal_col=6, cohort_col=16):
        self.n_col = n_col
        self.personal_col = personal_col
        self.cohort_col = cohort_col

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        personal, cohort = X[:, self.personal_col], X[:, self.cohort_col]
        m = np.isfinite(personal) & np.isfinite(cohort) & np.isfinite(y)
        tau2 = float(np.var(personal[m] - cohort[m], ddof=1)) if m.sum() > 1 else 1.0
        sigma2 = float(np.var(y[m] - personal[m], ddof=1)) if m.sum() > 1 else 1.0
        self.k_ = float(np.clip(sigma2 / max(tau2, 1e-9), 1.0, 400.0))
        self.bias_ = float(np.median(y[m] - personal[m])) if m.any() else 0.0
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        n = np.nan_to_num(X[:, self.n_col], nan=0.0)
        personal = X[:, self.personal_col]
        cohort = X[:, self.cohort_col]
        personal = np.where(np.isfinite(personal), personal, cohort)
        w = n / (n + self.k_)
        return w * personal + (1 - w) * cohort + self.bias_


def offset_candidates() -> dict:
    """name -> (estimator, native, cost). `native` decides whether a conformal shift applies."""
    return {
        "linear regression":     (_lin(LinearRegression()), "mean", "low"),
        "ridge":                 (_lin(Ridge(alpha=10.0)), "mean", "low"),
        "elastic net":           (_lin(ElasticNet(alpha=.05, l1_ratio=.5,
                                                  random_state=SEED)), "mean", "low"),
        "huber (robust)":        (_lin(HuberRegressor(epsilon=1.35, max_iter=500)),
                                  "mean", "low"),
        "kNN (mean)":            (_lin(KNeighborsRegressor(n_neighbors=25,
                                                           weights="distance")),
                                  "mean", "low"),
        "MLP (2x32)":            (_lin(MLPRegressor(hidden_layer_sizes=(32, 32),
                                                    max_iter=800, early_stopping=True,
                                                    random_state=SEED)), "mean", "medium"),
        "random forest (mean)":  (Pipeline([("impute", SimpleImputer(strategy="median")),
                                            ("model", RandomForestRegressor(
                                                n_estimators=300, min_samples_leaf=5,
                                                n_jobs=-1, random_state=SEED))]),
                                  "mean", "medium"),
        "extra trees (mean)":    (Pipeline([("impute", SimpleImputer(strategy="median")),
                                            ("model", ExtraTreesRegressor(
                                                n_estimators=300, min_samples_leaf=5,
                                                n_jobs=-1, random_state=SEED))]),
                                  "mean", "medium"),
        "GBM (pinball q90)":     (Pipeline([("impute", SimpleImputer(strategy="median")),
                                            ("model", GradientBoostingRegressor(
                                                loss="quantile", alpha=OFFSET_TARGET_Q,
                                                n_estimators=300, learning_rate=.05,
                                                random_state=SEED))]),
                                  "quantile", "medium"),
        "quantile forest":       (QuantileForest(), "quantile", "medium"),
        "kNN quantile":          (KNNQuantile(k=25), "quantile", "low"),
        "linear quantile":       (_lin(QuantileRegressor(quantile=OFFSET_TARGET_Q,
                                                         alpha=1e-3, solver="highs")),
                                  "quantile", "low"),
        "empirical-Bayes blend": (EmpiricalBayesBlend(), "quantile", "low"),
    }


def train_learned_offset(F: pd.DataFrame, config, offset_model, seed: int = SEED):
    """Fit, calibrate, select and decide. Returns (board, best, artifacts).

    `best` is None when no learned candidate beat the blend on a paired test -- which is a
    real outcome and the reason the blend stays in the bundle as the reference.
    """
    try:
        OFR = build_offset_frame(F, config, offset_model.cohort_prior,
                                 offset_model.global_prior)
        if len(OFR) < 200:
            logging.warning("learned offset: only %d training rows; keeping the blend",
                            len(OFR))
            return pd.DataFrame(), None, {}
        OFR = add_offset_split(OFR)
        counts = OFR.offset_split.value_counts().to_dict()
        logging.info("learned offset frame: %d rows over %d patients | %s",
                     len(OFR), OFR.series_id.nunique(), counts)

        tr = OFR[OFR.offset_split == "train"]
        ca = OFR[OFR.offset_split == "calib"]
        se = OFR[OFR.offset_split == "select"]
        ho = OFR[OFR.offset_split == "holdout"]
        if min(len(tr), len(ca), len(se)) < 30:
            logging.warning("learned offset: a split is too thin (%s); keeping the blend",
                            counts)
            return pd.DataFrame(), None, {}

        Xtr, ytr = tr[OFFSET_FEATURES].to_numpy(float), tr.y.to_numpy(float)
        rows, fitted = [], {}
        for name, (est, native, cost) in offset_candidates().items():
            try:
                m = clone(est).fit(Xtr, ytr)
                # Mean-native learners predict the middle of the next 30 sessions; the
                # target is its q0.90. The shift is the q0.90 of their own residuals,
                # measured on patients they never saw. Quantile-native learners already
                # aim at q0.90, so shifting them again would double-count.
                shift = 0.0
                if native == "mean":
                    resid = ca.y.to_numpy(float) - m.predict(ca[OFFSET_FEATURES].to_numpy(float))
                    resid = resid[np.isfinite(resid)]
                    shift = float(np.quantile(resid, OFFSET_TARGET_Q)) if len(resid) else 0.0

                r = dict(model=name, family="learned", native=native, cost=cost,
                         conformal_shift=round(shift, 3))
                for label, part in (("select", se), ("holdout", ho)):
                    if len(part) < 20:
                        continue
                    p = apply_caps(m.predict(part[OFFSET_FEATURES].to_numpy(float)) + shift,
                                   config)
                    y = part.y.to_numpy(float)
                    r[f"{label}_pinball"] = round(pinball(y, p), 4)
                    r[f"{label}_MAE"] = round(float(mean_absolute_error(y, p)), 3)
                    # coverage is the metric that says whether a q0.90 threshold IS a
                    # q0.90 threshold. A model can have excellent MAE and cover 50%.
                    r[f"{label}_coverage"] = round(float(np.mean(y <= p)), 3)
                    r[f"{label}_within10"] = round(float(np.mean(np.abs(y - p) <= 10)), 3)
                rows.append(r)
                fitted[name] = (m, shift, native)
            except Exception as exc:                                  # noqa: BLE001
                logging.warning("offset candidate %s failed: %s", name, exc)

        # References, scored on exactly the same rows and under the same caps.
        for name, col in (("capped blend (reference)", None),
                          ("personal q90 only", "hist_q90"),
                          ("cohort prior only", "cohort_prior")):
            r = dict(model=name, family="reference", native="-", cost="low",
                     conformal_shift=0.0)
            for label, part in (("select", se), ("holdout", ho)):
                if len(part) < 20:
                    continue
                if col is None:
                    w = part.hist_n / (part.hist_n + offset_model.k)
                    raw = (w * part.hist_q90.fillna(part.cohort_prior)
                           + (1 - w) * part.cohort_prior).to_numpy(float)
                else:
                    raw = part[col].fillna(part.cohort_prior).to_numpy(float)
                p = apply_caps(raw, config)
                y = part.y.to_numpy(float)
                r[f"{label}_pinball"] = round(pinball(y, p), 4)
                r[f"{label}_MAE"] = round(float(mean_absolute_error(y, p)), 3)
                r[f"{label}_coverage"] = round(float(np.mean(y <= p)), 3)
                r[f"{label}_within10"] = round(float(np.mean(np.abs(y - p) <= 10)), 3)
            rows.append(r)
        rows.append(_population_row(se, ho, config))

        board = pd.DataFrame(rows).sort_values("select_pinball")
        best, decision = _promote(board, fitted, ho, config, seed, offset_model)
        return board, best, decision
    except Exception as e:
        raise CustomException(e, sys)


def _population_row(se, ho, config) -> dict:
    r = dict(model="population constant", family="reference", native="-", cost="low",
             conformal_shift=0.0)
    for label, part in (("select", se), ("holdout", ho)):
        if len(part) < 20:
            continue
        p = np.full(len(part), config.population_threshold_mmHg)
        y = part.y.to_numpy(float)
        r[f"{label}_pinball"] = round(pinball(y, p), 4)
        r[f"{label}_MAE"] = round(float(mean_absolute_error(y, p)), 3)
        r[f"{label}_coverage"] = round(float(np.mean(y <= p)), 3)
        r[f"{label}_within10"] = round(float(np.mean(np.abs(y - p) <= 10)), 3)
    return r


def _promote(board, fitted, ho, config, seed, offset_model=None):
    """Select on `select`, then require a paired win on `holdout` to promote.

    Selection and the promotion test run on different patients on purpose. A candidate that
    wins on `select` has been chosen for winning there; asking whether it also beats the
    reference on `holdout`, paired row by row, is the only question whose answer was not
    already decided by the selection.

    `offset_model` is the FITTED blend, and it has to be, because `k` is searched over
    (10, 30, 60) by `search_offset`. This function used to rebuild the blend reference from
    `config.offset_k` -- the unsearched default of 30 -- while the board above built it from
    `offset_model.k`. Whenever the search picked anything other than 30, the promotion test
    was therefore run against a blend that neither appears on the board nor ships in the
    bundle, so a learned candidate was promoted or rejected against a model no patient is
    ever scored by.
    """
    k_blend = float(getattr(offset_model, "k", None) or config.offset_k)
    learned = board[(board.family == "learned") & board.select_pinball.notna()]
    refs = board[(board.family == "reference") & board.holdout_pinball.notna()]
    if learned.empty or refs.empty or len(ho) < 30:
        return None, {"verdict": "no learned candidate scorable", "promoted": False}

    name = str(learned.sort_values("select_pinball").iloc[0].model)
    ref_name = str(refs.sort_values("holdout_pinball").iloc[0].model)
    m, shift, _native = fitted[name]

    y = ho.y.to_numpy(float)
    p_learned = apply_caps(m.predict(ho[OFFSET_FEATURES].to_numpy(float)) + shift, config)
    if ref_name == "population constant":
        raw = np.full(len(ho), config.population_threshold_mmHg)
    elif ref_name == "personal q90 only":
        raw = ho.hist_q90.fillna(ho.cohort_prior).to_numpy(float)
    elif ref_name == "cohort prior only":
        raw = ho.cohort_prior.to_numpy(float)
    else:
        w = ho.hist_n / (ho.hist_n + k_blend)
        raw = (w * ho.hist_q90.fillna(ho.cohort_prior) + (1 - w) * ho.cohort_prior).to_numpy(float)
    p_ref = apply_caps(raw, config)

    q = OFFSET_TARGET_Q
    e_l = np.maximum(q * (y - p_learned), (q - 1) * (y - p_learned))
    e_r = np.maximum(q * (y - p_ref), (q - 1) * (y - p_ref))
    delta, (lo, hi), n = paired_delta(e_l, e_r, n_boot=config.n_boot, seed=seed)

    # hi < 0 means the learned model's pinball loss is lower across the WHOLE interval.
    promoted = bool(np.isfinite(hi) and hi < 0)
    mx = float(np.max(p_learned)) if len(p_learned) else np.nan
    assert not np.isfinite(mx) or mx < config.emergency_floor_mmHg, (
        f"learned offset predicted {mx:.1f} mmHg, at or above the "
        f"{config.emergency_floor_mmHg:.0f} mmHg emergency floor")

    logging.info("learned offset: %s vs %s -> delta %.4f [%.4f, %.4f] on %d rows -> %s",
                 name, ref_name, delta, lo, hi, n,
                 "PROMOTED" if promoted else "blend retained")
    return ((name, m, shift) if promoted else None), dict(
        candidate=name, reference=ref_name, delta_pinball=round(float(delta), 4),
        ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4), n=int(n),
        promoted=promoted, max_prediction=round(mx, 1),
        emergency_floor=config.emergency_floor_mmHg)
