"""The Model 1 architecture registry: 24 candidates across 9 families.

Port of notebook sections 6.1-6.3. Every architecture is registered with three pieces of
metadata that the selection rule actually reads:

  family    what kind of thing it is, for reporting and for the diverse-legs ensembles
  cost      low | medium | high -- and the tie-break rule prefers CHEAP. Two models that are
            statistically indistinguishable are not equally good: the cheaper one is.
  scope     global | local. A `local` model is refit per patient and is allowed to see that
            patient's own earlier rows, which is legitimate but makes it unfreezable.

Three shapes of candidate live here, and they do not share an interface:

  1. TABULAR   -- an sklearn estimator over the feature matrix. `make_model` builds it.
  2. SCORER    -- `fn(train, eval, signal, horizon) -> predictions`. Local models, the
                  classical smoothers and the ensembles cannot be expressed as a fitted
                  sklearn object over the feature matrix, because they refit per patient,
                  recurse over raw history, or combine other candidates.
  3. FROZEN    -- `FinalForecaster`, the servable form. A candidate that wins the bake-off
                  must survive into the bundle, and `fit_final` returns None for the ones
                  that cannot. That is not a rounding error: `local_ar` is a plausible
                  winner on this corpus and it is unfreezable by construction, so the
                  fallback path is a first-class part of the design rather than a guard.

The delta reparameterisations deserve a note. `ridge_delta` fits on `y - sbp_lag1` and adds
lag1 back at predict time. That is not a different model, it is a different question: the
tabular form has to learn "what is this patient's blood pressure", the delta form only has to
learn "how much will it move". Persistence is a strong baseline here, so handing the model
the persistence part for free and asking it for the residual is often the single biggest win
available, and it costs one line of arithmetic.
"""

import hashlib
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants.training_pipeline import (
    ENS_BASE,
    GOVERNANCE_KEYS,
    KERNEL_FIT_ROWS,
    LOCAL_MAX_FEATURES,
    LOCAL_MIN_TRAIN_ROWS,
    MODEL_TIER,
    RUN_LOCAL_SCOPE,
    SEED,
)
from src.exception.custom_exception import CustomException
from src.logging.logger import logging

# --------------------------------------------------------------------------- factory

#: sklearn estimator kinds. Everything here is fittable over the feature matrix directly.
TABULAR_KINDS = ("ridge", "elasticnet", "huber", "hgb", "hgb_mae", "hgb_mse",
                 "random_forest", "extra_trees", "knn", "mlp")

#: Kinds whose fit cost grows badly with rows; sampled down to KERNEL_FIT_ROWS.
KERNEL_KINDS = {"knn", "mlp"}


def _lin(model) -> Pipeline:
    """Impute then scale then fit. Every non-tree learner needs both."""
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("model", model)])


def _hgb(loss: str, **kw) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss, random_state=SEED,
        # sklearn turns early stopping on by itself above 10,000 rows and carves out
        # validation_fraction=0.1 -- rows drawn IID across patients AND across time. On a
        # lagged panel that is a leaky holdout: a patient's step 50 lands in the internal
        # validation while steps 49 and 51 stay in training, which is near-duplicate
        # information, so the stopping signal is optimistic. It also made the fit depend on
        # ROW ORDER -- the split is positional, so shuffling the same rows moved test
        # predictions by up to 13 mmHg. Selection happens on our own val split and tuning on
        # forward-chained within-patient folds, so this holdout was redundant as well as
        # wrong. Capacity is controlled by max_iter, which the search tunes properly.
        early_stopping=False,
        max_iter=kw.get("max_iter", 250), learning_rate=kw.get("learning_rate", .07),
        max_leaf_nodes=kw.get("max_leaf_nodes", 31),
        min_samples_leaf=kw.get("min_samples_leaf", 20),
        l2_regularization=kw.get("l2_regularization", 0.0),
        max_features=kw.get("max_features", 1.0))


def make_model(kind: str, **kw) -> BaseEstimator:
    """Single construction point, so training, tuning, evaluation and serving cannot drift."""
    if kind == "ridge":
        return _lin(Ridge(alpha=kw.get("alpha", 10.0)))
    if kind == "elasticnet":
        return _lin(ElasticNet(alpha=kw.get("alpha", .05), l1_ratio=kw.get("l1_ratio", .5),
                               random_state=SEED))
    if kind == "huber":
        # Robust loss on the LEVEL. The upper tail of this corpus is operationally censored
        # (any SBP > 200 was re-measured and the later value kept), so squared error spends
        # capacity fitting an artefact.
        return _lin(HuberRegressor(epsilon=kw.get("epsilon", 1.35),
                                   alpha=kw.get("alpha", 1e-4), max_iter=300))
    if kind in ("hgb", "hgb_mae"):
        return _hgb("absolute_error", **kw)
    if kind == "hgb_mse":
        # A pure loss ablation against hgb_mae: same capacity, different objective. Worth a
        # slot because MAE and RMSE disagree about which errors matter and the ship decision
        # is made on MAE.
        return _hgb("squared_error", **kw)
    if kind == "random_forest":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("model", RandomForestRegressor(
                             n_estimators=kw.get("n_estimators", 200),
                             min_samples_leaf=kw.get("min_samples_leaf", 20),
                             max_features=kw.get("max_features", 1.0),
                             n_jobs=-1, random_state=SEED))])
    if kind == "extra_trees":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("model", ExtraTreesRegressor(
                             n_estimators=kw.get("n_estimators", 200),
                             min_samples_leaf=kw.get("min_samples_leaf", 20),
                             max_features=kw.get("max_features", 1.0),
                             n_jobs=-1, random_state=SEED))])
    if kind == "knn":
        return _lin(KNeighborsRegressor(n_neighbors=kw.get("n_neighbors", 25),
                                        weights=kw.get("weights", "distance"), n_jobs=-1))
    if kind == "mlp":
        # early_stopping stays False for the same reason as the boosters: sklearn's internal
        # holdout is drawn IID from a lagged panel. An MLP without it can under-converge at
        # max_iter=200 and will say so in a ConvergenceWarning; that warning is the honest
        # outcome and is preferable to a leaky stopping signal on this data.
        return _lin(MLPRegressor(hidden_layer_sizes=kw.get("hidden_layer_sizes", (64, 64)),
                                 alpha=kw.get("alpha", 1e-4), max_iter=200,
                                 early_stopping=False, random_state=SEED))
    raise ValueError(f"unknown model kind: {kind}")


# ------------------------------------------------------------------------- the registry

COST_RANK = {"low": 0, "medium": 1, "high": 2}
_TIER_ORDER = {"low": 0, "medium": 1, "high": 2, "all": 2}


def _spec(family, cost, scope="global", freezable=True, note=""):
    return dict(family=family, cost=cost, scope=scope, freezable=freezable, note=note)


MODEL_SPEC = {
    # --- linear -------------------------------------------------------------------
    "ridge":            _spec("linear", "low", note="L2 on the level"),
    "elasticnet":       _spec("linear", "low", note="L1+L2, sparsity on a wide matrix"),
    "huber":            _spec("linear", "low", note="robust loss, censored upper tail"),
    # --- trees --------------------------------------------------------------------
    "hgb_mae":          _spec("trees", "medium", note="gradient boosting, absolute error"),
    "hgb_mse":          _spec("trees", "medium", note="loss ablation against hgb_mae"),
    "random_forest":    _spec("trees", "high", note="bagged trees"),
    "extra_trees":      _spec("trees", "high", note="extremely randomised trees"),
    # --- kernel / neural ----------------------------------------------------------
    "knn":              _spec("kernel", "high", note="distance-weighted neighbours"),
    "mlp":              _spec("neural", "medium", note="2x64 dense net"),
    # --- reparameterisation -------------------------------------------------------
    "ridge_delta":      _spec("reparam", "low", note="ridge on y - lag1"),
    "hgb_delta":        _spec("reparam", "medium", note="boosting on y - lag1"),
    # --- representation -----------------------------------------------------------
    "window_linear":    _spec("representation", "low", note="ridge on the raw lag window"),
    "window_delta":     _spec("representation", "low", note="window ridge on y - lag1"),
    # --- scope --------------------------------------------------------------------
    "global_plus_intercept": _spec("scope", "low", note="ridge + per-patient mean residual"),
    "local_ar":         _spec("scope", "medium", scope="local", freezable=False,
                              note="per-patient least squares on the lag window"),
    "local_ridge":      _spec("scope", "medium", scope="local", freezable=False,
                              note="per-patient ridge on a compact feature set"),
    "local_ridge_delta": _spec("scope", "medium", scope="local", freezable=False,
                               note="per-patient ridge on y - lag1"),
    "local_hgb":        _spec("scope", "high", scope="local", freezable=False,
                              note="per-patient boosting; expected to overfit, included "
                                   "as the honest test of that expectation"),
    # --- classical ----------------------------------------------------------------
    "holt_damped":      _spec("classical", "low", note="damped-trend exponential smoothing"),
    "theta":            _spec("classical", "low", note="SES plus half the fitted drift"),
    # --- ensemble -----------------------------------------------------------------
    "ens_mean":         _spec("ensemble", "medium", note="mean of the base legs"),
    "ens_median":       _spec("ensemble", "medium", note="median of the base legs"),
    "ens_inv_mae":      _spec("ensemble", "medium", note="inverse-MAE weights, inner split"),
    "ens_stack_ridge":  _spec("ensemble", "high", note="out-of-fold stack, NNLS weights"),
}

#: alias so existing callers and tuned-parameter keys keyed on "hgb" keep working
MODEL_SPEC["hgb"] = dict(MODEL_SPEC["hgb_mae"], note="alias of hgb_mae")


def enabled_kinds(tier: str = None) -> tuple:
    """Architectures admitted at the configured cost tier."""
    tier = (tier or MODEL_TIER).lower()
    limit = _TIER_ORDER.get(tier, 1)
    out = [k for k, s in MODEL_SPEC.items()
           if k != "hgb"                                   # the alias never competes twice
           and COST_RANK[s["cost"]] <= limit
           and (s["scope"] != "local" or RUN_LOCAL_SCOPE)]
    return tuple(out)


#: The kinds the sweep will actually fit. Kept as a module attribute because `run_sweep`
#: and the tuner both iterate it and a tier change must move both.
MODEL_KINDS = enabled_kinds()


def cost_of(name: str) -> str:
    return MODEL_SPEC.get(name, {}).get("cost", "medium")


def is_freezable(name: str) -> bool:
    return bool(MODEL_SPEC.get(name, {}).get("freezable", True))


# ------------------------------------------------------------------- search spaces

SEARCH_SPACE = {
    "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0, 300.0]},
    "elasticnet": {"alpha": [0.005, 0.02, 0.05, 0.2], "l1_ratio": [0.1, 0.5, 0.9]},
    "huber": {"epsilon": [1.1, 1.35, 1.8], "alpha": [1e-5, 1e-4, 1e-3]},
    "hgb": {"max_iter": [150, 250, 400], "learning_rate": [0.03, 0.07, 0.12],
            "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [20, 50, 100],
            "l2_regularization": [0.0, 0.5, 2.0], "max_features": [0.6, 0.8, 1.0]},
    "random_forest": {"n_estimators": [200, 400], "min_samples_leaf": [5, 20, 50],
                      "max_features": [0.3, 0.6, 1.0]},
    "extra_trees": {"n_estimators": [200, 400], "min_samples_leaf": [5, 20, 50],
                    "max_features": [0.3, 0.6, 1.0]},
    "knn": {"n_neighbors": [10, 25, 50], "weights": ["uniform", "distance"]},
}

# Reduced grid for the exhaustive backend. The full hgb SEARCH_SPACE is 3^6 = 729
# configurations, which is a grid search in name and a week in practice.
GRID_SPACE = {
    "ridge": {"alpha": [1.0, 10.0, 100.0]},
    "elasticnet": {"alpha": [0.005, 0.05, 0.2], "l1_ratio": [0.1, 0.5, 0.9]},
    "hgb": {"max_iter": [250, 400], "learning_rate": [0.03, 0.07],
            "max_leaf_nodes": [31, 63], "min_samples_leaf": [20, 50]},
    "random_forest": {"n_estimators": [200, 400], "min_samples_leaf": [5, 20],
                      "max_features": [0.3, 0.6]},
}

# The delta and window variants reuse their base kind's tuned parameters rather than being
# searched separately: they are the same estimator on a different target.
TUNE_ALIAS = {"ridge_delta": "ridge", "hgb_delta": "hgb",
              "window_linear": "ridge", "window_delta": "ridge",
              "global_plus_intercept": "ridge", "local_ridge": "ridge",
              "local_ridge_delta": "ridge", "local_hgb": "hgb", "hgb_mae": "hgb",
              "hgb_mse": "hgb"}

_SEARCHED_KEYS = ({k for g in SEARCH_SPACE.values() for k in g}
                  | {k for g in GRID_SPACE.values() for k in g})
# A governance parameter that leaked into a hyperparameter grid would be silently tuned
# against the data, which is exactly what "never searched" is meant to prevent.
assert not (set(GOVERNANCE_KEYS) & _SEARCHED_KEYS), \
    "a governance parameter leaked into a hyperparameter grid"


# ------------------------------------------------------------------------- helpers

def window_cols(signal: str, config) -> list:
    return [f"{signal}_lag{lag}" for lag in config.lags]


def local_feature_set(signal: str, config, available: list) -> list:
    """A compact feature set for per-patient models.

    At MIN_SESSIONS=60 a patient contributes roughly 0.6 * 60 = 36 training rows. Handing
    that 114 features is not a model, it is an interpolation of the training rows, and the
    per-patient candidates would lose the bake-off for the wrong reason.
    """
    want = (window_cols(signal, config)
            + [f"{signal}_ewm0.3", f"{signal}_mean7", f"{signal}_slope7",
               f"{signal}_base_mean", "days_since_last"])
    seen, out = set(), []
    for c in want:
        if c in available and c not in seen:
            seen.add(c)
            out.append(c)
    return out[:LOCAL_MAX_FEATURES]


def _fit_predict_tabular(kind, tr, ev, features, target, params, seed=SEED):
    if kind in KERNEL_KINDS and len(tr) > KERNEL_FIT_ROWS:
        tr = tr.sample(KERNEL_FIT_ROWS, random_state=seed)
    m = make_model(kind, **(params or {})).fit(tr[features], tr[target].to_numpy(float))
    return np.asarray(m.predict(ev[features]), dtype=float), m


# ------------------------------------------------------------------ classical smoothers

def holt_damped(y: np.ndarray, h: int, alpha=0.3, beta=0.05, phi=0.9) -> float:
    """Damped-trend exponential smoothing, h steps ahead of the end of `y`."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return float(y[-1]) if len(y) else np.nan
    level, trend = float(y[0]), float(y[1] - y[0])
    for v in y[1:]:
        prev = level
        level = alpha * v + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - prev) + (1 - beta) * phi * trend
    damp = sum(phi ** i for i in range(1, h + 1))
    return float(level + damp * trend)


def theta_method(y: np.ndarray, h: int, alpha=0.3) -> float:
    """SES level plus half the fitted drift -- the Theta method's classical form."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return float(y[-1]) if len(y) else np.nan
    level = float(y[0])
    for v in y[1:]:
        level = alpha * v + (1 - alpha) * level
    drift = float((y[-1] - y[0]) / max(len(y) - 1, 1))
    return float(level + 0.5 * h * drift)


def _history_map(full: pd.DataFrame, rows: pd.Index, signal: str) -> dict:
    """Row index -> that patient's signal history strictly before that row.

    The classical smoothers recurse over the raw series rather than reading feature columns,
    so they need the history itself. `full` must span BOTH the training and evaluation rows:
    an evaluation row sits late in the patient's timeline, and slicing history out of the
    evaluation frame alone would hand the smoother a series that starts at the val boundary
    and throw away everything the patient did before it.
    """
    want = set(rows)
    out = {}
    for _, g in full.groupby("series_id", sort=False):
        g = g.sort_values("step")
        vals = g[signal].to_numpy(dtype=float)
        idx = g.index.to_numpy()
        for i, ix in enumerate(idx):
            if ix in want:
                out[ix] = vals[:i] if i else vals[:1]
    return out


def _combined(tr: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """tr and ev as one timeline, deduplicated, ordered. Never sees the target."""
    both = pd.concat([tr, ev]) if len(tr) else ev
    both = both[~both.index.duplicated(keep="first")]
    return both.sort_values(["series_id", "step"])


_HOLT_CACHE: dict = {}


def _holt_cache_key(tr: pd.DataFrame, signal: str, h: int, seed: int):
    """Identify a training frame by content, not by object identity.

    The index alone would very nearly do -- the same rows give the same histories -- but
    hashing the signal column too means a frame that was re-sampled or re-scaled between
    stages cannot collide with the earlier one and silently reuse its parameters.
    """
    idx = np.ascontiguousarray(tr.index.to_numpy())
    col = np.ascontiguousarray(tr[signal].to_numpy(dtype=float)) if signal in tr else b""
    dig = hashlib.blake2b(idx.tobytes(), digest_size=16)
    dig.update(col.tobytes() if isinstance(col, np.ndarray) else col)
    return (signal, int(h), int(seed), len(tr), dig.hexdigest())


def _fit_holt_params(tr: pd.DataFrame, signal: str, h: int, seed=SEED) -> tuple:
    """Grid-fit alpha/beta/phi on training rows. 18 configurations, fitted once, then frozen.

    Memoised, and the reason is worth stating. This searches 18 configurations with a Python
    loop over 20,000 rows, which measured as 92% of a bake-off cell on its own. It is called
    12 times per cell with byte-identical arguments -- once for each of the three evaluation
    arms, and again for each arm's three ensemble legs, since `holt` is one of them. Nothing
    in the grid depends on the evaluation frame, only on `(tr, signal, h, seed)`, so all
    twelve calls were recomputing the same triple.

    The cache is keyed on training-frame content, so the tuned sweep does not inherit the
    untuned sweep's answer unless the frame really is identical.
    """
    key = _holt_cache_key(tr, signal, h, seed)
    if key in _HOLT_CACHE:
        return _HOLT_CACHE[key]
    d = tr if len(tr) <= 20_000 else tr.sample(20_000, random_state=seed)
    target = f"y_{signal}_h{h}"
    hist = _history_map(_combined(tr, d), d.index, signal)
    y = d[target].to_numpy(dtype=float)
    best, best_p = np.inf, (0.3, 0.05, 0.9)
    for a in (0.1, 0.3, 0.5):
        for b in (0.0, 0.05, 0.2):
            for p in (0.8, 0.98):
                pred = np.array([holt_damped(hist.get(i, np.empty(0)), h, a, b, p)
                                 for i in d.index], dtype=float)
                m = np.isfinite(pred) & np.isfinite(y)
                if m.sum() < 50:
                    continue
                e = float(np.mean(np.abs(pred[m] - y[m])))
                if e < best:
                    best, best_p = e, (a, b, p)
    _HOLT_CACHE[key] = best_p
    return best_p


# ------------------------------------------------------------------------ the scorers
# Every candidate that is not a plain sklearn fit over the feature matrix is scored through
# `score_architecture`, which returns predictions aligned to `ev.index`. One dispatch point
# rather than the notebook's decorator registry, so the shapes stay visible side by side.

def _delta_target(d: pd.DataFrame, target: str, signal: str) -> np.ndarray:
    return d[target].to_numpy(float) - d[f"{signal}_lag1"].to_numpy(float)


def _fit_local(kind, tr, ev, signal, h, config, features, params, delta=False, seed=SEED):
    """Refit per patient, falling back to a pooled fit where a patient has too little history.

    A local model is allowed to read its own patient's earlier rows -- that is the point of
    it, and it is legitimate because at serving time the patient's past really is available.
    It is also why these models cannot be frozen into the bundle: there is no single fitted
    object, only a recipe for refitting.
    """
    target = f"y_{signal}_h{h}"
    cols = local_feature_set(signal, config, features)
    lag1 = f"{signal}_lag1"
    out = pd.Series(np.nan, index=ev.index, dtype=float)

    # Pooled fallback, fitted once: patients below the row floor use this instead of a fit
    # on a handful of rows. Without it the local candidates score NaN for short histories and
    # the bake-off silently compares them on an easier subset.
    pooled = None
    if len(tr) >= 200:
        y_tr = _delta_target(tr, target, signal) if delta else tr[target].to_numpy(float)
        m = np.isfinite(y_tr)
        if m.sum() >= 200:
            pooled = make_model(kind, **(params or {})).fit(tr.loc[tr.index[m], cols],
                                                            y_tr[m])

    for sid, g_ev in ev.groupby("series_id", sort=False):
        g_tr = tr[tr.series_id == sid]
        mdl = pooled
        if len(g_tr) >= LOCAL_MIN_TRAIN_ROWS:
            y = _delta_target(g_tr, target, signal) if delta else g_tr[target].to_numpy(float)
            m = np.isfinite(y)
            if m.sum() >= LOCAL_MIN_TRAIN_ROWS:
                try:
                    mdl = make_model(kind, **(params or {})).fit(g_tr.loc[g_tr.index[m], cols],
                                                                 y[m])
                except Exception:                                    # noqa: BLE001
                    mdl = pooled
        if mdl is None:
            continue
        p = np.asarray(mdl.predict(g_ev[cols]), dtype=float)
        if delta:
            p = p + g_ev[lag1].to_numpy(float)
        out.loc[g_ev.index] = p
    return out.to_numpy(dtype=float)


def _local_ar(tr, ev, signal, h, config, features, seed=SEED):
    """Per-patient least squares on the raw lag window, persistence where history is thin."""
    target = f"y_{signal}_h{h}"
    cols = [c for c in window_cols(signal, config) if c in features]
    lag1 = f"{signal}_lag1"
    gmean = float(np.nanmean(tr[target].to_numpy(float))) if len(tr) else np.nan
    out = pd.Series(np.nan, index=ev.index, dtype=float)

    for sid, g_ev in ev.groupby("series_id", sort=False):
        g_tr = tr[tr.series_id == sid]
        X = g_tr[cols].to_numpy(float)
        y = g_tr[target].to_numpy(float)
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if m.sum() < 40:
            # Not enough of the patient's own history to fit an autoregression. Persistence
            # is the honest answer, not a pooled model wearing a local label.
            out.loc[g_ev.index] = g_ev[lag1].fillna(gmean).to_numpy(float)
            continue
        A = np.column_stack([np.ones(m.sum()), X[m]])
        coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        Xe = g_ev[cols].to_numpy(float)
        Ae = np.column_stack([np.ones(len(Xe)), np.nan_to_num(Xe, nan=np.nanmean(X[m]))])
        out.loc[g_ev.index] = Ae @ coef
    return out.to_numpy(dtype=float)


def _ens_legs(tr, ev, signal, h, config, features, params, seed=SEED) -> dict:
    """The base legs every ensemble combines: two tabular models plus a classical smoother."""
    target = f"y_{signal}_h{h}"
    legs = {}
    for kind in ENS_BASE:
        if kind not in MODEL_SPEC:
            continue
        try:
            p, _ = _fit_predict_tabular(kind, tr, ev, features, target,
                                        (params or {}).get((TUNE_ALIAS.get(kind, kind),
                                                            signal), {}), seed)
            legs[kind] = p
        except Exception as exc:                                     # noqa: BLE001
            logging.debug("ensemble leg %s unavailable: %s", kind, exc)
    try:
        legs["holt"] = score_architecture("holt_damped", tr, ev, signal, h, config,
                                          features, params, seed)
    except Exception as exc:                                         # noqa: BLE001
        logging.debug("ensemble leg holt unavailable: %s", exc)
    return legs


def _inner_split(tr: pd.DataFrame):
    """75/25 by within-patient percentile rank, for learning ensemble weights.

    Weights learned on the same rows the legs were fitted on would reward the leg that
    overfits hardest. This split never touches val or test.
    """
    q = tr.groupby("series_id").step.transform(lambda s: s.rank(pct=True))
    return tr[q <= 0.75], tr[q > 0.75]


def score_architecture(name, tr, ev, signal, h, config, features, params=None, seed=SEED):
    """Predictions for one architecture on `ev`, aligned to `ev.index`.

    Tabular kinds go through `make_model`; everything else has a shape that cannot be
    expressed as a fitted estimator over the feature matrix and is handled here.
    """
    try:
        target = f"y_{signal}_h{h}"
        params = params or {}
        tuned = params.get((TUNE_ALIAS.get(name, name), signal), {})

        if name in TABULAR_KINDS:
            return _fit_predict_tabular(name, tr, ev, features, target, tuned, seed)[0]

        if name in ("ridge_delta", "hgb_delta"):
            base = "ridge" if name.startswith("ridge") else "hgb"
            y = _delta_target(tr, target, signal)
            m = np.isfinite(y)
            mdl = make_model(base, **tuned).fit(tr.loc[tr.index[m], features], y[m])
            return (np.asarray(mdl.predict(ev[features]), float)
                    + ev[f"{signal}_lag1"].to_numpy(float))

        if name in ("window_linear", "window_delta"):
            cols = [c for c in window_cols(signal, config) if c in features]
            if not cols:
                return np.full(len(ev), np.nan)
            delta = name.endswith("delta")
            y = _delta_target(tr, target, signal) if delta else tr[target].to_numpy(float)
            m = np.isfinite(y)
            mdl = make_model("ridge", alpha=1.0).fit(tr.loc[tr.index[m], cols], y[m])
            p = np.asarray(mdl.predict(ev[cols]), float)
            return p + ev[f"{signal}_lag1"].to_numpy(float) if delta else p

        if name == "global_plus_intercept":
            mdl = make_model("ridge", **tuned).fit(tr[features], tr[target].to_numpy(float))
            resid = tr[target].to_numpy(float) - np.asarray(mdl.predict(tr[features]), float)
            bias = pd.Series(resid, index=tr.index).groupby(tr.series_id).mean()
            return (np.asarray(mdl.predict(ev[features]), float)
                    + ev.series_id.map(bias).fillna(0.0).to_numpy(float))

        if name == "local_ar":
            return _local_ar(tr, ev, signal, h, config, features, seed)
        if name in ("local_ridge", "local_ridge_delta"):
            return _fit_local("ridge", tr, ev, signal, h, config, features,
                              {"alpha": 1.0}, delta=name.endswith("delta"), seed=seed)
        if name == "local_hgb":
            return _fit_local("hgb", tr, ev, signal, h, config, features,
                              {"max_iter": 80, "max_leaf_nodes": 8, "min_samples_leaf": 8},
                              seed=seed)

        if name in ("holt_damped", "theta"):
            full = _combined(tr, ev)
            hist = _history_map(full, ev.index, signal)
            if name == "theta":
                return np.array([theta_method(hist.get(i, np.empty(0)), h + 1)
                                 for i in ev.index], dtype=float)
            a, b, p = _fit_holt_params(tr, signal, h, seed)
            return np.array([holt_damped(hist.get(i, np.empty(0)), h + 1, a, b, p)
                             for i in ev.index], dtype=float)

        if name.startswith("ens_"):
            legs = _ens_legs(tr, ev, signal, h, config, features, params, seed)
            if not legs:
                return np.full(len(ev), np.nan)
            P = np.column_stack(list(legs.values()))
            if name == "ens_mean":
                return np.nanmean(P, axis=1)
            if name == "ens_median":
                return np.nanmedian(P, axis=1)
            w = _ensemble_weights(name, tr, signal, h, config, features, params,
                                  list(legs), seed)
            if w is None:
                return np.nanmean(P, axis=1)
            M = np.isfinite(P)
            Pz = np.where(M, P, 0.0)
            wsum = (M * w).sum(axis=1)
            # Renormalise per row over the legs that produced a finite value, so one leg
            # returning NaN shrinks the prediction toward zero instead of being ignored.
            return np.where(wsum > 0, (Pz * w).sum(axis=1) / np.where(wsum > 0, wsum, 1),
                            np.nan)

        raise ValueError(f"no scorer for architecture {name!r}")
    except Exception as e:
        raise CustomException(e, sys)


def _ensemble_weights(name, tr, signal, h, config, features, params, leg_names, seed=SEED):
    """Non-negative weights summing to 1, learned without touching val or test."""
    inner_tr, inner_va = _inner_split(tr)
    if len(inner_tr) < 200 or len(inner_va) < 50:
        return None
    target = f"y_{signal}_h{h}"
    legs = _ens_legs(inner_tr, inner_va, signal, h, config, features, params, seed)
    legs = {k: v for k, v in legs.items() if k in leg_names}
    if len(legs) != len(leg_names):
        return None
    P = np.column_stack([legs[k] for k in leg_names])
    y = inner_va[target].to_numpy(float)
    m = np.isfinite(y) & np.isfinite(P).all(axis=1)
    if m.sum() < 50:
        return None

    if name == "ens_inv_mae":
        mae = np.array([max(float(np.mean(np.abs(P[m, j] - y[m]))), 1e-6)
                        for j in range(P.shape[1])])
        w = 1.0 / mae
    elif name == "ens_stack_ridge":
        from scipy.optimize import nnls
        # Non-negative least squares, not an unconstrained stack: a negative weight means
        # the ensemble is betting against one of its own legs, which is unstable out of
        # sample and impossible to explain to a clinician.
        try:
            w, _ = nnls(P[m], y[m])
        except Exception:                                            # noqa: BLE001
            w = np.ones(P.shape[1])
    else:
        w = np.ones(P.shape[1])
    s = float(np.sum(w))
    return (w / s) if s > 0 else None


# ------------------------------------------------------------------------ the freeze layer

class FinalForecaster(BaseEstimator):
    """The servable form of a bake-off winner.

    Everything the bundle holds must satisfy one contract -- `.predict(X) -> array` over a
    feature-matrix slice -- because `BPPredictor.predict`, the backtest and the dashboard all
    call it the same way. The candidates do not naturally share that shape: a delta model has
    to add lag1 back, a window model reads a different column set, and an ensemble has to
    call its legs. This wrapper absorbs those differences once, so nothing downstream has to
    know which family won.

    `kind` is the wrapper's own dispatch, not the estimator kind:

      tabular           fitted estimator over `features`
      delta             fitted estimator on y - lag1; lag1 added back at predict
      window            fitted estimator over the raw lag window only
      window_delta      both of the above
      global_intercept  fitted estimator plus a per-patient mean residual
      classical         no fitted object; recursion over the patient's own recent history
      ensemble          several FinalForecasters combined by fixed weights
    """

    def __init__(self, kind, signal, horizon, model=None, cols=None, bias=None,
                 params=None, legs=None, weights=None, name=""):
        self.kind = kind
        self.signal = signal
        self.horizon = horizon
        self.model = model
        self.cols = cols
        self.bias = bias if bias is not None else {}
        self.params = params or {}
        self.legs = legs or []
        self.weights = weights
        self.name = name

    def fit(self, X=None, y=None):
        return self

    def _lag1(self, d: pd.DataFrame) -> np.ndarray:
        c = f"{self.signal}_lag1"
        return d[c].to_numpy(dtype=float) if c in d else np.full(len(d), np.nan)

    def predict(self, X) -> np.ndarray:
        d = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.kind in ("tabular", "delta"):
            p = np.asarray(self.model.predict(d[self.cols]), dtype=float)
            return p + self._lag1(d) if self.kind == "delta" else p
        if self.kind in ("window", "window_delta"):
            p = np.asarray(self.model.predict(d[self.cols]), dtype=float)
            return p + self._lag1(d) if self.kind == "window_delta" else p
        if self.kind == "global_intercept":
            p = np.asarray(self.model.predict(d[self.cols]), dtype=float)
            if "series_id" in d:
                p = p + d.series_id.map(self.bias).fillna(0.0).to_numpy(dtype=float)
            return p
        if self.kind == "classical":
            # At serving time the frame is one row whose lag columns already carry the
            # patient's recent history, so the smoother recurses over that window in
            # chronological order. It is the same history every tabular candidate sees.
            lags = [c for c in self.cols if c in d.columns]
            a, b, ph = self.params.get("abphi", (0.3, 0.05, 0.9))
            out = []
            for _, r in d[lags].iterrows():
                hist = r.to_numpy(dtype=float)[::-1]          # lag14..lag1 -> oldest first
                hist = hist[np.isfinite(hist)]
                out.append(theta_method(hist, self.horizon + 1) if self.name == "theta"
                           else holt_damped(hist, self.horizon + 1, a, b, ph))
            return np.asarray(out, dtype=float)
        if self.kind == "ensemble":
            P = np.column_stack([leg.predict(d) for leg in self.legs])
            w = (np.asarray(self.weights, dtype=float) if self.weights is not None
                 else np.full(P.shape[1], 1.0 / P.shape[1]))
            M = np.isfinite(P)
            wsum = (M * w).sum(axis=1)
            # Renormalise per row over the legs that produced a finite value: one leg
            # returning NaN must shrink the ensemble toward the others, not toward zero.
            num = (np.where(M, P, 0.0) * w).sum(axis=1)
            return np.where(wsum > 0, num / np.where(wsum > 0, wsum, 1.0), np.nan)
        raise ValueError(f"unknown FinalForecaster kind: {self.kind}")

    def __repr__(self) -> str:
        return f"FinalForecaster({self.name or self.kind}, {self.signal}, h{self.horizon})"


def fit_final(name, tr, signal, h, config, features, params=None, seed=SEED):
    """Freeze one architecture into a servable object, or None if it cannot be frozen.

    Returning None is a real outcome, not an error path. Every `scope="local"` candidate
    refits per patient and has no single fitted object to store, so if one wins the bake-off
    the shipped model is necessarily a different one. `BPPredictor.build` handles that
    explicitly and a safety gate reports the substitution -- silently shipping a fallback
    while the scorecard names a different winner is the failure this exists to prevent.
    """
    try:
        if not is_freezable(name):
            return None
        target = f"y_{signal}_h{h}"
        params = params or {}
        tuned = params.get((TUNE_ALIAS.get(name, name), signal), {})
        d = tr[tr[target].notna()]
        if len(d) < 100:
            return None

        if name in TABULAR_KINDS:
            m = make_model(name, **tuned).fit(d[features], d[target].to_numpy(float))
            return FinalForecaster("tabular", signal, h, model=m, cols=list(features),
                                   name=name)

        if name in ("ridge_delta", "hgb_delta"):
            base = "ridge" if name.startswith("ridge") else "hgb"
            y = _delta_target(d, target, signal)
            msk = np.isfinite(y)
            m = make_model(base, **tuned).fit(d.loc[d.index[msk], features], y[msk])
            return FinalForecaster("delta", signal, h, model=m, cols=list(features),
                                   name=name)

        if name in ("window_linear", "window_delta"):
            cols = [c for c in window_cols(signal, config) if c in features]
            if not cols:
                return None
            delta = name.endswith("delta")
            y = _delta_target(d, target, signal) if delta else d[target].to_numpy(float)
            msk = np.isfinite(y)
            m = make_model("ridge", alpha=1.0).fit(d.loc[d.index[msk], cols], y[msk])
            return FinalForecaster("window_delta" if delta else "window", signal, h,
                                   model=m, cols=cols, name=name)

        if name == "global_plus_intercept":
            m = make_model("ridge", **tuned).fit(d[features], d[target].to_numpy(float))
            resid = d[target].to_numpy(float) - np.asarray(m.predict(d[features]), float)
            bias = pd.Series(resid, index=d.index).groupby(d.series_id).mean().to_dict()
            return FinalForecaster("global_intercept", signal, h, model=m,
                                   cols=list(features), bias=bias, name=name)

        if name in ("holt_damped", "theta"):
            cols = [c for c in window_cols(signal, config) if c in features]
            if not cols:
                return None
            abphi = (_fit_holt_params(d, signal, h, seed) if name == "holt_damped"
                     else (0.3, 0.05, 0.9))
            return FinalForecaster("classical", signal, h, cols=cols,
                                   params={"abphi": abphi}, name=name)

        if name.startswith("ens_"):
            legs, leg_names = [], []
            for kind in ENS_BASE:
                leg = fit_final("hgb_mae" if kind == "hgb" else kind, d, signal, h, config,
                                features, params, seed)
                if leg is not None:
                    legs.append(leg)
                    leg_names.append(kind)
            holt = fit_final("holt_damped", d, signal, h, config, features, params, seed)
            if holt is not None:
                legs.append(holt)
                leg_names.append("holt")
            if not legs:
                return None
            w = None
            if name in ("ens_inv_mae", "ens_stack_ridge"):
                w = _ensemble_weights(name, d, signal, h, config, features, params,
                                      leg_names, seed)
            if name == "ens_median":
                # A frozen median is not expressible as fixed weights. The equal-weight mean
                # is the closest servable form; `name` records what was actually selected so
                # the substitution is visible in the bundle rather than implied.
                logging.info("ens_median freezes as an equal-weight mean")
            return FinalForecaster("ensemble", signal, h, legs=legs, weights=w, name=name)

        return None
    except Exception as exc:                                         # noqa: BLE001
        logging.warning("could not freeze %s for %s h%s: %s", name, signal, h, exc)
        return None
