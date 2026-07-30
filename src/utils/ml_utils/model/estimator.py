"""Model factory, tuning, sweep and the frozen serving artifact.

Port of notebook sections 6, 6.1, 6.2, 6.4 and 10. `make_model` is the single place a
model is constructed, so training, tuning, evaluation and serving cannot drift apart.
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.constants.training_pipeline import SEED, STALE_FORECAST_MAX_DAYS
from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.utils.ml_utils.feature.causal_features import CausalFeatureBuilder
from src.utils.ml_utils.metric.regression_metric import BASELINE_SPECS, evaluate, forecast_baselines

# The architecture registry is the single source of truth for what can compete, what it
# costs, and whether it can be frozen. `make_model` is re-exported here because the trainer,
# the fairness audit and the missingness sweep all import it from this module.
from src.utils.ml_utils.model.architectures import (
    COST_RANK,
    GRID_SPACE,
    MODEL_KINDS,
    MODEL_SPEC,
    SEARCH_SPACE,
    TUNE_ALIAS,
    FinalForecaster,
    cost_of,
    enabled_kinds,
    fit_final,
    is_freezable,
    make_model,
    score_architecture,
)
from src.utils.ml_utils.model.offset import OffsetModel

__all__ = ["MODEL_KINDS", "MODEL_SPEC", "SEARCH_SPACE", "GRID_SPACE", "COST_RANK",
           "TUNE_ALIAS", "make_model", "fit_final", "is_freezable", "cost_of",
           "score_architecture", "FinalForecaster", "BaselineForecaster", "run_sweep",
           "forward_chain_folds", "cv_score", "random_search", "fit_quantile_interval",
           "BPPredictor", "explain_prediction", "champion_challenger"]


class BaselineForecaster(BaseEstimator):
    """A baseline wearing the estimator interface, so the ship decision can be obeyed.

    `ship_decision` can rule that no learned model earned its place. Acting on that
    verdict means the baseline has to satisfy the same `.predict(X)` contract the
    bundle, the backtest and the serving path all assume -- otherwise the rule stays
    a report and the learned model serves anyway.

    There is nothing to fit: every baseline is a closed form over columns the feature
    matrix already carries, taken from the same `BASELINE_SPECS` that scored it. A
    shipped baseline therefore cannot behave differently from the one that won.
    """

    def __init__(self, name: str, signal: str, horizon: int):
        self.name = name
        self.signal = signal
        self.horizon = horizon

    def fit(self, X=None, y=None) -> "BaselineForecaster":
        return self

    def predict(self, X) -> np.ndarray:
        d = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return np.asarray(BASELINE_SPECS[self.name](d, self.signal, self.horizon), float)

    def __repr__(self) -> str:
        return f"BaselineForecaster({self.name}, {self.signal}, h{self.horizon})"


# ------------------------------------------------------------------------------ sweep

def run_sweep(F: pd.DataFrame, features: list, config, params: dict = None, seed: int = SEED):
    """Score every admitted architecture on three evaluation arms, per signal x horizon.

    The three arms answer three different questions and the pipeline needs all of them:

      val         later sessions of FIT patients -- used for selection
      test        latest sessions of FIT patients -- unseen time, seen patient
      holdout_pt  latest sessions of HOLDOUT patients -- unseen time AND unseen patient

    `holdout_pt` is the cold-start arm, and it is the only measurement of what a NEW user
    gets. For a journaling app that is the dominant case, not an edge case: most people
    submitting a history have never been seen by the model. Reporting only `test` describes
    the experience of a patient the model has already trained on.

    `scope="local"` candidates are trained on `tr_own` -- train rows of EVERY patient,
    including the holdout ones -- because a per-patient model legitimately reads its own
    patient's past, and at serving time that past really is available. Global candidates
    only ever see `tr_fit`.
    """
    params = params or {}
    # Resolved from the config, not from the module global: MODEL_KINDS is captured at
    # import, so a run that sets config.model_tier would otherwise be silently ignored and
    # the "low tier" smoke test would quietly run the medium fleet.
    kinds = enabled_kinds(getattr(config, "model_tier", None))
    rows, preds, fitted = [], {}, {}
    for signal in config.signals:
        for h in config.horizons:
            target = f"y_{signal}_h{h}"
            if target not in F.columns:
                continue
            d = F[F[target].notna()]
            fit_pt = d[d.patient_split == "fit"] if "patient_split" in d else d
            hold_pt = d[d.patient_split == "holdout"] if "patient_split" in d else d.iloc[:0]

            tr_fit = fit_pt[fit_pt.split == "train"]
            tr_own = d[d.split == "train"]
            va = fit_pt[fit_pt.split == "val"]
            te = fit_pt[fit_pt.split == "test"]
            ho = hold_pt[hold_pt.split == "test"]
            if len(tr_fit) < 200 or len(te) < 40:
                continue
            if len(tr_fit) > config.max_train_rows:
                tr_fit = tr_fit.sample(config.max_train_rows, random_state=seed)
            if len(tr_own) > config.max_train_rows:
                tr_own = tr_own.sample(config.max_train_rows, random_state=seed)
            va = va.sample(min(len(va), config.max_eval_rows), random_state=seed)
            te = te.sample(min(len(te), config.max_eval_rows), random_state=seed)
            ho = ho.sample(min(len(ho), config.max_eval_rows), random_state=seed)
            arms = [("val", va), ("test", te), ("holdout_pt", ho)]

            # Baselines are scored on every arm: a baseline can be the thing that ships, the
            # drift monitor needs its val MAE as a reference, and its cold-start penalty is
            # the number a learned model has to beat to justify itself for new patients.
            for label, part in arms:
                if len(part) < 40:
                    continue
                for name, pred in forecast_baselines(part, signal, h).items():
                    r = evaluate(part[target].values, pred, part[f"{signal}_lag1"].values,
                                 model=name, family="baseline", cost="low", scope="global",
                                 signal=signal, horizon=h, split=label)
                    if r:
                        rows.append(r)
                    # Baseline per-row errors go into `preds` alongside the learned ones.
                    # The ship rule compares the winner against the best BASELINE, so
                    # without these the paired test has nothing to pair against and silently
                    # degrades to comparing two marginal confidence intervals -- which is
                    # both weaker and the exact comparison the paired test exists to replace.
                    if (label == "test" and signal == "sbp"
                            and h == config.horizons[min(1, len(config.horizons) - 1)]):
                        preds[name] = (part[target].values, np.asarray(pred, float),
                                       part[f"{signal}_lag1"].values,
                                       part.series_id.to_numpy())

            for kind in kinds:
                spec = MODEL_SPEC[kind]
                tr = tr_own if spec["scope"] == "local" else tr_fit
                for label, part in arms:
                    if len(part) < 40:
                        continue
                    try:
                        pred = score_architecture(kind, tr, part, signal, h, config,
                                                  features, params, seed)
                    except Exception as exc:                          # noqa: BLE001
                        # One architecture failing must not take the bake-off down with it;
                        # the failure is recorded so it cannot be mistaken for absence.
                        logging.warning("%s failed on %s h%s (%s): %s",
                                        kind, signal, h, label, exc)
                        rows.append(dict(model=kind, family=spec["family"],
                                         cost=spec["cost"], scope=spec["scope"],
                                         signal=signal, horizon=h, split=label,
                                         note=f"{type(exc).__name__}: {exc}"[:160]))
                        continue
                    r = evaluate(part[target].values, pred,
                                 part[f"{signal}_lag1"].values, model=kind,
                                 family="learned", cost=spec["cost"], scope=spec["scope"],
                                 signal=signal, horizon=h, split=label)
                    if r:
                        rows.append(r)
                    if (label == "test" and signal == "sbp"
                            and h == config.horizons[min(1, len(config.horizons) - 1)]):
                        # Per-row errors, kept so the ship decision can run a PAIRED
                        # bootstrap rather than comparing two marginal confidence intervals.
                        preds[kind] = (part[target].values, np.asarray(pred, float),
                                       part[f"{signal}_lag1"].values,
                                       part.series_id.to_numpy())
    return pd.DataFrame(rows), preds, fitted


# ----------------------------------------------------------------------------- tuning

def forward_chain_folds(d: pd.DataFrame, n_folds: int):
    """Expanding-window folds inside each patient's own timeline."""
    q = d.groupby("series_id").step.transform(lambda s: s.rank(pct=True))
    edges = np.linspace(0.4, 1.0, n_folds + 1)
    for i in range(n_folds):
        yield (q <= edges[i]).values, ((q > edges[i]) & (q <= edges[i + 1])).values


def cv_score(kind: str, params: dict, d: pd.DataFrame, target: str,
             features: list, n_folds: int, trial=None) -> float:
    """Mean MAE across forward-chained, within-patient folds.

    `trial` is an optional Optuna trial. Reporting per fold lets the median pruner kill a
    losing configuration after the first fold instead of paying for all three, which is most
    of where the trial budget goes.
    """
    errs = []
    for i, (tr_m, va_m) in enumerate(forward_chain_folds(d, n_folds)):
        tr, va = d[tr_m], d[va_m]
        if len(tr) < 200 or len(va) < 50:
            continue
        m = make_model(kind, **params).fit(tr[features], tr[target].values)
        errs.append(mean_absolute_error(va[target].values, m.predict(va[features])))
        if trial is not None:
            trial.report(float(np.mean(errs)), i)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
    return float(np.mean(errs)) if errs else np.nan


#: Only the tabular kinds have a search space. Everything else either reuses a base kind's
#: parameters (see TUNE_ALIAS) or has no hyperparameters worth searching.
def _tunable_kinds(config=None) -> list:
    kinds = enabled_kinds(getattr(config, "model_tier", None)) if config else MODEL_KINDS
    base = {TUNE_ALIAS.get(k, k) for k in kinds}
    return sorted(k for k in base if k in SEARCH_SPACE)


def _tuning_frame(F, target, config, seed):
    d = F[(F[target].notna()) & (F.split != "test")]
    if len(d) > config.tune_sample_rows:
        d = d.sample(config.tune_sample_rows,
                     random_state=seed).sort_values(["series_id", "step"])
    return d


def _log_row(kind, signal, base, best, best_p, n_configs, secs, tuner):
    return dict(model=kind, signal=signal, tuner=tuner,
                cv_mae_default=round(base, 4) if np.isfinite(base) else np.nan,
                cv_mae_tuned=round(best, 4) if np.isfinite(best) else np.nan,
                gain_pct=(round(100 * (base - best) / base, 2)
                          if np.isfinite(base) and np.isfinite(best) and base else np.nan),
                n_configs=n_configs, secs=round(secs, 1), params=str(best_p))


def random_search(F: pd.DataFrame, features: list, config, best_params: dict,
                  seed: int = SEED) -> pd.DataFrame:
    """Random draws from SEARCH_SPACE scored by forward-chained CV. Fills `best_params`."""
    rng = np.random.default_rng(seed)
    log = []
    for signal in config.signals:
        target = f"y_{signal}_h{config.horizons[0]}"
        if target not in F.columns:
            continue
        d = _tuning_frame(F, target, config, seed)
        for kind in _tunable_kinds(config):
            t0 = time.perf_counter()
            try:
                base = cv_score(kind, {}, d, target, features, config.tune_folds)
                best, best_p = base, {}
                for _ in range(config.tune_draws):
                    grid = SEARCH_SPACE[kind]
                    cand = {k: (float(rng.choice(v)) if isinstance(v[0], float)
                                else rng.choice(v).item())
                            for k, v in grid.items()}
                    sc = cv_score(kind, cand, d, target, features, config.tune_folds)
                    if np.isfinite(sc) and np.isfinite(best) and sc < best:
                        best, best_p = sc, cand
            except Exception as exc:                                  # noqa: BLE001
                # One family failing must not lose the whole search. Defaults are a valid
                # configuration; an exception is not a reason to abandon the other families.
                logging.warning("%s: random search failed, keeping defaults -- %s", kind, exc)
                base, best, best_p = np.nan, np.nan, {}
            if not np.isfinite(best) or (np.isfinite(base) and best >= base):
                # Never ship a configuration that lost to the defaults on the same folds.
                best, best_p = base, {}
            best_params[(kind, signal)] = best_p
            log.append(_log_row(kind, signal, base, best, best_p, config.tune_draws,
                                time.perf_counter() - t0, "random"))
    return pd.DataFrame(log)


def optuna_params(kind: str, trial) -> dict:
    """Continuous, log-scaled ranges. A grid over `alpha` cannot express 3.7."""
    if kind == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-2, 1e3, log=True)}
    if kind == "elasticnet":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95)}
    if kind == "huber":
        return {"epsilon": trial.suggest_float("epsilon", 1.05, 2.0),
                "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True)}
    if kind == "hgb":
        return {"max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 127, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0,
                                                         log=True),
                "max_features": trial.suggest_float("max_features", 0.4, 1.0)}
    if kind in ("random_forest", "extra_trees"):
        return {"n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 80, log=True),
                "max_features": trial.suggest_float("max_features", 0.2, 1.0)}
    if kind == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 5, 80, log=True)}
    return {}


def optuna_search(F: pd.DataFrame, features: list, config, best_params: dict,
                  seed: int = SEED) -> pd.DataFrame:
    """TPE search with median pruning over the same forward-chained folds."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log = []
    for signal in config.signals:
        target = f"y_{signal}_h{config.horizons[0]}"
        if target not in F.columns:
            continue
        d = _tuning_frame(F, target, config, seed)
        for kind in _tunable_kinds(config):
            t0 = time.perf_counter()
            base = cv_score(kind, {}, d, target, features, config.tune_folds)
            try:
                study = optuna.create_study(
                    direction="minimize", study_name=f"{kind}-{signal}",
                    sampler=optuna.samplers.TPESampler(
                        seed=seed, n_startup_trials=config.tpe_startup_trials),
                    pruner=optuna.pruners.MedianPruner(
                        n_startup_trials=config.pruner_startup_trials,
                        n_warmup_steps=config.pruner_warmup_steps))
                # Loop variables bound as defaults rather than captured. `study.optimize`
                # is synchronous, so the closure happens to be correct today -- but a
                # lazily-evaluated objective would silently tune the LAST family in the
                # loop for every study, and nothing downstream would look wrong.
                def _objective(t, kind=kind, d=d, target=target):
                    return cv_score(kind, optuna_params(kind, t), d, target, features,
                                    config.tune_folds, trial=t)

                study.optimize(_objective, n_trials=config.tune_trials,
                               catch=(Exception,))
                done = [t for t in study.trials if t.value is not None]
                best = float(study.best_value) if done else np.nan
                best_p = dict(study.best_params) if done else {}
                n_pruned = sum(1 for t in study.trials
                               if t.state == optuna.trial.TrialState.PRUNED)
            except Exception as exc:                                  # noqa: BLE001
                logging.warning("%s: optuna search failed, keeping defaults -- %s", kind, exc)
                best, best_p, n_pruned = np.nan, {}, 0
            if not np.isfinite(best) or (np.isfinite(base) and best >= base):
                best, best_p = base, {}
            best_params[(kind, signal)] = best_p
            row = _log_row(kind, signal, base, best, best_p, config.tune_trials,
                           time.perf_counter() - t0, "optuna")
            row["n_pruned"] = n_pruned
            log.append(row)
    return pd.DataFrame(log)


def grid_search(F: pd.DataFrame, features: list, config, best_params: dict,
                seed: int = SEED) -> pd.DataFrame:
    """Exhaustive search over the reduced GRID_SPACE."""
    from itertools import product
    log = []
    for signal in config.signals:
        target = f"y_{signal}_h{config.horizons[0]}"
        if target not in F.columns:
            continue
        d = _tuning_frame(F, target, config, seed)
        for kind in _tunable_kinds(config):
            grid = GRID_SPACE.get(kind)
            if not grid:
                best_params[(kind, signal)] = {}
                continue
            t0 = time.perf_counter()
            base = cv_score(kind, {}, d, target, features, config.tune_folds)
            best, best_p, n = base, {}, 0
            for combo in product(*grid.values()):
                cand = dict(zip(grid.keys(), combo))
                n += 1
                sc = cv_score(kind, cand, d, target, features, config.tune_folds)
                if np.isfinite(sc) and np.isfinite(best) and sc < best:
                    best, best_p = sc, cand
            if not np.isfinite(best) or (np.isfinite(base) and best >= base):
                best, best_p = base, {}
            best_params[(kind, signal)] = best_p
            log.append(_log_row(kind, signal, base, best, best_p, n,
                                time.perf_counter() - t0, "grid"))
    return pd.DataFrame(log)


def tune(F: pd.DataFrame, features: list, config, best_params: dict,
         seed: int = SEED) -> pd.DataFrame:
    """Dispatch to the configured tuner, falling back loudly rather than silently.

    The notebook this is ported from had `TUNER = "optuna"` inside a bare try/except import,
    so a machine without optuna quietly ran a 12-draw random search instead of a 40-trial TPE
    study and reported it as tuning. The fallback here is logged at WARNING for that reason.
    """
    tuner = str(getattr(config, "tuner", "random")).lower()
    if tuner == "optuna":
        try:
            import optuna  # noqa: F401
        except ImportError:
            logging.warning("optuna is not installed; falling back to a %d-draw random "
                            "search. `pip install optuna` for the TPE study the config asks "
                            "for.", config.tune_draws)
            tuner = "random"
    fn = {"optuna": optuna_search, "grid": grid_search}.get(tuner, random_search)
    logging.info("tuning with %s over %d families x %d signals",
                 tuner, len(_tunable_kinds(config)), len(config.signals))
    return fn(F, features, config, best_params, seed)


def fit_quantile_interval(F: pd.DataFrame, features: list, config, best_params: dict,
                          signal: str, horizon: int, seed: int = SEED):
    """Quantile GBM band fit on train, conformalised on val.

    The conformal widening is calibrated on val and deliberately NOT refit later --
    refitting it on data the band already saw would void the coverage guarantee.
    """
    target = f"y_{signal}_h{horizon}"
    if target not in F.columns:
        return {}, 0.0
    di = F[F[target].notna()]
    tr, va = di[di.split == "train"], di[di.split == "val"]
    if len(tr) < 200 or len(va) < 50:
        return {}, 0.0
    if len(tr) > config.max_train_rows:
        tr = tr.sample(config.max_train_rows, random_state=seed)

    hgb_p = {k: v for k, v in best_params.get(("hgb", signal), {}).items() if k != "loss"}
    qmodels = {}
    for q, name in [(.1, "lo"), (.5, "mid"), (.9, "hi")]:
        # early_stopping=False for the same reason as make_model: sklearn's automatic
        # internal holdout is drawn IID from a lagged panel, which leaks. These models
        # produce the 80% band, so a leaky stopping signal would show up as an interval
        # that is narrower than its own coverage guarantee.
        qmodels[name] = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, random_state=seed, early_stopping=False,
            **{**dict(max_iter=250, learning_rate=.07), **hgb_p}
        ).fit(tr[features], tr[target].values)

    va_lo = qmodels["lo"].predict(va[features])
    va_hi = qmodels["hi"].predict(va[features])
    conformity = np.maximum(va_lo - va[target].values, va[target].values - va_hi)
    finite = conformity[np.isfinite(conformity)]
    qhat = float(np.percentile(finite, 80)) if len(finite) else 0.0
    logging.info("conformal widening qhat = %.2f mmHg", qhat)
    return qmodels, qhat


# ---------------------------------------------------------------------------- serving

def _symptom_mechanism_map(models) -> dict:
    from src.utils.ml_utils.rule_engine.symptom_layer import SYMPTOM_MECHANISM
    return {k: SYMPTOM_MECHANISM.get(k.rsplit("_h", 1)[0], "group") for k in (models or {})}


def _symptom_red_flags(models) -> list:
    from src.utils.ml_utils.rule_engine.symptom_layer import SYMPTOM_RED_FLAG
    return sorted(k for k in (models or {}) if k.rsplit("_h", 1)[0] in SYMPTOM_RED_FLAG)


def _first_freezable(fit_rows, signal, config, features, best_params, winner) -> str:
    """The cheapest architecture that will actually freeze, as a fallback.

    Walks the registry in cost order rather than picking a hardcoded default: if the local
    winner has to be replaced, the replacement should be the cheapest thing that can do the
    job, which is the same principle the tie-break uses.
    """
    probe = fit_rows.head(800)
    for name in sorted((k for k in enabled_kinds(getattr(config, "model_tier", None))
                    if is_freezable(k)),
                       key=lambda k: (COST_RANK.get(cost_of(k), 1), k)):
        try:
            if fit_final(name, probe, signal, config.horizons[0], config, features,
                         best_params, seed=config.seed) is not None:
                return name
        except Exception:                                             # noqa: BLE001
            continue
    return "ridge"


class BPPredictor:
    """Frozen serving artifact. The only state is the bundle; no module globals are read."""

    def __init__(self, bundle: dict):
        # The bundle stores plain data plus sklearn estimators only. Project-defined classes
        # are reconstructed here rather than pickled, so a service process can load the
        # bundle without carrying the training package's class graph.
        self.b = bundle
        self.config = bundle["config"]
        self.fb = CausalFeatureBuilder(self.config)
        o = bundle["offset"]
        self.offset = OffsetModel(
            self.config, warm=o["warm"], k=o["k"], q=o["q"],
            cohort_prior=o["cohort_prior"], global_prior=o["global_prior"])

    # ---------- construction ----------
    @classmethod
    def build(cls, F, features, config, winner, best_params, offset_model,
              qmodels, qhat, interval_signal, interval_horizon,
              detector, imputer, dense_cols, detector_cut,
              shipped: dict = None, symptom_models: dict = None,
              symptom_cuts: dict = None) -> "BPPredictor":
        """Refit the forecasters on train+val, then freeze everything into one bundle.

        `shipped` maps signal -> ('learned', kind) | ('baseline', name) and is the ship
        decision made real: a signal whose learned model did not clear the bar serves
        the baseline that beat it, not the learned model that lost.
        """
        fit_mask = F.split.isin(["train", "val"])
        shipped = shipped or {s: ("learned", winner.get(s, "hgb")) for s in config.signals}

        forecasters, unservable = {}, {}
        for s in config.signals:
            family, name = shipped.get(s, ("learned", winner.get(s, "hgb")))
            served = name
            if family == "learned" and not is_freezable(name):
                # The bake-off winner refits per patient and has no single fitted object to
                # store. Something has to ship, so the cheapest freezable candidate does --
                # but the substitution is recorded in the bundle and raised as a safety-gate
                # WARN, because a scorecard naming one model while another serves every
                # request is exactly the kind of drift nobody notices until it matters.
                served = _first_freezable(F[fit_mask], s, config, features, best_params,
                                          winner)
                unservable[s] = name
                logging.warning("%s: selected %s cannot be frozen (scope=local); shipping "
                                "%s instead", s, name, served)
            for h in config.horizons:
                tgt = f"y_{s}_h{h}"
                if tgt not in F.columns:
                    continue
                if family == "baseline":
                    forecasters[(s, h)] = BaselineForecaster(name, s, h)
                    continue
                d = F[fit_mask & F[tgt].notna()]
                if len(d) < 200:
                    continue
                if len(d) > config.max_train_rows:
                    d = d.sample(config.max_train_rows, random_state=config.seed)
                # fit_final knows every wrapper shape -- delta, window, classical, ensemble
                # -- so a winner from any family freezes through one call. The plain
                # make_model path stays as the fallback for a kind fit_final declines.
                frozen = fit_final(served, d, s, h, config, features, best_params,
                                   seed=config.seed)
                forecasters[(s, h)] = frozen if frozen is not None else make_model(
                    served, **best_params.get((TUNE_ALIAS.get(served, served), s), {})
                ).fit(d[features], d[tgt].values)
        logging.info("shipped forecasters: %s",
                     {s: f"{f}:{n}" for s, (f, n) in shipped.items()})
        if unservable:
            logging.warning("selected-but-unservable: %s", unservable)

        bundle = dict(
            model_version=f"hemobp-bp-{config.run_id}", run_id=config.run_id,
            config=config, feature_names=features,
            selected_family=winner, shipped=shipped, forecasters=forecasters,
            # Non-empty means the model that serves is NOT the model the scorecard selected.
            selected_but_unservable=unservable,
            interval=dict(models=qmodels, qhat=qhat, signal=interval_signal,
                          horizon=interval_horizon, fit_on="train", calibrated_on="val"),
            offset=dict(warm=offset_model.warm, k=offset_model.k, q=offset_model.q,
                        cohort_prior=offset_model.cohort_prior,
                        global_prior=offset_model.global_prior,
                        label="capped shrinkage blend"),
            # Model 4. `symptom_labels_synthetic` is asserted by a safety gate and echoed
            # in every API response: it is the only thing stopping a generated hazard model
            # from being read as a clinical prediction.
            symptom_models=symptom_models or {}, symptom_cuts=symptom_cuts or {},
            symptom_labels_synthetic=True,
            symptom_mechanism=_symptom_mechanism_map(symptom_models),
            symptom_red_flags=_symptom_red_flags(symptom_models),
            detector=dict(model=detector, name=getattr(detector, "name", "d_isoforest"),
                          imputer=imputer, cols=dense_cols, cut=detector_cut,
                          budget_pct=config.alert_budget_pct, warn_window=config.warn_window,
                          event_quantile=config.event_quantile),
        )
        return cls(bundle)

    # ---------- persistence ----------
    def save(self, path: str) -> str:
        """Write the bundle atomically.

        `save_object` was already made write-then-rename for `final_model/model.pkl`; this
        path was left writing in place, and it is the one the serving registry falls back to
        when nothing has been promoted yet. A reader that stats the file mid-write sees a
        fresh mtime and a truncated joblib archive. That is not hypothetical -- it surfaced
        as a flaky API contract test while a training run was writing this exact file.
        """
        import os

        import joblib
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        joblib.dump(self.b, tmp)
        os.replace(tmp, path)          # atomic on the same filesystem, POSIX and Windows
        return path

    @classmethod
    def load(cls, path: str) -> "BPPredictor":
        import joblib
        return cls(joblib.load(path))

    # ---------- inference ----------
    def detector_score(self, F: pd.DataFrame, sbp_history: pd.Series = None) -> np.ndarray:
        """Model 3 over one or many feature rows -- the only path that scores the detector.

        Serving and the dashboard's trend view both come through here, so the number in
        the advisory and the number on the chart cannot come from different detectors.
        The bundle carries a ServingDetector; the fallback covers a bundle frozen before
        the detector became selectable, which held a bare score_samples estimator.
        """
        det = self.b["detector"]
        model = det.get("model")
        if hasattr(model, "score") and hasattr(model, "kind"):
            return np.asarray(model.score(F, sbp_history), float)
        return -np.asarray(
            model.score_samples(det["imputer"].transform(F.reindex(columns=det["cols"]))),
            float)

    def _tier(self, n: int) -> str:
        c = self.config
        if n < c.cold_start_min_readings:
            return "cold_start"
        return "bootstrapping" if n < c.steady_state_readings else "steady"

    def predict(self, history: pd.DataFrame, as_of=None) -> dict:
        """One patient's raw session history -> a serialisable advisory."""
        try:
            t0 = time.perf_counter()
            c, b = self.config, self.b
            g = history.sort_values("ts").copy()
            pid = str(g.patient_id.iloc[0])
            n_obs = int(g.sbp.notna().sum())
            tier = self._tier(n_obs)
            age = float(g.age.iloc[0]) if "age" in g and pd.notna(g.age.iloc[0]) else 65.0
            male = int(g.is_male.iloc[0]) if "is_male" in g else 0

            pers = self.offset.threshold_for(g.sbp, age, male)
            last_ts = g.ts.max()
            now = pd.Timestamp(as_of) if as_of is not None else last_ts
            age_days = float((now - last_ts).total_seconds() / 86400.0)
            out = dict(patient_id=pid, as_of=str(now),
                       model_version=b["model_version"], n_observations=n_obs,
                       confidence_tier=tier, personalisation=pers, forecast={},
                       early_warning=None, emergency_floor_mmHg=c.emergency_floor_mmHg,
                       staleness=dict(last_reading=str(last_ts),
                                      days_since_last_reading=round(age_days, 1),
                                      max_forecast_age_days=STALE_FORECAST_MAX_DAYS))

            if tier == "cold_start":
                out["note"] = (f"fewer than {c.cold_start_min_readings} readings; cohort "
                               f"threshold only, no forecast issued")
                out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                return out

            # A forecast built on a history this old asserts a continuity the data does not
            # support: every lag, window and EWMA describes a patient who has since been
            # unobserved for longer than the engine's own staleness threshold. Refusing is
            # the honest answer, and it mirrors the cold-start return directly above --
            # personalisation still stands, because a threshold is a property of the
            # patient's band, not of how recently they logged.
            if age_days > STALE_FORECAST_MAX_DAYS:
                out["confidence_tier"] = "stale"
                out["note"] = (f"last reading is {age_days:.0f} days old, beyond the "
                               f"{STALE_FORECAST_MAX_DAYS}-day limit; personalised "
                               f"threshold only, no forecast and no early warning issued")
                out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                return out

            row = self.fb.transform_for_inference(g, as_of=as_of)
            X = row.reindex(columns=b["feature_names"])
            med_gap = float(g.ts.diff().dt.days.median() or 2)

            for (sig, h), mdl in b["forecasters"].items():
                try:
                    p = float(mdl.predict(X)[0])
                except Exception:
                    continue
                # A baseline is a closed form over lag columns, so a short history
                # yields NaN rather than raising. NaN is not valid JSON, so it is
                # dropped here instead of being serialised into the advisory.
                if not np.isfinite(p):
                    continue
                # h is the target shift; the reading being predicted is h+1 readings after
                # the last one observed, because the features at the placeholder row see
                # only up to the newest actual reading. Reporting h here understated every
                # horizon by one gap -- "in ~2 days" for a ~4-day forecast.
                out["forecast"].setdefault(sig, {})[f"h{h}"] = dict(
                    point=round(p, 1), readings_ahead=h + 1, steps_ahead=h + 1,
                    days_ahead_est=round((h + 1) * med_gap, 1))

            iv = b["interval"]
            node = out["forecast"].get(iv["signal"], {}).get(f"h{iv['horizon']}")
            if node is not None and iv["models"]:
                lo = float(iv["models"]["lo"].predict(X)[0]) - iv["qhat"]
                hi = float(iv["models"]["hi"].predict(X)[0]) + iv["qhat"]
                node.update(lo80=round(lo, 1), hi80=round(hi, 1),
                            interval_basis=f"quantile GBM fit on {iv['fit_on']}, "
                                           f"conformal on {iv['calibrated_on']}")

            det = b["detector"]
            score = float(np.ravel(self.detector_score(row, g.sbp))[0])
            out["early_warning"] = dict(
                detector=det.get("name", "d_isoforest"),
                score=(round(score, 4) if np.isfinite(score) else None),
                cut=round(det["cut"], 4),
                flagged=bool(np.isfinite(score) and score >= det["cut"]),
                budget_pct=det["budget_pct"],
                event_definition=(f"SBP exceeds this patient's own p"
                                  f"{int(det['event_quantile'] * 100)} within the next "
                                  f"{det['warn_window']} sessions"),
                est_lead_days=round(det["warn_window"] * med_gap, 1))

            if tier == "bootstrapping":
                out["note"] = (f"{n_obs} readings, below the {c.steady_state_readings}-reading "
                               f"steady state; cohort-weighted, render a low-confidence badge")
            out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return out
        except Exception as e:
            raise CustomException(e, sys)


def explain_prediction(model, X_row, feature_names, reference, top_n: int = 5) -> dict:
    """Top-N contributing features for a single prediction.

    Local perturbation attribution: how much does the prediction move when each feature is
    replaced by its TRAINING median? Model-agnostic, cheap enough to run per advisory, and
    honest about being an approximation of a Shapley value rather than one.

    `reference` must be the training-set medians. Using the row's own values as the reference
    makes every perturbation a no-op and every contribution exactly zero -- which looks like a
    working explainer right up until someone reads the output.
    """
    base = float(model.predict(X_row)[0])
    contrib = {}
    for f in feature_names:
        ref = reference.get(f, np.nan)
        pert = X_row.copy()
        pert[f] = ref if np.isfinite(ref) else 0.0
        contrib[f] = base - float(model.predict(pert)[0])
    s = pd.Series(contrib)
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    return s.head(top_n).round(3).to_dict()


def champion_challenger(F, features, config, champ_kind: str, chall_kind: str,
                        signal: str = "sbp") -> pd.DataFrame:
    """Score an incumbent and a candidate on identical rows, so the delta is paired."""
    target = f"y_{signal}_h{config.horizons[0]}"
    if target not in F.columns:
        return pd.DataFrame()
    d = F[F[target].notna()]
    tr, te = d[d.split == "train"], d[d.split == "test"]
    if len(tr) < 200 or len(te) < 40:
        return pd.DataFrame()
    if len(tr) > config.max_train_rows:
        tr = tr.sample(config.max_train_rows, random_state=config.seed)
    rows = []
    for label, kind in (("champion", champ_kind), ("challenger", chall_kind)):
        m = make_model(kind).fit(tr[features], tr[target].values)
        r = evaluate(te[target].values, m.predict(te[features]), te[f"{signal}_lag1"].values,
                     model=kind, role=label, signal=signal)
        if r:
            rows.append(r)
    return pd.DataFrame(rows)
