"""Point-forecast metrics, baselines and the ship decision.

Port of notebook sections 5 and 6B.5. The decision rule is the point of this module:
a learned model ships only when it beats the best baseline by more than the bootstrap
CI width. A smoother needs no registry, no drift monitor and no change-control plan.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.constants.training_pipeline import N_BOOT, SEED
from src.entity.artifact_entity import ForecastMetricArtifact
from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.utils.ml_utils.model.architectures import COST_RANK


def bootstrap_ci(fn, y: np.ndarray, p: np.ndarray,
                 n_boot: int = None, seed: int = SEED):
    """Point estimate plus a percentile bootstrap 95% interval for `fn`."""
    n_boot = int(n_boot or N_BOOT)
    rng = np.random.default_rng(seed)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 10:
        return np.nan, (np.nan, np.nan)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    vals = np.array([fn(y[i], p[i]) for i in idx])
    return float(fn(y, p)), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def evaluate(y, p, last=None, lo=None, hi=None, **meta) -> dict:
    """Point-forecast metrics with CI, bias, direction and (optionally) interval coverage."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 10:
        return {}
    mae, (lo_ci, hi_ci) = bootstrap_ci(mean_absolute_error, y, p)
    out = dict(meta, n=len(y), MAE=round(mae, 3), MAE_lo=round(lo_ci, 3), MAE_hi=round(hi_ci, 3),
               RMSE=round(float(np.sqrt(mean_squared_error(y, p))), 3),
               R2=round(float(r2_score(y, p)), 3), bias=round(float(np.mean(p - y)), 2),
               target_sd=round(float(np.std(y)), 2), unit="per patient-step")
    if last is not None:
        b = np.asarray(last, float)[m]
        ok = np.isfinite(b)
        dt, dp = y[ok] - b[ok], p[ok] - b[ok]
        nz = np.abs(dt) > 1e-9
        out["DirAcc"] = (round(float(np.mean(np.sign(dt[nz]) == np.sign(dp[nz]))), 3)
                         if nz.sum() and np.abs(dp).max() > 1e-9 else np.nan)
    if lo is not None:
        lo_a, hi_a = np.asarray(lo, float)[m], np.asarray(hi, float)[m]
        out["Cover80"] = round(float(np.mean((y >= lo_a) & (y <= hi_a))), 3)
        out["IntWidth"] = round(float(np.mean(hi_a - lo_a)), 1)
    return out


# Each baseline as a closed form over columns the feature matrix already carries. This is
# the single definition: `forecast_baselines` scores them and `BaselineForecaster` serves
# them, so a baseline that wins the ship decision cannot behave differently once shipped.
BASELINE_SPECS = {
    "persistence": lambda d, s, h: d[f"{s}_lag1"].to_numpy(float),
    "seasonal_naive_7": lambda d, s, h: (d[f"{s}_lag7"].to_numpy(float)
                                         if f"{s}_lag7" in d
                                         else np.full(len(d), np.nan)),
    "drift": lambda d, s, h: (d[f"{s}_lag1"].to_numpy(float)
                              + h * (d[f"{s}_lag1"].to_numpy(float)
                                     - d[f"{s}_lag2"].to_numpy(float))),
    "personal_mean": lambda d, s, h: d[f"{s}_base_mean"].to_numpy(float),
    "ewma": lambda d, s, h: d[f"{s}_ewm0.3"].to_numpy(float),
}


#: The feature columns each baseline READS. Kept beside BASELINE_SPECS so the two are edited
#: together, and consumed by feature selection: a baseline whose input has been selected away
#: is a shipped model that raises KeyError at serving. `_lag7` is omitted because
#: `seasonal_naive_7` already guards its own absence and falls back to NaN.
BASELINE_REQUIRED_SUFFIXES: tuple = ("_lag1", "_lag2", "_base_mean", "_ewm0.3")


def baseline_required_columns(signals) -> set:
    """Columns that must survive feature selection for a baseline to be shippable.

    `ship_decision` can rule that no learned model earned its place for a signal, and then a
    BaselineForecaster serves. It is a closed form over these columns, so if selection has
    dropped one the shipped model cannot run -- and `BPPredictor.predict` swallows the
    KeyError per forecaster, so the signal simply disappears from the advisory with no error
    anywhere. That is exactly what happened to dbp: MUST_KEEP protected sbp_ewm0.3 alone, the
    EWMA baseline shipped for dbp, and the API returned an sbp-only forecast while reporting
    a dbp gain of 0.335 mmHg in the ship decision.
    """
    return {f"{s}{suf}" for s in signals for suf in BASELINE_REQUIRED_SUFFIXES}


def forecast_baselines(df: pd.DataFrame, signal: str, h: int) -> dict:
    """The floor every learned model must clear, per signal and horizon."""
    return {name: fn(df, signal, h) for name, fn in BASELINE_SPECS.items()}


def ship_decision(learned_mae: float, baseline_mae: float, ci_width: float):
    """learned model ships iff (baseline MAE - learned MAE) > bootstrap CI width."""
    gain = baseline_mae - learned_mae
    return ("learned model" if gain > ci_width else "BASELINE"), gain


def paired_delta(err_a: np.ndarray, err_b: np.ndarray, n_boot: int = None,
                 seed: int = SEED):
    """Paired bootstrap on per-row error differences: mean delta, 95% CI, and n.

    Paired rather than independent because both models see the same rows; the unpaired
    interval is far wider than the evidence warrants. Inputs are per-row ABSOLUTE errors.

    The 30-row floor is higher than the 10 used elsewhere on purpose: a bootstrap over
    fewer than ~30 paired differences resamples the same handful of rows and reports a
    tight interval around whatever those rows happened to say, which is exactly the
    situation where a ship decision must not be made.
    """
    n_boot = int(n_boot or N_BOOT)
    a, b = np.asarray(err_a, float), np.asarray(err_b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if len(d) < 30:
        return np.nan, (np.nan, np.nan), int(len(d))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return (float(d.mean()),
            (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))),
            int(len(d)))


def get_forecast_score(row: dict) -> ForecastMetricArtifact:
    """Turn one `evaluate` row into a typed artifact."""
    try:
        return ForecastMetricArtifact(
            signal=row.get("signal", ""),
            horizon=int(row.get("horizon", 0)),
            model=row.get("model", ""),
            n=int(row.get("n", 0)),
            mae=float(row.get("MAE", np.nan)),
            mae_lo=float(row.get("MAE_lo", np.nan)),
            mae_hi=float(row.get("MAE_hi", np.nan)),
            rmse=float(row.get("RMSE", np.nan)),
            r2=float(row.get("R2", np.nan)),
            bias=float(row.get("bias", np.nan)),
            target_sd=float(row.get("target_sd", np.nan)),
            dir_acc=(float(row["DirAcc"]) if row.get("DirAcc") is not None
                     and np.isfinite(row.get("DirAcc", np.nan)) else None),
            split=row.get("split", "test"),
        )
    except Exception as e:
        raise CustomException(e, sys)


def tie_set(preds: dict, leader: str, n_boot: int = None, seed: int = SEED):
    """Architectures statistically indistinguishable from the leader, by paired bootstrap.

    `preds` maps model name -> (y, pred, lag1, series_id) on the SAME test rows, so the
    comparison can be paired. Pairing matters more here than usual: every candidate is
    predicting the same patients on the same days, so most of the variance in MAE is
    variance in the rows, not in the models. An unpaired comparison of two marginal
    confidence intervals treats that shared variance as evidence of a tie and hides real
    differences; the paired interval removes it.

    Returns (frame, tie_names). `ties_leader` is `lo <= 0 <= hi` -- the delta against the
    leader is not distinguishable from zero. `patient_win_rate` is reported and deliberately
    NOT used to gate: it answers "does this beat the leader for the MEDIAN patient", which is
    a different and often more honest question than the pooled mean, and it belongs in front
    of a human rather than inside an automatic rule.
    """
    if leader not in preds:
        return pd.DataFrame(), {leader}
    y_l, p_l, _, sid_l = preds[leader]
    ref = np.abs(np.asarray(p_l, float) - np.asarray(y_l, float))
    rows, ties = [], {leader}
    for name, (y, p, _lag, sid) in preds.items():
        err = np.abs(np.asarray(p, float) - np.asarray(y, float))
        if err.shape != ref.shape:
            continue
        delta, (lo, hi), n = paired_delta(err, ref, n_boot=n_boot, seed=seed)
        pp = pd.DataFrame({"sid": sid, "e": err, "eref": ref}).dropna()
        win = float((pp.groupby("sid").e.mean() < pp.groupby("sid").eref.mean()).mean()) \
            if len(pp) else np.nan
        tied = bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi)
        if tied:
            ties.add(name)
        rows.append(dict(model=name, delta_vs_leader=round(delta, 4) if np.isfinite(delta)
                         else np.nan,
                         ci_lo=round(lo, 4) if np.isfinite(lo) else np.nan,
                         ci_hi=round(hi, 4) if np.isfinite(hi) else np.nan,
                         n=n, ties_leader=tied,
                         patient_win_rate=round(win, 3) if np.isfinite(win) else np.nan))
    return pd.DataFrame(rows).sort_values("delta_vs_leader"), ties


def select_and_decide(R: pd.DataFrame, config, preds: dict = None):
    """Select on validation, break ties by COST, then apply the paired ship rule on test.

    Two rules, and the second is the one that usually gets dropped in a port:

    1. The winner is the cheapest member of the statistical tie set, not the lowest MAE.
       Two models that cannot be distinguished on the evidence are not equally good -- the
       cheaper one is, because it costs less to serve, less to explain and less to debug.
       Taking `idxmin()` instead reliably ships a stacked ensemble to win 0.02 mmHg that the
       confidence interval says is noise.

    2. Shipping requires beating the best baseline on a PAIRED bootstrap lower bound, not on
       a scalar gain against a marginal CI width. Both sides are averaged over the same
       horizons: scoring the learned model across h0..h2 against the single best baseline
       cell -- always h0, because MAE grows with horizon -- would make the learned model
       answer for the hard horizons while the baseline is credited with the easy one.

    Both rules now apply to EVERY signal. `preds` was collected for sbp alone, so dbp fell
    through to the marginal-CI fallback and to `idxmin()` on val -- the two behaviours this
    function exists to avoid. The fallback is still reached when a signal has no stored
    predictions at all, which is the honest degradation; it is no longer the normal path for
    half the outputs.
    """
    try:
        learned = R[(R.split == "val") & (R.family == "learned") & R.MAE.notna()]
        if learned.empty:
            return {}, pd.DataFrame(), pd.DataFrame()
        val = learned.groupby(["signal", "model"]).MAE.mean().unstack()
        cost_by_model = (R.dropna(subset=["MAE"]).groupby("model").cost.first().to_dict()
                         if "cost" in R.columns else {})

        winner, cmp_frames = {}, []
        for s in val.index:
            ranked = val.loc[s].dropna().sort_values()
            if ranked.empty:
                continue
            leader = str(ranked.index[0])
            ties = {leader}
            # Per-signal. `preds` is {signal: {model: (y, pred, lag1, series_id)}} and used to
            # be populated for sbp alone, so dbp got neither the tie set nor the cost
            # tie-break below -- its winner was whatever had the lowest val MAE.
            sp = (preds or {}).get(s) or {}
            if sp:
                cmp_df, ties = tie_set(sp, leader, n_boot=getattr(config, "n_boot", None),
                                       seed=config.seed)
                if len(cmp_df):
                    cmp_frames.append(cmp_df.assign(signal=s, leader=leader))
            cand = [m for m in ranked.index if m in ties] or [leader]
            # cheapest first, MAE as the tie-break within a cost class
            winner[s] = min(cand, key=lambda m: (COST_RANK.get(cost_by_model.get(m, "medium"),
                                                               1), float(ranked[m]), m))
            if winner[s] != leader:
                logging.info("%s: %s ties the leader %s and is cheaper (%s vs %s) -- shipped",
                             s, winner[s], leader, cost_by_model.get(winner[s]),
                             cost_by_model.get(leader))

        test = R[R.split == "test"]
        rows = []
        for s, sel_model in winner.items():
            sel = test[(test.signal == s) & (test.model == sel_model)]
            base = test[(test.signal == s) & (test.family == "baseline")]
            if not len(sel) or not len(base):
                continue
            horizons = sorted(set(sel.horizon))
            base = base[base.horizon.isin(horizons)]
            per_base = base.groupby("model").MAE.mean()
            covers = base.groupby("model").horizon.nunique() == len(horizons)
            per_base = per_base[covers[covers].index] if covers.any() else per_base
            if per_base.empty:
                continue
            best_name = str(per_base.idxmin())
            learned_mae = float(sel.MAE.mean())

            verdict, gain, lo, hi = None, np.nan, np.nan, np.nan
            sp = (preds or {}).get(s) or {}
            if sel_model in sp and best_name in sp:
                y, p, _l, _sid = sp[sel_model]
                yb, pb, _l2, _s2 = sp[best_name]
                delta, (lo, hi), _n = paired_delta(
                    np.abs(np.asarray(pb, float) - np.asarray(yb, float)),
                    np.abs(np.asarray(p, float) - np.asarray(y, float)),
                    n_boot=getattr(config, "n_boot", None), seed=config.seed)
                gain = delta
                # ships only if the whole interval is on the right side of zero
                verdict = "learned model" if np.isfinite(lo) and lo > 0 else "BASELINE"
            if verdict is None:
                ci_w = float(sel.MAE_hi.mean() - sel.MAE_lo.mean())
                verdict, gain = ship_decision(learned_mae, float(per_base.min()), ci_w)
                lo = hi = np.nan

            # The cold-start arm: how much worse the SHIPPED model is on a patient it has
            # never seen. Reported beside the ship decision because a model that wins on
            # seen patients and collapses on new ones is not a model a new user can use.
            ho = R[(R.split == "holdout_pt") & (R.signal == s) & (R.model == sel_model)]
            penalty = (float(ho.MAE.mean()) - float(sel.MAE.mean())
                       if len(ho) else np.nan)

            rows.append(dict(signal=s, selected=sel_model,
                             cost=cost_by_model.get(sel_model, ""),
                             learned_MAE=round(learned_mae, 3), best_baseline=best_name,
                             baseline_MAE=round(float(per_base.min()), 3),
                             gain_mmHg=round(gain, 3) if np.isfinite(gain) else np.nan,
                             paired_ci_lo=round(lo, 3) if np.isfinite(lo) else np.nan,
                             paired_ci_hi=round(hi, 3) if np.isfinite(hi) else np.nan,
                             holdout_pt_MAE=round(float(ho.MAE.mean()), 3) if len(ho)
                             else np.nan,
                             cold_start_penalty=round(penalty, 3) if np.isfinite(penalty)
                             else np.nan,
                             horizons=len(horizons), ship=verdict))
        cmp_out = (pd.concat(cmp_frames, ignore_index=True) if cmp_frames
                   else pd.DataFrame())
        return winner, pd.DataFrame(rows), cmp_out
    except Exception as e:
        raise CustomException(e, sys)


def shipped_forecasters(decision: pd.DataFrame, winner: dict) -> dict:
    """signal -> ('learned', kind) or ('baseline', name), honouring the ship decision.

    Without this the verdict is a report nobody reads: `build` froze the learned winner
    whatever the rule said, so a model that never cleared the bar still served every
    request.
    """
    out = {s: ("learned", k) for s, k in winner.items()}
    for r in decision.itertuples() if len(decision) else []:
        if str(r.ship).upper() == "BASELINE":
            out[r.signal] = ("baseline", r.best_baseline)
    return out
