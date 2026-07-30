"""How the forecaster degrades when patients skip sessions.

The pipeline never imputes a skipped session. That is the right call -- a session that did
not happen has no blood pressure, and inventing one puts a fabricated reading into the lag
window that every downstream feature reads. But "correct in principle" is not evidence, and
the users who skip sessions are exactly the users an alerting system exists for.

So: drop a fraction of each patient's sessions, rebuild the causal features from what is
left, refit the shipped forecaster, and measure. The dropout is seeded and per-patient, so
the sweep is reproducible and every patient contributes to every fraction rather than whole
patients dropping out at the high fractions.

Rebuilding features is the expensive part and also the point. Simply masking feature columns
would answer a different, easier question: it would leave the lag structure intact and only
hide values. Dropping the session changes what `_lag1` MEANS -- the previous reading is now
two sessions ago -- which is what actually happens when a patient skips.
"""

import sys

import numpy as np
import pandas as pd

from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.utils.ml_utils.feature.causal_features import CausalFeatureBuilder
from src.utils.ml_utils.metric.regression_metric import evaluate
from src.utils.ml_utils.model.estimator import BaselineForecaster, make_model


def _thin(panel: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Drop `frac` of each patient's sessions, keeping the first and last.

    Per-patient rather than global so the cohort composition is held fixed: a global sample
    would quietly drop short-history patients entirely at the high fractions and report the
    resulting easier problem as robustness.
    """
    if frac <= 0:
        return panel.copy()
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in panel.groupby("series_id", sort=False):
        g = g.sort_values("ts")
        n = len(g)
        if n <= 3:
            keep.append(g)
            continue
        # The endpoints anchor the timeline: dropping the last session would shorten every
        # patient's history rather than perforate it, which is a different experiment.
        middle = np.arange(1, n - 1)
        n_drop = int(round(frac * len(middle)))
        drop = set(rng.choice(middle, size=n_drop, replace=False)) if n_drop else set()
        keep.append(g.iloc[[i for i in range(n) if i not in drop]])
    return pd.concat(keep, ignore_index=True)


def missingness_sweep(panel: pd.DataFrame, features: list, config, shipped: dict,
                      best_params: dict, signal: str = "sbp") -> pd.DataFrame:
    """Refit-and-score the shipped forecaster across session-dropout fractions."""
    try:
        h = config.horizons[0]
        target = f"y_{signal}_h{h}"
        family, kind = (shipped or {}).get(signal, ("learned", "ridge"))
        rows = []

        for frac in config.missingness_fracs:
            thinned = _thin(panel, float(frac), config.seed)
            # `step` is renumbered because it indexes position in the patient's own
            # timeline, and every rolling window and cumcount downstream assumes it is
            # contiguous. `split` is deliberately NOT recomputed: keeping each row's
            # original label holds the test set fixed across fractions, so the MAE
            # differences below are caused by the dropout and not by a moving test set.
            thinned["step"] = thinned.groupby("series_id").cumcount()
            F = CausalFeatureBuilder(config).transform(thinned)
            d = F[F[target].notna()]
            tr, te = d[d.split == "train"], d[d.split == "test"]
            if len(tr) < 200 or len(te) < 50:
                rows.append(dict(drop_frac=float(frac), n_rows=len(d), n_train=len(tr),
                                 n_test=len(te), MAE=np.nan,
                                 note="too few rows to score at this fraction"))
                continue
            if len(tr) > config.max_train_rows:
                tr = tr.sample(config.max_train_rows, random_state=config.seed)

            cols = [c for c in features if c in F.columns]
            if family == "baseline":
                # Baselines read raw lag columns straight off the frame, and feature
                # selection may legitimately have dropped some of them from `features`.
                # Hand them the whole frame; BASELINE_SPECS already tolerates an absent
                # column by returning NaN for that baseline.
                m, X_te = BaselineForecaster(kind, signal, h), te
            else:
                m = make_model(kind, **(best_params or {}).get((kind, signal), {}))
                m.fit(tr[cols], tr[target].to_numpy(float))
                X_te = te[cols]
            pred = np.asarray(m.predict(X_te), dtype=float)
            sc = evaluate(te[target].to_numpy(float), pred)
            if not sc:
                # evaluate() returns {} below 10 finite pairs rather than a noisy MAE.
                rows.append(dict(drop_frac=float(frac), n_rows=len(d), n_train=len(tr),
                                 n_test=len(te), MAE=np.nan,
                                 note="fewer than 10 finite prediction/target pairs"))
                continue
            rows.append(dict(drop_frac=float(frac), n_rows=len(d), n_train=len(tr),
                             n_test=len(te), n_features=len(cols),
                             MAE=round(float(sc["MAE"]), 3),
                             RMSE=round(float(sc["RMSE"]), 3),
                             coverage=round(float(len(d) / max(len(F), 1)), 4),
                             model=f"{family}:{kind}", note=""))

        out = pd.DataFrame(rows)
        if len(out) and out.MAE.notna().any():
            base = out.dropna(subset=["MAE"]).sort_values("drop_frac").MAE.iloc[0]
            out["delta_MAE"] = (out.MAE - base).round(3)
            logging.info("missingness sweep on %s:h%d -> %s", signal, h,
                         ", ".join(f"{100 * r.drop_frac:.0f}%={r.MAE:.2f}"
                                   for r in out.dropna(subset=["MAE"]).itertuples()))
        return out
    except Exception as e:
        raise CustomException(e, sys)
