"""Training-time evidence for the chained symptom path: cuts, the shift it causes, and a board.

## The sequencing problem this solves

The obvious home for this is inside `train_symptom_heads`. It cannot live there: the heads are
trained at model_trainer section 7b and `BPPredictor.build` -- which freezes the forecasters --
is section 9. At head-training time there is nothing to forecast with. So this runs as a
POST-BUILD stage against the frozen bundle, which is also the more honest place: it measures
the forecaster that will actually serve rather than a sweep artefact.

## Why the rows are built per-row rather than by substituting a panel

The tempting shortcut is to replace every reading in the panel with its forecast and run one
vectorised `transform`. That is much faster and it measures the wrong thing. At serving, only
the LAST reading is a forecast; everything before it is observed. A fully substituted panel
gives `sbp_mean7` seven forecasts where serving gives it six observations and one forecast, so
the variance collapse -- the very thing the shift report exists to quantify -- would be
overstated several times over.

So each sampled row is reconstructed the way serving does it: observed history up to s-2,
the forecast for s-1 appended, features rebuilt. That is one `transform_for_inference` per
sampled row, which is why this samples rather than sweeping the whole split.

## What the numbers mean

Nothing here measures clinical accuracy. The labels come from `symptom_layer.py`, so every
figure below describes how well an estimator recovers a generator this repository wrote. The
board is still worth having: it answers whether chaining recovers MORE of that generator than
scoring the observed present does, which is a real question about the architecture even when
the labels are not real.
"""

import numpy as np
import pandas as pd

from src.logging.logger import logging
from src.utils.ml_utils.metric.timeseries_metric import DriftMonitor
from src.utils.ml_utils.model.symptom_chain import chained_history

#: Rows sampled per split. Each costs one feature build, so this is the wall-clock dial.
CHAIN_SAMPLE_ROWS: int = 1500

#: A patient needs this many readings before a chained row is even constructible (history up
#: to s-2, plus the appended forecast, plus enough left for the rolling windows to mean much).
MIN_HISTORY: int = 12


def _patient_histories(panel: pd.DataFrame) -> dict:
    cols = [c for c in panel.columns if c not in ("split", "patient_split")]
    return {sid: g[cols].sort_values("ts").reset_index(drop=True)
            for sid, g in panel.groupby("series_id", sort=False)}


def build_chained_rows(panel, F, predictor, features, sample_idx, seed: int = 42):
    """Feature rows as the CHAINED serving path would build them, for the sampled positions.

    Returns `(chained_frame, kept_positions)`. `kept_positions` indexes back into
    `sample_idx`, because a row whose patient is too short is dropped rather than faked.
    """
    fc_model = (predictor.b.get("forecasters") or {}).get(("sbp", 0))
    if fc_model is None:
        return pd.DataFrame(), []

    hists = _patient_histories(panel)
    rows, kept = [], []
    for pos in sample_idx:
        sid = F.series_id.iloc[pos]
        step = int(F.step.iloc[pos])
        h = hists.get(sid)
        if h is None or step < MIN_HISTORY or step >= len(h):
            continue
        # Observed history up to s-2, then the forecast for s-1 appended. The forecast is the
        # shipped model's own prediction of reading s-1 from features that read <= s-2, which
        # is exactly what a client would have had at that moment.
        past = h.iloc[:step - 1]
        try:
            x_prev = F.iloc[[pos - 1]][features]
            yhat = float(np.asarray(fc_model.predict(x_prev), dtype=float)[0])
            if not np.isfinite(yhat):
                continue
            fake_fc = {"sbp": {"h0": {"point": yhat, "days_ahead_est": None}}}
            ext = chained_history(past, fake_fc, upto_h=0)
            rows.append(predictor.fb.transform_for_inference(ext).iloc[0])
            kept.append(pos)
        except Exception:                                             # noqa: BLE001
            continue
    if not rows:
        return pd.DataFrame(), []
    return pd.DataFrame(rows).reset_index(drop=True), kept


def chain_shift(F_obs: pd.DataFrame, F_chain: pd.DataFrame, features) -> pd.DataFrame:
    """PSI between the observed and chained feature distributions.

    Reuses `DriftMonitor`, the same instrument the pipeline points at deployment drift, turned
    inward at a shift the architecture creates on purpose. The expected signature is the
    variance family -- `sbp_std3`, `sbp_range3`, `sbp_vol_ratio`, `sbp_z` -- collapsing,
    because substituting a conditional mean for a draw removes spread by construction. Seeing
    that in the report is confirmation the rows were built correctly; NOT seeing it would mean
    something is wrong with the construction, not that the shift is absent.
    """
    cols = [c for c in features if c in F_obs.columns and c in F_chain.columns]
    if not cols or F_chain.empty:
        return pd.DataFrame()
    try:
        return DriftMonitor(F_obs[cols], cols).feature_drift(F_chain[cols])
    except Exception as exc:                                          # noqa: BLE001
        logging.warning("chain shift report unavailable: %s", exc)
        return pd.DataFrame()


def _score(models, key, X):
    try:
        return np.asarray(models[key].predict_proba(X)[:, 1], dtype=float)
    except Exception:                                                 # noqa: BLE001
        return None


def chained_cuts(predictor, F_chain, kept, F, config) -> dict:
    """Operating cuts for the CHAINED path, chosen on chained probabilities.

    The observed-row cut is a percentile of the probabilities the head produced on observed
    rows. A chained row is a different input distribution, so that percentile does not carry
    its alert-budget meaning across -- reusing it silently breaks the budget the
    patient-disjoint calibration/threshold split exists to protect. This is the same
    construction on the distribution the cut will actually be applied to.
    """
    models = predictor.b.get("symptom_models") or {}
    if F_chain.empty or not models:
        return {}
    budget = float(config.alert_budget_pct)
    X = F_chain.reindex(columns=predictor.b["feature_names"])
    out = {}
    for key in [k for k in models if k.endswith("_h0")]:
        p = _score(models, key, X)
        if p is None or not np.isfinite(p).any():
            continue
        out[key] = float(np.percentile(p[np.isfinite(p)], 100 - budget))
    return out


def chain_board(predictor, F, F_chain, kept, config) -> pd.DataFrame:
    """Direct vs chained vs the patient's own rate, on identical rows.

    The third column is the one that decides whether any of this is worth serving: a head that
    cannot beat quoting the patient's own 30-session rate is not earning its place, however it
    is scored.
    """
    models = predictor.b.get("symptom_models") or {}
    if F_chain.empty or not models or not kept:
        return pd.DataFrame()

    obs = F.iloc[kept].reset_index(drop=True)
    Xo = obs.reindex(columns=predictor.b["feature_names"])
    Xc = F_chain.reindex(columns=predictor.b["feature_names"])
    reds = set(predictor.b.get("symptom_red_flags") or [])
    mech = predictor.b.get("symptom_mechanism") or {}

    rows = []
    for key in sorted(k for k in models if k.endswith("_h0")):
        base = key.rsplit("_h", 1)[0]
        tgt = f"y_sym_{base}_h0"
        if tgt not in obs.columns:
            continue
        y = obs[tgt].to_numpy(dtype=float)
        pd_, pc = _score(models, key, Xo), _score(models, key, Xc)
        if pd_ is None or pc is None:
            continue
        m = np.isfinite(y) & np.isfinite(pd_) & np.isfinite(pc)
        if m.sum() < 50 or y[m].sum() < 5:
            continue
        yy, a, b = y[m], pd_[m], pc[m]

        rate_col = f"sym_{base}_rate30"
        personal = (obs[rate_col].to_numpy(dtype=float)[m]
                    if rate_col in obs.columns else np.full(m.sum(), yy.mean()))
        personal = np.where(np.isfinite(personal), personal, yy.mean())

        # Paired on identical rows, the same discipline the forecaster's ship rule uses: two
        # marginal Brier scores can overlap while the per-row difference is decisive.
        from src.utils.ml_utils.metric.regression_metric import paired_delta
        d_direct, d_chain = (a - yy) ** 2, (b - yy) ** 2
        try:
            delta, (lo, hi), _ = paired_delta(d_direct, d_chain)
        except Exception:                                             # noqa: BLE001
            delta, lo, hi = np.nan, np.nan, np.nan

        rows.append(dict(
            symptom=base, mechanism=mech.get(key), red_flag=base in reds,
            n=int(m.sum()), base_rate=round(float(yy.mean()), 5),
            brier_direct=round(float(d_direct.mean()), 5),
            brier_chained=round(float(d_chain.mean()), 5),
            brier_personal_rate=round(float(((personal - yy) ** 2).mean()), 5),
            paired_delta=round(float(delta), 6) if np.isfinite(delta) else None,
            ci_lo=round(float(lo), 6) if np.isfinite(lo) else None,
            ci_hi=round(float(hi), 6) if np.isfinite(hi) else None,
            chained_wins=bool(np.isfinite(lo) and lo > 0),
            beats_personal_rate=bool(d_chain.mean() < ((personal - yy) ** 2).mean()),
            labels="SYNTHETIC"))
    return pd.DataFrame(rows)


def run_chain_evaluation(panel, F, predictor, features, config, seed: int = 42):
    """The whole post-build stage. Returns `(board, shift, cuts)`; any may be empty."""
    eval_rows = F[(F.split == "test")] if "split" in F else F
    if not len(eval_rows):
        return pd.DataFrame(), pd.DataFrame(), {}
    rng = np.random.default_rng(seed)
    positions = np.flatnonzero(F.split.to_numpy() == "test") if "split" in F \
        else np.arange(len(F))
    if positions.size > CHAIN_SAMPLE_ROWS:
        positions = rng.choice(positions, CHAIN_SAMPLE_ROWS, replace=False)
    positions = np.sort(positions)

    F_chain, kept = build_chained_rows(panel, F, predictor, features, positions, seed=seed)
    if F_chain.empty:
        logging.warning("chained evaluation produced no rows; skipping")
        return pd.DataFrame(), pd.DataFrame(), {}
    logging.info("  chained evaluation: %d of %d sampled rows reconstructed",
                 len(kept), len(positions))

    shift = chain_shift(F.iloc[kept].reset_index(drop=True), F_chain, features)
    cuts = chained_cuts(predictor, F_chain, kept, F, config)
    board = chain_board(predictor, F, F_chain, kept, config)
    return board, shift, cuts
