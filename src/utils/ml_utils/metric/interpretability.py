"""Block permutation importance and serving-parity checks.

Two things live here that a single-column permutation cannot do.

BLOCK PERMUTATION. Permuting one feature at a time understates the importance of every
feature that has a surviving twin: the model reads the twin, the prediction barely moves, and
a column that genuinely drives the forecast reports near-zero. On this panel `sbp_mean7` and
`sbp_mean14` correlate above 0.97, so single-column permutation would call both unimportant
and the report would say the model runs on nothing. Feature selection already computed which
columns collapse into which cluster; permuting the whole cluster with ONE shared index is
what makes the number mean "how much does this signal matter".

WALK-FORWARD PARITY. The offline scorecard and the serving path build features by different
code (`transform` over a panel versus `transform_for_inference` over a history). They are
supposed to agree. Nothing checked that they did, and a divergence there is invisible in every
offline metric while being wrong in every served advisory.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.constants.training_pipeline import SEED
from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.utils.ml_utils.feature.causal_features import feature_group


def block_importance(model, X: pd.DataFrame, y: np.ndarray, blocks: dict,
                     n_repeats: int = 3, seed: int = SEED) -> pd.DataFrame:
    """MAE increase when each block of correlated features is permuted together.

    `blocks` maps a label to the list of columns permuted as a unit. One shared permutation
    index across the whole block, not an independent shuffle per column: independent shuffles
    would also destroy the correlation structure between the block's members, which measures
    something nobody asked about.
    """
    try:
        rng = np.random.default_rng(seed)
        base = float(mean_absolute_error(y, model.predict(X)))
        rows = []
        for label, cols in blocks.items():
            cols = [c for c in cols if c in X.columns]
            if not cols:
                continue
            deltas = []
            for _ in range(n_repeats):
                idx = rng.permutation(len(X))
                Xp = X.copy()
                for c in cols:
                    Xp[c] = X[c].to_numpy()[idx]
                deltas.append(float(mean_absolute_error(y, model.predict(Xp))) - base)
            rows.append(dict(block=label, n_features=len(cols),
                             group=feature_group(label),
                             mae_increase=round(float(np.mean(deltas)), 4),
                             sd=round(float(np.std(deltas)), 4),
                             members=", ".join(cols[:6]) + ("…" if len(cols) > 6 else "")))
        out = pd.DataFrame(rows).sort_values("mae_increase", ascending=False)
        if len(out):
            worth = out[out.mae_increase > 0.10]
            logging.info("block importance: %d/%d blocks worth more than 0.10 mmHg | top: %s",
                         len(worth), len(out),
                         ", ".join(f"{r.block} {r.mae_increase:+.2f}"
                                   for r in out.head(4).itertuples()))
        return out
    except Exception as e:
        raise CustomException(e, sys)


def error_concentration(y, pred, series_id) -> pd.DataFrame:
    """How much of the total error the worst patients carry.

    A mean MAE of 14 mmHg can mean everyone is 14 mmHg out, or that most patients are at 8
    and a tail is at 40. Those call for completely different responses, and the mean cannot
    tell them apart.
    """
    d = pd.DataFrame({"sid": series_id, "e": np.abs(np.asarray(pred, float)
                                                    - np.asarray(y, float))}).dropna()
    if not len(d):
        return pd.DataFrame()
    per = d.groupby("sid").e.mean().sort_values(ascending=False)
    total = float(per.sum())
    rows = []
    for q in (0.05, 0.10, 0.25, 0.50):
        k = max(1, int(round(q * len(per))))
        rows.append(dict(worst_share=q, n_patients=k,
                         pct_of_total_error=round(100 * float(per.head(k).sum()) / total, 1),
                         mean_MAE=round(float(per.head(k).mean()), 2)))
    return pd.DataFrame(rows)


def walk_forward_parity(predictor, panel: pd.DataFrame, offline_mae: float,
                        n_patients: int = 12, cuts_per_patient: int = 4,
                        tolerance: float = 2.0, seed: int = SEED) -> dict:
    """Re-serve at rolling cut points and compare against the offline test MAE.

    The only check that catches train/serve feature drift. Everything else in the pipeline
    scores the OFFLINE path; this one drives `predictor.predict` exactly as the API does and
    asks whether the answer is the same.
    """
    try:
        rng = np.random.default_rng(seed)
        sids = panel.series_id.dropna().unique()
        if not len(sids):
            return {"status": "SKIP", "detail": "no patients"}
        pick = rng.choice(sids, size=min(n_patients, len(sids)), replace=False)
        errs, n_calls = [], 0
        for sid in pick:
            g = panel[panel.series_id == sid].sort_values("ts").reset_index(drop=True)
            if len(g) < 25:
                continue
            lo, hi = int(len(g) * 0.7), len(g) - 2
            if hi <= lo:
                continue
            for cut in np.linspace(lo, hi, cuts_per_patient, dtype=int):
                hist = g.iloc[:cut]
                try:
                    a = predictor.predict(hist)
                except Exception:                                     # noqa: BLE001
                    continue
                n_calls += 1
                node = (a.get("forecast") or {}).get("sbp") or {}
                if not node:
                    continue
                first = node[sorted(node)[0]]
                actual = float(g.sbp.iloc[cut])
                if np.isfinite(first.get("point", np.nan)) and np.isfinite(actual):
                    errs.append(abs(first["point"] - actual))

        if len(errs) < 10:
            return {"status": "SKIP", "n": len(errs), "calls": n_calls,
                    "detail": "too few servable cut points to compare"}
        serve_mae = float(np.mean(errs))
        gap = abs(serve_mae - offline_mae) if np.isfinite(offline_mae) else np.nan
        ok = bool(np.isfinite(gap) and gap < tolerance)
        out = {"status": "PASS" if ok else "INVESTIGATE", "n": len(errs), "calls": n_calls,
               "serve_mae": round(serve_mae, 3),
               "offline_mae": round(float(offline_mae), 3) if np.isfinite(offline_mae) else None,
               "gap_mmHg": round(gap, 3) if np.isfinite(gap) else None,
               "tolerance": tolerance}
        logging.info("walk-forward parity: serve %.2f vs offline %.2f mmHg over %d re-serves "
                     "-> %s", serve_mae, offline_mae, len(errs), out["status"])
        return out
    except Exception as e:
        raise CustomException(e, sys)


def batch_latency(predictor, panel: pd.DataFrame, config, n_patients: int = 40,
                  seed: int = SEED) -> dict:
    """Median and p95 serving latency, and the realised alert rate, against the budget.

    `LATENCY_BUDGET_MS` was defined, surfaced on the config and then never read by anything.
    A budget nothing measures is a comment.
    """
    try:
        import time
        rng = np.random.default_rng(seed)
        sids = panel.series_id.dropna().unique()
        pick = rng.choice(sids, size=min(n_patients, len(sids)), replace=False)
        lat, flags, n = [], 0, 0
        for sid in pick:
            g = panel[panel.series_id == sid].sort_values("ts")
            if len(g) < 10:
                continue
            t0 = time.perf_counter()
            try:
                a = predictor.predict(g)
            except Exception:                                         # noqa: BLE001
                continue
            lat.append((time.perf_counter() - t0) * 1000.0)
            n += 1
            if (a.get("early_warning") or {}).get("flagged"):
                flags += 1
        if not lat:
            return {"status": "SKIP", "detail": "no scorable patients"}
        med, p95 = float(np.median(lat)), float(np.percentile(lat, 95))
        budget = float(config.latency_budget_ms)
        rate = flags / max(n, 1)
        out = {"status": "PASS" if p95 <= budget else "FAIL",
               "n": n, "median_ms": round(med, 1), "p95_ms": round(p95, 1),
               "budget_ms": budget,
               "realised_alert_rate": round(rate, 4),
               "alert_budget": round(config.alert_budget_pct / 100.0, 4)}
        logging.info("batch latency over %d patients: median %.1f ms, p95 %.1f ms "
                     "(budget %.0f) | alert rate %.1f%% against a %.0f%% budget -> %s",
                     n, med, p95, budget, 100 * rate, config.alert_budget_pct, out["status"])
        return out
    except Exception as e:
        raise CustomException(e, sys)


def bundle_round_trip(predictor, reloaded, history: pd.DataFrame) -> dict:
    """The reloaded bundle must produce the same advisory as the in-memory model.

    The trainer already reloads the bundle; it just never compared the two. A pickle that
    loads but scores differently -- a fitted object that did not survive serialisation, a
    lambda that silently became a default -- is exactly the failure this catches, and it is
    invisible everywhere else.
    """
    import json
    try:
        a = predictor.predict(history)
        b = reloaded.predict(history)
        drop = {"latency_ms"}
        ja = json.dumps({k: v for k, v in a.items() if k not in drop},
                        sort_keys=True, default=str)
        jb = json.dumps({k: v for k, v in b.items() if k not in drop},
                        sort_keys=True, default=str)
        same = ja == jb
        if not same:
            ka = {k for k in a if k not in drop}
            diff = sorted(k for k in ka | {k for k in b if k not in drop}
                          if json.dumps(a.get(k), sort_keys=True, default=str)
                          != json.dumps(b.get(k), sort_keys=True, default=str))
            logging.error("bundle round-trip MISMATCH on: %s", diff)
            return {"status": "FAIL", "differing_keys": diff}
        logging.info("bundle round-trip: the reloaded bundle matches the in-memory model")
        return {"status": "PASS"}
    except Exception as e:
        raise CustomException(e, sys)
