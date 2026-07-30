"""The hybrid ML layer: features at t -> the tier the rule engine fires at t+h.

Port of notebook section 16. The engine's output is the ground truth this layer forecasts
and the vocabulary it translates into -- never something the ML layer produces.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.constants.training_pipeline import SEED
from src.logging.logger import logging


def build_classifier_frame(F: pd.DataFrame, alerts: pd.DataFrame, h: int) -> pd.DataFrame:
    """Features at t -> the tier the engine fires at t+h. Causal, gate-aware."""
    a = alerts[["series_id", "step", "tier", "rule_id", "gate_reason"]].copy()
    a["step"] = a.step - h                      # shift the label back to the decision point
    a = a.rename(columns={"tier": "tier_future", "rule_id": "rule_future",
                          "gate_reason": "gate_future"})
    X = F.merge(a, on=["series_id", "step"], how="left")
    X["tier_future"] = X.tier_future.fillna("NO_ALERT")
    X["gate_future"] = X.gate_future.fillna("NONE")
    # Rows the engine could not evaluate are not clean negatives. Including them would
    # teach the model that "gated" looks like "healthy".
    X["clean_negative"] = (X.tier_future == "NO_ALERT") & (X.gate_future == "NONE")
    X["trainable"] = (X.tier_future != "NO_ALERT") | X.clean_negative
    return X


def train_tier_head(CLS: pd.DataFrame, features: list, config, seed: int = SEED):
    """8-way primary head over tiers, plus a per-tier secondary head over rules.

    Tier probability and rule probability are calibrated and thresholded independently.
    """
    tr_c = CLS[(CLS.split == "train") & CLS.trainable]
    te_c = CLS[(CLS.split == "test") & CLS.trainable]
    if len(tr_c) < 100 or tr_c.tier_future.nunique() < 2:
        logging.warning("not enough trainable rows for the tier head; skipping")
        return None, {}, pd.DataFrame(), te_c, None

    if len(tr_c) > 40_000:
        tr_c = tr_c.sample(40_000, random_state=seed)

    # early_stopping=False: sklearn enables it automatically above 10,000 rows and carves
    # the holdout IID from a lagged panel, so a patient's neighbouring steps land on both
    # sides of it. Selection here happens on the held-out test split, not internally.
    tier_clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=.08,
                                              random_state=seed, class_weight="balanced",
                                              early_stopping=False)
    tier_clf.fit(tr_c[features], tr_c.tier_future.values)
    classes = list(tier_clf.classes_)
    proba_te = tier_clf.predict_proba(te_c[features]) if len(te_c) else None
    logging.info("tier head trained on %d rows, %d classes: %s", len(tr_c), len(classes), classes)

    # Rule-secondary head: within each fired tier, which rule?
    rule_clf, rule_report = {}, []
    for tier in [t for t in classes if t != "NO_ALERT"]:
        sub = tr_c[(tr_c.tier_future == tier) & tr_c.rule_future.notna()]
        if sub.rule_future.nunique() < 2 or len(sub) < 100:
            rule_report.append(dict(tier=tier, n_rules=int(sub.rule_future.nunique()),
                                    status="single rule or too few rows -- head not required"))
            continue
        clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=.1,
                                             random_state=seed, early_stopping=False)
        clf.fit(sub[features], sub.rule_future.values)
        rule_clf[tier] = clf
        sub_te = te_c[(te_c.tier_future == tier) & te_c.rule_future.notna()]
        acc = (float((clf.predict(sub_te[features]) == sub_te.rule_future.values).mean())
               if len(sub_te) else np.nan)
        rule_report.append(dict(tier=tier, n_rules=int(sub.rule_future.nunique()),
                                status=(f"trained; test top-1 accuracy {acc:.3f}"
                                        if np.isfinite(acc) else "trained; no test rows")))

    return tier_clf, rule_clf, pd.DataFrame(rule_report), te_c, proba_te


# ============================================================================= MODEL 4
# Symptom heads. Trained on SYNTHETIC labels -- see rule_engine/symptom_layer.py. Everything
# below is honest about that: the metrics are the ones that survive a 0.1-12% base rate, and
# the flag saying "synthetic" travels with the bundle rather than living in a docstring.

MIN_POSITIVES = 60


def _calibrated(est, X, y):
    """Isotonic calibration around a frozen fitted estimator."""
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(est), method="isotonic").fit(X, y)
    except ImportError:                                              # sklearn < 1.6
        return CalibratedClassifierCV(est, method="isotonic", cv="prefit").fit(X, y)


def make_symptom_model(seed: int = SEED) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=seed, max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        # Same reason as every other booster here: sklearn's automatic IID holdout leaks on
        # a lagged panel and makes the fit row-order dependent.
        early_stopping=False)


def eval_symptom(y, p, cut: float, budget_pct: float) -> dict:
    """Metrics that survive a rare-event base rate.

    Accuracy and bare ROC-AUC are omitted deliberately. These symptoms occur in 0.1-12% of
    sessions, where predicting "never" scores above 88% accurate and ROC-AUC is dominated by
    the negative class. PR-AUC as a MULTIPLE of the base rate, Brier skill, and precision at
    the operating budget are the numbers that can tell a useful head from a constant.
    """
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    m = np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 50 or y.sum() < 5:
        return {}
    base = float(y.mean())
    if base in (0.0, 1.0):
        return {}
    flag = p >= cut
    tp = int((flag & (y == 1)).sum())
    prec = tp / max(int(flag.sum()), 1)
    rec = tp / max(int((y == 1).sum()), 1)
    ap = float(average_precision_score(y, p))
    brier = float(brier_score_loss(y, p))
    return dict(n=len(y), base_rate=round(base, 5),
                PR_AUC=round(ap, 4), PR_lift=round(ap / max(base, 1e-9), 2),
                ROC_AUC=round(float(roc_auc_score(y, p)), 4),
                Brier_skill=round(1 - brier / max(base * (1 - base), 1e-9), 4),
                prec_at_budget=round(prec, 4), recall_at_budget=round(rec, 4),
                lift_at_budget=round(prec / max(base, 1e-9), 2),
                alert_rate=round(float(flag.mean()), 4), cut=round(float(cut), 5),
                budget_pct=budget_pct)


def train_symptom_heads(F: pd.DataFrame, features: list, config, seed: int = SEED):
    """Fit, calibrate, threshold and score every symptom head.

    THE CALIBRATION SPLIT IS THE POINT. The notebook fitted isotonic on the validation rows
    and then took the operating cut from `predict_proba` on those same rows. The cut is then
    chosen on in-sample calibration output, so the realised alert rate drifts away from the
    budget the moment it meets test data -- the threshold was fitted to noise the calibrator
    had already absorbed.

    Here the validation split is halved BY PATIENT: `va_cal` fits the isotonic map, `va_thr`
    picks the cut. Patient-level rather than row-level, because a patient's neighbouring
    sessions are near-duplicates and a row split would put the same patient on both sides,
    reproducing the leak in a subtler form.
    """
    from src.utils.main_utils.utils import deterministic_patient_split
    from src.utils.ml_utils.rule_engine.symptom_layer import (
        SYMPTOM_GROUPS,
        SYMPTOM_MECHANISM,
        SYMPTOM_RED_FLAG,
        SYMPTOMS,
    )

    targets = ([f"y_sym_{g}_h{h}" for g in SYMPTOM_GROUPS for h in config.horizons]
               + [f"y_sym_{s}_h{h}" for s in SYMPTOMS for h in config.horizons])
    targets = [t for t in targets if t in F.columns]
    if not targets:
        logging.info("no symptom targets in the feature frame; Model 4 skipped")
        return {}, {}, pd.DataFrame(), pd.DataFrame()

    va = F[F.split == "val"]
    assign = deterministic_patient_split(sorted(va.series_id.unique()), 0.5,
                                         salt="symptom-calibration")
    cal_ids = {k for k, v in assign.items() if v == "holdout"}
    va_cal, va_thr = va[va.series_id.isin(cal_ids)], va[~va.series_id.isin(cal_ids)]
    assert not (set(va_cal.series_id) & set(va_thr.series_id)), \
        "the calibration and threshold pools share a patient"
    if min(len(va_cal), len(va_thr)) < 100:
        logging.warning("validation split too thin to separate calibration from threshold "
                        "selection (%d / %d rows); Model 4 skipped",
                        len(va_cal), len(va_thr))
        return {}, {}, pd.DataFrame(), pd.DataFrame()

    tr, te = F[F.split == "train"], F[F.split == "test"]
    if len(tr) > config.max_train_rows:
        tr = tr.sample(config.max_train_rows, random_state=seed)
    budget = float(config.alert_budget_pct)

    models, cuts, rows, skipped = {}, {}, [], []
    for tgt in targets:
        key = tgt[len("y_sym_"):].rsplit("_h", 1)[0]
        horizon = int(tgt.rsplit("_h", 1)[1])
        pos = int(tr[tgt].sum()) if tr[tgt].notna().any() else 0
        if pos < MIN_POSITIVES:
            rate = float(F[tgt].mean()) if F[tgt].notna().any() else 0.0
            skipped.append(dict(target=tgt, symptom=key, horizon=horizon, positives=pos,
                                rate=round(rate, 5),
                                train_rows_needed=int(MIN_POSITIVES / max(rate, 1e-6))))
            continue
        try:
            m = tr[tgt].notna()
            est = make_symptom_model(seed).fit(tr.loc[m, features],
                                               tr.loc[m, tgt].astype(int))
            mc = va_cal[tgt].notna()
            if mc.sum() < 50 or va_cal.loc[mc, tgt].nunique() < 2:
                continue
            cal = _calibrated(est, va_cal.loc[mc, features],
                              va_cal.loc[mc, tgt].astype(int))

            mt = va_thr[tgt].notna()
            p_thr = cal.predict_proba(va_thr.loc[mt, features])[:, 1]
            cut = float(np.percentile(p_thr, 100 - budget))
            name = f"{key}_h{horizon}"
            models[name], cuts[name] = cal, cut

            r = dict(target=tgt, symptom=key, horizon=horizon,
                     mechanism=SYMPTOM_MECHANISM.get(key, "group"),
                     red_flag=key in SYMPTOM_RED_FLAG, train_positives=pos,
                     labels="SYNTHETIC")
            # The realised rate on va_thr and on test are reported side by side. Had the
            # calibration pool not been separated they would diverge, and that divergence
            # is precisely what this column exists to expose.
            r["thr_alert_rate"] = (eval_symptom(va_thr.loc[mt, tgt].astype(int), p_thr,
                                                cut, budget) or {}).get("alert_rate")
            mte = te[tgt].notna()
            if mte.sum() >= 50:
                p_te = cal.predict_proba(te.loc[mte, features])[:, 1]
                r.update(eval_symptom(te.loc[mte, tgt].astype(int), p_te, cut, budget))
                # Does the head beat simply quoting the patient's own recent rate?
                ref = f"sym_{key}_rate30"
                if ref in te.columns:
                    rm = eval_symptom(te.loc[mte, tgt].astype(int),
                                      te.loc[mte, ref].fillna(0).to_numpy(float),
                                      cut, budget) or {}
                    r["ref_PR_AUC"] = rm.get("PR_AUC")
                    r["beats_personal_rate"] = bool(
                        (r.get("PR_AUC") or 0) > (rm.get("PR_AUC") or 0))
            rows.append(r)
        except Exception as exc:                                     # noqa: BLE001
            logging.warning("symptom head %s failed: %s", tgt, exc)

    board = pd.DataFrame(rows)
    if len(board) and "alert_rate" in board and "thr_alert_rate" in board:
        drift = float((board.thr_alert_rate - board.alert_rate).abs().median())
        logging.warning("[SYNTHETIC LABELS] %d symptom heads trained, %d skipped below %d "
                        "positives. Median |alert-rate drift| from the threshold pool to "
                        "test: %.4f. These labels are generated -- the metrics describe "
                        "symptom_layer.py, not physiology.",
                        len(board), len(skipped), MIN_POSITIVES, drift)
    return models, cuts, board, pd.DataFrame(skipped)
