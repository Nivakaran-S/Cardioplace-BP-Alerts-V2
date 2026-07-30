"""Detector, offset, fairness, calibration and drift metrics.

Port of notebook sections 7 (scorecard), 8.3, 9, 11 and 19. These answer "is this
clinically useful", as distinct from the point-forecast metrics in `regression_metric`.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from src.utils.ml_utils.metric.regression_metric import bootstrap_ci

FAIR_MARGIN_PP = 0.05


# --------------------------------------------------------------------------- detector

def detector_scorecard(df: pd.DataFrame, detectors: list, sessions_per_week: float,
                       budgets=(1, 2, 5, 10)) -> pd.DataFrame:
    """Precision / recall / lead time for each detector at each alert budget."""
    rows = []
    for col in detectors:
        s = df[[col, "event_next", "event_step", "step", "days_since_last"]].copy()
        # An infinite score survives dropna and then wins every ranking: np.percentile
        # returns inf for the cut, so the budget selects nothing, and roc_auc_score treats
        # it as the most confident prediction in the set. Fusion legs and the EWMA gap both
        # produce inf when a patient's baseline std is zero.
        s[col] = s[col].replace([np.inf, -np.inf], np.nan)
        s = s.dropna(subset=[col])
        if s.event_next.nunique() < 2 or len(s) < 50:
            continue
        base = float(s.event_next.mean())
        auc = roc_auc_score(s.event_next, s[col])
        ap = average_precision_score(s.event_next, s[col])
        for b in budgets:
            cut = np.percentile(s[col], 100 - b)
            flag = (s[col] >= cut).astype(int)
            tp = int(((flag == 1) & (s.event_next == 1)).sum())
            fp = int(((flag == 1) & (s.event_next == 0)).sum())
            fn = int(((flag == 0) & (s.event_next == 1)).sum())
            prec = tp / (tp + fp) if tp + fp else np.nan
            hit = s[(flag == 1) & (s.event_next == 1)]
            lead_sessions = hit.event_step - hit.step
            lead_days = (lead_sessions * hit.days_since_last.median() if len(hit)
                         else pd.Series(dtype=float))
            rows.append(dict(
                detector=col, budget_pct=b, base_rate=round(base, 4),
                precision=round(prec, 3) if np.isfinite(prec) else np.nan,
                lift=round(prec / base, 2) if base and np.isfinite(prec) else np.nan,
                recall=round(tp / (tp + fn), 3) if tp + fn else np.nan,
                auc_roc=round(auc, 3), auc_pr=round(ap, 3), auc_pr_lift=round(ap / base, 2),
                lead_days_p10=round(float(lead_days.quantile(.10)), 2) if len(hit) else np.nan,
                lead_days_median=round(float(lead_days.median()), 2) if len(hit) else np.nan,
                lead_days_p90=round(float(lead_days.quantile(.90)), 2) if len(hit) else np.nan,
                alerts_per_patient_week=round(b / 100 * sessions_per_week, 2)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- offset

def offset_scorecard(M: pd.DataFrame, label: str, population_threshold: float,
                     config=None) -> pd.DataFrame:
    """Learned blend vs personal-only, cohort-only and the population constant.

    Every candidate is put through the same governance caps before it is scored. Raw
    personal and cohort bands are not shippable alternatives: on this cohort the raw
    personal band reaches 195 mmHg and would hand 24 of 150 patients a threshold at or
    above the 180 emergency floor, which safety gate 3 exists to forbid. Scoring the
    capped blend against an uncapped rival measures the cost of the cap and calls it a
    modelling loss -- and the ship decision downstream then rules against the only legal
    option. The caps are applied here so the comparison is between things that could
    actually ship; `legal_raw` records which candidates were already compliant.
    """
    a = M.actual.values

    def cap(v):
        v = np.asarray(v, float)
        if config is None:
            return v
        return population_threshold + np.clip(v - population_threshold,
                                              -config.offset_cap_tighten,
                                              config.offset_cap_loosen)

    candidates = {"learned (capped blend)": M.threshold.values,
                  "personal only": M.personal.values,
                  "cohort only": M.cohort.values,
                  "population constant": np.full(len(M), population_threshold)}
    rows = []
    for name, raw in candidates.items():
        p = cap(raw)
        mae, (lo, hi) = bootstrap_ci(mean_absolute_error, a, p)
        rows.append(dict(split=label, model=name, MAE=round(mae, 2), lo=round(lo, 2),
                         hi=round(hi, 2), R2=round(r2_score(a, p), 3),
                         within_10mmHg=round(float(np.mean(np.abs(a - p) <= 10)), 3),
                         legal_raw=bool(np.allclose(np.asarray(raw, float), p, equal_nan=True))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- fairness

def slice_gate(df: pd.DataFrame, group_cols, metric, margin: float, label: str,
               min_n: int = 30, reference=None, higher_is_better: bool = False
               ) -> pd.DataFrame:
    """Per-subgroup performance against a pass/fail margin.

    ## Why `reference` exists

    Comparing a subgroup's RAW metric against the overall value conflates two very different
    things: a model that serves a group badly, and a group that is intrinsically harder to
    predict. On this corpus that is not hypothetical -- it produced a false failure that
    blocked promotion.

    Under-50 dialysis patients have 24% more variable systolic pressure (SD 30.6 vs 24.7),
    so every predictor has higher error for them. The EWMA baseline -- which does no learning
    whatsoever -- is 2.50 mmHg worse for that group. The shipped model was 2.33 mmHg worse,
    i.e. it CLOSED part of the gap, and the gate failed it for the difficulty of the data.

    The same mechanism applies to precision at a fixed alert budget: precision is bounded by
    the subgroup's event base rate, so a group with fewer events cannot reach the same
    precision however good the detector is.

    Passing `reference` -- a callable giving what is ACHIEVABLE for each subgroup, such as the
    baseline forecaster's error or the subgroup's own event rate -- changes the comparison to
    the improvement the model delivers over that floor. That is the fairness question worth
    asking: does this model help every group about equally? It keeps the margin in its
    original units, because the score is still a difference rather than a ratio.

    The raw value is still reported alongside, so nothing is hidden -- only the pass/fail
    basis changes.
    """
    overall_v = metric(df)
    overall_r = reference(df) if reference is not None else None

    def _score(v, r):
        if r is None or not np.isfinite(r):
            return v
        return (v - r) if higher_is_better else (r - v)

    overall_s = _score(overall_v, overall_r)
    rows = []
    for gc in group_cols:
        if gc not in df.columns:
            continue
        for level, sub in df.groupby(gc, observed=True):
            if len(sub) < min_n:
                continue
            v = metric(sub)
            if not np.isfinite(v):
                continue
            r = reference(sub) if reference is not None else None
            sc = _score(v, r)
            gap = sc - overall_s
            rows.append(dict(
                metric=label, axis=gc, level=str(level), n=len(sub),
                value=round(v, 4), overall=round(overall_v, 4),
                reference=(round(r, 4) if r is not None and np.isfinite(r) else None),
                score=round(sc, 4), overall_score=round(overall_s, 4),
                raw_gap=round(v - overall_v, 4),
                gap=round(gap, 4),
                basis=("improvement over the subgroup's own reference"
                       if reference is not None else "raw metric vs overall"),
                passes=bool(abs(gap) <= margin)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ calibration/utility

def expected_calibration_error(y, p, bins: int = 10) -> float:
    df = pd.DataFrame({"y": y, "p": p}).dropna()
    if len(df) < 50:
        return np.nan
    df["bin"] = pd.qcut(df.p, min(bins, df.p.nunique()), duplicates="drop")
    g = df.groupby("bin", observed=True).agg(conf=("p", "mean"), acc=("y", "mean"),
                                             n=("y", "size"))
    return float((g.n / g.n.sum() * (g.conf - g.acc).abs()).sum())


def tier_metrics(y_true_bin, p, thr: float, prevalence: float,
                 unit: str = "per patient-step") -> dict:
    """Sensitivity/specificity/PPV/NPV/NNA at one operating point, base rate disclosed.

    Every row discloses its base rate: a Brier of 0.15 at a prevalence of 0.005 is nearly
    perfect; the same 0.15 at 0.6 is barely better than random. Without prevalence the
    number is uninterpretable.
    """
    y_true_bin = np.asarray(y_true_bin)
    p = np.asarray(p, float)
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y_true_bin == 1)).sum())
    fp = int(((yhat == 1) & (y_true_bin == 0)).sum())
    fn = int(((yhat == 0) & (y_true_bin == 1)).sum())
    tn = int(((yhat == 0) & (y_true_bin == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return dict(prevalence=round(prevalence, 4), threshold=round(thr, 3),
                sensitivity=round(sens, 3), specificity=round(spec, 3),
                PPV=round(ppv, 3), NPV=round(npv, 3),
                NNA=round(1 / ppv, 1) if ppv and np.isfinite(ppv) and ppv > 0 else np.nan,
                AUC_PR=round(average_precision_score(y_true_bin, p), 3),
                AUC_ROC=round(roc_auc_score(y_true_bin, p), 3),
                Brier=round(brier_score_loss(y_true_bin, p), 4),
                ECE=round(expected_calibration_error(y_true_bin, p), 4),
                unit=unit, n=len(p))


def decision_curve(y, p, thresholds=None) -> pd.DataFrame:
    """Net benefit across threshold probabilities (Vickers & Elkin 2006).

    net_benefit = TP/n - FP/n * pt/(1-pt). Compared against 'treat all' and 'treat none'.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.60, 40)
    y = np.asarray(y)
    p = np.asarray(p, float)
    n = len(y)
    out = []
    for pt in thresholds:
        yhat = (p >= pt).astype(int)
        tp = float(((yhat == 1) & (y == 1)).sum())
        fp = float(((yhat == 1) & (y == 0)).sum())
        nb = tp / n - fp / n * (pt / (1 - pt))
        prev = float(np.mean(y))
        nb_all = prev - (1 - prev) * (pt / (1 - pt))
        out.append(dict(threshold=pt, net_benefit=nb, nb_treat_all=nb_all, nb_treat_none=0.0))
    return pd.DataFrame(out)


def elicitation_table(y, p, sessions_per_week: float,
                      nna_options=(3, 5, 10, 15, 20)) -> pd.DataFrame:
    """The harm:benefit trade-off rendered as a table a clinician can answer from."""
    y = np.asarray(y)
    p = np.asarray(p, float)
    rows = []
    for nna in nna_options:
        pt = 1 / nna
        yhat = (p >= pt).astype(int)
        tp = float(((yhat == 1) & (y == 1)).sum())
        fp = float(((yhat == 1) & (y == 0)).sum())
        fn = float(((yhat == 0) & (y == 1)).sum())
        rows.append(dict(
            question_answer=f"I'd review {nna} advisories to catch 1 true event",
            implied_p_t=round(pt, 3),
            sensitivity=round(tp / (tp + fn), 3) if tp + fn else np.nan,
            PPV=round(tp / (tp + fp), 3) if tp + fp else np.nan,
            alerts_per_patient_week=round(float(yhat.mean()) * sessions_per_week, 2)))
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- monitoring

def population_stability_index(expected: np.ndarray, actual: np.ndarray,
                               bins: int = 10) -> float:
    """PSI between a reference and a current distribution. >0.2 is the conventional alarm."""
    e = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    a = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()
    if len(e) < 50 or len(a) < 50 or e.nunique() < 3:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


class DriftMonitor:
    """Feature drift (PSI), forecast-error drift and alert-rate drift, with a response ladder."""

    PSI_WARN, PSI_ALARM = 0.10, 0.20
    ERROR_ALARM_REL = 0.10      # >10% relative MAE degradation
    RATE_ALARM_REL = 0.50       # alert rate off target by >50% relative

    def __init__(self, reference: pd.DataFrame, features: list):
        self.ref, self.features = reference, features

    def feature_drift(self, current: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        rows = [dict(feature=f,
                     psi=population_stability_index(self.ref[f].values, current[f].values))
                for f in self.features if f in self.ref.columns and f in current.columns]
        out = pd.DataFrame(rows).dropna().sort_values("psi", ascending=False)
        if out.empty:
            return out
        out["status"] = np.where(out.psi >= self.PSI_ALARM, "ALARM",
                                 np.where(out.psi >= self.PSI_WARN, "WARN", "ok"))
        return out.head(top_n)

    def error_drift(self, ref_mae: float, cur_mae: float) -> dict:
        rel = (cur_mae - ref_mae) / max(ref_mae, 1e-9)
        return dict(signal="forecast error", reference=round(ref_mae, 3),
                    current=round(cur_mae, 3), relative_change=round(rel, 3),
                    status="ALARM" if rel > self.ERROR_ALARM_REL else "ok",
                    needs_labels=False, detectable_within="1-7 days (self-supervised)")

    def alert_rate_drift(self, observed_rate: float, target_rate: float) -> dict:
        rel = abs(observed_rate - target_rate) / max(target_rate, 1e-9)
        return dict(signal="alert rate", reference=round(target_rate, 4),
                    current=round(observed_rate, 4), relative_change=round(rel, 3),
                    status="ALARM" if rel > self.RATE_ALARM_REL else "ok",
                    needs_labels=False, detectable_within="days")

    def cadence_drift(self, ref_gap, cur_gap) -> dict:
        """PSI on the RAW gap, as its own signal rather than one row of feature_drift.

        Two reasons it cannot ride along with `feature_drift`: that method returns only the
        top 20 features by PSI, so cadence drops off the report exactly when other features
        are moving; and it reads `days_since_last`, which is clipped, so a cohort drifting
        from 30-day to 200-day gaps registers as no change at all.

        This is the signal that fires when a product built for daily journaling starts
        seeing the irregular, long gaps a dialysis corpus never contained.
        """
        r = pd.Series(ref_gap).replace([np.inf, -np.inf], np.nan).dropna()
        c = pd.Series(cur_gap).replace([np.inf, -np.inf], np.nan).dropna()
        psi = population_stability_index(r.values, c.values) if len(r) and len(c) else np.nan
        status = ("ALARM" if np.isfinite(psi) and psi >= self.PSI_ALARM
                  else "WARN" if np.isfinite(psi) and psi >= self.PSI_WARN else "ok")
        return dict(signal="session cadence",
                    reference=round(float(r.median()), 2) if len(r) else np.nan,
                    current=round(float(c.median()), 2) if len(c) else np.nan,
                    relative_change=round(float(psi), 3) if np.isfinite(psi) else np.nan,
                    status=status, needs_labels=False,
                    detectable_within="days (median gap, PSI on the unclipped value)")

    @staticmethod
    def response_ladder(status: str) -> str:
        if status != "ALARM":
            return "no action; continue monitoring"
        return ("1) recalibrate (cheap, frequent) -> 2) full refit only if recalibration does not "
                "restore the metric (governed) -> 3) roll back if neither does. Log which step "
                "resolved it; that distribution is what sets the real cadence.")
