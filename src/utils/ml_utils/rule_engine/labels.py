"""Label engineering: the disposition catalog and its simulated stream.

Port of notebook section 15. The clinician judgement is proxied by a forward-looking
outcome. It is a proxy and is labelled as such -- it stands in for a real disposition
stream, it does not claim to be one.
"""

import numpy as np
import pandas as pd

from src.constants.training_pipeline import SEED
from src.logging.logger import logging
from src.utils.ml_utils.rule_engine.engine import RuleEngine

# 22 actions, grouped by the ML interpretation assigned to them.
DISPOSITION_CATALOG = {
    "positive": ["TIER1_DISCONTINUED", "TIER1_CHANGE_ORDERED", "TIER1_ACKNOWLEDGED",
                 "TIER2_WILL_CONTACT", "TIER2_CHANGE_ORDERED", "BP_L2_CONTACTED_MED_ADJUSTED",
                 "BP_L2_CONTACTED_ADVISED_ED", "BP_L2_SEEN_IN_OFFICE", "ANGIO_ADVISED_ED",
                 "ANGIO_CONFIRMED_ED", "ANGIO_ACE_DISCONTINUED", "ANGIO_SEEN_IN_OFFICE"],
    "negative": ["TIER1_FALSE_POSITIVE", "TIER2_REVIEWED_NO_ACTION",
                 "BP_L2_REVIEWED_TRENDING_DOWN", "ANGIO_FALSE_ALARM"],
    "deferred": ["TIER1_DEFERRED", "TIER2_DEFERRED", "TIER2_PHARMACY_RECONCILE",
                 "BP_L2_CONTACTED_RECHECK"],
    "unreachable": ["BP_L2_UNABLE_TO_REACH_RETRY", "ANGIO_UNABLE_TO_REACH"],
}

DISPOSITIONABLE_TIERS = {"BP_LEVEL_2", "BP_LEVEL_2_SYMPTOM_OVERRIDE", "TIER_1_ANGIOEDEMA",
                         "TIER_1_CONTRAINDICATION", "TIER_2_DISCREPANCY"}
LABEL_LATENCY_MAX_DAYS = 14
DEFERRED_LOSS_WEIGHT = 0.4
DOUBLE_REVIEW_RATE = 0.10


def disposition_coverage(registry: pd.DataFrame, tiers: list) -> pd.DataFrame:
    """Which rules have a resolution path at all -- the label-coverage gap, reproduced not hidden.

    The forecaster and detector cover these rules' readings regardless, because neither
    needs the label stream. What the gap constrains is classifier-head vocabulary accuracy
    and retrospective confirmation that a flagged drift was clinically real.
    """
    cov = registry.assign(has_disposition=registry.tier.isin(DISPOSITIONABLE_TIERS))
    tbl = (cov.groupby(["tier", "has_disposition"]).size().unstack(fill_value=0)
           .reindex(tiers).fillna(0).astype(int).reset_index())
    n_with = int(cov.has_disposition.sum())
    logging.info("rules with a disposition path: %d of %d (%.0f%%)",
                 n_with, len(cov), 100 * n_with / max(len(cov), 1))
    return tbl


def simulate_dispositions(alerts: pd.DataFrame, panel: pd.DataFrame,
                          seed: int = SEED) -> pd.DataFrame:
    """Attach dispositions ONLY to tiers that have a resolution catalog."""
    rng = np.random.default_rng(seed)
    fired = alerts[alerts.fired].copy()
    if fired.empty:
        return fired
    nxt = (panel.sort_values(["series_id", "step"])
           .assign(sbp_next3=lambda d: d.groupby("series_id").sbp
                   .shift(-1).rolling(3, min_periods=1).max())
           [["series_id", "step", "sbp_next3"]])
    fired = fired.merge(nxt, on=["series_id", "step"], how="left")

    fired["dispositionable"] = fired.tier.isin(DISPOSITIONABLE_TIERS)
    persistent = (fired.sbp_next3 >= RuleEngine.SBP_HIGH).fillna(False)

    dispo, cls = [], []
    for is_d, pers in zip(fired.dispositionable, persistent):
        if not is_d:
            dispo.append(None)
            cls.append(None)
            continue
        u = rng.random()
        if u < 0.08:
            dispo.append(rng.choice(DISPOSITION_CATALOG["deferred"]))
            cls.append("deferred")
        elif u < 0.13:
            dispo.append(rng.choice(DISPOSITION_CATALOG["unreachable"]))
            cls.append("unreachable")
        elif pers:
            dispo.append(rng.choice(DISPOSITION_CATALOG["positive"]))
            cls.append("positive")
        else:
            dispo.append(rng.choice(DISPOSITION_CATALOG["negative"]))
            cls.append("negative")
    fired["disposition"] = dispo
    fired["disposition_class"] = cls

    # Label latency: dispositions arrive days later, hard outcomes weeks later.
    fired["disposition_lag_days"] = np.where(
        fired.disposition.notna(),
        rng.integers(0, LABEL_LATENCY_MAX_DAYS + 1, len(fired)), np.nan)
    fired["label_available_at"] = fired.ts + pd.to_timedelta(fired.disposition_lag_days, unit="D")

    # Weighting: deferred is a weak positive, unreachable is operational not clinical.
    fired["loss_weight"] = np.select(
        [fired.disposition_class == "deferred", fired.disposition_class == "unreachable"],
        [DEFERRED_LOSS_WEIGHT, 0.0], default=1.0)
    fired["binary_label"] = np.select(
        [fired.disposition_class == "positive", fired.disposition_class == "deferred",
         fired.disposition_class == "negative"], [1, 1, 0], default=np.nan)

    # Inter-rater reliability: a random sample gets a second reviewer.
    second = rng.random(len(fired)) < DOUBLE_REVIEW_RATE
    flip = second & (rng.random(len(fired)) < 0.15)
    fired["second_review"] = second
    fired["second_label"] = np.where(second, np.where(flip, 1 - fired.binary_label,
                                                      fired.binary_label), np.nan)
    return fired


def label_quality(dispo: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Training terminus and inter-rater agreement.

    Undocumented label noise is indistinguishable from model error. Measuring it is the
    cheapest way to know how much headroom a classifier actually has.
    """
    if dispo.empty:
        return {"status": "no alerts fired"}
    cutoff = panel.ts.max() - pd.Timedelta(days=LABEL_LATENCY_MAX_DAYS)
    out = {
        "alerts_fired": int(len(dispo)),
        "with_disposition_path": int(dispo.disposition.notna().sum()),
        "training_terminus": str(cutoff.date()),
        "labels_not_yet_landed": int((dispo.ts > cutoff).sum()),
    }
    rev = dispo[dispo.second_review & dispo.binary_label.notna() & dispo.second_label.notna()]
    if len(rev) > 10:
        agree = float((rev.binary_label == rev.second_label).mean())
        out.update(double_reviewed=int(len(rev)), raw_agreement=round(agree, 4),
                   kappa=round((agree - 0.5) / 0.5, 3))
    return out
