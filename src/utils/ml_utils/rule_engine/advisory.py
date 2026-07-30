"""Advisory objects and arbitration.

Port of notebook section 17. An Advisory is provider-visible decision support written to a
runtime store. It is NOT a DeviationAlert and is never written to the patient dataset --
no code path here produces one.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.utils.ml_utils.rule_engine.registry import ADVISORY_TIER_RANK


@dataclass
class Advisory:
    """Runtime decision-support object. NOT a DeviationAlert."""
    advisory_type: str
    patient_id: str
    as_of: str
    horizon_days: float
    confidence: float
    surfaced_tier: str
    rationale: str
    evidence: dict
    status: str = "PROVISIONAL"        # PROVISIONAL -> SHOWN -> SUPPRESSED
    data_quality: str = "FULL"         # FULL | PROXY | BLOCKED
    provider_visible_only: bool = True

    def to_row(self) -> dict:
        return asdict(self)


def build_advisories(predictor, history: pd.DataFrame, tier_clf, rule_clf,
                     features: list, config) -> list:
    """Turn one patient's prediction into zero or more provider-facing advisories."""
    a = predictor.predict(history)
    pid, as_of = a["patient_id"], a["as_of"]
    out = []
    if a["confidence_tier"] == "cold_start":
        return out

    thr = a["personalisation"]["threshold"]
    med_gap = float(history.ts.diff().dt.days.median() or 2)

    # 1. PREDICT_BP_DRIFT_TOWARD_L1 -- fully supported
    for hk, node in a["forecast"].get("sbp", {}).items():
        if node["point"] >= thr:
            margin = node["point"] - thr
            conf = float(np.clip(0.5 + margin / 40.0, 0, 0.99))
            out.append(Advisory("PREDICT_BP_DRIFT_TOWARD_L1", pid, as_of,
                                node["days_ahead_est"], round(conf, 3), "BP_LEVEL_1_HIGH",
                                f"forecast SBP {node['point']} >= personalised threshold {thr}",
                                dict(forecast=node, threshold=thr,
                                     interval=[node.get("lo80"), node.get("hi80")])))
            break                                   # earliest breaching horizon only

    # 2. PREDICT_HF_DECOMP_IMMINENT -- PROXY: idwg trend only, no edema / diuretic adherence
    if "idwg" in history and history.idwg.notna().sum() >= 8:
        recent = history.idwg.dropna().tail(8)
        slope = float(np.polyfit(np.arange(len(recent)), recent.values, 1)[0])
        if slope > 0.05 and float(recent.tail(3).mean()) >= 3.0:
            out.append(Advisory("PREDICT_HF_DECOMP_IMMINENT", pid, as_of, 3 * med_gap,
                                round(float(np.clip(0.4 + slope, 0, .95)), 3), "BP_LEVEL_1_LOW",
                                "rising interdialytic weight gain (fluid-overload proxy)",
                                dict(idwg_slope=round(slope, 3),
                                     idwg_recent_mean=round(float(recent.tail(3).mean()), 2)),
                                data_quality="PROXY"))

    # 3. early-warning detector, translated into the provider's tier vocabulary by the head
    ew = a.get("early_warning") or {}
    if ew.get("flagged"):
        row = predictor.fb.transform_for_inference(history).reindex(columns=features)
        tier_hat, conf = "BP_LEVEL_1_HIGH", 0.5
        if tier_clf is not None:
            try:
                p = tier_clf.predict_proba(row)[0]
                k = int(np.argmax(p))
                tier_hat, conf = tier_clf.classes_[k], float(p[k])
                if tier_hat == "NO_ALERT" and len(p) > 1:
                    order = np.argsort(p)[::-1]
                    for j in order:
                        if tier_clf.classes_[j] != "NO_ALERT":
                            tier_hat, conf = tier_clf.classes_[j], float(p[j])
                            break
            except Exception:
                pass
        rule_hat = None
        if tier_hat in (rule_clf or {}):
            try:
                rp = rule_clf[tier_hat].predict_proba(row)[0]
                j = int(np.argmax(rp))
                rule_hat = f"{rule_clf[tier_hat].classes_[j]} at {rp[j]:.0%}"
            except Exception:
                pass
        out.append(Advisory("PREDICT_ANOMALY_TREND", pid, as_of, ew["est_lead_days"],
                            round(conf, 3), tier_hat,
                            f"detector score {ew['score']} >= cut {ew['cut']}"
                            + (f"; most likely {rule_hat}" if rule_hat else ""),
                            dict(early_warning=ew)))
    return out


def arbitrate(advisories: list, engine_fired_tiers_today: set, tau: float = 0.55):
    """Highest tier wins, ties by earliest horizon, confidence-thresholded, engine-aware."""
    log = dict(received=len(advisories), suppressed_low_confidence=0,
               suppressed_engine_already_fired=0, bundled=0)
    kept = []
    for adv in advisories:
        if adv.confidence < tau:
            adv.status = "SUPPRESSED"
            log["suppressed_low_confidence"] += 1
            continue
        if adv.surfaced_tier in engine_fired_tiers_today:
            adv.status = "SUPPRESSED"
            log["suppressed_engine_already_fired"] += 1
            continue
        kept.append(adv)
    kept.sort(key=lambda a: (ADVISORY_TIER_RANK.get(a.surfaced_tier, 99), a.horizon_days))
    for a in kept:
        a.status = "SHOWN"
    log["bundled"] = max(0, len(kept) - 1)
    log["shown"] = len(kept)
    return kept, log


# An override asserts the model was wrong to fire at all; a disposition resolves a fired
# alert. Folding one into the other loses the distinction that makes either useful.
