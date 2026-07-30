"""The deterministic rule engine -- the permanent floor.

Port of notebook section 14. It is never trained, never tuned, and never consults the ML
layer. Its output is the ground truth the ML layer forecasts and the vocabulary it
translates into -- never something the ML layer produces.
"""

import numpy as np
import pandas as pd

from src.utils.ml_utils.rule_engine.registry import EMERGENCY_TIERS

GATE_REASONS = {
    "NONE": "evaluated normally",
    "HISTORICAL_ENTRY": "stale entry; no L2 fire, excluded from CMS 99454 count",
    "SINGLE_READING_NON_EMERGENCY": "session-average required; lone reading cannot fire non-emergency",
    "UNCONFIRMED_TERMINAL": "Option-D pair-resolution already fired",
    "NOT_SIMULABLE": "gate exists upstream but its inputs are absent from this corpus",
}


class RuleEngine:
    """Deterministic alert engine. Fixed stage order, hard short-circuits, explicit gates."""

    SBP_EMERGENCY, DBP_EMERGENCY = 180.0, 120.0
    SBP_HIGH, DBP_HIGH = 140.0, 90.0
    SBP_LOW, DBP_LOW = 90.0, 60.0
    SBP_LOW_ELDERLY = 100.0
    PP_WIDE, PP_NARROW = 60.0, 25.0
    STALE_GAP_DAYS = 14

    def __init__(self, registry: pd.DataFrame, thresholds: dict = None):
        self.registry = registry
        self.personal = thresholds or {}      # series_id -> personalised SBP threshold
        self.evaluable = set(registry.loc[registry.status == "EVALUABLE", "rule_id"])

    # ---- gates --------------------------------------------------------------
    def _gate(self, row) -> str:
        """`HISTORICAL_ENTRY` must mean "this reading is old", not "the one before it was".

        `days_since_last` is the spacing to the PREVIOUS reading. Gating on it inverted the
        policy: a patient who logs SBP 200 today after a three-week absence has a gap of 21,
        so the reading was classified historical and `_stage_emergency` was skipped -- the
        emergency was suppressed precisely for the user most likely to need it.

        The policy is unchanged: a genuinely back-dated entry must not fire a live L2, and
        is excluded from the CMS 99454 count. Only the variable that detects it is corrected.
        `reading_age_days` is the age of the reading at the moment of evaluation, which the
        serving path supplies from `as_of`. The training panel has no such column, so the
        fallback keeps offline behaviour bit-identical -- proved by differential execution.
        """
        age = getattr(row, "reading_age_days", None)
        if age is not None and np.isfinite(age):
            if age > self.STALE_GAP_DAYS:
                return "HISTORICAL_ENTRY"
        elif row.days_since_last > self.STALE_GAP_DAYS:
            return "HISTORICAL_ENTRY"
        if getattr(row, "n_meas", 99) < 2:
            return "SINGLE_READING_NON_EMERGENCY"
        return "NONE"

    # ---- stages -------------------------------------------------------------
    def _stage_emergency(self, row):
        if row.sbp >= self.SBP_EMERGENCY or row.dbp >= self.DBP_EMERGENCY:
            return ("BP_LEVEL_2", "RULE_ABSOLUTE_EMERGENCY")
        return None

    def _stage_contraindication(self, row):
        # Option-D: an emergency-range reading that was never retaken within the session.
        if (row.sbp >= self.SBP_EMERGENCY or row.dbp >= self.DBP_EMERGENCY) \
                and getattr(row, "n_meas", 99) < 2:
            return ("TIER_1_CONTRAINDICATION", "RULE_UNCONFIRMED_EMERGENCY")
        return None

    def _stage_l1(self, row, personal_thr):
        # first-claimant-wins: personalised beats standard purely by iteration order
        if personal_thr is not None and row.sbp >= personal_thr:
            return ("BP_LEVEL_1_HIGH", "RULE_PERSONALIZED_HIGH")
        if row.sbp >= self.SBP_HIGH or row.dbp >= self.DBP_HIGH:
            return ("BP_LEVEL_1_HIGH", "RULE_STANDARD_L1_HIGH")
        if row.age >= 65 and row.sbp < self.SBP_LOW_ELDERLY:
            return ("BP_LEVEL_1_LOW", "RULE_AGE_65_LOW")
        if row.sbp < self.SBP_LOW or row.dbp < self.DBP_LOW:
            return ("BP_LEVEL_1_LOW", "RULE_STANDARD_L1_LOW")
        _idwg = getattr(row, "idwg", np.nan)
        if np.isfinite(_idwg) and _idwg >= 4.0:
            return ("BP_LEVEL_1_LOW", "RULE_HF_DECOMPENSATION")   # fluid-overload proxy
        return None

    def _stage_info(self, row, prev_emergency):
        pp = row.sbp - row.dbp
        if prev_emergency and row.sbp < self.SBP_HIGH:
            return ("TIER_3_INFO", "RULE_EMERGENCY_RANGE_CONFIRMED_NORMAL")
        if pp >= self.PP_WIDE:
            return ("TIER_3_INFO", "RULE_PULSE_PRESSURE_WIDE")
        if pp <= self.PP_NARROW:
            return ("TIER_3_INFO", "RULE_PULSE_PRESSURE_NARROW")
        if row.step < 12:
            return ("TIER_3_INFO", "RULE_FIRST_MONTH_ADHERENCE_NUDGE")
        return None

    # ---- evaluation ---------------------------------------------------------
    def evaluate_row(self, row, personal_thr, prev_emergency) -> dict:
        gate = self._gate(row)
        out = dict(tier=None, rule_id=None, mode="STANDARD", gate_reason=gate,
                   short_circuited=False)

        hit = self._stage_emergency(row)          # emergency runs even when gated for staleness
        if hit and gate != "HISTORICAL_ENTRY":
            out.update(tier=hit[0], rule_id=hit[1], short_circuited=True)
            return out                            # emergency exclusivity: later stages never run

        if gate != "NONE":
            return out                            # gated: not an alert, and NOT a clean negative

        for stage in (self._stage_contraindication,):
            hit = stage(row)
            if hit:
                out.update(tier=hit[0], rule_id=hit[1])
                return out

        hit = self._stage_l1(row, personal_thr)
        if hit:
            out.update(tier=hit[0], rule_id=hit[1],
                       mode="PERSONALIZED" if hit[1].startswith("RULE_PERSONALIZED")
                       else "STANDARD")
            return out

        hit = self._stage_info(row, prev_emergency)
        if hit:
            out.update(tier=hit[0], rule_id=hit[1])
        return out

    def run(self, panel: pd.DataFrame, use_personalisation: bool = False) -> pd.DataFrame:
        rows = []
        for sid, g in panel.groupby("series_id", sort=False):
            g = g.sort_values("step")
            thr = self.personal.get(sid) if use_personalisation else None
            prev_emerg = False
            for row in g.itertuples():
                r = self.evaluate_row(row, thr, prev_emerg)
                prev_emerg = (r["tier"] == "BP_LEVEL_2")
                rows.append(dict(series_id=sid, step=int(row.step), ts=row.ts,
                                 sbp=float(row.sbp), dbp=float(row.dbp), **r))
        out = pd.DataFrame(rows)
        out["fired"] = out.tier.notna()
        out["is_emergency"] = out.tier.isin(EMERGENCY_TIERS)
        return out
