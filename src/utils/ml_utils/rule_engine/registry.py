"""The complete clinical inventory: all 56 rules across 8 tiers.

Port of notebook sections 1.1b, 13 (1a/1b). `needs` names the inputs a rule requires. A
rule whose needs are absent from HEMOBP is marked BLOCKED_ON_INPUTS -- it remains a
prediction target and simply cannot be fired by this pipeline's engine. None are dropped.
"""

import glob
import os

import numpy as np
import pandas as pd

from src.constants.training_pipeline import PULSE_CANDIDATES, RAW_DATA_DIR, VIP_FILE_NAME
from src.logging.logger import logging

TIERS = ["BP_LEVEL_2", "BP_LEVEL_2_SYMPTOM_OVERRIDE", "TIER_1_ANGIOEDEMA",
         "TIER_1_CONTRAINDICATION", "TIER_2_DISCREPANCY", "BP_LEVEL_1_HIGH",
         "BP_LEVEL_1_LOW", "TIER_3_INFO"]

EMERGENCY_TIERS = {"BP_LEVEL_2", "BP_LEVEL_2_SYMPTOM_OVERRIDE", "TIER_1_ANGIOEDEMA"}

ADVISORY_TIER_RANK = {t: i for i, t in enumerate(TIERS)}   # lower index = higher severity

# What each layer may do, per tier. This table is a safety contract, not documentation.
TIER_POLICY = pd.DataFrame([
    ("BP_LEVEL_2",                  False, "metrics only", "emergency floor"),
    ("BP_LEVEL_2_SYMPTOM_OVERRIDE", False, "metrics only", "emergency floor"),
    ("TIER_1_ANGIOEDEMA",           False, "metrics only", "acute airway; no leading signal"),
    ("TIER_1_CONTRAINDICATION",     None,  "days-ahead",    "candidate, clinician-gated"),
    ("TIER_2_DISCREPANCY",          True,  "days-ahead",    "adherence trend"),
    ("BP_LEVEL_1_HIGH",             True,  "1-3 days",      "BP drift"),
    ("BP_LEVEL_1_LOW",              True,  "1-3 days",      "BP drift"),
    ("TIER_3_INFO",                 True,  "days-ahead",    "informational"),
], columns=["tier", "personalisable", "forecast_horizon", "note"])

RULES = [
    # (rule_id, tier, needs, code_note)
    ("RULE_ABSOLUTE_EMERGENCY",          "BP_LEVEL_2", ["sbp", "dbp"], ""),
    ("RULE_PREGNANCY_L2",                "BP_LEVEL_2", ["pregnancy"], ""),
    ("RULE_SYMPTOM_OVERRIDE_GENERAL",    "BP_LEVEL_2_SYMPTOM_OVERRIDE", ["symptoms"], ""),
    ("RULE_SYMPTOM_OVERRIDE_PREGNANCY",  "BP_LEVEL_2_SYMPTOM_OVERRIDE", ["symptoms", "pregnancy"], ""),
    ("RULE_ACE_ANGIOEDEMA",              "TIER_1_ANGIOEDEMA", ["symptoms", "medications"], ""),
    ("RULE_GENERIC_ANGIOEDEMA",          "TIER_1_ANGIOEDEMA", ["symptoms"], ""),
    ("RULE_PREGNANCY_ACE_ARB",           "TIER_1_CONTRAINDICATION", ["pregnancy", "medications"], ""),
    ("RULE_NDHP_HFREF",                  "TIER_1_CONTRAINDICATION", ["conditions", "medications"], ""),
    ("RULE_BRADY_ABSOLUTE",              "TIER_1_CONTRAINDICATION", ["pulse", "conditions"], ""),
    ("RULE_UNCONFIRMED_EMERGENCY",       "TIER_1_CONTRAINDICATION", ["session_structure"], "Option-D terminal"),
    ("RULE_MEDICATION_MISSED",           "TIER_2_DISCREPANCY", ["adherence"], ""),
    ("RULE_BETA_BLOCKER_SOB_HF",         "TIER_2_DISCREPANCY", ["symptoms", "conditions", "medications"], ""),
    ("RULE_BRADY_SURVEILLANCE",          "TIER_2_DISCREPANCY", ["pulse"], "spans two tiers"),
    ("RULE_STANDARD_L1_HIGH",            "BP_LEVEL_1_HIGH", ["sbp", "dbp"], ""),
    ("RULE_PERSONALIZED_HIGH",           "BP_LEVEL_1_HIGH", ["sbp", "provider_target"], ""),
    ("RULE_PREGNANCY_L1_HIGH",           "BP_LEVEL_1_HIGH", ["pregnancy"], ""),
    ("RULE_HFREF_HIGH",                  "BP_LEVEL_1_HIGH", ["conditions"], ""),
    ("RULE_HFPEF_HIGH",                  "BP_LEVEL_1_HIGH", ["conditions"], ""),
    ("RULE_CAD_HIGH",                    "BP_LEVEL_1_HIGH", ["conditions"], "env-dependent threshold"),
    ("RULE_CAD_DBP_HIGH",                "BP_LEVEL_1_HIGH", ["conditions"], "env-dependent threshold"),
    ("RULE_HCM_HIGH",                    "BP_LEVEL_1_HIGH", ["conditions"], ""),
    ("RULE_DCM_HIGH",                    "BP_LEVEL_1_HIGH", ["conditions"], ""),
    ("RULE_AORTIC_STENOSIS_HIGH",        "BP_LEVEL_1_HIGH", ["conditions"], ""),
    ("RULE_AFIB_HR_HIGH",                "BP_LEVEL_1_HIGH", ["pulse", "conditions"], ""),
    ("RULE_TACHY_HR",                    "BP_LEVEL_1_HIGH", ["pulse"], ""),
    ("RULE_TACHY_WITH_PALPITATIONS",     "BP_LEVEL_1_HIGH", ["pulse", "symptoms"], ""),
    ("RULE_STANDARD_L1_LOW",             "BP_LEVEL_1_LOW", ["sbp", "dbp"], ""),
    ("RULE_AGE_65_LOW",                  "BP_LEVEL_1_LOW", ["sbp", "age"], ""),
    ("RULE_PERSONALIZED_LOW",            "BP_LEVEL_1_LOW", ["sbp", "provider_target"], ""),
    ("RULE_HFREF_LOW",                   "BP_LEVEL_1_LOW", ["conditions"], ""),
    ("RULE_HFPEF_LOW",                   "BP_LEVEL_1_LOW", ["conditions"], ""),
    ("RULE_HCM_LOW",                     "BP_LEVEL_1_LOW", ["conditions"], ""),
    ("RULE_DCM_LOW",                     "BP_LEVEL_1_LOW", ["conditions"], ""),
    ("RULE_AORTIC_STENOSIS_LOW",         "BP_LEVEL_1_LOW", ["conditions"], ""),
    ("RULE_CAD_DBP_CRITICAL",            "BP_LEVEL_1_LOW", ["conditions", "dbp"], ""),
    ("RULE_AFIB_HR_LOW",                 "BP_LEVEL_1_LOW", ["pulse", "conditions"], ""),
    ("RULE_BRADY_HR_SYMPTOMATIC",        "BP_LEVEL_1_LOW", ["pulse", "symptoms"], ""),
    ("RULE_BRADY_HR_ASYMPTOMATIC",       "BP_LEVEL_1_LOW", ["pulse"], "DEAD IN CODE upstream"),
    ("RULE_HF_DECOMPENSATION",           "BP_LEVEL_1_LOW", ["weight"], "kg/lbs bug fixed 2026-07-14"),
    ("RULE_ORTHOSTATIC_HYPOTENSION",     "BP_LEVEL_1_LOW", ["position"], "trigger scope defect open"),
    ("RULE_AFIB_PALPITATIONS",           "BP_LEVEL_1_LOW", ["symptoms", "conditions"], ""),
    ("RULE_SYNCOPE_GENERAL",             "BP_LEVEL_1_LOW", ["symptoms"], ""),
    ("RULE_PULSE_PRESSURE_WIDE",         "TIER_3_INFO", ["sbp", "dbp"], ""),
    ("RULE_PULSE_PRESSURE_NARROW",       "TIER_3_INFO", ["sbp", "dbp"], ""),
    ("RULE_LOOP_DIURETIC_HYPOTENSION",   "TIER_3_INFO", ["medications"], ""),
    ("RULE_HCM_VASODILATOR",             "TIER_3_INFO", ["conditions", "medications"], ""),
    ("RULE_DHP_CCB_LEG_SWELLING",        "TIER_3_INFO", ["symptoms", "medications"], ""),
    ("RULE_BETA_BLOCKER_DIZZINESS",      "TIER_3_INFO", ["symptoms", "medications"], ""),
    ("RULE_BETA_BLOCKER_FATIGUE",        "TIER_3_INFO", ["symptoms", "medications"], ""),
    ("RULE_BETA_BLOCKER_SOB_NON_HF",     "TIER_3_INFO", ["symptoms", "medications"], ""),
    ("RULE_PALPITATIONS_GENERAL",        "TIER_3_INFO", ["symptoms"], ""),
    ("RULE_NSAID_ANTIHTN_INTERACTION",   "TIER_3_INFO", ["symptoms", "medications"], ""),
    ("RULE_ACE_COUGH",                   "TIER_3_INFO", ["symptoms", "medications"], ""),
    ("RULE_HF_CAREGIVER_EDEMA",          "TIER_3_INFO", ["symptoms"], ""),
    ("RULE_FIRST_MONTH_ADHERENCE_NUDGE", "TIER_3_INFO", ["cadence"], ""),
    ("RULE_EMERGENCY_RANGE_CONFIRMED_NORMAL", "TIER_3_INFO", ["sbp", "session_structure"], ""),
    ("RULE_BRADY_SURVEILLANCE",           "TIER_3_INFO", ["pulse"], "same rule, run < 3 days"),
]

# What HEMOBP actually supplies. `provider_target` is supplied by the personalisation offset,
# standing in for a clinician-set PatientThreshold.
BASE_AVAILABLE_INPUTS = {"sbp", "dbp", "age", "weight", "cadence",
                         "session_structure", "provider_target"}


def detect_pulse(vip_path: str = None) -> tuple:
    """Return (available, matched_columns) by reading only the header row of vip.csv.

    The data descriptor's Methods state that pulse rate was measured at 30-minute intervals,
    but the published column list for Hemrec_VIP does not include a pulse column. Rather
    than assume either way, detect it and let the registry adapt.
    """
    cands = [vip_path, os.path.join(RAW_DATA_DIR, VIP_FILE_NAME)]
    cands += sorted(glob.glob(os.path.join(RAW_DATA_DIR, "**", "*vip*.csv"), recursive=True))
    path = next((p for p in cands if p and os.path.exists(p)), None)
    if path is None:
        logging.warning("vip.csv not found (looked in %s) -- treating pulse as ABSENT. A false "
                        "negative here blocks every HR-gated rule for no reason.",
                        [c for c in cands if c][:3])
        return False, []
    header = pd.read_csv(path, nrows=0).columns.str.strip().str.lower().tolist()
    hits = [c for c in header if c in PULSE_CANDIDATES or "pulse" in c or "heart" in c]
    return (len(hits) > 0), hits


def build_registry(vip_path: str = None) -> pd.DataFrame:
    """The rule inventory annotated with what this corpus can and cannot evaluate."""
    pulse_available, pulse_cols = detect_pulse(vip_path)
    available = set(BASE_AVAILABLE_INPUTS)
    if pulse_available:
        available |= {"pulse"}
        logging.info("pulse detected in the corpus (%s) -- HR-gated rules move to EVALUABLE",
                     pulse_cols)
    else:
        logging.info("pulse absent from the corpus -- HR-gated rules stay BLOCKED_ON_INPUTS "
                     "and remain prediction targets")

    reg = pd.DataFrame(RULES, columns=["rule_id", "tier", "needs", "code_note"])
    reg["blocked_by"] = reg.needs.apply(lambda ns: sorted(set(ns) - available))
    reg["status"] = np.where(reg.blocked_by.str.len() == 0, "EVALUABLE", "BLOCKED_ON_INPUTS")
    reg["is_emergency"] = reg.tier.isin(EMERGENCY_TIERS)
    reg["prediction_target"] = True     # none are ever dropped
    reg["pulse_available"] = pulse_available

    logging.info("inventory: %d slots across %d distinct rules over %d tiers "
                 "| EVALUABLE %d | BLOCKED_ON_INPUTS %d",
                 len(reg), reg.rule_id.nunique(), len(TIERS),
                 int((reg.status == "EVALUABLE").sum()),
                 int((reg.status == "BLOCKED_ON_INPUTS").sum()))
    return reg
