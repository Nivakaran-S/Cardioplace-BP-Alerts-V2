"""Synthetic daily cohort and the extended engine -- rule coverage, not evidence.

Port of notebook section 25. HEMOBP cannot reach most of the 56 rules: it has no symptoms,
medications or conditions. This module generates a fully synthetic journaling-shaped panel
so every branch of the engine can be exercised at least once, and runs the same engine
contract over it with the blocked axes populated.

Nothing produced here is evidence about real patients, and nothing here is ever merged
with HEMOBP rows -- `is_synthetic` is stamped on every row and the provenance guard in
`safety.gates` checks it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.ml_utils.feature.cadence import attach_cadence
from src.utils.ml_utils.rule_engine.engine import RuleEngine
from src.utils.ml_utils.rule_engine.registry import EMERGENCY_TIERS


@dataclass(frozen=True)
class SynthConfig:
    """Every parameter declared here with a provenance note; nothing buried in a body."""
    n_patients: int = 120
    n_days: int = 180                  # 6 months: past the ~2-3 month personalisation warm-up
    seed: int = 20260727
    # engagement / missingness (provenance: telemonitoring adherence literature, order-of-magnitude)
    p_log_weekday: float = 0.82
    p_log_weekend: float = 0.61
    # adherence: 3-state persistent chain, NOT iid (provenance: illustrative)
    p_adh_to_lapsing: float = 0.035
    p_lapsing_to_lapsed: float = 0.30
    p_lapsing_to_adh: float = 0.45
    p_lapsed_to_adh: float = 0.10
    # effect sizes, mmHg / bpm per unit (provenance: illustrative, direction is what matters)
    sbp_per_missed_day: float = 3.2
    sbp_ar_phi: float = 0.72
    sbp_noise: float = 7.5
    # seeded rare branches, so every rule has positive examples (coverage target, not prevalence)
    frac_pregnant: float = 0.10
    frac_angioedema_event: float = 0.06
    frac_brady_absolute: float = 0.05
    frac_syncope: float = 0.05
    # Rare clinical branches do not appear at plausible prevalence in a 120-patient draw, so a
    # fixed slice of the cohort is ASSIGNED to each. This is coverage engineering and is stated
    # as such: the resulting mix is not a prevalence estimate.
    archetype_frac: float = 0.34


SCFG = SynthConfig()

CONDITION_KEYS = ["has_hf", "has_afib", "has_cad", "has_hcm", "has_dcm", "has_as",
                  "has_tachy", "has_brady", "dx_htn"]
MED_KEYS = ["on_ace", "on_arb", "on_bb", "on_dhp_ccb", "on_ndhp_ccb", "on_loop",
            "on_thiazide", "on_mra", "on_nitrate", "on_antiarr", "on_nsaid"]
SYMPTOM_KEYS = ["severe_headache", "visual_changes", "altered_mental", "chest_pain_dyspnea",
                "focal_neuro", "severe_epigastric", "new_onset_headache", "ruq_pain", "edema",
                "dizziness", "syncope", "palpitations", "leg_swelling", "fatigue", "sob",
                "dry_cough", "nsaid_use", "face_swelling", "throat_tightness"]

ARCHETYPES = ["PREGNANT_HTN", "PREGNANT_ON_ACE", "ANGIO_NO_ACE", "HYPOTENSIVE_HFREF",
              "HYPOTENSIVE_DCM", "HYPOTENSIVE_AS", "HYPOTENSIVE_HFPEF", "HYPOTENSIVE_HCM",
              "NARROW_PP", "LONE_EMERGENCY_READING", "ELDERLY_LOW", "AFIB_RVR",
              "EMERGENCY_THEN_NORMAL", "GENERAL", "GENERAL"]


def synth_param_table(cfg: SynthConfig = SCFG) -> pd.DataFrame:
    return pd.DataFrame([(f.name, getattr(cfg, f.name))
                         for f in cfg.__dataclass_fields__.values()],
                        columns=["parameter", "value"])


def _apply_archetype(p: dict, tag: str, rng) -> dict:
    """Force the patient onto a branch the random draw would almost never reach."""
    if tag.startswith("PREGNANT"):
        p.update(is_male=0, age=int(rng.integers(23, 42)), is_pregnant=1,
                 sbp_base=float(rng.uniform(148, 172)), dbp_base=float(rng.uniform(96, 116)))
        p["on_ace"] = int(tag == "PREGNANT_ON_ACE")
        p["on_arb"] = 0
    elif tag == "ANGIO_NO_ACE":
        p.update(on_ace=0, on_arb=0)                      # so GENERIC, not ACE, claims the axis
        p["angio_day"] = int(rng.integers(15, 150))
    elif tag == "HYPOTENSIVE_HFREF":
        p.update(has_hf=1, hf_type="HFREF", sbp_base=float(rng.uniform(88, 98)))
    elif tag == "HYPOTENSIVE_HFPEF":
        p.update(has_hf=1, hf_type="HFPEF", sbp_base=float(rng.uniform(88, 98)))
    elif tag == "HYPOTENSIVE_DCM":
        p.update(has_dcm=1, sbp_base=float(rng.uniform(92, 102)))
    elif tag == "HYPOTENSIVE_AS":
        p.update(has_as=1, sbp_base=float(rng.uniform(92, 102)))
    elif tag == "HYPOTENSIVE_HCM":
        p.update(has_hcm=1, sbp_base=float(rng.uniform(92, 102)))
    elif tag == "NARROW_PP":
        p["dbp_base"] = float(p["sbp_base"] - rng.uniform(12, 22))
    elif tag == "AFIB_RVR":
        p.update(has_afib=1, on_bb=0, hr_base=float(rng.uniform(104, 126)))
    elif tag == "ELDERLY_LOW":
        p.update(age=int(rng.integers(70, 88)), sbp_base=float(rng.uniform(88, 99)))
    if "HYPOTENSIVE" in tag or tag == "ELDERLY_LOW":
        p["dbp_base"] = float(np.clip(p["sbp_base"] * 0.62, 44, 74))
        p["provider_target"] = float(np.clip(p["sbp_base"] + 26, 110, 150))
    p["archetype"] = tag
    return p


def _assign_patient(rng, i: int, cfg: SynthConfig) -> dict:
    """Conditions first, then a medication portfolio that is consistent with them."""
    age = int(np.clip(rng.normal(63, 13), 22, 92))
    p = dict(series_id=f"SYN{i:04d}", age=age, is_male=int(rng.random() < 0.52))
    p["is_pregnant"] = int(p["is_male"] == 0 and age < 45 and rng.random() < cfg.frac_pregnant)
    p["history_hdp"] = int(p["is_pregnant"] and rng.random() < 0.3)
    p["has_hf"] = int(rng.random() < 0.34)
    p["hf_type"] = rng.choice(["HFREF", "HFPEF"]) if p["has_hf"] else "NONE"
    p["has_afib"] = int(rng.random() < 0.18)
    p["has_cad"] = int(rng.random() < 0.26)
    p["has_hcm"] = int(rng.random() < 0.05)
    p["has_dcm"] = int(rng.random() < 0.05)
    p["has_as"] = int(rng.random() < 0.06)
    p["has_tachy"] = int(rng.random() < 0.08)
    p["has_brady"] = int(rng.random() < cfg.frac_brady_absolute + 0.05)
    p["dx_htn"] = int(rng.random() < 0.72)

    # portfolio conditional on conditions
    p["on_ace"] = int(rng.random() < (0.55 if p["has_hf"] or p["dx_htn"] else 0.15))
    p["on_arb"] = int((not p["on_ace"]) and rng.random() < 0.30)
    p["on_bb"] = int(rng.random() < (0.80 if (p["has_hf"] or p["has_afib"] or p["has_hcm"])
                                     else 0.25))
    p["on_dhp_ccb"] = int(rng.random() < 0.30)
    # non-DHP CCB in HFrEF is a contraindication -- seeded deliberately, at a low rate
    p["on_ndhp_ccb"] = int(rng.random() < (0.08 if p["hf_type"] == "HFREF" else 0.12))
    p["on_loop"] = int(rng.random() < (0.75 if p["has_hf"] else 0.10))
    p["on_thiazide"] = int(rng.random() < 0.28)
    p["on_mra"] = int(p["has_hf"] and rng.random() < 0.45)
    p["on_nitrate"] = int(rng.random() < (0.30 if p["has_cad"] else 0.05))
    p["on_antiarr"] = int(p["has_afib"] and rng.random() < 0.35)
    p["on_nsaid"] = int(rng.random() < 0.14)

    p["sbp_base"] = float(np.clip(rng.normal(124 + 5 * p["dx_htn"], 11), 92, 162))
    p["dbp_base"] = float(np.clip(p["sbp_base"] * 0.62 + rng.normal(2, 4), 54, 104))
    p["hr_base"] = float(np.clip(rng.normal(74 - 8 * p["on_bb"], 9), 42, 108))
    p["weight_base"] = float(np.clip(rng.normal(78, 15), 42, 140))
    p["provider_target"] = float(np.clip(p["sbp_base"] + rng.normal(-4, 6), 110, 150))
    p["angio_day"] = (int(rng.integers(20, cfg.n_days - 5))
                      if rng.random() < cfg.frac_angioedema_event else -1)
    p["syncope_day"] = (int(rng.integers(20, cfg.n_days - 5))
                        if rng.random() < cfg.frac_syncope else -1)
    p["archetype"] = "GENERAL"
    p["lone_reading_days"] = set()
    p["spike_days"] = set()
    tag = (ARCHETYPES[i % len(ARCHETYPES)]
           if (i / max(1, cfg.n_patients)) < cfg.archetype_frac else "GENERAL")
    if tag == "EMERGENCY_THEN_NORMAL":
        p["archetype"] = tag
        p["sbp_base"] = float(rng.uniform(116, 126))
        p["spike_days"] = {int(d) for d in rng.integers(15, cfg.n_days - 5, size=8)}
    elif tag == "LONE_EMERGENCY_READING":
        # Option-D: an emergency-range reading with no retake in the session. The engine must
        # route this to RULE_UNCONFIRMED_EMERGENCY, not to RULE_ABSOLUTE_EMERGENCY.
        p["archetype"] = tag
        p["lone_reading_days"] = {int(d) for d in rng.integers(15, cfg.n_days - 5, size=6)}
        p["sbp_base"] = float(rng.uniform(150, 166))
    elif tag != "GENERAL":
        p = _apply_archetype(p, tag, rng)
    return p


def generate_synthetic_cohort(cfg: SynthConfig = SCFG) -> pd.DataFrame:
    """Daily journaling-shaped panel. Fully synthetic: never merged with HEMOBP rows."""
    rng = np.random.default_rng(cfg.seed)
    t0 = pd.Timestamp("2026-01-05")                       # a Monday
    rows = []
    for i in range(cfg.n_patients):
        p = _assign_patient(rng, i, cfg)
        adh_state, resid, miss_hist, brady_run = "ADHERENT", 0.0, [], 0
        step, prev_hr_high_day = 0, -99
        for d in range(cfg.n_days):
            ts = t0 + pd.Timedelta(days=d)
            weekend = ts.dayofweek >= 5
            # adherence state machine (persistent, weekday-modulated)
            u = rng.random()
            if adh_state == "ADHERENT" and u < cfg.p_adh_to_lapsing * (1.8 if weekend else 1.0):
                adh_state = "LAPSING"
            elif adh_state == "LAPSING":
                adh_state = "LAPSED" if u < cfg.p_lapsing_to_lapsed else (
                    "ADHERENT" if u > 1 - cfg.p_lapsing_to_adh else "LAPSING")
            elif adh_state == "LAPSED" and u < cfg.p_lapsed_to_adh:
                adh_state = "ADHERENT"
            missed = int(adh_state == "LAPSED"
                         or (adh_state == "LAPSING" and rng.random() < 0.5))
            miss_hist.append(missed)
            miss_hist = miss_hist[-30:]

            # did the patient log today? (missingness is absence, not imputation)
            if rng.random() > (cfg.p_log_weekend if weekend else cfg.p_log_weekday):
                continue

            # vitals: adherence drives drift, AR(1) carries the trend
            miss3 = int(sum(miss_hist[-3:]))
            drift = cfg.sbp_per_missed_day * sum(miss_hist[-7:])
            resid = cfg.sbp_ar_phi * resid + rng.normal(0, cfg.sbp_noise)
            dow_effect = 2.5 if weekend else 0.0
            sbp = p["sbp_base"] + drift + resid + dow_effect
            dbp = p["dbp_base"] + 0.42 * (drift + resid) + rng.normal(0, 4)
            hr = p["hr_base"] + 0.25 * drift + rng.normal(0, 6)
            if p["is_pregnant"] and d > cfg.n_days * 0.6:
                sbp += 12
                dbp += 8                                   # gestational trajectory
            if p["has_brady"] and rng.random() < 0.25:
                hr = rng.uniform(33, 49)
            if p["has_tachy"] and rng.random() < 0.25:
                hr = rng.uniform(101, 138)
            if d in p["spike_days"]:
                sbp = max(sbp, 186.0)     # confirmed emergency; next reading returns to normal
            if d in p["lone_reading_days"]:
                sbp = max(sbp, 182.0)     # emergency range, single reading, never retaken
            weight = p["weight_base"] + 0.35 * sum(miss_hist[-5:]) + rng.normal(0, 0.5)
            wdelta = (1.4 if (p["has_hf"] and adh_state == "LAPSED" and rng.random() < 0.4)
                      else float(rng.normal(0, 0.3)))

            # symptoms: hazard conditional on drift, drug class, condition
            def bern(pr):
                return int(rng.random() < np.clip(pr, 0, 0.98))

            sev = max(0.0, (sbp - 165) / 60)
            s = {k: 0 for k in SYMPTOM_KEYS}
            s["severe_headache"] = bern(0.01 + 0.16 * sev)
            s["visual_changes"] = bern(0.005 + 0.10 * sev)
            s["altered_mental"] = bern(0.003 + 0.07 * sev)
            s["chest_pain_dyspnea"] = bern(0.008 + 0.09 * sev + 0.02 * p["has_cad"])
            s["focal_neuro"] = bern(0.002 + 0.06 * sev)
            s["severe_epigastric"] = bern(0.003 + 0.05 * sev)
            if p["is_pregnant"]:
                s["new_onset_headache"] = bern(0.02 + 0.14 * sev)
                s["ruq_pain"] = bern(0.01 + 0.08 * sev)
            s["edema"] = bern(0.006 + 0.12 * p["has_hf"] * (adh_state == "LAPSED"))
            s["dizziness"] = bern(0.005 + 0.20 * (hr < 50) + 0.04 * p["on_bb"] * (hr < 55))
            s["syncope"] = bern(0.40 if d == p["syncope_day"] else 0.002 + 0.05 * (hr < 45))
            s["palpitations"] = bern(0.004 + 0.20 * (hr > 110) + 0.02 * p["has_afib"])
            s["leg_swelling"] = bern(0.004 + 0.05 * p["on_dhp_ccb"])
            s["fatigue"] = bern(0.006 + 0.04 * p["on_bb"])
            s["sob"] = bern(0.005 + 0.05 * p["has_hf"] + 0.02 * p["on_bb"])
            s["dry_cough"] = bern(0.004 + 0.05 * p["on_ace"])
            s["nsaid_use"] = bern(0.10 * p["on_nsaid"])
            if d == p["angio_day"]:
                s["face_swelling"], s["throat_tightness"] = 1, 1

            brady_run = brady_run + 1 if 40 <= hr <= 49 and not s["dizziness"] else 0
            hr_high_recent = int(d - prev_hr_high_day <= 1)
            if hr > 100:
                prev_hr_high_day = d

            rows.append(dict(
                series_id=p["series_id"], step=step, ts=ts,
                n_meas=(1 if d in p["lone_reading_days"] else int(rng.integers(2, 4))),
                sbp=float(sbp), dbp=float(dbp), pulse=float(hr), weight=float(weight),
                weight_delta_24h=float(wdelta), idwg=np.nan,
                age=p["age"], is_male=p["is_male"], is_dm=int(rng.random() < 0.3),
                is_pregnant=p["is_pregnant"], history_hdp=p["history_hdp"],
                hf_type=p["hf_type"], provider_target=p["provider_target"],
                position=("STANDING" if rng.random() < 0.15 else "SITTING"),
                adherence_state=adh_state, missed_today=missed, missed_3d=miss3,
                missed_bb_today=int(missed and p["on_bb"]),
                adherence_7d=float(1 - np.mean(miss_hist[-7:])),
                brady_run_len=int(brady_run), hr_high_recent=hr_high_recent,
                archetype=p["archetype"],
                **{k: p[k] for k in CONDITION_KEYS}, **{k: p[k] for k in MED_KEYS}, **s))
            step += 1

    df = pd.DataFrame(rows)
    # This cohort is generated daily, so it does NOT share the real panel's conventions: a
    # first row is 1 day not 2, and gaps stay unclipped so the generator's deliberate skips
    # reach the stale gate at their true length. Routing it through the same function with
    # explicit arguments turns what used to be an accidental divergence into a declared one.
    df = attach_cadence(df, by="series_id", first=1.0, lo=None, hi=None)
    df["is_synthetic"] = True
    df["patient_split"] = np.where(
        pd.util.hash_pandas_object(df.series_id, index=False) % 10 < 3, "holdout", "fit")
    return df


# --------------------------------------------------------------- extended engine

EMIT_DEAD_RULE = False   # RULE_BRADY_HR_ASYMPTOMATIC is dead in code upstream. Keeping it dead
                         # here is the faithful choice; flip only to test the post-fix world.

CAD_SBP_THRESHOLD, CAD_DBP_THRESHOLD = 130.0, 80.0   # env-dependent, snapshotted at label time


def _rv(row, name, default=0):
    v = getattr(row, name, default)
    return default if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


def _personalised_high(r) -> bool:
    """Model 2's learned threshold when one is supplied, else the engine's fixed
    provider-target + 20.

    This is the learned generalisation of the engine's existing personalised mode:
    the ML layer supplies the offset instead of the hard-coded 20 mmHg. `run` puts
    the threshold on the row as `ml_threshold`, because predicates only receive the
    row.

    It cannot suppress an alert. The predicate only ever ADDS a way for the L1 axis
    to fire; nothing downstream is removed, and RULE_STANDARD_L1_HIGH still sits
    directly underneath it in the same tier, so SBP >= 140 fires with or without a
    learned value. The caps put the threshold in [115, 155] around the population
    140, which splits into two regimes:

      141-155  loosening -- the standard rule already claimed the reading, so this
               only changes which rule_id is recorded.
      115-139  tightening -- the reading is abnormal for THIS patient though below
               the population floor, and fires where the population run was silent.

    Tightening is the point of the asymmetric caps (-25 tighten vs +15 loosen): the
    engine is deliberately more willing to tighten than to loosen. Differential
    execution over the synthetic cohort confirms the direction -- 94 newly-fired and
    21 escalated rows, zero suppressed and zero softened.

    The emergency axis runs earlier and short-circuits, and Gate 3 asserts every
    threshold stays under the 180 mmHg floor, so no learned value can reach it.
    """
    ml = _rv(r, "ml_threshold", np.nan)
    if np.isfinite(ml) and r.sbp >= ml:
        return True
    pt = _rv(r, "provider_target", np.nan)
    return bool(np.isfinite(pt) and r.sbp >= pt + 20)


def _personalised_low(r) -> bool:
    """Model 2 produces an upper threshold only, so the low branch keeps the
    engine's provider-target - 20."""
    pt = _rv(r, "provider_target", np.nan)
    return bool(np.isfinite(pt) and r.sbp <= pt - 20)


# axis -> ordered [(rule_id, tier, predicate)]. Order IS the clinical semantics.
AXIS_RULES = {
    "EMERGENCY": [
        ("RULE_ACE_ANGIOEDEMA", "TIER_1_ANGIOEDEMA",
         lambda r: (_rv(r, "face_swelling") or _rv(r, "throat_tightness"))
         and (_rv(r, "on_ace") or _rv(r, "on_arb"))),
        ("RULE_GENERIC_ANGIOEDEMA", "TIER_1_ANGIOEDEMA",
         lambda r: (_rv(r, "face_swelling") or _rv(r, "throat_tightness"))),
        ("RULE_SYMPTOM_OVERRIDE_PREGNANCY", "BP_LEVEL_2_SYMPTOM_OVERRIDE",
         lambda r: _rv(r, "is_pregnant") and (_rv(r, "new_onset_headache")
                                              or _rv(r, "ruq_pain") or _rv(r, "edema"))),
        ("RULE_SYMPTOM_OVERRIDE_GENERAL", "BP_LEVEL_2_SYMPTOM_OVERRIDE",
         lambda r: any(_rv(r, k) for k in ("severe_headache", "visual_changes", "altered_mental",
                                           "chest_pain_dyspnea", "focal_neuro",
                                           "severe_epigastric"))),
        ("RULE_PREGNANCY_L2", "BP_LEVEL_2",
         lambda r: _rv(r, "is_pregnant") and (r.sbp >= 160 or r.dbp >= 110)),
        ("RULE_ABSOLUTE_EMERGENCY", "BP_LEVEL_2",
         lambda r: (r.sbp >= 180 or r.dbp >= 120) and _rv(r, "n_meas", 99) >= 2),
    ],
    "CONTRAINDICATION": [
        ("RULE_PREGNANCY_ACE_ARB", "TIER_1_CONTRAINDICATION",
         lambda r: _rv(r, "is_pregnant") and (_rv(r, "on_ace") or _rv(r, "on_arb"))),
        ("RULE_NDHP_HFREF", "TIER_1_CONTRAINDICATION",
         lambda r: str(_rv(r, "hf_type", "NONE")) == "HFREF" and _rv(r, "on_ndhp_ccb")),
        ("RULE_BRADY_ABSOLUTE", "TIER_1_CONTRAINDICATION",
         lambda r: _rv(r, "pulse", 99) < 40 and (_rv(r, "has_brady") or _rv(r, "on_bb"))),
        ("RULE_UNCONFIRMED_EMERGENCY", "TIER_1_CONTRAINDICATION",
         lambda r: (r.sbp >= 180 or r.dbp >= 120) and _rv(r, "n_meas", 99) < 2),
    ],
    "TIER_2": [
        ("RULE_MEDICATION_MISSED", "TIER_2_DISCREPANCY",
         lambda r: _rv(r, "missed_3d") >= 2
         or (_rv(r, "missed_bb_today") and (_rv(r, "has_hf") or _rv(r, "has_hcm")
                                            or _rv(r, "has_afib")))),
        ("RULE_BETA_BLOCKER_SOB_HF", "TIER_2_DISCREPANCY",
         lambda r: _rv(r, "sob") and _rv(r, "has_hf") and _rv(r, "on_bb")),
        ("RULE_BRADY_SURVEILLANCE", "TIER_2_DISCREPANCY",
         lambda r: _rv(r, "brady_run_len") >= 3),
    ],
    "L1": [
        ("RULE_PREGNANCY_L1_HIGH", "BP_LEVEL_1_HIGH",
         lambda r: _rv(r, "is_pregnant") and (r.sbp >= 140 or r.dbp >= 90)),
        ("RULE_HFREF_HIGH", "BP_LEVEL_1_HIGH",
         lambda r: str(_rv(r, "hf_type", "NONE")) == "HFREF" and r.sbp >= 130),
        ("RULE_HFPEF_HIGH", "BP_LEVEL_1_HIGH",
         lambda r: str(_rv(r, "hf_type", "NONE")) == "HFPEF" and r.sbp >= 130),
        ("RULE_HCM_HIGH", "BP_LEVEL_1_HIGH", lambda r: _rv(r, "has_hcm") and r.sbp >= 135),
        ("RULE_DCM_HIGH", "BP_LEVEL_1_HIGH", lambda r: _rv(r, "has_dcm") and r.sbp >= 135),
        ("RULE_AORTIC_STENOSIS_HIGH", "BP_LEVEL_1_HIGH",
         lambda r: _rv(r, "has_as") and r.sbp >= 140),
        ("RULE_CAD_HIGH", "BP_LEVEL_1_HIGH",
         lambda r: _rv(r, "has_cad") and r.sbp >= CAD_SBP_THRESHOLD),
        ("RULE_CAD_DBP_HIGH", "BP_LEVEL_1_HIGH",
         lambda r: _rv(r, "has_cad") and r.dbp >= CAD_DBP_THRESHOLD),
        ("RULE_AFIB_HR_HIGH", "BP_LEVEL_1_HIGH",
         lambda r: _rv(r, "has_afib") and _rv(r, "pulse", 0) > 110),
        ("RULE_TACHY_WITH_PALPITATIONS", "BP_LEVEL_1_HIGH",
         lambda r: _rv(r, "pulse", 0) > 100 and _rv(r, "palpitations")),
        ("RULE_TACHY_HR", "BP_LEVEL_1_HIGH",
         lambda r: _rv(r, "pulse", 0) > 130
         or (_rv(r, "pulse", 0) > 100 and _rv(r, "hr_high_recent"))),
        ("RULE_PERSONALIZED_HIGH", "BP_LEVEL_1_HIGH", _personalised_high),
        ("RULE_STANDARD_L1_HIGH", "BP_LEVEL_1_HIGH", lambda r: r.sbp >= 140 or r.dbp >= 90),
        # --- low branch ---
        ("RULE_SYNCOPE_GENERAL", "BP_LEVEL_1_LOW", lambda r: _rv(r, "syncope")),
        ("RULE_HF_DECOMPENSATION", "BP_LEVEL_1_LOW",
         lambda r: _rv(r, "has_hf") and _rv(r, "weight_delta_24h", 0) >= 1.0),
        ("RULE_ORTHOSTATIC_HYPOTENSION", "BP_LEVEL_1_LOW",
         lambda r: str(_rv(r, "position", "SITTING")) == "STANDING" and r.sbp < 100),
        ("RULE_AFIB_PALPITATIONS", "BP_LEVEL_1_LOW",
         lambda r: _rv(r, "has_afib") and _rv(r, "palpitations")),
        ("RULE_BRADY_HR_SYMPTOMATIC", "BP_LEVEL_1_LOW",
         lambda r: 40 <= _rv(r, "pulse", 99) <= 49
         and (_rv(r, "dizziness") or _rv(r, "syncope"))),
        ("RULE_BRADY_HR_ASYMPTOMATIC", "BP_LEVEL_1_LOW",
         lambda r: EMIT_DEAD_RULE and 40 <= _rv(r, "pulse", 99) <= 49),
        ("RULE_AFIB_HR_LOW", "BP_LEVEL_1_LOW",
         lambda r: _rv(r, "has_afib") and _rv(r, "pulse", 99) < 60),
        ("RULE_CAD_DBP_CRITICAL", "BP_LEVEL_1_LOW",
         lambda r: _rv(r, "has_cad") and r.dbp < 60),
        ("RULE_HFREF_LOW", "BP_LEVEL_1_LOW",
         lambda r: str(_rv(r, "hf_type", "NONE")) == "HFREF" and r.sbp < 100),
        ("RULE_HFPEF_LOW", "BP_LEVEL_1_LOW",
         lambda r: str(_rv(r, "hf_type", "NONE")) == "HFPEF" and r.sbp < 100),
        ("RULE_HCM_LOW", "BP_LEVEL_1_LOW", lambda r: _rv(r, "has_hcm") and r.sbp < 105),
        ("RULE_DCM_LOW", "BP_LEVEL_1_LOW", lambda r: _rv(r, "has_dcm") and r.sbp < 105),
        ("RULE_AORTIC_STENOSIS_LOW", "BP_LEVEL_1_LOW",
         lambda r: _rv(r, "has_as") and r.sbp < 105),
        ("RULE_AGE_65_LOW", "BP_LEVEL_1_LOW",
         lambda r: _rv(r, "age", 0) >= 65 and r.sbp < 100),
        ("RULE_PERSONALIZED_LOW", "BP_LEVEL_1_LOW", _personalised_low),
        ("RULE_STANDARD_L1_LOW", "BP_LEVEL_1_LOW", lambda r: r.sbp < 90 or r.dbp < 60),
    ],
    "TIER_3": [
        ("RULE_EMERGENCY_RANGE_CONFIRMED_NORMAL", "TIER_3_INFO",
         lambda r: _rv(r, "prev_emergency") and r.sbp < 140),
        ("RULE_HCM_VASODILATOR", "TIER_3_INFO",
         lambda r: _rv(r, "has_hcm") and (_rv(r, "on_dhp_ccb") or _rv(r, "on_nitrate"))),
        ("RULE_LOOP_DIURETIC_HYPOTENSION", "TIER_3_INFO",
         lambda r: _rv(r, "on_loop") and r.sbp < 105),
        ("RULE_ACE_COUGH", "TIER_3_INFO",
         lambda r: _rv(r, "on_ace") and _rv(r, "dry_cough")),
        ("RULE_DHP_CCB_LEG_SWELLING", "TIER_3_INFO",
         lambda r: _rv(r, "on_dhp_ccb") and _rv(r, "leg_swelling")),
        ("RULE_BETA_BLOCKER_DIZZINESS", "TIER_3_INFO",
         lambda r: _rv(r, "on_bb") and _rv(r, "dizziness")),
        ("RULE_BETA_BLOCKER_SOB_NON_HF", "TIER_3_INFO",
         lambda r: _rv(r, "on_bb") and _rv(r, "sob") and not _rv(r, "has_hf")),
        ("RULE_BETA_BLOCKER_FATIGUE", "TIER_3_INFO",
         lambda r: _rv(r, "on_bb") and _rv(r, "fatigue")),
        ("RULE_NSAID_ANTIHTN_INTERACTION", "TIER_3_INFO",
         lambda r: _rv(r, "nsaid_use") and (_rv(r, "on_ace") or _rv(r, "on_arb")
                                            or _rv(r, "on_thiazide"))),
        ("RULE_HF_CAREGIVER_EDEMA", "TIER_3_INFO",
         lambda r: _rv(r, "has_hf") and _rv(r, "edema")),
        ("RULE_PALPITATIONS_GENERAL", "TIER_3_INFO", lambda r: _rv(r, "palpitations")),
        ("RULE_BRADY_SURVEILLANCE", "TIER_3_INFO",
         lambda r: 0 < _rv(r, "brady_run_len") < 3),
        ("RULE_PULSE_PRESSURE_WIDE", "TIER_3_INFO", lambda r: (r.sbp - r.dbp) >= 60),
        ("RULE_PULSE_PRESSURE_NARROW", "TIER_3_INFO", lambda r: (r.sbp - r.dbp) <= 25),
        ("RULE_FIRST_MONTH_ADHERENCE_NUDGE", "TIER_3_INFO",
         lambda r: _rv(r, "step", 99) < 30 and _rv(r, "adherence_7d", 1.0) < 0.8),
    ],
}


class ExtendedRuleEngine(RuleEngine):
    """The same engine contract, with the symptom / medication / condition axes populated."""

    def evaluate_row(self, row, personal_thr, prev_emergency) -> dict:
        # `personal_thr` is kept for the base-class signature but is not read here:
        # this engine dispatches through AXIS_RULES, whose predicates take the row
        # alone, so `run` hands the threshold over as the `ml_threshold` column.
        gate = self._gate(row)
        out = dict(tier=None, rule_id=None, mode="STANDARD", gate_reason=gate,
                   short_circuited=False)

        for rid, tier, pred in AXIS_RULES["EMERGENCY"]:
            if pred(row):
                if gate == "HISTORICAL_ENTRY":
                    return out                    # stale entry: no L2 fire
                out.update(tier=tier, rule_id=rid, short_circuited=True)
                return out                        # emergency exclusivity

        if gate == "SINGLE_READING_NON_EMERGENCY":
            # Option-D: a lone emergency-range reading never retaken is not a silent drop --
            # it routes to the contraindication axis. Every other rule stays gated.
            rid, tier, pred = AXIS_RULES["CONTRAINDICATION"][3]
            if pred(row):
                out.update(tier=tier, rule_id=rid, gate_reason="UNCONFIRMED_TERMINAL")
            return out
        if gate != "NONE":
            return out                            # gated: not an alert, NOT a clean negative

        for axis in ("CONTRAINDICATION", "TIER_2", "L1"):
            for rid, tier, pred in AXIS_RULES[axis]:
                if pred(row):
                    out.update(tier=tier, rule_id=rid,
                               mode=("PERSONALIZED" if rid.startswith("RULE_PERSONALIZED")
                                     else "STANDARD"))
                    return out

        for rid, tier, pred in AXIS_RULES["TIER_3"]:
            if pred(row):
                out.update(tier=tier, rule_id=rid)
                return out
        return out

    def run(self, panel: pd.DataFrame, use_personalisation: bool = False) -> pd.DataFrame:
        rows = []
        for sid, g in panel.groupby("series_id", sort=False):
            g = g.sort_values("step")
            thr = self.personal.get(sid) if use_personalisation else None
            # Predicates only receive the row, so Model 2's threshold rides along as a
            # column. Without this the learned offset is computed, reported, and then
            # silently ignored by every rule.
            if thr is not None:
                g = g.assign(ml_threshold=float(thr))
            prev_emerg = False
            for row in g.itertuples():
                r = self.evaluate_row(row, thr, prev_emerg)
                prev_emerg = bool(r["tier"] in ("BP_LEVEL_2", "BP_LEVEL_2_SYMPTOM_OVERRIDE"))
                rows.append(dict(series_id=sid, step=int(row.step), ts=row.ts,
                                 sbp=float(row.sbp), dbp=float(row.dbp), **r))
        out = pd.DataFrame(rows)
        out["fired"] = out.tier.notna()
        out["is_emergency"] = out.tier.isin(EMERGENCY_TIERS)
        return out


def prepare_synthetic_run(synth: pd.DataFrame):
    """Attach the columns the extended predicates read, and the per-patient thresholds.

    `prev_emergency` is consumed by RULE_EMERGENCY_RANGE_CONFIRMED_NORMAL; itertuples rows
    are immutable namedtuples, so it is carried as a column rather than mutated per row.
    """
    run = synth.copy()
    run["prev_emergency"] = (run.groupby("series_id").sbp.shift(1) >= 180).fillna(False)
    personal = {sid: float(np.clip(g.sbp.head(48).quantile(0.90), 120, 165))
                for sid, g in run.groupby("series_id")}
    return run, personal


def rule_coverage(registry: pd.DataFrame, alerts_real: pd.DataFrame,
                  alerts_synth: pd.DataFrame) -> pd.DataFrame:
    """Which rules fired where. Synthetic fires prove reachability, never prevalence."""
    real_fired = set(alerts_real.loc[alerts_real.fired, "rule_id"].dropna().unique())
    syn_fired = set(alerts_synth.loc[alerts_synth.fired, "rule_id"].dropna().unique()) \
        if alerts_synth is not None and len(alerts_synth) else set()
    cov = registry[["rule_id", "tier", "status"]].drop_duplicates("rule_id").copy()
    cov["fired_on_hemobp"] = cov.rule_id.isin(real_fired)
    cov["fired_on_synthetic"] = cov.rule_id.isin(syn_fired)
    cov["covered"] = cov.fired_on_hemobp | cov.fired_on_synthetic
    cov["evidence"] = np.where(cov.fired_on_hemobp, "real data",
                               np.where(cov.fired_on_synthetic, "synthetic (reachability only)",
                                        "never fired"))
    return cov
