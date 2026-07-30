"""Model 4's label source: a structural causal model overlaid on the REAL panel.

## Read this before trusting any symptom metric

**No real symptom was ever observed in HEMOBP.** The corpus carries blood pressure, weight,
temperature and ultrafiltration. It carries no symptoms, no medications, no diagnosed
conditions and no adherence.

Everything this module produces is generated. A model trained on generated labels recovers
the generating process and nothing else, so a high symptom AUC downstream is a statement
about the code in this file, not about physiology. The bundle carries
`symptom_labels_synthetic=True`, a safety gate asserts that flag survives, and the API
attaches a warning to every symptom response. Those are not decoration.

## Why generate them at all

Two defensible reasons, and neither is "to get a number".

1. The rule engine has 56 rules and 45 of them are `BLOCKED_ON_INPUTS` against this corpus.
   The symptom axis is how the serving path exercises them, and the heads give the API a
   populated shape to serve so the contract is real rather than hypothetical.
2. The heads' *plumbing* -- targets, calibration, threshold selection, the operating budget
   -- is identical to what a real symptom corpus would need. Building it against generated
   labels means the day real labels arrive, only this module is replaced.

## What differs from `synthetic.py`

`synthetic.py` builds a standalone 120-patient cohort for rule reachability, with its own
19 keys. This is an OVERLAY: it merges onto the real panel by `(series_id, step)`, so each
generated symptom is conditioned on that patient's actual blood pressure on that actual day.
The two vocabularies are reconciled through `SYMPTOM_KEY_ALIAS`.

The hazards are conditioned on real physiology and calibrated to published rates, so the
generator is at least a plausible one -- but plausible is not observed, and the distinction
is the whole point of this docstring.
"""

import sys

import numpy as np
import pandas as pd

from src.constants.training_pipeline import SEED
from src.exception.custom_exception import CustomException
from src.logging.logger import logging

#: Mechanism -> the physiology that drives it. Separated because the four are driven by
#: different processes and a clinician reads them differently; averaging them into one
#: "symptom score" would destroy exactly the information that makes them useful.
SYMPTOM_MECHANISM = {
    "severe_headache": "hypertensive", "visual_changes": "hypertensive",
    "altered_mental_status": "hypertensive", "focal_neuro_deficit": "hypertensive",
    "severe_epigastric_pain": "hypertensive", "chest_pain": "hypertensive",
    "dizziness": "hypotensive", "syncope": "hypotensive",
    "palpitations": "hypotensive", "fatigue": "hypotensive",
    "leg_swelling": "volume", "shortness_of_breath": "volume",
    "dry_cough": "drug", "face_swelling": "drug", "throat_tightness": "drug",
}
SYMPTOMS = list(SYMPTOM_MECHANISM)

#: Never averaged against fatigue. A red flag next to a common symptom is not half a red
#: flag, and a mechanism score that mixes them hides the one that matters.
SYMPTOM_RED_FLAG = {"altered_mental_status", "focal_neuro_deficit", "visual_changes",
                    "chest_pain", "throat_tightness", "syncope"}

SYMPTOM_GROUPS = ["any", "red_flag", "mech_hypertensive", "mech_hypotensive",
                  "mech_volume", "mech_drug"]

#: Base rate per session, anchored to published haemodialysis cohort literature. These set
#: the intercept; the covariates below move each session away from it.
SYMPTOM_SPEC = {
    "severe_headache":        dict(rate=0.030, sbp=0.055, drop=0.000, vol=0.0, ace=0.0),
    "visual_changes":         dict(rate=0.008, sbp=0.050, drop=0.000, vol=0.0, ace=0.0),
    "altered_mental_status":  dict(rate=0.004, sbp=0.055, drop=0.020, vol=0.0, ace=0.0),
    "focal_neuro_deficit":    dict(rate=0.002, sbp=0.060, drop=0.000, vol=0.0, ace=0.0),
    "severe_epigastric_pain": dict(rate=0.006, sbp=0.040, drop=0.000, vol=0.0, ace=0.0),
    "chest_pain":             dict(rate=0.018, sbp=0.030, drop=0.020, vol=0.3, ace=0.0),
    "dizziness":              dict(rate=0.049, sbp=0.000, drop=0.055, vol=0.0, ace=0.6),
    "syncope":                dict(rate=0.006, sbp=0.000, drop=0.070, vol=0.0, ace=0.4),
    "palpitations":           dict(rate=0.030, sbp=0.010, drop=0.035, vol=0.0, ace=0.0),
    "fatigue":                dict(rate=0.120, sbp=0.005, drop=0.030, vol=0.4, ace=0.0),
    "leg_swelling":           dict(rate=0.070, sbp=0.010, drop=0.000, vol=1.6, ace=0.0),
    "shortness_of_breath":    dict(rate=0.055, sbp=0.015, drop=0.010, vol=1.8, ace=0.0),
    "dry_cough":              dict(rate=0.040, sbp=0.000, drop=0.000, vol=0.2, ace=4.2),
    "face_swelling":          dict(rate=0.003, sbp=0.000, drop=0.000, vol=0.3, ace=3.0),
    "throat_tightness":       dict(rate=0.001, sbp=0.000, drop=0.000, vol=0.0, ace=5.0),
}

#: Bridge to `synthetic.py`'s 19 keys, so the serving vocabulary and the training labels
#: describe the same clinical concepts even though the two modules name them differently.
SYMPTOM_KEY_ALIAS = {
    "altered_mental_status": "altered_mental", "focal_neuro_deficit": "focal_neuro",
    "severe_epigastric_pain": "severe_epigastric", "chest_pain": "chest_pain_dyspnea",
    "shortness_of_breath": "sob",
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def synth_session_layer(panel: pd.DataFrame, seed: int = SEED):
    """Generate a per-session clinical + symptom layer conditioned on the real panel.

    Returns (S, rates). `S` is keyed on (series_id, step) so it merges straight onto the
    panel. Patient-level traits are drawn once per patient and held constant, which is what
    makes the labels autocorrelated the way real symptoms are -- generating them iid per
    session would make the whole exercise trivially unpredictable and the metrics
    meaningless in the opposite direction.
    """
    try:
        rng = np.random.default_rng(seed)
        sids = panel.series_id.astype(str).unique()

        # ---- patient-level traits ------------------------------------------------
        # `adherence_trait` and `frailty` are LATENT: they drive the labels and are not
        # observable in any deployment. They are emitted so the audit can prove they never
        # became features, and CausalFeatureBuilder.DROP plus a leakage probe enforce it.
        n = len(sids)
        traits = pd.DataFrame({
            "series_id": sids,
            "adherence_trait": np.clip(rng.beta(6, 2, n), 0, 1),
            "frailty": np.clip(rng.beta(2, 5, n), 0, 1),
            "has_hf": (rng.random(n) < 0.40).astype(int),
            "has_ihd": (rng.random(n) < 0.393).astype(int),
            "has_afib": (rng.random(n) < 0.265).astype(int),
            "on_ace": (rng.random(n) < 0.28).astype(int),
            "on_arb": 0,
            "on_bb": (rng.random(n) < 0.55).astype(int),
            "on_loop": (rng.random(n) < 0.62).astype(int),
            "on_nsaid": (rng.random(n) < 0.09).astype(int),
        })
        # Dual RAAS blockade is contraindicated; the generator must not produce patients on
        # both, or the drug-mechanism symptoms learn from a combination that does not occur.
        both = (traits.on_ace == 1) & (traits.on_arb == 1)
        traits.loc[both, "on_arb"] = 0
        traits["on_arb"] = np.where((traits.on_ace == 0) & (rng.random(n) < 0.22), 1, 0)
        traits["n_medications"] = traits[["on_ace", "on_arb", "on_bb", "on_loop",
                                          "on_nsaid"]].sum(axis=1)

        P = panel[["series_id", "step", "sbp", "sbp_drop", "idwg", "dryweight"]].copy()
        P["series_id"] = P.series_id.astype(str)
        P = P.merge(traits, on="series_id", how="left")

        # ---- session covariates, standardised so the coefficients are interpretable ----
        sbp_ex = (P.sbp.astype(float) - 140.0).clip(lower=0)          # mmHg above threshold
        drop = P.sbp_drop.astype(float).fillna(0).clip(lower=0)        # intradialytic fall
        vol = (100.0 * P.idwg / P.dryweight).clip(0, 12).fillna(0)     # % dry weight gained

        # Adherence: driven by the latent trait, so a non-adherent patient is persistently
        # non-adherent rather than randomly so.
        adh_p = 0.35 + 0.6 * P.adherence_trait
        P["took_all_meds"] = (rng.random(len(P)) < adh_p).astype(int)
        P["missed_antihypertensive"] = ((P.took_all_meds == 0)
                                        & (P[["on_ace", "on_arb", "on_bb"]].sum(axis=1) > 0)
                                        ).astype(int)
        # Heart rate: a rate-limiting agent lowers it, frailty raises variability.
        P["heart_rate"] = np.clip(
            rng.normal(78 - 9 * P.on_bb, 9 + 4 * P.frailty), 38, 165).round()

        rates = {}
        for s, spec in SYMPTOM_SPEC.items():
            # logit intercept placed so the marginal rate lands on the literature anchor
            b0 = np.log(spec["rate"] / (1 - spec["rate"]))
            lin = (b0
                   + spec["sbp"] * sbp_ex
                   + spec["drop"] * drop
                   + spec["vol"] * (vol / 6.0)
                   + np.log(max(spec["ace"], 1e-6)) * (P.on_ace if spec["ace"] else 0)
                   + 0.8 * P.frailty
                   + 0.5 * P.missed_antihypertensive)
            p = _sigmoid(lin)
            P[f"sym_{s}"] = (rng.random(len(P)) < p).astype(int)
            rates[s] = float(P[f"sym_{s}"].mean())

        cols = [f"sym_{s}" for s in SYMPTOMS]
        P["sym_any"] = P[cols].max(axis=1)
        P["sym_count"] = P[cols].sum(axis=1)
        P["sym_red_flag"] = P[[f"sym_{s}" for s in SYMPTOM_RED_FLAG]].max(axis=1)
        for mech in ("hypertensive", "hypotensive", "volume", "drug"):
            members = [f"sym_{s}" for s, m in SYMPTOM_MECHANISM.items() if m == mech]
            P[f"sym_mech_{mech}"] = P[members].max(axis=1)

        # KDOQI intradialytic hypotension: a >=20 mmHg fall WITH symptoms.
        P["idh_kdoqi"] = ((drop >= 20) & (P.sym_any == 1)).astype(int)

        keep = (["series_id", "step", "adherence_trait", "frailty", "took_all_meds",
                 "missed_antihypertensive", "heart_rate", "n_medications", "idh_kdoqi",
                 "sym_any", "sym_count", "sym_red_flag"]
                + [c for c in P.columns if c.startswith("sym_mech_")]
                + cols
                + ["has_hf", "has_ihd", "has_afib", "on_ace", "on_arb", "on_bb",
                   "on_loop", "on_nsaid"])
        S = P[keep].copy()
        logging.warning("[SYNTHETIC LABELS] generated a symptom layer over %d sessions for "
                        "%d patients. No real symptom exists in HEMOBP; every label below "
                        "comes from symptom_layer.py and describes that generator.",
                        len(S), len(sids))
        logging.info("symptom rates: %s",
                     {k: round(v, 4) for k, v in sorted(rates.items(),
                                                        key=lambda kv: -kv[1])[:6]})
        return S, rates
    except Exception as e:
        raise CustomException(e, sys)


def symptom_param_table() -> pd.DataFrame:
    """The generator's parameters, written to the reports so the labels are auditable."""
    rows = []
    for s, spec in SYMPTOM_SPEC.items():
        rows.append(dict(symptom=s, mechanism=SYMPTOM_MECHANISM[s],
                         red_flag=s in SYMPTOM_RED_FLAG,
                         base_rate=spec["rate"], beta_sbp_excess=spec["sbp"],
                         beta_intradialytic_drop=spec["drop"],
                         beta_volume=spec["vol"], odds_ratio_ace=spec["ace"],
                         serving_key=SYMPTOM_KEY_ALIAS.get(s, s)))
    return pd.DataFrame(rows)
