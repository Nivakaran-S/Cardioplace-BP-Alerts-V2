"""The 39 clinical checkbox keys, their labels, groups and red-flag status.

Served from `/api/schema` rather than duplicated in JavaScript. The old SPA hardcoded a
parallel copy of these lists and drifted from them: its red-flag set used camelCase
(`severeHeadache`) against the snake_case keys the engine actually reads, so `indexOf` was
always -1 and no red-flag symptom ever got its status colour. Anything the client needs to
know about a key now comes from the server that evaluates it.
"""

from src.utils.ml_utils.rule_engine.synthetic import CONDITION_KEYS, MED_KEYS, SYMPTOM_KEYS

#: Symptoms whose presence changes the tier regardless of the blood-pressure number.
#: Kept in one place because both the engine's override rules and the UI's colour coding
#: have to agree about what counts as a red flag.
RED_FLAGS = {"severe_headache", "visual_changes", "altered_mental", "focal_neuro",
             "chest_pain_dyspnea", "syncope", "throat_tightness", "severe_epigastric"}

#: Which mechanism a symptom points at. Grouping them in the form is not cosmetic: the four
#: mechanisms are driven by different physiology and a clinician reads them differently.
SYMPTOM_GROUP = {
    "severe_headache": "Hypertensive", "visual_changes": "Hypertensive",
    "altered_mental": "Hypertensive", "focal_neuro": "Hypertensive",
    "severe_epigastric": "Hypertensive", "chest_pain_dyspnea": "Hypertensive",
    "new_onset_headache": "Hypertensive", "ruq_pain": "Hypertensive",
    "dizziness": "Hypotensive", "syncope": "Hypotensive",
    "palpitations": "Hypotensive", "fatigue": "Hypotensive",
    "leg_swelling": "Volume", "sob": "Volume", "edema": "Volume",
    "dry_cough": "Drug effect", "face_swelling": "Drug effect",
    "throat_tightness": "Drug effect", "nsaid_use": "Drug effect",
}

LABELS = {
    # conditions
    "has_hf": "Heart failure", "has_afib": "Atrial fibrillation",
    "has_cad": "Coronary artery disease", "has_hcm": "Hypertrophic cardiomyopathy",
    "has_dcm": "Dilated cardiomyopathy", "has_as": "Aortic stenosis",
    "has_tachy": "Tachyarrhythmia", "has_brady": "Bradyarrhythmia",
    "dx_htn": "Diagnosed hypertension",
    # medications
    "on_ace": "ACE inhibitor", "on_arb": "ARB", "on_bb": "Beta blocker",
    "on_dhp_ccb": "Dihydropyridine CCB", "on_ndhp_ccb": "Non-dihydropyridine CCB",
    "on_loop": "Loop diuretic", "on_thiazide": "Thiazide",
    "on_mra": "Mineralocorticoid antagonist", "on_nitrate": "Nitrate",
    "on_antiarr": "Antiarrhythmic", "on_nsaid": "NSAID",
    # symptoms
    "severe_headache": "Severe headache", "visual_changes": "Visual changes",
    "altered_mental": "Altered mental status", "chest_pain_dyspnea": "Chest pain / dyspnoea",
    "focal_neuro": "Focal neurological deficit", "severe_epigastric": "Severe epigastric pain",
    "new_onset_headache": "New-onset headache", "ruq_pain": "Right upper quadrant pain",
    "edema": "Oedema", "dizziness": "Dizziness", "syncope": "Syncope",
    "palpitations": "Palpitations", "leg_swelling": "Leg swelling", "fatigue": "Fatigue",
    "sob": "Shortness of breath", "dry_cough": "Dry cough", "nsaid_use": "Recent NSAID use",
    "face_swelling": "Facial swelling", "throat_tightness": "Throat tightness",
}


def _entry(key: str, group: str = None) -> dict:
    return {"key": key,
            "label": LABELS.get(key, key.replace("_", " ").capitalize()),
            "group": group or SYMPTOM_GROUP.get(key, "Other"),
            "red_flag": key in RED_FLAGS}


def build_vocabulary(registry=None) -> dict:
    """The field vocabulary plus how much of the rule registry the corpus can support."""
    rules = {}
    if registry is not None and len(registry):
        blocked = int((registry.status == "BLOCKED_ON_INPUTS").sum())
        rules = {
            "total_slots": int(len(registry)),
            "evaluable_on_corpus": int((registry.status == "EVALUABLE").sum()),
            "blocked_on_corpus": blocked,
            # The training corpus has no symptom, medication or condition columns, so most
            # rules are unreachable there. At SERVING time the user supplies exactly those
            # axes, so the extended engine evaluates them. Saying so matters: otherwise the
            # engine panel and this count look like they contradict each other.
            "note": (f"{blocked} of {len(registry)} rule slots are BLOCKED_ON_INPUTS "
                     f"against the HEMOBP training corpus, which carries no symptom, "
                     f"medication or condition columns. They are evaluated here from what "
                     f"you enter."),
        }
    return {
        "conditions": [_entry(k, "Condition") for k in CONDITION_KEYS],
        "medications": [_entry(k, "Medication") for k in MED_KEYS],
        "symptoms": [_entry(k) for k in SYMPTOM_KEYS],
        "hf_types": ["NONE", "HFREF", "HFPEF"],
        "positions": ["SITTING", "STANDING", "LYING"],
        "rules": rules,
    }


CONDITION_SET = set(CONDITION_KEYS)
MED_SET = set(MED_KEYS)
SYMPTOM_SET = set(SYMPTOM_KEYS)
