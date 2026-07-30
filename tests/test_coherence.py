"""Coherence guard on the predicted (systolic, diastolic) pair.

Every check is exercised in both directions -- a violating pair that must flag AND a valid pair
that must not. A guard that flags everything is as useless as one that flags nothing, and only
the paired form can tell them apart.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.constants.training_pipeline import INGEST_RANGES  # noqa: E402
from src.utils.ml_utils.model.coherence import (  # noqa: E402
    check_bp_pair,
    coherence_reference,
    pair_coherence,
)

FAILS = []


def chk(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"  <- {extra}"))
    if not cond:
        FAILS.append(name)


def _ordering():
    bad = check_bp_pair(142, 149)
    chk("*** inverted pair is flagged (dbp above sbp) ***", not bad["ok"], bad)
    chk("    and the reason names the inversion",
        any("not below" in v for v in bad["violations"]), bad["violations"])
    chk("    with a stable machine-readable code", bad["codes"] == ["inverted"], bad["codes"])
    chk("    pulse pressure is reported signed", bad["pp"] == -7.0, bad["pp"])

    ok = check_bp_pair(142, 88)
    chk("    a normal pair is NOT flagged (the check can pass)", ok["ok"], ok)
    chk("    and its pulse pressure is right", ok["pp"] == 54.0, ok["pp"])

    eq = check_bp_pair(130, 130)
    chk("    equal legs are flagged, not treated as a zero-width pass", not eq["ok"], eq)


def _floor():
    lim = float(INGEST_RANGES["min_pulse_pressure"])
    just_under = check_bp_pair(130, 130 - (lim - 1))
    chk(f"pulse pressure just under the {lim:.0f} mmHg input floor is flagged",
        not just_under["ok"], just_under)
    just_over = check_bp_pair(130, 130 - (lim + 1))
    chk("    and just over it is not", just_over["ok"], just_over)

    # The floor must be the SAME constant the API enforces on input, not a second copy.
    from pydantic import ValidationError

    from src.serving.schemas import Reading
    narrow = {"date": "2026-01-01", "sbp": 130, "dbp": int(130 - (lim - 1))}
    try:
        Reading(**narrow)
        chk("    the same pair would be rejected as an INPUT reading", False,
            "schemas.Reading accepted it")
    except ValidationError:
        chk("    the same pair would be rejected as an INPUT reading", True)


def _ranges():
    hi = check_bp_pair(400, 90)
    chk("a systolic outside the admission range is flagged", not hi["ok"], hi)
    lo = check_bp_pair(50, 20)
    chk("    so is a pair below it", not lo["ok"], lo)
    chk("    but a pair at the edge of the range is fine",
        check_bp_pair(259, 100)["ok"], check_bp_pair(259, 100))


def _patient_baseline():
    bounds = {"pp_abs_dev_p99": 20.0}

    # Both legs individually unremarkable, but the JOINT is wrong for THIS patient. This is the
    # only check that can see that, which is the reason it exists.
    far = check_bp_pair(150, 70, pp_ref=45.0, bounds=bounds)   # pp 80 vs a usual 45
    chk("*** a pair far from the patient's own pulse pressure is flagged ***",
        not far["ok"], far)
    chk("    both legs are individually in range (so only the joint check caught it)",
        all("outside" not in v and "not below" not in v for v in far["violations"]),
        far["violations"])

    near = check_bp_pair(150, 100, pp_ref=45.0, bounds=bounds)  # pp 50 vs a usual 45
    chk("    a pair near their baseline is not flagged", near["ok"], near)

    # Absent evidence must record as SKIPPED, never as a pass.
    no_ref = check_bp_pair(150, 70, pp_ref=None, bounds=bounds)
    chk("no patient baseline -> the check is skipped, not silently passed",
        any("pp_near_patient_baseline" in s for s in no_ref["skipped"]), no_ref)
    chk("    and it is not listed as checked",
        "pp_near_patient_baseline" not in no_ref["checked"], no_ref["checked"])

    no_bounds = check_bp_pair(150, 70, pp_ref=45.0, bounds=None)
    chk("no bundle reference -> also skipped (old bundles degrade, not lie)",
        any("pp_near_patient_baseline" in s for s in no_bounds["skipped"]), no_bounds)


def _missing():
    for label, pair in (("None", (None, 80)), ("NaN", (np.nan, 80)),
                        ("inf", (np.inf, 80))):
        r = check_bp_pair(*pair)
        chk(f"a {label} leg is flagged rather than crashing", not r["ok"], r)


def _reference():
    rng = np.random.default_rng(0)
    sbp = rng.normal(145, 18, 4000)
    panel = pd.DataFrame({"sbp": sbp, "dbp": sbp - rng.normal(52, 9, 4000)})
    ref = coherence_reference(panel)
    chk("reference reports the six numbers the guard needs",
        {"pp_floor", "pp_p001", "pp_p50", "pp_p999", "pp_abs_dev_p99", "n"} <= set(ref), ref)
    chk("    the median pulse pressure is about right",
        48 < ref["pp_p50"] < 56, ref["pp_p50"])
    chk("    the dispersion bound is positive and finite",
        0 < ref["pp_abs_dev_p99"] < 100, ref["pp_abs_dev_p99"])
    chk("    an empty panel degrades instead of raising",
        "basis" in coherence_reference(pd.DataFrame({"sbp": [], "dbp": []})))


def _batch():
    # 100 clean pairs, 5 deliberately inverted.
    rng = np.random.default_rng(1)
    s = rng.normal(145, 10, 105)
    d = s - rng.normal(50, 6, 105)
    d[:5] = s[:5] + 4.0                                     # inverted
    rep = pair_coherence(s, d, horizon=0)
    chk("batch report counts exactly the injected violations",
        rep["n_violations"] == 5 and rep["n_inverted"] == 5, rep)
    # The report rounds to 6 dp on purpose; assert against that, not against full precision.
    chk("    and the rate is right", abs(rep["violation_rate"] - 5 / 105) < 1e-6, rep)
    chk("    reasons aggregate by violation TYPE, not by mmHg value",
        rep["top_reasons"] == "inverted x5", rep["top_reasons"])
    chk("    a clean batch reports zero (the report can say 'fine')",
        pair_coherence(s[5:], d[5:])["n_violations"] == 0)
    chk("    the reason breakdown is populated", bool(rep["top_reasons"]), rep)


def run():
    for title, fn in (("ordering", _ordering),
                      ("input-contract floor", _floor),
                      ("admission ranges", _ranges),
                      ("patient baseline", _patient_baseline),
                      ("missing / non-finite", _missing),
                      ("cohort reference", _reference),
                      ("batch report", _batch)):
        print(f"\n--- {title} ---")
        fn()
    print("\n" + ("ALL COHERENCE TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


def test_coherence():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
