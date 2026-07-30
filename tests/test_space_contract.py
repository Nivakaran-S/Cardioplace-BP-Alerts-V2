"""Hugging Face Space contract.

The Space is a second front end over the same clinical logic, and the failure mode worth
guarding against is not a crash -- it is silent divergence: the Space quietly showing a
different threshold, tier or verdict than `POST /api/predict` for the same patient. The
equivalence check below is the reason this file exists; everything else is a smoke test.

Every check here is written so it can fail. Where that is not obvious the assertion is
paired with a negative case in the same function.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient  # noqa: E402

import app as A  # noqa: E402
import gradio_app as G  # noqa: E402

FAILS = []


def chk(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"  <- {extra}"))
    if not cond:
        FAILS.append(name)


DEFAULTS = dict(patient_id="t1", age=68, sex="Male", diabetic=False, pregnant=False,
                hf_type="NONE", provider_target=None, conditions=[], medications=[],
                symptoms=[], position="SITTING", missed_3d=0, adherence_7d=100)


def call(text=None, **over):
    kw = {**DEFAULTS, **over}
    return G.assess(text if text is not None else G.SAMPLE, kw["patient_id"], kw["age"],
                    kw["sex"], kw["diabetic"], kw["pregnant"], kw["hf_type"],
                    kw["provider_target"], kw["conditions"], kw["medications"],
                    kw["symptoms"], kw["position"], kw["missed_3d"], kw["adherence_7d"])


def _parsing():
    rows = G.parse_readings("2026-01-01, 140, 80\n\n# a comment\n2026-01-03, 145, 82  ")
    chk("parser: skips blanks and comments", len(rows) == 2, rows)
    chk("parser: coerces to the int the schema wants",
        isinstance(rows[0]["sbp"], int) and rows[0]["sbp"] == 140)
    chk("parser: takes an optional pulse",
        G.parse_readings("2026-01-01, 140, 80, 72")[0].get("pulse") == 72.0)
    for bad, why in (("2026-01-01, 140", "too few fields"),
                     ("nope, 140, 80", "unparseable date"),
                     ("", "empty input"),
                     ("# only a comment", "no readings")):
        try:
            G.parse_readings(bad)
            chk(f"parser rejects {why}", False, f"accepted {bad!r}")
        except ValueError:
            chk(f"parser rejects {why}", True)


def _rendering():
    b, tiles, fc, chart, eng, sym, raw = call()
    chk("assess returns the 7 outputs the UI declares",
        all(x is not None for x in (b, tiles, fc, chart, eng, sym, raw)))
    chk("banner is rendered HTML", b.startswith("<div") and "</div>" in b)
    chk("engine table has one row per reading",
        len(eng) == len(G.parse_readings(G.SAMPLE)), f"{len(eng)} rows")
    chk("engine table carries the SBP joined back from the input",
        "SBP" in eng.columns and eng.SBP.notna().all())
    chk("summary names the emergency floor", "180" in tiles)
    chk("chart carries the observed series and the floor",
        {"Observed SBP", "Emergency floor (180)"} <= set(chart.series))
    chk("raw tab is strict JSON with no NaN and no python reprs",
        _strict(raw) and "<bound method" not in json.dumps(raw)
        and "object at 0x" not in json.dumps(raw))


def _strict(obj):
    try:
        json.loads(json.dumps(obj),
                   parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        return True
    except (ValueError, TypeError):
        return False


def _clinical():
    # The claim the whole degraded path rests on: an emergency reading is an emergency with
    # no model on disk. Paired with a normal history so a green result cannot be vacuous.
    hi = G.SAMPLE.rsplit("\n", 1)[0] + "\n2026-06-03, 195, 100"
    b_hi, *_, raw_hi = call(hi)
    b_ok, *_, raw_ok = call()
    cur_hi = (raw_hi.get("rule_engine") or {}).get("current") or {}
    cur_ok = (raw_ok.get("rule_engine") or {}).get("current") or {}
    chk("*** SBP 195 -> emergency banner, no model loaded ***",
        cur_hi.get("is_emergency") is True and "emergency" in b_hi.lower(), json.dumps(cur_hi))
    chk("    and a 158 reading is NOT an emergency (the check can fail)",
        cur_ok.get("is_emergency") is False)
    chk("axis_mode is a real engine value",
        cur_hi.get("axis_mode") in {"STANDARD", "PERSONALIZED", None}, cur_hi.get("axis_mode"))

    # Symptoms attach to the latest reading, not the profile, and a red flag must move the
    # verdict. Without the symptom the same history must not fire the override.
    _b, *_, raw_sym = call(symptoms=["Severe headache ⚑"])
    rule_with = ((raw_sym.get("rule_engine") or {}).get("current") or {}).get("rule_id")
    chk("a red-flag symptom changes the fired rule",
        rule_with != cur_ok.get("rule_id"), f"{rule_with} == {cur_ok.get('rule_id')}")

    b_bad, *_ = call(sex="Male", pregnant=True)
    chk("male + pregnant is rejected, not scored", "Rejected by validation" in b_bad)
    b_parse, *_ = call("garbage")
    chk("an unparseable history reports the line, does not crash",
        "Could not read the history" in b_parse)


def _equivalence():
    """The Space and the API must produce the same advisory for the same patient."""
    A.REGISTRY.refresh(force=True)
    G.REGISTRY.refresh(force=True)
    rows = G.parse_readings(G.SAMPLE)
    rows[-1]["symptoms"] = []
    rows[-1]["position"] = "SITTING"
    body = {"patient_id": "t1", "readings": rows,
            "profile": {"age": 68.0, "is_male": 1, "is_dm": 0, "is_pregnant": 0,
                        "hf_type": "NONE", "conditions": [], "medications": [],
                        "missed_3d": 0, "adherence_7d": 1.0}}
    api = TestClient(A.app).post("/api/predict", json=body).json()
    *_, space = call()

    # Timings are wall-clock and will never match; everything else must.
    drop = {"timings", "budget"}
    a = {k: v for k, v in api.items() if k not in drop}
    s = {k: v for k, v in space.items() if k not in drop}
    same = json.dumps(a, sort_keys=True, default=str) == json.dumps(s, sort_keys=True,
                                                                    default=str)
    if not same:
        diff = [k for k in set(a) | set(s)
                if json.dumps(a.get(k), sort_keys=True, default=str)
                != json.dumps(s.get(k), sort_keys=True, default=str)]
        chk("*** Space and API return the identical advisory ***", False, f"differ: {diff}")
    else:
        chk("*** Space and API return the identical advisory ***", True)

    # Guard against the above passing because both sides are empty.
    chk("    the compared advisory is non-trivial",
        bool(a.get("rule_engine")) and len(a) > 6, sorted(a))


def run():
    print("\n--- parsing ---")
    _parsing()
    print("\n--- rendering ---")
    _rendering()
    print("\n--- clinical behaviour (no model on disk) ---")
    _clinical()
    print("\n--- Space / API equivalence ---")
    _equivalence()
    print("\n" + ("ALL SPACE CONTRACT TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


def test_space_contract():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
