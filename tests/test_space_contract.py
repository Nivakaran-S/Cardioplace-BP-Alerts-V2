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
import re  # noqa: E402
import warnings  # noqa: E402

import pandas as pd  # noqa: E402

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
                     ("# only a comment", "no readings"),
                     ("2026-01-01, 140, 80, w=heavy", "a non-numeric keyed value"),
                     ("2026-01-01, 140, 80, meds=maybe", "a non-boolean meds="),
                     ("2026-01-01, 140, 80, wieght=72", "a misspelled field name")):
        try:
            G.parse_readings(bad)
            chk(f"parser rejects {why}", False, f"accepted {bad!r}")
        except ValueError:
            chk(f"parser rejects {why}", True)

    # The keyed tail. These are the per-session model inputs, so a token that parsed to the
    # wrong field would be worse than one that failed loudly.
    r = G.parse_readings("2026-01-01, 140, 80, 72, w=74.2, idwg=2.1, meds=n, uf=2.4, "
                         "hrs=4, drop=18, sym=dizziness+fatigue")[0]
    for k, v in (("weight", 74.2), ("idwg", 2.1), ("took_all_meds", False),
                 ("uf_total", 2.4), ("session_hours", 4.0), ("sbp_drop", 18.0),
                 ("pulse", 72.0)):
        chk(f"parser reads {k} from the keyed tail", r.get(k) == v, f"{k}={r.get(k)!r}")
    chk("parser splits sym= on +", r.get("symptoms") == ["dizziness", "fatigue"],
        r.get("symptoms"))
    chk("    meds=y is the other boolean",
        G.parse_readings("2026-01-01, 140, 80, meds=y")[0]["took_all_meds"] is True)
    chk("    a line with no keyed tail is unchanged",
        set(G.parse_readings("2026-01-01, 140, 80")[0]) == {"date", "sbp", "dbp"})
    # The SPA and the Space must accept the SAME history, or one paste gives two different
    # forecasts depending on which front end the user opened. Read out of app.js rather than
    # restated, so the two lists cannot be edited apart.
    js = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(encoding="utf-8")
    js_tokens = set(re.findall(r"(\w+):\s*\[", js.split("var TOKENS = {")[1].split("};")[0]))
    chk("*** both front ends accept the same reading-line tokens ***",
        set(G.TOKENS) == js_tokens, f"space={sorted(G.TOKENS)} spa={sorted(js_tokens)}")
    chk("    (control) the SPA token list was actually extracted",
        {"w", "idwg", "uf"} <= js_tokens, sorted(js_tokens))


def _rendering():
    b, tiles, fc, chart, eng, sym, chain, cov, raw = call()
    chk("assess returns the outputs the UI declares",
        all(x is not None for x in (b, tiles, fc, chart, eng, sym, chain, cov, raw)))
    # Width is asserted against the declared count rather than a literal, because every error
    # path pads to it -- one of them used to pad to seven against eight components, which
    # paired the raw-JSON pane with a table and showed nothing wrong.
    chk("every error path returns exactly as many outputs as the UI declares",
        all(len(G._error("x")) == G._N_OUTPUTS for _ in (0,))
        and len(call("garbage")) == G._N_OUTPUTS, len(call("garbage")))
    chk("    the coverage table names the fitted-input count",
        "features" in getattr(cov, "columns", []) or "note" in getattr(cov, "columns", []),
        list(getattr(cov, "columns", [])))
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


def _forecast_node_contract():
    """The Space must read the keys `BPPredictor.predict` actually writes.

    This is a unit test on the renderers, not an integration test, so it runs with no bundle
    on disk -- which matters, because the bug it guards against was invisible exactly when no
    model was loaded: an empty forecast table looked like "no model" rather than like a
    misread key. The node shape below mirrors `estimator.py:736-747`.
    """
    node = {"point": 158.4, "readings_ahead": 1, "steps_ahead": 1, "days_ahead_est": 2.0,
            "lo80": 148.1, "hi80": 168.7, "interval_basis": "quantile GBM, conformal on val"}
    plain = {"point": 161.2, "readings_ahead": 2, "steps_ahead": 2, "days_ahead_est": 4.0}
    d = {"forecast": {"sbp": {"h0": node, "h1": plain}}, "personalisation": {"threshold": 145.0}}

    fc = G.forecast_frame(d)
    chk("forecast table renders a row per horizon", len(fc) == 2, fc.to_string())
    chk("forecast table shows the predicted value, not an em-dash",
        "158.4" in fc.to_string(), fc.to_string())
    chk("forecast table shows the 80% interval where one exists",
        "148.1" in fc.to_string() and "168.7" in fc.to_string(), fc.to_string())
    chk("forecast table says so where no interval was fitted",
        "not fitted" in fc.to_string(), fc.to_string())
    chk("forecast table shows days ahead", "2.0" in fc.to_string(), fc.to_string())

    # No numeric column may be entirely blank -- that is the signature of a misread key.
    for c in ("predicted", "days ahead"):
        col = fc[c].astype(str)
        chk(f"forecast column {c!r} is not all-empty",
            not col.str.strip().isin({"—", "", "None", "nan"}).all(), col.to_list())

    rows = G.parse_readings(G.SAMPLE)
    ch = G.chart_frame(d, rows)
    fser = ch[ch.series == "Forecast SBP"]
    chk("chart plots the forecast series", len(fser) == 2, ch.series.value_counts().to_dict())
    chk("chart uses the forecast point values",
        set(fser.mmHg.round(1)) == {158.4, 161.2}, fser.mmHg.to_list())

    # Prove the checks bite: the OLD key names must produce an empty table.
    stale = {"forecast": {"sbp": {"h0": {"sbp": 158.4, "lo": 148.1, "hi": 168.7,
                                         "days_ahead": 2.0}}}}
    bad = G.forecast_frame(stale)
    chk("    (control) the pre-fix key names render no value",
        "158.4" not in bad.to_string(), bad.to_string())


def _clinical():
    # The claim the whole degraded path rests on: an emergency reading is an emergency with
    # no model on disk. Paired with a normal history so a green result cannot be vacuous.
    # Derived from the sample, not hardcoded: the literal date silently became a DUPLICATE
    # when the sample was extended, the schema rejected the whole request, and this check was
    # then measuring the validation path while still claiming to measure the emergency one.
    last_ts = pd.Timestamp(G.parse_readings(G.SAMPLE)[-1]["date"])
    hi = G.SAMPLE + f"\n{(last_ts + pd.Timedelta(days=2)).date()}, 195, 100"
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
                        "missed_3d": 0, "adherence_7d": 1.0},
            # The enrich flags must match whatever the Space sends, or this compares two
            # different questions and reports the difference as drift. It has now caught that
            # twice: once when the Space started requesting the chained block, and again when
            # the block was made opt-in and the Space stopped. Mirrors the Space's default.
            "enrich": {"symptom_chained": False}}
    api = TestClient(A.app).post("/api/predict", json=body).json()
    *_, space = call()

    # Wall-clock fields will never match; everything else must. `latency_ms` is measured
    # inside predict() and belongs with `timings` rather than with the advisory's content.
    drop = {"timings", "budget", "latency_ms"}
    a = {k: v for k, v in api.items() if k not in drop}
    s_ = {k: v for k, v in space.items() if k not in drop}
    s = s_
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
    chk("    the chained block is OFF by default on both sides (it costs ~5 s)",
        "symptom_chained" not in a and "symptom_chained" not in s_)
    chk("    the compared advisory is non-trivial",
        bool(a.get("rule_engine")) and len(a) > 6, sorted(a))


def run():
    print("\n--- parsing ---")
    _parsing()
    print("\n--- rendering ---")
    _rendering()
    print("\n--- forecast node contract ---")
    _forecast_node_contract()
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
