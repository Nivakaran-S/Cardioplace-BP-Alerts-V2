"""API contract smoke.

Exercises BOTH the no-model and the with-model paths explicitly, by forcing the registry
into each state rather than depending on whatever happens to be on disk. The no-model path
is the one that matters most: it is the primary path on a fresh checkout, and the claim it
has to support is that an emergency reading still produces an emergency without any ML.
"""


import sys
from pathlib import Path

# `python tests/test_x.py` puts tests/ on sys.path, not the repo root, so `import src` fails.
# The README documents running these files directly, so the fix belongs here rather than in a
# PYTHONPATH the reader has to know to set. CI hit exactly this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

import app as A  # noqa: E402
from src.serving import settings as S  # noqa: E402

FAILS = []


def chk(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"  <- {extra}"))
    if not cond:
        FAILS.append(name)


def hist(n=40, last_sbp=None):
    rows, d = [], datetime.date(2026, 4, 6)
    for i in range(n):
        rows.append({"date": d.isoformat(), "sbp": 138 + (i % 7) * 3 - (i % 3),
                     "dbp": 78 + (i % 5)})
        d += datetime.timedelta(days=2 if i % 3 else 3)
    if last_sbp:
        rows[-1]["sbp"] = last_sbp
    return rows


def _static_and_schema(c):
    r = c.get("/")
    chk("GET / serves the shell", r.status_code == 200 and "Cardioplace" in r.text)
    for p in ("/static/app.js", "/static/style.css", "/static/charts.js"):
        chk(f"GET {p}", c.get(p).status_code == 200)

    # Every $("id") the controller reaches for must exist in the shell. A vanilla-JS
    # rebuild breaks silently this way more than any other.
    with open(f"{S.TEMPLATES_DIR}/index.html", encoding="utf-8") as fh:
        html = fh.read()
    with open(f"{S.STATIC_DIR}/app.js", encoding="utf-8") as fh:
        js = fh.read()
    ids = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r'\$\("([^"]+)"\)', js))
    chk("every $(id) in app.js exists in index.html", used <= ids, sorted(used - ids))

    j = c.get("/api/schema").json()
    chk("schema: 19 symptoms / 9 conditions / 11 medications",
        len(j["symptoms"]) == 19 and len(j["conditions"]) == 9 and len(j["medications"]) == 11)
    chk("schema: red flags surfaced", any(x["red_flag"] for x in j["symptoms"]))
    chk("schema: rule inventory present", "total_slots" in (j.get("rules") or {}))
    chk("GET /api/health", c.get("/api/health").status_code == 200)


def _validation(c):
    bad = {"patient_id": "t", "readings": [{"date": "2026-01-01", "sbp": 120, "dbp": 118}]}
    chk("pulse pressure < 10 -> 422", c.post("/api/predict", json=bad).status_code == 422)
    dup = [{"date": "2026-01-01", "sbp": 140, "dbp": 80},
           {"date": "2026-01-01", "sbp": 141, "dbp": 81}]
    chk("duplicate dates -> 422",
        c.post("/api/predict", json={"patient_id": "t", "readings": dup}).status_code == 422)
    for label, prof in (("unknown condition", {"conditions": ["has_zzz"]}),
                        ("unknown medication", {"medications": ["on_zzz"]}),
                        ("male + pregnant", {"is_male": 1, "is_pregnant": 1})):
        body = {"patient_id": "t", "readings": hist(3), "profile": prof}
        chk(f"{label} -> 422", c.post("/api/predict", json=body).status_code == 422)
    chk("unknown top-level field -> 422",
        c.post("/api/predict",
               json={"patient_id": "t", "readings": hist(3),
                     "readings_typo": []}).status_code == 422)


def _json_safety(raw, label):
    chk(f"{label}: no NaN/Infinity tokens", "NaN" not in raw and "Infinity" not in raw)
    try:
        json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        chk(f"{label}: strict-parseable JSON", True)
    except ValueError as e:
        chk(f"{label}: strict-parseable JSON", False, str(e))


def run():
    c = TestClient(A.app)
    A.REGISTRY.refresh(force=True)
    saved = A.REGISTRY.predictor
    real_refresh = A.REGISTRY.refresh

    # ---------------------------------------------------------- no-model path
    print("\n--- NO MODEL LOADED (the primary path on a fresh checkout) ---")
    A.REGISTRY.predictor = None
    A.REGISTRY.error = "forced empty for the contract test"
    A.REGISTRY.refresh = lambda force=False: False      # pin the state for this block
    _static_and_schema(c)
    _validation(c)

    r = c.post("/api/predict", json={"patient_id": "t1", "readings": hist(40),
                                     "profile": {"age": 68, "is_male": 1}})
    chk("POST /api/predict -> 200", r.status_code == 200, r.text[:160])
    d = r.json()
    chk("  rule engine still evaluated", bool((d.get("rule_engine") or {}).get("history")))
    chk("  degraded block present", (d.get("degraded") or {}).get("model_loaded") is False)
    chk("  emergency floor from constants", d.get("emergency_floor_mmHg") == 180.0)
    chk("  no forecast issued", d.get("forecast") == {})
    chk("GET /api/model -> 404", c.get("/api/model").status_code == 404)

    # `current.axis_mode` is read off a pandas Series whose index carries a label named
    # `mode` -- which is also a Series METHOD. Attribute access returns the bound method, and
    # to_jsonable stringifies it rather than raising, so the bug is invisible in a 200 response
    # and only shows up as a consumer that never matches. The SPA's "· personalised" marker was
    # dead for exactly this reason. Assert the value is one the engine actually emits.
    axis = ((d.get("rule_engine") or {}).get("current") or {}).get("axis_mode")
    chk("  current.axis_mode is a real engine value, not a bound method",
        axis in {"STANDARD", "PERSONALIZED", None}, repr(axis))
    chk("  no stringified python object anywhere in the body",
        "<bound method" not in r.content.decode() and "object at 0x" not in r.content.decode())

    r2 = c.post("/api/predict", json={"patient_id": "t2", "readings": hist(40, last_sbp=195),
                                      "profile": {"age": 68, "is_male": 1}})
    cur = (r2.json().get("rule_engine") or {}).get("current") or {}
    chk("*** SBP 195 -> emergency WITH NO MODEL ***",
        r2.status_code == 200 and cur.get("is_emergency") is True, json.dumps(cur))
    _json_safety(r2.content.decode(), "no-model body")

    # ---------------------------------------------------------- with-model path
    A.REGISTRY.refresh = real_refresh
    A.REGISTRY.predictor = saved
    A.REGISTRY.refresh(force=True)
    if not A.REGISTRY.loaded:
        print("\n--- WITH MODEL: skipped, no artifact on disk ---")
        print("    run `python main.py` to exercise this half")
    else:
        print(f"\n--- WITH MODEL LOADED ({A.REGISTRY.source}) ---")
        m = c.get("/api/model")
        chk("GET /api/model -> 200", m.status_code == 200)
        mj = m.json()
        chk("  reports model_version + sklearn runtime",
            bool(mj.get("model_version")) and bool(mj.get("sklearn_runtime")))
        chk("  governance block complete",
            {"emergency_floor_mmHg", "population_threshold_mmHg", "alert_budget_pct"}
            <= set(mj.get("governance", {})))

        r3 = c.post("/api/predict", json={"patient_id": "t3", "readings": hist(60),
                                          "profile": {"age": 68, "is_male": 1}})
        chk("POST /api/predict -> 200", r3.status_code == 200, r3.text[:250])
        d3 = r3.json()
        chk("  forecast issued", bool(d3.get("forecast", {}).get("sbp")))
        chk("  personalised threshold below the emergency floor",
            (d3.get("personalisation") or {}).get("threshold", 999) < 180.0)
        chk("  early warning scored", d3.get("early_warning") is not None)
        chk("  anomaly points returned", bool((d3.get("anomaly") or {}).get("points")))
        chk("  backtest horizons returned", bool((d3.get("backtest") or {}).get("horizons")))
        chk("  predicted_alert horizons returned",
            bool((d3.get("predicted_alert") or {}).get("horizons")))
        chk("  symptom block always carries the synthetic warning",
            bool((d3.get("symptom_risk") or {}).get("warning")))
        t = d3.get("timings") or {}
        budget = (d3.get("budget") or {}).get("latency_budget_ms", 200.0)
        chk("  timings reported", "predict_ms" in t)
        # LATENCY_BUDGET_MS is a production-hardware SLO, and this test runs wherever it
        # runs -- a 2-core shared CI runner, or a laptop with a training job saturating it.
        # Asserting the SLO here would produce a permanently flaky build that says nothing
        # about the code. The pipeline measures it properly on the training machine and
        # writes batch_latency.yaml; this only catches an order-of-magnitude regression.
        pm = t.get("predict_ms", 0)
        within = pm <= budget
        print(f"  {'PASS' if within else 'INFO'}    core predict {pm} ms vs a {budget} ms "
              f"SLO{'' if within else ' (over -- machine-dependent, see batch_latency.yaml)'}")
        chk(f"  core predict not an order of magnitude over ({pm} ms <= {budget * 10} ms)",
            pm <= budget * 10, json.dumps(t))
        _json_safety(r3.content.decode(), "with-model body")

        r4 = c.post("/api/predict", json={"patient_id": "t4",
                                          "readings": hist(60, last_sbp=195),
                                          "profile": {"age": 68, "is_male": 1}})
        cur4 = (r4.json().get("rule_engine") or {}).get("current") or {}
        chk("SBP 195 -> emergency WITH a model", cur4.get("is_emergency") is True,
            json.dumps(cur4))

    print("\n" + ("ALL API CONTRACT TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


def test_api_contract():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
