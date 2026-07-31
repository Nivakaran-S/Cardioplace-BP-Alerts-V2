"""The SPA must be wired to the markup, to the stylesheet, and to the real payload.

The dashboard is three files that reference each other entirely by string: `app.js` reaches
for elements by id, both scripts read colours by custom-property name, and every renderer
indexes the advisory by key. Nothing in that chain is checked by a compiler, and every link in
it has broken silently at least once in this repo -- a renderer reading `f.get("sbp")` when
`predict()` writes `point` rendered a table of em-dashes that looked exactly like "no model
loaded".

So the checks here are the compiler that does not exist:

  1. every id `app.js` asks for is in `index.html`
  2. every custom property the scripts read is defined in `style.css`
  3. every `Charts.*` primitive the app calls is exported
  4. every class the scripts emit is styled
  5. every advisory key the renderers read is present in a live payload
  6. no interface file describes the symptom labels as generated

Check 5 is the one that catches real regressions, and it runs against a live `POST
/api/predict` rather than a fixture, so it fails when the model changes shape.

Every check is written so it can fail; where that is not self-evident the assertion is paired
with a control in the same function.
"""

import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
CHARTS_JS = (ROOT / "static" / "charts.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
HTML = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")

FAILS = []


def chk(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"  <- {extra}"))
    if not cond:
        FAILS.append(name)


# --------------------------------------------------------------------------- 1. ids

def _ids_used(js):
    """`$("x")` is the app's element lookup; `num("x")` reaches for one indirectly, so both
    are extracted -- otherwise the reverse-direction note misreports live form fields as dead
    markup."""
    return (set(re.findall(r'\$\("([a-z0-9-]+)"\)', js))
            | set(re.findall(r'getElementById\("([a-z0-9-]+)"\)', js))
            | set(re.findall(r'\bnum\("([a-z0-9-]+)"\)', js)))


def _ids_declared(html):
    return set(re.findall(r'\bid="([a-zA-Z0-9_-]+)"', html))


def _ids():
    used, have = _ids_used(APP_JS), _ids_declared(HTML)
    missing = sorted(used - have)
    chk("*** every element id app.js reaches for exists in the markup ***",
        not missing, f"absent from index.html: {missing}")

    # Prove the check bites: an id that is not in the template must be caught.
    chk("    (control) a fabricated id is detected as missing",
        "totally-not-an-element" not in have)

    # The reverse direction is a warning, not a failure -- static markup may legitimately
    # carry ids the script never touches.
    unused = sorted(have - used - {"form", "banner-title", "banner-detail", "readings-hint"})
    if unused:
        print(f"  note   ids in the markup that app.js never reads: {unused}")


# ------------------------------------------------------------------ 2. custom properties

def _css_vars():
    read = set(re.findall(r'cssVar\("(--[a-z0-9-]+)"\)', APP_JS + CHARTS_JS))
    read |= set(re.findall(r'var\((--[a-z0-9-]+)\)', APP_JS))
    defined = set(re.findall(r'^\s*(--[a-z0-9-]+)\s*:', CSS, re.M))
    missing = sorted(read - defined)
    chk("*** every custom property the scripts read is defined in style.css ***",
        not missing, f"undefined: {missing} (an undefined var resolves to '' and the mark "
                     f"renders black or invisible)")
    chk("    (control) the property list was actually extracted",
        len(read) >= 6 and "--series-1" in read, sorted(read))

    # Both themes must define every series and status colour, or a mode silently loses a hue.
    for block, label in ((re.search(r':root\[data-theme="dark"\]\s*\{(.*?)\}', CSS, re.S), "dark"),
                         (re.search(r'^:root\s*\{(.*?)\}', CSS, re.S | re.M), "light")):
        body = block.group(1) if block else ""
        have = set(re.findall(r'(--[a-z0-9-]+)\s*:', body))
        need = {"--series-1", "--series-2", "--series-3", "--grid", "--axis", "--ink-muted"}
        chk(f"    the {label} theme defines every colour a chart reads",
            need <= have, f"missing {sorted(need - have)}")


# ----------------------------------------------------------------------- 3. primitives

def _primitives():
    called = set(re.findall(r'\bC\.([a-zA-Z]+)\(', APP_JS))
    exported = set(re.findall(r'(\w+):\s*\w+[,}]', CHARTS_JS.split("global.Charts =")[-1]))
    missing = sorted(called - exported)
    chk("*** every Charts primitive app.js calls is exported ***", not missing,
        f"not on the Charts object: {missing}")
    chk("    (control) the call list was actually extracted",
        {"axes", "line", "domain"} <= called, sorted(called))


# -------------------------------------------------------------------------- 4. classes

def _classes():
    """Classes the scripts CREATE must be styled -- markup classes are visible in review,
    generated ones are not."""
    # The charset stops the match at the quote OR at a `+` concatenation, so a class list
    # built as `class="tile' + (accent ? ...)` still yields "tile". A token left dangling by
    # that cut -- `"banner is-"` -- ends in a hyphen and is a fragment, not a class.
    emitted = set()
    for pat in (r'className\s*=\s*"([a-z0-9 _-]*)', r'class="([a-z0-9 _-]*)'):
        for m in re.findall(pat, APP_JS + CHARTS_JS):
            emitted |= {c for c in m.split() if c and not c.endswith("-")}
    # States appended by ternary rather than written inline; enumerated from the renderers.
    emitted |= {"is-" + k for k in ("critical", "watch", "good", "info")}
    emitted |= {"is-live", "is-degraded", "over", "dash",
                "good", "warning", "critical", "muted", "accent-good", "accent-critical"}
    styled = set(re.findall(r'\.([a-zA-Z][a-zA-Z0-9_-]*)', CSS))
    missing = sorted(c for c in emitted - styled if not c.startswith("sr-"))
    chk("*** every class the scripts emit has a rule in style.css ***", not missing,
        f"unstyled: {missing}")
    chk("    (control) the emitted-class list was actually extracted",
        {"tile", "pill", "banner", "chain-block"} <= emitted, sorted(emitted))
    chk("    (control) an unstyled class would be caught",
        "no-such-class-anywhere" not in styled)


# ------------------------------------------------------------- 5. payload keys, live

def _payload():
    from fastapi.testclient import TestClient

    import app as A
    import gradio_app as G

    A.REGISTRY.refresh(force=True)
    rows = G.parse_readings(G.SAMPLE)
    body = {"patient_id": "ui-contract", "readings": rows,
            "profile": {"age": 68.0, "is_male": 1, "is_dm": 0, "is_pregnant": 0,
                        "hf_type": "NONE", "conditions": [], "medications": [],
                        "missed_3d": 0, "adherence_7d": 1.0, "dryweight": 72.0},
            "enrich": {"symptom_chained": True}}
    res = TestClient(A.app).post("/api/predict", json=body)
    chk("the API answers the dashboard's own request shape", res.status_code == 200,
        res.text[:300])
    if res.status_code != 200:
        return
    d = res.json()

    def at(path):
        cur = d
        for p in path.split("."):
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            elif isinstance(cur, list) and cur and isinstance(cur[0], dict) and p in cur[0]:
                cur = cur[0][p]
            else:
                return False
        return True

    # Half this file's checks need a loaded bundle, and when there is none they ALL fail at
    # once -- which reads like the payload is broken when the real message is one line long.
    # CI hit exactly that: `load_object` on an unfetched LFS pointer raises `KeyError: 118`
    # (118 is `v`, the first byte of "version https://git-lfs.github.com/..."), the registry
    # kept `predictor = None`, and the run reported thirty-eight missing keys.
    #
    # So the model's absence is now ONE named check, and the model-dependent block is skipped
    # rather than allowed to fail forty times. Strictness is the caller's decision:
    # CP_REQUIRE_MODEL=1 (set in CI) turns the skip into a failure, so a silent regression in
    # the LFS fetch cannot leave this suite green.
    import os

    health = A.REGISTRY.health()
    has_model = bool(health.get("model_loaded"))
    require = os.environ.get("CP_REQUIRE_MODEL", "").strip() not in ("", "0", "false")
    if not has_model:
        detail = health.get("detail") or "no detail reported"
        if require:
            chk("*** a bundle is loaded (CP_REQUIRE_MODEL is set) ***", False,
                f"{detail} -- if this says KeyError: 118, the checkout did not fetch the "
                f"LFS object and final_model/model.pkl is still a pointer file")
        else:
            print(f"  SKIP   no bundle loaded ({detail}); skipping every model-dependent "
                  f"payload check. Set CP_REQUIRE_MODEL=1 to make this a failure.")

    # Keys the degraded path emits too -- checked whether or not a bundle is present, because
    # the rule engine is the safety-critical layer and does not need a model.
    ALWAYS = [
        "confidence_tier", "n_observations", "rule_engine.history.ts", "rule_engine.fired_count",
        "governance.emergency_floor_mmHg", "governance.population_threshold_mmHg",
    ]
    missing_always = [k for k in ALWAYS if not at(k)]
    chk("*** every key the renderers read on the degraded path is present ***",
        not missing_always, f"absent: {missing_always}")

    if not has_model:
        chk("    (control) a key the API does not emit is reported missing",
            not at("forecast.sbp.h0.definitely_not_a_field"))
        return

    # Every key a renderer indexes. Absent ones do not crash -- they render an em-dash or hide
    # a section, which is exactly why they need asserting.
    REQUIRED = [
        "model_version",
        "forecast.sbp.h0.point", "forecast.sbp.h0.days_ahead_est",
        "forecast.sbp.h0.coherence.ok", "forecast.dbp.h0.point",
        "personalisation.threshold", "personalisation.offset", "personalisation.cohort_key",
        "early_warning.score", "early_warning.cut", "early_warning.flagged",
        "early_warning.est_lead_days", "early_warning.budget_pct",
        "anomaly.points.ts", "anomaly.points.score", "anomaly.points.flagged",
        "anomaly.cut", "anomaly.n_flagged", "anomaly.n_settled",
        "rule_engine.current.is_emergency", "rule_engine.current.fired",
        "rule_engine.current.rule_id",
        "predicted_alert.horizons.days_ahead", "predicted_alert.horizons.sbp",
        "predicted_alert.horizons.fired", "predicted_alert.horizons.is_emergency",
        "backtest.horizons.mae", "backtest.horizons.within_10", "backtest.horizons.n",
        "backtest.series.h1",
        "history.ts", "history.sbp", "history.dbp",
        "symptom_chained.available", "symptom_chained.items", "symptom_chained.sessions_ahead",
        "symptom_chained.items.prob", "symptom_chained.items.sessions_ahead",
        "symptom_chained.items.mechanism", "symptom_chained.items.jensen_gap",
        "symptom_chained.uncertainty_basis",
        "feature_coverage.fitted", "feature_coverage.resolved", "feature_coverage.pct",
        "feature_coverage.gaps", "feature_coverage.needs_more_sessions",
    ]
    missing = [k for k in REQUIRED if not at(k)]
    chk("*** every advisory key the renderers read is present in a live payload ***",
        not missing, f"absent: {missing}")
    chk("    (control) a key the API does not emit is reported missing",
        not at("forecast.sbp.h0.definitely_not_a_field"))

    # The panels must be non-empty, or the checks above pass vacuously on a stub payload.
    chk("    the trend chart has enough history to draw", len(d.get("history") or []) >= 2)
    chk("    the rule engine returned a per-session history",
        len((d.get("rule_engine") or {}).get("history") or []) >= 2)
    chk("    the backtest returned a scored series",
        len(((d.get("backtest") or {}).get("series") or {}).get("h1") or []) >= 2)
    chk("    the symptom outlook returned items for more than one session",
        len({i["sessions_ahead"] for i in
             ((d.get("symptom_chained") or {}).get("items") or [])}) >= 2)

    # A signal in the bundle that cannot be scored must be declared, not dropped, or the UI
    # cannot tell "no such model" from "model could not run here".
    un = d.get("forecast_unavailable")
    if un:
        chk("    an unscorable signal is declared with a reason",
            bool(un.get("signals")) and bool(un.get("note")), un)

    # The renderer treats `cut_applies` as authoritative for whether to show a flag. If the
    # API ever starts flagging chained rows, the UI must be revisited deliberately.
    items = (d.get("symptom_chained") or {}).get("items") or []
    chk("    no chained row claims an alert flag (the cut is not budget-valid here)",
        all(not i.get("cut_applies") for i in items),
        [i["key"] for i in items if i.get("cut_applies")][:5])
    return d


# ------------------------------------------------------------------- 6. interface text

def _wording():
    files = {"gradio_app.py": (ROOT / "gradio_app.py").read_text(encoding="utf-8"),
             "index.html": HTML, "app.js": APP_JS, "style.css": CSS}
    bad = {f: [ln for ln in t.splitlines() if "synthetic" in ln.lower()]
           for f, t in files.items()}
    offenders = {f: v for f, v in bad.items() if v}
    chk("*** no rendered interface text describes the symptom labels as generated ***",
        not offenders, {f: v[:2] for f, v in offenders.items()})

    # ...but the API payload and the provenance flag must KEEP it. Removing the wording from
    # the interface is a presentation decision; deleting the provenance record would be a
    # governance change, and gate 14 asserts it.
    enrich = (ROOT / "src" / "serving" / "enrich.py").read_text(encoding="utf-8")
    chk("    the API payload still carries the provenance disclosure",
        "SYNTHETIC_WARNING" in enrich and "labels_are_synthetic" in enrich)


def run():
    print("\n--- element ids ---")
    _ids()
    print("\n--- custom properties ---")
    _css_vars()
    print("\n--- chart primitives ---")
    _primitives()
    print("\n--- emitted classes ---")
    _classes()
    print("\n--- live payload keys ---")
    _payload()
    print("\n--- interface wording ---")
    _wording()
    print("\n" + ("ALL UI CONTRACT TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


def test_ui_contract():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
