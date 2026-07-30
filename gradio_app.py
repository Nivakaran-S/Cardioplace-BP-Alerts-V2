"""Hugging Face Space entry point -- the Gradio SDK front end.

Why this file exists alongside `app.py`. The FastAPI app and its vanilla-JS SPA are the
richer interface, but serving them on a Space needs `sdk: docker`, and Docker Spaces are a
PRO feature: a free account gets `402 Payment Required` at push time. The Gradio SDK is free,
so this module is what the Space actually runs.

It is a front end and nothing more. Every number on this page comes from
`src.serving.advisory.build_advisory` -- the same function `POST /api/predict` calls -- so the
Space and the API cannot drift into disagreeing about the same patient. Nothing clinical is
decided here; this file only lays out what that function returned.

`app.py` is unchanged and still runs locally or under Docker:  python app.py

Run locally:  python gradio_app.py
"""

import os
import re

import gradio as gr
import pandas as pd
from pydantic import ValidationError

from src.constants.training_pipeline import EMERGENCY_FLOOR_MMHG
from src.logging.logger import logging
from src.serving.advisory import banner_for, build_advisory
from src.serving.jsonify import to_jsonable
from src.serving.model_registry import ModelRegistry
from src.serving.schemas import PredictRequest
from src.serving.vocabulary import build_vocabulary
from src.utils.ml_utils.rule_engine.registry import build_registry

# ZeroGPU refuses to serve a Space that exposes no GPU entry point: it starts the app,
# health-checks it, then kills it with "No @spaces.GPU function detected during startup".
# This app has no GPU work to offer it -- the forecaster is fitted scikit-learn and the rule
# engine is arithmetic -- so the decorated function below exists purely to satisfy that
# contract, and is never called on any request path.
#
# The import is guarded because `spaces` is injected by the ZeroGPU image and is deliberately
# NOT in requirements.txt: it pulls torch, which would add gigabytes to CI for a package that
# does nothing off-Space. Off-Space this block is skipped entirely.
try:
    import spaces
except ImportError:  # local runs, CI, Docker, `python app.py`
    spaces = None

if spaces is not None:

    @spaces.GPU(duration=1)
    def _zerogpu_entry_point() -> str:
        """Declare a GPU entry point so ZeroGPU lets the Space start. Never invoked."""
        return "ok"


REGISTRY = ModelRegistry()
RULES = build_registry()
VOCAB = build_vocabulary(RULES)
BLOCKED_NOTE = (VOCAB.get("rules") or {}).get("note", "")

CONDITIONS = {c["label"]: c["key"] for c in VOCAB["conditions"]}
MEDICATIONS = {m["label"]: m["key"] for m in VOCAB["medications"]}
# Grouped by mechanism, because a red flag is surfaced individually and never averaged into a
# mechanism score. The group is the clinical reason the symptom matters.
SYM_BY_GROUP: dict[str, list[dict]] = {}
for s in VOCAB["symptoms"]:
    SYM_BY_GROUP.setdefault(s["group"], []).append(s)
SYM_LABEL_TO_KEY = {f"{s['label']}{' ⚑' if s['red_flag'] else ''}": s["key"]
                    for s in VOCAB["symptoms"]}

SEV_COLOUR = {"critical": ("#b3261e", "#fdeceb"), "watch": ("#8a5300", "#fff6e5"),
              "info": ("#1b5e8a", "#e9f3fa"), "good": ("#1c6b3f", "#e8f5ed")}

# Bounds for the optional provider target. The ceiling is derived from the emergency floor
# rather than written as 179, so the two cannot drift: a provider target at or above the
# floor would ask the app to treat an emergency reading as this patient's normal, and that
# floor is the one value the governance contract says is never personalised.
_TARGET_LO = 100
_TARGET_HI = int(EMERGENCY_FLOOR_MMHG) - 1

SAMPLE = """2026-05-01, 136, 78
2026-05-04, 139, 80
2026-05-06, 141, 79
2026-05-08, 138, 77
2026-05-11, 144, 82
2026-05-13, 142, 80
2026-05-15, 147, 83
2026-05-18, 145, 81
2026-05-20, 150, 84
2026-05-22, 148, 82
2026-05-25, 152, 85
2026-05-27, 149, 83
2026-05-29, 155, 86
2026-06-01, 153, 84
2026-06-03, 158, 87"""


# --------------------------------------------------------------------------- parsing

def parse_readings(text: str) -> list[dict]:
    """`date, sbp, dbp [, pulse]` per line. Blank lines and `#` comments ignored.

    Raises ValueError with the offending line number. A silent skip would let a typo drop a
    reading out of the history without anyone noticing it went missing.
    """
    rows = []
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in re.split(r"[,\t;]+", line) if p.strip()]
        if len(parts) < 3:
            raise ValueError(f"line {i}: expected `date, sbp, dbp`, got {raw!r}")
        try:
            row = {"date": str(pd.Timestamp(parts[0]).date()),
                   "sbp": int(round(float(parts[1]))), "dbp": int(round(float(parts[2])))}
        except Exception as exc:                                      # noqa: BLE001
            raise ValueError(f"line {i}: {exc}") from exc
        if len(parts) > 3:
            row["pulse"] = float(parts[3])
        rows.append(row)
    if not rows:
        raise ValueError("no readings entered")
    return rows


def _num(x, nd=1):
    return "—" if x is None else (f"{float(x):.{nd}f}" if isinstance(x, (int, float)) else x)


def _pretty(s):
    return str(s or "—").replace("RULE_", "").replace("_", " ").capitalize()


# --------------------------------------------------------------------------- rendering

def banner_html(d: dict) -> str:
    sev, title, detail = banner_for(d)
    fg, bg = SEV_COLOUR[sev]
    return (f'<div style="border-left:6px solid {fg};background:{bg};color:#111;'
            f'padding:14px 16px;border-radius:8px">'
            f'<div style="font-weight:700;color:{fg};font-size:1.05rem">{title}</div>'
            f'<div style="margin-top:4px;font-size:.92rem">{detail}</div></div>')


def tiles_md(d: dict) -> str:
    pers = d.get("personalisation") or {}
    ew = d.get("early_warning") or {}
    eng = d.get("rule_engine") or {}
    gov = d.get("governance") or {}
    stale = d.get("staleness") or {}
    rows = [
        ("Confidence tier", d.get("confidence_tier") or "—"),
        ("Personalised threshold",
         f"{_num(pers.get('threshold'))} mmHg" if pers.get("threshold") is not None
         else "not issued"),
        ("Offset vs population",
         f"{pers.get('offset'):+.1f} mmHg" if pers.get("offset") is not None else "—"),
        ("Early-warning score",
         f"{_num(ew.get('score'), 3)} (cut {_num(ew.get('cut'), 3)})"
         if ew.get("score") is not None else "not scored"),
        ("Rules fired", f"{eng.get('fired_count', 0)} of {len(eng.get('history') or [])} "
                        f"readings"),
        ("Observations", d.get("n_observations")),
        ("Days since last reading", _num(stale.get("days_since_last_reading"))),
        ("Emergency floor (never personalised)", f"{_num(gov.get('emergency_floor_mmHg'))} mmHg"),
        ("Population threshold", f"{_num(gov.get('population_threshold_mmHg'))} mmHg"),
        ("Model version", d.get("model_version") or "none loaded"),
    ]
    out = "| | |\n|---|---|\n"
    return out + "\n".join(f"| **{k}** | {v} |" for k, v in rows)


def forecast_frame(d: dict) -> pd.DataFrame:
    """One row per (signal, horizon), with the 80% conformal interval where one exists.

    The key names are a contract with `BPPredictor.predict`, which writes `point`,
    `readings_ahead`, `steps_ahead`, `days_ahead_est` and -- on the single node that has an
    interval -- `lo80`, `hi80`, `interval_basis`. Guessing `sbp`/`lo`/`days_ahead` here
    produced a table of em-dashes that looked like "no forecast" rather than like a bug, which
    is why `_forecast_cols` is asserted in tests/test_space_contract.py.

    Only one node carries a band: `fit_quantile_interval` is fitted for sbp at one horizon
    only, so a blank interval on the other rows is the true state of the bundle, not a
    rendering failure. The basis column says which.
    """
    rows = []
    for sig, per_h in (d.get("forecast") or {}).items():
        for key, f in (per_h or {}).items():
            if not isinstance(f, dict):
                continue
            lo, hi = f.get("lo80"), f.get("hi80")
            rows.append({
                "signal": sig.upper(), "horizon": key,
                "readings ahead": f.get("readings_ahead"),
                "days ahead": _num(f.get("days_ahead_est")),
                "predicted": _num(f.get("point")),
                "80% interval": (f"{_num(lo)} – {_num(hi)}" if lo is not None and hi is not None
                                 else "not fitted at this horizon"),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        {"note": ["no forecast issued — see the banner for why"]})


def chart_frame(d: dict, readings: list[dict]) -> pd.DataFrame:
    """Observed SBP plus the forecast points and the two reference lines, tidy-format."""
    rows = [{"date": pd.Timestamp(r["date"]), "mmHg": r["sbp"], "series": "Observed SBP"}
            for r in readings]
    last = max((r["date"] for r in readings), default=None)
    if last is not None:
        base = pd.Timestamp(last)
        # `point` and `days_ahead_est`, not `sbp`/`days_ahead` -- see forecast_frame.
        for f in ((d.get("forecast") or {}).get("sbp") or {}).values():
            if isinstance(f, dict) and f.get("point") is not None:
                rows.append({
                    "date": base + pd.Timedelta(days=float(f.get("days_ahead_est") or 0)),
                    "mmHg": float(f["point"]), "series": "Forecast SBP"})
        pers = (d.get("personalisation") or {}).get("threshold")
        span = [pd.Timestamp(min(r["date"] for r in readings)), base + pd.Timedelta(days=4)]
        for t in span:
            rows.append({"date": t, "mmHg": 180.0, "series": "Emergency floor (180)"})
            if pers is not None:
                rows.append({"date": t, "mmHg": float(pers),
                             "series": "Personalised threshold"})
    return pd.DataFrame(rows)


def engine_frame(d: dict, readings: list[dict]) -> pd.DataFrame:
    """Per-reading verdicts.

    The engine block returns `ts/fired/tier/rule_id` and no blood pressure, so the SBP/DBP
    shown here are joined back from what was submitted, keyed on the date rather than on
    position -- the enrichment tail can be shorter than the history.
    """
    hist = (d.get("rule_engine") or {}).get("history") or []
    if not hist:
        return pd.DataFrame({"note": ["rule engine not evaluated"]})
    by_date = {r["date"]: r for r in readings}
    rows = []
    for h in hist:
        src = by_date.get(str(h.get("ts")), {})
        rows.append({"date": h.get("ts"), "SBP": src.get("sbp"), "DBP": src.get("dbp"),
                     "fired": "yes" if h.get("fired") else "",
                     "tier": _pretty(h.get("tier")) if h.get("tier") else "",
                     "rule": _pretty(h.get("rule_id")) if h.get("rule_id") else ""})
    return pd.DataFrame(rows)


def chained_frame(d: dict) -> pd.DataFrame:
    """Symptom risk conditioned on the FORECAST -- the joint answer, one row per horizon.

    `prob` is integrated over the forecast's own uncertainty where a conformal band exists.
    `point` is what a plug-in cascade would have reported, and the gap between them is shown
    because near the 140 mmHg threshold it is the difference between "no elevated risk" and a
    real one -- see symptom_chain.py.
    """
    blk = d.get("symptom_chained") or {}
    items = blk.get("items") or []
    if not items:
        return pd.DataFrame({"note": [blk.get("reason")
                                      or "not requested; set enrich.symptom_chained"]})
    return pd.DataFrame([{
        "symptom": _pretty(i.get("key")),
        "at session": i.get("sessions_ahead"),
        "conditioned through": f"session {i.get('conditioned_through_session')} "
                               f"(SBP {_num(i.get('conditioned_through_sbp'))})",
        "risk": f"{100 * float(i['prob']):.1f}%",
        "uncertainty correction": f"{100 * float(i['jensen_gap']):+.1f} pp",
        "mechanism": i.get("mechanism") or "—",
        "red flag": "⚑" if i.get("red_flag") else "",
    } for i in items])


def symptom_frame(d: dict) -> pd.DataFrame:
    blk = d.get("symptom_risk") or {}
    items = blk.get("items") or []
    if not items:
        return pd.DataFrame({"note": [blk.get("note")
                                      or "no symptom heads in this bundle"]})
    return pd.DataFrame([{"symptom": s.get("label") or _pretty(s.get("key")),
                          "horizon": s.get("horizon"),
                          "mechanism": s.get("mechanism") or "—",
                          "red flag": "⚑" if s.get("red_flag") else "",
                          "probability": _num(s.get("prob"), 3),
                          "cut": _num(s.get("cut"), 3),
                          "flagged": "yes" if s.get("flagged") else ""}
                         for s in items])


# --------------------------------------------------------------------------- callback

def assess(readings_text, patient_id, age, sex, diabetic, pregnant, hf_type, provider_target,
           conditions, medications, symptoms, position, missed_3d, adherence_7d):
    empty = pd.DataFrame()
    try:
        rows = parse_readings(readings_text)
    except ValueError as exc:
        return (f'<div style="color:#b3261e;font-weight:600">Could not read the history — '
                f'{exc}</div>', "", empty, empty, empty, empty, empty, {})

    # Symptoms and position are per-READING, not per-profile: the schema models what the
    # patient felt at a given measurement. The form asks "right now", so they attach to the
    # most recent reading only. Back-filling them across the history would invent a symptom
    # record that was never reported.
    sym_keys = [SYM_LABEL_TO_KEY[s] for s in (symptoms or []) if s in SYM_LABEL_TO_KEY]
    rows[-1]["symptoms"] = sym_keys
    rows[-1]["position"] = position or "SITTING"

    cond_keys = [CONDITIONS[c] for c in (conditions or []) if c in CONDITIONS]
    profile = {
        "age": float(age) if age else 60.0,
        "is_male": 1 if sex == "Male" else 0,
        # Diabetes is a Profile field, not a rule-axis condition: it is in the corpus as a
        # cohort key, and CONDITION_KEYS carries no `has_dm`. It gets its own control.
        "is_dm": 1 if diabetic else 0,
        "is_pregnant": 1 if pregnant else 0,
        "hf_type": hf_type or "NONE",
        "conditions": cond_keys,
        "medications": [MEDICATIONS[m] for m in (medications or []) if m in MEDICATIONS],
        "missed_3d": int(missed_3d or 0),
        "adherence_7d": float(adherence_7d) / 100.0 if adherence_7d is not None else 1.0,
    }
    # Falsy covers both ways this optional field arrives empty: None from the initial render,
    # and 0 from a box the user cleared. Neither is a target of 0 mmHg.
    if provider_target:
        if not _TARGET_LO <= float(provider_target) <= _TARGET_HI:
            return (f'<div style="color:#b3261e;font-weight:600">Provider target SBP must be '
                    f'between {_TARGET_LO} and {_TARGET_HI} mmHg, or left blank — '
                    f'got {_num(provider_target, 0)}. The {_num(EMERGENCY_FLOOR_MMHG, 0)} '
                    f'emergency floor is never personalised.</div>',
                    "", empty, empty, empty, empty, {})
        profile["provider_target"] = float(provider_target)

    try:
        req = PredictRequest(patient_id=patient_id or "space-user", readings=rows,
                             profile=profile,
                             enrich={"symptom_chained": True})
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first.get("loc", []))
        return (f'<div style="color:#b3261e;font-weight:600">Rejected by validation — '
                f'{loc}: {first.get("msg")}</div>', "", empty, empty, empty, empty, empty,
                {})

    REGISTRY.refresh()
    try:
        d = build_advisory(req, REGISTRY.predictor, RULES, BLOCKED_NOTE,
                           degraded_reason=REGISTRY.health()["detail"])
    except Exception as exc:                                          # noqa: BLE001
        logging.exception("space assessment failed")
        return (f'<div style="color:#b3261e;font-weight:600">Scoring failed — '
                f'{type(exc).__name__}</div>', "", empty, empty, empty, empty, empty, {})

    return (banner_html(d), tiles_md(d), forecast_frame(d), chart_frame(d, rows),
            engine_frame(d, rows), symptom_frame(d), chained_frame(d),
            to_jsonable(d))


# --------------------------------------------------------------------------- layout

DISCLAIMER = """
> **Not a medical device.** This is a research prototype for provider-visible decision
> support. It does not diagnose, and it must not be used to make a treatment decision.
> Trained on the HEMOBP haemodialysis corpus, which is a specific population.
"""

CHAIN_NOTE = """
**The symptom heads scored on the *forecast* trajectory rather than on today's readings.**

Read the two columns together: *at session 3, conditioned through session 2 (SBP 158)*. Three
things about that are worth knowing, and none of them are obvious from the number alone.

**Sessions start at 2.** Scoring the very next session needs no forecast at all — that is the
*Symptom risk* tab. The chain only has something to add once at least one predicted reading is
in the history.

**The head never sees the blood pressure of the session it scores.** Features at a session read
only up to the one before, and the heads were trained that way — while the label generator
drives the symptom from the *contemporaneous* pressure. So this conditions on the forecast
trajectory up to the preceding session and helps through autocorrelation. That is a smaller
claim than "conditioned on your predicted blood pressure", and it is the accurate one.

**The uncertainty correction** is what integrating over the forecast's own spread adds. It
matters most just below the 140 mmHg threshold, where a plug-in point estimate reports
**exactly zero** excess risk. No alert flag is raised: the operating cut was chosen on observed
rows and does not carry its budget meaning here.

Only the systolic and weight-gain drivers are forecast, so the four hypotensive-mechanism
symptoms — dizziness, syncope, palpitations, fatigue — gain nothing from this view, and
diastolic pressure does not enter the symptom model at all.
"""

SYNTH_WARNING = """
> ⚠️ **Symptom labels are synthetic.** The HEMOBP corpus carries no symptom columns, so the
> symptom heads were trained on labels generated by a structural causal model. Every number
> in this section describes that generator, **not** any observed patient outcome. It is shown
> because hiding it would be worse, not because it is evidence.
"""


def build_demo():
    with gr.Blocks(title="Cardioplace BP Alerts", theme=gr.themes.Soft(),
                   analytics_enabled=False) as demo:
        gr.Markdown("# 🫀 Cardioplace BP Alerts\n"
                    "Blood-pressure forecasting, a personalised alert threshold, an "
                    "early-warning detector and a deterministic 56-rule clinical engine.")
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            # -------------------------------------------------------------- inputs
            with gr.Column(scale=2):
                gr.Markdown("### Reading history")
                readings = gr.Textbox(
                    value=SAMPLE, lines=12, max_lines=24, label="",
                    info="One reading per line: date, systolic, diastolic [, pulse]")
                with gr.Row():
                    patient_id = gr.Textbox("space-user", label="Patient ID", scale=2)
                    position = gr.Dropdown(VOCAB["positions"], value="SITTING",
                                           label="Position", scale=1)

                with gr.Accordion("Patient profile", open=True):
                    with gr.Row():
                        # Bounds omitted for the same reason as provider_target below: a
                        # cleared box submits 0 and Gradio would reject it before `assess`
                        # runs. 0 falls back to the default there, and a real out-of-range
                        # age is caught by PredictRequest and shown in the results panel.
                        age = gr.Number(68, label="Age", info="18–110")
                        sex = gr.Radio(["Female", "Male"], value="Male", label="Sex")
                    with gr.Row():
                        diabetic = gr.Checkbox(False, label="Diabetic")
                        pregnant = gr.Checkbox(False, label="Pregnant")
                    with gr.Row():
                        hf_type = gr.Dropdown(VOCAB["hf_types"], value="NONE",
                                              label="Heart-failure type")
                    # No minimum/maximum here, deliberately. This field is optional, and a
                    # blank gr.Number submits 0 rather than None -- so component-level bounds
                    # made an untouched form fail with "Value 0 is less than minimum value
                    # 100." from inside Gradio's preprocess, before `assess` ever ran. The
                    # range is enforced in `assess` instead, where 0 can be read as "absent"
                    # and a genuine out-of-range entry can be reported in the results panel
                    # like every other input problem.
                    provider_target = gr.Number(
                        None, label="Provider target SBP (optional)",
                        info=f"Between {_TARGET_LO} and {_TARGET_HI}. Overrides the population "
                             "threshold when no model is loaded. The 180 emergency floor is "
                             "never personalised.")

                with gr.Accordion("Conditions and medications", open=False):
                    conditions = gr.CheckboxGroup(list(CONDITIONS), label="Conditions")
                    medications = gr.CheckboxGroup(list(MEDICATIONS), label="Medications")
                    with gr.Row():
                        missed_3d = gr.Slider(0, 3, value=0, step=1,
                                              label="Doses missed in the last 3 days")
                        adherence_7d = gr.Slider(0, 100, value=100, step=5,
                                                 label="7-day adherence (%)")
                    gr.Markdown(
                        "*Ticking a condition can change the answer a lot. `AXIS_RULES[\"L1\"]`"
                        " order is the clinical semantics: with CAD ticked, `RULE_CAD_HIGH` "
                        "fires at SBP ≥ 130, ahead of the personalised rule — so nearly every "
                        "reading fires. That ordering is deliberate and is not a bug.*")

                with gr.Accordion("Symptoms right now (⚑ = red flag)", open=False):
                    sym_boxes = []
                    for group, items in SYM_BY_GROUP.items():
                        sym_boxes.append(gr.CheckboxGroup(
                            [f"{s['label']}{' ⚑' if s['red_flag'] else ''}" for s in items],
                            label=group))

                go = gr.Button("Assess", variant="primary", size="lg")

            # ------------------------------------------------------------- outputs
            with gr.Column(scale=3):
                banner = gr.HTML()
                with gr.Tab("Summary"):
                    tiles = gr.Markdown()
                with gr.Tab("Forecast"):
                    chart = gr.LinePlot(x="date", y="mmHg", color="series", height=320,
                                        x_title="", y_title="mmHg")
                    fc_tbl = gr.Dataframe(label="Forecast with the 80% conformal interval",
                                          wrap=True)
                with gr.Tab("Rule engine"):
                    gr.Markdown(f"*{BLOCKED_NOTE}*" if BLOCKED_NOTE else "")
                    eng_tbl = gr.Dataframe(wrap=True)
                with gr.Tab("Symptom risk"):
                    gr.Markdown(SYNTH_WARNING)
                    gr.Markdown("**Today** — the heads scored on the observed history.")
                    sym_tbl = gr.Dataframe(wrap=True)
                with gr.Tab("Predicted symptoms"):
                    gr.Markdown(SYNTH_WARNING)
                    gr.Markdown(CHAIN_NOTE)
                    chain_tbl = gr.Dataframe(wrap=True)
                with gr.Tab("Raw response"):
                    gr.Markdown("Identical to the `POST /api/predict` body — same function.")
                    raw = gr.JSON()

        # Several CheckboxGroups, one flat symptom list: merge at call time rather than
        # threading a nested structure through the callback signature.
        def _assess(rt, pid, ag, sx, dm, pg, hf, pt, cond, med, pos, m3, a7, *sym_groups):
            flat = [s for grp in sym_groups for s in (grp or [])]
            return assess(rt, pid, ag, sx, dm, pg, hf, pt, cond, med, flat, pos, m3, a7)

        go.click(_assess,
                 inputs=[readings, patient_id, age, sex, diabetic, pregnant, hf_type,
                         provider_target, conditions, medications, position, missed_3d,
                         adherence_7d, *sym_boxes],
                 outputs=[banner, tiles, fc_tbl, chart, eng_tbl, sym_tbl, chain_tbl, raw])
        demo.load(_assess,
                  inputs=[readings, patient_id, age, sex, diabetic, pregnant, hf_type,
                          provider_target, conditions, medications, position, missed_3d,
                          adherence_7d, *sym_boxes],
                  outputs=[banner, tiles, fc_tbl, chart, eng_tbl, sym_tbl, chain_tbl, raw])
    return demo


if __name__ == "__main__":
    REGISTRY.refresh(force=True)
    h = REGISTRY.health()
    logging.info("Space ready | model_loaded=%s | %s", h["model_loaded"],
                 h.get("path") or h.get("detail"))
    build_demo().queue(max_size=16).launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
