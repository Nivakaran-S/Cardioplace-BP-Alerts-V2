"""Post-run verification: did this training run produce a model worth trusting?

`python main.py` exiting 0 means no exception escaped. It does not mean the run is good --
a pipeline can complete cleanly and still ship a forecaster that lost to persistence, an
offset whose 90th-percentile threshold covers 60% of readings, or a detector that is the
rule-engine reference it was supposed to beat. This reads the artifacts and says which.

Usage:  python tests/verify_run.py [artifact_dir]

Three outcomes, and the distinction is the point:

  PASS   the evidence exists and says what it should
  FAIL   the evidence exists and says something is wrong
  WARN   the evidence is ABSENT

A missing report is never a PASS. That rule is the whole reason this file can be trusted:
the easiest way to make a verification suite green is to stop producing the thing it checks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.constants.training_pipeline import (  # noqa: E402
    ALERT_BUDGET_PCT,
    EMERGENCY_FLOOR_MMHG,
    OFFSET_CAP_LOOSEN,
    OFFSET_CAP_TIGHTEN,
    POPULATION_THRESHOLD_MMHG,
)

R = {"PASS": 0, "FAIL": 0, "WARN": 0}
LINES = []


def rec(status, name, detail=""):
    R[status] += 1
    LINES.append((status, name, detail))
    mark = {"PASS": "  PASS  ", "FAIL": "  FAIL  ", "WARN": "  WARN  "}[status]
    print(f"{mark}{name}" + (f"   <- {detail}" if detail else ""))


def chk(name, cond, detail=""):
    rec("PASS" if cond else "FAIL", name, "" if cond else detail)


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def load(path, what):
    """Read a report, or record its absence as WARN and return None."""
    if not path or not os.path.exists(path):
        rec("WARN", f"{what}: report absent", os.path.basename(str(path)))
        return None
    try:
        if str(path).endswith((".yaml", ".yml")):
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        if str(path).endswith(".json"):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        d = pd.read_csv(path)
        if d.empty:
            rec("WARN", f"{what}: report is empty", str(path))
            return None
        return d
    except Exception as exc:                                          # noqa: BLE001
        rec("FAIL", f"{what}: report unreadable", f"{type(exc).__name__}: {exc}")
        return None


def newest_artifact_dir():
    dirs = [d for d in glob.glob("Artifacts/*") if os.path.isdir(d)]
    return max(dirs, key=os.path.getmtime) if dirs else None


def col(d, *names):
    """First matching column, so a renamed report degrades to WARN not a KeyError."""
    for n in names:
        if d is not None and n in d.columns:
            return n
    return None


# --------------------------------------------------------------------------- stages

def check_ingestion(art):
    section("1. DATA INGESTION -- session reconciliation")
    rp = os.path.join(art, "data_ingestion", "ingested", "session_decomposition.csv")
    d = load(rp, "session decomposition")
    if d is None:
        return
    print(d.to_string(index=False)[:1400])
    c = col(d, "count", "n", "rows", "sessions")
    if c:
        chk("decomposition attributes a non-zero delta", d[c].abs().sum() > 0)


def check_validation(art):
    section("2. DATA VALIDATION -- contract and drift")
    base = os.path.join(art, "data_validation")
    for fn, what in (("report/contract_report.csv", "contract"),
                     ("drift_report/report.yaml", "drift")):
        p = os.path.join(base, *fn.split("/"))
        d = load(p, what)
        if d is None or not isinstance(d, pd.DataFrame):
            continue
        st = col(d, "status", "passed", "ok")
        if st:
            bad = d[d[st].astype(str).str.upper().isin(["FAIL", "FALSE"])]
            chk(f"{what}: no failing checks", bad.empty,
                bad.to_string(index=False)[:400])
            print(f"         {len(d)} checks, {len(d) - len(bad)} passing")


def check_transformation(art):
    section("3. DATA TRANSFORMATION -- leakage, cadence, feature selection")
    base = os.path.join(art, "data_transformation")
    aud = load(os.path.join(base, "report", "leakage_audit.csv"), "leakage audit")
    if aud is not None:
        st = col(aud, "status")
        if st:
            bad = aud[aud[st].astype(str).str.upper() != "PASS"]
            chk("*** every leakage probe PASSES ***", bad.empty,
                bad.to_string(index=False)[:600])
            print(f"         {len(aud)} probes")
            # A suite of zero probes would pass vacuously.
            chk("    the leakage suite is non-trivial", len(aud) >= 10, f"{len(aud)} probes")

    sel = load(os.path.join(base, "report", "feature_selection.csv"), "feature selection")
    fl = load(os.path.join(base, "transformed", "feature_names.yaml"), "feature list")
    feats = None
    if isinstance(fl, dict):
        feats = fl.get("features") or fl.get("selected") or []
    if feats:
        chk("at least 20 features survive selection", len(feats) >= 20, f"{len(feats)}")
        chk("sbp_ewm0.3 survived (gate 6 blocks promotion without it)",
            "sbp_ewm0.3" in feats)
        must = {"sbp_lag1", "sbp_z", "sbp_ewm0.1", "sbp_ewm0.3", "sbp_slope7", "sbp_mean7",
                "sbp_base_mean", "sbp_base_std", "days_since_last", "is_weekend"}
        missing = must - set(feats)
        chk("MUST_KEEP is a subset of the selected features", not missing, sorted(missing))
        print(f"         {len(feats)} features selected")
    if sel is not None:
        print(f"         selection report: {len(sel)} rows")


def check_model1(art, rep):
    section("4. MODEL 1 -- forecaster bake-off, cold start, ship decision")
    board = load(os.path.join(rep, "forecast_scorecard.csv"), "forecast scorecard")
    if board is not None:
        sp = col(board, "split", "arm")
        chk("all three evaluation arms scored",
            sp is not None and {"val", "test", "holdout_pt"} <= set(board[sp].unique()),
            sorted(board[sp].unique()) if sp else "no split column")

        mae = col(board, "MAE", "mae")
        fam = col(board, "family")
        if mae and fam and sp:
            te = board[board[sp] == "test"]
            learned = te[te[fam] == "learned"]
            basel = te[te[fam] == "baseline"]
            if not learned.empty and not basel.empty:
                bl, bb = learned[mae].min(), basel[mae].min()
                chk("*** the best learned model beats the best baseline on test ***",
                    bl < bb, f"learned {bl:.3f} vs baseline {bb:.3f} mmHg")
                print(f"         best learned {bl:.3f} | best baseline {bb:.3f} mmHg")
                # An MAE of 0 would mean the target leaked into the features.
                chk("    and its MAE is not implausibly small (leak check)",
                    bl > 1.0, f"{bl:.4f} mmHg")

            # Cold start: the unseen-patient arm is the only measure of what a new user gets.
            ho = board[(board[sp] == "holdout_pt") & (board[fam] == "learned")]
            if not ho.empty and not learned.empty:
                pen = ho[mae].min() - learned[mae].min()
                chk("cold-start penalty is finite", np.isfinite(pen), str(pen))
                print(f"         cold-start penalty {pen:+.3f} mmHg "
                      f"(holdout_pt {ho[mae].min():.3f} vs test {learned[mae].min():.3f})")

        nt = col(board, "note")
        if nt:
            failed = board[board[nt].notna()]
            if not failed.empty:
                mc = col(failed, "model")
                rec("WARN", "some architectures failed and were recorded",
                    ", ".join(sorted(failed[mc].unique())) if mc else str(len(failed)))

    dec = load(os.path.join(rep, "ship_decision.csv"), "ship decision")
    if dec is not None:
        print("\n" + dec.to_string(index=False)[:900])
        lo = col(dec, "lo", "ci_lo", "lower")
        v = col(dec, "verdict", "decision")
        if lo:
            chk("ship decision rests on a paired lower bound > 0",
                (dec[lo].astype(float) > 0).any(),
                f"max lower bound {dec[lo].astype(float).max():.4f}")
        if v:
            print(f"         verdict(s): {sorted(dec[v].astype(str).unique())}")

    tie = load(os.path.join(rep, "paired_comparison.csv"), "paired comparison")
    if tie is not None:
        print(f"\n         tie-set comparison: {len(tie)} candidates")
        print(tie.to_string(index=False)[:800])


def check_tuning(rep):
    section("5. TUNING -- Optuna must not make things worse")
    cmp_ = load(os.path.join(rep, "tuning_default_vs_tuned.csv"), "tuning comparison")
    if cmp_ is not None:
        print(cmp_.to_string(index=False)[:1000])
        dcol = col(cmp_, "cv_mae_default", "mae_default", "default")
        tcol = col(cmp_, "cv_mae_tuned", "mae_tuned", "tuned")
        if dcol and tcol:
            worse = cmp_[cmp_[tcol].astype(float) > cmp_[dcol].astype(float) + 1e-9]
            chk("*** tuning never selected a config worse than the default ***",
                worse.empty, worse.to_string(index=False)[:400])
    log = load(os.path.join(rep, "tuning_log.csv"), "tuning log")
    if log is not None:
        print(f"\n         {len(log)} tuning rows")
        n = col(log, "n_configs")
        s = col(log, "secs", "seconds")
        if n and s:
            print(f"         {log[n].sum():.0f} configurations in {log[s].sum():.0f}s")


def check_model2(rep):
    section("6. MODEL 2 -- learned personalised offset")
    b = load(os.path.join(rep, "offset_learned_board.csv"), "learned offset board")
    if b is None:
        return
    show = [c for c in b.columns if any(k in c.lower() for k in
            ("model", "cand", "pinball", "coverage", "mae", "legal"))][:8]
    print(b[show].to_string(index=False)[:1600] if show else b.to_string(index=False)[:1600])

    cov = col(b, "holdout_coverage", "coverage", "select_coverage")
    if cov:
        c = pd.to_numeric(b[cov], errors="coerce").dropna()
        near = c[(c > 0.80) & (c < 0.97)]
        chk("*** at least one candidate's q90 threshold really covers ~90% ***",
            not near.empty, f"coverages ranged {c.min():.3f}-{c.max():.3f}")
    pin = col(b, "select_pinball", "pinball", "holdout_pinball")
    if pin:
        p = pd.to_numeric(b[pin], errors="coerce").dropna()
        chk("pinball loss is reported and finite", not p.empty and np.isfinite(p).all())

    # Governance: the cap is asymmetric on purpose and the floor is absolute.
    mx = col(b, "max_pred", "max_threshold", "pred_max")
    if mx:
        m = pd.to_numeric(b[mx], errors="coerce").max()
        chk(f"*** no learned threshold reaches the {EMERGENCY_FLOOR_MMHG:.0f} mmHg floor ***",
            m < EMERGENCY_FLOOR_MMHG, f"max {m}")
    print(f"\n         caps: -{OFFSET_CAP_TIGHTEN} / +{OFFSET_CAP_LOOSEN} mmHg around "
          f"{POPULATION_THRESHOLD_MMHG:.0f}")


def check_model3(rep):
    section("7. MODEL 3 -- early-warning detector")
    sc = load(os.path.join(rep, "detector_scorecard.csv"), "detector scorecard")
    if sc is None:
        return
    d = col(sc, "detector", "model")
    auc = col(sc, "auc_pr", "AUC_PR", "ap")
    if d and auc:
        s = sc.sort_values(auc, ascending=False)
        print(s[[c for c in (d, auc, "precision", "recall", "alert_rate")
                 if c in s.columns]].head(10).to_string(index=False))
        top = s.iloc[0][d]
        chk("*** the top-ranked detector is not the rule-engine reference ***",
            top != "d_fixed_threshold", f"top is {top}")
        fixed = sc[sc[d] == "d_fixed_threshold"]
        if not fixed.empty:
            chk("    and it beats that reference on AUC-PR",
                float(s.iloc[0][auc]) > float(fixed.iloc[0][auc]),
                f"{s.iloc[0][auc]:.4f} vs {fixed.iloc[0][auc]:.4f}")
    ar = col(sc, "alert_rate")
    if ar:
        a = pd.to_numeric(sc[ar], errors="coerce").dropna()
        near = a[(a <= ALERT_BUDGET_PCT / 100.0 * 1.6)]
        chk(f"a detector holds the {ALERT_BUDGET_PCT}% staffing budget",
            not near.empty, f"rates {a.min():.4f}-{a.max():.4f}")


def check_model4(rep):
    section("8. MODEL 4 -- symptom heads (labels are SYNTHETIC)")
    b = load(os.path.join(rep, "symptom_board.csv"), "symptom board")
    if b is None:
        return
    print(f"         {len(b)} heads trained")
    show = [c for c in b.columns if any(k in c.lower() for k in
            ("target", "symptom", "base_rate", "pr_auc", "lift", "alert_rate", "brier"))][:8]
    print(b[show].head(12).to_string(index=False)[:1400] if show else "")
    lift = col(b, "PR_lift", "pr_lift", "lift_at_budget")
    if lift:
        v = pd.to_numeric(b[lift], errors="coerce").dropna()
        chk("at least one head beats its own base rate", (v > 1.0).any(),
            f"max lift {v.max():.2f}")
    rec("WARN", "symptom metrics describe a synthetic generator, not patients",
        "excluded from promotion gates by construction")


def check_evaluation(rep):
    section("9. EVALUATION -- importance, latency, parity, robustness")
    imp = load(os.path.join(rep, "block_importance.csv"), "block importance")
    if imp is not None:
        v = col(imp, "delta_mae", "increase", "importance")
        if v:
            s = pd.to_numeric(imp[v], errors="coerce").dropna()
            chk("*** block importance is not all-zero ***", float(s.abs().max()) > 0,
                f"max {s.abs().max():.6f}")
            top = imp.sort_values(v, ascending=False).head(5)
            print(top.to_string(index=False)[:600])

    lat = load(os.path.join(rep, "batch_latency.yaml"), "batch latency")
    if isinstance(lat, dict):
        print(f"         {json.dumps(lat)[:400]}")
        p95 = lat.get("p95_ms") or lat.get("p95")
        budget = lat.get("latency_budget_ms", 200.0)
        if p95 is not None:
            chk(f"p95 serving latency within the {budget} ms budget",
                float(p95) <= float(budget), f"{p95} ms")

    par = load(os.path.join(rep, "walk_forward_parity.csv"), "walk-forward parity")
    if par is None:
        par = load(os.path.join(rep, "walk_forward_parity.yaml"), "walk-forward parity")
    if isinstance(par, dict):
        print(f"         {json.dumps(par)[:300]}")
        chk("train/serve parity verdict is consistent",
            str(par.get("verdict", "")).lower() == "consistent", json.dumps(par)[:200])
    elif isinstance(par, pd.DataFrame):
        print(par.to_string(index=False)[:500])

    conc = load(os.path.join(rep, "error_concentration.csv"), "error concentration")
    if conc is not None:
        print(conc.to_string(index=False)[:400])

    miss = load(os.path.join(rep, "missingness_robustness.csv"), "missingness sweep")
    if miss is not None:
        print(miss.to_string(index=False)[:600])
        dm = col(miss, "delta_MAE", "delta_mae")
        if dm:
            chk("degradation under missingness is bounded",
                pd.to_numeric(miss[dm], errors="coerce").abs().max() < 15.0,
                miss[dm].to_list())


def check_gates(rep):
    section("10. SAFETY GATES -- the promotion decision")
    g = load(os.path.join(rep, "safety_gates.csv"), "safety gates")
    if g is None:
        rec("FAIL", "*** no safety gate report -- promotion cannot be justified ***")
        return None
    print(g.to_string(index=False)[:2500])
    st = col(g, "status")
    cr = col(g, "critical")
    if not st:
        rec("FAIL", "gate report has no status column")
        return None
    crit = g[g[cr].astype(str).str.upper().isin(["TRUE", "1", "YES"])] if cr else g
    failed_crit = crit[crit[st].astype(str).str.upper() == "FAIL"]
    chk("*** no CRITICAL safety gate fails ***", failed_crit.empty,
        failed_crit.to_string(index=False)[:600])
    warns = g[g[st].astype(str).str.upper() == "WARN"]
    print(f"\n         {len(g)} gates | {len(crit)} critical | {len(warns)} warnings")
    if not warns.empty:
        nm = col(warns, "gate", "name")
        rec("WARN", f"{len(warns)} gates warn",
            ", ".join(warns[nm].astype(str))[:300] if nm else "")
    # A gate suite that is all-WARN proves nothing; there must be real PASSes.
    chk("    the gate suite actually asserted something",
        (g[st].astype(str).str.upper() == "PASS").sum() >= 5,
        f"{(g[st].astype(str).str.upper() == 'PASS').sum()} PASS")
    return failed_crit.empty


def check_bundle(promotable):
    section("11. THE SHIPPED BUNDLE -- does it load and predict?")
    paths = ["final_model/model.pkl"] + sorted(
        glob.glob("Artifacts/*/model_trainer/trained_model/*.pkl")
        + glob.glob("Artifacts/*/model_trainer/trained_model/*.joblib"))
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        rec("WARN" if promotable is False else "FAIL",
            "no bundle on disk",
            "expected, the run was not promotable" if promotable is False else "")
        return
    print(f"         loading {path}")
    from src.serving.model_registry import ModelRegistry
    reg = ModelRegistry()
    reg.refresh(force=True)
    chk("registry loads the bundle and it survives the smoke predict", reg.loaded,
        reg.health().get("detail", ""))
    if not reg.loaded:
        return
    p = reg.predictor
    b = p.b
    print(f"         version {b.get('model_version')} | "
          f"{len(b.get('feature_names') or [])} features | "
          f"forecasters {len(b.get('forecasters') or {})}")
    chk("bundle records that symptom labels are synthetic",
        b.get("symptom_labels_synthetic") is True, b.get("symptom_labels_synthetic"))
    chk("bundle carries the emergency floor",
        float(getattr(p.config, "emergency_floor_mmHg", 0)) == EMERGENCY_FLOOR_MMHG)
    unserv = b.get("selected_but_unservable")
    if unserv:
        rec("WARN", "the bake-off winner was unservable and was substituted", str(unserv))

    # An end-to-end advisory through the real serving path.
    from src.serving.advisory import build_advisory
    from src.serving.schemas import PredictRequest
    from src.serving.vocabulary import build_vocabulary
    from src.utils.ml_utils.rule_engine.registry import build_registry
    rules = build_registry()
    note = (build_vocabulary(rules).get("rules") or {}).get("note", "")
    rows, day = [], pd.Timestamp("2026-04-01")
    for i in range(60):
        rows.append({"date": str((day + pd.Timedelta(days=2 * i)).date()),
                     "sbp": int(138 + (i % 7) * 3 - (i % 3)), "dbp": int(78 + i % 5)})
    d = build_advisory(PredictRequest(patient_id="v1", readings=rows,
                                      profile={"age": 68.0, "is_male": 1}), p, rules, note)
    fc = (d.get("forecast") or {}).get("sbp") or {}
    chk("*** the shipped bundle issues a forecast ***", bool(fc), d.get("confidence_tier"))
    if fc:
        # `point` / `lo80` / `hi80`, the keys BPPredictor.predict actually writes. Reading
        # `sbp` / `lo` / `hi` here silently compared None against a range and reported a
        # failure that was in this file, not in the model.
        vals = [f.get("point") for f in fc.values() if isinstance(f, dict)]
        chk("    every horizon returns a plausible blood pressure",
            bool(vals) and all(v is not None and 60 < v < 260 for v in vals), vals)

        # Exactly one node carries a band: fit_quantile_interval runs for sbp at one horizon.
        # Asserting every node has one would be asserting something false about the bundle.
        banded = [f for f in fc.values()
                  if isinstance(f, dict) and f.get("lo80") is not None]
        chk("    one horizon carries an 80% conformal interval", len(banded) >= 1,
            f"{len(banded)} of {len(fc)} nodes banded")
        for f in banded:
            chk("    the interval brackets the point estimate",
                f["lo80"] <= f["point"] <= f["hi80"], f)
            chk("    the interval has positive width", f["hi80"] > f["lo80"], f)
    pers = d.get("personalisation") or {}
    if pers.get("threshold") is not None:
        chk("    personalised threshold sits below the emergency floor",
            pers["threshold"] < EMERGENCY_FLOOR_MMHG, pers["threshold"])
    chk("    early warning is scored", d.get("early_warning") is not None)
    print(f"         tier={d.get('confidence_tier')} "
          f"threshold={pers.get('threshold')} "
          f"predict={(d.get('timings') or {}).get('predict_ms')} ms")

    # The emergency floor must survive a model being present.
    rows2 = list(rows)
    rows2[-1] = {**rows2[-1], "sbp": 195, "dbp": 100}
    d2 = build_advisory(PredictRequest(patient_id="v2", readings=rows2,
                                       profile={"age": 68.0, "is_male": 1}), p, rules, note)
    chk("*** SBP 195 is an emergency WITH a model loaded ***",
        ((d2.get("rule_engine") or {}).get("current") or {}).get("is_emergency") is True)


def main():
    art = sys.argv[1] if len(sys.argv) > 1 else newest_artifact_dir()
    if not art:
        print("no Artifacts/ directory -- run `python main.py` first")
        return 1
    rep = os.path.join(art, "model_trainer", "report")
    print(f"verifying {art}")
    print(f"reports   {rep}" + ("" if os.path.isdir(rep) else "   (ABSENT)"))

    check_ingestion(art)
    check_validation(art)
    check_transformation(art)
    check_model1(art, rep)
    check_tuning(rep)
    check_model2(rep)
    check_model3(rep)
    check_model4(rep)
    check_evaluation(rep)
    promotable = check_gates(rep)
    check_bundle(promotable)

    section("VERDICT")
    print(f"  {R['PASS']} passed   {R['FAIL']} failed   {R['WARN']} warnings (absent evidence)")
    if R["FAIL"]:
        print("\n  FAILED:")
        for s, n, d in LINES:
            if s == "FAIL":
                print(f"    - {n}" + (f"  ({d[:120]})" if d else ""))
    if R["WARN"]:
        print("\n  ABSENT / UNPROVEN:")
        for s, n, d in LINES:
            if s == "WARN":
                print(f"    - {n}" + (f"  ({d[:120]})" if d else ""))
    print()
    return 1 if R["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
