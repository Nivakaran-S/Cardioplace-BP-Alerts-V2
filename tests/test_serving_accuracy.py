"""Is the SERVED model as accurate as the training run said it was?

Every other suite in this repo tests wiring: that blocks are produced, that features resolve,
that renderers read the right keys. A model predicting 150 mmHg for every patient would pass
all of them. This one asks whether the bundle on disk, driven through `to_history` ->
`BPPredictor.predict` exactly as the API drives it, actually forecasts.

That is a different question from the training scorecard, and the difference is where serving
defects live. A dropped column, a renamed field, a mapping that NaNs a feature group -- each
degrades this while leaving every training report untouched, because the reports were computed
from the training panel and never travel through `to_history`. Sixty-five of a hundred and
seventy-five features were arriving NaN and no report showed it.

The data is the held-out split of the real HEMOBP corpus. Symptoms, medications and heart rate
are absent from it -- the corpus has no such columns, which is why the symptom labels are
synthetic -- so those features are legitimately missing here, and the detector's degradation
against its training figure is measured rather than assumed.

Thresholds are deliberately loose. This is a regression guard against a serving path that has
broken, not a re-derivation of the ship decision; the paired bootstrap in the training run is
what decides whether a model ships. A tight bound here would fail on sampling noise from 400
rows and get muted, which is worse than no test.
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.serving.mapping import to_history  # noqa: E402
from src.serving.model_registry import ModelRegistry  # noqa: E402
from src.serving.schemas import PredictRequest  # noqa: E402

FAILS = []

MIN_HIST = 30          # the 30-session windows must be able to fill
N_PATIENTS = 40
CUTS_PER_PATIENT = 6

#: What the run that produced the promoted bundle reported on its own held-out split. Serving
#: is compared against these, with room for the sample being smaller and differently drawn.
TRAIN_REPORTED = {"sbp_mae": 14.135, "dbp_mae": 7.298, "idwg_mae": 0.480,
                  "detector_auc": 0.765, "detector_ap_lift": 2.33}


def chk(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"  <- {extra}"))
    if not cond:
        FAILS.append(name)


def _latest_run():
    art = ROOT / "Artifacts"
    if not art.exists():
        return None
    runs = [d for d in art.iterdir()
            if (d / "data_validation/validated/test.csv").exists()]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def _request(g, upto):
    """A PredictRequest over real sessions 0..upto-1 -- the shape a client would send."""
    h = g.iloc[:upto]
    rows = []
    for r in h.itertuples():
        row = {"date": str(pd.Timestamp(r.ts).date()),
               "sbp": int(round(r.sbp)), "dbp": int(round(r.dbp))}
        for c in ("weight", "idwg", "uf_total", "session_hours", "sbp_drop"):
            v = getattr(r, c, np.nan)
            if pd.notna(v):
                row[c] = float(v)
        rows.append(row)
    last = h.iloc[-1]
    prof = {"age": float(last.age) if pd.notna(last.age) else 65.0,
            "is_male": int(str(last.gender).upper().startswith("M"))
            if pd.notna(last.gender) else 0,
            "is_dm": int(bool(last.DM)) if pd.notna(last.DM) else 0}
    if pd.notna(last.dryweight):
        prof["dryweight"] = float(last.dryweight)
    if pd.notna(last.first_dialysis_ts):
        prof["first_dialysis"] = str(pd.Timestamp(last.first_dialysis_ts).date())
    return PredictRequest(patient_id=str(last.patient_id), readings=rows, profile=prof)


def score(P, test):
    recs, rejected = [], 0
    counts = test.groupby("patient_id").size().sort_values(ascending=False)
    usable = [p for p in counts.index if counts[p] >= MIN_HIST + 4][:N_PATIENTS]
    for pid in usable:
        g = test[test.patient_id == pid].sort_values("ts").reset_index(drop=True)
        g = g[g.sbp.notna() & g.dbp.notna()].reset_index(drop=True)
        if len(g) < MIN_HIST + 4:
            continue
        for upto in sorted(set(np.linspace(MIN_HIST, len(g) - 4,
                                           CUTS_PER_PATIENT).astype(int))):
            try:
                out = P.predict(to_history(_request(g, upto)))
            except Exception:                                          # noqa: BLE001
                rejected += 1
                continue
            f = out.get("forecast") or {}
            hs = g.sbp.iloc[:upto]
            rec = {"pid": pid,
                   "sbp_ewma": float(hs.ewm(alpha=0.3).mean().iloc[-1]),
                   "dbp_ewma": float(g.dbp.iloc[:upto].ewm(alpha=0.3).mean().iloc[-1])}
            for h, key in enumerate(("h0", "h1", "h2")):
                if upto + h >= len(g):
                    continue
                act = g.iloc[upto + h]
                for sig in ("sbp", "dbp", "idwg"):
                    node = (f.get(sig) or {}).get(key) or {}
                    pt = node.get("point")
                    if pt is not None and np.isfinite(pt) and pd.notna(act.get(sig)):
                        rec[f"{sig}_{key}_pred"] = float(pt)
                        rec[f"{sig}_{key}_act"] = float(act[sig])
                n = (f.get("sbp") or {}).get(key) or {}
                if n.get("lo80") is not None and pd.notna(act.sbp):
                    rec[f"cover_{key}"] = int(n["lo80"] <= act.sbp <= n["hi80"])
            ew = out.get("early_warning") or {}
            rec["ew_score"] = ew.get("score")
            rec["ew_flag"] = ew.get("flagged")
            # The detector's own event definition, matched to `EarlyWarningDataset.build`:
            # this patient's p95 reached (>=, not >) within the warn window.
            thr = float(hs.quantile(0.95))
            fut = g.sbp.iloc[upto:upto + 3]
            rec["event"] = int((fut >= thr).any()) if len(fut) else None
            pers = out.get("personalisation") or {}
            rec["thresh"] = pers.get("threshold")
            rec["capped"] = pers.get("capped")
            recs.append(rec)
    return pd.DataFrame(recs), rejected


def mae(a, b):
    m = a.notna() & b.notna()
    return float(np.mean(np.abs(a[m] - b[m]))) if m.sum() else float("nan")


def run():
    run_dir = _latest_run()
    if run_dir is None:
        print("  SKIP    no Artifacts run with a validated test split; nothing to score against")
        return 0
    reg = ModelRegistry()
    reg.refresh(force=True)
    P = reg.predictor
    if P is None:
        print("  SKIP    no bundle on disk")
        return 0
    print(f"bundle {P.b.get('model_version')}  shipped {P.b.get('shipped')}")
    print(f"scoring against {run_dir.name}")

    test = pd.read_csv(run_dir / "data_validation/validated/test.csv", parse_dates=["ts"])

    # The scored split comes from the LATEST run, which may not be the run that produced the
    # bundle. That is only safe because the split is deterministic -- so verify it, rather
    # than trust it. If a scored row was ever trained on, every number below is optimistic and
    # the file is worse than useless. Checked at row level: the split is temporal, so patient
    # overlap is expected and correct while ROW overlap is disqualifying.
    train_path = run_dir / "data_validation/validated/train.csv"
    bundle_run = ROOT / "Artifacts" / str(P.b.get("run_id") or "")
    bundle_train = bundle_run / "data_validation/validated/train.csv"
    for label, path in (("this run", train_path), ("the bundle's run", bundle_train)):
        if not path.exists():
            print(f"  note   no train split for {label}; overlap unverified")
            continue
        tr = pd.read_csv(path, usecols=["patient_id", "ts"])
        tr["ts"] = pd.to_datetime(tr.ts)
        seen = set(map(tuple, tr.values))
        rows = set(map(tuple, test[["patient_id", "ts"]].values))
        overlap = len(seen & rows)
        chk(f"*** no scored row was trained on ({label}) ***", overlap == 0,
            f"{overlap} of {len(rows)} scored rows appear in that training split")

    D, rejected = score(P, test)
    print(f"\nscored {len(D)} patient-cut pairs over {D.pid.nunique()} patients "
          f"({rejected} requests rejected)")
    chk("the sample is large enough to say anything", len(D) >= 150, len(D))
    # A schema that rejects real held-out sessions is a bug in the schema, not in the data.
    chk("the schema accepts essentially all real held-out sessions",
        rejected <= 0.10 * (len(D) + rejected), f"{rejected} rejected of {len(D) + rejected}")

    print("\n--- forecast, against the EWMA baseline on the same rows ---")
    for sig in ("sbp", "dbp", "idwg"):
        for key in ("h0", "h1", "h2"):
            pc, ac = f"{sig}_{key}_pred", f"{sig}_{key}_act"
            if pc not in D:
                continue
            m = D[pc].notna() & D[ac].notna()
            if m.sum() < 30:
                continue
            served = mae(D.loc[m, pc], D.loc[m, ac])
            base = (mae(D.loc[m, f"{sig}_ewma"], D.loc[m, ac])
                    if f"{sig}_ewma" in D else float("nan"))
            print(f"    {sig} {key}: served {served:6.3f}   ewma "
                  f"{base if np.isfinite(base) else float('nan'):6.3f}   n={int(m.sum())}")

    # The headline guard. Not "beats the baseline" -- the training run's own paired bootstrap
    # puts the sbp gain at 0.33 mmHg on a 14 mmHg error, which 400 rows cannot resolve. This
    # asks the answerable question: has serving accuracy COLLAPSED relative to the baseline?
    for sig, tol in (("sbp", 1.25), ("dbp", 1.25), ("idwg", 1.40)):
        pc, ac, bc = f"{sig}_h0_pred", f"{sig}_h0_act", f"{sig}_ewma"
        if pc not in D or bc not in D:
            continue
        m = D[pc].notna() & D[ac].notna()
        served, base = mae(D.loc[m, pc], D.loc[m, ac]), mae(D.loc[m, bc], D.loc[m, ac])
        if not np.isfinite(base):
            continue
        chk(f"*** served {sig} forecast is not far worse than a 3-line EWMA ***",
            served <= tol * base, f"{served:.3f} vs {base:.3f} ({served / base:.2f}x)")

    # DBP ships the EWMA baseline itself, so serving must REPRODUCE it, not merely approach
    # it. This is the sharpest single check in the file: it is the one place where the
    # expected answer is known exactly, so any drift in the serving path shows up undiluted.
    if P.b.get("shipped", {}).get("dbp", ("", ""))[0] == "baseline":
        m = D.dbp_h0_pred.notna() & D.dbp_h0_act.notna()
        gap = abs(mae(D.loc[m, "dbp_h0_pred"], D.loc[m, "dbp_h0_act"])
                  - mae(D.loc[m, "dbp_ewma"], D.loc[m, "dbp_h0_act"]))
        chk("*** DBP ships the EWMA baseline, so serving reproduces it exactly ***",
            gap < 0.05, f"MAE differs by {gap:.4f} mmHg -- the serving path is not faithful")

    print("\n--- 80% interval coverage ---")
    for key in ("h0", "h1", "h2"):
        c = f"cover_{key}"
        if c in D and D[c].notna().sum() >= 30:
            cov = float(D[c].mean())
            print(f"    {key}: {cov:.3f} on n={int(D[c].notna().sum())}")
            chk(f"    the 80% interval covers roughly 80% at {key}", 0.68 <= cov <= 0.92,
                f"{cov:.3f} -- conformal calibration does not hold at serving")

    print("\n--- early-warning detector ---")
    m = D.ew_score.notna() & D.event.notna()
    if m.sum() >= 60:
        from sklearn.metrics import average_precision_score, roc_auc_score
        y, s = D.loc[m, "event"].astype(int), D.loc[m, "ew_score"].astype(float)
        if 0 < y.sum() < len(y):
            auc = roc_auc_score(y, s)
            lift = average_precision_score(y, s) / y.mean()
            print(f"    n={int(m.sum())}  base rate {y.mean():.3f}  "
                  f"AUC-ROC {auc:.3f}  AUC-PR lift {lift:.2f}x")
            print(f"    [training run: AUC-ROC {TRAIN_REPORTED['detector_auc']}, "
                  f"lift {TRAIN_REPORTED['detector_ap_lift']}x -- on a panel where the "
                  f"symptom, adherence and heart-rate columns were populated]")
            chk("*** the detector still separates the event at serving ***", auc >= 0.60,
                f"AUC {auc:.3f} -- at 0.5 it is not ranking at all")
            chk("    and still beats the base rate on precision", lift >= 1.3, f"{lift:.2f}x")
            fl = D.loc[m, "ew_flag"].astype(bool)
            if fl.any():
                print(f"    at the shipped cut: alert rate {fl.mean():.3f}, "
                      f"precision {y[fl].mean():.3f}, "
                      f"recall {y[fl].sum() / max(y.sum(), 1):.3f}")

    print("\n--- personalised offset ---")
    if D.thresh.notna().any():
        t = D.thresh.dropna()
        print(f"    threshold min {t.min():.0f} / median {t.median():.0f} / max {t.max():.0f}")
        print(f"    capped at the governance limit on {D.capped.mean():.1%} of rows")
        # Governance, not accuracy: these two are the values the whole safety case rests on.
        chk("*** no personalised threshold reaches the 180 emergency floor ***",
            bool((t < 180).all()), f"max {t.max()}")
        chk("*** no personalised threshold falls below the 115 mmHg tightening limit ***",
            bool((t >= 115).all()), f"min {t.min()}")

    print("\n" + ("ALL SERVING ACCURACY TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


def test_serving_accuracy():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
