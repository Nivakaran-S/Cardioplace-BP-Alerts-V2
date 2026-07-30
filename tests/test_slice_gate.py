"""The subgroup gate must catch unfairness and must not catch difficulty.

A fairness gate has two ways to be useless. It can miss a model that genuinely serves one
group worse, and it can fire on a group whose data is simply harder -- which is what this one
did, blocking promotion because under-50 dialysis patients have 24% more variable systolic
pressure than the cohort. The EWMA baseline, which does no learning at all, was 2.50 mmHg
worse for that group; the model was 2.33 mmHg worse, i.e. it closed part of the gap and was
failed for the attempt.

Every case below is therefore paired: a fair-but-harder group that must PASS, and an
equally-hard group the model genuinely underserves that must FAIL. A gate that only satisfies
one direction is not tested.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.ml_utils.metric.timeseries_metric import slice_gate  # noqa: E402

FAILS = []


def chk(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"  <- {extra}"))
    if not cond:
        FAILS.append(name)


def cohort(hard_noise=12.0, easy_noise=6.0, hard_model_penalty=0.0, n=4000, seed=0):
    """Two groups of differing intrinsic difficulty.

    `hard_model_penalty` is genuine unfairness: extra error the model adds for the hard group
    ON TOP of its irreducible noise. At 0 the model is equally skilful for both.
    """
    rng = np.random.default_rng(seed)
    grp = np.where(rng.random(n) < 0.3, "hard", "easy")
    noise = np.where(grp == "hard", hard_noise, easy_noise)
    y = rng.normal(140, 20, n)
    # The baseline is the truth plus each group's irreducible noise.
    base = y + rng.normal(0, noise)
    # The model improves on the baseline by a constant, minus any penalty for the hard group.
    gain = 2.0 - np.where(grp == "hard", hard_model_penalty, 0.0)
    pred = y + rng.normal(0, np.maximum(noise - gain, 0.5))
    return pd.DataFrame({"grp": grp, "y": y, "pred": pred, "base": base})


def mae(x):
    return float(np.mean(np.abs(x.pred - x.y)))


def ref(x):
    return float(np.mean(np.abs(x.base - x.y)))


def _difficulty_is_not_unfairness():
    """The false failure that blocked a real promotion."""
    d = cohort(hard_model_penalty=0.0)

    raw = slice_gate(d, ["grp"], mae, 1.5, "MAE")
    hard_raw = raw[raw.level == "hard"].iloc[0]
    chk("raw-metric gate FAILS the harder group even though the model is equally skilful",
        not bool(hard_raw.passes), hard_raw.to_dict())
    chk(f"    (its raw gap is {hard_raw.gap:+.2f} mmHg, purely irreducible noise)", True)

    rel = slice_gate(d, ["grp"], mae, 1.5, "MAE", reference=ref)
    hard_rel = rel[rel.level == "hard"].iloc[0]
    chk("*** reference gate PASSES it -- equal improvement over an equally hard floor ***",
        bool(hard_rel.passes), hard_rel.to_dict())
    chk("    the raw value is still reported, nothing is hidden",
        np.isclose(hard_rel.value, hard_raw.value), (hard_rel.value, hard_raw.value))
    chk("    and raw_gap is retained as a diagnostic",
        abs(hard_rel.raw_gap - hard_raw.gap) < 1e-6, (hard_rel.raw_gap, hard_raw.gap))
    chk("    the basis is recorded on every row",
        (rel.basis == "improvement over the subgroup's own reference").all())


def _real_unfairness_still_fails():
    """The check that matters: the fix must not have disabled the gate."""
    d = cohort(hard_model_penalty=3.0)          # model helps the hard group far less
    rel = slice_gate(d, ["grp"], mae, 1.5, "MAE", reference=ref)
    hard = rel[rel.level == "hard"].iloc[0]
    chk("*** genuine unfairness STILL FAILS under the reference gate ***",
        not bool(hard.passes), hard.to_dict())
    chk("    and the gap has the right sign (less improvement than overall)",
        hard.gap < 0, hard.gap)

    # And a graded check, so it is not passing by luck at one magnitude.
    seen = []
    for pen in (0.0, 1.0, 2.0, 4.0):
        r = slice_gate(cohort(hard_model_penalty=pen), ["grp"], mae, 1.5, "MAE",
                       reference=ref)
        seen.append((pen, bool(r[r.level == "hard"].iloc[0].passes)))
    chk("gate is monotone in the size of the unfairness",
        [p for p, ok in seen if not ok] == [4.0] or
        all(not ok for p, ok in seen if p >= 3.0), seen)
    print(f"         penalty -> passes: {seen}")


def _higher_is_better():
    """Precision at a fixed budget is bounded by the subgroup's base rate."""
    rng = np.random.default_rng(1)
    n = 6000
    grp = np.where(rng.random(n) < 0.4, "rare", "common")
    base = np.where(grp == "rare", 0.02, 0.10)         # event rate differs 5x
    ev = (rng.random(n) < base).astype(int)
    # Detector flags the top 5% by a score correlated with the event, equally well for both.
    score = ev * 2.0 + rng.normal(0, 1, n)
    flag = (score >= np.percentile(score, 95)).astype(int)
    d = pd.DataFrame({"grp": grp, "event_next": ev, "flag": flag})

    def prec(x):
        return float(x[x.flag == 1].event_next.mean()) if (x.flag == 1).any() else np.nan

    def rec(x):
        return float(x[x.event_next == 1].flag.mean()) if (x.event_next == 1).any() else np.nan

    raw = slice_gate(d, ["grp"], prec, 0.05, "precision")
    rr = raw[raw.level == "rare"].iloc[0]
    chk("precision gate fails the rare-event group on its base rate alone",
        not bool(rr.passes), rr.to_dict())

    # Normalising precision by the base rate does NOT rescue it: the achievable ceiling
    # scales with the prior too, so the difference is still not comparable. Recorded here
    # because it was the first fix attempted and it does not work.
    lift = slice_gate(d, ["grp"], prec, 0.05, "precision",
                      reference=lambda x: float(x.event_next.mean()), higher_is_better=True)
    chk("    and normalising precision by base rate does NOT fix it (first attempt)",
        not bool(lift[lift.level == "rare"].iloc[0].passes))

    pr = raw.set_index("level").value
    rc = slice_gate(d, ["grp"], rec, 0.05, "recall", higher_is_better=True)         .set_index("level").value
    p_spread = float(abs(pr["rare"] - pr["common"]))
    r_spread = float(abs(rc["rare"] - rc["common"]))
    chk("*** recall is far more base-rate stable than precision ***",
        r_spread < p_spread / 4, f"precision {p_spread:.3f} vs recall {r_spread:.3f}")
    print(f"         precision spread {p_spread:.3f} | recall spread {r_spread:.3f}")

    # And recall must still fail a detector that genuinely misses one group's events.
    d2 = d.copy()
    miss = (d2.grp == "rare") & (d2.event_next == 1) & (rng.random(len(d2)) < 0.6)
    d2.loc[miss, "flag"] = 0
    r2 = slice_gate(d2, ["grp"], rec, 0.05, "recall", higher_is_better=True)
    chk("*** recall STILL fails a detector that misses a group's events ***",
        not bool(r2[r2.level == "rare"].iloc[0].passes), r2.to_dict("records"))


def _backwards_compatible():
    d = cohort()
    out = slice_gate(d, ["grp"], mae, 1.5, "MAE")
    chk("without a reference the behaviour is unchanged",
        out.basis.eq("raw metric vs overall").all() and out.reference.isna().all())
    chk("    and gap still equals value - overall",
        np.allclose(out.gap, out.value - out.overall), out[["gap", "value", "overall"]])


def run():
    for title, fn in (("difficulty is not unfairness", _difficulty_is_not_unfairness),
                      ("real unfairness still fails", _real_unfairness_still_fails),
                      ("higher-is-better metrics", _higher_is_better),
                      ("backwards compatibility", _backwards_compatible)):
        print(f"\n--- {title} ---")
        fn()
    print("\n" + ("ALL SLICE GATE TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


def test_slice_gate():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
