"""The one definition of how far apart two readings are.

A skipped reading is an absent row, never an imputed one: the panel is indexed by session,
not by calendar day, and the spacing lives in `days_since_last`. That decision is sound --
a daily grid would fabricate most of its own rows and teach the forecaster to recover the
interpolator that made them -- but the expression implementing it had been copy-pasted into
six places, one of which had already drifted. This module is the single source.

Two columns come out of `attach_cadence`, and the difference between them matters:

  days_since_last      the MODEL INPUT. Filled on the first row, clipped to [1, 30]. Bounded
                       because `days_since_last` has p25 == p75 == 2, so a single 1132-day
                       value would inflate its standard deviation by two orders of magnitude
                       and destroy the feature for every scaled linear model.
  days_since_last_raw  the TRUTH. No fill, no clip, NaN on a patient's first row. Everything
                       that is not a model input -- contract checks, drift, staleness, reports
                       -- reads this one, because the clipped column cannot distinguish a
                       30-day absence from a 1132-day one.

`days_since_last_raw` MUST stay in `CausalFeatureBuilder.DROP`. That class derives its
feature list from every numeric column not in `DROP`, so a new column silently becomes a
model input, changes every estimator's input matrix and re-opens the ship decision.
`cadence_audit` asserts this directly rather than trusting it.
"""

import numpy as np
import pandas as pd

GAP_FIRST_ROW_DAYS = 2.0    # no predecessor exists; the corpus cadence is ~2 days
GAP_MIN_DAYS = 1.0
GAP_CLIP_HI_DAYS = 30.0     # model-input ceiling; cadence_audit reports how often it binds


def raw_gap_days(ts, by=None, fractional: bool = False) -> pd.Series:
    """Calendar days since the previous reading. No fill, no clip.

    NaN on the first row of each series -- "no previous reading" is not a gap of zero, and
    conflating the two is what made the old contract check unfalsifiable.

    `by` is None for a single-patient frame, or an array/Series aligned to `ts`.
    `fractional=True` keeps sub-day resolution; the default floors to whole days, matching
    the corpus, whose timestamps carry no time component. A journaling app sends real
    timestamps, and there two readings 23 hours apart would otherwise measure a gap of 0.
    """
    s = pd.to_datetime(pd.Series(ts))
    d = s.diff() if by is None else s.groupby(np.asarray(by)).diff()
    if fractional:
        return d.dt.total_seconds() / 86400.0
    return d.dt.days.astype("float64")


def session_gap_days(ts, by=None, first: float = GAP_FIRST_ROW_DAYS,
                     lo: float = GAP_MIN_DAYS, hi: float = GAP_CLIP_HI_DAYS) -> pd.Series:
    """The model-input cadence feature.

    The default arguments reproduce the six inlined copies bit for bit; `frame_digest` over
    the feature matrix is the proof. Pass `lo=None`/`hi=None` to disable a bound.
    """
    g = raw_gap_days(ts, by=by).fillna(first)
    if lo is not None:
        g = g.clip(lower=lo)
    if hi is not None:
        g = g.clip(upper=hi)
    return g


def attach_cadence(df: pd.DataFrame, by="series_id", ts: str = "ts", **kw) -> pd.DataFrame:
    """Return a copy carrying both the clipped model input and the unclipped truth."""
    out = df.copy()
    grp = None if by is None else (out[by] if isinstance(by, str) else by)
    out["days_since_last_raw"] = raw_gap_days(out[ts], by=grp)
    out["days_since_last"] = session_gap_days(out[ts], by=grp, **kw)
    return out


def cadence_audit(panel: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """Structural probes over the cadence columns, in `leakage_audit`'s PASS/WARN/FAIL shape.

    These test the contract directly rather than restating how the column was built. The
    old check asserted `1 <= days_since_last <= 30` on a column that had just been clipped
    to [1, 30], so it could not fail; every probe here can.
    """
    rows = []

    def add(name, ok, detail="", status=None):
        rows.append(dict(probe=name, status=status or ("PASS" if ok else "FAIL"),
                         detail=detail))

    have_raw = "days_since_last_raw" in panel.columns
    add("days_since_last_raw is present", have_raw,
        "attach_cadence must supply the unclipped column")
    if not have_raw:
        return pd.DataFrame(rows)

    # The highest-risk line in the whole change: feature_names_ is derived from any numeric
    # column not in DROP, so this column becoming a feature is a silent, total regression.
    if features is not None:
        add("days_since_last_raw is NOT a model feature",
            "days_since_last_raw" not in features,
            f"{len(features)} features; a leak here changes every estimator's input matrix")

    grp = panel.series_id if "series_id" in panel.columns else None
    canon = session_gap_days(panel.ts, by=grp)
    stored = panel.days_since_last.astype("float64")
    add("days_since_last equals the canonical recomputation",
        bool(np.allclose(stored.values, canon.values, atol=1e-9)),
        "a re-sort, re-index or suffixing merge would break this")

    lo, hi = float(stored.min()), float(stored.max())
    add("days_since_last stays inside the model-input bounds",
        bool(lo >= GAP_MIN_DAYS and hi <= GAP_CLIP_HI_DAYS), f"[{lo:.0f}, {hi:.0f}] days")

    raw = panel.days_since_last_raw.dropna()
    add("raw gaps are strictly positive", bool((raw > 0).all()),
        f"{int((raw <= 0).sum())} non-positive of {len(raw):,}")

    over = float((raw > GAP_CLIP_HI_DAYS).mean()) if len(raw) else 0.0
    add("the clip is a guard, not a transform", over < 0.01,
        f"{over:.3%} of gaps exceed the {GAP_CLIP_HI_DAYS:.0f}-day ceiling "
        f"(max {raw.max():.0f}d); above 1% the feature no longer means what it is named",
        status=None if over < 0.01 else "FAIL")

    stale = float((raw > 14).mean()) if len(raw) else 0.0
    add("sessions following a stale gap are rare", True,
        f"{stale:.3%} of gaps exceed the 14-day staleness threshold",
        status="PASS" if stale < 0.005 else "WARN")

    return pd.DataFrame(rows)
