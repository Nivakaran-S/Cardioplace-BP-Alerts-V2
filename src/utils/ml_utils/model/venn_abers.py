"""Inductive Venn-Abers calibration: a probability that reports its own uncertainty.

`CalibratedClassifierCV(method="isotonic")` returns a single number and offers no guarantee
about it. Venn-Abers (Vovk & Petej, arXiv:1211.0025) returns a PAIR (p0, p1) that is
guaranteed, under exchangeability alone, to bracket a perfectly calibrated probability -- and
the width of that pair is itself the honest statement of how little the calibration set knows
about this region of the score.

That matters here more than usual. Several symptom heads sit at a base rate near 0.1%, where a
point probability of "0.004" is reported with the same apparent authority whether it rests on
four hundred calibration examples or on two. The interval says which.

## The construction

Given calibration scores `s_i` with labels `y_i`, and a test score `s`:

    p0 = isotonic fit on {(s_i, y_i)} + (s, 0), evaluated at s
    p1 = isotonic fit on {(s_i, y_i)} + (s, 1), evaluated at s

`p0 <= p1` always, and the calibrated point estimate is `p1 / (1 - p0 + p1)`.

## The one approximation, stated plainly

Done literally, that is two isotonic regressions per test point -- O(n_test * n_cal log n_cal),
which is far too slow for 45 heads over thousands of rows. There is an exact O(n log n)
algorithm via the cumulative-sum diagram; it is fiddly and easy to get subtly wrong.

Instead the pair is computed exactly on a GRID of calibration-score quantiles and interpolated
between grid points. The error is bounded by how much p0/p1 can move between adjacent
quantiles, which shrinks as `GRID` grows, and monotonicity is preserved by construction because
both curves are non-decreasing and interpolation between non-decreasing points is
non-decreasing. `GRID = 256` puts the grid spacing well below the calibration noise at any base
rate this project sees.

This is an approximation of the interval, not of the guarantee's shape: the returned p0 is
still <= p1, and both still come from real isotonic fits on the real calibration set.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression

#: Grid points at which the pair is computed exactly. Everything between is interpolated.
GRID: int = 256

#: Below this many calibration rows the interval is too wide to be informative and the whole
#: exercise is theatre; the caller is told to fall back to isotonic.
MIN_CALIBRATION: int = 50


def _iso(x, y):
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)


def venn_abers_grid(cal_scores, cal_labels, grid: int = GRID):
    """Precompute the (score, p0, p1) curve once. Returns None if not identifiable.

    Split out from `venn_abers_pairs` because it is the expensive half -- 2 isotonic fits per
    grid point -- and it depends ONLY on the calibration set. Recomputing it per prediction
    made `predict_proba` O(n_test * n_cal log n_cal): fine for the single fit at training
    time, ruinous at serving, where 45 heads are scored per request. Fitted once, stored on
    the estimator, and reduced to an interpolation at predict time.
    """
    s = np.asarray(cal_scores, dtype=float)
    y = np.asarray(cal_labels, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    s, y = s[m], y[m]
    if s.size < MIN_CALIBRATION or len(np.unique(y)) < 2:
        return None
    q = np.unique(np.quantile(s, np.linspace(0.0, 1.0, min(grid, s.size))))
    if q.size < 2:
        return None
    p0 = np.empty(q.size)
    p1 = np.empty(q.size)
    for i, v in enumerate(q):
        xs = np.append(s, v)
        p0[i] = float(_iso(xs, np.append(y, 0.0)).predict([v])[0])
        p1[i] = float(_iso(xs, np.append(y, 1.0)).predict([v])[0])
    return q, np.maximum.accumulate(p0), np.maximum.accumulate(p1)


def venn_abers_pairs(cal_scores, cal_labels, test_scores, grid: int = GRID):
    """`(p0, p1)` arrays for `test_scores`. `p0 <= p1` elementwise."""
    s = np.asarray(cal_scores, dtype=float)
    y = np.asarray(cal_labels, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    s, y = s[m], y[m]
    t = np.asarray(test_scores, dtype=float)

    built = venn_abers_grid(s, y, grid=grid)
    if built is None:
        # Nothing to calibrate against. Returning the raw score as a degenerate pair would
        # claim a guarantee that does not hold, so signal absence instead.
        return None, None
    return _interp(built, t)


def _interp(built, test_scores):
    """Interpolate a precomputed grid at `test_scores`, keeping p0 <= p1."""
    q, p0, p1 = built
    t = np.asarray(test_scores, dtype=float)
    lo = np.interp(t, q, p0)
    hi = np.interp(t, q, p1)
    return np.minimum(lo, hi), np.maximum(lo, hi)


class VennAbersCalibrator(BaseEstimator, ClassifierMixin):
    """Wraps a fitted classifier so `predict_proba` is Venn-Abers calibrated.

    Drop-in for `CalibratedClassifierCV`: same `predict_proba(X)[:, 1]` contract, so nothing
    downstream changes. `predict_interval(X)` is the extra -- the (p0, p1) pair that the point
    estimate is a summary of.

    The base estimator is used as fitted and never refitted, matching the `FrozenEstimator`
    intent of the isotonic path it replaces: the calibration split must stay disjoint from the
    training split, and refitting here would quietly destroy that.
    """

    def __init__(self, base_estimator=None, grid: int = GRID):
        self.base_estimator = base_estimator
        self.grid = grid

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.cal_scores_ = np.asarray(
            self.base_estimator.predict_proba(X)[:, 1], dtype=float)
        self.cal_labels_ = np.asarray(y, dtype=float)
        # The whole cost of Venn-Abers lives here, paid once.
        self.grid_ = venn_abers_grid(self.cal_scores_, self.cal_labels_, grid=self.grid)
        self.usable_ = self.grid_ is not None
        return self

    def _raw(self, X):
        return np.asarray(self.base_estimator.predict_proba(X)[:, 1], dtype=float)

    def predict_interval(self, X):
        """`(p0, p1)` per row, or `(None, None)` when the calibration set was too small."""
        if getattr(self, "grid_", None) is None:
            return None, None
        return _interp(self.grid_, self._raw(X))

    def predict_proba(self, X):
        raw = self._raw(X)
        if getattr(self, "grid_", None) is None:
            lo = hi = None
        else:
            lo, hi = _interp(self.grid_, raw)
        if lo is None:
            # Too little calibration data. Passing the raw score through is honest -- it is
            # what an uncalibrated head says -- and `usable_` records that no guarantee holds.
            p = raw
        else:
            # The Venn-Abers point estimate. Note this is NOT the midpoint: it is the value
            # that makes the pair self-consistent as a probability.
            denom = 1.0 - lo + hi
            p = np.where(denom > 0, hi / np.where(denom > 0, denom, 1.0), raw)
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
