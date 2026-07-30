"""Feature selection by redundancy clustering.

Port of notebook section 4.3. The feature builder emits ~180 columns for three signals, and
most of them are the same column wearing a different window: `sbp_mean7` and `sbp_mean14`
correlate above 0.97 on this corpus, as do every adjacent lag pair. That redundancy is not
free. It splits permutation importance across interchangeable columns so nothing looks
important, it makes the linear pipelines ill-conditioned, and it spends tree depth on choices
between equivalent splits.

Three screens, in order:

1. DEGENERATE  -- constant or almost entirely missing. Nothing to learn from.
2. RELEVANCE   -- max(|pearson|, |spearman|) against the targets. Spearman as well as
                  Pearson because several features are monotone-but-not-linear in the target
                  (the volatility ratio especially), and a Pearson-only screen drops them.
3. REDUNDANCY  -- average-linkage hierarchical clustering on 1 - |r|, cut at 1 - 0.95, then
                  one representative per cluster. Clustering rather than a greedy pairwise
                  pass because redundancy here is transitive: mean7/mean14/mean30 form a
                  block, and dropping pairs one at a time depends on iteration order.

`MUST_KEEP` overrides all three. Those features are load-bearing at serving time -- the
baselines are closed forms over them, and dropping one silently turns a shipped baseline into
NaN. They are also exempt from the final redundancy assertion, because several of them are
deliberately collinear with each other and that is the point.

The cluster assignment is returned, not just the surviving list: permutation importance
permutes whole clusters with one shared index, so it needs to know what was collapsed into
what. Without it, importance is measured against a feature whose twin is still in the matrix
and every contribution reads as zero.
"""

import sys

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from src.constants.training_pipeline import SEED, SIGNALS
from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.utils.ml_utils.metric.regression_metric import baseline_required_columns

# |r| at or above which two features are treated as one feature.
REDUNDANCY_R: float = 0.95
# max(|pearson|, |spearman|) below which a feature carries no signal about the target.
MIN_TARGET_R: float = 0.02
# Rows sampled for the correlation matrices. The full panel is ~190k rows and a 180x180
# correlation matrix converges long before that.
SELECT_ROWS: int = 40_000
# Minimum surviving features. Below this the screen is misconfigured, not selective.
MIN_FEATURES: int = 20

MUST_KEEP: set = {
    "sbp_lag1",        # persistence baseline, and the anchor of every delta reparameterisation
    "sbp_z",           # personal z-score: the detector's primary control-chart input
    "sbp_ewm0.1",      # smoother, short
    "sbp_ewm0.3",      # the EWMA BASELINE reads this column. If selection drops it, a shipped
                       # EWMA baseline predicts NaN and the explanation gate reports all-zero
                       # contributions -- which is a promotion block, correctly, for a cause
                       # that has nothing to do with the model.
    "sbp_slope7",      # trend
    "sbp_mean7",       # level
    "sbp_base_mean",   # personal-mean baseline
    "sbp_base_std",    # scale for the z-score and the offset band
    "days_since_last",  # cadence; the model must be able to tell a 2-day gap from a 10-day one
    "is_weekend",
}

# The sbp_ewm0.3 entry above fixed this failure for ONE signal. The same trap exists for every
# other signal a baseline can ship for, and it sprang: the EWMA baseline shipped for dbp,
# dbp_ewm0.3 had been selected away as redundant with dbp_ewm0.1, and the dbp forecast vanished
# from every advisory without an error. Derived from BASELINE_SPECS rather than listed, so a
# new baseline or a new signal cannot reintroduce it.
MUST_KEEP |= baseline_required_columns(SIGNALS)


def _relevance(X: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """max(|pearson|, |spearman|) of each feature against any target, ignoring NaN."""
    best = pd.Series(0.0, index=X.columns, dtype=float)
    for t in targets.columns:
        y = targets[t]
        m = y.notna()
        if int(m.sum()) < 200:
            continue
        for method in ("pearson", "spearman"):
            r = X[m].corrwith(y[m], method=method).abs().fillna(0.0)
            best = np.maximum(best, r)
    return best


def select_features(F: pd.DataFrame, features: list, config, seed: int = SEED):
    """Return (selected, audit, cluster_of).

    Fitted on TRAIN rows only. Selection reads the target, so fitting it on val or test rows
    would leak the evaluation split into the feature set -- a subtle version of the same
    mistake the causal contract exists to prevent.
    """
    try:
        cols = [c for c in features if c in F.columns]
        tr = F[F.split == "train"] if "split" in F.columns else F
        if len(tr) > SELECT_ROWS:
            tr = tr.sample(SELECT_ROWS, random_state=seed)
        X = tr[cols].astype(float)

        target_names = [f"y_{s}_h{h}" for s in config.signals for h in config.horizons]
        target_names = [t for t in target_names if t in tr.columns]
        # Extra targets (the symptom heads, once they exist) join here so a feature that
        # only matters for a secondary head is not screened out by the SBP targets alone.
        for extra in getattr(config, "selection_extra_targets", ()) or ():
            if extra in tr.columns:
                target_names.append(extra)
        targets = tr[target_names]

        rows, dropped = [], {}

        # ---- 1. degenerate ---------------------------------------------------------
        nun = X.nunique(dropna=True)
        dens = X.notna().mean()
        degenerate = set(nun[nun < 2].index) | set(dens[dens < 0.05].index)
        degenerate -= MUST_KEEP
        for c in degenerate:
            dropped[c] = "degenerate"

        pool = [c for c in cols if c not in degenerate]

        # ---- 2. relevance ----------------------------------------------------------
        rel = _relevance(X[pool], targets)
        weak = set(rel[rel < MIN_TARGET_R].index) - MUST_KEEP
        for c in weak:
            dropped[c] = "irrelevant"
        pool = [c for c in pool if c not in weak]

        # ---- 3. redundancy ---------------------------------------------------------
        cluster_of = {}
        if len(pool) > 1:
            R = X[pool].corr().abs().fillna(0.0).to_numpy(dtype=float, copy=True)
            np.fill_diagonal(R, 1.0)
            D = np.clip(1.0 - R, 0.0, None)
            # squareform needs an exactly symmetric matrix with a zero diagonal; corr() can
            # return values differing in the last bit, which raises rather than warns.
            D = (D + D.T) / 2.0
            np.fill_diagonal(D, 0.0)
            link = hierarchy.linkage(squareform(D, checks=False), method="average")
            labels = hierarchy.fcluster(link, t=1.0 - REDUNDANCY_R, criterion="distance")
            cluster_of = {c: int(k) for c, k in zip(pool, labels)}

            miss = X[pool].isna().mean()
            keep = []
            for k in sorted(set(labels)):
                members = [c for c in pool if cluster_of[c] == k]
                forced = [c for c in members if c in MUST_KEEP]
                if forced:
                    # Every MUST_KEEP member survives, not just one: they are kept for
                    # serving reasons, and "this cluster already has a representative" is
                    # not a reason to drop a column a baseline reads.
                    keep.extend(forced)
                    for c in members:
                        if c not in forced:
                            dropped[c] = f"redundant (cluster {k})"
                    continue
                rep = sorted(members, key=lambda c: (-float(rel.get(c, 0.0)),
                                                     float(miss.get(c, 1.0)), c))[0]
                keep.append(rep)
                for c in members:
                    if c != rep:
                        dropped[c] = f"redundant (cluster {k})"
            selected = [c for c in pool if c in set(keep)]
        else:
            selected = list(pool)

        # ---- 3b. greedy pass -------------------------------------------------------
        # fcluster bounds the AVERAGE distance within a cluster, not the maximum between
        # representatives, so two representatives can still sit above REDUNDANCY_R. Ordered
        # MUST_KEEP first, then by descending relevance, so the survivor of a colliding pair
        # is the more informative one rather than whichever came first alphabetically.
        if len(selected) > 1:
            Rsel = X[selected].corr().abs().fillna(0.0)
            order = sorted(selected, key=lambda c: (c not in MUST_KEEP,
                                                    -float(rel.get(c, 0.0)), c))
            kept = []
            for c in order:
                if c in MUST_KEEP or all(float(Rsel.loc[c, k]) < REDUNDANCY_R for k in kept):
                    kept.append(c)
                else:
                    dropped[c] = "redundant (greedy pass)"
            selected = [c for c in selected if c in set(kept)]

        # ---- contract --------------------------------------------------------------
        assert len(selected) >= MIN_FEATURES, (
            f"selection kept only {len(selected)} features (floor {MIN_FEATURES}); "
            f"raise REDUNDANCY_R or lower MIN_TARGET_R")
        missing = MUST_KEEP - set(selected)
        assert not missing, f"a serving-critical feature was dropped: {sorted(missing)}"
        if len(selected) > 1:
            Rf = X[selected].corr().abs().fillna(0.0).to_numpy(dtype=float, copy=True)
            np.fill_diagonal(Rf, 0.0)
            free = np.array([c not in MUST_KEEP for c in selected])
            # Zero the forced-vs-forced block: those pairs are collinear on purpose.
            Rf[np.outer(~free, ~free)] = 0.0
            worst = float(Rf.max()) if Rf.size else 0.0
            assert worst < REDUNDANCY_R, (
                f"redundancy invariant violated: two freely-selected features correlate at "
                f"{worst:.4f} (limit {REDUNDANCY_R})")
        else:
            worst = 0.0

        for c in cols:
            rows.append(dict(feature=c, kept=c in set(selected),
                             reason=dropped.get(c, "kept"),
                             relevance=round(float(rel.get(c, np.nan)), 4)
                             if c in rel.index else np.nan,
                             cluster=cluster_of.get(c, -1),
                             density=round(float(dens.get(c, np.nan)), 4)))
        audit = pd.DataFrame(rows).sort_values(["kept", "relevance"], ascending=[False, False])

        logging.info("feature selection: %d -> %d (%d degenerate, %d irrelevant, "
                     "%d redundant across %d clusters); worst free |r| %.3f",
                     len(cols), len(selected), len(degenerate), len(weak),
                     len(cols) - len(selected) - len(degenerate) - len(weak),
                     len(set(cluster_of.values())), worst)
        return selected, audit, cluster_of
    except Exception as e:
        raise CustomException(e, sys)


def cluster_blocks(cluster_of: dict, selected: list) -> dict:
    """Representative -> every feature that collapsed into it, for block permutation.

    Permuting a representative alone understates its importance whenever a correlated twin
    survived elsewhere in the matrix; the model reads the twin and the prediction barely
    moves. The block is what has to be permuted together.
    """
    blocks = {}
    for rep in selected:
        k = cluster_of.get(rep)
        members = [c for c, kk in cluster_of.items() if kk == k] if k is not None else [rep]
        blocks[rep] = sorted(set(members) | {rep})
    return blocks
