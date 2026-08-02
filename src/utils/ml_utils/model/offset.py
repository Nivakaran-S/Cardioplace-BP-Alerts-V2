"""Model 2 -- the personalisation offset.

Port of notebook section 7. A capped shrinkage blend of a patient's own band and a
demographic cohort prior. The caps are governance inputs: they are never searched, and
the emergency floor is asserted on every threshold this module produces.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.logging.logger import logging


def cohort_key(age: float, is_male: int) -> str:
    band = "<50" if age < 50 else "50-64" if age < 65 else "65-74" if age < 75 else "75+"
    return f"{band}|{'M' if is_male == 1 else 'F'}"


class OffsetModel:
    """Capped shrinkage blend of a patient's own band and a demographic cohort prior."""

    def __init__(self, config, warm: int = 48, k: float = 30.0, q: float = 0.90,
                 cohort_prior: dict = None, global_prior: float = 140.0):
        self.config = config
        self.warm = warm
        self.k = k
        self.q = q
        self.cohort_prior = cohort_prior if cohort_prior is not None else {}
        self.global_prior = global_prior

    def fit(self, F: pd.DataFrame, prior_pool: str = "fit") -> "OffsetModel":
        head = F[F.step < self.warm].copy()
        head["ck"] = [cohort_key(a, m) for a, m in zip(head.age.fillna(65), head.is_male)]
        pool = head[head.patient_split == prior_pool] if prior_pool else head
        if pool.empty:
            pool = head
        self.cohort_prior = pool.groupby("ck").sbp.quantile(self.q).to_dict()
        self.global_prior = float(pool.sbp.quantile(self.q))
        return self

    def threshold_for(self, sbp_history: pd.Series, age: float, is_male: int) -> dict:
        """Personalised threshold from the patient's warm-up window.

        `n` and `personal` must come from the SAME readings. They did not: `n` counted the
        whole history while `personal` was the quantile of `head(warm)` alone, so a patient
        submitting 372 readings got w = 0.925 on an estimate built from 72 of them -- the
        weight asserted five times the evidence the estimate actually used, and the extra
        readings moved the threshold by 1.1 mmHg while the patient's own band had risen by 30.

        It is also a train/serve skew, which is what makes it a defect rather than a
        preference. `transform` hands this method `F[F.step < warm]`, so at training `n` is
        capped at `warm` and w can never exceed warm/(warm+k) = 0.706. Serving passed the full
        history and ran at weights the blend was never fitted or validated at. Capping `n` to
        the same window makes serving reproduce training instead of extrapolating past it.
        """
        c = self.config
        h = pd.Series(sbp_history, dtype=float).dropna()
        head = h.head(self.warm)
        n = int(len(head))
        ck = cohort_key(age, is_male)
        cohort = float(self.cohort_prior.get(ck, self.global_prior))
        if n >= 5:
            personal = float(head.quantile(self.q))
            w = n / (n + self.k)
        else:
            personal, w = cohort, 0.0
        blend = w * personal + (1 - w) * cohort
        offset = float(np.clip(blend - c.population_threshold_mmHg,
                               -c.offset_cap_tighten, c.offset_cap_loosen))
        thr = c.population_threshold_mmHg + offset
        assert thr < c.emergency_floor_mmHg, "offset breached the emergency floor"
        return dict(threshold=round(thr, 1), offset=round(offset, 1), cohort_key=ck,
                    cohort=round(cohort, 1), personal=round(personal, 1), n_warm=n,
                    shrinkage_w=round(w, 3),
                    capped=abs(blend - c.population_threshold_mmHg - offset) > 1e-6)

    def transform(self, F: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for sid, g in F[F.step < self.warm].groupby("series_id", sort=False):
            if int(g.sbp.notna().sum()) < 5:
                continue
            r = self.threshold_for(g.sbp, float(g.age.fillna(65).iloc[0]),
                                   int(g.is_male.iloc[0]))
            rows.append(dict(series_id=sid, patient_split=g.patient_split.iloc[0],
                             is_male=int(g.is_male.iloc[0]), is_dm=int(g.is_dm.iloc[0]),
                             age=float(g.age.iloc[0]) if pd.notna(g.age.iloc[0]) else np.nan,
                             **r))
        return pd.DataFrame(rows)


def search_offset(F: pd.DataFrame, config):
    """Search warm/k/q on `fit` patients only; report on `holdout`.

    The caps are absent from the grid on purpose: a safety cap that moves to fit the data
    is not a safety cap.
    """
    assert not ({"offset_cap_loosen", "offset_cap_tighten"} & {"warm", "k", "q"})
    scores = []
    for w in config.offset_search_warm:
        for k in config.offset_search_k:
            for q in config.offset_search_q:
                om = OffsetModel(config, warm=w, k=k, q=q).fit(F)
                band = om.transform(F)
                if band.empty:
                    continue
                tl = (F[F.step >= w].groupby("series_id").sbp
                      .quantile(0.90).rename("actual").reset_index())
                MM = band.merge(tl, on="series_id").dropna(subset=["actual"])
                fit_m = MM[MM.patient_split == "fit"]
                if len(fit_m) < 20:
                    continue
                scores.append(dict(warm=w, k=k, q=q, n=len(fit_m),
                                   mae_fit=mean_absolute_error(fit_m.actual, fit_m.threshold)))
    S = pd.DataFrame(scores).sort_values("mae_fit") if scores else pd.DataFrame()
    if S.empty:
        logging.warning("offset search produced no scorable configuration; using defaults")
        return OffsetModel(config, warm=config.offset_warm, k=config.offset_k,
                           q=config.offset_q).fit(F), S
    best = OffsetModel(config, warm=int(S.iloc[0].warm), k=float(S.iloc[0].k),
                       q=float(S.iloc[0].q)).fit(F)
    logging.info("offset selected on fit patients: warm=%s k=%s q=%s (control was 48/30/0.90)",
                 best.warm, best.k, best.q)
    return best, S


def observed_band(F: pd.DataFrame, warm: int, q: float = 0.90) -> pd.DataFrame:
    """The band each patient actually occupies after the warm-up window."""
    return (F[F.step >= warm].groupby("series_id").sbp
            .quantile(q).rename("actual").reset_index())
