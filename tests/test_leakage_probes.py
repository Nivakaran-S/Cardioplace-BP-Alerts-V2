"""The leakage audit must pass on clean features AND fail on deliberately leaky ones.

Same discipline as the safety gates: a probe that cannot detect the corruption it exists for
reports PASS forever and is read as evidence. The future-corruption probe is the one that
matters most -- it is the only check in the pipeline that can catch a feature reading forward,
and it has no structural analogue.
"""

import sys

import numpy as np
import pandas as pd

from src.entity.config_entity import ModelTrainerConfig, TrainingPipelineConfig
from src.utils.ml_utils.feature.cadence import attach_cadence
from src.utils.ml_utils.feature.causal_features import CausalFeatureBuilder, leakage_audit

FAILS = []


def chk(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"  <- {extra}"))
    if not cond:
        FAILS.append(name)


def panel(n_pat=8, n=90, seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for p in range(n_pat):
        ts = pd.Timestamp("2024-01-01") + pd.to_timedelta(
            np.cumsum(rng.choice([2, 2, 3], n)), "D")
        walk = np.cumsum(rng.normal(0, 3, n))
        out.append(pd.DataFrame(dict(
            patient_id=f"p{p}", series_id=f"p{p}", ts=ts,
            sbp=(140 + walk + rng.normal(0, 6, n)).round(),
            dbp=(80 + walk * .4 + rng.normal(0, 5, n)).round(),
            idwg=rng.normal(2.4, .8, n), weight=rng.normal(70, 2, n), dryweight=68.0,
            uf_total=rng.normal(2.2, .6, n), sbp_post=rng.normal(130, 12, n).round(),
            sbp_min=rng.normal(120, 12, n).round(), temperature=36.5, n_meas=2,
            age=60.0 + p, is_male=p % 2, is_dm=0, DM=0.0, is_weekend=0)))
    d = pd.concat(out, ignore_index=True)
    d["sbp_drop"] = d.sbp - d.sbp_min
    d = attach_cadence(d, by="series_id")
    d["step"] = d.groupby("series_id").cumcount()
    d["split"] = np.where(d.step > 71, "test", np.where(d.step > 55, "val", "train"))
    d["patient_split"] = np.where(d.patient_id.isin(["p0", "p1"]), "holdout", "fit")
    return d


def run():
    cfg = ModelTrainerConfig(TrainingPipelineConfig())
    P = panel()
    fb = CausalFeatureBuilder(cfg)
    F = fb.transform(P)
    feats = fb.feature_names_

    print("\n--- clean features: every probe must PASS ---")
    a = leakage_audit(F, feats, cfg, fb=fb)
    for r in a.itertuples():
        print(f"  {r.status:4s}  {r.probe}")
    chk("no probe FAILs on clean features", int((a.status == "FAIL").sum()) == 0,
        list(a[a.status == "FAIL"].probe))
    for expect in ("corrupting the future leaves past features unchanged",
                   "a shuffled-target model does not beat persistence",
                   "latent generator variables are not features"):
        chk(f"probe present: {expect}", expect in set(a.probe))
    chk("target-alignment probe runs for every horizon",
        sum(1 for p in a.probe if "steps later" in p) == len(cfg.horizons))

    print("\n--- deliberately leaky features: the probes must BITE ---")

    # 1. A feature that reads the future directly. This is the failure the whole causal
    #    contract exists to prevent, and only the future-corruption probe can see it.
    Fl = F.copy()
    Fl["sbp_peek"] = P.sort_values(["series_id", "step"]).groupby("series_id").sbp.shift(-1).values

    class Peeking(CausalFeatureBuilder):
        def _one_series(self, g):
            out = super()._one_series(g)
            out["sbp_peek"] = out["sbp"].shift(-1)      # reads t+1
            return out

    fb2 = Peeking(cfg)
    F2 = fb2.transform(P)
    a2 = leakage_audit(F2, fb2.feature_names_, cfg, fb=fb2)
    hit = a2[a2.probe == "corrupting the future leaves past features unchanged"]
    chk("future-peeking feature is caught",
        len(hit) and hit.iloc[0].status == "FAIL",
        hit.iloc[0].detail if len(hit) else "probe missing")

    # 2. A target that is off by one. Every metric would look self-consistent.
    F3 = F.copy()
    tgt = f"y_sbp_h{cfg.horizons[0]}"
    F3[tgt] = F3.groupby("series_id")[tgt].shift(-1)
    a3 = leakage_audit(F3, feats, cfg)
    hit3 = a3[a3.probe.str.contains("steps later")]
    chk("misaligned target is caught", (hit3.status == "FAIL").any(),
        hit3[["probe", "status"]].to_string(index=False))

    # 3. A latent generator variable reaching the feature list.
    F4 = F.copy()
    F4["frailty"] = np.random.default_rng(0).normal(0, 1, len(F4))
    a4 = leakage_audit(F4, feats + ["frailty"], cfg)
    hit4 = a4[a4.probe == "latent generator variables are not features"]
    chk("latent generator variable is caught",
        len(hit4) and hit4.iloc[0].status == "FAIL",
        hit4.iloc[0].detail if len(hit4) else "probe missing")

    # 4. A feature that is a copy of the current reading.
    F5 = F.copy()
    F5["sbp_copy"] = F5["sbp"]
    a5 = leakage_audit(F5, feats + ["sbp_copy"], cfg)
    hit5 = a5[a5.probe == "no feature is a copy of the current reading"]
    chk("straight copy of the target is caught",
        len(hit5) and hit5.iloc[0].status == "FAIL",
        hit5.iloc[0].detail if len(hit5) else "probe missing")

    print("\n" + ("ALL LEAKAGE PROBE TESTS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: " + ", ".join(FAILS)))
    return 1 if FAILS else 0


def test_leakage_probes():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
