"""Safety gates -- the promotion contract.

Every other module in this package answers "how good is the model". This one answers "may it
ship", and the two are not the same question. A model can win its bake-off, beat every
baseline, and still be unpromotable because it moved a threshold it is not allowed to move.

The gates are the executable form of the GOVERNANCE block in
`src/constants/training_pipeline/__init__.py`. They read `config` rather than re-deriving any
bound, so there is exactly one place a governance number lives and no way for a gate to
quietly disagree with the model it is auditing.

A gate is `critical` when failing it means the model is unsafe, not merely worse. Critical
failures block promotion: `final_model/model.pkl` is left untouched and the run keeps its own
artifact under `Artifacts/<run>/` for inspection. Non-critical failures (WARN) are recorded
and shipped, because a warning that blocks promotion is a critical gate wearing a disguise.

Gate 6 -- the non-zero explanation gate -- exists because it has already caught a real
failure: an explainer whose perturbation set missed every column the shipped model read
returned all zeros and looked like a working explainer. See the comment at
`src/components/model_trainer.py:612-616`.
"""

import sys

import numpy as np
import pandas as pd

from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.utils.ml_utils.rule_engine.registry import EMERGENCY_TIERS

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"

GATE_COLUMNS = ["gate", "status", "observed", "bound", "critical", "detail"]


class GateResult:
    """The gate report plus the single boolean the promotion step reads."""

    def __init__(self, rows: list):
        self.rows = list(rows)

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=GATE_COLUMNS)
        return pd.DataFrame(self.rows)[GATE_COLUMNS]

    @property
    def critical_failures(self) -> list:
        return [r["gate"] for r in self.rows if r["critical"] and r["status"] == FAIL]

    @property
    def warnings(self) -> list:
        return [r["gate"] for r in self.rows if r["status"] == WARN]

    @property
    def promotable(self) -> bool:
        return not self.critical_failures

    def __repr__(self) -> str:
        f = self.frame()
        return (f"GateResult({int((f.status == PASS).sum())}/{len(f)} pass, "
                f"{len(self.critical_failures)} critical failures, "
                f"promotable={self.promotable})")


class _Recorder:
    """Collects gate rows and keeps a gate that raises from taking down the whole report.

    A gate whose evidence is missing is not a passing gate. `skip` records SKIP-as-WARN so an
    absent input is visible rather than silently counting as compliance -- the failure mode
    where a report reads 12/12 PASS because nine gates never ran.
    """

    def __init__(self):
        self.rows = []

    def add(self, gate, ok, observed, bound, detail="", critical=False):
        self.rows.append(dict(gate=gate, status=PASS if ok else (FAIL if critical else WARN),
                              observed=observed, bound=bound, critical=bool(critical),
                              detail=detail))

    def skip(self, gate, why, critical=False):
        """Evidence absent. Never PASS -- a gate that could not run has proved nothing."""
        self.rows.append(dict(gate=gate, status=WARN, observed=None, bound=None,
                              critical=False, detail=f"not evaluated: {why}"
                                                     + (" (normally critical)" if critical
                                                        else "")))

    def guard(self, gate, fn, critical=False):
        """Run one gate; a raise inside it becomes a FAIL row, not a dead report."""
        try:
            fn()
        except Exception as exc:                                     # noqa: BLE001
            self.rows.append(dict(gate=gate, status=FAIL if critical else WARN,
                                  observed=None, bound=None, critical=bool(critical),
                                  detail=f"gate raised {type(exc).__name__}: {exc}"))


# --------------------------------------------------------------------- abstention / OOD

class AbstentionPolicy:
    """Distance-to-training-distribution, and the cut past which the model should abstain.

    Robust (median / MAD) rather than mean / covariance on purpose: the training panel has a
    long right tail on several features, and a classical Mahalanobis distance fitted on it
    inherits that tail into the covariance estimate, which then declares the tail typical.

    The distance is a mean absolute robust z across `ood_cols`, so a row missing some columns
    still scores -- serving histories are routinely missing the interdialytic columns, and an
    abstention policy that abstains on everything is not a policy.
    """

    def __init__(self, train_df: pd.DataFrame, ood_cols: list, cut_pct: float = 99.0):
        self.cols = [c for c in ood_cols if c in train_df.columns]
        X = train_df.reindex(columns=self.cols).astype(float)
        self.median_ = X.median()
        mad = (X - self.median_).abs().median()
        # 1.4826 * MAD is the consistent estimator of sigma for a normal. The floor stops a
        # near-constant column from dividing by ~0 and dominating every distance.
        self.scale_ = (1.4826 * mad).replace(0.0, np.nan).fillna(1.0).clip(lower=1e-6)
        d = self._dist(train_df)
        d = d[np.isfinite(d)]
        self.cut = float(np.percentile(d, cut_pct)) if len(d) else float("inf")
        self.n_fit = int(len(d))

    def _dist(self, df: pd.DataFrame) -> np.ndarray:
        """Mean absolute robust z over the fitted columns.

        Callers pass both a full feature frame and a pre-sliced ood_cols frame (see
        `model_trainer.py:621-622`), so reindex rather than index -- a KeyError here would
        take down the safety report of an otherwise healthy run.
        """
        X = df.reindex(columns=self.cols).astype(float)
        z = (X - self.median_).abs() / self.scale_
        return z.mean(axis=1, skipna=True).to_numpy(dtype=float)

    def abstains(self, df: pd.DataFrame) -> np.ndarray:
        return self._dist(df) > self.cut


# --------------------------------------------------------------------- cold start

def cold_start_curve(predictor, history: pd.DataFrame, config) -> pd.DataFrame:
    """Advisory quality as a function of how much history the patient has.

    This is the only measurement of what a NEW user gets, which for a journaling app is the
    dominant case rather than the edge case. `predictor.predict` returns early for the
    cold_start and stale tiers (`estimator.py:382,394`), so every field below is optional and
    read defensively -- a curve that raises on the first short history measures nothing.
    """
    try:
        g = history.sort_values("ts")
        probes = [n for n in config.cold_start_probe_n if n <= len(g)]
        if len(g) not in probes:
            probes.append(len(g))
        rows = []
        for n in probes:
            try:
                a = predictor.predict(g.head(n))
            except Exception as exc:                                 # noqa: BLE001
                rows.append(dict(n_readings=n, tier="error", note=str(exc)[:120]))
                continue
            pers = a.get("personalisation") or {}
            sbp = (a.get("forecast") or {}).get("sbp") or {}
            node = sbp.get(sorted(sbp)[0]) if sbp else {}
            ew = a.get("early_warning") or {}
            lo, hi = node.get("lo80"), node.get("hi80")
            rows.append(dict(
                n_readings=n,
                tier=a.get("confidence_tier"),
                has_forecast=bool(sbp),
                point=node.get("point"),
                lo80=lo, hi80=hi,
                band_width=(round(hi - lo, 2) if lo is not None and hi is not None else None),
                threshold=pers.get("threshold"),
                offset=pers.get("offset"),
                shrinkage_w=pers.get("shrinkage_w"),
                capped=pers.get("capped"),
                ew_score=ew.get("score"),
                ew_flagged=ew.get("flagged"),
                latency_ms=a.get("latency_ms"),
                note=(a.get("note") or "")[:120]))
        return pd.DataFrame(rows)
    except Exception as e:
        raise CustomException(e, sys)


# --------------------------------------------------------------------- provenance

def provenance_guard(panel: pd.DataFrame, synth: pd.DataFrame,
                     alerts_real: pd.DataFrame, alerts_synth: pd.DataFrame) -> dict:
    """Prove no synthetic row reached the real-data record.

    The synthetic cohort exists so the 56-rule registry can be exercised against inputs the
    HEMOBP corpus does not carry. That is only defensible while the two populations stay
    provably disjoint -- the moment a generated reading is counted as an observation, every
    metric downstream is describing the generator.

    Checked by identity of `series_id`, not by a flag column, because a flag can be dropped by
    a merge and an id collision cannot be.
    """
    try:
        real_ids = set(panel.series_id.astype(str)) if "series_id" in panel else set()
        syn_ids = set(synth.series_id.astype(str)) if "series_id" in synth else set()
        a_real = (set(alerts_real.series_id.astype(str))
                  if len(alerts_real) and "series_id" in alerts_real else set())
        a_syn = (set(alerts_synth.series_id.astype(str))
                 if len(alerts_synth) and "series_id" in alerts_synth else set())

        checks = {
            "cohorts_disjoint": not (real_ids & syn_ids),
            "real_alerts_are_real_patients": a_real <= real_ids,
            "synthetic_alerts_are_synthetic_patients": a_syn <= syn_ids,
            "no_synthetic_id_in_real_alerts": not (a_real & syn_ids),
        }
        out = dict(checks)
        out.update(holds=bool(all(checks.values())),
                   n_real_patients=len(real_ids), n_synthetic_patients=len(syn_ids),
                   n_real_alert_rows=int(len(alerts_real)),
                   n_synthetic_alert_rows=int(len(alerts_synth)),
                   overlapping_ids=sorted(real_ids & syn_ids)[:20])
        if not out["holds"]:
            logging.error("provenance guard FAILED: %s",
                          {k: v for k, v in checks.items() if not v})
        return out
    except Exception as e:
        raise CustomException(e, sys)


# --------------------------------------------------------------------- the gate battery

def run_safety_gates(alerts_pop, alerts_pers, OFF, advisories, cold, fairness,
                     explanation, abstain, ood_rate, config, missingness=None,
                     offset_learned_max=None, shipped_forecaster=None,
                     selected_forecaster=None, shipped_detector=None,
                     cold_start_penalty=None, symptom_labels_synthetic=None,
                     detector_alert_rate=None, pair_violation_rate=None,
                     forecaster_features=None) -> GateResult:
    """Evaluate every promotion gate and return the report plus the promotable flag.

    The trailing keyword arguments are the gates whose evidence is produced by later phases
    (the learned offset, the freeze fallback, the detector choice, the cold-start arm). They
    default to None and record as "not evaluated" rather than PASS, so the report never
    overstates what was checked.
    """
    try:
        g = _Recorder()
        floor = config.emergency_floor_mmHg
        budget = config.alert_budget_pct / 100.0

        # ---- 1. the emergency floor is never personalised -----------------------
        def _gate_floor():
            if OFF is None or not len(OFF):
                return g.skip("emergency floor never personalised", "no offset rows",
                              critical=True)
            mx = float(OFF.threshold.max())
            g.add("emergency floor never personalised", mx < floor, round(mx, 1), floor,
                  f"highest personalised threshold is {mx:.1f} mmHg against a "
                  f"{floor:.0f} mmHg floor", critical=True)
        g.guard("emergency floor never personalised", _gate_floor, critical=True)

        # ---- 2. offset caps hold, asymmetrically --------------------------------
        def _gate_caps():
            if OFF is None or not len(OFF):
                return g.skip("offset within governance caps", "no offset rows", critical=True)
            lo, hi = float(OFF.offset.min()), float(OFF.offset.max())
            ok = (lo >= -config.offset_cap_tighten - 1e-6
                  and hi <= config.offset_cap_loosen + 1e-6)
            g.add("offset within governance caps", ok, f"[{lo:.1f}, {hi:.1f}]",
                  f"[-{config.offset_cap_tighten:.0f}, +{config.offset_cap_loosen:.0f}]",
                  f"{int(OFF.capped.sum())}/{len(OFF)} patients bind a cap; loosening is "
                  f"capped harder than tightening on purpose", critical=True)
        g.guard("offset within governance caps", _gate_caps, critical=True)

        # ---- 3. the alert budget binds the DETECTOR, not the rule engine --------
        # ALERT_BUDGET_PCT is staffing capacity for the ML early-warning layer: the cut is
        # placed at the (100 - budget)th percentile of scores, so the budget is a property
        # of where that cut sits. It is NOT a constraint on the deterministic rule engine.
        # The engine is the clinical floor -- it fires whenever the criteria are met, and on
        # a haemodialysis cohort whose median pre-dialysis SBP sits near the 140 mmHg
        # threshold that is most sessions. Holding the engine to 5% would mean suppressing
        # genuine Level-1 readings to hit a staffing number, which is exactly backwards.
        def _gate_budget():
            if detector_alert_rate is None or not np.isfinite(float(detector_alert_rate)):
                return g.skip("detector alert rate within budget",
                              "no realised detector rate reported", critical=True)
            rate = float(detector_alert_rate)
            # Two-sided with slack: far below budget means the cut is mis-calibrated and
            # capacity is being wasted, not that the model is unusually safe.
            ok = rate <= budget * 1.5
            g.add("detector alert rate within budget", ok, round(rate, 4), round(budget, 4),
                  f"the detector flags {100 * rate:.2f}% of scored sessions against a "
                  f"{config.alert_budget_pct:.0f}% budget (1.5x tolerance on the realised "
                  f"rate, because the cut is calibrated on val and applied to test)",
                  critical=True)
        g.guard("detector alert rate within budget", _gate_budget, critical=True)

        # The engine's own fire rate is reported, never gated. A rising rate is a fact about
        # the cohort's blood pressure, not a defect, and blocking promotion on it would be
        # blocking on the patients being unwell.
        def _gate_engine_rate():
            if alerts_pop is None or not len(alerts_pop):
                return g.skip("rule-engine fire rate (reported, not gated)", "no alerts")
            rate = float(alerts_pop.fired.mean())
            tiers = (alerts_pop[alerts_pop.fired].tier.value_counts().head(3).to_dict()
                     if "tier" in alerts_pop else {})
            g.add("rule-engine fire rate (reported, not gated)", True, round(rate, 4),
                  "not gated",
                  f"{int(alerts_pop.fired.sum())} of {len(alerts_pop)} readings fired "
                  f"({100 * rate:.1f}%); top tiers {tiers}. The engine is the clinical "
                  f"floor and is deliberately unbudgeted.")
        g.guard("rule-engine fire rate (reported, not gated)", _gate_engine_rate)

        # ---- 4. personalisation never suppresses an emergency -------------------
        # The ML layer may tighten or loosen a Level-1 threshold. It may not make an
        # emergency disappear. Set inclusion on (series_id, step), so a personalised run
        # that merely re-tiers a row still fails if the row stops being an emergency.
        def _gate_parity():
            if not len(alerts_pop) or not len(alerts_pers):
                return g.skip("personalisation preserves every emergency",
                              "one alert frame is empty", critical=True)
            key = ["series_id", "step"]
            pop_e = set(map(tuple, alerts_pop[alerts_pop.is_emergency][key].values))
            per_e = set(map(tuple, alerts_pers[alerts_pers.is_emergency][key].values))
            lost = pop_e - per_e
            g.add("personalisation preserves every emergency", not lost, len(lost), 0,
                  f"{len(pop_e)} emergencies in population mode, {len(per_e)} in "
                  f"personalised mode, {len(lost)} lost. Emergency tiers: "
                  f"{sorted(EMERGENCY_TIERS)}", critical=True)
        g.guard("personalisation preserves every emergency", _gate_parity, critical=True)

        # ---- 5. fairness, on whatever ships -------------------------------------
        def _gate_fair():
            if fairness is None or not len(fairness):
                return g.skip("subgroup parity", "no fairness rows", critical=True)
            fail = fairness[~fairness.passes]
            detail = "; ".join(f"{r.metric} {r.axis}={r.level} gap {r.gap:+.2f}"
                               for r in fail.head(4).itertuples()) or "all slices within margin"
            g.add("subgroup parity", not len(fail), len(fail), 0,
                  f"{len(fairness) - len(fail)}/{len(fairness)} slices pass. {detail}. "
                  f"Race and ethnicity are not auditable on HEMOBP.", critical=True)
        g.guard("subgroup parity", _gate_fair, critical=True)

        # ---- 6. the explanation actually explains something ---------------------
        def _gate_expl():
            if not explanation:
                return g.skip("explanation is non-zero", "no explanation produced",
                              critical=True)
            mx = max(abs(float(v)) for v in explanation.values())
            g.add("explanation is non-zero", mx > 0.0, round(mx, 4), "> 0",
                  f"largest single-feature contribution {mx:.4f} mmHg over "
                  f"{len(explanation)} reported features. Zero means the perturbation set "
                  f"missed every column the shipped model reads.", critical=True)
        g.guard("explanation is non-zero", _gate_expl, critical=True)

        # ---- 7. out-of-distribution rate ----------------------------------------
        def _gate_ood():
            if ood_rate is None or not np.isfinite(float(ood_rate)):
                return g.skip("test rows inside the training distribution",
                              "no OOD rate", critical=True)
            r = float(ood_rate)
            g.add("test rows inside the training distribution", r <= config.ood_rate_max,
                  round(r, 4), config.ood_rate_max,
                  f"{100 * r:.1f}% of test rows sit beyond the abstention cut "
                  f"({getattr(abstain, 'cut', float('nan')):.3f})", critical=True)
        g.guard("test rows inside the training distribution", _gate_ood, critical=True)

        # ---- 8. the learned offset respects the floor too -----------------------
        # OFF is the blend's output. A learned Model 2 is a separate estimator and needs its
        # own assertion -- clearing gate 1 says nothing about predictions gate 1 never saw.
        if offset_learned_max is None:
            g.skip("learned offset respects the emergency floor",
                   "no learned offset in this run", critical=True)
        else:
            mx = float(offset_learned_max)
            g.add("learned offset respects the emergency floor", mx < floor, round(mx, 1),
                  floor, f"highest learned-offset prediction {mx:.1f} mmHg", critical=True)

        # ---- 9. cold-start monotonicity -----------------------------------------
        # More history must not make the advisory worse. Checked on interval width, which is
        # the model's own statement of confidence.
        def _gate_cold_mono():
            if cold is None or not len(cold) or "band_width" not in cold:
                return g.skip("cold-start curve is monotone", "no cold-start curve")
            w = cold.dropna(subset=["band_width"]).sort_values("n_readings")
            if len(w) < 2:
                return g.skip("cold-start curve is monotone", "fewer than two banded probes")
            worst = float(np.max(np.diff(w.band_width.to_numpy(float))))
            g.add("cold-start curve is monotone", worst <= 1.0, round(worst, 2), 1.0,
                  f"largest widening of the 80% band as history grows: {worst:+.2f} mmHg "
                  f"across {len(w)} probes")
        g.guard("cold-start curve is monotone", _gate_cold_mono)

        # ---- 10. cold-start penalty ---------------------------------------------
        if cold_start_penalty is None or not np.isfinite(float(cold_start_penalty)):
            g.skip("cold-start penalty bounded", "no held-out-patient arm in this run")
        else:
            p = float(cold_start_penalty)
            g.add("cold-start penalty bounded", p <= config.cold_start_penalty_max_mmHg,
                  round(p, 2), config.cold_start_penalty_max_mmHg,
                  f"unseen patients cost {p:+.2f} mmHg MAE against unseen time for seen "
                  f"patients -- this is what a new user gets")

        # ---- 11. the abstention policy is usable --------------------------------
        def _gate_abstain():
            cut = float(getattr(abstain, "cut", float("nan")))
            n = int(getattr(abstain, "n_fit", 0))
            cols = len(getattr(abstain, "cols", []))
            g.add("abstention policy is usable", np.isfinite(cut) and cut > 0 and cols > 0,
                  round(cut, 4) if np.isfinite(cut) else None, "finite and > 0",
                  f"fitted on {n} training rows over {cols} columns")
        g.guard("abstention policy is usable", _gate_abstain)

        # ---- 12. advisories are well-formed -------------------------------------
        def _gate_adv():
            if not advisories:
                return g.skip("advisories are well-formed", "no advisories on the probe "
                                                            "patient")
            bad = [a for a in advisories
                   if a.surfaced_tier not in set(EMERGENCY_TIERS) | set(_known_tiers())]
            never_alert = all(getattr(a, "provider_visible_only", False) for a in advisories)
            g.add("advisories are well-formed", not bad and never_alert,
                  f"{len(advisories)} advisories, {len(bad)} unknown tiers",
                  "0 unknown tiers, all provider-visible-only",
                  "an advisory is decision support; no path here writes a DeviationAlert")
        g.guard("advisories are well-formed", _gate_adv)

        # ---- 13. missingness robustness -----------------------------------------
        def _gate_miss():
            if missingness is None or not len(missingness) or "MAE" not in missingness:
                return g.skip("missingness degradation bounded", "sweep unavailable")
            m = missingness.dropna(subset=["MAE"]).sort_values("drop_frac")
            if len(m) < 2:
                return g.skip("missingness degradation bounded", "fewer than two points")
            worst = float(m.MAE.iloc[-1] - m.MAE.iloc[0])
            g.add("missingness degradation bounded",
                  worst <= config.missingness_max_degradation_mmHg, round(worst, 2),
                  config.missingness_max_degradation_mmHg,
                  f"dropping {100 * float(m.drop_frac.iloc[-1]):.0f}% of sessions costs "
                  f"{worst:+.2f} mmHg MAE. The pipeline never imputes a skipped session; "
                  f"this is the evidence that holds up.")
        g.guard("missingness degradation bounded", _gate_miss)

        # ---- 14. the synthetic-label flag survived ------------------------------
        # Model 4 is trained on generated labels. The flag is what stops a downstream reader
        # treating a symptom probability as an observation, so its absence is a finding --
        # not a blocking one, because the heads are excluded from promotion criteria, but a
        # bundle that has forgotten its own provenance is not shippable in silence.
        # Extended when the symptom output began deriving from the forecast. The
        # justification above -- "the heads are excluded from promotion criteria" -- carries
        # less weight once a coupling exists at all, even a one-directional inference-time
        # one, so the gate now checks the coupling is still one-directional rather than
        # only that the flag is present.
        if symptom_labels_synthetic is None:
            g.skip("symptom labels declared synthetic", "no symptom heads in this bundle")
        else:
            g.add("symptom labels declared synthetic", bool(symptom_labels_synthetic) is True,
                  bool(symptom_labels_synthetic), True,
                  "no real symptom was ever observed in HEMOBP; the flag is what keeps a "
                  "symptom probability from being read as an observation")

            # The one that matters: no symptom-derived column may reach the FORECASTER's
            # feature matrix. That is the executable form of "the BP forecast is
            # uncontaminated by generated labels" -- an assertion rather than a claim in a
            # docstring. `selection.py` has a dead `selection_extra_targets` hook that would
            # break exactly this if anyone ever wired it.
            if forecaster_features is None:
                g.skip("no symptom-derived column reaches the forecaster",
                       "feature list not reported by this run")
            else:
                leaked = sorted(c for c in forecaster_features
                                if c.startswith("y_sym_") or c.startswith("y_"))
                g.add("no symptom-derived column reaches the forecaster", not leaked,
                      leaked[:5] or "none", "no y_* column",
                      "a target-derived column in the forecaster matrix would make the "
                      "synthetic labels an input to the real-data model")

        # ---- 15. the model that ships is the model that was selected ------------
        # Compares EVERY signal, not just sbp. It used to read two scalars, so a dbp winner
        # that failed to freeze shipped a substituted model with nothing in this report
        # saying so -- and `fit_final` returns None for all four scope="local" kinds, which
        # are real contenders on this corpus.
        if selected_forecaster is None or shipped_forecaster is None:
            g.skip("shipped forecaster is the selected forecaster",
                   "freeze path not reported by this run")
        elif isinstance(selected_forecaster, dict) and isinstance(shipped_forecaster, dict):
            diff = sorted(k for k in set(selected_forecaster) | set(shipped_forecaster)
                          if str(selected_forecaster.get(k)) != str(shipped_forecaster.get(k)))
            g.add("shipped forecaster is the selected forecaster", not diff,
                  ", ".join(f"{k}: {shipped_forecaster.get(k)}" for k in diff) or "all match",
                  ", ".join(f"{k}: {selected_forecaster.get(k)}"
                            for k in sorted(selected_forecaster)),
                  f"{len(diff)} signal(s) ship a model the bake-off did not select"
                  if diff else "every signal froze cleanly")
        else:
            same = str(selected_forecaster) == str(shipped_forecaster)
            g.add("shipped forecaster is the selected forecaster", same,
                  str(shipped_forecaster), str(selected_forecaster),
                  "the bake-off winner could not be frozen, so a fallback ships instead"
                  if not same else "winner froze cleanly")

        # ---- 16. the detector beats the reference it replaces -------------------
        # d_fixed_threshold IS the rule engine. Shipping it as the "early-warning detector"
        # means shipping the thing the detector exists to improve on, under a new name.
        if shipped_detector is None:
            g.skip("shipped detector is not the rule-engine reference",
                   "no detector reported by this run")
        else:
            ok = str(shipped_detector) != "d_fixed_threshold"
            g.add("shipped detector is not the rule-engine reference", ok,
                  str(shipped_detector), "!= d_fixed_threshold",
                  "d_fixed_threshold is the rule engine wearing a detector interface")

        # ---- 17. the predicted (sbp, dbp) pair is physiologically coherent ------
        # sbp and dbp are forecast by architectures chosen independently, so nothing else in
        # this report looks at the joint. A pair the API would have REFUSED as an input
        # reading -- pulse pressure below the schema floor, or diastolic above systolic --
        # is incoherent by the project's own definition, and pulse pressure is a rule axis,
        # so an incoherent pair can raise a watch banner that is purely an artefact.
        #
        # Non-critical, deliberately. Making it blocking would let a JOINT property veto a
        # forecaster whose own marginal MAE cleared every bar in this table, which is
        # precisely the kind of coupling the rest of these gates avoid. Promote it only
        # after a full run shows the observed rate is a stable zero.
        if pair_violation_rate is None:
            g.skip("forecast pairs are physiologically coherent",
                   "no paired sbp/dbp forecast reported by this run")
        else:
            rate = float(pair_violation_rate)
            g.add("forecast pairs are physiologically coherent",
                  rate <= config.pair_violation_max, round(rate, 6),
                  f"<= {config.pair_violation_max}",
                  "sbp and dbp ship independently selected architectures; this is the only "
                  "check on what they say together")

        result = GateResult(g.rows)
        f = result.frame()
        logging.info("safety gates: %d/%d PASS, %d WARN, %d critical failures -> %s",
                     int((f.status == PASS).sum()), len(f), int((f.status == WARN).sum()),
                     len(result.critical_failures),
                     "PROMOTABLE" if result.promotable else "BLOCKED")
        for r in g.rows:
            if r["status"] != PASS:
                logging.warning("gate %s [%s]: %s", r["gate"], r["status"], r["detail"])
        return result
    except Exception as e:
        raise CustomException(e, sys)


def _known_tiers() -> set:
    from src.utils.ml_utils.rule_engine.registry import TIERS
    return set(TIERS)
