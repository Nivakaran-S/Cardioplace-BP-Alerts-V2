---
title: Cardioplace BP Alerts
emoji: 🫀
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.36.2
# Must stay >= pyproject.toml's requires-python. The Space image defaults to 3.10, on
# which the pinned numeric stack does not exist: numpy 2.4.2 and scipy 1.16.0 both
# declare Requires-Python >=3.11, so the build dies at `pip install -r requirements.txt`
# with "No matching distribution found for numpy==2.4.2".
python_version: "3.11"
app_file: gradio_app.py
pinned: false
---

# Cardioplace BP Alerts

Blood-pressure forecasting, personalised alert thresholds and early-warning detection for
haemodialysis patients — sitting **behind** a deterministic clinical rule engine that the ML
layer is never allowed to override.

> **Not a medical device.** Provider-visible decision support only. Nothing here diagnoses,
> and no code path writes a patient-facing alert. See [Clinical disclaimer](#clinical-disclaimer).

---

## What it is

Five layers. Four learn; the fifth does not, and the fifth is the one that decides.

| | Layer | What it answers | Where |
|---|---|---|---|
| **M1** | Forecaster | Where will this patient's SBP / DBP / IDWG be in 1–3 sessions? | `utils/ml_utils/model/architectures.py`, `estimator.py` |
| **M2** | Personalised offset | What counts as "high" *for this patient*? | `model/offset.py`, `model/offset_learned.py` |
| **M3** | Early-warning detector | Is this patient about to breach their own band? | `model/detector.py` |
| **M4** | Symptom heads | Which symptoms are likely? **Synthetic labels — see below.** | `model/classifier_head.py` |
| **—** | **Rule engine** | **Does this reading require action, right now?** | `rule_engine/` |

The rule engine is 56 rules across 8 tiers. It is deterministic, it needs no model, and it is
authoritative. The API serves it whether or not a bundle exists: **an SBP of 195 produces a red
emergency banner on a fresh checkout with an empty `final_model/`.** That is a design property
with a test behind it (`tests/test_api_contract.py`), not an accident.

## The governance contract

These values are clinician and operations inputs. They are never searched, never tuned, and
never derived from data. `architectures.py` asserts at import that none of them has leaked into
a hyperparameter grid, and the safety gates assert at run time that the fitted models respected
them.

| Constant | Value | Meaning |
|---|---|---|
| `EMERGENCY_FLOOR_MMHG` | 180 | Never personalised. Asserted over every threshold *and* every learned prediction. |
| `POPULATION_THRESHOLD_MMHG` | 140 | The anchor personalisation moves away from. |
| `OFFSET_CAP_LOOSEN` / `TIGHTEN` | +15 / −25 | Asymmetric on purpose: loosening a threshold is the hazardous direction. |
| `ALERT_BUDGET_PCT` | 5 | Staffing capacity for the **detector**. The rule engine is deliberately unbudgeted. |
| `WARN_WINDOW` | 3 sessions | Lead time the detector must deliver. |
| `EVENT_QUANTILE` | 0.95 | An "event" is the patient exceeding *their own* p95. |
| `STALE_FORECAST_MAX_DAYS` | 14 | Asserted equal to `RuleEngine.STALE_GAP_DAYS`, so the two layers cannot disagree about "stale". |

## The causal contract

A feature at step *t* may read only steps ≤ *t−1*. The single deliberate exception is same-day
adherence, which genuinely is known when a forecast is made for the *next* session.

This is tested, not asserted. `leakage_audit` rebuilds features on a patient whose future
readings have been corrupted by +50 mmHg and requires every past feature row to come back
bit-identical. `cadence_audit` proves the 30-day gap clip is a guard rather than a transform.
Both block the run on failure.

## Dataset

[HEMOBP](https://doi.org/10.6084/m9.figshare.c.4590516), a public haemodialysis vitals corpus:
4,366,298 BP readings over 165,986 sessions and 1,075 patients. Not committed — see
[`data/README.md`](data/README.md).

### A session is not a calendar date

Worth calling out, because getting it wrong cost more than anything else in this pipeline.

The corpus runs three dialysis shifts a day and the last one crosses midnight. Keying sessions
on `(patient, calendar date)` split every one of those in two. The visible symptom was a derived
session count 43.8% above the published figure. The damaging symptom was invisible: the phantom
half-sessions had no matching record in `d1.csv`, so **`weight`, `dryweight` and `idwg` were
silently NaN on 53.6% of the corpus** — three model features, missing on half the data, with
nothing in any report saying so.

A session is now a contiguous run of readings (gap > 6 h starts a new one), dated to the
dialysis day beginning at 08:00 — which is not a fitted parameter but d1's own convention, since
every `keyindate` in that file is stamped exactly `08:00:00`.

| | calendar keying | contiguous-run keying |
|---|---|---|
| d1 sessions matched | 66.8% | **98.3%** |
| Sunday sessions (d1 records 2) | 16,959 | **22** |
| `dryweight` / `idwg` missing | 53.6% | **17.3%** |
| median session duration | — | **3.98 h** |

The residual ratio of 1.19 is real and decomposed line by line in
`session_decomposition.csv`: `vip.csv` carries BP readings for ~31k dialysis days that `d1.csv`
never recorded, so the published 165,986 is a *floor* on the session count, not the count.

## Install and run

```bash
uv sync                     # or: pip install -r requirements.txt
```

**Train** (needs the corpus; tens of minutes at the default tier, hours at `MODEL_TIER=all`):

```bash
python main.py
```

**Serve** (works with or without a trained bundle):

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1
```

One worker, deliberately: two would mean two independently hot-reloading predictors and two
training managers racing for the same single-flight lock.

**Test:**

```bash
pytest                              # or run either file directly
python tests/test_safety_gates.py   # every critical gate must be able to FAIL
python tests/test_api_contract.py   # both the no-model and with-model paths
```

### Tuning the cost/quality dial

`MODEL_TIER` in `src/constants/training_pipeline/__init__.py` decides how many of the 24
registered architectures compete. It changes *how many* can win, never *which* can.

| tier | architectures | rough cost |
|---|---|---|
| `low` | 9 — linear, classical smoothers, window/delta reparameterisations | seconds |
| `medium` (default) | 19 — adds boosting, per-patient local models, ensembles | minutes |
| `high` / `all` | 24 — adds random forest, extra trees, kNN, local HGB, stacked ensemble | hours |

`FAST_MODE = True` additionally shrinks row caps, the tuning budget and bootstrap draws.

## API

| Method | Route | |
|---|---|---|
| `GET` | `/` | The dashboard |
| `GET` | `/api/schema` | Field vocabulary: 19 symptoms, 9 conditions, 11 medications, rule inventory |
| `GET` | `/api/health` | Model state, sklearn runtime, version warnings, training status |
| `GET` | `/api/model` | Version, shipped families, governance block (404 with no model) |
| `POST` | `/api/predict` | The advisory plus every dashboard block. **200 even with no model.** |
| `POST` | `/api/train` · `GET /api/train/status` · `POST /api/train/cancel` | Single-flight training subprocess |

`src/serving/**` and `app.py` may not import `src/components/**` or `src/pipeline/**`. CI
enforces it with a grep. That rule is what keeps a broken training module from taking down an
API whose most important job is the one it does without any model.

## Artifacts

```
Artifacts/<run_id>/
  data_ingestion/     sessions.csv, static.csv, ingest_reconciliation.csv,
                      session_decomposition.csv, ingested/{train,test}.csv
  data_validation/    data_contract.csv, drift_report/report.yaml, validated/
  data_transformation/ panel.parquet, features.parquet, feature_names.yaml,
                      feature_dictionary.csv, leakage_audit.csv, feature_selection.csv
  model_trainer/
    trained_model/    model.pkl, predictor.joblib
    reports/          ~25 scorecards: forecast, paired_comparison, offset_learned_board,
                      detector, fairness, safety_gates, cold_start, missingness, drift, …
final_model/model.pkl   the promotion target -- written only when every critical gate passes
```

### Two on-disk shapes, on purpose

`final_model/model.pkl` is a pickled `BPPredictor` *instance* (`save_object`);
`predictor.joblib` is the bundle *dict* (`predictor.save`). Both are valid, and
`src/serving/model_registry.py` sniffs which it has. Loading one as the other yields
`TypeError: 'BPPredictor' object is not subscriptable` a long way from the cause.

## Tests

Three suites, and all three follow the same rule: **a check that cannot fail is worse than no
check**, because it reports PASS forever and is read as evidence. Each suite therefore proves
its checks bite by feeding them the corruption they exist for.

| Suite | Proves |
|---|---|
| `tests/test_safety_gates.py` | Every one of the 8 *critical* gates blocks promotion when violated; every non-critical one warns without blocking; and absent evidence records as WARN, never PASS. |
| `tests/test_leakage_probes.py` | The audit passes on clean features **and** catches a future-peeking builder, a misaligned target, a latent generator variable and a straight copy of the target. |
| `tests/test_api_contract.py` | 41 assertions across both the no-model and with-model paths, including that SBP 195 fires an emergency with nothing on disk, that no response contains a `NaN` token, and that every `$("id")` in the SPA exists in the shell. |

## Known findings and limitations

Stated plainly, because a system that hides these is worse than one that has them.

- **Subgroup gap, under-50s.** The forecaster is 2.39 mmHg worse for patients under 50 than
  overall (n = 2,199, MAE 16.49 vs 14.09). Well-powered, not noise. The fairness gate is
  *critical* and blocks promotion on it — correctly. It needs a modelling response, not a
  looser margin.
- **Personalisation coverage is ~27%, not 90%.** The offset targets each patient's next-30
  q0.90, but the governance cap pins thresholds at 155 mmHg, so for the most hypertensive
  patients the threshold cannot follow. That is the safety cap working as designed; the
  `coverage` metric now measures it instead of leaving it implicit.
- **The learned offset does not currently ship.** All 13 candidates are trained and scored,
  and the best does not beat the capped blend on a paired test. The blend is retained.
- **Journaling input is missing three training features.** `idwg`, `sbp_drop` and `uf_total`
  are intradialytic; a phone cannot supply them. They are served as missing, never zero-filled,
  and the missingness sweep quantifies the cost.
- **No pulse column.** HEMOBP has none, so all HR-gated rules stay `BLOCKED_ON_INPUTS` against
  the corpus. At serving they are evaluated from what the user enters.
- **`CAD` fires early by design.** `RULE_CAD_HIGH` triggers at SBP ≥ 130, ahead of
  `RULE_PERSONALIZED_HIGH`. Ticking "coronary artery disease" makes nearly every reading fire.
  That ordering is the clinical semantics; it is not a bug and it is not reordered.

## ⚠️ Model 4 symptom labels are synthetic

**No real symptom was ever observed in HEMOBP.** The corpus carries blood pressure, weight and
ultrafiltration — no symptoms, no medications, no diagnosed conditions.

Every symptom label the symptom heads train on is generated by a structural causal model in
`src/utils/ml_utils/rule_engine/synthetic.py`, calibrated to published literature rates. A model
trained on generated labels recovers the generating process and nothing else. **A high symptom
AUC is a statement about that generator, not about physiology.**

The bundle carries `symptom_labels_synthetic=True`, a safety gate asserts the flag survives,
the API attaches a warning to every symptom response, and the dashboard renders it as a
non-dismissible banner. If you remove those, you are removing the only thing standing between a
synthetic hazard model and a reader who thinks it is a clinical prediction.

## Clinical disclaimer

Provider-visible decision support. **Not a medical device, not a diagnosis, and not a substitute
for clinical judgement.**

The ML layer never writes a DeviationAlert and never moves an emergency threshold. The
180 / 120 mmHg emergency floor is fixed, is never personalised, and is asserted in the offset
model, in the learned offset, and again at the promotion gate. Where the ML layer and the rule
engine disagree, the rule engine wins.

Race and ethnicity are **not auditable** on this corpus — HEMOBP does not record them — so the
fairness gate covers sex, age band and diabetes status only. That is a real limitation of the
audit, not evidence of parity on the axes it cannot see.
