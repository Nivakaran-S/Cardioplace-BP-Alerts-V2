"""Request models. Pydantic v2, `extra="forbid"`.

Forbidding unknown fields is deliberate: a mistyped `symtoms` should be a 422 naming the
field, not a silently ignored key that makes the engine evaluate a patient with no symptoms
and return a reassuring answer.
"""

from typing import Literal

import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from src.constants.training_pipeline import INGEST_RANGES, SCHEMA_RANGES
from src.serving.settings import MAX_READINGS
from src.serving.vocabulary import CONDITION_SET, MED_SET, SYMPTOM_SET

_SBP_LO, _SBP_HI = INGEST_RANGES["sbp"]
_DBP_LO, _DBP_HI = INGEST_RANGES["dbp"]
_MIN_PP = INGEST_RANGES["min_pulse_pressure"]
_W_LO, _W_HI = SCHEMA_RANGES.get("weight", (25, 220))


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Reading(_Base):
    """One dialysis session as the user reports it."""

    date: str = Field(validation_alias=AliasChoices("date", "ts"))
    sbp: int = Field(ge=_SBP_LO, le=_SBP_HI)
    dbp: int = Field(ge=_DBP_LO, le=_DBP_HI)
    pulse: float | None = Field(None, ge=20, le=250)
    weight: float | None = Field(None, ge=_W_LO, le=_W_HI)
    idwg: float | None = Field(None, ge=-5, le=12)
    symptoms: list[str] = Field(default_factory=list)
    position: Literal["SITTING", "STANDING", "LYING"] = "SITTING"
    #: Confirmatory measurements. Load-bearing, not cosmetic: the engine gates every
    #: non-emergency rule behind n_meas >= 2, so a single unconfirmed reading is reported
    #: as gated rather than acted on.
    n_meas: int = Field(2, ge=1, le=10)

    @field_validator("date")
    @classmethod
    def _parse_date(cls, v):
        ts = pd.to_datetime(v, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"unparseable date {v!r}; use YYYY-MM-DD")
        return str(pd.Timestamp(ts).normalize().date())

    @field_validator("symptoms")
    @classmethod
    def _known_symptoms(cls, v):
        bad = sorted(set(v) - SYMPTOM_SET)
        if bad:
            raise ValueError(f"unknown symptom keys: {bad}")
        return v

    @model_validator(mode="after")
    def _pulse_pressure(self):
        if self.sbp - self.dbp < _MIN_PP:
            raise ValueError(
                f"pulse pressure {self.sbp - self.dbp} is below the {_MIN_PP} mmHg "
                f"physiological floor; check the systolic and diastolic values")
        return self


class Profile(_Base):
    age: float = Field(65.0, ge=18, le=110)
    is_male: Literal[0, 1] = 0
    is_dm: Literal[0, 1] = 0
    is_pregnant: Literal[0, 1] = 0
    hf_type: Literal["NONE", "HFREF", "HFPEF"] = "NONE"
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    provider_target: float | None = Field(None, ge=80, le=200)
    #: Target post-dialysis weight, in kg. Optional, and absent for anyone not on dialysis.
    #: Training derived `idwg_rel` and `uf_rate` from it, so without it those stay NaN --
    #: which is the honest state, not a reason to substitute the measured weight. Bounds match
    #: SCHEMA_RANGES["weight"].
    dryweight: float | None = Field(None, ge=25, le=220)
    missed_3d: int = Field(0, ge=0, le=3)
    adherence_7d: float = Field(1.0, ge=0.0, le=1.0)
    #: Where this history sits in the patient's real timeline. Without it a returning
    #: patient who pastes their last 20 readings looks like a patient at step 0-19, and
    #: RULE_FIRST_MONTH_ADHERENCE_NUDGE (step < 30) fires for everyone.
    step_offset: int = Field(0, ge=0, le=100_000)

    @field_validator("conditions")
    @classmethod
    def _known_conditions(cls, v):
        bad = sorted(set(v) - CONDITION_SET)
        if bad:
            raise ValueError(f"unknown condition keys: {bad}")
        return v

    @field_validator("medications")
    @classmethod
    def _known_meds(cls, v):
        bad = sorted(set(v) - MED_SET)
        if bad:
            raise ValueError(f"unknown medication keys: {bad}")
        return v

    @model_validator(mode="after")
    def _coherent(self):
        # Enforced server-side rather than trusted to the client. The old SPA had to
        # remember to push has_hf itself, and a second client would have had to remember too.
        if self.hf_type != "NONE" and "has_hf" not in self.conditions:
            self.conditions = [*self.conditions, "has_hf"]
        if self.is_pregnant and self.is_male:
            raise ValueError("is_pregnant and is_male are both set; the pregnancy rule axis "
                             "would fire on a male patient")
        return self


class EnrichFlags(_Base):
    """Per-block switches, so a latency measurement can isolate the core predict path."""
    rule_engine: bool = True
    predicted_alert: bool = True
    anomaly: bool = True
    backtest: bool = True
    symptom_risk: bool = True
    #: Score every horizon rather than just the next session. Off by default:
    #: the full set is 45 separate calibrated models scored one row at a time,
    #: measured at ~1s, which is most of the request.
    symptom_risk_full: bool = False
    #: Score the symptom heads on the FORECAST vitals instead of only on today's, giving one
    #: coherent "next session: SBP 158, DBP 87, dizziness 11%" answer. Off by default because
    #: it costs one feature build and ~15 head calls per horizon; the observed-row block is
    #: still returned alongside it, never replaced.
    symptom_chained: bool = False


class PredictRequest(_Base):
    patient_id: str = Field("demo", min_length=1, max_length=64,
                            pattern=r"^[A-Za-z0-9._:\- ]+$")
    profile: Profile = Field(default_factory=Profile)
    readings: list[Reading] = Field(min_length=1, max_length=MAX_READINGS)
    as_of: str | None = None
    enrich: EnrichFlags = Field(default_factory=EnrichFlags)

    @model_validator(mode="after")
    def _sort_and_dedupe(self):
        self.readings = sorted(self.readings, key=lambda r: r.date)
        dates = [r.date for r in self.readings]
        dupes = sorted({d for d in dates if dates.count(d) > 1})
        if dupes:
            # The corpus is one row per dialysis SESSION. Two rows on one date give a
            # zero-day gap, which the cadence clip silently rewrites to 1 -- so the model
            # would be told the readings are a day apart when they are hours apart.
            raise ValueError(
                f"more than one reading on {', '.join(dupes[:3])}. Merge same-day readings "
                f"into one session and set n_meas to the number of measurements.")
        return self
