"""Configuration contracts for feasible process search."""

from math import ulp
from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

from src.config.pipeline_common import (
    FiniteNumber,
    NonemptyText,
    NonnegativeInt,
    NonnegativeNumber,
    ParameterUnits,
    PositiveInt,
    PositiveNumber,
    StrictConfig,
)
from src.dataset.contracts import validate_label

_Coverage = Annotated[float, Field(strict=True, ge=0, le=1)]
_Bit = Annotated[int, Field(strict=True, ge=0, le=1)]
_Operation = Literal["deposition", "treatment"]
_Label = Annotated[str, Field(min_length=1)]


class NumericRange(StrictConfig):
    """Inclusive bounds with an optional exact grid and engineering scale."""

    lower: FiniteNumber
    upper: FiniteNumber
    spacing: PositiveNumber | None
    support_scale: PositiveNumber | None

    @model_validator(mode="after")
    def check_range(self) -> "NumericRange":
        """Check bounds, usable precision, and support-scale consistency."""
        if self.lower > self.upper:
            raise ValueError(f"numeric range {self.lower}..{self.upper} requires lower <= upper")
        if self.lower == self.upper and self.support_scale is not None:
            raise ValueError(f"fixed numeric range {self.lower} requires support_scale=null")
        if self.spacing is not None and self.lower < self.upper:
            # Keep exact lattice points farther apart than any float rounding cell in the interval.
            resolution = ulp(max(abs(self.lower), abs(self.upper)))
            if self.spacing <= resolution:
                raise ValueError(f"spacing {self.spacing} for {self.lower}..{self.upper}; "
                                 f"expected spacing greater than float resolution {resolution}")
        return self


class TreatmentRanges(StrictConfig):
    """Numeric choices for a treatment."""

    time: NumericRange
    temperature: NumericRange


class DepositionRanges(TreatmentRanges):
    """Numeric choices for a deposition."""

    thickness: NumericRange


class SearchDomain(StrictConfig):
    """Process catalog and operation limits for an equipment profile."""

    source: Literal["equipment"]
    parameter_units: ParameterUnits
    deposition_catalog: dict[_Label, list[_Label]]
    treatment_catalog: list[_Label]
    deposition_ranges: DepositionRanges
    treatment_ranges: TreatmentRanges
    vacuum_breaks: list[_Bit]
    min_active_count: PositiveInt
    max_active_count: PositiveInt
    min_depositions: NonnegativeInt
    max_treatments: NonnegativeInt
    allow_treatment_first: StrictBool

    @model_validator(mode="after")
    def check_domain(self) -> "SearchDomain":
        """Check catalog and limits for a well-formed search domain."""
        if not (self.deposition_catalog or self.treatment_catalog):
            raise ValueError("empty process catalog; expected a deposition or treatment entry")
        if self.min_active_count > self.max_active_count:
            raise ValueError(f"active counts {self.min_active_count}..{self.max_active_count} are out of order")
        if (self.min_depositions > self.max_active_count or self.min_depositions and not self.deposition_catalog):
            raise ValueError(f"min_depositions={self.min_depositions} exceeds the available deposition slots")
        if not self.vacuum_breaks or len(set(self.vacuum_breaks)) != len(self.vacuum_breaks):
            raise ValueError(f"vacuum_breaks={self.vacuum_breaks!r}; expected a nonempty unique list of bits")
        self._check_catalog()
        return self

    def _check_catalog(self) -> None:
        """Reject ambiguous or reserved entries before enumeration."""
        all_labels = list(self.deposition_catalog) + self.treatment_catalog
        for material, recipes in self.deposition_catalog.items():
            if not recipes or len(set(recipes)) != len(recipes):
                raise ValueError(f"deposition_catalog[{material!r}]={recipes!r}; expected nonempty unique recipes")
            all_labels.extend(recipes)
        if len(set(self.treatment_catalog)) != len(self.treatment_catalog):
            raise ValueError(f"treatment_catalog={self.treatment_catalog!r}; expected unique recipes")
        for label in all_labels:
            validate_label(label, "catalog label")
            if label != label.strip():
                raise ValueError(f"catalog label {label!r} has outer whitespace; expected {label.strip()!r}")


class SlotPin(StrictConfig):
    """A partial set of fields held fixed in one candidate slot."""

    # An omitted pin has no effect. These are presence markers and carry no run defaults.
    active: StrictBool | None = None
    operation: _Operation | None = None
    material: _Label | None = None
    recipe: _Label | None = None
    time: FiniteNumber | None = None
    temperature: FiniteNumber | None = None
    thickness: FiniteNumber | None = None
    vacuum_break_after: _Bit | None = None

    @model_validator(mode="after")
    def check_pin(self) -> "SlotPin":
        """Ensure that at least one field is pinned and absence stands alone."""
        fields = self.model_dump(exclude_none=True)
        if not fields or self.model_fields_set != fields.keys():
            raise ValueError(f"slot pin {self.model_fields_set!r}; expected at least one concrete field and no nulls")
        if self.active is False and len(fields) > 1:
            raise ValueError(f"inactive slot pin {fields!r}; expected no accompanying process or transition pins")
        return self


class SearchConfig(StrictConfig):
    """Fixed/free pin specification and deterministic search budgets."""

    mode: Literal["fixed", "free"]
    N_step_search: PositiveInt
    active_count: PositiveInt | None
    pins: dict[PositiveInt, SlotPin]
    seed: NonnegativeInt
    max_skeletons: PositiveInt
    max_candidates: PositiveInt
    max_attempts: PositiveInt
    batch_size: PositiveInt
    top_k: PositiveInt

    @model_validator(mode="after")
    def check_pins(self) -> "SearchConfig":
        """Validate mode and user-pinned slot indices."""
        has_pins = bool(self.pins) or self.active_count is not None
        if (self.mode == "fixed") != has_pins:
            raise ValueError(f"{self.mode} mode and pins={has_pins}; expected pins in fixed mode only")
        indices = [*self.pins, self.active_count or 1]
        if max(indices) > self.N_step_search:
            raise ValueError(f"slot indices and length {indices!r}; expected 1..{self.N_step_search}")
        return self


class ObjectiveConfig(StrictConfig):
    """Stress matching and optional equipment preference weights."""

    target_stress: FiniteNumber
    target_unit: NonemptyText
    stress_scale: PositiveNumber
    stress_weight: NonnegativeNumber
    cost_weight: NonnegativeNumber
    penalty_weight: NonnegativeNumber
    cost_rule: Literal["steps_fraction"] | None
    penalty_rule: Literal["vacuum_break_fraction"] | None

    @model_validator(mode="after")
    def check_weights(self) -> "ObjectiveConfig":
        """Require a meaningful objective and a rule for each weighted term."""
        weights = (self.stress_weight, self.cost_weight, self.penalty_weight)
        if not any(weights):
            raise ValueError(f"objective weights {weights!r}; expected at least one positive weight")
        if self.cost_weight and self.cost_rule is None:
            raise ValueError(f"cost_weight={self.cost_weight} requires cost_rule")
        if self.penalty_weight and self.penalty_rule is None:
            raise ValueError(f"penalty_weight={self.penalty_weight} requires penalty_rule")
        return self


class SupportConfig(StrictConfig):
    """Eligibility and neighborhood thresholds for data support."""

    radius: NonnegativeNumber
    min_coverage: _Coverage
