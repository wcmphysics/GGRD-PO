"""Canonical process histories and aligned wafer-level training data."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

PADDING = "<padding>"
NOT_APPLICABLE = "<not_applicable>"
NUMERIC_FIELDS = ("time", "temperature", "thickness")


def validate_label(value: object, field: str) -> None:
    """Require a nonempty label outside the reserved feature sentinels."""
    if not isinstance(value, str) or not value.strip() or value in {PADDING, NOT_APPLICABLE}:
        raise ValueError(f"Invalid {field}={value!r}; expected a nonempty label outside reserved feature labels")


def _validate_parameters(step: object, fields: tuple[str, ...]) -> None:
    """Validate finite recorded settings on a historical or candidate step."""
    for field in fields:
        value = getattr(step, field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)):
            raise ValueError(f"Invalid {field}={value!r}; expected a finite number or missing value")


@dataclass(frozen=True, slots=True)
class DepositionStep:
    """A deposition step with possibly unrecorded numeric settings."""

    operation: ClassVar[Literal["deposition"]] = "deposition"
    material: str
    recipe: str
    time: float | None
    temperature: float | None
    thickness: float | None

    def __post_init__(self) -> None:
        """Validate the deposition fields."""
        validate_label(self.material, "material")
        validate_label(self.recipe, "recipe")
        _validate_parameters(self, NUMERIC_FIELDS)


@dataclass(frozen=True, slots=True)
class TreatmentStep:
    """A treatment step with an opaque recipe label."""

    operation: ClassVar[Literal["treatment"]] = "treatment"
    recipe: str
    time: float | None
    temperature: float | None

    def __post_init__(self) -> None:
        """Validate the treatment fields."""
        validate_label(self.recipe, "recipe")
        _validate_parameters(self, NUMERIC_FIELDS[:2])


type ProcessStep = DepositionStep | TreatmentStep


@dataclass(frozen=True, slots=True)
class ProcessSequence:
    """Active process steps and their internal transition bits."""

    steps: tuple[ProcessStep, ...]
    vacuum_breaks: tuple[int, ...]

    def __post_init__(self) -> None:
        """Require a complete active sequence and one bit per internal transition."""
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError(f"Invalid steps={self.steps!r}; expected a nonempty tuple of process steps")
        for step in self.steps:
            if not isinstance(step, (DepositionStep, TreatmentStep)):
                raise ValueError(f"Invalid process step={step!r}; expected deposition or treatment")
        expected = len(self.steps) - 1
        if not isinstance(self.vacuum_breaks, tuple) or len(self.vacuum_breaks) != expected:
            raise ValueError(f"Invalid vacuum_breaks={self.vacuum_breaks!r}; expected {expected} internal bits")
        for bit in self.vacuum_breaks:
            if type(bit) is not int or bit not in (0, 1):
                raise ValueError(f"Invalid vacuum break={bit!r}; expected integer 0 or 1")


def sequence_payload(sequence: ProcessSequence) -> dict[str, list]:
    """Represent a canonical sequence in the prediction JSON format."""
    steps = []
    for step in sequence.steps:
        fields = {"operation": step.operation, "recipe": step.recipe, "time": step.time,
                  "temperature": step.temperature}
        if isinstance(step, DepositionStep):
            fields.update(material=step.material, thickness=step.thickness)
        steps.append(fields)
    return {"steps": steps, "vacuum_breaks": list(sequence.vacuum_breaks)}


@dataclass(frozen=True, slots=True)
class WaferSample:
    """One measured wafer and its pre-measurement history."""

    wafer_id: str
    group: str
    sequence: ProcessSequence
    target: float


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Features and metadata sharing the same positional wafer indices."""

    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    wafer_ids: np.ndarray
    row_positions: np.ndarray

    def __post_init__(self) -> None:
        """Check one-dimensional metadata alignment against features."""
        for field, values in (("y", self.y), ("groups", self.groups), ("wafer_ids", self.wafer_ids),
                              ("row_positions", self.row_positions)):
            if values.shape != (len(self.X),):
                raise ValueError(f"Invalid {field} shape={values.shape}; "
                                 f"expected one item per row of X ({len(self.X)})")

    def take_positions(self, indices: np.ndarray) -> "TrainingBatch":
        """Return an aligned batch indexed by positions local to this batch."""
        return TrainingBatch(self.X.iloc[indices], self.y[indices], self.groups[indices],
                             self.wafer_ids[indices], self.row_positions[indices])
