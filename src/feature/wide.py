"""Stateless wide feature encoding of canonical process histories."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import pandas as pd

from src.dataset.contracts import (
    DepositionStep,
    NOT_APPLICABLE,
    NUMERIC_FIELDS,
    PADDING,
    ProcessSequence,
    ProcessStep,
    TrainingBatch,
    WaferSample,
)

_CATEGORICAL_FIELDS = ("operation", "material", "recipe", "vacuum_break_after")


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Ordered slot schema shared by historical and unlabeled sequences."""

    N_step: int
    include_transitions: bool = True

    def __post_init__(self) -> None:
        """Require positive integer capacity for this slot schema."""
        if isinstance(self.N_step, bool) or not isinstance(self.N_step, Integral) or self.N_step < 1:
            raise ValueError(f"Invalid N_step={self.N_step!r}; expected a positive integer slot capacity")

    @property
    def _layout(self) -> tuple[tuple[str, str], ...]:
        """Return the ordered column names and explicit pandas dtypes."""
        categories = _CATEGORICAL_FIELDS if self.include_transitions else _CATEGORICAL_FIELDS[:-1]
        slots = [("active", "int64"), *((field, "object") for field in categories)]
        for field in NUMERIC_FIELDS:
            slots.extend(((field, "float64"), (f"{field}_applicable", "int64"), (f"{field}_observed", "int64")))
        return tuple((f"step_{index}_{field}", dtype) for index in range(1, self.N_step + 1)
                     for field, dtype in slots)

    @property
    def columns(self) -> tuple[str, ...]:
        """Return all raw wide feature names in canonical order."""
        return tuple(name for name, _ in self._layout)

    @property
    def dtypes(self) -> tuple[str, ...]:
        """Return pandas raw wide dtypes matching canonical column order."""
        return tuple(dtype for _, dtype in self._layout)

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        """Return observed or missing raw numeric fields."""
        return self._by_dtype("float64")

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        """Return categorical field names."""
        return self._by_dtype("object")

    @property
    def flag_columns(self) -> tuple[str, ...]:
        """Return activation, applicability and observation flags."""
        return self._by_dtype("int64")

    def _by_dtype(self, expected: str) -> tuple[str, ...]:
        """Select ordered features with the requested raw dtype."""
        return tuple(name for name, dtype in self._layout if dtype == expected)

    def without_transitions(self) -> "FeatureSchema":
        """Return the matching feature layout with transition information removed."""
        return replace(self, include_transitions=False)


def _slot_values(step: ProcessStep | None, transition: str, include_transitions: bool) -> Iterable[object]:
    """Yield one slot's category, value and flag fields in schema order."""
    yield int(step is not None)
    if step is None:
        yield from (PADDING,) * (4 if include_transitions else 3)
    else:
        yield step.operation
        yield step.material if isinstance(step, DepositionStep) else NOT_APPLICABLE
        yield step.recipe
        if include_transitions:
            yield transition
    for field in NUMERIC_FIELDS:
        applicable = step is not None and (field != "thickness" or isinstance(step, DepositionStep))
        value = getattr(step, field) if applicable else None
        yield np.nan if value is None else value
        yield int(applicable)
        yield int(value is not None)


def _row_values(sequence: ProcessSequence, schema: FeatureSchema) -> tuple[object, ...]:
    """Widen one sequence, preserving missing values before any model fitting."""
    count = len(sequence.steps)
    if count > schema.N_step:
        raise ValueError(f"Sequence has {count} active steps; expected at most N_step={schema.N_step}")
    slots = (*sequence.steps, *((None,) * (schema.N_step - count)))
    transitions = tuple(str(bit) for bit in sequence.vacuum_breaks) + (NOT_APPLICABLE,) * (schema.N_step - count + 1)
    return tuple(value for step, transition in zip(slots, transitions, strict=True)
                 for value in _slot_values(step, transition, schema.include_transitions))


def build_features(sequences: Iterable[ProcessSequence], schema: FeatureSchema) -> pd.DataFrame:
    """Build one ordered raw feature row per canonical sequence."""
    rows = [_row_values(sequence, schema) for sequence in sequences]
    return pd.DataFrame.from_records(rows, columns=schema.columns).astype(dict(schema._layout))


def build_training_batch(samples: Iterable[WaferSample], schema: FeatureSchema) -> TrainingBatch:
    """Build feature and metadata arrays without using labels for preprocessing."""
    wafers = tuple(samples)
    for wafer in wafers:
        if len(wafer.sequence.steps) > schema.N_step:
            raise ValueError(f"Wafer {wafer.wafer_id!r} has {len(wafer.sequence.steps)} steps; "
                             f"expected at most N_step={schema.N_step}")
    return TrainingBatch(
        X=build_features((wafer.sequence for wafer in wafers), schema),
        y=np.array([wafer.target for wafer in wafers], dtype=float),
        groups=np.array([wafer.group for wafer in wafers], dtype=object),
        wafer_ids=np.array([wafer.wafer_id for wafer in wafers], dtype=object),
        row_positions=np.arange(len(wafers)),
    )
