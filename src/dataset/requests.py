"""Prediction request parsing for the canonical process sequence contract."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, TypeAdapter

from src.dataset.contracts import DepositionStep, ProcessSequence, TreatmentStep


class _RequestStep(BaseModel):
    """Shared fields in a strictly typed JSON process step."""

    model_config = ConfigDict(extra="forbid", strict=True)
    recipe: str
    time: FiniteFloat | None
    temperature: FiniteFloat | None


class _DepositionRequest(_RequestStep):
    """Required fields in one JSON deposition step."""

    operation: Literal["deposition"]
    material: str
    thickness: FiniteFloat | None


class _TreatmentRequest(_RequestStep):
    """Required fields in one JSON treatment step."""

    operation: Literal["treatment"]


class _RequestSequence(BaseModel):
    """JSON representation of active steps and internal breaks."""

    model_config = ConfigDict(extra="forbid", strict=True)
    steps: list[Annotated[_DepositionRequest | _TreatmentRequest, Field(discriminator="operation")]]
    vacuum_breaks: list[int]


class _Request(BaseModel):
    """One labeled prediction request."""

    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str
    sequence: _RequestSequence


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    """Unlabeled process history with an external request identifier."""

    request_id: str
    sequence: ProcessSequence


_REQUESTS = TypeAdapter(Annotated[list[_Request], Field(min_length=1)])


def _unique_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate object members before schema validation."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}; expected each field exactly once in an object")
        result[key] = value
    return result


def _canonical_sequence(raw_sequence: _RequestSequence) -> ProcessSequence:
    """Convert schema-validated JSON steps to the shared immutable process type."""
    steps = []
    for index, raw_step in enumerate(raw_sequence.steps, start=1):
        try:
            if isinstance(raw_step, _DepositionRequest):
                step = DepositionStep(raw_step.material, raw_step.recipe, raw_step.time, raw_step.temperature,
                                      raw_step.thickness)
            else:
                step = TreatmentStep(raw_step.recipe, raw_step.time, raw_step.temperature)
        except ValueError as error:
            raise ValueError(f"Step {index}: {error}") from error
        steps.append(step)
    return ProcessSequence(tuple(steps), tuple(raw_sequence.vacuum_breaks))


def read_requests(path: Path) -> tuple[PredictionRequest, ...]:
    """Read a nonempty ordered JSON array of unique prediction requests."""
    try:
        raw_requests = _REQUESTS.validate_python(json.loads(path.read_text(encoding="utf-8-sig"),
                                                           object_pairs_hook=_unique_members))
        requests = []
        request_ids: set[str] = set()
        for raw in raw_requests:
            if not raw.request_id.strip() or raw.request_id in request_ids:
                raise ValueError(f"Invalid request_id={raw.request_id!r}; expected a unique nonempty ID")
            request_ids.add(raw.request_id)
            try:
                sequence = _canonical_sequence(raw.sequence)
            except ValueError as error:
                raise ValueError(f"Invalid request_id={raw.request_id!r}: {error}") from error
            requests.append(PredictionRequest(raw.request_id, sequence))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read prediction requests {path}: {error}") from error
    return tuple(requests)
