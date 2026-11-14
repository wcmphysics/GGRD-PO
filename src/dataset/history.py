"""Read long-form wafer records into canonical measured histories."""

import csv
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
import re
from typing import TextIO

from src.dataset.contracts import DepositionStep, ProcessSequence, ProcessStep, TreatmentStep, WaferSample

_HEADER = ("group", "wafer_id", "step", "operation", "parameter", "value")
_PARAMETERS = {
    "deposition": {"material", "recipe", "time", "temperature", "thickness"},
    "treatment": {"recipe", "time", "temperature"},
    "transition": {"vacuum_break"},
    "measurement": {"stress"},
}


@dataclass
class _WaferRecords:
    """Temporary records sharing one wafer and group."""

    group: str
    steps: dict[int, dict[str, dict[str, str]]] = field(default_factory=dict)


def _collect_records(handle: TextIO) -> dict[str, _WaferRecords]:
    """Read text rows and group them by canonical integer step and wafer."""
    reader = csv.reader(handle, strict=True)
    header = next(reader, [])
    if len(header) != len(_HEADER) or set(header) != set(_HEADER):
        raise ValueError(f"Invalid CSV header={header!r}; expected exactly {list(_HEADER)!r}")
    positions = [header.index(name) for name in _HEADER]
    wafers: dict[str, _WaferRecords] = {}
    for row in reader:
        if len(row) != len(_HEADER):
            raise ValueError(f"Invalid CSV row at line {reader.line_num}={row!r}; expected six fields")
        group, wafer, step, operation, parameter, value = (row[position] for position in positions)
        _validate_record(group, wafer, step, operation, parameter)
        records = wafers.setdefault(wafer, _WaferRecords(group))
        if records.group != group:
            raise ValueError(f"Wafer {wafer!r} belongs to group {group!r}; "
                             f"expected its existing group {records.group!r}")
        parameters = records.steps.setdefault(int(step), {}).setdefault(operation, {})
        if parameter in parameters:
            raise ValueError(f"Duplicate wafer={wafer!r}, step={step}, operation={operation}, parameter={parameter}; "
                             "expected one record per key")
        parameters[parameter] = value
    if not wafers:
        raise ValueError("CSV contains zero wafer records; expected at least one wafer")
    return wafers


def _validate_record(group: str, wafer: str, step: str, operation: str, parameter: str) -> None:
    """Validate raw keys while preserving identifiers and labels as text."""
    context = f"wafer={wafer!r}, step={step!r}, operation={operation!r}, parameter={parameter!r}"
    if not group.strip() or not wafer.strip():
        raise ValueError(f"Invalid {context}, group={group!r}; expected nonempty wafer and group identifiers")
    if re.fullmatch(r"[0-9]+", step) is None or int(step) == 0:
        raise ValueError(f"Invalid {context}; expected a positive integer step")
    if parameter not in _PARAMETERS.get(operation, ()):
        raise ValueError(f"Invalid {context}; expected a documented operation and its allowed parameter")


def _finite_value(value: str, parameter: str) -> float:
    """Convert a supplied raw numeric cell to a finite float."""
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"Invalid {parameter}={value!r}; expected a finite numeric value") from error
    if not isfinite(number):
        raise ValueError(f"Invalid {parameter}={value!r}; expected a finite numeric value")
    return number


def _number_if_present(fields: dict[str, str], parameter: str) -> float | None:
    """Keep an omitted numeric record missing and validate a supplied record."""
    return _finite_value(fields[parameter], parameter) if parameter in fields else None


def _process_step(operations: dict[str, dict[str, str]]) -> tuple[ProcessStep, int]:
    """Parse one active step and accept only one attached transition."""
    process = set(operations) - {"transition"}
    if process not in ({"deposition"}, {"treatment"}):
        raise ValueError(f"Invalid operations={sorted(operations)!r}; expected one deposition or treatment")
    transition = operations.get("transition", {}).get("vacuum_break")
    if transition not in {"0", "1"}:
        raise ValueError(f"Invalid vacuum_break={transition!r}; expected one transition record with 0 or 1")
    operation = next(iter(process))
    settings = operations[operation]
    time = _number_if_present(settings, "time")
    temperature = _number_if_present(settings, "temperature")
    if operation == "deposition":
        step = DepositionStep(settings.get("material", ""), settings.get("recipe", ""), time, temperature,
                              _number_if_present(settings, "thickness"))
        return step, int(transition)
    return TreatmentStep(settings.get("recipe", ""), time, temperature), int(transition)


def _canonical_wafer(wafer_id: str, records: _WaferRecords) -> WaferSample:
    """Validate ordered process and measurement steps and drop the terminal break."""
    step_numbers = sorted(records.steps)
    if len(step_numbers) < 2 or step_numbers != list(range(1, len(step_numbers) + 1)):
        raise ValueError(f"Wafer {wafer_id!r} has steps={step_numbers!r}; expected contiguous process steps and target")
    last = step_numbers[-1]
    measurement = records.steps[last]
    if set(measurement) != {"measurement"} or set(measurement["measurement"]) != {"stress"}:
        raise ValueError(f"Wafer {wafer_id!r}, step={last} has records={measurement!r}; "
                         "expected only a terminal measurement/stress record")
    try:
        target = _finite_value(measurement["measurement"]["stress"], "stress")
    except ValueError as error:
        raise ValueError(f"Wafer {wafer_id!r}, step={last}: {error}") from error
    steps = []
    breaks = []
    for index in step_numbers[:-1]:
        try:
            step, transition = _process_step(records.steps[index])
        except ValueError as error:
            raise ValueError(f"Wafer {wafer_id!r}, step={index}: {error}") from error
        steps.append(step)
        breaks.append(transition)
    if breaks[-1] != 1:
        raise ValueError(f"Wafer {wafer_id!r}, step={last - 1} has terminal vacuum_break={breaks[-1]}; expected 1")
    return WaferSample(wafer_id, records.group, ProcessSequence(tuple(steps), tuple(breaks[:-1])), target)


def _check_replicates(samples: tuple[WaferSample, ...]) -> None:
    """Enforce canonical history equality within each declared group."""
    first_samples: dict[str, WaferSample] = {}
    for sample in samples:
        first = first_samples.setdefault(sample.group, sample)
        if first.sequence != sample.sequence:
            raise ValueError(f"Group {sample.group!r} has a conflicting history in wafer {sample.wafer_id!r}; "
                             f"expected the same process settings and missing states as wafer {first.wafer_id!r}")


def duplicate_histories(samples: tuple[WaferSample, ...]) -> dict[ProcessSequence, tuple[str, ...]]:
    """List identical histories appearing under different declared groups."""
    groups: dict[ProcessSequence, set[str]] = {}
    for sample in samples:
        groups.setdefault(sample.sequence, set()).add(sample.group)
    return {sequence: tuple(sorted(labels)) for sequence, labels in groups.items() if len(labels) > 1}


def read_history(path: Path) -> tuple[WaferSample, ...]:
    """Return validated measured wafers ordered by their original text IDs."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            records = _collect_records(handle)
        samples = tuple(_canonical_wafer(wafer_id, records[wafer_id]) for wafer_id in sorted(records))
        _check_replicates(samples)
    except (OSError, csv.Error, ValueError) as error:
        raise ValueError(f"Cannot read wafer history {path}: {error}") from error
    return samples
