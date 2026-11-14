"""Describe canonical wafer histories before fitting any model state."""

import json
from pathlib import Path

import pandas as pd

from src.dataset.contracts import WaferSample
from src.dataset.history import duplicate_histories


def _numeric_missing(samples: tuple[WaferSample, ...]) -> list[dict[str, str | int]]:
    """Count missing applicable values by operation among all wafers."""
    counts: dict[tuple[str, str], list[int]] = {}
    for sample in samples:
        for step in sample.sequence.steps:
            parameters = ("time", "temperature", "thickness") if step.operation == "deposition" else (
                "time", "temperature"
            )
            for parameter in parameters:
                tally = counts.setdefault((step.operation, parameter), [0, 0])
                tally[0] += 1
                tally[1] += getattr(step, parameter) is None
    return [
        {"operation": operation, "parameter": parameter, "applicable": applicable, "missing": missing}
        for (operation, parameter), (applicable, missing) in sorted(counts.items())
    ]


def write_exploration(samples: tuple[WaferSample, ...], raw_rows: int, destination: Path) -> None:
    """Write group sizes, process lengths, missingness, and measured targets."""
    destination.mkdir()
    wafer_table = pd.DataFrame.from_records(
        {
            "row_id": sample.wafer_id,
            "group_id": sample.group,
            "target_stress": sample.target,
            "active_count": len(sample.sequence.steps),
        }
        for sample in samples
    )
    group_table = wafer_table.groupby("group_id", sort=True).size().rename("n_wafers").reset_index()
    duplicates = list(duplicate_histories(samples).values())
    summary = {
        "raw_rows": raw_rows,
        "wafer_count": len(wafer_table),
        "group_count": len(group_table),
        "duplicate_history_count": len(duplicates),
        "groups_sharing_histories": duplicates,
        "numeric_missing": _numeric_missing(samples),
    }
    wafer_table.to_csv(destination / "wafers.csv", index=False)
    group_table.to_csv(destination / "groups.csv", index=False)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
