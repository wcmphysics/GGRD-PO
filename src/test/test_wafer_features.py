"""Acceptance coverage for canonical histories and fit-local wide preprocessing."""

import csv
import json
from pathlib import Path

import numpy as np
from pandas.testing import assert_frame_equal
import pytest

from src.dataset.contracts import DepositionStep, ProcessSequence, TreatmentStep, sequence_payload
from src.dataset.history import duplicate_histories, read_history
from src.dataset.requests import read_requests
from src.feature.preprocess import make_preprocessor, warn_empty_columns
from src.feature.wide import FeatureSchema, build_features, build_training_batch


def _wafer_rows(wafer: str, group: str, target: str) -> list[tuple[str, ...]]:
    """Create a two-step history that includes real zero and missing parameters."""
    records = [
        ("1", "deposition", "material", "NA"),
        ("1", "deposition", "recipe", "NA"),
        ("1", "deposition", "time", "0"),
        ("1", "transition", "vacuum_break", "0"),
        ("2", "treatment", "recipe", "anneal_v1"),
        ("2", "transition", "vacuum_break", "1"),
        ("3", "measurement", "stress", target),
    ]
    return [(group, wafer, *record) for record in records]


def _write_history(path: Path, rows: list[tuple[str, ...]]) -> None:
    """Write fixture rows using the raw history schema."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("group", "wafer_id", "step", "operation", "parameter", "value"))
        writer.writerows(rows)


def test_canonical_ingestion_and_parity(tmp_path: Path) -> None:
    """Raw row order, replicates and nullable request input preserve exact feature values."""
    path = tmp_path / "example.csv"
    rows = (_wafer_rows("0002", "001", "2") + _wafer_rows("0001", "001", "-1")
            + _wafer_rows("0003", "002", "3"))
    _write_history(path, rows)
    samples = read_history(path)
    _write_history(path, rows[::-1])
    assert read_history(path) == samples
    assert [sample.wafer_id for sample in samples] == ["0001", "0002", "0003"]
    assert list(duplicate_histories(samples).values()) == [("001", "002")]
    assert samples[0].sequence.vacuum_breaks == (0,)

    candidate = ProcessSequence((DepositionStep("NA", "NA", 0, None, None),
                                 TreatmentStep("anneal_v1", None, None)), (0,))
    request_path = tmp_path / "input.json"
    request_path.write_text(json.dumps([{"request_id": "01", "sequence": sequence_payload(candidate)}]))
    request = read_requests(request_path)[0]
    assert request.request_id == "01"
    schema = FeatureSchema(3)
    batch = build_training_batch(samples, schema)
    assert_frame_equal(batch.X.iloc[:1], build_features((request.sequence,), schema))
    assert_frame_equal(batch.X.iloc[:1], build_features((candidate,), schema))
    reordered = batch.take_positions(np.array([2, 0]))
    assert reordered.y.tolist() == [3, -1]
    assert reordered.groups.tolist() == ["002", "001"]
    assert reordered.wafer_ids.tolist() == ["0003", "0001"]
    assert reordered.row_positions.tolist() == [2, 0]


@pytest.mark.parametrize(("replace_index", "new_record", "error"), [
    (None, ("001", "0001", "1", "deposition", "material", "NA"), "Duplicate.*parameter=material"),
    (0, ("001", "0001", "1", "deposition", "unknown", "NA"), "expected a documented operation"),
    (2, ("001", "0001", "1", "deposition", "time", ""), "time=.*finite numeric"),
    (3, ("002", "0001", "1", "transition", "vacuum_break", "0"), "expected its existing group"),
    (3, ("001", "0001", "1", "transition", "vacuum_break", "2"), "expected one transition"),
    (5, ("001", "0001", "2", "transition", "vacuum_break", "0"), "terminal vacuum_break=0; expected 1"),
    (6, ("001", "0001", "4", "measurement", "stress", "1"), "expected contiguous"),
    (6, ("001", "0001", "3", "measurement", "stress", "inf"), "stress=.*finite numeric"),
    (None, ("001", "0001", "3", "transition", "vacuum_break", "1"), "expected only a terminal"),
])
def test_invalid_history_records(tmp_path: Path, replace_index: int | None,
                                 new_record: tuple[str, ...], error: str) -> None:
    """Malformed raw records fail with context and an expected value."""
    rows = _wafer_rows("0001", "001", "1")
    if replace_index is None:
        rows.append(new_record)
    else:
        rows[replace_index] = new_record
    path = tmp_path / "invalid.csv"
    _write_history(path, rows)
    with pytest.raises(ValueError, match=error):
        read_history(path)


def test_conflicting_group_history(tmp_path: Path) -> None:
    """Replicates with distinct process observations cannot share a group."""
    rows = _wafer_rows("0001", "001", "1") + _wafer_rows("0002", "001", "2")
    rows.append(("001", "0002", "2", "treatment", "time", "400"))
    path = tmp_path / "invalid.csv"
    _write_history(path, rows)
    with pytest.raises(ValueError, match="Group '001'.*conflicting history.*'0002'"):
        read_history(path)


@pytest.mark.parametrize(("modification", "error"), [
    ({"time": "60"}, "valid number"),
    ({"temperature": float("inf")}, "finite number"),
    ({"thickness": 2}, "Extra inputs"),
    ({"recipe": "<padding>"}, "reserved feature labels"),
])
def test_request_value_validation(tmp_path: Path, modification: dict[str, object], error: str) -> None:
    """JSON requests reject wrong types, invalid numbers, and incoherent fields."""
    step = {"operation": "treatment", "recipe": "anneal", "time": 60, "temperature": None} | modification
    path = tmp_path / "requests.json"
    request = {"request_id": "one", "sequence": {"steps": [step], "vacuum_breaks": []}}
    path.write_text(json.dumps([request]))
    with pytest.raises(ValueError, match=error):
        read_requests(path)


def test_direct_step_numbers_must_fit_float() -> None:
    """Reject an integer too large for the feature dtype with field context."""
    with pytest.raises(ValueError, match=r"Invalid time=.*expected a finite number"):
        TreatmentStep("anneal", 10**500, None)


def test_wide_states_and_transitions() -> None:
    """Wide slots distinguish active missing values, inapplicable values and padding."""
    schema = FeatureSchema(3)
    steps = (DepositionStep("TiN", "recipe", 0, None, 5), TreatmentStep("anneal", 100, 120))
    X = build_features((ProcessSequence(steps, (0,)), ProcessSequence(steps, (1,))), schema)
    assert tuple(X.columns) == schema.columns
    assert tuple(X.dtypes.astype(str)) == schema.dtypes
    assert set(X.columns).isdisjoint({"wafer_id", "group", "row_positions", "target"})
    assert X["step_1_vacuum_break_after"].tolist() == ["0", "1"]
    assert X.at[0, "step_2_vacuum_break_after"] == "<not_applicable>"
    assert X.at[0, "step_2_material"] == "<not_applicable>"
    for field in ("operation", "material", "recipe", "vacuum_break_after"):
        assert X.at[0, f"step_3_{field}"] == "<padding>"
    expected = [("step_1_time", (1, 1), 0), ("step_1_temperature", (1, 0), None),
                ("step_2_thickness", (0, 0), None), ("step_3_time", (0, 0), None)]
    for column, flags, value in expected:
        assert (X.at[0, f"{column}_applicable"], X.at[0, f"{column}_observed"]) == flags
        assert np.isnan(X.at[0, column]) if value is None else X.at[0, column] == value
    assert X["step_3_active"].tolist() == [0, 0]
    ablated = schema.without_transitions()
    assert_frame_equal(build_features((ProcessSequence(steps, (0,)),), ablated),
                       X.loc[:0, list(ablated.columns)])


def test_fit_local_imputation_audit(caplog: pytest.LogCaptureFixture) -> None:
    """Both empty states get warned and holdout-only observations survive transformation."""
    schema = FeatureSchema(2)
    train = build_features((ProcessSequence((DepositionStep("TiN", "recipe", 0, None, 2),), ()),), schema)
    held_out = ProcessSequence((DepositionStep("TiAlN", "unseen", None, 120, None),
                               TreatmentStep("anneal", 4, 5)), (1,))
    holdout = build_features((held_out,), schema)
    warn_empty_columns(train, schema, {"model": "ridge", "stage": "outer", "fold": 1})
    messages = [record.message for record in caplog.records if record.levelname == "WARNING"]
    assert len(messages) == 1
    assert "ALL_MISSING_NUMERIC_FEATURES" in messages[0]
    assert "'step_1_temperature': 'applicable_but_unrecorded'" in messages[0]
    assert "'step_2_time': 'no_applicable_rows'" in messages[0]
    assert "'step_1_time'" not in messages[0]
    assert "model=ridge stage=outer fold=1 training_rows=1" in messages[0]

    pipeline = make_preprocessor(schema, scale_numeric=True)
    numeric = pipeline.fit(train).named_transformers_["numeric"]
    imputed = numeric.named_steps["impute"].transform(holdout.loc[:, list(schema.numeric_columns)])
    np.testing.assert_array_equal(imputed[0], [0, 120, 2, 4, 5, 0])
    transformed = pipeline.transform(holdout)
    assert np.isfinite(transformed).all()
    flags = transformed[:, -len(schema.flag_columns):]
    np.testing.assert_array_equal(flags, holdout.loc[:, list(schema.flag_columns)].to_numpy())
