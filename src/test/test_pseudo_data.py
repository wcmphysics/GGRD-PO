"""Comprehensive unit and integration test suite for wafer pseudo-data generator.

This test suite verifies:
1. Configuration schema loading, strict Pydantic validation, extra/missing key rejection, and key alignment.
2. Long-table structure, group replicate consistency, and wafer identifier formatting.
3. Sequence step bounds, operation sampling, transition rows, and vacuum break rules.
4. Missing parameter dropout behavior (boundary rates 0.0, 1.0, and structural preservation).
5. Physical thermal stress model, deterministic calculation, noise, and outlier bounds.
6. Determinism and reproducibility across seeds.
7. End-to-end CLI entrypoint execution and file persistence.
8. Implementation plan preservation.
"""

from copy import deepcopy
import math
from pathlib import Path
import re
import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
import yaml

from src.config.pseudo_data_schema import (
    DepositionConfig,
    PseudoDataConfig,
    StressModelConfig,
    TransitionConfig,
    TreatmentConfig,
    load_pseudo_data_config,
)
from src.main import main
from src.utility.pseudo_data import generate_pseudo_data, save_pseudo_data

DEFAULT_CONFIG_PATH = Path("src/config/pseudo_data_config.yaml")


@pytest.fixture
def default_config() -> PseudoDataConfig:
    """Fixture providing the default validated configuration instance."""
    return load_pseudo_data_config(DEFAULT_CONFIG_PATH)


@pytest.fixture
def raw_config_dict() -> dict:
    """Fixture providing raw configuration dictionary from YAML."""
    with open(DEFAULT_CONFIG_PATH, mode="r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


# ---------------------------------------------------------------------------
# 1. Configuration & Schema Validation Tests
# ---------------------------------------------------------------------------


def test_load_default_config_success(default_config: PseudoDataConfig) -> None:
    """Verify that default YAML configuration loads without errors with expected types."""
    expected_scalars = {
        "random_seed": 42,
        "n_groups": 100,
        "wafers_per_group_min": 1,
        "wafers_per_group_max": 5,
        "wafer_id_prefix": "ABC",
        "wafer_id_digits": 6,
        "total_steps_min": 3,
        "total_steps_max": 5,
        "min_depositions": 1,
        "max_treatments": 2,
        "missing_parameter_rate": 0.05,
    }
    for field_name, expected_value in expected_scalars.items():
        assert getattr(default_config, field_name) == expected_value

    assert "TiN" in default_config.deposition.materials
    assert "anneal" in default_config.treatment.recipes
    assert default_config.transition.vacuum_break_before_measure == 1
    assert default_config.stress_model.noise_std >= 0
    assert 0 <= default_config.stress_model.outlier_rate <= 1


def test_schema_rejects_extra_keys(raw_config_dict: dict) -> None:
    """Verify ConfigDict(extra='forbid') rejects unknown top-level and nested keys."""
    extra_top = deepcopy(raw_config_dict)
    extra_top["unexpected_field"] = "unexpected_value"
    with pytest.raises(ValidationError) as exc_info:
        PseudoDataConfig(**extra_top)
    assert "extra_forbidden" in str(exc_info.value) or "Extra inputs are not permitted" in str(exc_info.value)

    extra_nested = deepcopy(raw_config_dict)
    extra_nested["deposition"]["unexpected_dep_field"] = 999
    with pytest.raises(ValidationError):
        PseudoDataConfig(**extra_nested)


def test_schema_rejects_missing_keys(raw_config_dict: dict) -> None:
    """Verify that schema rejects configuration missing required fields."""
    missing_top = deepcopy(raw_config_dict)
    del missing_top["random_seed"]
    with pytest.raises(ValidationError):
        PseudoDataConfig(**missing_top)

    missing_nested = deepcopy(raw_config_dict)
    del missing_nested["stress_model"]["noise_std"]
    with pytest.raises(ValidationError):
        PseudoDataConfig(**missing_nested)


def test_schema_models_have_no_defaults() -> None:
    """Verify AGENTS.md §5 mandate: all Pydantic fields must have is_required() == True."""
    models_to_check = [
        PseudoDataConfig,
        DepositionConfig,
        TreatmentConfig,
        TransitionConfig,
        StressModelConfig,
    ]
    for model_class in models_to_check:
        for field_name, field_info in model_class.model_fields.items():
            assert field_info.is_required(), (
                f"Model {model_class.__name__} field '{field_name}' must not have a default value."
            )


@pytest.mark.parametrize(
    ("section", "field", "invalid_value"),
    [
        ("deposition", "time_min", 700),
        ("deposition", "temperature_min", 1200),
        ("deposition", "thickness_min", 60),
        ("treatment", "time_min", 700),
        ("treatment", "temperature_min", 1200),
        ("stress_model", "noise_std", -0.01),
        ("stress_model", "outlier_rate", 1.5),
        ("stress_model", "outlier_min_abs", 12.0),
        ("transition", "vacuum_break_before_measure", 2),
    ],
)
def test_schema_rejects_invalid_field_values(
    raw_config_dict: dict, section: str, field: str, invalid_value: object
) -> None:
    """Verify validation on boundary values and inverted bounds."""
    modified_config = deepcopy(raw_config_dict)
    modified_config[section][field] = invalid_value
    with pytest.raises((ValidationError, ValueError)):
        PseudoDataConfig(**modified_config)


def test_schema_rejects_material_key_mismatch(raw_config_dict: dict) -> None:
    """Verify cross-field validator catches mismatched materials and stress constants."""
    mismatched = deepcopy(raw_config_dict)
    mismatched["deposition"]["materials"]["ExtraMetal"] = ["recipe_v1"]
    with pytest.raises(ValidationError) as exc_info:
        PseudoDataConfig(**mismatched)
    assert "Key mismatch between deposition.materials and stress_model.material_constants" in str(exc_info.value)


def test_schema_rejects_treatment_key_mismatch(raw_config_dict: dict) -> None:
    """Verify cross-field validator catches mismatched treatments and stress constants."""
    mismatched = deepcopy(raw_config_dict)
    mismatched["treatment"]["recipes"]["plasma_clean"] = ["recipe_v1"]
    with pytest.raises(ValidationError) as exc_info:
        PseudoDataConfig(**mismatched)
    assert "Key mismatch between treatment.recipes and stress_model.treatment_constants" in str(exc_info.value)


def test_load_nonexistent_config_raises_file_not_found() -> None:
    """Verify that loading a non-existent configuration path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pseudo_data_config("nonexistent_path_to_config.yaml")


def test_load_invalid_yaml_raises_value_error(tmp_path: Path) -> None:
    """Verify that loading non-dictionary YAML content raises ValueError."""
    invalid_file = tmp_path / "invalid_content.yaml"
    invalid_file.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pseudo_data_config(invalid_file)


@pytest.mark.parametrize("invalid_seed", [-1, -42, -999999])
def test_negative_seed_rejected(raw_config_dict: dict, invalid_seed: int) -> None:
    """Verify that negative random_seed values are rejected by schema validation."""
    modified_config = deepcopy(raw_config_dict)
    modified_config["random_seed"] = invalid_seed
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        PseudoDataConfig(**modified_config)
    error_message = str(exc_info.value).lower()
    assert "random_seed" in error_message
    assert ">= 0" in error_message or "non-negative" in error_message


@pytest.mark.parametrize("valid_seed", [0, 1, 42])
def test_valid_seed_boundary(raw_config_dict: dict, valid_seed: int) -> None:
    """Verify that non-negative random_seed values (including boundary 0) are accepted."""
    modified_config = deepcopy(raw_config_dict)
    modified_config["random_seed"] = valid_seed
    config_obj = PseudoDataConfig(**modified_config)
    assert config_obj.random_seed == valid_seed


@pytest.mark.parametrize(
    ("stress_min", "stress_max", "outlier_min_abs"),
    [
        (1.0, 2.0, 0.5),    # Unipolar positive bounds (Challenger 2 reproduction)
        (0.0, 5.0, 0.5),    # Zero stress_min bound
        (2.0, 10.0, 1.0),   # Strictly positive lower bound
        (-2.0, -1.0, 0.5),  # Unipolar negative bounds
        (-5.0, 0.0, 0.5),   # Zero stress_max bound
        (-10.0, -0.5, 0.1), # Strictly negative upper bound
        (5.0, -5.0, 1.5),   # Inverted bounds (min > max)
        (2.0, 2.0, 1.0),    # Equal bounds (min == max)
    ],
)
def test_unipolar_bounds_rejected(
    raw_config_dict: dict, stress_min: float, stress_max: float, outlier_min_abs: float
) -> None:
    """Verify schema rejects unipolar or inverted stress bounds unable to bracket zero."""
    modified_config = deepcopy(raw_config_dict)
    modified_config["stress_model"]["stress_min"] = stress_min
    modified_config["stress_model"]["stress_max"] = stress_max
    modified_config["stress_model"]["outlier_min_abs"] = outlier_min_abs
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        PseudoDataConfig(**modified_config)
    error_message = str(exc_info.value).lower()
    assert "stress" in error_message


@pytest.mark.parametrize(
    ("stress_min", "stress_max", "outlier_min_abs"),
    [
        (-10.0, 10.0, 10.0),  # Equal to both bounds (interval width zero)
        (-10.0, 10.0, 12.0),  # Greater than both bounds
        (-2.0, 5.0, 2.0),     # Equal to abs(stress_min)
        (-2.0, 5.0, 2.5),     # Greater than abs(stress_min)
        (-5.0, 2.0, 2.0),     # Equal to abs(stress_max)
        (-5.0, 2.0, 2.5),     # Greater than abs(stress_max)
        (-10.0, 10.0, -0.5),  # Negative outlier_min_abs
        (-10.0, 10.0, -2.0),  # Negative outlier_min_abs
    ],
)
def test_invalid_outlier_rejected(
    raw_config_dict: dict, stress_min: float, stress_max: float, outlier_min_abs: float
) -> None:
    """Verify schema rejects outlier_min_abs when not strictly less than bounds or negative."""
    modified_config = deepcopy(raw_config_dict)
    modified_config["stress_model"]["stress_min"] = stress_min
    modified_config["stress_model"]["stress_max"] = stress_max
    modified_config["stress_model"]["outlier_min_abs"] = outlier_min_abs
    with pytest.raises((ValidationError, ValueError)) as exc_info:
        PseudoDataConfig(**modified_config)
    error_message = str(exc_info.value).lower()
    assert "outlier_min_abs" in error_message or "stress" in error_message


@pytest.mark.parametrize(
    ("stress_min", "stress_max", "outlier_min_abs"),
    [
        (-10.0, 10.0, 0.0),  # Zero outlier threshold (contiguous lobes)
        (-2.0, 5.0, 1.9),    # Near upper limit of abs(stress_min)
        (-5.0, 2.0, 1.9),    # Near upper limit of abs(stress_max)
        (-0.5, 0.5, 0.1),    # Narrow symmetrical bounds
    ],
)
def test_valid_outlier_bounds(
    raw_config_dict: dict, stress_min: float, stress_max: float, outlier_min_abs: float
) -> None:
    """Verify schema accepts valid bipolar bounds with outlier_min_abs strictly inside bounds."""
    modified_config = deepcopy(raw_config_dict)
    modified_config["stress_model"]["stress_min"] = stress_min
    modified_config["stress_model"]["stress_max"] = stress_max
    modified_config["stress_model"]["outlier_min_abs"] = outlier_min_abs
    config_obj = PseudoDataConfig(**modified_config)
    assert config_obj.stress_model.stress_min == stress_min
    assert config_obj.stress_model.stress_max == stress_max
    assert config_obj.stress_model.outlier_min_abs == outlier_min_abs


# ---------------------------------------------------------------------------
# 2. Long-Table Format & Replicate Structure Tests
# ---------------------------------------------------------------------------


def test_generate_dataframe_columns_and_group_count(default_config: PseudoDataConfig) -> None:
    """Verify exact DataFrame columns, total groups, and wafer counts."""
    dataframe = generate_pseudo_data(default_config)
    expected_columns = ["group", "wafer_id", "step", "operation", "parameter", "value"]
    assert list(dataframe.columns) == expected_columns
    assert dataframe["group"].nunique() == default_config.n_groups

    # Check wafer count per group is within [wafers_per_group_min, wafers_per_group_max]
    group_wafer_counts = dataframe.groupby("group")["wafer_id"].nunique()
    assert (group_wafer_counts >= default_config.wafers_per_group_min).all()
    assert (group_wafer_counts <= default_config.wafers_per_group_max).all()


def test_wafer_id_format_and_uniqueness(default_config: PseudoDataConfig) -> None:
    """Verify wafer IDs match prefix, digits, and are globally unique."""
    dataframe = generate_pseudo_data(default_config)
    unique_wafers = dataframe["wafer_id"].unique()
    pattern = rf"^{default_config.wafer_id_prefix}\d{{{default_config.wafer_id_digits}}}$"
    regex = re.compile(pattern)

    for wafer_id in unique_wafers:
        assert regex.match(str(wafer_id)), f"Wafer ID '{wafer_id}' does not match pattern {pattern}"

    total_wafers_across_groups = dataframe.groupby("group")["wafer_id"].nunique().sum()
    assert len(unique_wafers) == total_wafers_across_groups


def test_group_wafers_share_identical_process_history(default_config: PseudoDataConfig) -> None:
    """Verify wafers in the same group have identical process steps, parameters, and values."""
    dataframe = generate_pseudo_data(default_config)

    for group_id, group_df in dataframe.groupby("group"):
        wafers_in_group = group_df["wafer_id"].unique()
        if len(wafers_in_group) < 2:
            continue

        # Extract process history excluding final measurement row
        process_df = group_df[group_df["operation"] != "measurement"]
        reference_wafer = wafers_in_group[0]
        reference_history = process_df[process_df["wafer_id"] == reference_wafer][
            ["step", "operation", "parameter", "value"]
        ].reset_index(drop=True)

        for other_wafer in wafers_in_group[1:]:
            other_history = process_df[process_df["wafer_id"] == other_wafer][
                ["step", "operation", "parameter", "value"]
            ].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                reference_history,
                other_history,
                obj=f"Process history mismatch between {reference_wafer} and {other_wafer} in group {group_id}",
            )


# ---------------------------------------------------------------------------
# 3. Sequence & Transition Constraint Tests
# ---------------------------------------------------------------------------


def test_sequence_step_counts_and_measurement_step(default_config: PseudoDataConfig) -> None:
    """Verify step counts lie in [total_steps_min, total_steps_max] and last step is measurement."""
    dataframe = generate_pseudo_data(default_config)

    for wafer_id, wafer_df in dataframe.groupby("wafer_id"):
        max_step = wafer_df["step"].max()
        assert default_config.total_steps_min <= max_step <= default_config.total_steps_max

        final_rows = wafer_df[wafer_df["step"] == max_step]
        assert len(final_rows) == 1
        final_row = final_rows.iloc[0]
        assert final_row["operation"] == "measurement"
        assert final_row["parameter"] == "stress"


def test_deposition_and_treatment_count_constraints(default_config: PseudoDataConfig) -> None:
    """Verify >= min_depositions and <= max_treatments per wafer."""
    dataframe = generate_pseudo_data(default_config)

    for _, wafer_df in dataframe.groupby("wafer_id"):
        # Count distinct process steps of each operation
        process_steps_df = wafer_df[wafer_df["operation"].isin(["deposition", "treatment"])]
        operation_per_step = process_steps_df.groupby("step")["operation"].first()

        deposition_count = int((operation_per_step == "deposition").sum())
        treatment_count = int((operation_per_step == "treatment").sum())

        assert deposition_count >= default_config.min_depositions
        assert treatment_count <= default_config.max_treatments


def test_deposition_allowed_after_step_one(default_config: PseudoDataConfig) -> None:
    """Verify that deposition is not restricted to step 1 across all generated sequences."""
    dataframe = generate_pseudo_data(default_config)
    process_df = dataframe[dataframe["operation"].isin(["deposition", "treatment"])]
    first_ops = process_df[process_df["step"] == 1].groupby("group")["operation"].first()

    # Across 100 groups, treatment should occur at step 1 in some groups
    treatment_first_count = int((first_ops == "treatment").sum())
    assert treatment_first_count > 0, "Expected at least one group to begin with treatment at step 1."


def test_transition_row_presence_and_vacuum_break(default_config: PseudoDataConfig) -> None:
    """Verify transition rows follow each process step and vacuum_break == 1 before measurement."""
    dataframe = generate_pseudo_data(default_config)

    for _, wafer_df in dataframe.groupby("wafer_id"):
        max_step = wafer_df["step"].max()
        process_step_numbers = sorted(wafer_df[wafer_df["step"] < max_step]["step"].unique())

        for step_number in process_step_numbers:
            step_rows = wafer_df[wafer_df["step"] == step_number]
            transition_rows = step_rows[
                (step_rows["operation"] == "transition") & (step_rows["parameter"] == "vacuum_break")
            ]
            assert len(transition_rows) == 1, f"Missing or duplicate transition row at step {step_number}"

            transition_value = int(transition_rows.iloc[0]["value"])
            assert transition_value in (0, 1)

            # Step immediately preceding measurement must have vacuum_break == 1
            if step_number == max_step - 1:
                assert transition_value == default_config.transition.vacuum_break_before_measure


# ---------------------------------------------------------------------------
# 4. Missing Parameter Handling Tests
# ---------------------------------------------------------------------------


def test_missing_parameter_rate_zero(raw_config_dict: dict) -> None:
    """Verify missing_parameter_rate == 0.0 retains all numeric process parameters."""
    zero_drop_config = deepcopy(raw_config_dict)
    zero_drop_config["missing_parameter_rate"] = 0.0
    zero_drop_config["n_groups"] = 10
    config_obj = PseudoDataConfig(**zero_drop_config)

    dataframe = generate_pseudo_data(config_obj)
    for _, step_df in dataframe[dataframe["operation"] == "deposition"].groupby(["wafer_id", "step"]):
        parameters_in_step = set(step_df["parameter"])
        assert {"material", "recipe", "time", "temperature", "thickness"}.issubset(parameters_in_step)

    for _, step_df in dataframe[dataframe["operation"] == "treatment"].groupby(["wafer_id", "step"]):
        parameters_in_step = set(step_df["parameter"])
        assert {"recipe", "time", "temperature"}.issubset(parameters_in_step)


def test_missing_parameter_rate_one(raw_config_dict: dict) -> None:
    """Verify missing_parameter_rate == 1.0 drops all numeric parameters while keeping structural rows."""
    full_drop_config = deepcopy(raw_config_dict)
    full_drop_config["missing_parameter_rate"] = 1.0
    full_drop_config["n_groups"] = 10
    config_obj = PseudoDataConfig(**full_drop_config)

    dataframe = generate_pseudo_data(config_obj)

    # Numeric process parameters must not appear anywhere
    droppable_numeric = {"time", "temperature", "thickness"}
    process_parameters = set(dataframe[dataframe["operation"].isin(["deposition", "treatment"])]["parameter"])
    assert process_parameters.isdisjoint(droppable_numeric)

    # Structural parameters must still be 100% present
    dep_parameters = set(dataframe[dataframe["operation"] == "deposition"]["parameter"])
    assert dep_parameters == {"material", "recipe"}
    trt_parameters = set(dataframe[dataframe["operation"] == "treatment"]["parameter"])
    assert trt_parameters == {"recipe"}


def _check_step_structural_parameters(step_dataframe: pd.DataFrame) -> None:
    """Helper to verify structural rows in a single step."""
    operations_present = set(step_dataframe["operation"])
    if "deposition" in operations_present:
        dep_params = set(step_dataframe[step_dataframe["operation"] == "deposition"]["parameter"])
        assert {"material", "recipe"}.issubset(dep_params)
        assert "vacuum_break" in set(step_dataframe[step_dataframe["operation"] == "transition"]["parameter"])
    elif "treatment" in operations_present:
        trt_params = set(step_dataframe[step_dataframe["operation"] == "treatment"]["parameter"])
        assert "recipe" in trt_params
        assert "vacuum_break" in set(step_dataframe[step_dataframe["operation"] == "transition"]["parameter"])
    elif "measurement" in operations_present:
        meas_params = set(step_dataframe[step_dataframe["operation"] == "measurement"]["parameter"])
        assert "stress" in meas_params


def test_structural_parameters_never_dropped(default_config: PseudoDataConfig) -> None:
    """Verify structural rows (material, recipe, transition, measurement) are never omitted."""
    dataframe = generate_pseudo_data(default_config)
    for _, wafer_df in dataframe.groupby("wafer_id"):
        for _, step_df in wafer_df.groupby("step"):
            _check_step_structural_parameters(step_df)


# ---------------------------------------------------------------------------
# 5. Physics Model, Noise, and Outlier Bounds Tests
# ---------------------------------------------------------------------------


def test_deterministic_stress_formula_zero_noise(raw_config_dict: dict) -> None:
    """Verify exact formula sum(A * T * d) + sum(B * t * T) when noise and outliers are disabled."""
    deterministic_dict = deepcopy(raw_config_dict)
    deterministic_dict["n_groups"] = 10
    deterministic_dict["missing_parameter_rate"] = 0.0  # keep parameters to recalculate
    deterministic_dict["stress_model"]["noise_std"] = 0.0
    deterministic_dict["stress_model"]["outlier_rate"] = 0.0
    config_obj = PseudoDataConfig(**deterministic_dict)

    dataframe = generate_pseudo_data(config_obj)
    mat_constants = config_obj.stress_model.material_constants
    trt_constants = config_obj.stress_model.treatment_constants

    for _, wafer_df in dataframe.groupby("wafer_id"):
        expected_base_stress = 0.0
        for _, step_df in wafer_df.groupby("step"):
            op = step_df["operation"].iloc[0]
            param_map = dict(zip(step_df["parameter"], step_df["value"]))
            if op == "deposition":
                mat = param_map["material"]
                coeff = mat_constants[mat]
                expected_base_stress += coeff * float(param_map["temperature"]) * float(param_map["thickness"])
            elif op == "treatment":
                # Determine treatment type from recipe name prefix
                recipe_name = param_map["recipe"]
                trt_type = "anneal" if "anneal" in recipe_name else "oxidation"
                coeff = trt_constants[trt_type]
                expected_base_stress += coeff * float(param_map["time"]) * float(param_map["temperature"])

        stress_measured = float(wafer_df[wafer_df["parameter"] == "stress"]["value"].iloc[0])
        expected_rounded = round(expected_base_stress, config_obj.stress_model.decimal_places)
        assert math.isclose(stress_measured, expected_rounded, abs_tol=1e-4)


def test_stress_values_strictly_bounded(default_config: PseudoDataConfig) -> None:
    """Verify 100% of measured stress values lie strictly in [stress_min, stress_max]."""
    dataframe = generate_pseudo_data(default_config)
    stress_series = dataframe[dataframe["parameter"] == "stress"]["value"].astype(float)

    assert (stress_series >= default_config.stress_model.stress_min).all()
    assert (stress_series <= default_config.stress_model.stress_max).all()


def test_outlier_values_satisfy_min_abs_bound(raw_config_dict: dict) -> None:
    """Verify with outlier_rate == 1.0 that all stress values satisfy |stress| >= outlier_min_abs."""
    outlier_dict = deepcopy(raw_config_dict)
    outlier_dict["n_groups"] = 20
    outlier_dict["stress_model"]["outlier_rate"] = 1.0
    config_obj = PseudoDataConfig(**outlier_dict)

    dataframe = generate_pseudo_data(config_obj)
    stress_series = dataframe[dataframe["parameter"] == "stress"]["value"].astype(float)
    min_abs = config_obj.stress_model.outlier_min_abs

    assert (stress_series.abs() >= min_abs - 1e-6).all()


def test_noise_and_outlier_distribution(default_config: PseudoDataConfig) -> None:
    """Verify empirical outlier frequency is within expected binomial bounds around 5%."""
    dataframe = generate_pseudo_data(default_config)
    stress_series = dataframe[dataframe["parameter"] == "stress"]["value"].astype(float)
    outlier_min_abs = default_config.stress_model.outlier_min_abs

    # Outliers satisfy |stress| >= 1.5
    is_outlier = stress_series.abs() >= outlier_min_abs
    empirical_rate = float(is_outlier.mean())
    total_samples = len(stress_series)

    # 3-sigma tolerance for Binomial(N=total_samples, p=0.05)
    std_error = math.sqrt(0.05 * 0.95 / total_samples)
    assert abs(empirical_rate - 0.05) < 3.5 * std_error


def test_defensive_stress_guard(default_config: PseudoDataConfig) -> None:
    """Verify _sample_wafer_stress defensive guard raises ValueError on invalid intervals."""
    from src.utility.pseudo_data import _sample_wafer_stress

    rng = np.random.default_rng(42)
    bypassed_config = deepcopy(default_config)
    object.__setattr__(bypassed_config.stress_model, "stress_min", 1.0)
    object.__setattr__(bypassed_config.stress_model, "stress_max", 2.0)
    object.__setattr__(bypassed_config.stress_model, "outlier_min_abs", 0.5)
    object.__setattr__(bypassed_config.stress_model, "outlier_rate", 1.0)

    with pytest.raises(ValueError) as exc_info:
        _sample_wafer_stress(0.0, bypassed_config, rng)
    error_message = str(exc_info.value).lower()
    assert "outlier interval" in error_message or "width" in error_message


# ---------------------------------------------------------------------------
# 6. Determinism & Reproducibility Tests
# ---------------------------------------------------------------------------


def test_identical_seed_produces_identical_dataframe(default_config: PseudoDataConfig) -> None:
    """Verify bit-exact reproducibility when executing with the same random seed."""
    dataframe_1 = generate_pseudo_data(default_config)
    dataframe_2 = generate_pseudo_data(default_config)
    pd.testing.assert_frame_equal(dataframe_1, dataframe_2)


def test_different_seeds_produce_different_dataframes(raw_config_dict: dict) -> None:
    """Verify that different random seeds produce different dataframes."""
    config_dict_1 = deepcopy(raw_config_dict)
    config_dict_1["random_seed"] = 42
    config_dict_2 = deepcopy(raw_config_dict)
    config_dict_2["random_seed"] = 999

    dataframe_1 = generate_pseudo_data(PseudoDataConfig(**config_dict_1))
    dataframe_2 = generate_pseudo_data(PseudoDataConfig(**config_dict_2))

    assert not dataframe_1.equals(dataframe_2)


# ---------------------------------------------------------------------------
# 7. File I/O & Entrypoint CLI Tests
# ---------------------------------------------------------------------------


def test_save_pseudo_data_roundtrip(default_config: PseudoDataConfig, tmp_path: Path) -> None:
    """Verify saving DataFrame to CSV and loading back preserves all records."""
    dataframe = generate_pseudo_data(default_config)
    output_file = tmp_path / "test_output.csv"
    saved_path = save_pseudo_data(dataframe, output_file)

    assert saved_path.is_file()
    loaded_df = pd.read_csv(saved_path, dtype={"value": str})
    assert len(loaded_df) == len(dataframe)
    assert list(loaded_df.columns) == list(dataframe.columns)


def test_main_cli_headless_execution(tmp_path: Path) -> None:
    """Verify main entrypoint executes via CLI arguments with headless flag."""
    custom_output = tmp_path / "cli_pseudo_data.csv"
    with open(DEFAULT_CONFIG_PATH, mode="r", encoding="utf-8") as file_handle:
        config_data = yaml.safe_load(file_handle)

    config_data["output_path"] = str(custom_output).replace("\\", "/")
    config_data["n_groups"] = 5
    custom_yaml = tmp_path / "cli_config.yaml"
    with open(custom_yaml, mode="w", encoding="utf-8") as file_handle:
        yaml.safe_dump(config_data, file_handle)

    returned_path = main(["--config", str(custom_yaml), "--headless"])
    assert returned_path.is_file()
    assert returned_path.resolve() == custom_output.resolve()

    persisted_df = pd.read_csv(returned_path)
    assert persisted_df["group"].nunique() == 5


# ---------------------------------------------------------------------------
# 8. Plan Preservation Test
# ---------------------------------------------------------------------------


def test_plan_preserved_in_doc() -> None:
    """Verify R1: approved implementation plan is preserved at doc/pseudo_data_plan.md."""
    plan_path = Path("doc/pseudo_data_plan.md")
    assert plan_path.is_file(), "doc/pseudo_data_plan.md does not exist."
    content = plan_path.read_text(encoding="utf-8")
    assert len(content) > 500, "doc/pseudo_data_plan.md appears too short or truncated."
    assert "Pseudo-Data Generator Implementation Plan" in content
