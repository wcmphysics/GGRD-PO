"""Wafer pseudo-data generator for process histories and thermal stress measurements.

This module provides functionality to generate and persist synthetic wafer processing histories
and thermal stress measurements in the long-table format specified in doc/data_example.md.

Physical Thermal Stress Model:
------------------------------
For each wafer group, the deterministic base stress is calculated from complete process parameters:
    stress_base = sum_{i in dep} A_{mat_i} * T_i * d_i + sum_{j in trt} B_{type_j} * t_j * T_j
where:
    - Deposition: A is material constant, T is temperature (°C), d is thickness (nm).
    - Treatment: B is treatment constant, t is time (s), T is temperature (°C).

Measurement Noise & Outliers:
-----------------------------
Wafers within each group represent repeated experiments with identical process parameters.
Individual wafer stress measurements are generated as:
    - Regular noise (1 - outlier_rate): Gaussian noise N(0, noise_std), clipped to [stress_min, stress_max].
    - Extreme outlier (outlier_rate): Uniform in [stress_min, -outlier_min_abs] U [outlier_min_abs, stress_max],
      clipped to [stress_min, stress_max].
Final values are rounded to config.stress_model.decimal_places.

Missing Parameter Emulation:
----------------------------
Numeric process parameters (time, temperature, thickness) are dropped independently at missing_parameter_rate
by omitting corresponding table rows. Structural parameters (material, recipe, transition, measurement) are
never dropped. Base thermal stress is always computed from unmasked parameters before dropping.
"""

import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from src.config.pseudo_data_schema import PseudoDataConfig


def _sample_process_operations(config: PseudoDataConfig, rng: np.random.Generator) -> list[str]:
    """Sample an ordered sequence of process operations.

    Selects total_steps uniformly, calculates allowable treatment counts ensuring at least
    min_depositions and at most max_treatments, and samples valid operation sequences uniformly.
    """
    total_steps = int(rng.integers(config.total_steps_min, config.total_steps_max, endpoint=True))
    process_steps_count = total_steps - 1
    max_treatments_allowed = min(config.max_treatments, process_steps_count - config.min_depositions)

    # Binomial weighting ensures all valid operation permutations have equal probability
    treatment_options = list(range(0, max_treatments_allowed + 1))
    combination_weights = [math.comb(process_steps_count, k) for k in treatment_options]
    total_weight = sum(combination_weights)
    probabilities = [w / total_weight for w in combination_weights]

    chosen_treatments = int(rng.choice(treatment_options, p=probabilities))
    operations = ["treatment"] * chosen_treatments + ["deposition"] * (process_steps_count - chosen_treatments)
    shuffled_operations = rng.permutation(operations).tolist()
    return [str(op) for op in shuffled_operations]


def _sample_deposition_step(config: PseudoDataConfig, rng: np.random.Generator) -> dict[str, Any]:
    """Sample raw physical parameters for a deposition step."""
    materials_list = list(config.deposition.materials.keys())
    chosen_material = str(rng.choice(materials_list))
    recipe_list = config.deposition.materials[chosen_material]
    chosen_recipe = str(rng.choice(recipe_list))

    deposition_config = config.deposition
    time_value = int(rng.integers(deposition_config.time_min, deposition_config.time_max, endpoint=True))
    temperature_value = int(
        rng.integers(deposition_config.temperature_min, deposition_config.temperature_max, endpoint=True)
    )
    thickness_value = int(
        rng.integers(deposition_config.thickness_min, deposition_config.thickness_max, endpoint=True)
    )

    return {
        "operation": "deposition",
        "material": chosen_material,
        "recipe": chosen_recipe,
        "time": time_value,
        "temperature": temperature_value,
        "thickness": thickness_value,
    }


def _sample_treatment_step(config: PseudoDataConfig, rng: np.random.Generator) -> dict[str, Any]:
    """Sample raw physical parameters for a treatment step."""
    treatment_list = list(config.treatment.recipes.keys())
    chosen_treatment = str(rng.choice(treatment_list))
    recipe_list = config.treatment.recipes[chosen_treatment]
    chosen_recipe = str(rng.choice(recipe_list))

    treatment_config = config.treatment
    time_value = int(rng.integers(treatment_config.time_min, treatment_config.time_max, endpoint=True))
    temperature_value = int(
        rng.integers(treatment_config.temperature_min, treatment_config.temperature_max, endpoint=True)
    )

    return {
        "operation": "treatment",
        "treatment_type": chosen_treatment,
        "recipe": chosen_recipe,
        "time": time_value,
        "temperature": temperature_value,
    }


def _compute_base_stress(process_steps: list[dict[str, Any]], config: PseudoDataConfig) -> float:
    """Compute deterministic group base thermal stress from complete parameters."""
    total_stress = 0.0
    for step_data in process_steps:
        operation_name = step_data["operation"]
        if operation_name == "deposition":
            material_name = step_data["material"]
            coefficient = config.stress_model.material_constants[material_name]
            temperature_val = float(step_data["temperature"])
            thickness_val = float(step_data["thickness"])
            total_stress += coefficient * temperature_val * thickness_val
        elif operation_name == "treatment":
            treatment_type = step_data["treatment_type"]
            coefficient = config.stress_model.treatment_constants[treatment_type]
            time_val = float(step_data["time"])
            temperature_val = float(step_data["temperature"])
            total_stress += coefficient * time_val * temperature_val
    return total_stress


def _drop_step_parameters(
    step_data: dict[str, Any], missing_rate: float, rng: np.random.Generator
) -> list[tuple[str, str, str]]:
    """Determine retained parameter rows for a process step after missing data dropout.

    Returns tuples of (operation, parameter_name, string_value).
    Structural parameters are always kept; numeric parameters are dropped at missing_rate.
    """
    operation_name = str(step_data["operation"])
    retained_rows: list[tuple[str, str, str]] = []

    if operation_name == "deposition":
        retained_rows.append((operation_name, "material", str(step_data["material"])))
        retained_rows.append((operation_name, "recipe", str(step_data["recipe"])))
        for numeric_parameter in ("time", "temperature", "thickness"):
            if float(rng.random()) >= missing_rate:
                retained_rows.append((operation_name, numeric_parameter, str(step_data[numeric_parameter])))
    elif operation_name == "treatment":
        retained_rows.append((operation_name, "recipe", str(step_data["recipe"])))
        for numeric_parameter in ("time", "temperature"):
            if float(rng.random()) >= missing_rate:
                retained_rows.append((operation_name, numeric_parameter, str(step_data[numeric_parameter])))

    return retained_rows


def _sample_wafer_stress(stress_base: float, config: PseudoDataConfig, rng: np.random.Generator) -> float:
    """Sample individual wafer stress with regular Gaussian noise or extreme outlier."""
    stress_config = config.stress_model
    is_outlier = bool(float(rng.random()) < stress_config.outlier_rate)

    if not is_outlier:
        noise = float(rng.normal(0.0, stress_config.noise_std))
        clamped_stress = float(np.clip(stress_base + noise, stress_config.stress_min, stress_config.stress_max))
        return round(clamped_stress, stress_config.decimal_places)

    # Disjoint outlier sampling intervals: [stress_min, -outlier_min_abs] U [outlier_min_abs, stress_max]
    negative_width = -stress_config.outlier_min_abs - stress_config.stress_min
    positive_width = stress_config.stress_max - stress_config.outlier_min_abs
    total_width = negative_width + positive_width

    if total_width <= 0.0 or negative_width <= 0.0 or positive_width <= 0.0:
        raise ValueError(
            f"Invalid outlier intervals: negative_width={negative_width}, "
            f"positive_width={positive_width}, total_width={total_width}; "
            "expected positive interval widths."
        )

    negative_probability = negative_width / total_width

    if float(rng.random()) < negative_probability:
        sampled_outlier = float(rng.uniform(stress_config.stress_min, -stress_config.outlier_min_abs))
    else:
        sampled_outlier = float(rng.uniform(stress_config.outlier_min_abs, stress_config.stress_max))

    clipped_outlier = float(np.clip(sampled_outlier, stress_config.stress_min, stress_config.stress_max))
    return round(clipped_outlier, stress_config.decimal_places)


def _format_wafer_id(prefix: str, digits: int, counter: int) -> str:
    """Format unique sequential wafer identifier."""
    return f"{prefix}{counter:0{digits}d}"


def _build_group_template_rows(
    process_steps: list[dict[str, Any]], config: PseudoDataConfig, rng: np.random.Generator
) -> list[tuple[int, str, str, str]]:
    """Build shared step rows (step, operation, parameter, value) for a group."""
    group_rows: list[tuple[int, str, str, str]] = []
    total_process_steps = len(process_steps)

    for step_index, step_data in enumerate(process_steps, start=1):
        retained = _drop_step_parameters(step_data, config.missing_parameter_rate, rng)
        for operation_name, parameter_name, parameter_value in retained:
            group_rows.append((step_index, operation_name, parameter_name, parameter_value))

        # Transition row appended after each process step
        if step_index == total_process_steps:
            vacuum_break_value = config.transition.vacuum_break_before_measure
        else:
            vacuum_break_value = int(rng.choice(config.transition.vacuum_break_values))
        group_rows.append((step_index, "transition", "vacuum_break", str(vacuum_break_value)))

    return group_rows


def generate_pseudo_data(config: PseudoDataConfig) -> pd.DataFrame:
    """Generate wafer processing histories and measurements dataset.

    Parameters
    ----------
    config : PseudoDataConfig
        Validated configuration instance.

    Returns
    -------
    pd.DataFrame
        Long-table DataFrame with columns:
        ['group', 'wafer_id', 'step', 'operation', 'parameter', 'value']
    """
    rng = np.random.default_rng(config.random_seed)
    accumulated_rows: list[dict[str, Any]] = []
    wafer_counter = 0

    # Iterate over groups to generate shared process histories
    for group_number in range(1, config.n_groups + 1):
        operations_sequence = _sample_process_operations(config, rng)
        process_steps: list[dict[str, Any]] = []
        for operation_name in operations_sequence:
            if operation_name == "deposition":
                process_steps.append(_sample_deposition_step(config, rng))
            else:
                process_steps.append(_sample_treatment_step(config, rng))

        stress_base = _compute_base_stress(process_steps, config)
        group_template_rows = _build_group_template_rows(process_steps, config, rng)
        measurement_step = len(process_steps) + 1

        wafers_in_group = int(
            rng.integers(config.wafers_per_group_min, config.wafers_per_group_max, endpoint=True)
        )

        # Generate individual wafer records within group
        for _ in range(wafers_in_group):
            wafer_counter += 1
            wafer_id = _format_wafer_id(config.wafer_id_prefix, config.wafer_id_digits, wafer_counter)

            for step_num, op_name, param_name, param_val in group_template_rows:
                accumulated_rows.append(
                    {
                        "group": group_number,
                        "wafer_id": wafer_id,
                        "step": step_num,
                        "operation": op_name,
                        "parameter": param_name,
                        "value": param_val,
                    }
                )

            wafer_stress = _sample_wafer_stress(stress_base, config, rng)
            stress_string = f"{wafer_stress:.{config.stress_model.decimal_places}f}"
            accumulated_rows.append(
                {
                    "group": group_number,
                    "wafer_id": wafer_id,
                    "step": measurement_step,
                    "operation": "measurement",
                    "parameter": "stress",
                    "value": stress_string,
                }
            )

    return pd.DataFrame(accumulated_rows)


def save_pseudo_data(dataframe: pd.DataFrame, output_path: Path | str) -> Path:
    """Save generated pseudo-data DataFrame to CSV file.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to persist.
    output_path : Path | str
        Destination file path.

    Returns
    -------
    Path
        Resolved path of the saved CSV file.
    """
    resolved_path = Path(output_path).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(resolved_path, index=False, encoding="utf-8")
    return resolved_path
