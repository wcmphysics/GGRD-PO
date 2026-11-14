"""Configuration schema and validation for wafer pseudo-data generation.

This module defines Pydantic v2 schemas adhering to AGENTS.md §5:
- YAML is the single source of truth.
- Extra fields are forbidden via ConfigDict(extra="forbid").
- Zero default values in Python code (all values must come from config).
- Cross-field key alignment is strictly validated.
"""

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class DepositionConfig(BaseModel):
    """Configuration for deposition process step sampling."""

    model_config = ConfigDict(extra="forbid")

    materials: dict[str, list[str]]
    time_min: int
    time_max: int
    temperature_min: int
    temperature_max: int
    thickness_min: int
    thickness_max: int

    @model_validator(mode="after")
    def validate_deposition_ranges(self) -> "DepositionConfig":
        """Validate deposition ranges and recipe definitions."""
        if not self.materials:
            raise ValueError("Deposition materials map cannot be empty.")
        for material_name, recipe_list in self.materials.items():
            if not recipe_list:
                raise ValueError(
                    f"Deposition material '{material_name}' has empty recipes; expected non-empty list of recipes."
                )
        if self.time_min <= 0 or self.time_min > self.time_max:
            raise ValueError(
                f"Invalid deposition time bounds: time_min={self.time_min}, time_max={self.time_max}; "
                "expected 0 < time_min <= time_max."
            )
        if self.temperature_min <= 0 or self.temperature_min > self.temperature_max:
            raise ValueError(
                f"Invalid deposition temperature bounds: min={self.temperature_min}, max={self.temperature_max}; "
                "expected 0 < temperature_min <= temperature_max."
            )
        if self.thickness_min <= 0 or self.thickness_min > self.thickness_max:
            raise ValueError(
                f"Invalid deposition thickness bounds: min={self.thickness_min}, max={self.thickness_max}; "
                "expected 0 < thickness_min <= thickness_max."
            )
        return self


class TreatmentConfig(BaseModel):
    """Configuration for treatment process step sampling."""

    model_config = ConfigDict(extra="forbid")

    recipes: dict[str, list[str]]
    time_min: int
    time_max: int
    temperature_min: int
    temperature_max: int

    @model_validator(mode="after")
    def validate_treatment_ranges(self) -> "TreatmentConfig":
        """Validate treatment ranges and recipe definitions."""
        if not self.recipes:
            raise ValueError("Treatment recipes map cannot be empty.")
        for treatment_name, recipe_list in self.recipes.items():
            if not recipe_list:
                raise ValueError(
                    f"Treatment '{treatment_name}' has empty recipes; expected non-empty list of recipes."
                )
        if self.time_min <= 0 or self.time_min > self.time_max:
            raise ValueError(
                f"Invalid treatment time bounds: time_min={self.time_min}, time_max={self.time_max}; "
                "expected 0 < time_min <= time_max."
            )
        if self.temperature_min <= 0 or self.temperature_min > self.temperature_max:
            raise ValueError(
                f"Invalid treatment temperature bounds: min={self.temperature_min}, max={self.temperature_max}; "
                "expected 0 < temperature_min <= temperature_max."
            )
        return self


class TransitionConfig(BaseModel):
    """Configuration for transition steps and vacuum break settings."""

    model_config = ConfigDict(extra="forbid")

    vacuum_break_before_measure: int
    vacuum_break_values: list[int]

    @model_validator(mode="after")
    def validate_transition_values(self) -> "TransitionConfig":
        """Validate vacuum break configuration options."""
        if self.vacuum_break_before_measure not in (0, 1):
            raise ValueError(
                f"Invalid vacuum_break_before_measure={self.vacuum_break_before_measure}; expected 0 or 1."
            )
        if not self.vacuum_break_values:
            raise ValueError("vacuum_break_values list cannot be empty.")
        for value in self.vacuum_break_values:
            if value not in (0, 1):
                raise ValueError(
                    f"Invalid vacuum break value {value} in vacuum_break_values; expected elements in {{0, 1}}."
                )
        return self


class StressModelConfig(BaseModel):
    """Configuration for thermal-stress physical simulation and noise."""

    model_config = ConfigDict(extra="forbid")

    material_constants: dict[str, float]
    treatment_constants: dict[str, float]
    noise_std: float
    outlier_rate: float
    outlier_min_abs: float
    stress_min: float
    stress_max: float
    decimal_places: int

    @model_validator(mode="after")
    def validate_stress_model(self) -> "StressModelConfig":
        """Validate physical constants, noise parameters, and stress bounds."""
        if self.noise_std < 0.0:
            raise ValueError(f"Invalid noise_std={self.noise_std}; expected non-negative value.")
        if not (0.0 <= self.outlier_rate <= 1.0):
            raise ValueError(f"Invalid outlier_rate={self.outlier_rate}; expected value in [0.0, 1.0].")
        if self.stress_min >= self.stress_max:
            raise ValueError(
                f"Invalid stress clipping bounds: min={self.stress_min}, max={self.stress_max}; "
                "expected stress_min < stress_max."
            )
        if self.stress_min >= 0.0 or self.stress_max <= 0.0:
            raise ValueError(
                f"Invalid stress clipping bounds: min={self.stress_min}, max={self.stress_max}; "
                "expected stress_min < 0 < stress_max to support two-lobe outlier simulation."
            )
        max_allowed_outlier_min = min(-self.stress_min, self.stress_max)
        if self.outlier_min_abs < 0.0 or self.outlier_min_abs >= max_allowed_outlier_min:
            raise ValueError(
                f"Invalid outlier_min_abs={self.outlier_min_abs}; expected 0.0 <= outlier_min_abs < "
                f"{max_allowed_outlier_min} (min(-stress_min, stress_max))."
            )
        if self.decimal_places < 0:
            raise ValueError(f"Invalid decimal_places={self.decimal_places}; expected non-negative integer.")
        return self


class PseudoDataConfig(BaseModel):
    """Top-level configuration for wafer pseudo-data generation."""

    model_config = ConfigDict(extra="forbid")

    random_seed: int
    output_path: str
    n_groups: int
    wafers_per_group_min: int
    wafers_per_group_max: int
    wafer_id_prefix: str
    wafer_id_digits: int
    total_steps_min: int
    total_steps_max: int
    min_depositions: int
    max_treatments: int
    missing_parameter_rate: float
    deposition: DepositionConfig
    treatment: TreatmentConfig
    transition: TransitionConfig
    stress_model: StressModelConfig

    @model_validator(mode="after")
    def validate_simulation_parameters(self) -> "PseudoDataConfig":
        """Validate generator sequence limits and cross-field key alignments."""
        self._validate_scalar_bounds()
        self._validate_sequence_bounds()
        self._validate_cross_field_keys()
        return self

    def _validate_scalar_bounds(self) -> None:
        """Validate basic scalar parameter boundaries."""
        if self.random_seed < 0:
            raise ValueError(f"Invalid random_seed={self.random_seed}; expected random_seed >= 0.")
        if self.n_groups < 1:
            raise ValueError(f"Invalid n_groups={self.n_groups}; expected n_groups >= 1.")
        if self.wafers_per_group_min < 1 or self.wafers_per_group_min > self.wafers_per_group_max:
            raise ValueError(
                f"Invalid wafer bounds: min={self.wafers_per_group_min}, max={self.wafers_per_group_max}; "
                "expected 1 <= wafers_per_group_min <= wafers_per_group_max."
            )
        if self.wafer_id_digits < 1:
            raise ValueError(f"Invalid wafer_id_digits={self.wafer_id_digits}; expected digits >= 1.")
        if not (0.0 <= self.missing_parameter_rate <= 1.0):
            raise ValueError(
                f"Invalid missing_parameter_rate={self.missing_parameter_rate}; expected rate in [0.0, 1.0]."
            )

    def _validate_sequence_bounds(self) -> None:
        """Validate sequence length and operation count bounds."""
        if self.total_steps_min < 2 or self.total_steps_min > self.total_steps_max:
            raise ValueError(
                f"Invalid step bounds: min={self.total_steps_min}, max={self.total_steps_max}; "
                "expected 2 <= total_steps_min <= total_steps_max."
            )
        if self.min_depositions < 1 or self.min_depositions > (self.total_steps_min - 1):
            raise ValueError(
                f"Invalid min_depositions={self.min_depositions} for total_steps_min={self.total_steps_min}; "
                f"expected 1 <= min_depositions <= {self.total_steps_min - 1}."
            )
        if self.max_treatments < 0:
            raise ValueError(f"Invalid max_treatments={self.max_treatments}; expected max_treatments >= 0.")

    def _validate_cross_field_keys(self) -> None:
        """Validate key alignments between processes and physical constants."""
        deposition_material_keys = set(self.deposition.materials.keys())
        stress_material_keys = set(self.stress_model.material_constants.keys())
        if deposition_material_keys != stress_material_keys:
            mismatch_keys = deposition_material_keys.symmetric_difference(stress_material_keys)
            raise ValueError(
                "Key mismatch between deposition.materials and stress_model.material_constants: "
                f"{sorted(mismatch_keys)}"
            )

        treatment_recipe_keys = set(self.treatment.recipes.keys())
        stress_treatment_keys = set(self.stress_model.treatment_constants.keys())
        if treatment_recipe_keys != stress_treatment_keys:
            mismatch_keys = treatment_recipe_keys.symmetric_difference(stress_treatment_keys)
            raise ValueError(
                "Key mismatch between treatment.recipes and stress_model.treatment_constants: "
                f"{sorted(mismatch_keys)}"
            )


def load_pseudo_data_config(config_path: Path | str) -> PseudoDataConfig:
    """Load and validate pseudo-data generator configuration from YAML file.

    Parameters
    ----------
    config_path : Path | str
        Path to the YAML configuration file.

    Returns
    -------
    PseudoDataConfig
        Validated configuration instance.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the file content is not a valid YAML dictionary.
    """
    path_instance = Path(config_path).resolve()
    if not path_instance.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path_instance}; expected existing YAML configuration file."
        )

    with open(path_instance, mode="r", encoding="utf-8") as file_handle:
        loaded_data: Any = yaml.safe_load(file_handle)

    if not isinstance(loaded_data, dict):
        raise ValueError(
            f"Invalid configuration content in {path_instance}: got {type(loaded_data).__name__}, "
            "expected dictionary mapping."
        )

    return PseudoDataConfig(**loaded_data)
