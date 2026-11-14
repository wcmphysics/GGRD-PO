"""YAML contracts for training, prediction, and process search."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StrictBool, ValidationError, model_validator
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from src.config.pipeline_common import (
    NonemptyText,
    NonnegativeInt,
    NonnegativeNumber,
    ParameterUnits,
    PositiveInt,
    PositiveNumber,
    StrictConfig,
)
from src.optimize.settings import ObjectiveConfig, SearchConfig, SearchDomain, SupportConfig

FoldCount = Annotated[int, Field(strict=True, ge=2)]
AlphaGrid = Annotated[list[NonnegativeNumber], Field(min_length=1)]
IntegerGrid = Annotated[list[PositiveInt], Field(min_length=1)]


class DataConfig(StrictConfig):
    """Source history and units for a training run."""

    csv_path: Path
    parameter_units: ParameterUnits
    target_unit: NonemptyText


class FeaturesConfig(StrictConfig):
    """Number of process slots retained in a wide feature row."""

    N_step: PositiveInt


class CrossValidationConfig(StrictConfig):
    """Group partition counts, root seed, and MAPE eligibility."""

    outer_folds: FoldCount
    outer_repeats: PositiveInt
    inner_folds: FoldCount
    search_folds: FoldCount
    root_seed: NonnegativeInt
    mape_min_abs_y: NonnegativeNumber


class RidgeParameters(StrictConfig):
    """Supported fixed parameters for Ridge."""

    alpha: NonnegativeNumber


class ForestParameters(StrictConfig):
    """Supported fixed parameters for the random forest."""

    n_estimators: PositiveInt
    min_samples_leaf: PositiveInt
    n_jobs: Annotated[int, Field(strict=True, ge=1, le=1)]


class RidgeGrid(StrictConfig):
    """Allowed Ridge search candidates."""

    alpha: AlphaGrid


class ForestGrid(StrictConfig):
    """Allowed forest search candidates, with null meaning fixed."""

    n_estimators: IntegerGrid | None
    min_samples_leaf: IntegerGrid | None

    @model_validator(mode="after")
    def require_candidates(self) -> "ForestGrid":
        """Require at least one searched parameter."""
        if self.n_estimators is None and self.min_samples_leaf is None:
            raise ValueError("forest grid found two null entries, expected at least one nonempty grid")
        return self


class BaseConfig(StrictConfig):
    """Common name and search policy for an ordered base model."""

    name: NonemptyText
    enable_search: StrictBool

    @model_validator(mode="after")
    def validate_search(self) -> "BaseConfig":
        """Ensure the grid and enable flag describe one clear policy."""
        if self.enable_search != (self.search_grid is not None):
            raise ValueError(f"enable_search={self.enable_search}, expected a grid exactly when search is enabled")
        return self


class RidgeConfig(BaseConfig):
    """Ridge base model settings."""

    estimator: Literal["ridge"]
    parameters: RidgeParameters
    search_grid: RidgeGrid | None


class ForestConfig(BaseConfig):
    """Random forest base model settings."""

    estimator: Literal["random_forest"]
    parameters: ForestParameters
    search_grid: ForestGrid | None


ModelConfig = Annotated[RidgeConfig | ForestConfig, Field(discriminator="estimator")]


class MetaConfig(StrictConfig):
    """Shrinkage and numerical precision for simplex fitting."""

    alpha: NonnegativeNumber
    max_iterations: PositiveInt
    solver_tolerance: PositiveNumber
    constraint_tolerance: PositiveNumber


class AnalysisConfig(StrictConfig):
    """Controls the paired transition feature ablation."""

    transition_ablation: StrictBool


class OutputConfig(StrictConfig):
    """Fresh output directory for one invocation."""

    directory: Path


class TrainConfig(StrictConfig):
    """Configuration of one complete evaluation and deployment run."""

    data: DataConfig
    features: FeaturesConfig
    cv: CrossValidationConfig
    models: Annotated[list[ModelConfig], Field(min_length=2)]
    meta: MetaConfig
    analysis: AnalysisConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_names(self) -> "TrainConfig":
        """Require an unambiguous name for every base matrix column."""
        names = [base.name for base in self.models]
        if len(set(names)) != len(names):
            raise ValueError(f"base model names={names!r}, expected unique names")
        return self


class PredictConfig(StrictConfig):
    """Bundle, request collection, and output for inference."""

    bundle_directory: Path
    request_path: Path
    output: OutputConfig


class SyntheticDomainConfig(StrictConfig):
    """Project an optimization domain from the generator's process catalog."""

    source: Literal["synthetic"]
    generator_config_path: Path
    parameter_units: ParameterUnits


class OptimizeConfig(StrictConfig):
    """Bundle and validated process domain for a fresh search run."""

    bundle_directory: Path
    domain: Annotated[SyntheticDomainConfig | SearchDomain, Field(discriminator="source")]
    search: SearchConfig
    objective: ObjectiveConfig
    support: SupportConfig
    output: OutputConfig


class _UniqueYamlLoader(yaml.SafeLoader):
    """Reject duplicate keys before Pydantic validates the decoded data."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict:
        """Build a mapping without silently replacing the first value."""
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise ConstructorError(None, None, f"Expected a string or integer mapping key, found {key!r}",
                                       key_node.start_mark)
            if key in result:
                raise ConstructorError(None, None, f"Duplicate YAML key {key!r}", key_node.start_mark)
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_config[ConfigType: StrictConfig](path: Path, schema: type[ConfigType]) -> ConfigType:
    """Read an explicit YAML file using the provided command schema."""
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueYamlLoader)
        return schema.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"Configuration {path}: expected valid {schema.__name__} YAML: {error}") from error
