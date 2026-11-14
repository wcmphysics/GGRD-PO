"""YAML contracts for training, prediction, and process search."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StrictBool, ValidationError, model_validator
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from src.config.base_estimator import ModelConfig
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
from src.config.pseudo_data_schema import PseudoDataConfig

FoldCount = Annotated[int, Field(strict=True, ge=2)]


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


class MetaConfig(StrictConfig):
    """Shrinkage and numerical precision for simplex fitting."""

    alpha: NonnegativeNumber
    max_iterations: PositiveInt
    solver_tolerance: PositiveNumber
    constraint_tolerance: PositiveNumber


class AnalysisConfig(StrictConfig):
    """Control paired feature ablation and fit diagnostics."""

    transition_ablation: StrictBool
    convergence: StrictBool
    convergence_every: PositiveInt


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


class StageConfig(StrictConfig):
    """Stage toggle and path to its typed settings."""

    enabled: StrictBool
    config: Path | None

    @model_validator(mode="after")
    def require_path(self) -> "StageConfig":
        """Require a settings file for an enabled stage."""
        if self.enabled and self.config is None:
            raise ValueError("enabled=true with config=null; expected a settings file path for the enabled stage")
        return self


class PipelineConfig(StrictConfig):
    """Control which stages main runs in workflow order."""

    generate: StageConfig
    train: StageConfig
    predict: StageConfig
    optimize: StageConfig

    @model_validator(mode="after")
    def require_stage(self) -> "PipelineConfig":
        """Reject a configuration that would produce no output."""
        if not any(stage.enabled for stage in (self.generate, self.train, self.predict, self.optimize)):
            raise ValueError("all stages disabled; expected at least one enabled stage")
        return self


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


def _read_yaml(path: Path) -> object:
    """Read YAML with source context and no duplicate keys."""
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueYamlLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Configuration {path}: expected readable YAML: {error}") from error


def load_config[ConfigType: StrictConfig](path: Path, schema: type[ConfigType]) -> ConfigType:
    """Read an explicit YAML file using the provided command schema."""
    try:
        return schema.model_validate(_read_yaml(path))
    except ValidationError as error:
        raise ValueError(f"Configuration {path}: expected valid {schema.__name__} YAML: {error}") from error


def load_entry_config(path: Path) -> PipelineConfig | PseudoDataConfig:
    """Validate a workflow file or an existing generator-only YAML file."""
    raw = _read_yaml(path)
    legacy_keys = {"n_groups", "wafer_id_prefix", "stress_model"}
    workflow_keys = {"generate", "train", "predict", "optimize"}
    legacy = isinstance(raw, dict) and bool(legacy_keys.intersection(raw)) and not workflow_keys.intersection(raw)
    schema = PseudoDataConfig if legacy else PipelineConfig
    try:
        return schema.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"Configuration {path}: expected valid {schema.__name__} YAML: {error}") from error
