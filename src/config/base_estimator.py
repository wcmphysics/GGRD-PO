"""Typed settings and train-scope requirements for base regressors."""

from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

from src.config.pipeline_common import (
    FiniteNumber,
    NonemptyText,
    NonnegativeInt,
    NonnegativeNumber,
    PositiveInt,
    PositiveNumber,
    StrictConfig,
)

type Choices[Item] = Annotated[list[Item], Field(min_length=1)]
type UnitFraction = Annotated[float, Field(strict=True, gt=0, le=1)]
type SingleThread = Annotated[int, Field(strict=True, ge=1, le=1)]
type ElasticSelection = Literal["cyclic", "random"]
type NeighborWeights = Literal["uniform", "distance"]
type NeighborDistance = Annotated[int, Field(strict=True, ge=1, le=2)]
type SvrKernel = Literal["rbf", "linear", "poly", "sigmoid"]
type SvrGamma = Literal["scale", "auto"] | PositiveNumber
type CatDepth = Annotated[int, Field(strict=True, ge=1, le=16)]
type LightDepth = Annotated[int, Field(strict=True, ge=-1)]
type LightLeaves = Annotated[int, Field(strict=True, ge=2)]


class _SearchGrid(StrictConfig):
    """Permit searching selected parameters while keeping the remaining values fixed."""

    # Optional members mark omitted search dimensions. Fixed run values still come from YAML.
    @model_validator(mode="after")
    def require_candidates(self) -> "_SearchGrid":
        """Require at least one nonempty candidate list."""
        if not self.model_dump(exclude_none=True):
            raise ValueError("empty search grid; expected at least one nonempty parameter list")
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


class RidgeParameters(StrictConfig):
    """Supported fixed parameters for Ridge."""

    alpha: NonnegativeNumber


class RidgeGrid(_SearchGrid):
    """Allowed Ridge search candidates."""

    alpha: Choices[NonnegativeNumber]


class RidgeConfig(BaseConfig):
    """Ridge base model settings."""

    estimator: Literal["ridge"]
    parameters: RidgeParameters
    search_grid: RidgeGrid | None


class ForestParameters(StrictConfig):
    """Supported fixed parameters for the random forest."""

    n_estimators: PositiveInt
    min_samples_leaf: PositiveInt
    n_jobs: SingleThread


class ForestGrid(_SearchGrid):
    """Allowed forest search candidates, with null meaning fixed."""

    n_estimators: Choices[PositiveInt] | None = None
    min_samples_leaf: Choices[PositiveInt] | None = None


class ForestConfig(BaseConfig):
    """Random forest base model settings."""

    estimator: Literal["random_forest"]
    parameters: ForestParameters
    search_grid: ForestGrid | None


class ElasticParameters(StrictConfig):
    """Positive Elastic Net penalty and native coordinate descent options."""

    alpha: PositiveNumber
    l1_ratio: UnitFraction
    max_iter: PositiveInt
    tol: PositiveNumber
    selection: ElasticSelection


class ElasticGrid(_SearchGrid):
    """Allowed Elastic Net search candidates."""

    alpha: Choices[PositiveNumber] | None = None
    l1_ratio: Choices[UnitFraction] | None = None
    max_iter: Choices[PositiveInt] | None = None
    tol: Choices[PositiveNumber] | None = None
    selection: Choices[ElasticSelection] | None = None


class ElasticNetConfig(BaseConfig):
    """Elastic Net base model settings."""

    estimator: Literal["elastic_net"]
    parameters: ElasticParameters
    search_grid: ElasticGrid | None


class _KernelSettings(StrictConfig):
    """Declarative isotropic kernel scale and optional optimization bounds."""

    length_scale: PositiveNumber
    length_scale_bounds: tuple[PositiveNumber, PositiveNumber] | Literal["fixed"]

    @model_validator(mode="after")
    def check_bounds(self) -> "_KernelSettings":
        """Keep an initial scale inside its finite optimization domain."""
        if self.length_scale_bounds != "fixed":
            lower, upper = self.length_scale_bounds
            if lower >= upper or not lower <= self.length_scale <= upper:
                raise ValueError(f"length_scale={self.length_scale}, bounds={self.length_scale_bounds}; "
                                 "expected increasing bounds containing length_scale, or 'fixed'")
        return self


class RbfSettings(_KernelSettings):
    """Radial basis function kernel settings."""

    kind: Literal["rbf"]


class MaternSettings(_KernelSettings):
    """Matérn kernel settings restricted to efficiently evaluated smoothness values."""

    kind: Literal["matern"]
    nu: Literal[0.5, 1.5, 2.5]


class GaussianParameters(StrictConfig):
    """Supported native Gaussian process fit settings."""

    kernel: Annotated[RbfSettings | MaternSettings, Field(discriminator="kind")]
    alpha: PositiveNumber
    optimizer: Literal["fmin_l_bfgs_b"] | None
    n_restarts_optimizer: NonnegativeInt
    normalize_y: StrictBool


class GaussianGrid(_SearchGrid):
    """Search scalar settings without placing kernel objects into trial records."""

    alpha: Choices[PositiveNumber] | None = None
    n_restarts_optimizer: Choices[NonnegativeInt] | None = None
    normalize_y: Choices[StrictBool] | None = None


class GaussianConfig(BaseConfig):
    """Gaussian process base model settings."""

    estimator: Literal["gaussian_process"]
    parameters: GaussianParameters
    search_grid: GaussianGrid | None

    @model_validator(mode="after")
    def check_restarts(self) -> "GaussianConfig":
        """Forbid nonzero restarts when no kernel optimization can run."""
        restarts = [self.parameters.n_restarts_optimizer]
        if self.search_grid is not None and self.search_grid.n_restarts_optimizer is not None:
            restarts.extend(self.search_grid.n_restarts_optimizer)
        kernel_is_fixed = self.parameters.kernel.length_scale_bounds == "fixed"
        if any(restarts) and (kernel_is_fixed or self.parameters.optimizer is None):
            raise ValueError(f"n_restarts_optimizer={restarts}; expected zero for fixed or unoptimized kernels")
        return self


class KnnParameters(StrictConfig):
    """Number and weighting of neighbors and Minkowski distance power."""

    n_neighbors: PositiveInt
    weights: NeighborWeights
    p: NeighborDistance


class KnnGrid(_SearchGrid):
    """Allowed KNN search candidates."""

    n_neighbors: Choices[PositiveInt] | None = None
    weights: Choices[NeighborWeights] | None = None
    p: Choices[NeighborDistance] | None = None


class KnnConfig(BaseConfig):
    """K-nearest neighbors base model settings."""

    estimator: Literal["knn"]
    parameters: KnnParameters
    search_grid: KnnGrid | None


class SvrParameters(StrictConfig):
    """Supported SVR regularization and native named kernel settings."""

    C: PositiveNumber
    epsilon: NonnegativeNumber
    kernel: SvrKernel
    gamma: SvrGamma
    degree: PositiveInt
    coef0: FiniteNumber


class SvrGrid(_SearchGrid):
    """Allowed SVR search candidates."""

    C: Choices[PositiveNumber] | None = None
    epsilon: Choices[NonnegativeNumber] | None = None
    kernel: Choices[SvrKernel] | None = None
    gamma: Choices[SvrGamma] | None = None
    degree: Choices[PositiveInt] | None = None
    coef0: Choices[FiniteNumber] | None = None


class SvrConfig(BaseConfig):
    """Support vector regression base model settings."""

    estimator: Literal["svr"]
    parameters: SvrParameters
    search_grid: SvrGrid | None


class _TreeBoostParameters(StrictConfig):
    """Shared fixed settings for count-based boosted tree libraries."""

    n_estimators: PositiveInt
    learning_rate: PositiveNumber
    reg_alpha: NonnegativeNumber
    reg_lambda: NonnegativeNumber


class _TreeBoostGrid(_SearchGrid):
    """Shared search candidates for count-based boosted tree libraries."""

    n_estimators: Choices[PositiveInt] | None = None
    learning_rate: Choices[PositiveNumber] | None = None
    reg_alpha: Choices[NonnegativeNumber] | None = None
    reg_lambda: Choices[NonnegativeNumber] | None = None


class LightParameters(_TreeBoostParameters):
    """CPU LightGBM tree growth settings."""

    num_leaves: LightLeaves
    min_child_samples: PositiveInt
    max_depth: LightDepth


class LightGrid(_TreeBoostGrid):
    """Allowed LightGBM search candidates."""

    num_leaves: Choices[LightLeaves] | None = None
    min_child_samples: Choices[PositiveInt] | None = None
    max_depth: Choices[LightDepth] | None = None


class LightGbmConfig(BaseConfig):
    """LightGBM base model settings."""

    estimator: Literal["lightgbm"]
    parameters: LightParameters
    search_grid: LightGrid | None


class CatParameters(StrictConfig):
    """CPU CatBoost tree growth settings."""

    iterations: PositiveInt
    learning_rate: PositiveNumber
    depth: CatDepth
    l2_leaf_reg: NonnegativeNumber


class CatGrid(_SearchGrid):
    """Allowed CatBoost search candidates."""

    iterations: Choices[PositiveInt] | None = None
    learning_rate: Choices[PositiveNumber] | None = None
    depth: Choices[CatDepth] | None = None
    l2_leaf_reg: Choices[NonnegativeNumber] | None = None


class CatBoostConfig(BaseConfig):
    """CatBoost base model settings."""

    estimator: Literal["catboost"]
    parameters: CatParameters
    search_grid: CatGrid | None


class XgbParameters(_TreeBoostParameters):
    """CPU XGBoost tree growth settings."""

    max_depth: PositiveInt
    min_child_weight: NonnegativeNumber
    subsample: UnitFraction
    colsample_bytree: UnitFraction


class XgbGrid(_TreeBoostGrid):
    """Allowed XGBoost search candidates."""

    max_depth: Choices[PositiveInt] | None = None
    min_child_weight: Choices[NonnegativeNumber] | None = None
    subsample: Choices[UnitFraction] | None = None
    colsample_bytree: Choices[UnitFraction] | None = None


class XgBoostConfig(BaseConfig):
    """XGBoost base model settings."""

    estimator: Literal["xgboost"]
    parameters: XgbParameters
    search_grid: XgbGrid | None


ModelConfig = Annotated[
    RidgeConfig | ForestConfig | ElasticNetConfig | GaussianConfig | KnnConfig | SvrConfig | LightGbmConfig
    | CatBoostConfig | XgBoostConfig,
    Field(discriminator="estimator"),
]


def required_training_rows(base: ModelConfig) -> int | None:
    """Return the minimum allowed KNN training size across the entire search grid."""
    if not isinstance(base, KnnConfig):
        return None
    if base.search_grid is not None and base.search_grid.n_neighbors is not None:
        return max(base.search_grid.n_neighbors)
    return base.parameters.n_neighbors
