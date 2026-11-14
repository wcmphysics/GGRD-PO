"""Base pipelines, constrained stacking weights, and shared prediction checks."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline

from src.config.pipeline_schema import MetaConfig, ModelConfig
from src.feature.preprocess import make_preprocessor
from src.feature.wide import FeatureSchema
from src.model.base_estimator import make_estimator, uses_numeric_scaling


@dataclass(frozen=True)
class FittedStack:
    """Immutable ordering and payload for a trained stack predictor."""

    schema: FeatureSchema
    base_names: tuple[str, ...]
    pipelines: tuple[Pipeline, ...]
    parameters: tuple[dict, ...]
    weights: NDArray[np.float64]
    constraint_tolerance: float


def make_pipeline(base: ModelConfig, schema: FeatureSchema, seed: int) -> Pipeline:
    """Construct one new preprocessing and base estimator pipeline."""
    return Pipeline([("features", make_preprocessor(schema, scale_numeric=uses_numeric_scaling(base))),
                     ("estimator", make_estimator(base, seed))])


def _feasible(weights: NDArray[np.float64], tolerance: float) -> bool:
    """Check numerical feasibility of a candidate simplex vector."""
    return bool(weights.ndim == 1 and weights.size and np.isfinite(weights).all()
                and (weights >= -tolerance).all() and (weights <= 1 + tolerance).all()
                and abs(weights.sum() - 1.0) <= tolerance)


def fit_weights(oof: NDArray[np.float64], targets: NDArray[np.float64], config: MetaConfig,
                context: str, trace: list[float] | None = None) -> NDArray[np.float64]:
    """Minimize original-unit squared error with regularized simplex weights."""
    if oof.ndim != 2 or targets.shape != (len(oof),) or not oof.shape[1] or not len(oof):
        raise ValueError(f"{context}: invalid meta dimensions {oof.shape}, {targets.shape}; expected aligned OOF rows.")
    if not np.isfinite(oof).all() or not np.isfinite(targets).all():
        raise ValueError(f"{context}: nonfinite meta input; expected finite predictions and labels.")

    def objective(weights: NDArray[np.float64]) -> float:
        """Evaluate mean squared prediction error and the shrinkage penalty."""
        return float(np.mean((oof @ weights - targets) ** 2) + config.alpha * np.dot(weights, weights))

    def gradient(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate the analytic objective gradient."""
        return 2 * (oof.T @ (oof @ weights - targets)) / len(oof) + 2 * config.alpha * weights

    def sum_constraint(weights: NDArray[np.float64]) -> float:
        """Constrain the weights to sum to one."""
        return float(weights.sum() - 1.0)

    def sum_gradient(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate the analytic equality constraint gradient."""
        return np.ones_like(weights)

    initial = np.full(oof.shape[1], 1 / oof.shape[1])
    callback = None
    if trace is not None:
        trace.append(objective(initial))

        def record_iterate(weights: NDArray[np.float64]) -> None:
            """Record the true objective at each major SLSQP iteration."""
            trace.append(objective(weights))

        callback = record_iterate
    # Solve against OOF predictions only. Outer holdout data never enters this call.
    result = minimize(objective, initial, method="SLSQP", jac=gradient, bounds=[(0, 1)] * len(initial),
                      constraints={"type": "eq", "fun": sum_constraint, "jac": sum_gradient},
                      options={"ftol": config.solver_tolerance, "maxiter": config.max_iterations}, callback=callback)
    tolerance = config.constraint_tolerance
    if not result.success or not _feasible(result.x, tolerance) or not np.isfinite(result.fun):
        raise ValueError(f"{context}: meta solver returned {result.message}, {result.x}; expected a feasible solution.")
    repaired = np.clip(result.x, 0, 1)
    repaired /= repaired.sum()
    score = objective(repaired)
    if not _feasible(repaired, tolerance) or not np.isfinite(score) or score > objective(initial) + tolerance:
        raise ValueError(f"{context}: meta objective {score}; expected a feasible score no worse than equal weights.")
    return repaired


def predict_base(pipeline: Pipeline, features: pd.DataFrame, context: str) -> NDArray[np.float64]:
    """Predict one base with explicit shape, finiteness and error context."""
    try:
        values = np.asarray(pipeline.predict(features), dtype=np.float64)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{context}: base prediction failed for {len(features)} rows: {error}") from error
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError(f"{context}: prediction shape {values.shape}; expected one finite prediction per wafer.")
    return values


def predict_base_matrix(fitted_stack: FittedStack, X: pd.DataFrame) -> pd.DataFrame:
    """Return base predictions with declared names, preserving source row order."""
    schema = fitted_stack.schema
    if tuple(X.columns) != schema.columns or tuple(X.dtypes.astype(str)) != schema.dtypes:
        raise ValueError(f"Prediction feature schema {tuple(X.columns)}; expected {schema.columns} and declared types.")
    if X.empty:
        return pd.DataFrame(index=X.index, columns=fitted_stack.base_names, dtype=np.float64)
    return pd.DataFrame({name: predict_base(pipeline, X, f"base={name}")
                         for name, pipeline in zip(fitted_stack.base_names, fitted_stack.pipelines, strict=True)},
                        index=X.index)


def predict_meta(fitted_stack: FittedStack, base_matrix: pd.DataFrame) -> NDArray[np.float64]:
    """Predict with the meta weights only for a valid named base matrix."""
    if tuple(base_matrix.columns) != fitted_stack.base_names:
        raise ValueError(f"Meta base names {tuple(base_matrix.columns)}; expected {fitted_stack.base_names}.")
    matrix = base_matrix.to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite meta input; expected a finite prediction from every base model.")
    if not len(matrix):
        return np.empty(0, dtype=np.float64)
    values = matrix @ fitted_stack.weights
    tolerance = fitted_stack.constraint_tolerance
    if (not np.isfinite(values).all() or np.any(values < matrix.min(axis=1) - tolerance)
            or np.any(values > matrix.max(axis=1) + tolerance)):
        raise ValueError("Invalid meta prediction; expected finite values within each base prediction range.")
    return values


def predict_stack(fitted_stack: FittedStack, X: pd.DataFrame) -> NDArray[np.float64]:
    """Apply the base pipelines and constrained meta-model to features."""
    return predict_meta(fitted_stack, predict_base_matrix(fitted_stack, X))
