"""Fit iterative estimators with optional, training-only convergence traces."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ElasticNet
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data

_TRACE_ROWS: ContextVar[int | None] = ContextVar("optimization_trace_rows", default=None)
type _Vector = NDArray[np.float64]


@contextmanager
def trace_optimization(training_rows: int) -> Iterator[None]:
    """Record only the fit using all permitted rows, skipping grouped search candidates."""
    token = _TRACE_ROWS.set(training_rows)
    try:
        yield
    finally:
        _TRACE_ROWS.reset(token)


@dataclass(frozen=True)
class _ElasticProblem:
    """Centered, sample-scaled coordinate-descent inputs."""

    features: NDArray[np.float64]
    targets: _Vector
    feature_means: _Vector
    target_mean: float
    squared_norms: _Vector
    lasso: float
    ridge: float
    positive: bool

    def gap(self, weights: _Vector, residual: _Vector) -> float:
        """Return the duality gap, or a first-order residual for unpenalized least squares."""
        correlations = self.features.T @ residual
        if not self.lasso:
            return self._smooth_gap(weights, correlations)
        dual = correlations - self.ridge * weights
        dual_norm = np.max(dual if self.positive else np.abs(dual))
        scale = self.lasso / dual_norm if dual_norm > self.lasso else 1.0
        quadratic = residual @ residual + self.ridge * (weights @ weights)
        primal = 0.5 * quadratic + self.lasso * np.abs(weights).sum()
        return float(primal + 0.5 * scale ** 2 * quadratic - scale * (residual @ self.targets))

    def _smooth_gap(self, weights: _Vector, correlations: _Vector) -> float:
        """Evaluate the smooth dual gap without cancellation at the optimum."""
        if not self.ridge:
            gradient = correlations
            if self.positive:
                gradient = np.where(weights > 0, correlations, np.maximum(correlations, 0))
            return float(gradient @ gradient)
        if self.positive:
            projected = np.maximum(correlations, 0)
            displacement = projected - self.ridge * weights
            return float(0.5 * (displacement @ displacement) / self.ridge + weights @ (projected - correlations))
        displacement = correlations - self.ridge * weights
        return float(0.5 * (displacement @ displacement) / self.ridge)


def _elastic_sweep(problem: _ElasticProblem, weights: _Vector, residual: _Vector,
                   coordinates: range | NDArray[np.int_]) -> tuple[float, float]:
    """Update one sweep of actual coordinates and return relative change components."""
    maximum_weight = maximum_change = 0.0
    for index in coordinates:
        norm = problem.squared_norms[index]
        if not norm:
            continue
        column = problem.features[:, index]
        old_weight = weights[index]
        update = column @ residual + old_weight * norm
        magnitude = max(abs(update) - problem.lasso, 0)
        new_weight = np.copysign(magnitude, update) / (norm + problem.ridge)
        if problem.positive:
            new_weight = max(new_weight, 0)
        change = new_weight - old_weight
        if change:
            weights[index] = new_weight
            residual -= change * column
        maximum_weight = max(maximum_weight, abs(new_weight))
        maximum_change = max(maximum_change, abs(change))
    return maximum_weight, maximum_change


def _elastic_record(problem: _ElasticProblem, weights: _Vector, residual: _Vector,
                    iteration: int) -> tuple[dict, tuple[_Vector, float]]:
    """Capture one true training objective and a linear predictor at the same step."""
    mse = float(residual @ residual / len(residual))
    penalty = (problem.lasso * np.abs(weights).sum() + 0.5 * problem.ridge * (weights @ weights)) / len(residual)
    record = {"iteration": iteration, "trajectory": 0, "objective": float(0.5 * mse + penalty),
              "objective_name": "elastic_net_objective", "train_mse": mse, "converged": None}
    intercept = float(problem.target_mean - problem.feature_means @ weights)
    return record, (weights.copy(), intercept)


def _coordinates(rng: np.random.RandomState, count: int, selection: str) -> range | NDArray[np.int_]:
    """Use one random draw with replacement for each coordinate update, or a cyclic sweep."""
    return rng.randint(count, size=count) if selection == "random" else range(count)


class TracedElasticNet(ElasticNet):
    """Dense, single-output Elastic Net with traceable coordinate descent sweeps.

    The objective and duality-gap threshold match sklearn ElasticNet. Cyclic
    uses ascending coordinates. Random draws with replacement from a fixed
    per-fit RNG. No active-set screening is applied.
    """

    def _problem(self, X: NDArray[np.float64], y: _Vector) -> _ElasticProblem:
        """Center dense input so the intercept can be set outside penalized updates."""
        feature_means = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        target_mean = float(y.mean()) if self.fit_intercept else 0.0
        features = np.asfortranarray(X - feature_means)
        targets = y - target_mean
        norms = np.einsum("ij,ij->j", features, features)
        return _ElasticProblem(features, targets, feature_means, target_mean, norms,
                               len(y) * self.alpha * self.l1_ratio, len(y) * self.alpha * (1 - self.l1_ratio),
                               self.positive)

    def _solve(self, problem: _ElasticProblem, weights: _Vector) -> tuple[int, float, float]:
        """Run real coordinate updates and capture each full training sweep when requested."""
        residual = problem.targets - problem.features @ weights
        threshold = self.tol * (problem.targets @ problem.targets)
        gap = problem.gap(weights, residual)
        recording = _TRACE_ROWS.get() == len(problem.targets)
        rng = check_random_state(self.random_state)
        records, iterates = [], []
        iteration = 0

        # Record the initialization and then only sweeps actually executed by this fit.
        if recording:
            metric, iterate = _elastic_record(problem, weights, residual, iteration)
            records.append(metric)
            iterates.append(iterate)
        if gap > threshold:
            for iteration in range(1, self.max_iter + 1):
                coordinates = _coordinates(rng, len(weights), self.selection)
                maximum_weight, maximum_change = _elastic_sweep(problem, weights, residual, coordinates)
                if recording:
                    metric, iterate = _elastic_record(problem, weights, residual, iteration)
                    records.append(metric)
                    iterates.append(iterate)
                if maximum_change <= self.tol * maximum_weight or iteration == self.max_iter:
                    gap = problem.gap(weights, residual)
                    if gap <= threshold:
                        break

        if records:
            records[-1]["converged"] = bool(gap <= threshold)
        self.convergence_, self.iterates_ = tuple(records), tuple(iterates)
        return iteration, gap, threshold

    def fit(self, X: NDArray[np.float64], y: _Vector, sample_weight: _Vector | None = None,
            check_input: bool = True) -> "TracedElasticNet":
        """Fit one dense, unweighted model and optionally retain its real iterates."""
        self.convergence_, self.iterates_ = (), ()
        self._validate_params()
        if sample_weight is not None or not check_input or self.precompute is not False:
            raise ValueError("Traced Elastic Net expects unweighted, validated rows and precompute=False.")
        features, targets = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
        problem = self._problem(features, np.asarray(targets, dtype=np.float64))
        weights = np.zeros(features.shape[1], dtype=np.float64)
        if self.warm_start and hasattr(self, "coef_"):
            if self.coef_.shape != weights.shape:
                raise ValueError(f"Warm-start Elastic Net coefficients {self.coef_.shape}; expected {weights.shape}.")
            weights[:] = self.coef_
        self.n_iter_, gap, threshold = self._solve(problem, weights)
        self.coef_ = weights
        self.dual_gap_ = gap / len(targets)
        self.intercept_ = float(problem.target_mean - problem.feature_means @ weights)
        if not np.isfinite(weights).all() or not np.isfinite([gap, threshold, self.intercept_]).all():
            raise ValueError("Elastic Net returned nonfinite coefficients, intercept or gap; expected finite output.")
        if gap > threshold:
            warnings.warn(f"Elastic Net exhausted {self.max_iter} iterations with gap={gap:.6g}; "
                          f"expected gap <= {threshold:.6g}.", ConvergenceWarning, stacklevel=2)
        return self


def _nll_record(iteration: int, trajectory: int, objective: float, converged: bool | None = None) -> dict:
    """Describe a real L-BFGS-B state with a portable scalar objective."""
    return {"iteration": iteration, "trajectory": trajectory, "objective": float(objective),
            "objective_name": "negative_log_marginal_likelihood", "converged": converged}


class _GaussianOptimizer:
    """Execute sklearn's L-BFGS-B settings and collect accepted iterates per restart."""

    def __init__(self) -> None:
        """Start an empty trace for one Gaussian Process fit."""
        self.traces: list[list[dict]] = []
        self.final_values: list[float] = []

    def __call__(self, objective: Callable, initial_theta: _Vector,
                 bounds: NDArray[np.float64]) -> tuple[_Vector, float]:
        """Run one native optimizer trajectory without recalculating its loss."""
        trajectory = len(self.traces)
        points: list[dict] = []
        initial_value = None

        def measured_objective(theta: _Vector) -> tuple[float, _Vector]:
            """Keep the real first evaluation if this trajectory accepts no iterates."""
            nonlocal initial_value
            value, gradient = objective(theta)
            if initial_value is None:
                initial_value = value
            return value, gradient

        def callback(intermediate_result: OptimizeResult) -> None:
            """Capture the cached loss at SciPy's accepted iterate."""
            points.append(_nll_record(len(points) + 1, trajectory, intermediate_result.fun))

        result = minimize(measured_objective, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                          callback=callback)
        if not points:
            if initial_value is None:
                raise ValueError(f"Gaussian Process restart={trajectory} returned no objective evaluations.")
            points.append(_nll_record(0, trajectory, initial_value))
        points[-1]["converged"] = bool(result.success)
        self.traces.append(points)
        self.final_values.append(result.fun)
        if not result.success:
            warnings.warn(f"Gaussian Process L-BFGS-B restart={trajectory} status={result.status}: {result.message}",
                          ConvergenceWarning, stacklevel=2)
        return result.x, result.fun

    def records(self) -> tuple[dict, ...]:
        """Flatten all real restart trajectories and flag the one selected by likelihood."""
        if not self.traces:
            return ()
        selected = int(np.argmin(self.final_values))
        return tuple(dict(record, selected=index == selected) for index, points in enumerate(self.traces)
                     for record in points)


class TracedGaussianProcess(GaussianProcessRegressor):
    """Use the native GP fit with optional accepted L-BFGS-B iteration records."""

    def fit(self, X: NDArray[np.float64], y: _Vector) -> "TracedGaussianProcess":
        """Trace optimizer restarts when requested and restore serializable parameters."""
        self.convergence_ = ()
        if self.optimizer != "fmin_l_bfgs_b" or _TRACE_ROWS.get() != len(X):
            super().fit(X, y)
            return self
        optimizer = _GaussianOptimizer()
        previous = self.optimizer
        try:
            self.optimizer = optimizer
            super().fit(X, y)
        finally:
            self.optimizer = previous
        self.convergence_ = optimizer.records()
        return self
