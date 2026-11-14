"""Verify genuine iterative traces and parity with the corresponding objectives."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import ElasticNet

from src.model.iterative import TracedElasticNet, TracedGaussianProcess, trace_optimization


@pytest.fixture
def regression_rows() -> tuple[np.ndarray, np.ndarray]:
    """Return correlated features and a nonzero target intercept."""
    rng = np.random.default_rng(55)
    features = rng.normal(size=(100, 9))
    features[:, 1] = 0.8 * features[:, 0] + 0.2 * features[:, 1]
    targets = 5 + features @ rng.normal(size=9) + 0.1 * rng.normal(size=100)
    return features, targets


@pytest.mark.parametrize("selection", ["cyclic", "random"])
@pytest.mark.parametrize("ratio", [0.0, 0.25, 1.0])
def test_elastic_trace(regression_rows: tuple[np.ndarray, np.ndarray], selection: str, ratio: float) -> None:
    """Recompute every real training objective and compare the fitted solution with sklearn."""
    features, targets = regression_rows
    parameters = dict(alpha=0.025, l1_ratio=ratio, max_iter=1500, tol=1e-10, selection=selection, random_state=42)
    actual = clone(TracedElasticNet(**parameters))
    with trace_optimization(len(features)):
        actual.fit(features, targets)
    expected = ElasticNet(**parameters).fit(features, targets)
    np.testing.assert_allclose(actual.predict(features), expected.predict(features), rtol=0, atol=2e-6)
    assert actual.n_iter_ > 1
    assert len(actual.iterates_) == actual.n_iter_ + 1
    assert [point["iteration"] for point in actual.convergence_] == list(range(actual.n_iter_ + 1))
    assert actual.convergence_[-1]["converged"] is True
    assert [point["converged"] for point in actual.convergence_[:-1]] == [None] * actual.n_iter_
    for point, (coefficients, intercept) in zip(actual.convergence_, actual.iterates_, strict=True):
        mse = np.mean((features @ coefficients + intercept - targets) ** 2)
        penalty = ratio * np.abs(coefficients).sum() + 0.5 * (1 - ratio) * (coefficients @ coefficients)
        assert point["train_mse"] == pytest.approx(mse, abs=1e-10)
        assert point["objective"] == pytest.approx(0.5 * mse + actual.alpha * penalty, abs=1e-10)
    objectives = [point["objective"] for point in actual.convergence_]
    assert objectives[-1] < objectives[0]
    assert np.all(np.diff(objectives) <= 1e-10)
    recorded_prediction = actual.predict(features)
    actual.fit(features, targets)
    assert actual.convergence_ == actual.iterates_ == ()
    np.testing.assert_array_equal(actual.predict(features), recorded_prediction)


def test_elastic_stopping(regression_rows: tuple[np.ndarray, np.ndarray]) -> None:
    """Keep initial convergence and exhausted iteration budget distinct."""
    features, targets = regression_rows
    initial = TracedElasticNet(alpha=100, max_iter=1)
    limited = TracedElasticNet(alpha=0.01, max_iter=1, tol=1e-12)
    with trace_optimization(len(features)):
        initial.fit(features, targets)
        with pytest.warns(ConvergenceWarning, match="exhausted 1 iterations"):
            limited.fit(features, targets)
    assert initial.n_iter_ == 0
    assert [point["converged"] for point in initial.convergence_] == [True]
    assert limited.n_iter_ == 1
    assert [point["converged"] for point in limited.convergence_] == [None, False]


def test_gaussian_trace(regression_rows: tuple[np.ndarray, np.ndarray]) -> None:
    """Compare native predictions, restart selection, and reported training log likelihood."""
    features, targets = regression_rows
    features, targets = features[:19, :3], targets[:19]
    parameters = dict(kernel=RBF(1.0, (0.1, 10)) + WhiteKernel(0.2, (0.01, 2)), alpha=0.1,
                      normalize_y=True, n_restarts_optimizer=2, random_state=33)
    model = clone(TracedGaussianProcess(**parameters))
    with trace_optimization(len(features)):
        model.fit(features, targets)
    expected = GaussianProcessRegressor(**parameters).fit(features, targets)
    np.testing.assert_array_equal(model.predict(features), expected.predict(features))
    assert model.optimizer == "fmin_l_bfgs_b"
    trajectories = {record["trajectory"] for record in model.convergence_}
    assert trajectories == {0, 1, 2}
    winners = [record for record in model.convergence_ if record["selected"]]
    assert len({record["trajectory"] for record in winners}) == 1
    assert winners[-1]["objective"] == pytest.approx(-model.log_marginal_likelihood_value_)
    for trajectory in trajectories:
        points = [record for record in model.convergence_ if record["trajectory"] == trajectory]
        assert [record["iteration"] for record in points] == list(range(1, len(points) + 1))
        assert points[-1]["converged"] is True
        assert np.all(np.diff([point["objective"] for point in points]) <= 1e-10)


def test_trace_context_skips_smaller_and_fixed_fits(regression_rows: tuple[np.ndarray, np.ndarray]) -> None:
    """Leave only a whole training refit trace and no trace for a fixed GP kernel."""
    features, targets = regression_rows
    elastic = TracedElasticNet(alpha=0.05, random_state=55)
    fixed = TracedGaussianProcess(kernel=RBF(1.0), optimizer=None, alpha=0.1)
    with trace_optimization(len(features)):
        elastic.fit(features[:-1], targets[:-1])
        fixed.fit(features, targets)
    assert elastic.convergence_ == elastic.iterates_ == fixed.convergence_ == ()
