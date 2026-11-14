"""Exercise parameter validation and every supported CPU estimator adapter."""

import json
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter, ValidationError
import pytest
from sklearn.base import clone

from src.config.base_estimator import ModelConfig, required_training_rows
from src.config.pipeline_schema import TrainConfig, load_config
from src.model.base_estimator import effective_parameters, make_estimator, uses_numeric_scaling

_PROFILE = Path("src/config/train_all_models.yaml")


def _settings(estimator: str) -> dict:
    """Return editable settings for a model in the all-model example."""
    return next(base.model_dump() for base in load_config(_PROFILE, TrainConfig).models if base.estimator == estimator)


@pytest.mark.parametrize(("estimator", "scale"), [
    ("ridge", True), ("random_forest", False), ("elastic_net", True), ("gaussian_process", True),
    ("knn", True), ("svr", True), ("lightgbm", False), ("catboost", False), ("xgboost", False),
])
def test_estimator_factory(estimator: str, scale: bool) -> None:
    """Fresh seeded adapters are cloneable, deterministic, and have serializable parameters."""
    base = TypeAdapter(ModelConfig).validate_python(_settings(estimator))
    values = np.random.default_rng(12).normal(size=(32, 4))
    targets = np.sin(values[:, 0]) + values[:, 1] * 0.3
    native_model = make_estimator(base, 2**32 - 1)
    trained = native_model.fit(values, targets)
    duplicate = clone(trained).fit(values, targets)
    np.testing.assert_allclose(trained.predict(values), duplicate.predict(values), atol=1e-12, rtol=0)
    assert np.isfinite(trained.predict(values)).all()
    assert uses_numeric_scaling(base) is scale
    recorded = effective_parameters(base, trained)
    if estimator == "gaussian_process":
        assert recorded.pop("fitted_kernel")["length_scale"] == trained.kernel_.length_scale
    assert json.loads(json.dumps(recorded, allow_nan=False)) == base.parameters.model_dump(mode="json")


@pytest.mark.parametrize(("estimator", "field", "invalid"), [
    ("elastic_net", "l1_ratio", 1.25), ("gaussian_process", "normalize_y", "true"), ("knn", "n_neighbors", 0),
    ("svr", "C", -1), ("lightgbm", "num_leaves", 1), ("catboost", "depth", 17), ("xgboost", "subsample", 0),
])
def test_invalid_parameter(estimator: str, field: str, invalid: object) -> None:
    """Reject invalid knobs at configuration load, before any training fit."""
    raw = _settings(estimator)
    raw["parameters"][field] = invalid
    with pytest.raises(ValidationError, match=field):
        TypeAdapter(ModelConfig).validate_python(raw)


def test_declarative_gaussian_kernel() -> None:
    """Accept the supported Matérn variant and reject an inconsistent optimization request."""
    raw = _settings("gaussian_process")
    raw["parameters"]["kernel"] = {"kind": "matern", "nu": 1.5, "length_scale": 1.0, "length_scale_bounds": "fixed"}
    base = TypeAdapter(ModelConfig).validate_python(raw)
    estimator = make_estimator(base, 42).set_params(alpha=0.02)
    assert estimator.kernel.nu == 1.5
    recorded = effective_parameters(base, estimator)
    assert json.loads(json.dumps(recorded, allow_nan=False))["kernel"] == raw["parameters"]["kernel"]
    assert recorded["alpha"] == 0.02
    raw["parameters"]["n_restarts_optimizer"] = 1
    with pytest.raises(ValidationError, match="expected zero for fixed or unoptimized kernels"):
        TypeAdapter(ModelConfig).validate_python(raw)


def test_knn_search_training_size() -> None:
    """Search preflight can account for candidates larger than the fixed neighbor count."""
    raw = _settings("knn")
    assert required_training_rows(TypeAdapter(ModelConfig).validate_python(raw)) == raw["parameters"]["n_neighbors"]
    raw.update(enable_search=True, search_grid={"n_neighbors": [3, 9], "weights": ["uniform"]})
    assert required_training_rows(TypeAdapter(ModelConfig).validate_python(raw)) == 9
    raw["search_grid"] = {}
    with pytest.raises(ValidationError, match="empty search grid"):
        TypeAdapter(ModelConfig).validate_python(raw)
