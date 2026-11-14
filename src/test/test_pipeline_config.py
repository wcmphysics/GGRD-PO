"""Validate real pipeline settings and conflicting adjustments."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from src.config.pipeline_schema import PredictConfig, TrainConfig, load_config


def test_demo_configs() -> None:
    """The committed train and predict YAML files validate independently."""
    train = load_config(Path("src/config/config.yaml"), TrainConfig)
    predict = load_config(Path("src/config/predict_demo.yaml"), PredictConfig)
    assert train.output.directory / "model" == predict.bundle_directory
    assert predict.request_path.is_file()


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "grid", "unknown"])
def test_validation_errors(invalid: str) -> None:
    """A malformed model policy or incomplete config fails validation."""
    raw = load_config(Path("src/config/config.yaml"), TrainConfig).model_dump()
    if invalid == "missing":
        del raw["cv"]["outer_folds"]
    elif invalid == "duplicate":
        raw["models"][1]["name"] = raw["models"][0]["name"]
    elif invalid == "grid":
        raw["models"][0]["enable_search"] = True
    else:
        raw["models"][0]["parameters"]["unknown"] = 42
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(raw)
