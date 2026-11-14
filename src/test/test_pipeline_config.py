"""Validate real pipeline settings and conflicting adjustments."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from src.config.pipeline_schema import PipelineConfig, PredictConfig, TrainConfig, load_config


def test_demo_configs() -> None:
    """The committed train and predict YAML files validate independently."""
    workflow = load_config(Path("src/config/config.yaml"), PipelineConfig)
    train = load_config(workflow.train.config, TrainConfig)
    predict = load_config(workflow.predict.config, PredictConfig)
    assert workflow.train.enabled
    assert train.output.directory / "model" == predict.bundle_directory
    assert predict.request_path.is_file()


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "grid", "unknown", "coercion"])
def test_validation_errors(invalid: str) -> None:
    """A malformed model policy or incomplete config fails validation."""
    raw = load_config(Path("src/config/train_demo.yaml"), TrainConfig).model_dump()
    if invalid == "missing":
        del raw["cv"]["outer_folds"]
    elif invalid == "duplicate":
        raw["models"][1]["name"] = raw["models"][0]["name"]
    elif invalid == "grid":
        raw["models"][0]["enable_search"] = True
    elif invalid == "unknown":
        raw["models"][0]["parameters"]["unknown"] = 42
    else:
        raw["analysis"]["transition_ablation"] = "yes"
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(raw)


def test_duplicate_yaml_key(tmp_path: Path) -> None:
    """A repeated YAML key fails before either value is used."""
    path = tmp_path / "repeated.yaml"
    path.write_text("features:\n  N_step: 4\n  N_step: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate YAML key 'N_step'"):
        load_config(path, TrainConfig)


def test_workflow_requires_enabled_stage_and_settings() -> None:
    """Reject a no-op workflow and an enabled stage with no settings file."""
    raw = {name: {"enabled": False, "config": None} for name in ("generate", "train", "predict", "optimize")}
    with pytest.raises(ValidationError, match="all stages disabled"):
        PipelineConfig.model_validate(raw)
    raw["train"]["enabled"] = True
    with pytest.raises(ValidationError, match="settings file path"):
        PipelineConfig.model_validate(raw)
