"""Exercise the public headless pipeline and staged failure behavior."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import main


def _demo_settings(name: str) -> dict:
    """Read the committed configuration as a base for a small integration run."""
    return yaml.safe_load((Path("src/config") / name).read_text(encoding="utf-8"))


def _run(stage: str, settings: dict, destination: Path) -> Path:
    """Run an actual stage dispatch against a temporary YAML file."""
    settings_path = destination / f"{stage}.yaml"
    settings_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return main([stage, "--config", str(settings_path), "--headless"])


def _training_settings(source: Path, output: Path) -> dict:
    """Prepare a small grouped job that enables both tuning and ablation."""
    settings = _demo_settings("config.yaml")
    settings["data"]["csv_path"] = str(source)
    settings["cv"].update(outer_folds=2, outer_repeats=2, inner_folds=2, search_folds=2)
    settings["models"][0].update(enable_search=True, search_grid={"alpha": [0.1, 1.0, 10.0]})
    settings["models"][1]["parameters"]["n_estimators"] = 5
    settings["output"]["directory"] = str(output)
    return settings


def _verify_search(mode: str, model: Path, root: Path) -> None:
    """Recompute fixed/free results from the persisted trial ledger."""
    settings = _demo_settings(f"optimize_{mode}_demo.yaml")
    settings["bundle_directory"] = str(model)
    settings["output"]["directory"] = str(root / mode)
    settings["search"].update(max_candidates=12, max_attempts=60, batch_size=5)
    run = _run("optimize", settings, root)
    summary = json.loads((run / "optimization.json").read_text(encoding="utf-8"))
    ledger = pd.read_json(run / "optimization_trials.jsonl", lines=True, precise_float=True)
    assert summary["scored"] == len(ledger) == 12
    assert summary["ranked_results"][0]["objective"] == pytest.approx(ledger.objective.min())
    assert ledger.best_so_far_objective.tolist() == pytest.approx(ledger.objective.cummin().tolist())
    assert (run / "synthetic_audit.json").is_file()
    assert (run / "figures" / "optimization.png").is_file()


def test_end_to_end(tmp_path: Path) -> None:
    """Generate, train with tuning and ablation, reload, predict, and run both search modes."""
    generator = _demo_settings("pseudo_data_config.yaml")
    generator.update(n_groups=16, output_path=str(tmp_path / "history.csv"))
    source = _run("generate", generator, tmp_path)
    train = _run("train", _training_settings(source, tmp_path / "train"), tmp_path)
    metrics = pd.read_csv(train / "repeat_metrics.csv")
    assert set(metrics.candidate_id) == {"base:ridge", "base:random_forest", "equal_mean", "stack"}
    assert len(metrics) == 8
    assert (train / "search_trials.jsonl").is_file()
    assert (train / "figures" / "transition_ablation.png").is_file()
    model = train / "model"
    predict = _demo_settings("predict_demo.yaml")
    predict.update(bundle_directory=str(model), output={"directory": str(tmp_path / "predict")})
    predicted = _run("predict", predict, tmp_path)
    table = pd.read_csv(predicted / "predictions.csv")
    assert table.request_id.tolist() == ["synthetic-example"]
    assert table.predicted_stress.notna().all()
    _verify_search("fixed", model, tmp_path)
    _verify_search("free", model, tmp_path)
    with pytest.raises(FileExistsError, match="fresh directory"):
        _run("predict", predict, tmp_path)


def test_failed_stage_is_not_published(tmp_path: Path) -> None:
    """A failed read does not publish any training or staging artifacts."""
    destination = tmp_path / "failed"
    settings = _training_settings(tmp_path / "absent.csv", destination)
    with pytest.raises(FileNotFoundError):
        _run("train", settings, tmp_path)
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed.stage-*"))
