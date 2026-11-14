"""Exercise the public headless pipeline and staged failure behavior."""

import json
from pathlib import Path
import re

import pandas as pd
import pytest
import yaml

from src.main import _resolve_bundle, main


def _demo_settings(name: str) -> dict:
    """Read the committed configuration as a base for a small integration run."""
    return yaml.safe_load((Path("src/config") / name).read_text(encoding="utf-8"))


def _run(destination: Path, **enabled: dict) -> Path:
    """Run the enabled stages through one workflow YAML."""
    workflow = {}
    for name in ("generate", "train", "predict", "optimize"):
        settings_path = destination / f"{name}.yaml"
        if name in enabled:
            settings_path.write_text(yaml.safe_dump(enabled[name]), encoding="utf-8")
        workflow[name] = {"enabled": name in enabled, "config": str(settings_path)}
    entry_path = destination / "workflow.yaml"
    entry_path.write_text(yaml.safe_dump(workflow), encoding="utf-8")
    return main(["--config", str(entry_path), "--headless"])


def _training_settings(source: Path, output: Path) -> dict:
    """Prepare a small grouped job that enables both tuning and ablation."""
    settings = _demo_settings("train_demo.yaml")
    settings["data"]["csv_path"] = str(source)
    settings["cv"].update(outer_folds=2, outer_repeats=2, inner_folds=2, search_folds=2)
    settings["models"][0].update(enable_search=True, search_grid={"alpha": [0.1, 1.0, 10.0]})
    settings["models"][1]["parameters"]["n_estimators"] = 5
    settings["output"]["directory"] = str(output)
    return settings


def _search_settings(mode: str, model: Path | None, output: Path) -> dict:
    """Build settings for a short search in the requested mode."""
    settings = _demo_settings(f"optimize_{mode}_demo.yaml")
    settings["bundle_directory"] = str(model) if model is not None else None
    settings["output"]["directory"] = str(output)
    settings["search"].update(max_candidates=12, max_attempts=60, batch_size=5)
    return settings


def _verify_search(run: Path) -> None:
    """Recompute fixed/free results from the persisted trial ledger."""
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
    source, train, predicted = (tmp_path / name for name in ("history.csv", "train", "predict"))
    generator.update(n_groups=16, output_path=str(source))
    predict = _demo_settings("predict_demo.yaml")
    predict.update(bundle_directory=None, output={"directory": str(predicted)})
    fixed = _search_settings("fixed", None, tmp_path / "fixed")
    result = _run(tmp_path, generate=generator, train=_training_settings(source, train),
                  predict=predict, optimize=fixed)
    stamp = result.name.removeprefix("fixed_")
    assert re.fullmatch(r"\d{8}T\d{12}Z", stamp)
    train, predicted = (tmp_path / f"{name}_{stamp}" for name in ("train", "predict"))
    model = train / "model"
    metrics = pd.read_csv(train / "repeat_metrics.csv")
    assert set(metrics.candidate_id) == {"base:ridge", "base:random_forest", "equal_mean", "stack"}
    assert len(metrics) == 8
    assert metrics.r2.notna().all()
    wide = pd.read_csv(train / "wide_table.csv")
    held_out = pd.read_csv(train / "outer_predictions.csv").query("repeat_id == 1 and candidate_id == 'stack'")
    joined = wide.merge(held_out, left_on="wafer_id", right_on="row_id", validate="one_to_one")
    assert len(joined) == len(wide) == len(held_out)
    assert joined.target_stress.tolist() == joined.y_true.tolist()
    assert "step_1_temperature_observed" in wide
    assert (train / "search_trials.jsonl").is_file()
    assert (train / "figures" / "transition_ablation.png").is_file()
    trace = pd.read_csv(train / "analysis" / "convergence.csv")
    fits = pd.read_csv(train / "analysis" / "convergence_fits.csv")
    assert set(trace["step_unit"]) == {"tree", "iteration"}
    assert len(fits.query("scope == 'meta'")) == 5
    assert set(fits.query("estimator == 'ridge'").status) == {"unavailable"}
    assert set(trace.query("scope == 'base_refit' and stage == 'outer'").split) == {"train", "validation"}
    assert set(trace.query("scope == 'base_refit' and stage == 'final'").split) == {"train"}
    assert (train / "figures" / "convergence_meta.png").is_file()
    table = pd.read_csv(predicted / "predictions.csv")
    assert table.request_id.tolist() == ["synthetic-example"]
    assert table.predicted_stress.notna().all()
    _verify_search(result)
    _verify_search(_run(tmp_path, optimize=_search_settings("free", model, tmp_path / "free")))
    predict["bundle_directory"] = str(model)
    again = _run(tmp_path, predict=predict)
    assert again != predicted and (again / "predictions.csv").is_file()


def test_failed_stage_is_not_published(tmp_path: Path) -> None:
    """A failed read does not publish any training or staging artifacts."""
    destination = tmp_path / "failed"
    settings = _training_settings(tmp_path / "absent.csv", destination)
    with pytest.raises(FileNotFoundError):
        _run(tmp_path, train=settings)
    assert not destination.exists()
    assert not list(tmp_path.glob("failed_*"))
    assert not list(tmp_path.glob(".failed_*.stage-*"))


def test_bundle_selection(tmp_path: Path) -> None:
    """Use a current bundle only on request and fail clearly when none exists."""
    current = tmp_path / "current"
    assert _resolve_bundle(tmp_path, None, current) == current
    assert _resolve_bundle(tmp_path, Path("explicit"), current) == tmp_path / "explicit"
    with pytest.raises(ValueError, match="bundle_directory=null"):
        _resolve_bundle(tmp_path, None, None)


@pytest.mark.parametrize("evaluate", [True, False])
def test_base_workflow(tmp_path: Path, evaluate: bool) -> None:
    """Publish one model with mode-specific reports and reuse it from a later workflow."""
    from src.dataset.requests import read_requests
    from src.inference.bundle import load_bundle
    from src.inference.predict import predict_sequences

    source = tmp_path / "history.csv"
    generator = _demo_settings("pseudo_data_config.yaml")
    generator.update(n_groups=12, output_path=str(source))
    settings = _training_settings(source, tmp_path / "base")
    settings.update(mode="base", active_model="ridge", evaluate=evaluate, meta=None)
    settings["models"][0].update(enable_search=False, search_grid=None)
    settings["analysis"]["transition_ablation"] = evaluate
    train = _run(tmp_path, generate=generator, train=settings)
    manifest = json.loads((train / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["mode"], manifest["active_model"], manifest["base_order"]) == ("base", "ridge", ["ridge"])
    assert manifest["evaluation_status"] == ("grouped_cv" if evaluate else "not_evaluated")
    assert (train / "wide_table.csv").is_file()
    assert (train / "figures" / "convergence_availability.png").is_file()
    assert not (train / "analysis" / "meta_weights.csv").exists()
    assert not (train / "paired_summary.csv").exists()
    for filename in ("outer_predictions.csv", "repeat_metrics.csv", "metrics_summary.csv",
                     "figures/held_out_parity.png", "analysis/ablation_repeat_metrics.csv",
                     "figures/transition_ablation.png"):
        assert (train / filename).is_file() == evaluate
    if evaluate:
        assert set(pd.read_csv(train / "repeat_metrics.csv").candidate_id) == {"base:ridge"}

    bundle_path = train / "model"
    predict = _demo_settings("predict_demo.yaml")
    predict.update(bundle_directory=str(bundle_path), output={"directory": str(tmp_path / "predicted")})
    result = _run(tmp_path, predict=predict)
    requests = read_requests(Path(predict["request_path"]))
    expected = predict_sequences(load_bundle(bundle_path), [request.sequence for request in requests])
    assert expected.values.tolist() == expected.base["ridge"].tolist()
    assert pd.read_csv(result / "predictions.csv").predicted_stress.tolist() == pytest.approx(expected.values)
    if evaluate:
        optimize = _run(tmp_path, optimize=_search_settings("fixed", bundle_path, tmp_path / "fixed"))
        _verify_search(optimize)
        best = json.loads((optimize / "optimization.json").read_text(encoding="utf-8"))["ranked_results"][0]
        assert best["base_predictions"] == [best["predicted_stress"]]
