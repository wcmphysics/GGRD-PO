"""Acceptance tests for grouped stacking, scoring and deployable inference."""

from dataclasses import replace
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config.pipeline_schema import RidgeGrid, TrainConfig, load_config
from src.dataset.contracts import DepositionStep, ProcessSequence, TrainingBatch, WaferSample
from src.feature.wide import FeatureSchema, build_training_batch
from src.inference.bundle import DeploymentBundle, load_bundle, save_bundle
from src.inference.predict import predict_sequences
from src.model.stack import fit_weights, predict_meta
from src.trainer import stacking
from src.trainer.reporting import score_repeats, write_evaluation
from src.trainer.splits import StackPlan, preflight_training, split_records
from src.trainer.stacking import Evaluation, evaluate_stack, fit_deployment


@pytest.fixture
def train_config() -> TrainConfig:
    """Use the demo schema with small enabled search and forest budgets."""
    config = load_config(Path("src/config/train_demo.yaml"), TrainConfig)
    config.cv.outer_folds = 2
    config.cv.inner_folds = 2
    config.cv.outer_repeats = 1
    config.cv.search_folds = 2
    config.features.N_step = 2
    config.models[0].enable_search = True
    config.models[0].search_grid = RidgeGrid(alpha=[0.1, 1.0])
    config.models[1].parameters.n_estimators = 2
    return config


@pytest.fixture
def samples() -> list[WaferSample]:
    """Supply replicated wafer groups, one truly missing field and one padded slot."""
    return [WaferSample(f"{group:02d}-{replicate}", f"group-{group:02d}",
                        ProcessSequence((DepositionStep("TiN", "recipe", None, 100.0 + group, 1.0),), ()),
                        group * 0.1 + replicate * 0.01)
            for group in range(12) for replicate in range(group % 2 + 1)]


def _fixture_batch(samples: list[WaferSample], config: TrainConfig) -> tuple[FeatureSchema, TrainingBatch]:
    """Use duplicate DataFrame labels to exercise positional alignment."""
    schema = FeatureSchema(config.features.N_step)
    batch = build_training_batch(samples, schema)
    batch.X.index = pd.Index(np.arange(len(batch.X)) % 3 + 20)
    return schema, batch


def _expected_fit_rows(subset: TrainingBatch, config: TrainConfig, plan: StackPlan) -> list[tuple[str, ...]]:
    """Assemble the exact fit rows for the fixture's Ridge search and forest."""
    expected = []
    allowed_folds = [(index, train) for index, (train, _) in enumerate(plan.inner.folds, start=1)]
    allowed_folds.append((None, np.arange(len(subset.X))))
    for inner_fold, allowed in allowed_folds:
        base_rows = subset.wafer_ids[allowed]
        search_training = plan.searches[inner_fold, "ridge"].folds
        for _ in config.models[0].search_grid.alpha:
            expected.extend(tuple(base_rows[search_train]) for search_train, _ in search_training)
        expected.extend((tuple(base_rows), tuple(base_rows)))
    return expected


def test_group_plans_and_fit_scope(train_config: TrainConfig, samples: list[WaferSample],
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """Every logged fold is group-safe and each real fit sees only its permitted rows."""
    schema, batch = _fixture_batch(samples, train_config)
    batch.X.index = batch.wafer_ids
    plan = preflight_training(batch, train_config)
    plans = list(split_records(batch, plan))
    assert plans == list(split_records(batch, preflight_training(batch, train_config)))
    for record in plans:
        assert {row["group_id"] for row in record["train"]}.isdisjoint(row["group_id"] for row in record["holdout"])

    from sklearn.pipeline import Pipeline

    original_fit = Pipeline.fit
    actual: list[tuple[str, ...]] = []

    # Observe pipeline fits after sklearn clones and slices the tuning candidates.
    def fit_spy(pipeline: Pipeline, features: pd.DataFrame, target: np.ndarray, **kwargs: object) -> Pipeline:
        """Record only the base pipeline fits, excluding the internal numeric pipeline."""
        if "estimator" in pipeline.named_steps:
            actual.append(tuple(features.index))
        return original_fit(pipeline, features, target, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fit_spy)
    fold_plan = plan.outer_stacks[1, 1]
    training, _ = plan.outer[0].folds[0]
    subset = batch.take_positions(training)
    context = stacking._FitContext(fold_plan, "full", plan.estimator_seeds)
    fitted = stacking._fit_stack(subset, schema, train_config, context)
    assert fitted.oof.shape == (len(subset.X), 2)
    assert actual == _expected_fit_rows(subset, train_config, fold_plan)


def test_heldout_label_isolation(train_config: TrainConfig, samples: list[WaferSample]) -> None:
    """Neither local holdout label nor outer holdout label reaches the predicting fit."""
    schema, batch = _fixture_batch(samples, train_config)
    plan = preflight_training(batch, train_config)
    training, outer_holdout = plan.outer[0].folds[0]
    fold_plan = plan.outer_stacks[1, 1]
    context = stacking._FitContext(fold_plan, "full", plan.estimator_seeds)
    subset = batch.take_positions(training)
    _, inner_holdout = fold_plan.inner.folds[0]
    changed_labels = subset.y.copy()
    changed_labels[inner_holdout] += 5
    before = stacking._fit_stack(subset, schema, train_config, context)
    after = stacking._fit_stack(replace(subset, y=changed_labels), schema, train_config, context)
    np.testing.assert_allclose(before.oof.iloc[inner_holdout], after.oof.iloc[inner_holdout], rtol=0, atol=1e-10)
    assert before.search_trials[0] == after.search_trials[0]

    outer_targets = batch.y.copy()
    outer_targets[outer_holdout] += 5
    original = evaluate_stack(batch, schema, train_config, plan)
    perturbed = evaluate_stack(replace(batch, y=outer_targets), schema, train_config, plan)
    first = original.predictions.query("outer_fold_id == 1")
    second = perturbed.predictions.query("outer_fold_id == 1")
    np.testing.assert_allclose(first.y_pred, second.y_pred, rtol=0, atol=1e-10)
    assert not np.allclose(first.y_true, second.y_true)
    assert [trial for trial in original.search_trials if trial["outer_fold_id"] == 1] == [
        trial for trial in perturbed.search_trials if trial["outer_fold_id"] == 1]


def test_final_fit_warnings(train_config: TrainConfig, samples: list[WaferSample],
                            caplog: pytest.LogCaptureFixture) -> None:
    """Exercise all-missing audit context and full-data base and meta refits."""
    schema, batch = _fixture_batch(samples, train_config)
    plan = preflight_training(batch, train_config)
    with caplog.at_level(logging.WARNING):
        final = fit_deployment(batch, schema, train_config, plan)
    warnings = [record.message for record in caplog.records if "ALL_MISSING_NUMERIC_FEATURES" in record.message]
    assert len(warnings) == (train_config.cv.inner_folds + 1) * (train_config.cv.search_folds + 2)
    assert all("applicable_but_unrecorded" in record and "no_applicable_rows" in record
               and "final" in record and "step_1_temperature" not in record for record in warnings)
    assert any("winner_refit" in record for record in warnings)
    assert all(trial["stage"] == "final" for trial in final.search_trials)
    assert list(final.oof.index) == list(batch.wafer_ids)
    assert np.isclose(final.stack.weights.sum(), 1) and np.all(final.stack.weights >= 0)


def test_final_bundle_reload(train_config: TrainConfig, samples: list[WaferSample], tmp_path: Path) -> None:
    """Check reload parity, ordered base predictions and compatibility validation."""
    schema, batch = _fixture_batch(samples, train_config)
    final = fit_deployment(batch, schema, train_config, preflight_training(batch, train_config))
    bundle = DeploymentBundle(final.stack, schema, {sample.group: sample.sequence for sample in samples},
                              train_config.model_dump(mode="json"), "synthetic-fixture", "stress_unit",
                              {"time": "s", "temperature": "C", "thickness": "nm"})
    directory = tmp_path / "model"
    save_bundle(bundle, directory)
    sequences = [samples[-1].sequence, samples[0].sequence]
    original = predict_sequences(bundle, sequences)
    reloaded = predict_sequences(load_bundle(directory), sequences)
    np.testing.assert_allclose(reloaded.values, original.values, rtol=0, atol=1e-10)
    assert np.all(reloaded.values >= reloaded.base.min(axis=1) - 1e-10)
    assert np.all(reloaded.values <= reloaded.base.max(axis=1) + 1e-10)
    with pytest.raises(ValueError, match="Meta base names"):
        predict_meta(bundle.stack, reloaded.base.iloc[:, ::-1])
    metadata = directory / "metadata.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["bundle_format"] = -1
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bundle_format"):
        load_bundle(directory)
    payload["bundle_format"] = 1
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with (directory / "bundle.joblib").open("ab") as stream:
        stream.write(b"changed payload")
    with pytest.raises(ValueError, match="payload_sha256"):
        load_bundle(directory)


def test_infeasible_search(train_config: TrainConfig, samples: list[WaferSample]) -> None:
    """Reject too many search folds and malformed labels before a fit can begin."""
    _, batch = _fixture_batch(samples, train_config)
    train_config.cv.search_folds = 5
    with pytest.raises(ValueError, match="inner=1 base=ridge search:.*expected at least 5 groups"):
        preflight_training(batch, train_config)
    groups = batch.groups.astype(object)
    groups[0] = None
    with pytest.raises(ValueError, match="group"):
        preflight_training(replace(batch, groups=groups), train_config)


def test_pooled_paired_metrics(tmp_path: Path) -> None:
    """Uneven folds, symmetric MAPE eligibility, nulls and paired signs are auditable."""
    frame = pd.DataFrame({"repeat_id": 1, "outer_fold_id": [1, 2, 2, 2],
                          "row_id": list("abcd"), "group_id": list("ABCD"),
                          "y_true": [100.0, -50.0, 0.0, 0.01], "y_pred": [110.0, -40.0, 3.0, 1.0],
                          "candidate_id": "stack"})
    predictions = pd.concat([frame, frame.assign(candidate_id="equal_mean", y_pred=[120.0, -30.0, 6.0, 2.0])])
    scores = write_evaluation(Evaluation(predictions, [], []), tmp_path, 0.01).set_index("candidate_id")
    assert scores.loc["stack", "n_mape_eligible"] == 2
    assert scores.loc["stack", "n_mape_excluded"] == 2
    assert scores.loc["stack", "mape_pct"] == pytest.approx(15)
    assert scores.loc["stack", "mae"] == pytest.approx((10 + 10 + 3 + 0.99) / 4)
    assert scores.loc["stack", "rmse"] == pytest.approx(np.sqrt((100 + 100 + 9 + 0.99**2) / 4))
    paired = pd.read_csv(tmp_path / "paired_summary.csv").set_index("metric")
    assert paired.loc["mape_pct", "mean_delta"] == pytest.approx(15)
    assert pd.isna(paired.loc["mae", "sample_sd_delta"])
    assert score_repeats(predictions, 100)["mape_pct"].isna().all()
    write_evaluation(Evaluation(predictions, [], []), tmp_path, 100)
    assert pd.read_csv(tmp_path / "paired_deltas.csv").query("metric == 'mape_pct'")["delta"].isna().all()


def test_meta_solver_error(train_config: TrainConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """A solver failure surfaces context instead of falling back to equal weights."""
    def fail_solver(*args: object, **kwargs: object) -> SimpleNamespace:
        """Supply a non-success status even with a numerically feasible vector."""
        return SimpleNamespace(success=False, x=np.array([0.5, 0.5]), message="iteration limit")

    monkeypatch.setattr("src.model.stack.minimize", fail_solver)
    with pytest.raises(ValueError, match="fixture: meta solver returned iteration limit"):
        fit_weights(np.array([[1.0, 2.0]]), np.array([1.0]), train_config.meta, "fixture")
