"""Calculate pooled held-out metrics and serialize split and fit diagnostics."""

from collections.abc import Iterable, Sequence
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.config.pipeline_schema import TrainConfig
from src.dataset.contracts import TrainingBatch
from src.feature.wide import FeatureSchema
from src.trainer.splits import TrainingPlan
from src.trainer.stacking import Evaluation, FittedResult


METRICS = ("mae", "rmse", "mape_pct")


def _score_candidate(held_out: pd.DataFrame, eligible: np.ndarray) -> dict:
    """Compute one candidate's metrics from its complete aligned repeat."""
    targets = held_out["y_true"].to_numpy()
    errors = np.abs(held_out["y_pred"].to_numpy() - targets)
    if not np.isfinite(errors).all() or not len(errors):
        raise ValueError("Invalid outer residuals; expected finite errors for every held-out wafer.")
    eligible_count = int(eligible.sum())
    mae, rmse = float(errors.mean()), float(np.sqrt(np.mean(errors ** 2)))
    mape = float(100 * np.mean(errors[eligible] / np.abs(targets[eligible]))) if eligible_count else None
    if not all(np.isfinite(metric) for metric in (mae, rmse, mape) if metric is not None):
        raise ValueError("Outer metrics overflowed; expected finite MAE, RMSE and any available MAPE.")
    return {"n_scored": len(errors), "mae": mae, "rmse": rmse, "mape_pct": mape,
            "n_mape_eligible": eligible_count, "n_mape_excluded": len(errors) - eligible_count}


def score_repeats(predictions: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """Score pooled outer predictions on identically matched wafers per repeat."""
    records = []
    expected_ids = tuple(sorted(predictions["row_id"].unique()))
    expected_candidates = set(predictions["candidate_id"])
    for repeat_id, repeat in predictions.groupby("repeat_id", sort=True):
        if set(repeat["candidate_id"]) != expected_candidates:
            raise ValueError(f"Repeat {repeat_id}: candidate set differs; expected {sorted(expected_candidates)}.")
        reference = repeat.loc[repeat["candidate_id"] == "stack"].sort_values("row_id")
        reference_targets = reference["y_true"].to_numpy()
        eligibility = np.abs(reference_targets) > cutoff
        for name, candidate in repeat.groupby("candidate_id", sort=False):
            rows = candidate.sort_values("row_id")
            if tuple(rows["row_id"]) != expected_ids or not np.array_equal(rows["y_true"], reference_targets):
                raise ValueError(f"Repeat {repeat_id} candidate {name}: expected identical wafer IDs and targets.")
            scores = _score_candidate(rows, eligibility)
            records.append(dict(scores, repeat_id=repeat_id, candidate_id=name))
    return pd.DataFrame(records, columns=["repeat_id", "candidate_id", "n_scored", *METRICS,
                                          "n_mape_eligible", "n_mape_excluded"])


def _summarize(series: pd.Series) -> dict:
    """Return repeat counts, mean and sample deviation with proper null handling."""
    values = series.dropna()
    return {"n_repeats": len(series), "n_valid_repeats": len(values),
            "mean": float(values.mean()) if len(values) else None,
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None}


def _metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize each candidate and metric across complete repeats."""
    return pd.DataFrame([dict(_summarize(group[metric]), candidate_id=name, metric=metric)
                         for name, group in metrics.groupby("candidate_id", sort=False) for metric in METRICS],
                        columns=["candidate_id", "metric", "n_repeats", "n_valid_repeats", "mean", "sample_sd"])


def _paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    """Subtract the stack metric from each matched baseline within each repeat."""
    primary = metrics.loc[metrics["candidate_id"] == "stack"].set_index("repeat_id")
    records = []
    for baseline_id, baseline in metrics.loc[metrics["candidate_id"] != "stack"].groupby("candidate_id", sort=False):
        deltas = baseline.set_index("repeat_id")[list(METRICS)] - primary[list(METRICS)]
        records.append(deltas.rename_axis(columns="metric").stack(future_stack=True).rename("delta")
                       .reset_index().assign(baseline_id=baseline_id))
    return pd.concat(records, ignore_index=True)[["repeat_id", "baseline_id", "metric", "delta"]]


def _paired_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired improvements and report positive counts separately."""
    records = []
    for (baseline, metric), rows in deltas.groupby(["baseline_id", "metric"], sort=False):
        summary = _summarize(rows["delta"])
        records.append(dict(summary, baseline_id=baseline, metric=metric, n_positive=int((rows["delta"] > 0).sum())))
    return pd.DataFrame(records).rename(columns={"mean": "mean_delta", "sample_sd": "sample_sd_delta"})


def write_evaluation(evaluation: Evaluation, destination: Path, cutoff: float) -> pd.DataFrame:
    """Persist complete repeat scores and paired comparisons for a full run."""
    metrics = score_repeats(evaluation.predictions, cutoff)
    deltas = _paired_deltas(metrics)
    tables = {"outer_predictions": evaluation.predictions, "repeat_metrics": metrics,
              "metrics_summary": _metric_summary(metrics), "paired_deltas": deltas,
              "paired_summary": _paired_summary(deltas)}
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False)
    return metrics


def training_manifest(batch: TrainingBatch, schema: FeatureSchema, config: TrainConfig,
                      plan: TrainingPlan, input_hashes: dict[str, str]) -> dict:
    """Describe inputs, effective settings, feature ordering, and runtime provenance."""
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "pydantic", "PyYAML", "matplotlib", "pytest")
    packages += tuple(sorted({base.estimator for base in config.models} & {"lightgbm", "catboost", "xgboost"}))
    runtime = {"python": platform.python_version(), "packages": {package: version(package) for package in packages}}
    return {"input_hashes": input_hashes, "settings": config.model_dump(mode="json"), "runtime": runtime,
            "base_order": [base.name for base in config.models], "features": list(zip(schema.columns, schema.dtypes)),
            "identity": {"row_id": "wafer_id", "group_id": "group", "row_order": "lexicographic wafer_id"},
            "n_rows": len(batch.X), "n_groups": len(set(batch.groups)), "estimator_seeds": plan.estimator_seeds,
            "distinct_outer_partitions": plan.distinct_outer_partitions}


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Stream strict JSON records to the run staging directory."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in records:
            stream.write(json.dumps(item, allow_nan=False, ensure_ascii=False) + "\n")


def write_trace_reports(destination: Path, splits: Iterable[dict], evaluations: Sequence[Evaluation],
                        final: FittedResult, manifest: dict) -> None:
    """Persist model and split diagnostics separately from the deployable bundle."""
    (destination / "manifest.json").write_text(json.dumps(manifest, allow_nan=False, indent=2), encoding="utf-8")
    _write_jsonl(destination / "splits.jsonl", splits)
    all_results = (*evaluations, final)
    searches = [trial for result in all_results for trial in result.search_trials]
    if searches:
        _write_jsonl(destination / "search_trials.jsonl", searches)
    analysis = destination / "analysis"
    analysis.mkdir(exist_ok=True)
    weights = pd.DataFrame([row for result in all_results for row in result.weights])
    weights.to_csv(analysis / "meta_weights.csv", index=False)
    fits = [fit for result in all_results for fit in result.convergence.fits]
    if fits:
        points = [point for result in all_results for point in result.convergence.points]
        pd.DataFrame(fits).to_csv(analysis / "convergence_fits.csv", index=False)
        pd.DataFrame(points).to_csv(analysis / "convergence.csv", index=False)
