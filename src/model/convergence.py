"""Measure fitted model trajectories without changing fold training or selection."""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from src.dataset.contracts import TrainingBatch


def _tree_stages(estimator: object, features: NDArray) -> Iterator[NDArray]:
    """Yield predictions from increasing prefixes of the fitted forest."""
    running = np.zeros(len(features), dtype=np.float64)
    for count, tree in enumerate(estimator.estimators_, start=1):
        running += tree.predict(features)
        yield running / count


def _boosting_stages(kind: str, estimator: object, features: NDArray) -> Iterator[NDArray]:
    """Yield the model's actual predictions at each fitted boosting round."""
    if kind == "catboost":
        yield from estimator.staged_predict(features, thread_count=1)
    elif kind == "lightgbm":
        booster = estimator.booster_
        for step in range(1, booster.current_iteration() + 1):
            yield booster.predict(features, num_iteration=step, num_threads=1)
    else:
        for step in range(1, estimator.get_booster().num_boosted_rounds() + 1):
            yield estimator.predict(features, iteration_range=(0, step))


def _score_path(predictions: Iterator[NDArray], targets: NDArray, unit: str, split: str,
                start: int = 1) -> Iterator[dict]:
    """Score successive fitted states in the original target units."""
    for step, prediction in enumerate(predictions, start=start):
        error = np.asarray(prediction).reshape(-1) - targets
        yield {"step": step, "step_unit": unit, "metric": "mse", "split": split,
               "value": float(np.mean(error ** 2))}


def _cached_path(estimator: object, holdout: TrainingBatch | None, pipeline: Pipeline) -> Iterator[dict]:
    """Export objectives recorded by the iterative solvers."""
    for point in getattr(estimator, "convergence_", ()):
        state = {"step": point["iteration"], "step_unit": "iteration", "split": "train",
                 "optimizer_restart": point["trajectory"], "solver_converged": point.get("converged"),
                 "optimizer_selected": point.get("selected")}
        yield dict(state, metric=point["objective_name"], value=point["objective"])
        if "train_mse" in point:
            yield dict(state, metric="mse", value=point["train_mse"])
    iterates = getattr(estimator, "iterates_", ())
    if holdout is not None and iterates:
        features = pipeline.named_steps["features"].transform(holdout.X)
        predictions = (features @ coefficients + intercept for coefficients, intercept in iterates)
        yield from _score_path(predictions, holdout.y, "iteration", "validation", start=0)


def _base_path(kind: str, pipeline: Pipeline, train: TrainingBatch,
               holdout: TrainingBatch | None) -> Iterator[dict]:
    """Read solver steps or score tree prefixes after a completed fit."""
    estimator = pipeline.named_steps["estimator"]
    if kind in {"elastic_net", "gaussian_process"}:
        yield from _cached_path(estimator, holdout, pipeline)
    elif kind in {"random_forest", "lightgbm", "catboost", "xgboost"}:
        for split, batch in (("train", train), ("validation", holdout)):
            if batch is None:
                continue
            features = pipeline.named_steps["features"].transform(batch.X)
            if kind == "random_forest":
                yield from _score_path(_tree_stages(estimator, features), batch.y, "tree", split)
            else:
                yield from _score_path(_boosting_stages(kind, estimator, features), batch.y, "boosting_round", split)


def _selected(points: list[dict], every: int) -> Iterator[dict]:
    """Retain the first, periodic, and final point of every reported series."""
    series: dict[tuple[object, ...], list[dict]] = {}
    for point in points:
        key = point["metric"], point["split"], point.get("optimizer_restart", 0)
        series.setdefault(key, []).append(point)
    for path in series.values():
        for index, point in enumerate(path):
            if index in {0, len(path) - 1} or point["step"] % every == 0:
                yield point


def _unavailable(kind: str) -> str:
    """Describe why an estimator has no recorded intermediate objectives."""
    return {
        "ridge": "The configured dense Ridge solver exposes no iteration history.",
        "knn": "Nearest-neighbor fitting has no iterative prediction loss.",
        "svr": "Libsvm does not expose per-iteration objective or predictions.",
        "gaussian_process": "Kernel hyperparameters are fixed or kernel optimization is disabled.",
    }.get(kind, "No fitted states were available.")


def base_convergence(labels: dict, pipeline: Pipeline, train: TrainingBatch, holdout: TrainingBatch | None,
                     every: int) -> tuple[list[dict], dict]:
    """Return labeled fit diagnostics, separate from persisted predictor state."""
    kind = labels["estimator"]
    estimator = pipeline.named_steps["estimator"]
    points = list(_base_path(kind, pipeline, train, holdout))
    for attribute in ("convergence_", "iterates_"):
        if hasattr(estimator, attribute):
            delattr(estimator, attribute)
    if any(not np.isfinite(point["value"]) for point in points):
        raise ValueError(f"Base convergence {labels}: expected finite objective values")
    reason = "" if points else _unavailable(kind)
    status = dict(labels, status="recorded" if points else "unavailable", reason=reason)
    return [dict(labels, **point) for point in _selected(points, every)], status
