"""Train fold-local stacks and generate matched held-out predictions."""

from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config.pipeline_schema import ModelConfig, TrainConfig
from src.dataset.contracts import TrainingBatch
from src.feature.preprocess import warn_empty_columns
from src.feature.wide import FeatureSchema
from src.model.base_estimator import effective_parameters
from src.model.convergence import base_convergence
from src.model.iterative import trace_optimization
from src.model.stack import FittedStack, fit_weights, make_pipeline, predict_base, predict_base_matrix, predict_meta
from src.trainer.splits import StackPlan, TrainingPlan, preflight_training, split_records


@dataclass
class FitDiagnostics:
    """Collect convergence points and fit-level availability outside the fitted stack."""

    points: list[dict] = field(default_factory=list)
    fits: list[dict] = field(default_factory=list)

    def extend(self, other: "FitDiagnostics") -> None:
        """Combine traces from independent outer fits without dropping identifiers."""
        self.points.extend(other.points)
        self.fits.extend(other.fits)


@dataclass(frozen=True)
class FittedResult:
    """A stack and the fitting diagnostics that are excluded from deployment."""

    stack: FittedStack
    oof: pd.DataFrame
    search_trials: list[dict]
    weights: list[dict]
    convergence: FitDiagnostics = field(default_factory=FitDiagnostics)


@dataclass(frozen=True)
class Evaluation:
    """Long-format held-out predictions and associated fitting diagnostics."""

    predictions: pd.DataFrame
    search_trials: list[dict]
    weights: list[dict]
    convergence: FitDiagnostics = field(default_factory=FitDiagnostics)


@dataclass(frozen=True)
class _FitContext:
    """Carry one preflighted fit, its reporting label, and estimator seeds."""

    plan: StackPlan
    variant: str
    estimator_seeds: dict[str, int]
    convergence_every: int | None = None
    diagnostics: FitDiagnostics = field(default_factory=FitDiagnostics)

    def labels(self) -> dict:
        """Identify this fit independently of its inner folds."""
        return {"stage": self.plan.stage, "variant": self.variant, "repeat_id": self.plan.repeat_id,
                "outer_fold_id": self.plan.outer_fold_id}

    def record_base(self, base: ModelConfig, pipeline: Pipeline, train: TrainingBatch,
                    holdout: TrainingBatch | None, inner_fold: int | None) -> None:
        """Score diagnostics only after the base fit and parameter search have finished."""
        if self.convergence_every is None:
            return
        labels = dict(self.labels(), model=base.name, estimator=base.estimator, inner_fold_id=inner_fold,
                      scope="base_refit" if inner_fold is None else "inner_oof")
        points, status = base_convergence(labels, pipeline, train, holdout, self.convergence_every)
        self.diagnostics.points.extend(points)
        self.diagnostics.fits.append(status)

    def record_meta(self, trace: list[float]) -> None:
        """Record the solver objective on the OOF matrix that fits the meta model."""
        every = self.convergence_every
        if every is None:
            return
        labels = dict(self.labels(), model="meta", estimator="slsqp", inner_fold_id=None, scope="meta")
        self.diagnostics.fits.append(dict(labels, status="recorded", reason=""))
        self.diagnostics.points.extend(dict(labels, step=step, step_unit="iteration", split="train",
                                            metric="simplex_objective", value=value)
                                       for step, value in enumerate(trace)
                                       if step % every == 0 or step == len(trace) - 1)


def _with_features(batch: TrainingBatch, schema: FeatureSchema) -> TrainingBatch:
    """Project onto one predeclared feature variant while retaining all row identities."""
    return TrainingBatch(batch.X.loc[:, list(schema.columns)], batch.y, batch.groups, batch.wafer_ids,
                         batch.row_positions)


def _search_record(search: GridSearchCV, context: dict, seed: int) -> dict:
    """Export GridSearchCV trial scores in actual evaluation order."""
    results = search.cv_results_
    candidates = []
    for index, setting in enumerate(results["params"]):
        candidates.append({"parameters": {name.removeprefix("estimator__"): value for name, value in setting.items()},
                           "mean_mae": -float(results["mean_test_score"][index]),
                           "sd_mae": float(results["std_test_score"][index])})
    return {**context, "seed": seed, "candidates": candidates,
            "winner_parameters": candidates[search.best_index_]["parameters"],
            "winner_mean_mae": -float(search.best_score_)}


def _fit_fresh_base(base: ModelConfig, batch: TrainingBatch, schema: FeatureSchema,
                    context: _FitContext, inner_fold: int | None) -> tuple[Pipeline, dict | None]:
    """Fit a fresh base, including optional train-only tuning on an existing plan."""
    labels = dict(context.labels(), model=base.name, inner_fold_id=inner_fold)
    pipeline = make_pipeline(base, schema, context.estimator_seeds[base.name])
    capture = trace_optimization(len(batch.X)) if context.convergence_every is not None else nullcontext()
    try:
        if not base.enable_search:
            warn_empty_columns(batch.X, schema, labels)
            with capture:
                return pipeline.fit(batch.X, batch.y), None
        search_plan = context.plan.searches[inner_fold, base.name]
        search_plan.validate(batch)
        for fold_id, (training, _) in enumerate(search_plan.folds, start=1):
            warn_empty_columns(batch.X.iloc[training], schema, dict(labels, search_fold_id=fold_id))
        warn_empty_columns(batch.X, schema, dict(labels, search_fit="winner_refit"))
        grid = {f"estimator__{name}": values for name, values in base.search_grid.model_dump(exclude_none=True).items()}
        search = GridSearchCV(pipeline, grid, cv=search_plan.folds, scoring="neg_mean_absolute_error",
                              refit=True, error_score="raise", n_jobs=1)
        with capture:
            search.fit(batch.X, batch.y)
        search_id = [context.plan.stage, context.plan.repeat_id, context.plan.outer_fold_id, inner_fold, base.name]
        return search.best_estimator_, _search_record(search, dict(labels, search_id=search_id), search_plan.seed)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"Base fit {labels} failed for {len(batch.X)} training rows: {error}") from error


def _oof_predictions(batch: TrainingBatch, schema: FeatureSchema, config: TrainConfig,
                     context: _FitContext) -> tuple[pd.DataFrame, list[dict]]:
    """Generate exactly one out-of-fold prediction per base and training wafer."""
    base_names = [base.name for base in config.models]
    values = np.full((len(batch.X), len(base_names)), np.nan)
    writes = np.zeros_like(values, dtype=int)
    searches = []
    for inner_fold, (train, holdout) in enumerate(context.plan.inner.folds, start=1):
        training = batch.take_positions(train)
        validation = batch.take_positions(holdout)
        for base_position, base in enumerate(config.models):
            fitted, search = _fit_fresh_base(base, training, schema, context, inner_fold)
            label = f"{context.labels()} inner_fold={inner_fold} base={base.name} OOF"
            values[holdout, base_position] = predict_base(fitted, validation.X, label)
            context.record_base(base, fitted, training, validation, inner_fold)
            writes[holdout, base_position] += 1
            if search is not None:
                searches.append(search)
    if not np.all(writes == 1) or not np.isfinite(values).all():
        raise ValueError(f"{context.labels()}: invalid OOF matrix; expected one finite prediction per wafer and base.")
    return pd.DataFrame(values, index=pd.Index(batch.wafer_ids, name="row_id"), columns=base_names), searches


def _fit_stack(batch: TrainingBatch, schema: FeatureSchema, config: TrainConfig, context: _FitContext,
               diagnostic_holdout: TrainingBatch | None = None) -> FittedResult:
    """Fit one meta model from OOF predictions, then refit the bases on this subset."""
    context.plan.inner.validate(batch)
    oof, searches = _oof_predictions(batch, schema, config, context)
    trace = [] if context.convergence_every is not None else None
    weights = fit_weights(oof.to_numpy(), batch.y, config.meta, str(context.labels()), trace)
    if trace is not None:
        context.record_meta(trace)
    pipelines = []
    for base in config.models:
        estimator, search = _fit_fresh_base(base, batch, schema, context, None)
        context.record_base(base, estimator, batch, diagnostic_holdout, None)
        pipelines.append(estimator)
        if search is not None:
            searches.append(search)
    parameters = tuple(effective_parameters(base, pipeline.named_steps["estimator"])
                       for base, pipeline in zip(config.models, pipelines, strict=True))
    stack = FittedStack(schema, tuple(oof.columns), tuple(pipelines), parameters, weights,
                        config.meta.constraint_tolerance)
    weight_records = [dict(context.labels(), model=name, weight=float(weight))
                      for name, weight in zip(stack.base_names, weights, strict=True)]
    return FittedResult(stack, oof, searches, weight_records, context.diagnostics)


def _outer_predictions(stack: FittedStack, batch: TrainingBatch, repeat: int, fold: int) -> pd.DataFrame:
    """Reuse one fitted base matrix across every matched outer comparison."""
    try:
        matrix = predict_base_matrix(stack, batch.X)
        predicted = {f"base:{name}": matrix[name].to_numpy() for name in stack.base_names}
        predicted["equal_mean"] = matrix.to_numpy().mean(axis=1)
        predicted["stack"] = predict_meta(stack, matrix)
    except ValueError as error:
        raise ValueError(f"Outer predictions repeat={repeat} fold={fold}: {error}") from error
    reference = pd.DataFrame({"repeat_id": repeat, "outer_fold_id": fold, "row_id": batch.wafer_ids,
                              "group_id": batch.groups, "y_true": batch.y})
    columns = ["repeat_id", "outer_fold_id", "row_id", "group_id", "candidate_id", "y_true", "y_pred"]
    return pd.concat([reference.assign(candidate_id=name, y_pred=values)[columns]
                      for name, values in predicted.items()], ignore_index=True)


def evaluate_stack(batch: TrainingBatch, schema: FeatureSchema, config: TrainConfig,
                   plan: TrainingPlan, variant: str = "full") -> Evaluation:
    """Evaluate a feature variant on a previously validated grouped outer plan."""
    plan.validate(batch, config)
    projected = _with_features(batch, schema)
    predictions = []
    searches = []
    weights = []
    diagnostics = FitDiagnostics()
    interval = config.analysis.convergence_every if config.analysis.convergence and variant == "full" else None
    for repeat, partition in enumerate(plan.outer, start=1):
        partition.validate(projected)
        for fold, (training, holdout) in enumerate(partition.folds, start=1):
            context = _FitContext(plan.outer_stacks[repeat, fold], variant, plan.estimator_seeds, interval)
            validation = projected.take_positions(holdout)
            fitted = _fit_stack(projected.take_positions(training), schema, config, context, validation)
            predictions.append(_outer_predictions(fitted.stack, validation, repeat, fold))
            searches.extend(fitted.search_trials)
            weights.extend(fitted.weights)
            diagnostics.extend(fitted.convergence)
    return Evaluation(pd.concat(predictions, ignore_index=True), searches, weights, diagnostics)


def fit_deployment(batch: TrainingBatch, schema: FeatureSchema, config: TrainConfig,
                   plan: TrainingPlan) -> FittedResult:
    """Build a fresh final OOF matrix and refit all bases on all labeled wafers."""
    plan.validate(batch, config)
    interval = config.analysis.convergence_every if config.analysis.convergence else None
    context = _FitContext(plan.final_stack, "full", plan.estimator_seeds, interval)
    return _fit_stack(_with_features(batch, schema), schema, config, context)


__all__ = ["Evaluation", "FittedResult", "evaluate_stack", "fit_deployment", "preflight_training", "split_records"]
