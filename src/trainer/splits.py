"""Preflight deterministic grouped splits and preserve their local index scopes."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from src.config.base_estimator import required_training_rows
from src.config.pipeline_schema import TrainConfig
from src.dataset.contracts import TrainingBatch


Positions = NDArray[np.intp]


@dataclass(frozen=True)
class Partition:
    """One complete grouped partition on the given wafer order."""

    seed: int
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    folds: tuple[tuple[Positions, Positions], ...]

    def validate(self, batch: TrainingBatch) -> None:
        """Ensure indices still address the exact preflighted rows."""
        if tuple(batch.wafer_ids) != self.row_ids or tuple(batch.groups) != self.group_ids:
            raise ValueError("Split membership differs from preflight; expected identical wafer and group order.")


@dataclass(frozen=True)
class StackPlan:
    """Plans local to one outer or final training call."""

    stage: str
    repeat_id: int | None
    outer_fold_id: int | None
    inner: Partition | None
    searches: dict[tuple[int | None, str], Partition]


@dataclass(frozen=True)
class TrainingPlan:
    """Complete outer, inner, tuning, and final partition preflight."""

    outer: tuple[Partition, ...]
    outer_stacks: dict[tuple[int, int], StackPlan]
    final_stack: StackPlan
    estimator_seeds: dict[str, int]
    distinct_outer_partitions: int
    split_settings: tuple[object, ...]
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]

    def validate(self, batch: TrainingBatch, config: TrainConfig) -> None:
        """Check that plan reuse preserves row order and split settings."""
        self.validate_rows(batch)
        if self.split_settings != _split_settings(config):
            raise ValueError("Train settings differ from preflight; expected the original CV and model setup.")

    def validate_rows(self, batch: TrainingBatch) -> None:
        """Bind even a split-free configuration to the preflighted row order."""
        if tuple(batch.wafer_ids) != self.row_ids or tuple(batch.groups) != self.group_ids:
            raise ValueError("Training membership differs from preflight; expected identical wafer and group order.")


def _seed(root_seed: int, *coordinates: int) -> int:
    """Derive an independent, stable unsigned estimator seed."""
    return int(np.random.SeedSequence([root_seed, *coordinates]).generate_state(1)[0])


def _build_partition(batch: TrainingBatch, count: int, seed: int, context: str) -> Partition:
    """Create, verify, and lock local indices for one complete partition."""
    group_count = len(set(batch.groups))
    if group_count < count:
        raise ValueError(f"{context}: {group_count} groups in {len(batch.X)} rows; expected at least {count} groups.")
    splitter = GroupKFold(n_splits=count, shuffle=True, random_state=seed)
    membership = np.zeros(len(batch.X), dtype=int)
    folds = tuple(splitter.split(batch.X, groups=batch.groups))
    seen_groups: set[str] = set()
    for train, holdout in folds:
        train_groups, holdout_groups = set(batch.groups[train]), set(batch.groups[holdout])
        if train_groups & holdout_groups or seen_groups & holdout_groups or not len(train) or not len(holdout):
            raise ValueError(f"{context}: invalid fold; expected nonempty, disjoint training and holdout groups.")
        seen_groups.update(holdout_groups)
        membership[holdout] += 1
        train.flags.writeable = holdout.flags.writeable = False
    if not np.all(membership == 1) or seen_groups != set(batch.groups):
        raise ValueError(f"{context}: incomplete grouped partition; expected exactly one holdout per wafer and group.")
    return Partition(seed, tuple(batch.wafer_ids), tuple(batch.groups), folds)


def _build_stack_plan(batch: TrainingBatch, config: TrainConfig, repeat: int, outer_fold: int) -> StackPlan:
    """Materialize all inner and tuning folds within one stack's training rows."""
    stage = "outer" if repeat else "final"
    context = f"{stage} repeat={repeat or None} fold={outer_fold or None}"
    root_seed = config.cv.root_seed
    inner = None
    fit_positions = []
    if config.mode == "stacked":
        inner_seed = _seed(root_seed, 2, repeat, outer_fold)
        inner = _build_partition(batch, config.cv.inner_folds, inner_seed, f"{context} inner")
        fit_positions.extend((index, train) for index, (train, _) in enumerate(inner.folds, start=1))
    fit_positions.append((None, np.arange(len(batch.X), dtype=np.intp)))
    searches = {}
    active_names = {base.name for base in config.active_models}
    for inner_fold, positions in fit_positions:
        subset = batch.take_positions(positions)
        for base_position, base in enumerate(config.models):
            if base.name not in active_names:
                continue
            label = f"{context} inner={inner_fold} base={base.name}"
            minimum_rows = required_training_rows(base)
            _check_training_size(len(subset.X), minimum_rows, label)
            if base.enable_search:
                search_seed = _seed(root_seed, 3, repeat, outer_fold, inner_fold or 0, base_position)
                search = _build_partition(subset, config.cv.search_folds, search_seed, f"{label} search")
                for search_fold, (training, _) in enumerate(search.folds, start=1):
                    _check_training_size(len(training), minimum_rows, f"{label} search_fold={search_fold}")
                searches[inner_fold, base.name] = search
    return StackPlan(stage, repeat or None, outer_fold or None, inner, searches)


def _split_settings(config: TrainConfig) -> tuple[object, ...]:
    """Extract configuration fields that determine split meaning and estimator seeds."""
    cv = config.cv
    active_names = {base.name for base in config.active_models}
    return (config.mode, config.evaluate, cv.outer_repeats, cv.outer_folds, cv.inner_folds, cv.search_folds, cv.root_seed,
            tuple((position, base.name, base.estimator, base.enable_search, required_training_rows(base))
                  for position, base in enumerate(config.models) if base.name in active_names))


def _check_training_size(count: int, minimum: int | None, context: str) -> None:
    """Reject KNN settings that cannot fit and predict in the local fold."""
    if minimum is not None and count < minimum:
        raise ValueError(f"{context}: {count} training rows; n_neighbors requires at least {minimum} rows.")


def _partition_signature(partition: Partition) -> tuple[tuple[str, ...], ...]:
    """Compare group holdouts independently of fold and row order."""
    return tuple(sorted(tuple(sorted({partition.group_ids[index] for index in holdout}))
                        for _, holdout in partition.folds))


def _validate_training_batch(batch: TrainingBatch) -> None:
    """Reject malformed alignment before constructing any fit plans."""
    arrays = (batch.groups, batch.wafer_ids, batch.y, batch.row_positions)
    if not len(batch.X) or any(array.shape != (len(batch.X),) for array in arrays):
        raise ValueError(f"Training batch has {len(batch.X)} rows; expected nonempty aligned one-dimensional metadata.")
    for name, values in (("group", batch.groups), ("wafer ID", batch.wafer_ids)):
        if pd.isna(values).any() or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"Invalid {name} in training rows; expected nonempty text labels.")
    if len(set(batch.wafer_ids)) != len(batch.X) or len(set(batch.row_positions)) != len(batch.X):
        raise ValueError("Duplicate training identity; expected unique wafer IDs and original row positions.")
    if not np.isfinite(batch.y).all():
        raise ValueError("Nonfinite wafer stress encountered; expected a finite target for every training row.")


def preflight_training(batch: TrainingBatch, config: TrainConfig) -> TrainingPlan:
    """Validate all nested fit scopes before the first estimator is fitted."""
    _validate_training_batch(batch)
    outer = []
    stacks = {}
    repeat_count = config.cv.outer_repeats if config.evaluate else 0
    for repeat in range(1, repeat_count + 1):
        partition = _build_partition(batch, config.cv.outer_folds, _seed(config.cv.root_seed, 1, repeat),
                                     f"outer repeat={repeat}")
        outer.append(partition)
        for fold, (training, _) in enumerate(partition.folds, start=1):
            stacks[repeat, fold] = _build_stack_plan(batch.take_positions(training), config, repeat, fold)
    final_stack = _build_stack_plan(batch, config, 0, 0)
    active_names = {base.name for base in config.active_models}
    estimator_seeds = {base.name: _seed(config.cv.root_seed, 4, index)
                       for index, base in enumerate(config.models) if base.name in active_names}
    return TrainingPlan(tuple(outer), stacks, final_stack, estimator_seeds,
                        len({_partition_signature(partition) for partition in outer}), _split_settings(config),
                        tuple(batch.wafer_ids), tuple(batch.groups))


def _records(partition: Partition, context: dict) -> Iterator[dict]:
    """Emit auditable train and holdout membership for each local fold."""
    for fold_id, folds in enumerate(partition.folds, start=1):
        record = dict(context, fold_id=fold_id, seed=partition.seed)
        for side, positions in zip(("train", "holdout"), folds):
            members = [{"row_id": partition.row_ids[index], "group_id": partition.group_ids[index]}
                       for index in positions]
            record[side] = members
            record[f"n_{side}_rows"] = len(members)
            record[f"n_{side}_groups"] = len({item["group_id"] for item in members})
        yield record


def split_records(batch: TrainingBatch, plan: TrainingPlan) -> Iterator[dict]:
    """Export every preflighted outer, inner, final and tuning partition."""
    plan.validate_rows(batch)
    for repeat_id, outer in enumerate(plan.outer, start=1):
        yield from _records(outer, {"split_kind": "outer", "stage": "outer", "repeat_id": repeat_id})
    for stack in (*plan.outer_stacks.values(), plan.final_stack):
        context = {"stage": stack.stage, "repeat_id": stack.repeat_id, "outer_fold_id": stack.outer_fold_id}
        if stack.inner is not None:
            yield from _records(stack.inner, dict(context, split_kind="inner"))
        for (inner_fold, base_name), search in stack.searches.items():
            search_id = [stack.stage, stack.repeat_id, stack.outer_fold_id, inner_fold, base_name]
            yield from _records(search, dict(context, split_kind="search", search_id=search_id))
