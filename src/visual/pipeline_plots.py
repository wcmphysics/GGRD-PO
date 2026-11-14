"""Render stage plots from materialized run tables."""

import json
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _plotting(headless: bool) -> ModuleType:
    """Select the headless backend before importing pyplot."""
    import matplotlib

    if headless:
        matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


def _save(figure: "Figure", pyplot: ModuleType, destination: Path) -> None:
    """Write a PNG and release the associated figure."""
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    pyplot.close(figure)


def _plot_exploration(run: Path, figures: Path, pyplot: ModuleType) -> None:
    """Plot target, replica and process-length distributions."""
    wafers = pd.read_csv(run / "exploration" / "wafers.csv")
    groups = pd.read_csv(run / "exploration" / "groups.csv")
    figure, axes = pyplot.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(wafers.target_stress, bins="auto")
    axes[0].set(xlabel="Measured stress", ylabel="Wafers")
    axes[1].bar(*zip(*groups.n_wafers.value_counts().sort_index().items()))
    axes[1].set(xlabel="Wafers per group", ylabel="Groups")
    axes[2].bar(*zip(*wafers.active_count.value_counts().sort_index().items()))
    axes[2].set(xlabel="Process steps", ylabel="Wafers")
    _save(figure, pyplot, figures / "exploration.png")


def _plot_parity(run: Path, figures: Path, pyplot: ModuleType) -> None:
    """Compare held-out measurements and predictions for the first repeat."""
    predictions = pd.read_csv(run / "outer_predictions.csv")
    first_repeat = predictions.repeat_id.min()
    stack = predictions.loc[(predictions.repeat_id == first_repeat) & (predictions.candidate_id == "stack")]
    lower = min(stack.y_true.min(), stack.y_pred.min())
    upper = max(stack.y_true.max(), stack.y_pred.max())
    figure, axis = pyplot.subplots()
    axis.scatter(stack.y_true, stack.y_pred, alpha=0.7)
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="0.4")
    axis.set(xlabel="Measured stress", ylabel="Predicted stress", title=f"Held-out wafers, repeat {first_repeat}")
    _save(figure, pyplot, figures / "held_out_parity.png")


def _plot_pairs(run: Path, figures: Path, pyplot: ModuleType) -> None:
    """Draw one paired MAE delta per repeat and baseline."""
    deltas = pd.read_csv(run / "paired_deltas.csv")
    figure, axis = pyplot.subplots()
    for baseline, rows in deltas.loc[deltas.metric == "mae"].groupby("baseline_id"):
        axis.plot(rows.repeat_id, rows.delta, marker="o", label=baseline)
    axis.axhline(0, linestyle="--", color="0.4")
    axis.set(xlabel="Outer repeat", ylabel="Baseline MAE minus stack MAE")
    axis.legend()
    _save(figure, pyplot, figures / "paired_mae.png")


def _plot_weights(run: Path, figures: Path, pyplot: ModuleType) -> None:
    """Display fitted deployment weights and optional held-out ablation."""
    weights = pd.read_csv(run / "analysis" / "meta_weights.csv")
    final = weights.loc[weights.stage == "final"]
    figure, axis = pyplot.subplots()
    axis.bar(final.model, final.weight)
    axis.set(ylabel="Final meta weight", ylim=(0, 1))
    _save(figure, pyplot, figures / "meta_weights.png")
    ablation_path = run / "analysis" / "ablation_repeat_metrics.csv"
    if ablation_path.is_file():
        ablation = pd.read_csv(ablation_path)
        figure, axis = pyplot.subplots()
        axis.bar(ablation.repeat_id, ablation.delta)
        axis.set(xlabel="Outer repeat", ylabel="Ablated MAE minus full MAE")
        _save(figure, pyplot, figures / "transition_ablation.png")


def plot_training(run: Path, headless: bool) -> None:
    """Render the completed numeric outputs of the training stage."""
    pyplot = _plotting(headless)
    figures = run / "figures"
    figures.mkdir()
    _plot_exploration(run, figures, pyplot)
    _plot_parity(run, figures, pyplot)
    _plot_pairs(run, figures, pyplot)
    _plot_weights(run, figures, pyplot)


def plot_optimization(run: Path, headless: bool) -> None:
    """Plot running best objective, target gap, and historical support."""
    pyplot = _plotting(headless)
    figures = run / "figures"
    figures.mkdir()
    trials = pd.read_json(run / "optimization_trials.jsonl", lines=True)
    summary = json.loads((run / "optimization.json").read_text(encoding="utf-8"))
    ranked = summary["ranked_results"]
    figure, axes = pyplot.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(trials.trial_index, trials.best_so_far_objective)
    axes[0].set(xlabel="Scored trial", ylabel="Best objective so far")
    ranks = [row["rank"] for row in ranked]
    axes[1].bar(ranks, [abs(row["predicted_stress"] - row["target_stress"]) for row in ranked])
    axes[1].set(xlabel="Rank", ylabel="Absolute target gap")
    axes[2].bar(ranks, [row["support"]["exact_structure_groups"] for row in ranked])
    axes[2].set(xlabel="Rank", ylabel="Matching historical groups")
    _save(figure, pyplot, figures / "optimization.png")
