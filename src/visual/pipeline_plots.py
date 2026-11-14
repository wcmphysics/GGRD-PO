"""Render stage plots from materialized run tables."""

import json
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.analysis.regression import held_out_r2

if TYPE_CHECKING:
    from matplotlib.axes import Axes
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


def _parity_limits(lower: float, upper: float) -> tuple[float, float]:
    """Give a constant-value plot usable bounds and pad nonconstant ranges."""
    margin = 0.05 * (upper - lower) if upper > lower else 0.05 * max(abs(lower), 1.0)
    return lower - margin, upper + margin


def _draw_parity(axis: "Axes", stack: pd.DataFrame, limits: tuple[float, float], title: str) -> None:
    """Draw all rows on shared measured and predicted scales."""
    axis.scatter(stack.y_true, stack.y_pred, alpha=0.65, s=17)
    axis.plot(limits, limits, linestyle="--", color="0.4")
    axis.set(xlabel="Measured stress", ylabel="Predicted stress", xlim=limits, ylim=limits, title=title)
    axis.set_aspect("equal", adjustable="box")


def _format_parity_score(stack: pd.DataFrame) -> str:
    """Report R² over all held-out pairs and mark undefined cases."""
    score = held_out_r2(stack.y_true.to_numpy(), stack.y_pred.to_numpy())
    return "undefined" if score is None else f"{score:.3f}"


def _plot_parity(run: Path, figures: Path, pyplot: ModuleType, candidate: str = "stack") -> None:
    """Show the full held-out range and a central view at equal scales."""
    predictions = pd.read_csv(run / "outer_predictions.csv")
    first_repeat = predictions.repeat_id.min()
    stack = predictions.loc[(predictions.repeat_id == first_repeat) & (predictions.candidate_id == candidate)]
    pooled = stack[["y_true", "y_pred"]].to_numpy().ravel()
    central_lower, central_upper = np.quantile(pooled, [0.05, 0.95])
    central = _parity_limits(central_lower, central_upper)
    inside = stack.y_true.between(*central) & stack.y_pred.between(*central)
    figure, axes = pyplot.subplots(1, 2, figsize=(9, 4.5))
    _draw_parity(axes[0], stack, _parity_limits(pooled.min(), pooled.max()), f"Full range ({len(stack)} wafers)")
    _draw_parity(axes[1], stack, central, f"Central view ({inside.sum()} of {len(stack)} wafers)")
    axes[1].text(0.03, 0.97, "Pooled 5th–95th percentiles, padded", transform=axes[1].transAxes, va="top")
    label = candidate.removeprefix("base:")
    score = _format_parity_score(stack)
    figure.suptitle(f"{label} · Held-out wafers, repeat {first_repeat} · R² = {score} (all wafers)")
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
    """Display the fitted deployment weights."""
    weights = pd.read_csv(run / "analysis" / "meta_weights.csv")
    final = weights.loc[weights.stage == "final"]
    figure, axis = pyplot.subplots()
    axis.bar(final.model, final.weight)
    axis.set(ylabel="Final meta weight", ylim=(0, 1))
    _save(figure, pyplot, figures / "meta_weights.png")


def _plot_ablation(run: Path, figures: Path, pyplot: ModuleType) -> None:
    """Plot the held-out change after removing transition features, when available."""
    ablation_path = run / "analysis" / "ablation_repeat_metrics.csv"
    if ablation_path.is_file():
        ablation = pd.read_csv(ablation_path)
        figure, axis = pyplot.subplots()
        axis.bar(ablation.repeat_id, ablation.delta)
        axis.set(xlabel="Outer repeat", ylabel="Ablated MAE minus full MAE")
        _save(figure, pyplot, figures / "transition_ablation.png")


def plot_training(run: Path, headless: bool, candidate: str | None) -> None:
    """Render the completed numeric outputs of the training stage."""
    pyplot = _plotting(headless)
    figures = run / "figures"
    figures.mkdir()
    _plot_exploration(run, figures, pyplot)
    if candidate is not None:
        _plot_parity(run, figures, pyplot, candidate)
    if (run / "paired_deltas.csv").is_file():
        _plot_pairs(run, figures, pyplot)
    if (run / "analysis" / "meta_weights.csv").is_file():
        _plot_weights(run, figures, pyplot)
    _plot_ablation(run, figures, pyplot)


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
