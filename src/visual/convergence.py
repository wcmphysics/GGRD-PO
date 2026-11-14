"""Render training histories grouped by model, evaluation scope, and metric."""

import hashlib
from pathlib import Path
import re
from textwrap import fill
from types import ModuleType
from typing import TYPE_CHECKING

import pandas as pd

from src.visual.pipeline_plots import _plotting, _save

if TYPE_CHECKING:
    from matplotlib.axes import Axes


_PANELS = {
    ("outer", "inner_oof"): "Outer CV · inner OOF",
    ("outer", "base_refit"): "Outer CV · base refit",
    ("outer", "meta"): "Outer CV · meta fit",
    ("final", "inner_oof"): "Final fit · inner OOF",
    ("final", "base_refit"): "Final fit · full-data base",
    ("final", "meta"): "Final fit · full-data meta",
}
_FIT_IDENTIFIERS = ("repeat_id", "outer_fold_id", "inner_fold_id", "optimizer_restart")
_SPLIT_COLORS = {"train": "tab:blue", "validation": "tab:orange"}


def _in_panel(rows: pd.DataFrame, panel: tuple[str, str]) -> pd.DataFrame:
    """Select a stage and training scope."""
    stage, scope = panel
    return rows.loc[(rows.stage == stage) & (rows.scope == scope)]


def _split_curves(axis: "Axes", rows: pd.DataFrame, split: str) -> None:
    """Draw distinct fits and their stepwise median for one data split."""
    identity = [name for name in _FIT_IDENTIFIERS if name in rows.columns]
    color = _SPLIT_COLORS[split]
    for _, trace in rows.groupby(identity, sort=False, dropna=False):
        ordered = trace.sort_values("step")
        axis.plot(ordered.step, ordered.value, color=color, alpha=0.17, linewidth=0.8, marker=".", markersize=2)
    median = rows.groupby("step", sort=True).value.median()
    axis.plot(median.index, median, label=split.capitalize(), color=color, linewidth=1.7, marker=".", markersize=3)


def _draw_panel(axis: "Axes", rows: pd.DataFrame, title: str, metric: str, step_unit: str) -> None:
    """Keep raw metric values and label axes in the estimator's own units."""
    axis.set(title=title, xlabel=step_unit.replace("_", " ").capitalize(), ylabel=metric.replace("_", " "))
    if rows.empty:
        axis.text(0.5, 0.5, "No recorded history", transform=axis.transAxes, ha="center", va="center")
        axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        return
    for split, split_rows in rows.groupby("split", sort=True):
        _split_curves(axis, split_rows, split)
    axis.grid(alpha=0.2)


def _model_token(model: str) -> str:
    """Use portable figure names while preserving simple model names."""
    if re.fullmatch(r"[a-z][a-z0-9_]{0,59}", model) and model != "availability":
        return model
    token = re.sub(r"[^a-z0-9]+", "_", model.casefold()).strip("_")[:40]
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
    return f"{token or 'model'}-{digest}"


def _plot_model(
    identity: tuple[str, str], rows: pd.DataFrame, destination: Path, pyplot: ModuleType
) -> None:
    """Compare like metrics across panels while retaining raw fold traces."""
    from matplotlib.ticker import MaxNLocator

    model, estimator = identity
    specifications = list(rows.groupby(["variant", "metric", "step_unit"], sort=True))
    present = set(zip(rows.stage, rows.scope, strict=True))
    panels = [panel for panel in _PANELS if panel in present]
    figure, axes = pyplot.subplots(
        len(specifications), len(panels), figsize=(3.9 * len(panels), 2.7 * len(specifications) + 1),
        squeeze=False, sharex="row", sharey="row", layout="constrained",
    )
    legend = {}
    # Keep one panel for each fit scope and collect at most one legend entry per split.
    for index, ((variant, metric, step_unit), subset) in enumerate(specifications):
        for column, panel in enumerate(panels):
            axis = axes[index, column]
            title = f"{_PANELS[panel]} · {variant}"
            _draw_panel(axis, _in_panel(subset, panel), title, metric, step_unit)
            handles, labels = axis.get_legend_handles_labels()
            legend.update(zip(labels, handles, strict=True))
        if subset.step.nunique() == 1:
            axes[index, 0].set_xticks([subset.step.iloc[0]])
        else:
            axes[index, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    figure.suptitle(f"{model} ({estimator})\nFaint: individual traces · Solid: median at each recorded step")
    figure.legend(legend.values(), legend.keys(), loc="outside lower center", ncols=len(legend))
    figure.savefig(destination, dpi=150)
    pyplot.close(figure)


def _status_text(statuses: pd.DataFrame) -> list[str]:
    """Summarize available curves and group their missing-history reasons."""
    lines: list[str] = []
    for (model, estimator), records in statuses.groupby(["model", "estimator"], sort=True):
        lines.append(f"{model} ({estimator})")
        for (stage, scope, variant), group in records.groupby(["stage", "scope", "variant"], sort=False):
            recorded = group.status.eq("recorded").sum()
            missing = group.loc[group.status == "unavailable", "reason"].value_counts(sort=False)
            line = f"{_PANELS[(stage, scope)]}, {variant}: {recorded}/{len(group)} fits with history"
            if not missing.empty:
                line += " · " + ", ".join(f"{reason} ({count})" for reason, count in missing.items())
            lines.extend(fill(line, width=105, initial_indent="  ", subsequent_indent="    ").splitlines())
    return lines


def _plot_availability(statuses: pd.DataFrame, figures: Path, pyplot: ModuleType) -> None:
    """Summarize fit coverage even for estimators without iterative histories."""
    lines = _status_text(statuses)
    figure, axis = pyplot.subplots(figsize=(11, max(2.2, 0.24 * len(lines) + 0.9)))
    axis.axis("off")
    axis.set_title("Convergence history availability by fit", loc="left")
    axis.text(0, 1, "\n".join(lines), va="top", transform=axis.transAxes, fontsize=9)
    _save(figure, pyplot, figures / "convergence_availability.png")


def plot_convergence(run: Path, headless: bool) -> None:
    """Render saved convergence traces and fit coverage in a training run."""
    analysis = run / "analysis"
    traces = pd.read_csv(analysis / "convergence.csv", keep_default_na=False)
    statuses = pd.read_csv(analysis / "convergence_fits.csv", keep_default_na=False)
    pyplot = _plotting(headless)
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    estimators_per_model = traces.groupby("model").estimator.nunique()
    for identity, rows in traces.groupby(["model", "estimator"], sort=True):
        model, estimator = identity
        filename_token = model if estimators_per_model[model] == 1 else f"{model}/{estimator}"
        _plot_model(identity, rows, figures / f"convergence_{_model_token(filename_token)}.png", pyplot)
    _plot_availability(statuses, figures, pyplot)
