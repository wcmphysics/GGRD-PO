"""Check the two held-out scales and trace rendering from persisted run data."""

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pandas as pd
from pytest import MonkeyPatch

from src.visual import pipeline_plots
from src.visual.convergence import plot_convergence

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def test_parity_draws_full_and_central_ranges(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Retain the outlier in the full view and use it for the shared score."""
    frame = pd.DataFrame({"repeat_id": 1, "candidate_id": "stack", "y_true": [0, 1, 2, 3, 4, 99],
                          "y_pred": [0.5, 1.5, 2.5, 3.5, 4.5, 4.0]})
    frame.to_csv(tmp_path / "outer_predictions.csv", index=False)
    pyplot = pipeline_plots._plotting(True)

    def inspect(figure: "Figure", backend: ModuleType, destination: Path) -> None:
        """Check the figure before rendering and releasing it."""
        full, center = figure.axes
        assert full.get_xlim() == full.get_ylim()
        assert center.get_xlim() == center.get_ylim()
        assert full.get_xlim()[1] > frame.y_true.max() > center.get_xlim()[1]
        assert len(full.collections[0].get_offsets()) == len(frame)
        assert "5 of 6 wafers" in center.get_title()
        assert "R² = -0.150 (all wafers)" in figure.get_suptitle()
        backend.close(figure)

    monkeypatch.setattr(pipeline_plots, "_save", inspect)
    pipeline_plots._plot_parity(tmp_path, tmp_path, pyplot)


def test_convergence_renders_scope_coverage(tmp_path: Path) -> None:
    """Render blank fold IDs, negative objectives, separate units, and unavailable models."""
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    inner = dict(variant="full", stage="outer", repeat_id=1, outer_fold_id=2, inner_fold_id=3,
                 model="forest", estimator="random_forest", scope="inner_oof")
    refit = {**inner, "inner_fold_id": "", "scope": "base_refit"}
    final = {**refit, "stage": "final", "repeat_id": "", "outer_fold_id": "", "scope": "meta",
             "model": "stack", "estimator": "simplex"}
    traces = [dict(inner, split="validation", step=1, step_unit="tree", metric="mse", value=0.5),
              dict(refit, split="train", step=1, step_unit="tree", metric="mse", value=0.4),
              dict(final, split="train", step=0, step_unit="iteration", metric="simplex_objective", value=-0.1)]
    unavailable = {**inner, "model": "ridge", "estimator": "ridge", "status": "unavailable",
                   "reason": "solver has no history"}
    statuses = [dict(fit, status="recorded", reason="") for fit in (inner, refit, final)] + [unavailable]
    pd.DataFrame(traces).to_csv(analysis / "convergence.csv", index=False)
    pd.DataFrame(statuses).to_csv(analysis / "convergence_fits.csv", index=False)

    plot_convergence(tmp_path, headless=True)

    assert {path.name for path in (tmp_path / "figures").glob("*.png")} == {
        "convergence_forest.png", "convergence_stack.png", "convergence_availability.png",
    }
