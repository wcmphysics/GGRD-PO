"""Score the transition feature family on paired held-out wafer rows."""

from pathlib import Path

import pandas as pd


def write_transition_ablation(full: pd.DataFrame, ablated: pd.DataFrame, destination: Path, candidate: str) -> None:
    """Write the paired change in primary-model MAE after removing transition features."""
    key = ["repeat_id", "outer_fold_id", "row_id", "group_id", "y_true"]
    full_stack = full.loc[full.candidate_id == candidate, key + ["y_pred"]]
    ablated_stack = ablated.loc[ablated.candidate_id == candidate, key + ["y_pred"]]
    paired = full_stack.merge(ablated_stack, on=key, suffixes=("_full", "_ablated"), validate="one_to_one")
    if paired.empty or len(paired) != len(full_stack) or len(paired) != len(ablated_stack):
        raise ValueError("Transition ablation: expected identical paired outer rows and labels for both variants")
    errors = paired.assign(
        mae_full=(paired.y_pred_full - paired.y_true).abs(),
        mae_ablated=(paired.y_pred_ablated - paired.y_true).abs(),
    )
    repeats = errors.groupby("repeat_id", sort=True)[["mae_full", "mae_ablated"]].mean().reset_index()
    repeats.insert(1, "family", "internal_transition")
    repeats["delta"] = repeats.mae_ablated - repeats.mae_full
    summary = repeats.groupby("family").delta.agg(mean="mean", sample_sd="std", n_valid_repeats="count",
                                                 n_repeats="size").reset_index()
    destination.mkdir(exist_ok=True)
    repeats.to_csv(destination / "ablation_repeat_metrics.csv", index=False)
    summary.to_csv(destination / "ablation_summary.csv", index=False)
