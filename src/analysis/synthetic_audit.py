"""Evaluate search results against the synthetic generator after optimization."""

from collections.abc import Mapping
from typing import Any

from src.config.pseudo_data_schema import PseudoDataConfig
from src.utility.pseudo_data import _compute_base_stress


def _recipe_types(config: PseudoDataConfig) -> dict[str, str]:
    """Reverse the generator's treatment catalog and reject ambiguous recipes."""
    mapping = {}
    for treatment, recipes in config.treatment.recipes.items():
        for recipe in recipes:
            if recipe in mapping and mapping[recipe] != treatment:
                raise ValueError(f"treatment recipe {recipe!r} belongs to both {mapping[recipe]!r} and {treatment!r}; "
                                 "synthetic audit requires a unique type per recipe")
            mapping[recipe] = treatment
    return mapping


def audit_latent_results(summary: Mapping[str, Any], config: PseudoDataConfig) -> list[dict[str, float | int]]:
    """Compute generator latent stress and target gap for ranked trials."""
    recipe_types = _recipe_types(config)
    audit = []
    for trial in summary["ranked_results"]:
        steps = [dict(step) for step in trial["sequence"]["steps"]]
        for step in steps:
            if step["operation"] == "treatment":
                recipe = step["recipe"]
                if recipe not in recipe_types:
                    raise ValueError(f"trial {trial['trial_index']} recipe={recipe!r}; expected a synthetic recipe")
                step["treatment_type"] = recipe_types[recipe]
        latent = _compute_base_stress(steps, config)
        audit.append({"trial_index": trial["trial_index"], "synthetic_latent_stress": latent,
                      "latent_target_gap": abs(latent - trial["target_stress"])})
    return audit
