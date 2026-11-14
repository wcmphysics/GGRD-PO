"""Compare a proposed process to observed histories of distinct training groups."""

from collections.abc import Mapping
from math import fsum, isfinite

from src.dataset.contracts import ProcessSequence
from src.optimize.domain import numeric_ranges
from src.optimize.settings import SearchDomain, SupportConfig


def structural_signature(sequence: ProcessSequence) -> tuple[tuple[tuple[str, str, str], ...], tuple[int, ...]]:
    """Represent structure using operations, catalog labels and internal transitions."""
    labels = tuple((step.operation, getattr(step, "material", ""), step.recipe) for step in sequence.steps)
    return labels, sequence.vacuum_breaks


def _comparable_distance(
    candidate: ProcessSequence, reference: ProcessSequence, domain: SearchDomain, min_coverage: float
) -> tuple[float, float] | None:
    """Return normalized absolute difference and coverage for one matched group."""
    count = 0
    differences: list[float] = []
    for step, historic in zip(candidate.steps, reference.steps, strict=True):
        for field, bounds in numeric_ranges(domain, step.operation).items():
            if bounds.support_scale is None:
                continue
            count += 1
            candidate_value, reference_value = getattr(step, field), getattr(historic, field)
            if candidate_value is not None and reference_value is not None:
                difference = abs(candidate_value - reference_value) / bounds.support_scale
                if not isfinite(difference):
                    raise ValueError(f"{step.operation}.{field} values {candidate_value}, {reference_value} with "
                                     f"scale={bounds.support_scale}; expected finite normalized distance")
                differences.append(difference)
    if not differences or len(differences) / count < min_coverage:
        return None
    return fsum(value / len(differences) for value in differences), len(differences) / count


def assess_support(
    candidate: ProcessSequence, references: Mapping[str, ProcessSequence], domain: SearchDomain, config: SupportConfig
) -> dict[str, str | int | float | None]:
    """Count matching groups and nearby observed values without imputation."""
    signature = structural_signature(candidate)
    matches = {group: seq for group, seq in references.items() if structural_signature(seq) == signature}
    eligible = []
    for group, sequence in matches.items():
        scored = _comparable_distance(candidate, sequence, domain, config.min_coverage)
        if scored is not None:
            distance, coverage = scored
            eligible.append((distance, -coverage, group))

    return _support_record(len(matches), eligible, config.radius)


def _support_record(
    matches: int, eligible: list[tuple[float, float, str]], radius: float
) -> dict[str, str | int | float | None]:
    """Create a consistent summary for comparable and unavailable distances."""
    nearest = min(eligible) if eligible else None
    status = "unseen_structure" if not matches else "insufficient_numeric_overlap"
    return {
        "status": "comparable" if nearest is not None else status,
        "exact_structure_groups": matches,
        "comparable_groups": len(eligible),
        "nearest_distance": None if nearest is None else nearest[0],
        "nearest_coverage": None if nearest is None else -nearest[1],
        "within_radius_groups": sum(distance <= radius for distance, _, _ in eligible),
    }
