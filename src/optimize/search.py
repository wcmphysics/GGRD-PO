"""Score legal process sequences under a bounded deterministic search policy."""

from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import batched
import json
from math import isfinite
from random import Random
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.analysis.support import assess_support
from src.dataset.contracts import ProcessSequence, sequence_payload
from src.optimize.domain import Skeleton, build_skeletons
from src.optimize.settings import ObjectiveConfig, SearchConfig, SearchDomain, SupportConfig

if TYPE_CHECKING:
    from src.config.pipeline_schema import OptimizeConfig


Predictor = Callable[[tuple[ProcessSequence, ...]], tuple[NDArray[np.float64], NDArray[np.float64]]]


@dataclass(frozen=True)
class SearchContext:
    """Bundle facts and unfilled references needed for inference and diagnostics."""

    feature_capacity: int
    target_unit: str
    base_names: tuple[str, ...]
    bundle_fingerprint: str
    references: Mapping[str, ProcessSequence]


@dataclass(frozen=True)
class SearchResult:
    """JSON-ready output records in ranking and scoring order."""

    summary: dict[str, Any]
    trials: tuple[dict[str, Any], ...]


@dataclass
class _CandidatePool:
    """Unique queued sequences and draw-accounting counters."""

    lengths: tuple[int, ...]
    candidates: dict[str, ProcessSequence] = field(default_factory=dict)
    attempted: Counter[int] = field(default_factory=Counter)
    scored: Counter[int] = field(default_factory=Counter)

    def add(self, sequence: ProcessSequence) -> None:
        """Record a draw and queue it only when the canonical key is new."""
        length = len(sequence.steps)
        self.attempted[length] += 1
        key = _canonical_json(sequence_payload(sequence))
        if key not in self.candidates:
            self.candidates[key] = sequence
            self.scored[length] += 1


def _canonical_json(value: object) -> str:
    """Serialize a payload with stable keys and finite values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _length_sizes(by_length: dict[int, list[Skeleton]], limit: int) -> dict[int, int | None]:
    """Count finite lengths with saturation without materializing candidates."""
    sizes: dict[int, int | None] = {}
    for length, structures in by_length.items():
        counts = [structure.count(limit) for structure in structures]
        sizes[length] = None if None in counts else min(limit, sum(count for count in counts if count is not None))
    return sizes


class _SkeletonCycle:
    """Visit randomly shuffled skeletons once per cycle without replacement."""

    def __init__(self, skeletons: list[Skeleton], generator: Random) -> None:
        """Start with a generator and unshuffled copy of the input structures."""
        self.skeletons = skeletons.copy()
        self.generator = generator
        self.position = len(skeletons)

    def next(self) -> Skeleton:
        """Get the next structure, reshuffling when a cycle is exhausted."""
        if self.position == len(self.skeletons):
            self.generator.shuffle(self.skeletons)
            self.position = 0
        choice = self.skeletons[self.position]
        self.position += 1
        return choice


def _sample_pool(
    by_length: dict[int, list[Skeleton]], sizes: dict[int, int | None], config: SearchConfig
) -> tuple[_CandidatePool, str]:
    """Draw across feasible lengths in deterministic round-robin order."""
    pool = _CandidatePool(tuple(by_length))
    generator = Random(config.seed)
    remaining = deque(sorted(by_length))
    cycles = {length: _SkeletonCycle(skeletons, generator) for length, skeletons in by_length.items()}
    for _ in range(config.max_attempts):
        length = remaining.popleft()
        pool.add(cycles[length].next().sample(generator))
        if len(pool.candidates) == config.max_candidates:
            return pool, "budget_reached"
        if pool.scored[length] != sizes[length]:
            remaining.append(length)
        if not remaining:
            return pool, "finite_space_exhausted"
    return pool, "attempt_limit"


def _candidate_pool(domain: SearchDomain, config: SearchConfig) -> tuple[_CandidatePool, str]:
    """Select all candidates on small grids and bounded samples otherwise."""
    by_length = build_skeletons(domain, config)
    sizes = _length_sizes(by_length, config.max_candidates + 1)
    expected = sum(size for size in sizes.values() if size is not None)
    if None in sizes.values() or expected > config.max_candidates:
        return _sample_pool(by_length, sizes, config)
    pool = _CandidatePool(tuple(by_length))
    for skeletons in by_length.values():
        for skeleton in skeletons:
            for sequence in skeleton.enumerate():
                pool.add(sequence)
    if len(pool.candidates) != expected:
        raise ValueError(f"finite grid collapsed to {len(pool.candidates)} values; expected {expected} distinct values")
    return pool, "finite_space_exhausted"


def _objective_terms(
    sequence: ProcessSequence, prediction: float, domain: SearchDomain, settings: ObjectiveConfig
) -> dict[str, float]:
    """Compute raw and weighted components and reject non-finite results."""
    residual = (prediction - settings.target_stress) / settings.stress_scale
    terms = {"stress_loss": residual * residual, "process_cost": 0.0, "soft_penalty": 0.0}
    if settings.cost_rule == "steps_fraction":
        terms["process_cost"] = len(sequence.steps) / domain.max_active_count
    if settings.penalty_rule == "vacuum_break_fraction" and sequence.vacuum_breaks:
        terms["soft_penalty"] = sum(sequence.vacuum_breaks) / len(sequence.vacuum_breaks)
    terms["weighted_stress_loss"] = terms["stress_loss"] * settings.stress_weight
    terms["weighted_process_cost"] = terms["process_cost"] * settings.cost_weight
    terms["weighted_soft_penalty"] = terms["soft_penalty"] * settings.penalty_weight
    terms["objective"] = terms["weighted_stress_loss"] + terms["weighted_process_cost"] + terms["weighted_soft_penalty"]
    if any(not isfinite(value) for value in terms.values()):
        raise ValueError(f"objective for prediction={prediction}, settings={settings!r}; expected finite terms")
    return terms


def _predict_batch(
    batch: tuple[ProcessSequence, ...], predictor: Predictor, base_count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Check the finite, aligned fitted predictor contract on one batch."""
    predictions, base_predictions = predictor(batch)
    expected = (len(batch), base_count)
    if predictions.shape != (len(batch),) or base_predictions.shape != expected:
        raise ValueError(f"prediction shapes {predictions.shape}, {base_predictions.shape}; "
                         f"expected {(len(batch),)}, {expected}")
    if not (np.isfinite(predictions).all() and np.isfinite(base_predictions).all()):
        raise ValueError("prediction batch has non-finite entries; expected finite stack and base predictions")
    return predictions, base_predictions


def _score_pool(
    pool: _CandidatePool, domain: SearchDomain, config: "OptimizeConfig", predictor: Predictor, context: SearchContext
) -> tuple[dict[str, Any], ...]:
    """Evaluate unique candidates in order and derive the running best objective."""
    trials = []
    best = float("inf")
    # Preserve candidate order across calls and report the first affected trial on inference failure.
    for items in batched(pool.candidates.items(), config.search.batch_size):
        keys, sequences = zip(*items, strict=True)
        try:
            predictions, base = _predict_batch(sequences, predictor, len(context.base_names))
        except (RuntimeError, ValueError, TypeError) as error:
            raise ValueError(f"search inference starting at trial {len(trials) + 1} failed: {error}") from error
        for key, sequence, value, base_row in zip(keys, sequences, predictions, base, strict=True):
            index = len(trials) + 1
            try:
                terms = _objective_terms(sequence, float(value), domain, config.objective)
            except ValueError as error:
                raise ValueError(f"trial {index} candidate={key} failed: {error}") from error
            score = terms.pop("objective")
            best = min(best, score)
            trials.append({
                "trial_index": index,
                "canonical_key": key,
                "sequence": sequence_payload(sequence),
                "active_count": len(sequence.steps),
                "predicted_stress": float(value),
                "target_unit": context.target_unit,
                "target_stress": config.objective.target_stress,
                "base_predictions": base_row.tolist(),
                "components": terms,
                "objective": score,
                "best_so_far_objective": best,
            })
    return tuple(trials)


def _ranked_results(
    trials: tuple[dict[str, Any], ...], pool: _CandidatePool, domain: SearchDomain,
    settings: tuple[SearchConfig, SupportConfig], context: SearchContext
) -> list[dict[str, Any]]:
    """Annotate top trials with independent support and prediction disagreement."""
    search, support = settings
    ranked = sorted(trials, key=lambda trial: (trial["objective"], trial["canonical_key"]))
    results = []
    for rank, trial in enumerate(ranked[:search.top_k], start=1):
        sequence = pool.candidates[trial["canonical_key"]]
        base_min, base_max = min(trial["base_predictions"]), max(trial["base_predictions"])
        base_range = base_max - base_min
        if not isfinite(base_range):
            raise ValueError(f"trial {trial['trial_index']} base range overflowed; expected finite disagreement")
        results.append({
            **trial,
            "rank": rank,
            "support": assess_support(sequence, context.references, domain, support),
            "base_min": base_min,
            "base_max": base_max,
            "base_range": base_range,
        })
    return results


def _summarize(
    pool: _CandidatePool, domain: SearchDomain, config: "OptimizeConfig", context: SearchContext,
    stop_reason: str, ranked: list[dict[str, Any]]
) -> dict[str, Any]:
    """Create the auditable run manifest and ranked trial summary."""
    attempted, scored = pool.attempted.total(), pool.scored.total()
    effective_domain = domain.model_dump(mode="json")
    slots = {str(index): pin.model_dump(exclude_none=True) for index, pin in config.search.pins.items()}
    return {
        "mode": config.search.mode,
        "pins": {"active_count": config.search.active_count, "slots": slots},
        "domain": effective_domain,
        "domain_fingerprint": sha256(_canonical_json(effective_domain).encode("utf-8")).hexdigest(),
        "bundle_fingerprint": context.bundle_fingerprint,
        "base_names": list(context.base_names),
        "search": {**config.search.model_dump(mode="json"), "pins": slots},
        "objective": config.objective.model_dump(mode="json"),
        "support": {**config.support.model_dump(mode="json"), "parameter_ranges": {
            "deposition": effective_domain["deposition_ranges"], "treatment": effective_domain["treatment_ranges"]}},
        "attempted": attempted,
        "scored": scored,
        "per_length": {str(length): {"attempted": pool.attempted[length], "scored": pool.scored[length]}
                       for length in pool.lengths},
        "duplicate_count": attempted - scored,
        "stop_reason": stop_reason,
        "candidate_budget_shortfall": config.search.max_candidates - scored if stop_reason == "attempt_limit" else 0,
        "ranked_results": ranked,
    }


def optimize_sequences(
    domain: SearchDomain, config: "OptimizeConfig", predictor: Predictor, context: SearchContext
) -> SearchResult:
    """Search legal sequences and return normalized objectives and diagnostics."""
    if config.search.N_step_search > context.feature_capacity:
        raise ValueError(f"N_step_search={config.search.N_step_search}; expected <= {context.feature_capacity}")
    if config.objective.target_unit != context.target_unit:
        raise ValueError(f"search target unit {config.objective.target_unit!r}; expected {context.target_unit!r}")
    if not context.base_names:
        raise ValueError("bundle has no base names; expected at least one named base prediction")
    pool, stop_reason = _candidate_pool(domain, config.search)
    if not pool.candidates:
        raise ValueError("search scored zero candidates; expected a nonempty feasible candidate pool")
    trials = _score_pool(pool, domain, config, predictor, context)
    ranked = _ranked_results(trials, pool, domain, (config.search, config.support), context)
    summary = _summarize(pool, domain, config, context, stop_reason, ranked)
    return SearchResult(summary, trials)
