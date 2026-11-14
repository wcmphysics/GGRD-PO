"""Search legality, reproducibility, objective audit, and group support."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from src.analysis.support import assess_support
from src.analysis.synthetic_audit import audit_latent_results
from src.config.pipeline_common import ParameterUnits
from src.config.pipeline_schema import OptimizeConfig
from src.config.pseudo_data_schema import load_pseudo_data_config
from src.dataset.contracts import DepositionStep, ProcessSequence
from src.optimize.domain import NumericChoice, build_skeletons, synthetic_domain
from src.optimize.search import SearchContext, optimize_sequences
from src.optimize.settings import NumericRange, SearchDomain, SupportConfig


@pytest.fixture
def domain() -> SearchDomain:
    """Build a tiny lattice whose exact best solution is computed independently."""
    fixed = {"lower": 2, "upper": 2, "spacing": 1, "support_scale": None}
    varying = {"lower": 0.0, "upper": 0.3, "spacing": 0.1, "support_scale": 0.4}
    return SearchDomain.model_validate({
        "source": "equipment",
        "parameter_units": {"time": "s", "temperature": "C", "thickness": "nm"},
        "deposition_catalog": {"A": ["a"]},
        "treatment_catalog": ["b"],
        "deposition_ranges": {"time": fixed, "temperature": varying, "thickness": fixed},
        "treatment_ranges": {"time": fixed, "temperature": fixed},
        "vacuum_breaks": [0, 1],
        "min_active_count": 1,
        "max_active_count": 2,
        "min_depositions": 1,
        "max_treatments": 1,
        "allow_treatment_first": True,
    })


def _search_config(domain: SearchDomain, **overrides: Any) -> OptimizeConfig:
    """Create a test job with a budget large enough for exact scoring."""
    search = dict(mode="free", N_step_search=2, active_count=None, pins={}, seed=42, max_skeletons=100,
                  max_candidates=100, max_attempts=1, batch_size=3, top_k=100)
    search.update(overrides)
    return OptimizeConfig.model_validate({
        "bundle_directory": "artifact/model",
        "domain": domain,
        "search": search,
        "objective": {"target_stress": 0.0, "target_unit": "stress", "stress_scale": 2.0,
                      "stress_weight": 1.0, "cost_weight": 0.0, "penalty_weight": 0.0,
                      "cost_rule": "steps_fraction", "penalty_rule": "vacuum_break_fraction"},
        "support": {"radius": 0.25, "min_coverage": 0.75},
        "output": {"directory": "artifact/optimize"},
    })


def _predict(sequences: tuple[ProcessSequence, ...]) -> tuple[np.ndarray, np.ndarray]:
    """An independent objective oracle where break, order and numeric value all matter."""
    predictions = []
    for sequence in sequences:
        steps = sequence.steps
        if len(steps) == 2 and steps[0].operation == "treatment" and sequence.vacuum_breaks == (1,):
            predictions.append(steps[1].temperature - 0.3)
        else:
            predictions.append(1.0)
    values = np.asarray(predictions, dtype=float)
    return values, np.column_stack((values - 0.5, values + 0.5))


def _context() -> SearchContext:
    """Match the synthetic scorer's two ordered base names."""
    return SearchContext(2, "stress", ("base_a", "base_b"), "fixture-hash", {})


def test_exact_grid_oracle(domain: SearchDomain) -> None:
    """Enumerate 52 legal sequences independent of the sampling attempt limit."""
    result = optimize_sequences(domain, _search_config(domain), _predict, _context())
    summary = result.summary
    assert summary["stop_reason"] == "finite_space_exhausted"
    assert summary["scored"] == summary["attempted"] == len(result.trials) == 52
    assert summary["per_length"] == {"1": {"attempted": 4, "scored": 4}, "2": {"attempted": 48, "scored": 48}}
    winner = summary["ranked_results"][0]
    assert [step["operation"] for step in winner["sequence"]["steps"]] == ["treatment", "deposition"]
    assert winner["sequence"]["steps"][1]["temperature"] == 0.3
    assert winner["sequence"]["vacuum_breaks"] == [1]
    assert winner["objective"] == 0.0
    assert winner["base_range"] == 1.0
    assert result.trials[winner["trial_index"] - 1]["canonical_key"] == winner["canonical_key"]
    order = [(trial["objective"], trial["canonical_key"]) for trial in summary["ranked_results"]]
    assert order == sorted(order)
    best_scores = np.minimum.accumulate([trial["objective"] for trial in result.trials])
    assert [trial["best_so_far_objective"] for trial in result.trials] == best_scores.tolist()
    for trial in result.trials:
        assert trial["objective"] == pytest.approx(((trial["predicted_stress"] - trial["target_stress"]) / 2) ** 2)
    json.dumps(summary, allow_nan=False)


def test_fully_pinned_and_fixed_cost(domain: SearchDomain) -> None:
    """Score exactly once and calculate costs using uncapped domain length."""
    config = _search_config(domain, mode="fixed", N_step_search=1, active_count=1, pins={1: {"temperature": 0.2}})
    config.objective.cost_weight = 3.0
    result = optimize_sequences(domain, config, _predict, _context())
    assert result.summary["ranked_results"][0]["sequence"]["steps"][0]["temperature"] == 0.2
    assert result.summary["scored"] == result.summary["attempted"] == 1
    components = result.trials[0]["components"]
    domain_capacity = result.summary["domain"]["max_active_count"]
    assert components["process_cost"] == 1 / domain_capacity == 0.5
    assert result.trials[0]["objective"] == 1.75


@pytest.mark.parametrize("pins", [{1: {"vacuum_break_after": 0}}, {1: {"temperature": 0.31}},
                                  {1: {"temperature": 0.25}}, {1: {"operation": "treatment", "thickness": 2}},
                                  {1: {"active": False}}, {1: {"material": "unknown"}}])
def test_infeasible_pins_before_inference(domain: SearchDomain, pins: dict) -> None:
    """Reject a terminal break, out-of-range/grid pins and incompatible structural fields."""
    config = _search_config(domain, mode="fixed", active_count=1, pins=pins)
    with pytest.raises(ValueError, match="expected at least one feasible structure"):
        optimize_sequences(domain, config, lambda _: pytest.fail("inference ran on invalid domain"), _context())


def test_validate_invalid_config(domain: SearchDomain) -> None:
    """Enforce mode constraints, non-null pins, slot caps, and typed settings."""
    with pytest.raises(ValidationError, match="expected pins in fixed mode only"):
        _search_config(domain, active_count=1)
    with pytest.raises(ValidationError, match="expected 1..2"):
        _search_config(domain, mode="fixed", pins={3: {"active": False}})
    with pytest.raises(ValidationError, match="no nulls"):
        _search_config(domain, mode="fixed", pins={1: {"material": None}})
    with pytest.raises(ValidationError):
        _search_config(domain, mode="fixed", pins={1: {"vacuum_break_after": True}})


def test_grid_spacing_cannot_alias_adjacent_float_values() -> None:
    """Reject lattice points that can collapse after exact-to-float conversion."""
    bounds = {"lower": 10_000_000_000_000_000, "upper": 10_000_000_000_000_010, "support_scale": 10}
    with pytest.raises(ValidationError, match="spacing 1.1.*float resolution 2.0"):
        NumericRange.model_validate({**bounds, "spacing": 1.1})
    assert NumericRange.model_validate({**bounds, "spacing": 4.0}).spacing == 4.0


def test_grid_payload_cannot_leave_lattice() -> None:
    """Reject an emitted float whose decimal representation is off the lattice."""
    bounds = NumericRange(lower=0.1, upper=0.5, spacing=0.10000000000000002, support_scale=0.4)
    choice = NumericChoice.from_range(bounds, None)
    assert choice is not None
    with pytest.raises(ValueError, match="grid index 1.*expected an exact lattice value"):
        list(choice.values())


def test_caps_and_predictor_failure(domain: SearchDomain) -> None:
    """Check the feature capacity, structural cap, and finite inference diagnostics."""
    with pytest.raises(ValueError, match="N_step_search=3"):
        optimize_sequences(domain, _search_config(domain, N_step_search=3), _predict, _context())
    with pytest.raises(ValueError, match="legal structures"):
        optimize_sequences(domain, _search_config(domain, max_skeletons=1), _predict, _context())

    def invalid(sequences: tuple[ProcessSequence, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Produce one invalid prediction per trial."""
        return np.full(len(sequences), np.nan), np.zeros((len(sequences), 2))

    with pytest.raises(ValueError, match="trial 1.*non-finite"):
        optimize_sequences(domain, _search_config(domain), invalid, _context())


def test_sampling_length_and_attempt_budget(domain: SearchDomain) -> None:
    """Round-robin affordable draws and preserve deterministic order and accounting."""
    config = _search_config(domain, max_candidates=10, max_attempts=6)
    result = optimize_sequences(domain, config, _predict, _context())
    assert result == optimize_sequences(domain, config, _predict, _context())
    assert result.summary["stop_reason"] == "attempt_limit"
    assert result.summary["attempted"] == 6
    assert [entry["attempted"] for entry in result.summary["per_length"].values()] == [3, 3]
    score_count = result.summary["scored"]
    assert result.summary["candidate_budget_shortfall"] == 10 - score_count
    assert result.summary["duplicate_count"] == 6 - score_count

    limited = optimize_sequences(domain, _search_config(domain, max_candidates=2, max_attempts=10), _predict,
                                 _context())
    assert limited.summary["stop_reason"] == "budget_reached"
    assert [trial["active_count"] for trial in limited.trials] == [1, 2]


def test_support_counts_only_observed_group_values(domain: SearchDomain) -> None:
    """Cover all statuses and include a distance exactly on the radius boundary."""
    settings = SupportConfig(radius=0.25, min_coverage=0.75)
    candidate = ProcessSequence((DepositionStep("A", "a", 2, 0, 2),), ())
    missing = ProcessSequence((DepositionStep("A", "a", 2, None, 2),), ())
    observed = ProcessSequence((DepositionStep("A", "a", 2, 0.1, 2),), ())
    assert assess_support(candidate, {}, domain, settings)["status"] == "unseen_structure"
    unavailable = assess_support(candidate, {"G1": missing}, domain, settings)
    assert unavailable["status"] == "insufficient_numeric_overlap"
    assert unavailable["nearest_distance"] is None
    support = assess_support(candidate, {"G1": observed, "G2": observed, "G3": missing}, domain, settings)
    assert support == {"status": "comparable", "exact_structure_groups": 3, "comparable_groups": 2,
                       "nearest_distance": 0.25, "nearest_coverage": 1.0, "within_radius_groups": 2}


def test_synthetic_catalog_and_post_search_audit() -> None:
    """Match generator bounds and run the explicit post-search latent diagnostic."""
    generator = load_pseudo_data_config(Path("src/config/pseudo_data_config.yaml"))
    domain = synthetic_domain(generator, ParameterUnits(time="s", temperature="C", thickness="nm"))
    config = _search_config(domain, max_skeletons=20000, N_step_search=4)
    assert {size: len(skeletons) for size, skeletons in build_skeletons(domain, config.search).items()} == {
        2: 90, 3: 1340, 4: 17800,
    }
    fixed = _search_config(domain, mode="fixed", active_count=2, max_candidates=2, max_attempts=2,
                           pins={1: {"operation": "treatment"}, 2: {"operation": "deposition"}})
    result = optimize_sequences(domain, fixed, _predict, _context())
    audited = audit_latent_results(result.summary, generator)
    assert len(audited) == 2
    ranked_trials = [row["trial_index"] for row in result.summary["ranked_results"]]
    assert [entry["trial_index"] for entry in audited] == ranked_trials
    assert all(np.isfinite(entry["synthetic_latent_stress"]) for entry in audited)
