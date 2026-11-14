"""Project search domains and enumerate structurally legal process choices."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from random import Random
from typing import Literal

from src.config.pipeline_common import ParameterUnits
from src.config.pseudo_data_schema import PseudoDataConfig
from src.dataset.contracts import DepositionStep, ProcessSequence, TreatmentStep
from src.optimize.settings import (
    DepositionRanges,
    NumericRange,
    SearchConfig,
    SearchDomain,
    SlotPin,
    TreatmentRanges,
)


def _synthetic_range(lower: int, upper: int) -> NumericRange:
    """Use the generator's inclusive integer lattice and engineering span."""
    return NumericRange(lower=lower, upper=upper, spacing=1, support_scale=upper - lower or None)


def synthetic_domain(config: PseudoDataConfig, parameter_units: ParameterUnits) -> SearchDomain:
    """Project the permitted settings independently of synthetic stress generation."""
    deposition, treatment = config.deposition, config.treatment
    return SearchDomain(
        source="equipment",
        parameter_units=parameter_units,
        deposition_catalog=deposition.materials,
        treatment_catalog=sorted(recipe for recipes in treatment.recipes.values() for recipe in recipes),
        deposition_ranges=DepositionRanges(
            time=_synthetic_range(deposition.time_min, deposition.time_max),
            temperature=_synthetic_range(deposition.temperature_min, deposition.temperature_max),
            thickness=_synthetic_range(deposition.thickness_min, deposition.thickness_max),
        ),
        treatment_ranges=TreatmentRanges(
            time=_synthetic_range(treatment.time_min, treatment.time_max),
            temperature=_synthetic_range(treatment.temperature_min, treatment.temperature_max),
        ),
        vacuum_breaks=sorted(config.transition.vacuum_break_values),
        min_active_count=config.total_steps_min - 1,
        max_active_count=config.total_steps_max - 1,
        min_depositions=config.min_depositions,
        max_treatments=config.max_treatments,
        allow_treatment_first=True,
    )


def numeric_ranges(domain: SearchDomain, operation: str) -> Mapping[str, NumericRange]:
    """Return the domain's applicable fields in canonical feature order."""
    if operation == "deposition":
        ranges = domain.deposition_ranges
        return {"time": ranges.time, "temperature": ranges.temperature, "thickness": ranges.thickness}
    if operation == "treatment":
        ranges = domain.treatment_ranges
        return {"time": ranges.time, "temperature": ranges.temperature}
    raise ValueError(f"operation={operation!r}; expected deposition or treatment")


def _fraction(value: float) -> Fraction:
    """Use the canonical decimal representation for exact lattice arithmetic."""
    return Fraction(str(value))


@dataclass(frozen=True)
class NumericChoice:
    """A fixed, gridded, or continuous numeric parameter."""

    lower: Fraction
    upper: Fraction
    spacing: Fraction | None

    @classmethod
    def from_range(cls, bounds: NumericRange, pin: float | None) -> "NumericChoice | None":
        """Compile a parameter choice, returning None for an incompatible pin."""
        lower, upper = _fraction(bounds.lower), _fraction(bounds.upper)
        spacing = None if bounds.spacing is None else _fraction(bounds.spacing)
        if pin is None:
            return cls(lower, upper, spacing)
        fixed = _fraction(pin)
        if not lower <= fixed <= upper or spacing is not None and (fixed - lower) % spacing:
            return None
        return cls(fixed, fixed, None)

    @property
    def size(self) -> int | None:
        """Return exact lattice cardinality or None for a continuous interval."""
        if self.lower == self.upper:
            return 1
        if self.spacing is None:
            return None
        return (self.upper - self.lower) // self.spacing + 1

    def values(self) -> Iterator[float]:
        """Enumerate a finite range from low to high."""
        count = self.size
        if count is None:
            raise ValueError(f"continuous interval {self!r}; expected a finite lattice for enumeration")
        return (self._grid_value(index) for index in range(count))

    def _grid_value(self, index: int) -> float:
        """Materialize one exact lattice point and reject an off-grid float payload."""
        exact = self.lower + index * (self.spacing or 0)
        value = float(exact)
        if _fraction(value) != exact:
            raise ValueError(f"grid index {index} in {self!r} converts to {value}; expected an exact lattice value")
        return value

    def sample(self, generator: Random) -> float:
        """Draw a numeric choice without integer size or float span overflow."""
        if self.lower == self.upper:
            return float(self.lower)
        if self.spacing is not None:
            count = (self.upper - self.lower) // self.spacing + 1
            return self._grid_value(generator.randrange(count))
        weight = generator.random()
        return (1 - weight) * float(self.lower) + weight * float(self.upper)


@dataclass(frozen=True)
class StepChoice:
    """A catalog entry with all compatible numeric choices for its slot."""

    operation: Literal["deposition", "treatment"]
    material: str
    recipe: str
    choices: tuple[NumericChoice, ...]

    def make_step(self, values: tuple[float, ...]) -> DepositionStep | TreatmentStep:
        """Construct a canonical step from chosen values in field order."""
        if self.operation == "deposition":
            return DepositionStep(self.material, self.recipe, values[0], values[1], values[2])
        return TreatmentStep(self.recipe, values[0], values[1])


@dataclass(frozen=True)
class Skeleton:
    """An admissible ordered process structure awaiting numeric values."""

    steps: tuple[StepChoice, ...]
    vacuum_breaks: tuple[int, ...]

    def count(self, limit: int) -> int | None:
        """Count numeric realizations, saturating at limit, or return None."""
        count = 1
        for step in self.steps:
            for choice in step.choices:
                cardinality = choice.size
                if cardinality is None:
                    return None
                count = min(limit, count * cardinality)
        return count

    def enumerate(self) -> Iterator[ProcessSequence]:
        """Yield every discretization in numeric order."""
        coordinate_sets = [tuple(choice.values()) for step in self.steps for choice in step.choices]
        for values in product(*coordinate_sets):
            offset = 0
            steps = []
            for step in self.steps:
                end = offset + len(step.choices)
                steps.append(step.make_step(values[offset:end]))
                offset = end
            yield ProcessSequence(tuple(steps), self.vacuum_breaks)

    def sample(self, generator: Random) -> ProcessSequence:
        """Draw a sequence with independently sampled compatible coordinates."""
        steps = tuple(step.make_step(tuple(choice.sample(generator) for choice in step.choices)) for step in self.steps)
        return ProcessSequence(steps, self.vacuum_breaks)


def _step_choice(entry: tuple[str, str, str], domain: SearchDomain, pin: SlotPin | None) -> StepChoice | None:
    """Compile one catalog entry if it satisfies the slot's operation and field pins."""
    operation, material, recipe = entry
    ranges = numeric_ranges(domain, operation)
    if pin is not None:
        fields = {"operation": operation, "material": material, "recipe": recipe}
        if any(getattr(pin, name) not in (None, value) for name, value in fields.items()):
            return None
        if pin.thickness is not None and "thickness" not in ranges:
            return None
    choices = []
    for field, bounds in ranges.items():
        choice = NumericChoice.from_range(bounds, None if pin is None else getattr(pin, field))
        if choice is None:
            return None
        choices.append(choice)
    return StepChoice(operation, material, recipe, tuple(choices))


def _catalog(domain: SearchDomain) -> tuple[tuple[str, str, str], ...]:
    """Return canonical (operation, material, recipe) triples."""
    depositions = [("deposition", material, recipe) for material, recipes in domain.deposition_catalog.items()
                   for recipe in recipes]
    treatments = [("treatment", "", recipe) for recipe in domain.treatment_catalog]
    return tuple(sorted(depositions + treatments))


def _pin_length(length: int, index: int, pin: SlotPin) -> bool:
    """Check structural presence and ensure any outgoing pin has a successor."""
    if pin.active is False:
        return index > length
    if pin.vacuum_break_after is not None:
        return index < length
    return index <= length


def _step_paths(
    slots: tuple[tuple[StepChoice, ...], ...], domain: SearchDomain, prefix: tuple[StepChoice, ...]
) -> Iterator[tuple[StepChoice, ...]]:
    """Walk eligible slots and prune branches violating operation limits."""
    if not slots:
        yield prefix
        return
    previous_treatments = sum(step.operation == "treatment" for step in prefix)
    for step in slots[0]:
        treatments = previous_treatments + (step.operation == "treatment")
        max_depositions = len(prefix) + len(slots) - treatments
        if treatments <= domain.max_treatments and max_depositions >= domain.min_depositions:
            yield from _step_paths(slots[1:], domain, (*prefix, step))


def _transition_options(
    domain: SearchDomain, config: SearchConfig, length: int
) -> tuple[tuple[int, ...], ...] | None:
    """Resolve possible internal breaks, or reject a conflicting pin."""
    allowed_bits = tuple(sorted(domain.vacuum_breaks))
    transitions = []
    for index in range(1, length):
        pin = config.pins.get(index)
        bit = None if pin is None else pin.vacuum_break_after
        if bit is not None and bit not in allowed_bits:
            return None
        transitions.append(allowed_bits if bit is None else (bit,))
    return tuple(transitions)


def _slot_options(domain: SearchDomain, config: SearchConfig, length: int) -> tuple[tuple[StepChoice, ...], ...]:
    """Compile per-slot catalog and numeric choices once for each length."""
    catalog = _catalog(domain)
    slots = []
    for index in range(1, length + 1):
        entries = [entry for entry in catalog if index > 1 or entry[0] != "treatment" or domain.allow_treatment_first]
        slots.append(tuple(choice for entry in entries
                           if (choice := _step_choice(entry, domain, config.pins.get(index))) is not None))
    return tuple(slots)


def _skeletons_for_length(domain: SearchDomain, config: SearchConfig, length: int) -> Iterator[Skeleton]:
    """Yield compatible skeletons of one length, including internal transition bits."""
    if any(not _pin_length(length, index, pin) for index, pin in config.pins.items()):
        return
    transitions = _transition_options(domain, config, length)
    if transitions is None:
        return
    slots = _slot_options(domain, config, length)
    if not all(slots):
        return
    for steps in _step_paths(slots, domain, ()):
        for bits in product(*transitions):
            yield Skeleton(steps, bits)


def build_skeletons(domain: SearchDomain, config: SearchConfig) -> dict[int, list[Skeleton]]:
    """Enumerate skeletons within budget or reject an infeasible domain before inference."""
    maximum = min(domain.max_active_count, config.N_step_search)
    lengths = range(domain.min_active_count, maximum + 1)
    if config.active_count is not None:
        lengths = range(config.active_count, config.active_count + 1) if config.active_count in lengths else range(0)
    by_length: dict[int, list[Skeleton]] = {}
    count = 0
    for length in lengths:
        for skeleton in _skeletons_for_length(domain, config, length):
            count += 1
            if count > config.max_skeletons:
                raise ValueError(f"at least {count} legal structures; expected at most {config.max_skeletons}")
            by_length.setdefault(length, []).append(skeleton)
    if not by_length:
        raise ValueError(f"search {config.mode} pins={config.model_dump()['pins']!r}, lengths={list(lengths)!r}; "
                         "expected at least one feasible structure")
    return by_length
