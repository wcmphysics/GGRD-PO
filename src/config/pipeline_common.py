"""Shared validation primitives for the pipeline YAML schemas."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveNumber = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
NonnegativeNumber = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictConfig(BaseModel):
    """Forbid undeclared settings in every pipeline schema."""

    model_config = ConfigDict(extra="forbid")


class ParameterUnits(StrictConfig):
    """Canonical units for the three process coordinates."""

    time: NonemptyText
    temperature: NonemptyText
    thickness: NonemptyText
