"""Fit-local preprocessing and numeric feature audits."""

from collections.abc import Mapping
import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature.wide import FeatureSchema

_LOGGER = logging.getLogger(__name__)


def make_preprocessor(schema: FeatureSchema, scale_numeric: bool) -> ColumnTransformer:
    """Construct fresh per-fit encoding with imputation restricted by the model pipeline."""
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), list(schema.numeric_columns)),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(schema.categorical_columns)),
        ("flags", "passthrough", list(schema.flag_columns)),
    ])


def warn_empty_columns(X: pd.DataFrame, schema: FeatureSchema, context: Mapping[str, object]) -> None:
    """Emit one warning listing training-only all-missing numeric features and their causes."""
    empty = X.loc[:, list(schema.numeric_columns)].isna().all()
    affected: dict[str, str] = {}
    for column in empty[empty].index:
        affected[column] = ("applicable_but_unrecorded" if X[f"{column}_applicable"].any() else "no_applicable_rows")
    if affected:
        fit = " ".join(f"{key}={value}" for key, value in context.items())
        _LOGGER.warning("ALL_MISSING_NUMERIC_FEATURES %s training_rows=%d columns=%s; retained columns fill missing "
                        "cells with zero before scaling; fit continues", fit, len(X), affected)
