"""Resolve process requests through the shared feature and prediction path."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.dataset.contracts import ProcessSequence
from src.feature.wide import build_features
from src.inference.bundle import DeploymentBundle
from src.model.stack import predict_base_matrix, predict_meta


@dataclass(frozen=True)
class PredictionBatch:
    """Original-scale predictions and their contributing base predictions."""

    values: NDArray[np.float64]
    base: pd.DataFrame


def predict_sequences(bundle: DeploymentBundle, sequences: Sequence[ProcessSequence]) -> PredictionBatch:
    """Widen canonical sequences and predict in the requested order."""
    features = build_features(sequences, bundle.schema)
    base = predict_base_matrix(bundle.stack, features)
    return PredictionBatch(predict_meta(bundle.stack, base), base)
