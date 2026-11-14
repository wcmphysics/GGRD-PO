"""Check ordinary held-out R² before presentation formatting."""

import numpy as np
import pytest

from src.analysis.regression import held_out_r2


def test_negative_r_squared_is_preserved() -> None:
    """An error exceeding the mean-target baseline yields a negative coefficient."""
    assert held_out_r2(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-3.0)


@pytest.mark.parametrize("truth,predictions", [([1], [3]), ([1, 1], [1, 1]), ([1, 1], [2, 3])])
def test_r_squared_undefined(truth: list[int], predictions: list[int]) -> None:
    """A singleton or constant targets leave R² undefined regardless of prediction accuracy."""
    assert held_out_r2(np.asarray(truth), np.asarray(predictions)) is None
