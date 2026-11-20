import numpy as np
import pytest

from hst.models.baseline import cosine_score


def test_cosine_score():
    assert cosine_score(np.array([1.0, 0.0]), np.array([0.8, 0.6])) == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([1.0], [1.0, 2.0]),
        ([[1.0, 0.0]], [[1.0, 0.0]]),
        ([0.0, 0.0], [1.0, 0.0]),
    ],
)
def test_cosine_score_rejects_invalid_vectors(left, right):
    with pytest.raises(ValueError):
        cosine_score(np.asarray(left), np.asarray(right))
