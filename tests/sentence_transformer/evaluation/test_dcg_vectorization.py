from __future__ import annotations

import math

import numpy as np
import pytest

from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator


@pytest.mark.parametrize("length", [0, 3, 80, 1024])
@pytest.mark.parametrize("cutoff", [0, 1, 3, 10, 2048])
@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dcg_matches_independent_scalar_reference(length, cutoff, as_array, dtype):
    values = np.random.default_rng(19).uniform(0, 5, size=length).astype(dtype)
    values[::3] = 0
    inputs = values if as_array else values.tolist()
    expected = math.fsum(float(values[index]) / math.log2(index + 2) for index in range(min(length, cutoff)))
    actual = InformationRetrievalEvaluator.compute_dcg_at_k(inputs, cutoff)
    assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)
    np.testing.assert_array_equal(inputs, values)


@pytest.mark.parametrize("cutoff", [-3, 0, 1, 5])
def test_dcg_empty_and_zero_relevance(cutoff):
    assert InformationRetrievalEvaluator.compute_dcg_at_k([], cutoff) == 0
    assert InformationRetrievalEvaluator.compute_dcg_at_k([0, 0, 0], cutoff) == 0


def test_dcg_preserves_extended_precision_relevances():
    values = np.array([1, 2, 3], dtype=np.longdouble)
    expected = sum(value / np.log2(index + 2) for index, value in enumerate(values))
    actual = InformationRetrievalEvaluator.compute_dcg_at_k(values, 3)
    assert actual.dtype == values.dtype
    np.testing.assert_allclose(actual, expected, rtol=10 * np.finfo(np.longdouble).eps, atol=0)
