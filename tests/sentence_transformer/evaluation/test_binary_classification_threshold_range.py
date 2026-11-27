from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from sentence_transformers.sentence_transformer.evaluation.binary_classification import BinaryClassificationEvaluator


def assert_threshold_reproduces_score(scores, high_score_more_similar, metric):
    scores = np.sort(scores)
    if high_score_more_similar:
        scores = scores[::-1]
    labels = np.array([1, 0])
    with np.errstate(over="ignore", invalid="ignore"):
        if metric == "accuracy":
            score, threshold = BinaryClassificationEvaluator.find_best_acc_and_threshold(
                scores, labels, high_score_more_similar
            )
            measure = accuracy_score
        else:
            score, _, _, threshold = BinaryClassificationEvaluator.find_best_f1_and_threshold(
                scores, labels, high_score_more_similar
            )
            measure = f1_score

    predictions = scores >= threshold if high_score_more_similar else scores <= threshold
    assert score == pytest.approx(1.0)
    assert measure(labels, predictions) == pytest.approx(score)
    assert np.isfinite(threshold)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int16, np.int64])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("high_score_more_similar", [True, False])
@pytest.mark.parametrize("metric", ["accuracy", "f1"])
def test_threshold_midpoint_does_not_overflow(dtype, sign, high_score_more_similar, metric):
    info = np.finfo(dtype) if np.issubdtype(dtype, np.floating) else np.iinfo(dtype)
    scores = sign * np.array([float(info.max) * 0.75, float(info.max) * 0.625], dtype=dtype)
    assert_threshold_reproduces_score(scores, high_score_more_similar, metric)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("high_score_more_similar", [True, False])
@pytest.mark.parametrize("metric", ["accuracy", "f1"])
@pytest.mark.parametrize("region", ["adjacent", "subnormal"])
def test_promoted_midpoint_retains_attainable_thresholds(dtype, high_score_more_similar, metric, region):
    if region == "adjacent":
        first = dtype(1.0)
        second = np.nextafter(first, dtype(2.0))
    else:
        first = np.nextafter(dtype(0.0), dtype(1.0))
        second = dtype(first * 2)
    assert_threshold_reproduces_score(np.array([first, second]), high_score_more_similar, metric)
