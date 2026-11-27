from __future__ import annotations

from math import log2

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from sentence_transformers.util.metrics import _binary_ndcg, _reciprocal_rank


@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize(
    ("labels", "scores", "k", "expected"),
    [
        pytest.param(
            [1] + [0] * 9, [1.0] * 10, 10, sum(1 / rank for rank in range(1, 11)) / 10, id="tie-ends-at-cutoff"
        ),
        pytest.param([1, 0, 0, 0, 0], [1.0] * 5, 1, 1 / 5, id="top-one"),
        pytest.param([0, 1, 0, 1], [1.0] * 4, 10, 13 / 18, id="interleaved-positives"),
        pytest.param([0, 1, 0, 1], [1.0] * 4, 1, 1 / 2, id="cutoff-inside-tie"),
        pytest.param([1, 1, 0], [1.0, 1.0, 0.0], 10, 1.0, id="only-positives-tied"),
        pytest.param([1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 2, 1, 1, 0], 2, 1 / 4, id="mixed-scores-cutoff"),
        pytest.param([1, 0, 0, 0], [1.0, 2.0, 2.0, 1.0], 2, 0.0, id="tie-below-cutoff"),
        pytest.param([0, 1, 0], [0.9, 0.8, 0.7], 10, 1 / 2, id="distinct-scores"),
        pytest.param([0, 1, 0], [0.9, 0.8, 0.7], 1, 0.0, id="distinct-scores-cutoff"),
        pytest.param([1, 1, 0], [0.0, 2.0, 1.0], 10, 1.0, id="best-positive-not-first"),
        pytest.param([0, 0], [1.0, 1.0], 10, 0.0, id="no-positives"),
        pytest.param([1, 1], [1.0, 1.0], 10, 1.0, id="all-positives"),
    ],
)
def test_reciprocal_rank(as_array, labels, scores, k, expected):
    if as_array:
        labels, scores = np.asarray(labels), np.asarray(scores)
    assert _reciprocal_rank(labels, scores, k=k) == pytest.approx(expected)


@pytest.mark.parametrize("k", [1, 2, 10])
@pytest.mark.parametrize(
    ("labels", "scores"),
    [([0, 1, 0], [0.9, 0.8, 0.7]), ([0, 1, 0, 1], [0.9, 0.8, 0.8, 0.7]), ([0, 1, 0, 1], [1.0] * 4)],
)
def test_binary_ndcg_matches_sklearn(labels, scores, k):
    expected = ndcg_score([labels], [scores], k=k)
    assert _binary_ndcg(np.asarray(labels), np.asarray(scores), k=k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("labels", "scores", "k", "num_positives", "expected"),
    [
        pytest.param([], [], 10, 2, 0.0, id="empty"),
        pytest.param([0, 0], [1.0, 0.0], 10, 2, 0.0, id="no-retrieved-positives"),
        pytest.param([1], [1.0], 10, None, 1.0, id="single-document"),
        pytest.param([1], [1.0], 10, 2, 1 / (1 + 1 / log2(3)), id="missing-positive"),
        pytest.param([1, 0], [0.0, 0.0], 1, 2, 0.5, id="missing-positive-with-ties"),
        pytest.param([1, 1], [1.0, 0.0], 10, 1, 1.0, id="underreported-positives"),
        pytest.param([1], [1.0], 0, 2, 0.0, id="zero-cutoff-single-document"),
        pytest.param([1, 0], [0.0, 0.0], 0, None, 0.0, id="zero-cutoff-ties"),
    ],
)
def test_binary_ndcg(labels, scores, k, num_positives, expected):
    assert _binary_ndcg(labels, scores, k=k, num_positives=num_positives) == pytest.approx(expected)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int64, np.bool_])
@pytest.mark.parametrize("num_positives", [128, 256, 300])
def test_binary_ndcg_counts_narrow_labels_without_overflow(dtype, num_positives):
    labels = np.array([1] * num_positives + [0] * 100, dtype=dtype)
    scores = np.arange(len(labels), dtype=float)
    expected = ndcg_score([labels], [scores], k=len(labels))
    assert _binary_ndcg(labels, scores, k=len(labels)) == pytest.approx(expected)


@pytest.mark.parametrize("evaluator_kind", ["cross_encoder", "reciprocal_rank_fusion"])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("num_retrieved", [128, 256, 300])
def test_evaluator_metrics_count_narrow_labels(evaluator_kind, dtype, num_retrieved):
    from sklearn.metrics import average_precision_score, dcg_score

    from sentence_transformers.cross_encoder.evaluation.reranking import CrossEncoderRerankingEvaluator
    from sentence_transformers.sparse_encoder.evaluation.reciprocal_rank_fusion import ReciprocalRankFusionEvaluator

    labels = np.array([1] * num_retrieved + [0] * 100, dtype=dtype)
    scores = np.arange(len(labels), dtype=float)
    num_positives = num_retrieved + 50
    if evaluator_kind == "cross_encoder":
        evaluator = CrossEncoderRerankingEvaluator(
            [{"query": "q", "positive": ["p"], "negative": ["n"]}], at_k=len(labels), write_csv=False
        )
    else:
        evaluator = ReciprocalRankFusionEvaluator([], [], at_k=len(labels), write_csv=False)
    mrr, ndcg, ap = evaluator.compute_metrics(labels, scores, num_positives=num_positives)
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, num_positives + 1))
    assert mrr == pytest.approx(1 / 101)
    assert ndcg == pytest.approx(dcg_score([labels], [scores]) / ideal_dcg)
    assert ap == pytest.approx(average_precision_score(labels, scores) * num_retrieved / num_positives)
