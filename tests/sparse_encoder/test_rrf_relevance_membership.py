from __future__ import annotations

import json
from copy import deepcopy
from math import log2

import pytest

from sentence_transformers.sparse_encoder.evaluation import ReciprocalRankFusionEvaluator


@pytest.mark.parametrize("positive", [["p1"], ["p1", "p1", "unretrieved"], "p1", []])
def test_rrf_preserves_positive_counts_order_and_predictions(tmp_path, positive) -> None:
    sample = {"query_id": "q", "query": "query", "positive": positive, "documents": ["p1", "n1"]}
    original = deepcopy(sample)
    evaluator = ReciprocalRankFusionEvaluator([sample], [sample], write_csv=False, write_predictions=True)
    metrics = evaluator(output_path=str(tmp_path))
    count = len(positive) if isinstance(positive, list) else 1
    expected_map = 1 / count if count else 0
    expected_ndcg = 1 / sum(1 / log2(rank + 1) for rank in range(1, count + 1)) if count else 0
    for prefix in ("dense_", "sparse_", ""):
        assert metrics[prefix + "map"] == pytest.approx(expected_map)
        assert metrics[prefix + "ndcg@10"] == pytest.approx(expected_ndcg)
        assert metrics[prefix + "mrr@10"] == (1 if count else 0)
    prediction = json.loads((tmp_path / evaluator.predictions_file).read_text())
    assert prediction["positive"] == ([positive] if isinstance(positive, str) else positive)
    assert prediction["documents"] == sample["documents"]
    assert sample == original


def test_rrf_uses_updated_relevance_judgments_on_each_call() -> None:
    sample = {"query_id": "q", "query": "query", "positive": ["p1"], "documents": ["p1", "p2"]}
    evaluator = ReciprocalRankFusionEvaluator([sample], [sample], write_csv=False)
    first = evaluator()
    sample["positive"].append("unretrieved")
    second = evaluator()
    assert first["map"] == 1.0
    assert second["map"] == 0.5
