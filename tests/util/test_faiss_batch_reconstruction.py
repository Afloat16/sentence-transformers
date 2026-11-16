from __future__ import annotations

import numpy as np
import pytest

from sentence_transformers.util.quantization import semantic_search_faiss

faiss = pytest.importorskip("faiss")


@pytest.mark.parametrize("index_kind", ["flat", "hnsw", "idmap"])
@pytest.mark.parametrize("top_k", [1, 5])
@pytest.mark.parametrize("query_count", [1, 3])
def test_uint8_rescoring_preserves_index_keys(index_kind, top_k, query_count):
    corpus = np.array([[3, 0], [0, 4], [2, 2]], dtype=np.uint8)
    queries = np.array([[1, 3], [5, 1], [2, 3]], dtype=np.float32)[:query_count]
    keys = np.arange(len(corpus), dtype=np.int64)
    if index_kind == "hnsw":
        index = faiss.IndexHNSWFlat(2, 16, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexFlatIP(2)
    if index_kind == "idmap":
        index = faiss.IndexIDMap2(index)
        keys = np.array([10, 50, 100], dtype=np.int64)
        index.add_with_ids(corpus, keys)
    else:
        index.add(corpus)

    results, _, returned_index = semantic_search_faiss(
        queries,
        corpus_index=index,
        corpus_precision="uint8",
        ranges=np.array([[0, 0], [255, 255]], dtype=np.float32),
        top_k=top_k,
        rescore=True,
        output_index=True,
    )

    expected_scores = queries @ corpus.astype(np.float32).T
    assert returned_index is index
    for scores, row in zip(expected_scores, results):
        order = np.argsort(-scores)[:top_k]
        assert row == [{"corpus_id": int(keys[i]), "score": float(scores[i])} for i in order]


def test_uint8_rescoring_empty_index():
    index = faiss.IndexFlatIP(2)
    results, _ = semantic_search_faiss(
        np.array([[1, 3], [5, 1]], dtype=np.float32),
        corpus_index=index,
        corpus_precision="uint8",
        ranges=np.array([[0, 0], [255, 255]], dtype=np.float32),
        top_k=5,
    )
    assert results == [[], []]
