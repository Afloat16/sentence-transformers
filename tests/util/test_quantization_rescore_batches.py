from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sentence_transformers.util.quantization import (
    quantize_embeddings,
    semantic_search_faiss,
    semantic_search_usearch,
)

BACKENDS = [
    ("faiss", semantic_search_faiss, "uint8"),
    ("faiss", semantic_search_faiss, "ubinary"),
    ("usearch", semantic_search_usearch, "int8"),
    ("usearch", semantic_search_usearch, "ubinary"),
    ("usearch", semantic_search_usearch, "binary"),
]


@pytest.mark.parametrize(("backend", "search", "precision"), BACKENDS)
@pytest.mark.parametrize("batch_size", [1, 3, 100])
@pytest.mark.parametrize("n_docs", [3, 17])
@pytest.mark.parametrize("query_dtype", [np.float32, np.int32])
def test_rescore_batches_preserve_candidates_and_scores(backend, search, precision, batch_size, n_docs, query_dtype):
    pytest.importorskip(backend)
    rng = np.random.default_rng(42)
    corpus = rng.uniform(-10, 10, (n_docs, 10)).astype(np.float32)
    # Include a constant coordinate, unequal quantization scales, and binary byte padding.
    corpus[:, 0] = 2
    ranges = np.array([[-10] * 10, [10] * 10], dtype=np.float32)
    ranges[:, 0] = 2
    ranges[1, 1] = 100
    queries = rng.uniform(-5, 5, (7, 10)).astype(query_dtype)
    quantized = quantize_embeddings(corpus, precision, ranges=ranges)
    quantized_queries = quantize_embeddings(queries, precision, ranges=ranges)
    candidates, _, index = search(
        quantized_queries,
        corpus_embeddings=quantized,
        corpus_precision=precision,
        top_k=20,
        rescore=False,
        output_index=True,
    )
    if precision in ("uint8", "int8"):
        steps = (ranges[1] - ranges[0]) / 255
        steps = np.where(steps == 0, 1, steps)
        restored = quantized.astype(np.float64)
        if precision == "int8":
            restored += 128
        restored = restored * steps + ranges[0]
    else:
        unsigned = quantized if precision == "ubinary" else quantized.view(np.uint8) ^ 0x80
        restored = np.unpackbits(unsigned, axis=-1)[:, :10].astype(int)
    expected = []
    for query, hits in zip(queries, candidates):
        ids = np.array([hit["corpus_id"] for hit in hits])
        scores = np.einsum("j,kj->k", query, restored[ids]).astype(np.float64)
        order = np.argsort(-scores)[:5]
        expected.append([{"corpus_id": int(ids[i]), "score": float(scores[i])} for i in order])
    queries_before = queries.copy()
    corpus_before = quantized.copy()
    with patch.object(index, "search", wraps=index.search) as native_search:
        actual, _, returned_index = search(
            queries,
            corpus_index=index,
            corpus_precision=precision,
            ranges=ranges,
            top_k=5,
            rescore_multiplier=4,
            rescore_batch_size=batch_size,
            output_index=True,
        )
    assert native_search.call_count == 1
    assert returned_index is index
    assert actual == expected
    np.testing.assert_array_equal(queries, queries_before)
    np.testing.assert_array_equal(quantized, corpus_before)


@pytest.mark.parametrize(("backend", "search", "precision"), BACKENDS)
def test_default_rescore_batch_bounds_candidate_storage(backend, search, precision):
    pytest.importorskip(backend)
    rng = np.random.default_rng(43)
    queries = rng.standard_normal((203, 33), dtype=np.float32)
    corpus = rng.standard_normal((25, 33), dtype=np.float32)
    ranges = np.vstack((corpus.min(axis=0), corpus.max(axis=0)))
    shapes = []
    original_einsum = np.einsum

    def measured_einsum(subscripts, query_batch, candidates):
        shapes.append(candidates.shape)
        return original_einsum(subscripts, query_batch, candidates)

    with patch.object(np, "einsum", side_effect=measured_einsum):
        results, _ = search(
            queries,
            corpus_embeddings=quantize_embeddings(corpus, precision, ranges=ranges),
            corpus_precision=precision,
            ranges=ranges,
            top_k=5,
        )
    assert len(results) == len(queries)
    assert all(len(hits) == 5 for hits in results)
    assert sum(shape[0] for shape in shapes) == len(queries)
    assert max(shape[0] for shape in shapes) <= 100
    assert all(shape[1:] == (10, 33) for shape in shapes)


def test_usearch_batches_preserve_large_keys_and_empty_match_rows():
    pytest.importorskip("usearch")
    keys = np.array([5, 2**63, 2**64 - 2], dtype=np.uint64)
    queries = np.arange(1, 21, dtype=np.float32).reshape(5, 4)
    vectors = {int(key): np.array([i, i + 1, i + 2, i + 3], dtype=np.int8) for i, key in enumerate(keys)}
    match_keys = np.array([[0, 0], keys[:2], [0, 0], keys[1:], keys[:2]], dtype=np.uint64)
    counts = np.array([0, 2, 0, 1, 2])

    class Index:
        def search(self, embeddings, count, exact):
            return SimpleNamespace(keys=match_keys.copy(), distances=np.zeros((5, 2)), counts=counts)

        def get(self, query_keys):
            # A padding key has no vector, even if an entire batch has no real matches.
            return np.array([vectors[int(key)] for key in query_keys])

    ranges = np.array([[-128] * 4, [127] * 4], dtype=np.float32)
    expected, _ = semantic_search_usearch(
        queries, corpus_index=Index(), corpus_precision="int8", ranges=ranges, top_k=2, rescore_batch_size=5
    )
    for batch_size in (1, 2, 3):
        actual, _ = semantic_search_usearch(
            queries,
            corpus_index=Index(),
            corpus_precision="int8",
            ranges=ranges,
            top_k=2,
            rescore_batch_size=batch_size,
        )
        assert actual == expected
        assert actual[0] == actual[2] == []
        assert actual[3][0]["corpus_id"] == int(keys[1])
    assert actual[1][0]["corpus_id"] == int(keys[1])


@pytest.mark.parametrize(
    ("backend", "search"), [("faiss", semantic_search_faiss), ("usearch", semantic_search_usearch)]
)
@pytest.mark.parametrize("batch_size", [0, -1])
def test_rescore_batch_size_must_be_positive(backend, search, batch_size):
    pytest.importorskip(backend)
    with pytest.raises(ValueError, match="rescore_batch_size"):
        search(np.ones((2, 4)), corpus_embeddings=np.ones((3, 4)), rescore_batch_size=batch_size)
