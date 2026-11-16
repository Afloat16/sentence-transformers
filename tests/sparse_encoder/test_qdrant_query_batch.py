from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sentence_transformers.sparse_encoder.search_engines import semantic_search_qdrant

qdrant_client = pytest.importorskip("qdrant_client")
models = qdrant_client.models


@pytest.mark.parametrize("query_batch_size", [1, 2, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("kind", ["single", "batch", "empty"])
def test_qdrant_batch_matches_dense_scores_and_preserves_empty_queries(query_batch_size, dtype, kind):
    corpus = torch.tensor([[0, 0.1, 0, 0.3, 0.2, 0, 0.5], [0.8, 0.2, 0, 0, 0.5, 0.7, 0], [0, 0, 0.9, 0, 0, 0.6, 0.3]])
    queries = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0.2, 0.4, 0, 0.1, 0, 0.3, 0],
            [0, 0, 0.5, 0, 0, 0, 0.2],
            [0.4, 0.2, 0.1, 0, 0.3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=dtype,
    )
    if kind == "single":
        queries = queries[1]
    elif kind == "empty":
        queries = queries[:0]
    sparse = queries.to_sparse()
    # Coalescing duplicate input coordinates must happen before row boundaries are found.
    sparse = torch.sparse_coo_tensor(sparse.indices().repeat(1, 2), sparse.values().repeat(2) / 2, sparse.shape)
    client = qdrant_client.QdrantClient(":memory:")
    try:
        client.create_collection(
            "test", vectors_config={}, sparse_vectors_config={"text": models.SparseVectorParams()}
        )
        points = []
        for index, row in enumerate(corpus):
            selected = row.nonzero().flatten()
            points.append(
                models.PointStruct(
                    id=index,
                    vector={"text": models.SparseVector(indices=selected.tolist(), values=row[selected].tolist())},
                )
            )
        client.upsert("test", points, wait=True)
        with patch.object(client, "query_batch_points", wraps=client.query_batch_points) as batched:
            actual, elapsed, returned_index = semantic_search_qdrant(
                sparse, corpus_index=(client, "test"), top_k=2, output_index=True, query_batch_size=query_batch_size
            )
        query_rows = queries.unsqueeze(0) if queries.ndim == 1 else queries
        expected_scores = query_rows.float() @ corpus.T
        assert len(actual) == len(query_rows)
        for hits, scores in zip(actual, expected_scores):
            order = scores.argsort(descending=True)
            order = [int(index) for index in order if scores[index] > 0][:2]
            assert [hit["corpus_id"] for hit in hits] == order
            np.testing.assert_allclose([hit["score"] for hit in hits], scores[order].numpy(), rtol=2e-6, atol=1e-7)
        assert elapsed >= 0
        assert returned_index == (client, "test")
        assert batched.call_count == math.ceil(len(query_rows) / query_batch_size)
        assert all(len(call.kwargs["requests"]) <= query_batch_size for call in batched.call_args_list)
    finally:
        client.close()


@pytest.mark.parametrize("query_batch_size", [0, -1])
def test_qdrant_rejects_nonpositive_query_batch_size(query_batch_size):
    client = qdrant_client.QdrantClient(":memory:")
    try:
        with pytest.raises(ValueError, match="query_batch_size"):
            semantic_search_qdrant(
                torch.empty(0, 7).to_sparse(), corpus_index=(client, "unused"), query_batch_size=query_batch_size
            )
    finally:
        client.close()
