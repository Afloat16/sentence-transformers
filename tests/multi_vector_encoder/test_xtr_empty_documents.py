from __future__ import annotations

import pytest
import torch

from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling
from sentence_transformers.multi_vector_encoder.scoring import xtr_kd_scores, xtr_scores, xtr_scores_pairwise


@pytest.mark.parametrize("scorer", [xtr_scores, xtr_kd_scores, xtr_scores_pairwise])
@pytest.mark.parametrize("chunk_elements", [None, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_xtr_scores_documents_with_no_retained_tokens(scorer, chunk_elements, dtype):
    queries = torch.ones(2, 3, 4, dtype=dtype, requires_grad=True)
    documents = torch.ones(2, 5, 4, dtype=dtype, requires_grad=True)
    pooler = HierarchicalTokenPooling(pool_factor=2, num_protected_tokens=0)
    pooled = pooler({"token_embeddings": documents, "attention_mask": torch.zeros(2, 5, dtype=torch.bool)})
    assert pooled["token_embeddings"].shape == (2, 0, 4)

    compressed = pooled["token_embeddings"].unsqueeze(1)
    compressed_mask = pooled["attention_mask"].unsqueeze(1)
    padded = torch.zeros(2, 1, 1, 4, dtype=dtype)
    padded_mask = torch.zeros(2, 1, 1, dtype=torch.bool)
    actual = scorer(queries, compressed, documents_mask=compressed_mask, chunk_elements=chunk_elements)
    expected = scorer(queries.detach(), padded, documents_mask=padded_mask, chunk_elements=chunk_elements)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert (actual < -1e8).all()

    actual.sum().backward()
    torch.testing.assert_close(queries.grad, torch.zeros_like(queries))
    torch.testing.assert_close(documents.grad, torch.zeros_like(documents))


@pytest.mark.parametrize("chunk_elements", [None, 1])
def test_xtr_infers_empty_document_masks(chunk_elements):
    queries = torch.ones(2, 3, 4)
    documents = torch.empty(2, 1, 0, 4)
    actual = xtr_scores(queries, documents, chunk_elements=chunk_elements)
    expected = xtr_scores(queries, torch.zeros(2, 1, 1, 4), chunk_elements=chunk_elements)
    torch.testing.assert_close(actual, expected)
