from __future__ import annotations

import pytest
import torch

from sentence_transformers.util import maxsim, maxsim_pairwise, mean_maxsim, mean_maxsim_pairwise


@pytest.mark.parametrize("score", [maxsim, maxsim_pairwise, mean_maxsim, mean_maxsim_pairwise])
@pytest.mark.parametrize("chunk_elements", [1, 10**9])
@pytest.mark.parametrize("masked_side", ["query", "document"])
@pytest.mark.parametrize("padding", [float("nan"), float("inf"), -float("inf")])
def test_masked_values_match_unpadded_scores_and_gradients(score, chunk_elements, masked_side, padding) -> None:
    queries = torch.tensor([[[1.0, -2.0], [-3.0, 1.0]]], requires_grad=True)
    documents = torch.tensor([[[-2.0, 1.0], [1.0, 3.0]]], requires_grad=True)
    expected = score(queries, documents)
    expected.sum().backward()

    padded_queries = queries.detach().clone()
    padded_documents = documents.detach().clone()
    kwargs = {"chunk_elements": chunk_elements}
    if masked_side == "query":
        padded_queries = torch.cat([padded_queries, torch.full((1, 1, 2), padding)], dim=1)
        kwargs["a_mask"] = torch.tensor([[1, 1, 0]])
    else:
        padded_documents = torch.cat([padded_documents, torch.full((1, 1, 2), padding)], dim=1)
        kwargs["b_mask"] = torch.tensor([[1, 1, 0]])
    padded_queries.requires_grad_()
    padded_documents.requires_grad_()
    original_queries = padded_queries.detach().clone()
    original_documents = padded_documents.detach().clone()

    actual = score(padded_queries, padded_documents, **kwargs)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(padded_queries.grad[:, :2], queries.grad)
    torch.testing.assert_close(padded_documents.grad[:, :2], documents.grad)
    padded_gradient = padded_queries.grad if masked_side == "query" else padded_documents.grad
    torch.testing.assert_close(padded_gradient[:, 2:], torch.zeros((1, 1, 2)))
    torch.testing.assert_close(padded_queries, original_queries, equal_nan=True)
    torch.testing.assert_close(padded_documents, original_documents, equal_nan=True)


@pytest.mark.parametrize("score", [maxsim, maxsim_pairwise, mean_maxsim, mean_maxsim_pairwise])
def test_masked_half_precision_query_does_not_overflow(score) -> None:
    queries = torch.tensor([[[1.0, 0.0], [512.0, 512.0]]], dtype=torch.float16, requires_grad=True)
    documents = torch.tensor([[[512.0, 512.0]]], dtype=torch.float16, requires_grad=True)
    expected = score(queries[:, :1], documents)
    actual = score(queries, documents, a_mask=torch.tensor([[1, 0]]))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(queries.grad).all()
    assert torch.isfinite(documents.grad).all()
    torch.testing.assert_close(queries.grad[:, 1:], torch.zeros_like(queries.grad[:, 1:]))
