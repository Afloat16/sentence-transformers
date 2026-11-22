from __future__ import annotations

import pytest
import torch

from sentence_transformers import SparseEncoder


@pytest.mark.parametrize("query_layout", ["dense", "coo"])
@pytest.mark.parametrize("document_layout", ["dense", "coo", "csr"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_intersection_matches_dense_product(query_layout, document_layout, dtype):
    query = torch.tensor([2.0, 0.0, -1.0, 4.0, 0.0], dtype=dtype, requires_grad=True)
    documents = torch.tensor(
        [[1.0, 3.0, 2.0, 0.0, 0.0], [-2.0, 0.0, -2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0, 0.0]],
        dtype=dtype,
        requires_grad=True,
    )
    query_input = query.to_sparse() if query_layout == "coo" else query
    document_input = documents
    if document_layout == "coo":
        document_input = documents.to_sparse()
    elif document_layout == "csr":
        document_input = documents.to_sparse_csr()

    actual = SparseEncoder.intersection(query_input, document_input)
    product = query * documents
    expected = torch.where(product > 0, product, 0)
    assert actual.is_sparse
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.to_dense(), expected)

    actual_grads = torch.autograd.grad(actual.to_dense().sum(), (query, documents), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (query, documents))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_batched_intersection_coalesces_duplicate_coordinates():
    query_values = torch.tensor([2.0, 3.0, -1.0], requires_grad=True)
    document_values = torch.tensor([4.0, 1.0, 2.0, 3.0], requires_grad=True)
    query = torch.sparse_coo_tensor([[2, 0, 2]], query_values, (5,))
    documents = torch.sparse_coo_tensor([[1, 0, 1, 0], [2, 0, 2, 4]], document_values, (3, 5))

    actual = SparseEncoder.intersection(query, documents).to_dense()
    product = query.to_dense() * documents.to_dense()
    expected = torch.where(product > 0, product, 0)
    torch.testing.assert_close(actual, expected)

    actual_grads = torch.autograd.grad(actual.sum(), (query_values, document_values), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (query_values, document_values))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("batch_size", [0, 3])
@pytest.mark.parametrize("empty_query", [False, True])
def test_batched_intersection_empty_inputs(batch_size, empty_query):
    query = torch.zeros(5) if empty_query else torch.ones(5)
    documents = torch.zeros(batch_size, 5)
    result = SparseEncoder.intersection(query.to_sparse(), documents.to_sparse())

    assert result.is_sparse
    assert result.shape == documents.shape
    assert result._nnz() == 0


def test_batched_intersection_keeps_large_vocabulary_sparse():
    vocabulary_size = 1_000_000_000
    query = torch.sparse_coo_tensor([[5, vocabulary_size - 1]], [2.0, 3.0], (vocabulary_size,))
    documents = torch.sparse_coo_tensor(
        [[0, 1, 1], [5, 8, vocabulary_size - 1]], [4.0, 5.0, 6.0], (2, vocabulary_size)
    )

    result = SparseEncoder.intersection(query, documents).coalesce()

    assert result.shape == (2, vocabulary_size)
    torch.testing.assert_close(result.indices(), torch.tensor([[0, 1], [5, vocabulary_size - 1]]))
    torch.testing.assert_close(result.values(), torch.tensor([8.0, 18.0]))


@pytest.mark.parametrize(
    "query_dtype,document_dtype", [(torch.float32, torch.float64), (torch.float64, torch.float32)]
)
def test_batched_intersection_promotes_mixed_dtypes(query_dtype, document_dtype):
    query = torch.tensor([1.0, 2.0, 0.0], dtype=query_dtype, requires_grad=True)
    documents = torch.tensor([[1.0, 3.0, 5.0], [2.0, 0.0, 4.0]], dtype=document_dtype, requires_grad=True)

    result = SparseEncoder.intersection(query.to_sparse(), documents.to_sparse())
    expected = query * documents
    assert result.dtype == torch.float64
    torch.testing.assert_close(result.to_dense(), expected)

    actual_grads = torch.autograd.grad(result.to_dense().sum(), (query, documents), retain_graph=True)
    expected_grads = torch.autograd.grad(torch.where(expected > 0, expected, 0).sum(), (query, documents))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_batched_intersection_empty_query_keeps_backward_connection():
    query_values = torch.empty(0, requires_grad=True)
    document_values = torch.tensor([3.0, 4.0], requires_grad=True)
    query = torch.sparse_coo_tensor(torch.empty((1, 0), dtype=torch.long), query_values, (3,))
    documents = torch.sparse_coo_tensor([[0, 1], [0, 1]], document_values, (2, 3))

    SparseEncoder.intersection(query, documents).to_dense().sum().backward()

    assert query_values.grad is not None
    assert query_values.grad.shape == (0,)
    torch.testing.assert_close(document_values.grad, torch.zeros_like(document_values))


@pytest.mark.parametrize("document_shape", [(6,), (2, 6)])
def test_intersection_rejects_mismatched_vocabulary(document_shape):
    with pytest.raises(ValueError, match="Vocab dimension mismatch"):
        SparseEncoder.intersection(torch.ones(5).to_sparse(), torch.ones(document_shape).to_sparse())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("empty_query", [False, True])
def test_batched_intersection_rejects_different_devices(empty_query):
    query = torch.zeros(5) if empty_query else torch.ones(5)
    documents = torch.ones(2, 5, device="cuda")
    with pytest.raises(RuntimeError, match="device"):
        SparseEncoder.intersection(query.to_sparse(), documents.to_sparse())
