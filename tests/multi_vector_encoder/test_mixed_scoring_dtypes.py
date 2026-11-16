from __future__ import annotations

import numpy as np
import pytest
import torch

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.scoring.xtr import xtr_kd_scores, xtr_scores, xtr_scores_pairwise
from sentence_transformers.sentence_transformer.modules import WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sentence_transformers.util.similarity import maxsim, maxsim_pairwise, mean_maxsim, mean_maxsim_pairwise


@pytest.mark.parametrize("similarity", ["maxsim", "meanmaxsim"])
def test_bfloat16_queries_with_numpy_documents(similarity):
    module = WordEmbeddings(WhitespaceTokenizer(vocab=["red", "blue", "green"]), torch.eye(3)).to(torch.bfloat16)
    model = MultiVectorEncoder(modules=[module], similarity_fn_name=similarity, device="cpu")
    queries = model.encode_query(["red blue", "green"], convert_to_numpy=False)
    documents = model.encode_document(["red green", "blue"], convert_to_numpy=True)
    assert queries[0].dtype == torch.bfloat16
    assert documents[0].dtype == np.float32

    expected = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    if similarity == "meanmaxsim":
        expected[0] /= 2
    torch.testing.assert_close(model.similarity(queries, documents), expected)
    torch.testing.assert_close(model.similarity_pairwise(queries, documents), expected.diag())


@pytest.mark.parametrize(
    "query_dtype,document_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float16, torch.bfloat16),
        (torch.float64, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("pairwise", [False, True])
@pytest.mark.parametrize("chunk_elements", [None, 1])
def test_maxsim_dtypes_preserve_masked_scores_and_gradients(query_dtype, document_dtype, pairwise, chunk_elements):
    queries = torch.tensor(
        [[[1, 0.25], [-0.5, 1], [10, 10]], [[-1, 0.5], [0, 0], [0, 0]]],
        dtype=query_dtype,
        requires_grad=True,
    )
    documents = torch.tensor(
        [[[0.5, 1], [-1, 0.25]], [[0.25, -1], [10, 10]]],
        dtype=document_dtype,
        requires_grad=True,
    )
    query_mask = torch.tensor([[True, True, False], [True, False, False]])
    document_mask = torch.tensor([[True, True], [True, False]])
    dtype = torch.promote_types(query_dtype, document_dtype)
    reference_queries = queries.detach().to(dtype).requires_grad_()
    reference_documents = documents.detach().to(dtype).requires_grad_()
    products = torch.einsum("ash,bth->abst", reference_queries, reference_documents)
    maxima = products.masked_fill(~document_mask[None, :, None, :], -torch.inf).max(-1).values.float()
    expected = (maxima * query_mask[:, None, :]).sum(-1)
    if pairwise:
        expected = expected.diag()
    expected.sum().backward()

    scorer = maxsim_pairwise if pairwise else maxsim
    scores = scorer(queries, documents, query_mask, document_mask, chunk_elements=chunk_elements)
    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores, expected)
    scores.sum().backward()
    torch.testing.assert_close(queries.grad, reference_queries.grad.to(query_dtype))
    torch.testing.assert_close(documents.grad, reference_documents.grad.to(document_dtype))

    mean_scorer = mean_maxsim_pairwise if pairwise else mean_maxsim
    counts = query_mask.sum(-1) if pairwise else query_mask.sum(-1).unsqueeze(-1)
    torch.testing.assert_close(
        mean_scorer(queries, documents, query_mask, document_mask, chunk_elements=chunk_elements), expected / counts
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
@pytest.mark.parametrize("scorer", [xtr_scores, xtr_kd_scores, xtr_scores_pairwise])
@pytest.mark.parametrize("chunk_elements", [None, 1])
def test_xtr_mixed_dtypes_match_explicit_promotion(dtype, scorer, chunk_elements):
    queries = torch.tensor([[[1, 0], [0, 1]], [[-1, 0], [0, -1]]], dtype=dtype, requires_grad=True)
    documents = torch.tensor([[[[0.75, 0.5], [-0.25, -1]]], [[[0.5, 0.25], [-0.5, -0.75]]]], requires_grad=True)
    promoted_dtype = torch.promote_types(dtype, documents.dtype)
    reference_queries = queries.detach().to(promoted_dtype).requires_grad_()
    reference_documents = documents.detach().to(promoted_dtype).requires_grad_()
    kwargs = {"top_k": 2, "chunk_elements": chunk_elements}
    expected = scorer(reference_queries, reference_documents, **kwargs)
    expected.sum().backward()

    scores = scorer(queries, documents, **kwargs)
    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores, expected)
    scores.sum().backward()
    torch.testing.assert_close(queries.grad, reference_queries.grad.to(dtype))
    torch.testing.assert_close(documents.grad, reference_documents.grad.to(documents.dtype))
