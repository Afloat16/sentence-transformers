from __future__ import annotations

import copy

import pytest
import torch

from sentence_transformers.sentence_transformer.losses import CosineSimilarityLoss
from sentence_transformers.sparse_encoder.losses import SparseCosineSimilarityLoss


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("loss_class", [CosineSimilarityLoss, SparseCosineSimilarityLoss])
@pytest.mark.parametrize("label_dtype", [torch.bool, torch.int64, torch.float32, torch.float64])
@pytest.mark.parametrize("column_labels", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cosine_similarity_loss_dtype_and_gradients(dtype, loss_class, label_dtype, column_labels, reduction) -> None:
    generator = torch.Generator().manual_seed(31)
    embeddings = [torch.randn(3, 8, dtype=dtype, generator=generator).requires_grad_() for _ in range(2)]
    reference_embeddings = [embedding.detach().clone().requires_grad_() for embedding in embeddings]
    labels = torch.tensor([0, 1, 0], dtype=label_dtype)
    if column_labels:
        labels = labels[:, None]
    loss_fn = loss_class(torch.nn.Identity(), loss_fct=torch.nn.MSELoss(reduction=reduction))

    actual = loss_fn.compute_loss_from_embeddings(embeddings, labels)
    reference_scores = torch.cosine_similarity(*reference_embeddings)
    if dtype in (torch.float16, torch.bfloat16):
        reference_scores = reference_scores.float()
    errors = (reference_scores - labels.flatten().to(reference_scores.dtype)).square()
    expected = errors.mean() if reduction == "mean" else errors.sum()

    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, embeddings)
    expected_grads = torch.autograd.grad(expected, reference_embeddings)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert actual_grad.dtype == dtype
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_cosine_similarity_loss_preserves_transform_dtype_and_gradient(dtype) -> None:
    class ScoreTransform(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.75, dtype=dtype))
            self.input_dtype = None

        def forward(self, scores):
            self.input_dtype = scores.dtype
            return (scores * self.scale).sigmoid()

    generator = torch.Generator().manual_seed(71)
    embeddings = [torch.randn(3, 8, dtype=dtype, generator=generator).requires_grad_() for _ in range(2)]
    reference_embeddings = [embedding.detach().clone().requires_grad_() for embedding in embeddings]
    labels = torch.tensor([0.1, 0.6, 0.9], dtype=torch.float64)
    transform = ScoreTransform()
    reference_transform = copy.deepcopy(transform)
    loss_fn = CosineSimilarityLoss(torch.nn.Identity(), cos_score_transformation=transform)

    actual = loss_fn.compute_loss_from_embeddings(embeddings, labels)
    scores = reference_transform(torch.cosine_similarity(*reference_embeddings))
    if dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    expected = (scores - labels.to(scores.dtype)).square().mean()

    assert transform.input_dtype == dtype
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, [*embeddings, transform.scale])
    expected_grads = torch.autograd.grad(expected, [*reference_embeddings, reference_transform.scale])
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("loss_class", [CosineSimilarityLoss, SparseCosineSimilarityLoss])
def test_cosine_similarity_loss_preserves_autocast_behavior(dtype, loss_class) -> None:
    generator = torch.Generator().manual_seed(53)
    embeddings = [torch.randn(3, 8, dtype=dtype, generator=generator).requires_grad_() for _ in range(2)]
    reference_embeddings = [embedding.detach().clone().requires_grad_() for embedding in embeddings]
    labels = torch.tensor([0.2, 0.7, 1.0])
    loss_fn = loss_class(torch.nn.Identity())

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = loss_fn.compute_loss_from_embeddings(embeddings, labels)
        expected = torch.nn.functional.mse_loss(torch.cosine_similarity(*reference_embeddings), labels)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual, embeddings)
    expected_grads = torch.autograd.grad(expected, reference_embeddings)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
