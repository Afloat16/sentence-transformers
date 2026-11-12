from __future__ import annotations

import pytest
import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.losses import FlopsLoss, SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.sparse_encoder.modules import SpladePooling


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("threshold", [None, 0])
@pytest.mark.parametrize(("dimensions", "value"), [(30000, 2.0), (3, 300.0)])
def test_flops_accumulates_before_squaring_and_summing(dtype, threshold, dimensions, value):
    embeddings = torch.full((2, dimensions), value, dtype=dtype, requires_grad=True)
    loss = FlopsLoss(torch.nn.Identity(), threshold=threshold).compute_loss_from_embeddings(embeddings)

    torch.testing.assert_close(loss.double(), torch.tensor(dimensions * value**2, dtype=torch.float64))
    loss.backward()
    torch.testing.assert_close(embeddings.grad, torch.full_like(embeddings, value))


def test_splade_weights_finite_flops_after_real_pooling():
    model = SparseEncoder(modules=[SpladePooling()], device="cpu")
    loss = SpladeLoss(
        model,
        SparseMultipleNegativesRankingLoss(model),
        document_regularizer_weight=3e-5,
        use_document_regularizer_only=True,
    )
    logits = [torch.full((2, 1, 30000), 6.0, dtype=torch.float16, requires_grad=True) for _ in range(2)]
    features = [
        {"token_embeddings": tensor, "attention_mask": torch.ones(2, 1, dtype=torch.long)} for tensor in logits
    ]
    output = loss(features)

    pooled_value = torch.tensor(6.0, dtype=torch.float16).log1p().item()
    expected = 30000 * pooled_value**2 * 3e-5
    assert output["document_regularizer_loss"].item() == pytest.approx(expected)
    assert all(torch.isfinite(value) for value in output.values())
    output["document_regularizer_loss"].backward()
    for tensor in logits:
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_flops_threshold_excludes_rows_without_changing_the_denominator():
    embeddings = torch.tensor([[300.0, 300.0], [300.0, 0.0]], dtype=torch.float16, requires_grad=True)
    loss = FlopsLoss(torch.nn.Identity(), threshold=1).compute_loss_from_embeddings(embeddings)
    torch.testing.assert_close(loss, torch.tensor(45000.0))
    loss.backward()
    torch.testing.assert_close(embeddings.grad, embeddings.new_tensor([[150.0, 150.0], [0.0, 0.0]]))
