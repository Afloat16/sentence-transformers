from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from sentence_transformers.sparse_encoder.losses import (
    CachedSpladeLoss,
    FlopsLoss,
    SparseMultipleNegativesRankingLoss,
    SpladeLoss,
)
from sentence_transformers.sparse_encoder.modules import SpladePooling
from sentence_transformers.util import cos_sim


class _ReferenceFlops(nn.Module):
    def __init__(self, threshold=None):
        super().__init__()
        self.threshold = threshold

    def compute_loss_from_embeddings(self, embeddings):
        values = embeddings.to(torch.float64 if embeddings.dtype == torch.float64 else torch.float32)
        if self.threshold is not None:
            values = values * ((embeddings != 0).sum(dim=1) > self.threshold).unsqueeze(1)
        return values.mean(dim=0).square().sum()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("threshold", [None, 0, 2, 5])
def test_flops_matches_full_precision_value_and_gradient(dtype, threshold):
    embeddings = torch.tensor(
        [[300, 0, 0, 0], [300, 1.25, 0, 0], [300, 1.75, -2.5, 0], [0, 0, 0, 0]],
        dtype=dtype,
        requires_grad=True,
    )
    reference_embeddings = embeddings.detach().clone().requires_grad_()
    actual = FlopsLoss(nn.Identity(), threshold=threshold).compute_loss_from_embeddings(embeddings)
    expected = _ReferenceFlops(threshold).compute_loss_from_embeddings(reference_embeddings)

    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The threshold masks rows but keeps the full batch as the mean's denominator.
    (actual * 0.125).backward()
    (expected * 0.125).backward()
    torch.testing.assert_close(embeddings.grad, reference_embeddings.grad, rtol=0, atol=0)


@pytest.mark.parametrize("shape,value", [((2, 3), 300), ((2, 4096), 8)])
def test_flops_avoids_half_precision_square_and_sum_overflow(shape, value):
    embeddings = torch.full(shape, value, dtype=torch.float16, requires_grad=True)
    loss = FlopsLoss(nn.Identity()).compute_loss_from_embeddings(embeddings)
    expected = torch.tensor(shape[1] * value**2, dtype=torch.float32)

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    (loss * 3e-5).backward()
    assert torch.isfinite(embeddings.grad).all()
    assert torch.count_nonzero(embeddings.grad) == embeddings.numel()


class _LogitsModel(nn.Module):
    """Trainable logits with the real SPLADE pooling, without a downloaded encoder."""

    def __init__(self, dtype, pooling_strategy, activation_function):
        super().__init__()
        generator = torch.Generator().manual_seed(31)
        weights = torch.randn(256, 512, generator=generator) * 0.5 + 0.5
        self.logits = nn.Embedding.from_pretrained(weights.to(dtype), freeze=False)
        self.pooling = SpladePooling(pooling_strategy=pooling_strategy, activation_function=activation_function)

    def forward(self, features):
        return self.pooling({**features, "token_embeddings": self.logits(features["input_ids"])})


@pytest.mark.parametrize("loss_class", [SpladeLoss, CachedSpladeLoss])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pooling_strategy", ["max", "sum"])
@pytest.mark.parametrize("activation_function", ["relu", "log1p_relu"])
@pytest.mark.parametrize("autocast", [False, True])
def test_splade_low_precision_regularizers_match_reference(
    loss_class, dtype, pooling_strategy, activation_function, autocast
):
    model = _LogitsModel(dtype, pooling_strategy, activation_function).train()
    reference_model = deepcopy(model)
    generator = torch.Generator().manual_seed(32)
    features = [
        {
            "input_ids": torch.randint(256, (4, 64), generator=generator),
            "attention_mask": torch.arange(64).unsqueeze(0) < torch.tensor([64, 56, 48, 40]).unsqueeze(1),
        }
        for _ in range(2)
    ]

    def evaluate(current_model, reference=False):
        options = {"mini_batch_size": 2} if loss_class is CachedSpladeLoss else {}
        if reference:
            options.update(document_regularizer=_ReferenceFlops(), query_regularizer=_ReferenceFlops())
        loss = loss_class(
            current_model,
            # Cosine scoring keeps the base loss finite independently of the FLOPS reduction.
            loss=SparseMultipleNegativesRankingLoss(current_model, similarity_fct=cos_sim, scale=20),
            document_regularizer_weight=3e-5,
            query_regularizer_weight=5e-5,
            **options,
        )
        with torch.autocast("cpu", dtype=dtype, enabled=autocast):
            components = loss(features, None)
            total = sum(components.values())
        # The cached loss must also propagate an upstream scale through its replay hook.
        (total * 128).backward()
        return components, current_model.logits.weight.grad

    actual, actual_grad = evaluate(model)
    expected, expected_grad = evaluate(reference_model, reference=True)
    assert actual.keys() == expected.keys()
    for name in actual:
        assert torch.isfinite(actual[name]), name
        if name != "base_loss":
            assert actual[name].dtype == torch.float32
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)
    assert torch.isfinite(actual_grad).all()
    assert torch.count_nonzero(actual_grad) > 0
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
