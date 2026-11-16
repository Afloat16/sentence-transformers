from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import ContrastiveTensionLoss
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [0.1, 32.0])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("inferred_labels", [False, True])
def test_contrastive_tension_matches_float64_dot_product(
    dtype: torch.dtype, scale: float, autocast: bool, inferred_labels: bool
) -> None:
    torch.manual_seed(37)
    first = (torch.randn(4, 257) * scale).to(dtype).requires_grad_()
    second = (torch.randn(4, 257) * scale).to(dtype).requires_grad_()
    reference_first = first.detach().double().requires_grad_()
    reference_second = second.detach().double().requires_grad_()
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    features = [
        {"sentence_embedding": first, "input_ids": torch.tensor([[1], [2], [3], [4]])},
        {"sentence_embedding": second, "input_ids": torch.tensor([[1], [9], [3], [9]])},
    ]
    loss = ContrastiveTensionLoss(torch.nn.Identity())
    with torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext():
        actual = loss(features, None if inferred_labels else labels)
    expected = F.binary_cross_entropy_with_logits(
        (reference_first * reference_second).sum(dim=-1), labels.double(), reduction="sum"
    )
    assert torch.isfinite(actual)
    assert actual.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-6)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(first.grad, reference_first.grad.to(dtype))
    torch.testing.assert_close(second.grad, reference_second.grad.to(dtype))


@pytest.mark.parametrize("inferred_labels", [False, True])
def test_contrastive_tension_fp16_encoder_training_remains_finite(inferred_labels: bool) -> None:
    tokenizer = WhitespaceTokenizer(vocab=["[PAD]", "first", "second"], stop_words=[])
    words = WordEmbeddings(tokenizer, torch.full((3, 128), 32.0), update_embeddings=True)
    model = SentenceTransformer(modules=[words, Pooling(128, "mean")]).half()
    loss = ContrastiveTensionLoss(model)
    features = [model.preprocess([text] * 2) for text in ("first", "second")]
    actual = loss(features, None if inferred_labels else torch.zeros(2))
    # Two negative pairs, each with dot product 128 * 32**2. Their softplus
    # losses are representable in float32 even though the dot products overflow FP16.
    torch.testing.assert_close(actual, torch.tensor(2 * 128 * 32.0**2), rtol=0, atol=0)
    actual.backward()
    for encoder in (loss.model1, loss.model2):
        gradient = encoder[0].emb_layer.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
