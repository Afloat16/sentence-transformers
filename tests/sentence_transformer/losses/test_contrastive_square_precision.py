from __future__ import annotations

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import ContrastiveLoss, OnlineContrastiveLoss
from sentence_transformers.sentence_transformer.losses.contrastive import SiameseDistanceMetric
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize("loss_type", [ContrastiveLoss, OnlineContrastiveLoss])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("label_values", [[1, 0, 1, 0, 1, 0], [0] * 6, [1] * 6])
def test_contrastive_squares_match_float64_reference(loss_type, dtype, large, label_values):
    values = [400, 320, 512, 96, 192, 480] if large else [0.5, 0.25, 0.75, 0.125, 0.375, 0.625]
    distances = torch.tensor(values, dtype=dtype, requires_grad=True)
    labels = torch.tensor(label_values)
    loss = loss_type(None, distance_metric=lambda first, second: first[:, 0], margin=0.5)
    actual = loss.compute_loss_from_embeddings([distances[:, None], torch.zeros(6, 1, dtype=dtype)], labels)

    # Select the contributing pairs using Python numbers, then differentiate the
    # mathematical squares in float64 rather than repeating the tensor implementation.
    reference = distances.detach().double().requires_grad_()
    positives = [index for index, label in enumerate(label_values) if label == 1]
    negatives = [index for index, label in enumerate(label_values) if label == 0]
    if loss_type is OnlineContrastiveLoss:
        negative_cutoff = max(values[index] for index in positives) if positives else sum(values) / len(values)
        positive_cutoff = min(values[index] for index in negatives) if negatives else sum(values) / len(values)
        positives = [index for index in positives if values[index] > positive_cutoff]
        negatives = [index for index in negatives if values[index] < negative_cutoff]
    expected = reference.sum() * 0
    for index in positives:
        expected = expected + reference[index] ** 2
    for index in negatives:
        if values[index] < 0.5:
            expected = expected + (0.5 - reference[index]) ** 2
    if loss_type is ContrastiveLoss:
        expected = expected / (2 * len(values))

    assert torch.isfinite(actual)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-7)
    actual.backward()
    expected.backward()
    assert torch.isfinite(distances.grad).all()
    torch.testing.assert_close(distances.grad, reference.grad.to(dtype))


@pytest.mark.parametrize("loss_type", [ContrastiveLoss, OnlineContrastiveLoss])
def test_fp16_contrastive_training_with_unnormalized_word_embeddings(loss_type):
    tokenizer = WhitespaceTokenizer(vocab=["[PAD]", "zero", "large", "small"], stop_words=[])
    weights = torch.tensor([0.0, 0.0, 32.0, 0.125])[:, None].expand(4, 128).clone()
    words = WordEmbeddings(tokenizer, weights, update_embeddings=True)
    model = SentenceTransformer(modules=[words, Pooling(128, "mean")]).half()
    loss = loss_type(model, distance_metric=SiameseDistanceMetric.EUCLIDEAN)
    actual = loss([model.preprocess(["zero", "zero"]), model.preprocess(["large", "small"])], torch.tensor([1, 0]))
    expected = 128 * 32.0**2
    if loss_type is ContrastiveLoss:
        expected /= 4
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual.float(), torch.tensor(expected), rtol=1e-3, atol=0)
    actual.backward()
    assert torch.isfinite(words.emb_layer.weight.grad).all()
    assert words.emb_layer.weight.grad.abs().sum() > 0
