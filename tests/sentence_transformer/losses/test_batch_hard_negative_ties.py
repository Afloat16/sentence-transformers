from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from torch import nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import BatchHardSoftMarginTripletLoss, BatchHardTripletLoss
from sentence_transformers.sentence_transformer.losses.batch_hard_triplet import BatchHardTripletLossDistanceFunction
from sentence_transformers.sentence_transformer.modules import StaticEmbedding

LOSSES = [BatchHardTripletLoss, BatchHardSoftMarginTripletLoss]
METRICS = [
    BatchHardTripletLossDistanceFunction.euclidean_distance,
    BatchHardTripletLossDistanceFunction.cosine_distance,
]


def valid_negative_reference(loss_fn, embeddings, labels):
    distances = loss_fn.distance_metric(embeddings)
    terms = []
    for anchor in range(len(labels)):
        positives = [j for j in range(len(labels)) if j != anchor and labels[j] == labels[anchor]]
        negatives = [j for j in range(len(labels)) if labels[j] != labels[anchor]]
        if not negatives:
            terms.append(distances[anchor].sum() * 0)
            continue
        hardest_positive = distances[anchor, positives].max(dim=0).values
        hardest_negative = distances[anchor, negatives].min(dim=0).values
        difference = hardest_positive - hardest_negative
        if isinstance(loss_fn, BatchHardSoftMarginTripletLoss):
            terms.append(nn.functional.softplus(difference))
        else:
            terms.append((difference + loss_fn.triplet_margin).clamp_min(0))
    return torch.stack(terms).mean()


@pytest.mark.parametrize("loss_class", LOSSES)
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("order", [[0, 1, 2, 3], [2, 0, 3, 1]])
def test_tied_negatives_only_receive_valid_pair_gradients(loss_class, metric, dtype, order):
    # Orthogonal embeddings have equal off-diagonal distances. The old row-maximum
    # sentinel lets an invalid diagonal tie with real negatives and win the minimum.
    embeddings = torch.eye(4, dtype=dtype)[order].requires_grad_()
    reference_embeddings = embeddings.detach().clone().requires_grad_()
    labels = torch.tensor([0, 0, 1, 1])[order]
    loss_fn = loss_class(nn.Identity(), distance_metric=metric)
    actual = loss_fn.compute_loss_from_embeddings([embeddings], labels)
    expected = valid_negative_reference(loss_fn, reference_embeddings, labels)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(embeddings.grad, reference_embeddings.grad)
    assert embeddings.grad[0].abs().sum() > 0


@pytest.mark.parametrize("loss_class", LOSSES)
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("case", ["random", "one_class", "all_coincident"])
def test_batch_hard_negative_mining_matches_valid_pair_reference(loss_class, metric, case):
    generator = torch.Generator().manual_seed(42)
    embeddings = torch.randn(8, 5, generator=generator, dtype=torch.float64)
    labels = torch.arange(8) // 2
    if case == "one_class":
        labels.zero_()
    elif case == "all_coincident":
        embeddings = embeddings[:1].expand_as(embeddings).clone()
    embeddings.requires_grad_()
    reference_embeddings = embeddings.detach().clone().requires_grad_()
    loss_fn = loss_class(nn.Identity(), distance_metric=metric)
    actual = loss_fn.compute_loss_from_embeddings([embeddings], labels)
    expected = valid_negative_reference(loss_fn, reference_embeddings, labels)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(embeddings.grad, reference_embeddings.grad)
    assert torch.isfinite(embeddings.grad).all()
    if case == "one_class":
        assert actual.item() == 0
        assert embeddings.grad.count_nonzero() == 0


@pytest.mark.parametrize("loss_class", LOSSES)
def test_batch_hard_static_embedding_training_matches_valid_pair_oracle(loss_class):
    def model():
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "one": 1, "two": 2, "three": 3, "four": 4}, "[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        weights = torch.cat([torch.zeros(1, 4), torch.eye(4)])
        return SentenceTransformer(modules=[StaticEmbedding(tokenizer, embedding_weights=weights)], device="cpu")

    actual_model = model()
    expected_model = model()
    actual_loss_fn = loss_class(actual_model)
    expected_loss_fn = loss_class(expected_model)
    actual_optimizer = torch.optim.SGD(actual_model.parameters(), lr=0.1)
    expected_optimizer = torch.optim.SGD(expected_model.parameters(), lr=0.1)
    labels = torch.tensor([0, 0, 1, 1])
    texts = ["one", "two", "three", "four"]
    initial = actual_model[0].embedding.weight.detach().clone()
    for _ in range(3):
        actual_optimizer.zero_grad()
        expected_optimizer.zero_grad()
        actual = actual_loss_fn([actual_model.preprocess(texts)], labels)
        expected_embeddings = expected_model(expected_model.preprocess(texts))["sentence_embedding"]
        expected = valid_negative_reference(expected_loss_fn, expected_embeddings, labels)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        for actual_parameter, expected_parameter in zip(actual_model.parameters(), expected_model.parameters()):
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)
        actual_optimizer.step()
        expected_optimizer.step()
        for actual_parameter, expected_parameter in zip(actual_model.parameters(), expected_model.parameters()):
            torch.testing.assert_close(actual_parameter, expected_parameter)
    assert not torch.equal(actual_model[0].embedding.weight, initial)
