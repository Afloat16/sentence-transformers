from __future__ import annotations

import math
from decimal import Decimal, localcontext

import pytest
import torch
from torch import nn

from sentence_transformers.cross_encoder.losses.lambda_loss import (
    LambdaLoss,
    LambdaRankScheme,
    NDCGLoss1Scheme,
    NDCGLoss2PPScheme,
    NDCGLoss2Scheme,
    NoWeightingScheme,
)


class ScoreModel(nn.Module):
    num_labels = 1

    def __init__(self, scores):
        super().__init__()
        self.scores = nn.Parameter(scores)

    @property
    def device(self):
        return self.scores.device

    def preprocess(self, pairs, **kwargs):
        return {"indices": torch.tensor([int(document) for _, document in pairs])}

    def forward(self, inputs):
        return {"scores": self.scores[inputs["indices"]].unsqueeze(-1)}


def reference_loss(scores, labels, scheme, k, reduction_log, eps):
    """Enumerate ordered pairs with 80-digit gains, without float32 exponentiation."""
    order = scores.detach().argsort(descending=True).tolist()
    count = len(order)
    cutoff = count if k is None else k
    discounts = [math.log2(rank + 2) for rank in range(count)]
    with localcontext() as context:
        context.prec = 80
        gains = [Decimal(2) ** Decimal(max(float(label), 0.0)) - 1 for label in labels]
        ideal = sum(
            gain / Decimal(discount) for gain, discount in zip(sorted(gains, reverse=True)[:cutoff], discounts)
        )
        denominator = max(ideal, Decimal(eps))
        normalized = [float(gain / denominator) for gain in gains]

    terms = []
    for rank_i, i in enumerate(order):
        for rank_j, j in enumerate(order):
            if rank_i >= cutoff and rank_j >= cutoff:
                continue
            if scheme != "ndcg1" and labels[i] <= labels[j]:
                continue
            difference = abs(normalized[i] - normalized[j])
            rank_distance = abs(rank_i - rank_j)
            delta = 1 / math.log2(rank_distance + 1) - 1 / math.log2(rank_distance + 2) if rank_distance else 0
            rank_weight = abs(1 / discounts[rank_i] - 1 / discounts[rank_j]) * difference
            if scheme == "ndcg1":
                weight = normalized[i] / discounts[rank_i]
            elif scheme == "ndcg2":
                weight = delta * difference
            elif scheme == "lambda_rank":
                weight = rank_weight
            elif scheme == "ndcg2pp":
                weight = 10 * delta * difference + rank_weight
            else:
                weight = 1
            terms.append(weight * torch.nn.functional.softplus(scores[j] - scores[i]))
    result = torch.stack(terms).mean() if terms else scores.sum() * 0
    return result / math.log(2) if reduction_log == "binary" else result


@pytest.mark.parametrize(
    "values",
    [
        [130.0, 129.0, 0.0],
        [1000.0, 999.0, 998.0],
        [1e-8, 0.0, 0.0],
        [1e-12, 0.0, 0.0],
        [3.0, 2.0, 0.0],
        [2.5, 2.5, 0.0],
        [0.0, 0.0, 0.0],
        [-2.0, -1.0, 0.0],
    ],
)
@pytest.mark.parametrize(
    ("scheme", "weighting_class"),
    [
        ("none", NoWeightingScheme),
        ("ndcg1", NDCGLoss1Scheme),
        ("ndcg2", NDCGLoss2Scheme),
        ("lambda_rank", LambdaRankScheme),
        ("ndcg2pp", NDCGLoss2PPScheme),
    ],
)
@pytest.mark.parametrize("k", [None, 1, 2])
@pytest.mark.parametrize("reduction_log", ["binary", "natural"])
def test_lambda_gains_match_high_precision(values, scheme, weighting_class, k, reduction_log):
    model = ScoreModel(torch.tensor([0.0, 1.0, 2.0]))
    labels = torch.tensor(values)
    loss_fn = LambdaLoss(model, weighting_scheme=weighting_class(), k=k, reduction_log=reduction_log)
    actual = loss_fn((["query"], [["0", "1", "2"]]), [labels])
    reference_scores = model.scores.detach().double().requires_grad_()
    expected = reference_loss(reference_scores, labels, scheme, k, reduction_log, loss_fn.eps)

    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected.float(), rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(model.scores.grad, reference_scores.grad.float(), rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("eps", [1e-10, 10.0])
def test_lambda_gain_scale_preserves_epsilon_and_padding(eps):
    model = ScoreModel(torch.tensor([0.0, 1.0, 2.0, 3.0, -1.0]))
    labels = [torch.tensor([130.0, 129.0, 0.0]), torch.tensor([0.1, 0.0])]
    actual = LambdaLoss(model, eps=eps)((["a", "b"], [["0", "1", "2"], ["3", "4"]]), labels)
    reference_scores = model.scores.detach().double().requires_grad_()
    # Three ordered pairs in the first query, one in the second.
    expected = (
        3 * reference_loss(reference_scores[:3], labels[0], "ndcg2pp", None, "binary", eps)
        + reference_loss(reference_scores[3:], labels[1], "ndcg2pp", None, "binary", eps)
    ) / 4
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected.float())
    torch.testing.assert_close(model.scores.grad, reference_scores.grad.float())
