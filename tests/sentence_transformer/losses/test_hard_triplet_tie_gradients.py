from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sentence_transformers.sentence_transformer.losses import (
    BatchHardSoftMarginTripletLoss,
    BatchHardTripletLoss,
    BatchHardTripletLossDistanceFunction,
)


@pytest.mark.parametrize("loss_class", [BatchHardTripletLoss, BatchHardSoftMarginTripletLoss])
@pytest.mark.parametrize("distance", ["euclidean", "cosine"])
@pytest.mark.parametrize("tied", [False, True])
def test_hard_triplet_gradients_select_real_negatives(loss_class, distance: str, tied: bool) -> None:
    if tied:
        values = [[0.0], [2.0], [-2.0], [-2.0]]
        if distance == "cosine":
            values = [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0]]
        embeddings = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    else:
        embeddings = torch.randn(4, 3, generator=torch.Generator().manual_seed(24), dtype=torch.float64)
        embeddings.requires_grad_()
    reference_embeddings = embeddings.detach().clone().requires_grad_()
    labels = torch.tensor([0, 0, 1, 1])
    distance_metric = getattr(BatchHardTripletLossDistanceFunction, f"{distance}_distance")

    actual = loss_class(None, distance_metric=distance_metric).compute_loss_from_embeddings([embeddings], labels)
    terms = []
    for anchor in range(len(labels)):
        positives, negatives = [], []
        for other in range(len(labels)):
            if anchor == other:
                continue
            if distance == "euclidean":
                value = torch.linalg.vector_norm(reference_embeddings[anchor] - reference_embeddings[other])
            else:
                value = 1 - F.cosine_similarity(reference_embeddings[anchor], reference_embeddings[other], dim=0)
            (positives if labels[anchor] == labels[other] else negatives).append(value)
        difference = torch.stack(positives).max(dim=0).values - torch.stack(negatives).min(dim=0).values
        terms.append(F.relu(difference + 5) if loss_class is BatchHardTripletLoss else F.softplus(difference))
    expected = torch.stack(terms).mean()

    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(embeddings.grad, reference_embeddings.grad)
