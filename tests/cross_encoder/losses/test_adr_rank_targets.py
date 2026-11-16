from __future__ import annotations

import math

import pytest
import torch

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import ADRMSELoss


class _ScoreTable(CrossEncoder):
    def __init__(self, scores):
        torch.nn.Module.__init__(self)
        self.scores = torch.nn.Parameter(scores)

    @property
    def num_labels(self):
        return 1

    @property
    def device(self):
        return self.scores.device

    def preprocess(self, pairs, **kwargs):
        return {"indices": torch.tensor([int(document) for _, document in pairs])}

    def forward(self, inputs):
        return {"scores": self.scores[inputs["indices"]].unsqueeze(-1)}


@pytest.mark.parametrize(
    "label_rows",
    [
        [[3, 1, 1, 0], [0, 0], [1]],
        [[-3, -1, -1, 2], [4, 1, 3]],
        [[0, 0, 0, 0], [2, 2]],
        [[-math.inf, math.inf, 2, 2], [math.inf], [-math.inf]],
        [[math.nan, 1, 2, 2], [math.nan], [1, math.nan]],
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("alpha", [0.1, 1.0, 4.0])
@pytest.mark.parametrize("mini_batch_size", [1, -1])
def test_adr_ranks_match_per_query_reference(label_rows, dtype, alpha, mini_batch_size):
    count = sum(map(len, label_rows))
    values = torch.linspace(-1.2, 1.4, count).to(dtype)
    model = _ScoreTable(values.clone())
    queries, documents, labels = [], [], []
    offset = 0
    for index, row in enumerate(label_rows):
        queries.append(str(index))
        documents.append([str(i) for i in range(offset, offset + len(row))])
        labels.append(torch.tensor(row))
        offset += len(row)
    actual = ADRMSELoss(model, alpha=alpha, mini_batch_size=mini_batch_size)((queries, documents), labels)

    # Rank each unpadded query with scalar comparisons. No sorting, search bounds,
    # padding sentinel or batched pairwise target matrix is used by this oracle.
    scores = values.float().double().requires_grad_()
    expected = scores.sum() * 0
    offset = 0
    for row in label_rows:
        for index, label in enumerate(row):
            greater = sum(other > label for other in row)
            equal = sum(other == label for other in row)
            rank = greater + (equal + 1) / 2
            approx = scores.new_tensor(1)
            for other in range(len(row)):
                if other != index:
                    approx = approx + torch.sigmoid(alpha * (scores[offset + other] - scores[offset + index]))
            expected = expected + (rank - approx).square() / math.log2(rank + 1)
        offset += len(row)
    expected = expected / count
    torch.testing.assert_close(actual.double(), expected, rtol=3e-6, atol=2e-7)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(
        model.scores.grad, scores.grad.to(dtype), rtol=0.02 if dtype == torch.bfloat16 else 0.002, atol=2e-7
    )


def test_adr_equal_labels_and_scores_have_zero_loss_and_gradients():
    model = _ScoreTable(torch.ones(7))
    actual = ADRMSELoss(model)(
        (["q0", "q1"], [["0", "1", "2", "3"], ["4", "5", "6"]]), [torch.zeros(4), torch.ones(3)]
    )
    assert actual.item() == 0
    actual.backward()
    torch.testing.assert_close(model.scores.grad, torch.zeros(7), rtol=0, atol=0)
