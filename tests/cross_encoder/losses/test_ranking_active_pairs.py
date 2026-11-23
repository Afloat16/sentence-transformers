from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses.lambda_loss import (
    BaseWeightingScheme,
    LambdaLoss,
    LambdaRankScheme,
    NDCGLoss1Scheme,
    NDCGLoss2PPScheme,
    NDCGLoss2Scheme,
    NoWeightingScheme,
)

SCHEMES = [NoWeightingScheme, NDCGLoss1Scheme, NDCGLoss2Scheme, LambdaRankScheme, NDCGLoss2PPScheme]


class ScoreModel(CrossEncoder):
    def __init__(self, scores):
        nn.Module.__init__(self)
        self.scores = nn.Parameter(scores.clone())

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


def reference(scores, labels, scheme, k, sigma, reduction_log):
    """Enumerate contributing pairs directly, without a padded pairwise score matrix."""
    max_docs = max(map(len, labels))
    k = k or max_docs
    discount = torch.log2(torch.arange(2, max_docs + 2, dtype=scores.dtype))[None, :]
    terms = []
    offset = 0
    for query_labels in labels:
        query_scores, order = scores[offset : offset + len(query_labels)].sort(descending=True)
        offset += len(query_labels)
        ordered_labels = query_labels[order]
        ideal = nn.functional.pad(
            query_labels.sort(descending=True).values.clamp_min(0), (0, max_docs - len(query_labels))
        )
        max_dcg = (((2**ideal - 1) / discount[0])[:k].sum()).clamp_min(1e-10)
        padded_labels = nn.functional.pad(ordered_labels.clamp_min(0), (0, max_docs - len(query_labels)))
        gain = ((2**padded_labels - 1) / max_dcg)[None, :]
        weights = scheme(gain, discount, padded_labels[None, :]).broadcast_to((1, max_docs, max_docs))
        for row in range(len(query_labels)):
            for column in range(len(query_labels)):
                if row >= k and column >= k:
                    continue
                if not isinstance(scheme, NDCGLoss1Scheme) and ordered_labels[row] <= ordered_labels[column]:
                    continue
                difference = (query_scores[row] - query_scores[column]).clamp(-1e8, 1e8)
                term = weights[0, row, column] * nn.functional.softplus(-sigma * difference)
                if reduction_log == "binary":
                    term = term / math.log(2)
                terms.append(term)
    return torch.stack(terms).mean() if terms else scores.sum() * 0


@pytest.mark.parametrize("scheme_class", SCHEMES)
@pytest.mark.parametrize("k", [1, 3, None, 10])
@pytest.mark.parametrize("case", ["mixed", "tied_scores", "no_ordered_pairs"])
@pytest.mark.parametrize("reduction_log", ["natural", "binary"])
def test_active_ranking_pairs_match_explicit_reference(scheme_class, k, case, reduction_log):
    scores = torch.tensor([0.3, -2, 1, 3, -1, 2, 0, 4, -2, 3, 1, 0, -1, 2])
    labels = [
        torch.tensor([0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 0.0]),
        torch.tensor([2.0, 0.0, 1.0, 2.0, 0.0]),
        torch.tensor([1.0]),
    ]
    if case == "tied_scores":
        scores = scores.round().remainder(2)
    elif case == "no_ordered_pairs":
        labels = [torch.zeros_like(row) for row in labels]
    model = ScoreModel(scores)
    scheme = scheme_class()
    documents = [[str(i) for i in range(8)], [str(i) for i in range(8, 13)], ["13"]]
    loss_fn = LambdaLoss(
        model, weighting_scheme=scheme, k=k, sigma=1.7, reduction_log=reduction_log, mini_batch_size=3
    )
    actual = loss_fn((["q0", "q1", "q2"], documents), labels)
    expected_scores = scores.clone().requires_grad_()
    expected = reference(expected_scores, labels, scheme_class(), k, 1.7, reduction_log)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.scores.grad, expected_scores.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("shape", ["scalar", "row", "column", "matrix"])
def test_active_ranking_pairs_preserve_custom_weight_gradients_and_hooks(shape):
    class LearnedWeights(BaseWeightingScheme):
        def __init__(self):
            super().__init__()
            self.factor = nn.Parameter(torch.tensor(1.3))

        def forward(self, gain, discount, true_sorted):
            if shape == "scalar":
                return self.factor
            if shape == "row":
                return self.factor * (gain[:, :, None] + 0.2)
            if shape == "column":
                return self.factor * (gain[:, None, :] + 0.2)
            return self.factor * (gain[:, :, None] + gain[:, None, :] + 0.2)

    scheme = LearnedWeights()
    expected_scheme = copy.deepcopy(scheme)
    calls = []
    scheme.register_forward_hook(lambda module, args, output: calls.append(output.shape))
    scores = torch.arange(8, dtype=torch.float32)
    labels = [torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0])]
    model = ScoreModel(scores)
    actual = LambdaLoss(model, weighting_scheme=scheme, k=1)((["query"], [[str(i) for i in range(8)]]), labels)
    expected_scores = scores.clone().requires_grad_()
    expected = reference(expected_scores, labels, expected_scheme, 1, 1.0, "binary")
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.scores.grad, expected_scores.grad)
    torch.testing.assert_close(scheme.factor.grad, expected_scheme.factor.grad)
    assert len(calls) == 1


def retained_scoring_bytes(scheme_class, k, n_docs=128):
    model = ScoreModel(torch.linspace(-2, 2, n_docs))
    labels = [torch.arange(n_docs).remainder(4).float()]
    live = {}

    class SavedTensor:
        def __init__(self, tensor):
            self.tensor = tensor.detach()
            storage = tensor.untyped_storage()
            self.key = storage.data_ptr()
            if self.key not in live:
                live[self.key] = [storage.nbytes(), 0]
            live[self.key][1] += 1

        def __del__(self):
            live[self.key][1] -= 1
            if live[self.key][1] == 0:
                del live[self.key]

    with torch.autograd.graph.saved_tensors_hooks(SavedTensor, lambda saved: saved.tensor):
        loss = LambdaLoss(model, weighting_scheme=scheme_class(), k=k, mini_batch_size=32)(
            (["query"], [[str(i) for i in range(n_docs)]]), labels
        )
    retained_bytes = sum(size for size, references in live.values())
    loss.backward()
    assert not live
    assert torch.isfinite(model.scores.grad).all()
    return retained_bytes


@pytest.mark.parametrize("scheme_class", SCHEMES)
def test_small_cutoff_reduces_retained_scoring_memory(scheme_class):
    """Count live, unique saved storage, rather than temporary tensors or Python references."""
    assert retained_scoring_bytes(scheme_class, k=4) < retained_scoring_bytes(scheme_class, k=None) / 2


def test_ndcg1_broad_cutoff_does_not_expand_row_weights():
    # The broad-cutoff path fits below four full float32 score matrices plus
    # linear input metadata; materializing its broadcast weights exceeds this.
    assert retained_scoring_bytes(NDCGLoss1Scheme, k=None) < 16 * 128**2 + 128 * 128
