from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from sentence_transformers.sentence_transformer.losses import CachedGISTEmbedLoss, GISTEmbedLoss


class _Tokenizer(PreTrainedTokenizerBase):
    def get_vocab(self):
        return {"token": 0}


class _EmbeddingInput(nn.Module):
    max_seq_length = 16

    def __init__(self):
        super().__init__()
        self.tokenizer = _Tokenizer()
        self.input_module = nn.Identity()

    def __getitem__(self, index):
        return self.input_module

    def forward(self, features):
        return {"sentence_embedding": features["sentence_embedding"]}


def _make_loss(loss_class, **kwargs):
    return loss_class(_EmbeddingInput(), _EmbeddingInput(), **kwargs)


@pytest.mark.parametrize("loss_class", [GISTEmbedLoss, CachedGISTEmbedLoss])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [0.0, 1e-10, 1.0, 1e8])
@pytest.mark.parametrize("autocast", [False, True])
def test_cosine_matrix_preserves_values_and_gradients(loss_class, dtype, scale, autocast):
    loss = _make_loss(loss_class)
    generator = torch.Generator().manual_seed(13)
    # Noncontiguous inputs and different row counts exercise rectangular candidate sets.
    left = (torch.randn(7, 3, generator=generator, dtype=dtype) * scale).T.requires_grad_()
    right = (torch.randn(7, 5, generator=generator, dtype=dtype) * scale).T.requires_grad_()
    weights = torch.randn(3, 5, generator=generator, dtype=dtype)
    context = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
    with context:
        expected = loss.similarity_fct(left.unsqueeze(1), right.unsqueeze(0))
        actual = loss.sim_matrix(left, right)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad((expected * weights).sum(), (left, right))
    actual_grads = torch.autograd.grad((actual * weights).sum(), (left, right))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        # Matrix multiplication sums the float32 gradient contributions in a different order.
        tolerance = {"rtol": 1e-5, "atol": 1e-5} if dtype == torch.float32 else {}
        torch.testing.assert_close(actual_grad, expected_grad, **tolerance)


@pytest.mark.parametrize("loss_class", [GISTEmbedLoss, CachedGISTEmbedLoss])
@pytest.mark.parametrize("eps", [1e-4, 0.5])
def test_cosine_matrix_preserves_configured_epsilon(loss_class, eps):
    loss = _make_loss(loss_class)
    loss.similarity_fct.eps = eps
    left = torch.tensor([[0.1, 0.2], [0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([[0.2, 0.1]], dtype=torch.float64, requires_grad=True)
    expected = loss.similarity_fct(left.unsqueeze(1), right.unsqueeze(0))
    actual = loss.sim_matrix(left, right)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.sum(), (left, right)), torch.autograd.grad(expected.sum(), (left, right))
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("loss_class", [GISTEmbedLoss, CachedGISTEmbedLoss])
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float64),
        (torch.bfloat16, torch.float32),
    ],
)
def test_cosine_matrix_preserves_other_dtypes(loss_class, dtypes):
    loss = _make_loss(loss_class)
    left = torch.tensor([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]], dtype=dtypes[0], requires_grad=True)
    right = torch.tensor([[3.0, 4.0, 1.0]], dtype=dtypes[1], requires_grad=True)
    actual = loss.sim_matrix(left, right)
    expected = loss.similarity_fct(left.unsqueeze(1), right.unsqueeze(0))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.sum(), (left, right)), torch.autograd.grad(expected.sum(), (left, right))
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


class _ModifiedCosine(nn.CosineSimilarity):
    def forward(self, left, right):
        return super().forward(left, right) + 0.25


@pytest.mark.parametrize("loss_class", [GISTEmbedLoss, CachedGISTEmbedLoss])
def test_cosine_matrix_respects_custom_similarity(loss_class):
    loss = _make_loss(loss_class)
    loss.similarity_fct = _ModifiedCosine(dim=-1)
    left = torch.eye(3)
    right = torch.ones(2, 3)
    torch.testing.assert_close(loss.sim_matrix(left, right), loss.similarity_fct(left[:, None], right[None, :]))


@pytest.mark.parametrize("loss_class", [GISTEmbedLoss, CachedGISTEmbedLoss])
@pytest.mark.parametrize("margin_strategy", ["absolute", "relative"])
@pytest.mark.parametrize("num_columns", [2, 3])
def test_gist_training_loss_matches_broadcast_cosine(loss_class, margin_strategy, num_columns, monkeypatch):
    kwargs = {"mini_batch_size": 2} if loss_class is CachedGISTEmbedLoss else {}
    loss = _make_loss(loss_class, margin_strategy=margin_strategy, margin=0.05, **kwargs)
    generator = torch.Generator().manual_seed(17)
    columns = [torch.randn(5, 7, generator=generator, requires_grad=True) for _ in range(num_columns)]

    def evaluate():
        if loss_class is CachedGISTEmbedLoss:
            reps = [list(column.split(2)) for column in columns]
            guide_reps = [[part.detach() for part in column] for column in reps]
            return loss.calculate_loss(reps, guide_reps)
        return loss([{"sentence_embedding": column} for column in columns], labels=None)

    actual = evaluate()
    actual_grads = torch.autograd.grad(actual, columns)
    monkeypatch.setattr(loss, "sim_matrix", lambda left, right: loss.similarity_fct(left[:, None], right[None, :]))
    expected = evaluate()
    expected_grads = torch.autograd.grad(expected, columns)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
