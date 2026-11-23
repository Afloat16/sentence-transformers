from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from sentence_transformers.cross_encoder.losses import ADRMSELoss


class ScoreModel(nn.Module):
    num_labels = 1

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    @property
    def device(self):
        return self.linear.weight.device

    def preprocess(self, pairs, **kwargs):
        indices = torch.tensor([int(document) for _, document in pairs], dtype=torch.float32)
        return {"features": torch.stack([indices.sin(), indices.cos(), indices / 1000], dim=-1)}

    def forward(self, features):
        return {"scores": self.linear(features["features"])}


def make_loss(chunk_size, **kwargs):
    loss = ADRMSELoss(ScoreModel(), **kwargs)
    # Setting the option after construction lets the storage test expose the old
    # allocation, rather than just failing because the new argument is missing.
    loss.rank_chunk_size = chunk_size
    return loss


def pairwise_reference(scores, mask, alpha):
    rows = []
    for row_scores, row_mask in zip(scores, mask):
        ranks = []
        for index, score in enumerate(row_scores):
            terms = [
                torch.sigmoid(alpha * (candidate - score))
                for other, candidate in enumerate(row_scores)
                if row_mask[other] and other != index
            ]
            ranks.append(1 + sum(terms, score * 0))
        rows.append(torch.stack(ranks))
    return torch.stack(rows)


@pytest.mark.parametrize("chunk_size", [1, 4, 8, 32, None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("alpha", [0.0, 0.3, 3.0])
def test_chunked_adr_ranks_and_gradients_match_pair_enumeration(chunk_size, dtype, alpha):
    generator = torch.Generator().manual_seed(42)
    scores = torch.randn(3, 9, generator=generator, dtype=dtype)
    scores[0, :3] = 0.5
    scores[1, 7:] = -1e16
    scores.requires_grad_()
    reference_scores = scores.detach().clone().requires_grad_()
    mask = torch.tensor([[True] * 9, [True] * 7 + [False] * 2, [False] * 9])
    loss = make_loss(chunk_size, alpha=alpha)
    actual = loss.approximate_ranks(scores, mask)
    expected = pairwise_reference(reference_scores, mask, alpha)
    torch.testing.assert_close(actual, expected)
    weights = torch.randn(actual.shape, generator=generator, dtype=dtype)
    actual_grad = torch.autograd.grad((actual * weights).sum(), scores)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), reference_scores)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("chunk_size", [1, 5, 32])
def test_chunked_adr_retains_second_derivatives_and_learned_alpha(chunk_size):
    scores = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
    alpha = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    loss = make_loss(chunk_size, alpha=alpha)
    mask = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])

    def fn(scores):
        return loss.approximate_ranks(scores, mask)

    assert torch.autograd.gradcheck(fn, (scores,), fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, (scores,), fast_mode=True)
    reference_scores = scores.detach().clone().requires_grad_()
    reference_alpha = alpha.detach().clone().requires_grad_()
    weights = torch.arange(14, dtype=scores.dtype).reshape(2, 7)
    actual = (fn(scores) * weights).sum()
    expected = (pairwise_reference(reference_scores, mask, reference_alpha) * weights).sum()
    actual_grad = torch.autograd.grad(actual, (scores, alpha), create_graph=True)
    expected_grad = torch.autograd.grad(expected, (reference_scores, reference_alpha), create_graph=True)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want)
    actual_second = torch.autograd.grad(sum(value.square().sum() for value in actual_grad), (scores, alpha))
    expected_second = torch.autograd.grad(
        sum(value.square().sum() for value in expected_grad), (reference_scores, reference_alpha)
    )
    for got, want in zip(actual_second, expected_second):
        torch.testing.assert_close(got, want)


def test_chunked_adr_bounds_retained_pairwise_storage():
    loss = make_loss(32)
    scores = torch.randn(2, 256, requires_grad=True)
    mask = torch.arange(256)[None, :] < torch.tensor([[256], [129]])
    live = {}

    class Saved:
        def __init__(self, tensor):
            self.tensor = tensor.detach()
            storage = tensor.untyped_storage()
            self.key = storage.data_ptr()
            live.setdefault(self.key, [storage.nbytes(), 0])
            live[self.key][1] += 1

        def __del__(self):
            live[self.key][1] -= 1
            if live[self.key][1] == 0:
                del live[self.key]

    with torch.autograd.graph.saved_tensors_hooks(Saved, lambda saved: saved.tensor):
        value = loss.approximate_ranks(scores, mask).sum()
    saved_bytes = sum(size for size, count in live.values())
    value.backward()
    assert not live
    assert saved_bytes < 50 * scores.numel() * scores.element_size()


@pytest.mark.parametrize("chunk_size", [1, 4, 32])
@pytest.mark.parametrize("autocast", [False, True])
def test_chunked_adr_public_training_matches_unbounded(chunk_size, autocast):
    torch.manual_seed(42)
    model = ScoreModel()
    reference_model = copy.deepcopy(model)
    loss = ADRMSELoss(model, mini_batch_size=4)
    loss.rank_chunk_size = chunk_size
    reference = ADRMSELoss(reference_model, mini_batch_size=4)
    inputs = (["first", "second", "single"], [[str(i) for i in range(9)], ["20", "21", "22"], ["30"]])
    labels = [torch.tensor([2, 2, 0, 1, 0, 3, 3, 1, 0]), torch.tensor([1, 1, 0]), torch.tensor([1])]
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.01) for m in (model, reference_model)]
    for _ in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            actual = loss(inputs, labels, prompt="prompt", task="task")
            expected = reference(inputs, labels, prompt="prompt", task="task")
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        actual.backward()
        expected.backward()
        for parameter, reference_parameter in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=2e-5, atol=2e-6)
        for optimizer in optimizers:
            optimizer.step()
        for parameter, reference_parameter in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(parameter, reference_parameter, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_chunked_adr_preserves_strided_inputs(grad_enabled):
    scores = torch.randn(2, 18, dtype=torch.float64, requires_grad=grad_enabled)[:, ::2]
    mask = torch.arange(18)[None, :].expand(2, -1)[:, ::2] < 11
    with torch.set_grad_enabled(grad_enabled):
        actual = make_loss(4).approximate_ranks(scores, mask)
        expected = make_loss(None).approximate_ranks(scores, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, "4"])
def test_chunked_adr_rejects_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="rank_chunk_size"):
        ADRMSELoss(ScoreModel(), rank_chunk_size=chunk_size)


def test_chunked_adr_config_round_trip():
    loss = ADRMSELoss(ScoreModel(), rank_chunk_size=8)
    assert loss.get_config_dict()["rank_chunk_size"] == 8
    assert ADRMSELoss(ScoreModel()).get_config_dict()["rank_chunk_size"] is None


def test_unchunked_adr_keeps_torch_func_support():
    loss = ADRMSELoss(ScoreModel())
    scores = torch.randn(2, 7, requires_grad=True)
    mask = torch.ones_like(scores, dtype=torch.bool)
    weights = torch.arange(14).reshape_as(scores)

    def objective(scores):
        return (loss.approximate_ranks(scores, mask) * weights).sum()

    actual = torch.func.grad(objective)(scores)
    expected = torch.autograd.grad(objective(scores), scores)[0]
    torch.testing.assert_close(actual, expected)


def test_chunked_adr_torch_compile_forward_and_backward():
    loss = ADRMSELoss(ScoreModel(), rank_chunk_size=4)
    scores = torch.randn(2, 9, requires_grad=True)
    reference_scores = scores.detach().clone().requires_grad_()
    mask = torch.arange(9)[None, :] < torch.tensor([[9], [5]])
    compiled = torch.compile(loss.approximate_ranks, backend="eager", fullgraph=True)
    actual = compiled(scores, mask)
    expected = make_loss(None).approximate_ranks(reference_scores, mask)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), scores)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), reference_scores)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=1e-5)


def test_chunked_adr_pretrained_forward_and_backward(reranker_bert_tiny_model_v54):
    model = reranker_bert_tiny_model_v54
    model.eval()
    inputs = (
        ["What is Python?", "What is PyTorch?"],
        [["A language.", "A snake.", "A tool."], ["A framework.", "A lunch."]],
    )
    labels = [torch.tensor([2.0, 0.0, 1.0]), torch.tensor([1.0, 0.0])]
    loss = ADRMSELoss(model, rank_chunk_size=2, mini_batch_size=2)
    reference = ADRMSELoss(model, mini_batch_size=2)
    actual = loss(inputs, labels)
    actual.backward()
    actual_grads = {
        name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    model.zero_grad()
    expected = reference(inputs, labels)
    expected.backward()
    torch.testing.assert_close(actual, expected)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(actual_grads[name], parameter.grad, rtol=2e-5, atol=2e-6)
