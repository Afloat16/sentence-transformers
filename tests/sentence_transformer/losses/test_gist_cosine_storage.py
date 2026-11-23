from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from sentence_transformers.sentence_transformer.losses import CachedGISTEmbedLoss, GISTEmbedLoss


class LocalTokenizer(PreTrainedTokenizerBase):
    def get_vocab(self):
        return {"word": 0}


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = LocalTokenizer()
        self.max_seq_length = 16
        self.linear = nn.Linear(16, 16, bias=False)

    def __getitem__(self, index):
        return self.linear

    def forward(self, features):
        return {"sentence_embedding": self.linear(features["input_ids"].float())}


def make_loss(loss_class, **kwargs):
    model = LinearModel()
    guide = LinearModel()
    if loss_class is CachedGISTEmbedLoss:
        kwargs["mini_batch_size"] = 3
    return loss_class(model, guide, temperature=0.1, **kwargs)


@pytest.mark.parametrize("loss_class", [GISTEmbedLoss, CachedGISTEmbedLoss])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["random", "small_norms", "zeros", "noncontiguous", "large_norms"])
def test_gist_cosine_values_and_gradients_match_native(loss_class, dtype, case):
    generator = torch.Generator().manual_seed(42)
    first = torch.randn(7, 16, generator=generator, dtype=dtype)
    second = torch.randn(5, 16, generator=generator, dtype=dtype)
    loss = make_loss(loss_class)
    eps = 0.25 if case in ("small_norms", "zeros") else 1e-8
    loss.similarity_fct.eps = eps
    if case == "small_norms":
        first *= 1e-4
        second *= 1e-4
    elif case == "zeros":
        first[::2] = 0
        second[::2] = 0
    elif case == "large_norms":
        first *= 3000
        second *= 3000
    elif case == "noncontiguous":
        first = first[:, ::2]
        second = second[:, ::2]
    first.requires_grad_()
    second.requires_grad_()
    reference_first = first.detach().clone().requires_grad_()
    reference_second = second.detach().clone().requires_grad_()
    actual = loss.sim_matrix(first, second)
    expected = nn.functional.cosine_similarity(
        reference_first[:, None, :], reference_second[None, :, :], dim=-1, eps=eps
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-13 if dtype == torch.float64 else 1e-6,
        atol=1e-15 if dtype == torch.float64 else 1e-7,
    )
    weights = torch.randn(actual.shape, generator=generator, dtype=dtype)
    actual_grad = torch.autograd.grad((actual * weights).sum(), (first, second))
    expected_grad = torch.autograd.grad((expected * weights).sum(), (reference_first, reference_second))
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(
            got,
            want,
            rtol=0.04 if dtype == torch.bfloat16 else 0.005 if dtype == torch.float16 else 1e-5,
            atol=1e-5 if dtype in (torch.float16, torch.bfloat16) else 1e-6,
        )


@pytest.mark.parametrize("loss_class", [GISTEmbedLoss, CachedGISTEmbedLoss])
def test_gist_cosine_retains_linear_embedding_storage(loss_class):
    loss = make_loss(loss_class)
    first = torch.randn(128, 64, requires_grad=True)
    second = torch.randn(127, 64, requires_grad=True)
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
        value = loss.sim_matrix(first, second).sum()
    saved_bytes = sum(size for size, references in live.values())
    value.backward()
    assert not live
    assert saved_bytes < 20 * (first.numel() + second.numel()) * first.element_size()


@pytest.mark.parametrize("loss_class", [GISTEmbedLoss, CachedGISTEmbedLoss])
@pytest.mark.parametrize("margin_strategy", ["absolute", "relative"])
@pytest.mark.parametrize("with_negative", [False, True])
@pytest.mark.parametrize("contrasts", [False, True])
def test_gist_training_and_filtering_match_native_cosine(loss_class, margin_strategy, with_negative, contrasts):
    torch.manual_seed(42)
    actual = make_loss(
        loss_class,
        margin_strategy=margin_strategy,
        margin=0.1,
        contrast_anchors=contrasts,
        contrast_positives=contrasts,
    )
    reference = copy.deepcopy(actual)
    reference.similarity_fct = nn.CosineSimilarity(dim=-1)
    features = [{"input_ids": torch.randint(0, 10, (8, 16))} for _ in range(3 if with_negative else 2)]
    optimizers = [torch.optim.SGD(loss.model.parameters(), lr=0.01) for loss in (actual, reference)]
    actual_masks, reference_masks = [], []
    actual.cross_entropy_loss.register_forward_pre_hook(
        lambda module, args: actual_masks.append(torch.isneginf(args[0]))
    )
    reference.cross_entropy_loss.register_forward_pre_hook(
        lambda module, args: reference_masks.append(torch.isneginf(args[0]))
    )
    for _ in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad()
        got = actual(copy.deepcopy(features), labels=None)
        want = reference(copy.deepcopy(features), labels=None)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
        got.backward()
        want.backward()
        for got_parameter, want_parameter in zip(actual.model.parameters(), reference.model.parameters()):
            torch.testing.assert_close(got_parameter.grad, want_parameter.grad, rtol=1e-4, atol=1e-6)
        for optimizer in optimizers:
            optimizer.step()
        for got_parameter, want_parameter in zip(actual.model.parameters(), reference.model.parameters()):
            torch.testing.assert_close(got_parameter, want_parameter, rtol=1e-5, atol=1e-6)
    assert len(actual_masks) == len(reference_masks)
    for got, want in zip(actual_masks, reference_masks):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize("loss_class", [GISTEmbedLoss, CachedGISTEmbedLoss])
def test_gist_cosine_preserves_autocast_and_module_hooks(loss_class):
    loss = make_loss(loss_class)
    inputs = (torch.randn(7, 16, requires_grad=True), torch.randn(5, 16, requires_grad=True))
    calls = []
    loss.similarity_fct.register_forward_hook(lambda module, args, output: calls.append(output.dtype))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = loss.sim_matrix(*inputs)
        expected = nn.functional.cosine_similarity(inputs[0][:, None, :], inputs[1][None, :, :], dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == expected.dtype == calls[0]
    replacement = nn.PairwiseDistance()
    loss.similarity_fct = replacement
    torch.testing.assert_close(loss.sim_matrix(*inputs), replacement(inputs[0][:, None, :], inputs[1][None, :, :]))
