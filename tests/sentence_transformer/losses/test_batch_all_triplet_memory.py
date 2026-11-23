from __future__ import annotations

import pytest
import torch
from torch import nn

from sentence_transformers.sentence_transformer.losses import BatchAllTripletLoss
from sentence_transformers.sentence_transformer.losses.batch_hard_triplet import BatchHardTripletLossDistanceFunction


def _explicit_loss(embeddings, labels, distance_metric, margin):
    distances = distance_metric(embeddings)
    triplets = [
        distances[anchor, positive] - distances[anchor, negative] + margin
        for anchor in range(len(labels))
        for positive in range(len(labels))
        for negative in range(len(labels))
        if anchor != positive and labels[anchor] == labels[positive] and labels[anchor] != labels[negative]
    ]
    if not triplets:
        return distances.to(torch.promote_types(distances.dtype, torch.float32)).sum() * 0
    losses = torch.stack(triplets)
    losses = losses.to(torch.promote_types(losses.dtype, torch.float32))
    losses = torch.where(losses < 0, 0.0, losses)
    return losses.sum() / ((losses > 1e-16).sum() + 1e-16)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("distance", ["euclidean_distance", "cosine_distance"])
@pytest.mark.parametrize("labels", [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 2, 2], [0] * 8, list(range(8))])
def test_large_label_groups_match_explicit_values_and_gradients(dtype, distance, labels):
    generator = torch.Generator().manual_seed(67)
    embeddings = torch.randn(8, 4, generator=generator, dtype=dtype, requires_grad=True)
    reference = embeddings.detach().clone().requires_grad_()
    labels = torch.tensor(labels)
    metric = getattr(BatchHardTripletLossDistanceFunction, distance)
    loss_fn = BatchAllTripletLoss(nn.Identity(), distance_metric=metric, margin=0.5)

    actual = loss_fn.compute_loss_from_embeddings([embeddings], labels)
    expected = _explicit_loss(reference, labels, metric, margin=0.5)
    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(embeddings.grad, reference.grad)


@pytest.mark.parametrize("positive,negative,margin", [(1.0, 1.1, 0.1), (0.3, 0.4, 0.1), (1.0, 2.0, 1.0)])
def test_hinge_boundaries_keep_the_original_subtraction_order(positive, negative, margin):
    labels = torch.tensor([0, 0, 1, 1])
    distances = torch.full((4, 4), negative)
    distances[labels[:, None] == labels[None, :]] = positive
    distances.fill_diagonal_(0)
    distances.requires_grad_()
    reference = distances.detach().clone().requires_grad_()

    def identity(values):
        return values

    loss_fn = BatchAllTripletLoss(nn.Identity(), distance_metric=identity, margin=margin)

    actual = loss_fn.compute_loss_from_embeddings([distances], labels)
    expected = _explicit_loss(reference, labels, identity, margin)
    actual.backward()
    expected.backward()

    # In float32, (0.3 - 0.4) + 0.1 is positive although 0.3 + 0.1 == 0.4.
    # At an exact zero hinge the existing implementation retains its derivative.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(distances.grad, reference.grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_custom_distance_precision_and_upstream_scaling(dtype):
    generator = torch.Generator().manual_seed(31)
    distances = (torch.randint(0, 16, (8, 8), generator=generator) / 4).to(dtype).requires_grad_()
    reference = distances.detach().clone().requires_grad_()
    labels = torch.tensor([0] * 4 + [1] * 4)

    def identity(values):
        return values

    loss_fn = BatchAllTripletLoss(nn.Identity(), distance_metric=identity, margin=0.5)
    actual = loss_fn.compute_loss_from_embeddings([distances], labels)
    expected = _explicit_loss(reference, labels, identity, margin=0.5)
    (actual * 0.37).backward()
    (expected * 0.37).backward()

    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(distances.grad, reference.grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["no_positives", "no_negatives", "all_easy", "zero_hinge"])
def test_low_precision_zero_denominator_keeps_inactive_gradients_zero(dtype, case):
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    if case == "no_positives":
        labels = torch.arange(8)
    elif case == "no_negatives":
        labels = torch.zeros(8, dtype=torch.long)
    distances = torch.full((8, 8), 2.0 if case == "zero_hinge" else 10.0, dtype=dtype)
    distances[labels[:, None] == labels[None, :]] = 1.0 if case == "zero_hinge" else 0.0
    distances.fill_diagonal_(0)
    distances.requires_grad_()
    reference = distances.detach().clone().requires_grad_()

    def identity(values):
        return values

    loss_fn = BatchAllTripletLoss(nn.Identity(), distance_metric=identity, margin=1.0)
    actual = loss_fn.compute_loss_from_embeddings([distances], labels)
    expected = _explicit_loss(reference, labels, identity, margin=1.0)
    actual.backward()
    expected.backward()

    assert actual == expected == 0
    torch.testing.assert_close(distances.grad, reference.grad)
    assert not torch.isnan(distances.grad).any()
    if case != "zero_hinge":
        assert not distances.grad.any()


def test_large_label_groups_support_second_order_gradients():
    embeddings = torch.randn(6, 3, generator=torch.Generator().manual_seed(71), dtype=torch.float64)
    embeddings.requires_grad_()
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    loss_fn = BatchAllTripletLoss(nn.Identity(), margin=4.0)

    def loss(values):
        return loss_fn.compute_loss_from_embeddings([values], labels)

    assert torch.autograd.gradcheck(loss, (embeddings,))
    assert torch.autograd.gradgradcheck(loss, (embeddings,))


def test_scalar_tensor_margin_keeps_its_gradient():
    embeddings = torch.randn(6, 3, generator=torch.Generator().manual_seed(71), dtype=torch.float64)
    embeddings.requires_grad_()
    reference = embeddings.detach().clone().requires_grad_()
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    margin = nn.Parameter(torch.tensor(4.0, dtype=embeddings.dtype))
    reference_margin = margin.detach().clone().requires_grad_()
    metric = BatchHardTripletLossDistanceFunction.euclidean_distance
    loss_fn = BatchAllTripletLoss(nn.Identity(), margin=margin)

    actual = loss_fn.compute_loss_from_embeddings([embeddings], labels)
    expected = _explicit_loss(reference, labels, metric, reference_margin)
    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(embeddings.grad, reference.grad)
    torch.testing.assert_close(margin.grad, reference_margin.grad)


@pytest.mark.parametrize("transform", ["grad", "jvp"])
def test_large_label_groups_support_functional_autograd(transform):
    generator = torch.Generator().manual_seed(71)
    embeddings = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    metric = BatchHardTripletLossDistanceFunction.euclidean_distance
    loss_fn = BatchAllTripletLoss(nn.Identity(), margin=4.0)

    def loss(values):
        return loss_fn.compute_loss_from_embeddings([values], labels)

    def reference(values):
        return _explicit_loss(values, labels, metric, margin=4.0)

    if transform == "grad":
        actual = torch.func.grad(loss)(embeddings)
        expected = torch.func.grad(reference)(embeddings)
    else:
        tangent = torch.randn(embeddings.shape, generator=generator, dtype=embeddings.dtype)
        actual = torch.func.jvp(loss, (embeddings,), (tangent,))
        expected = torch.func.jvp(reference, (embeddings,), (tangent,))
    torch.testing.assert_close(actual, expected)


def test_large_label_groups_save_only_quadratic_tensors():
    batch_size = 64
    embeddings = torch.randn(batch_size, 8, requires_grad=True)
    labels = torch.arange(batch_size) // (batch_size // 2)
    saved_sizes = []

    def pack(tensor):
        saved_sizes.append(tensor.numel())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = BatchAllTripletLoss(nn.Identity()).compute_loss_from_embeddings([embeddings], labels)
    loss.backward()

    assert saved_sizes
    assert max(saved_sizes) <= batch_size**2
    assert torch.isfinite(loss)
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0
