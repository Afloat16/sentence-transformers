from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

import sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking as cached_module
import sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking as plain_module
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
)


def _mock_gather(monkeypatch, rank: int) -> None:
    """Supply distinct remote candidates without launching a distributed process group."""

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [tensor if index == rank else tensor.detach().flip(0) + 0.25 * (index + 1) for index in range(3)]
        )

    for module in (plain_module, cached_module):
        monkeypatch.setattr(module, "all_gather_with_grad", gather)
        monkeypatch.setattr(module, "get_rank", lambda: rank)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("rank", [0, 2])
@pytest.mark.parametrize("num_negatives", [0, 2])
@pytest.mark.parametrize("hardness_mode", [None, "in_batch_negatives", "hard_negatives", "all_negatives"])
@pytest.mark.parametrize(
    ("directions", "partition_mode"),
    [
        (("query_to_doc",), "joint"),
        (("query_to_doc", "doc_to_query"), "per_direction"),
        (("query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"), "joint"),
    ],
)
def test_mnrl_masks_match_cached_loss_and_gradients(
    monkeypatch, dtype, rank, num_negatives, hardness_mode, directions, partition_mode
) -> None:
    _mock_gather(monkeypatch, rank)
    generator = torch.Generator().manual_seed(17)
    embeddings = [torch.randn(2, 7, dtype=dtype, generator=generator) for _ in range(2 + num_negatives)]
    plain_embeddings = [embedding.clone().requires_grad_() for embedding in embeddings]
    cached_embeddings = [embedding.clone().requires_grad_() for embedding in embeddings]
    config = {
        "scale": 7.0,
        "gather_across_devices": True,
        "directions": directions,
        "partition_mode": partition_mode,
        "hardness_mode": hardness_mode,
        "hardness_strength": 5.0,
    }
    plain = MultipleNegativesRankingLoss(torch.nn.Identity(), **config)
    # Match the local batch size so low-precision gradient accumulation order does not differ.
    cached = CachedMultipleNegativesRankingLoss(torch.nn.Sequential(torch.nn.Identity()), mini_batch_size=2, **config)

    actual = plain.compute_loss_from_embeddings(plain_embeddings, labels=None)
    expected = cached.calculate_loss([[embedding] for embedding in cached_embeddings])
    actual_grads = torch.autograd.grad(actual, plain_embeddings)
    expected_grads = torch.autograd.grad(expected, cached_embeddings)

    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("doc_to_doc", [False, True])
@pytest.mark.parametrize("hardness_mode", [None, "in_batch_negatives", "hard_negatives", "all_negatives"])
def test_mnrl_masks_do_not_allocate_global_square(monkeypatch, doc_to_doc, hardness_mode) -> None:
    _mock_gather(monkeypatch, rank=2)
    local_batch_size, world_batch_size, embedding_dim = 2, 6, 7

    class TrackGlobalSquares(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.global_square_shapes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            self.global_square_shapes.extend(
                tuple(value.shape)
                for value in tree_leaves(result)
                if isinstance(value, torch.Tensor) and tuple(value.shape) == (world_batch_size, world_batch_size)
            )
            return result

    loss_fn = MultipleNegativesRankingLoss(
        torch.nn.Identity(),
        gather_across_devices=True,
        directions=("query_to_doc", "doc_to_doc") if doc_to_doc else ("query_to_doc",),
        hardness_mode=hardness_mode,
        hardness_strength=5.0,
    )
    embeddings = [torch.randn(local_batch_size, embedding_dim, requires_grad=True) for _ in range(4)]

    with TrackGlobalSquares() as allocations:
        loss = loss_fn.compute_loss_from_embeddings(embeddings, labels=None)
        loss.backward()

    assert torch.isfinite(loss)
    assert allocations.global_square_shapes == [], "Only local rows should be allocated for global candidate masks"
