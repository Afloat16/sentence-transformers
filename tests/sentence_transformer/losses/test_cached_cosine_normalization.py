from __future__ import annotations

from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.losses import cached_multiple_negatives_ranking as cached_module
from sentence_transformers.util import cos_sim
from sentence_transformers.util import similarity as similarity_module
from tests.utils import skip_bfloat16_cpu_crash

ALL_DIRECTIONS = ("query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc")


class TinyEncoder(nn.Module):
    def __init__(self, output_dtype=None):
        super().__init__()
        self.linear = nn.Linear(5, 7)
        self.output_dtype = output_dtype

    def __getitem__(self, index):
        return self.linear

    def forward(self, features):
        embeddings = self.linear(features["values"])
        if self.output_dtype is not None:
            embeddings = embeddings.to(self.output_dtype)
        return {"sentence_embedding": embeddings}


def _embeddings(dtype=torch.float32, rank=0):
    generator = torch.Generator().manual_seed(91 + rank)
    return [torch.randn(9, 7, dtype=dtype, generator=generator).requires_grad_() for _ in range(3)]


def _chunks(embeddings, size=3):
    # These are leaf tensors, as produced by the embedding stage of GradCache.
    return [[chunk.detach().requires_grad_() for chunk in embedding.split(size)] for embedding in embeddings]


@pytest.mark.parametrize("with_backward", [False, True])
@pytest.mark.parametrize("mini_batch_size", [2, 4])
@pytest.mark.parametrize("directions", [("query_to_doc",), ALL_DIRECTIONS])
def test_cosine_normalization_work_is_linear_in_embedding_rows(
    monkeypatch, with_backward, mini_batch_size, directions
):
    normalized_rows = []
    normalize = similarity_module.normalize_embeddings

    def record(embeddings):
        normalized_rows.append(len(embeddings))
        return normalize(embeddings)

    monkeypatch.setattr(similarity_module, "normalize_embeddings", record)
    monkeypatch.setattr(cached_module, "normalize_embeddings", record, raising=False)
    embeddings = _embeddings()
    loss = CachedMultipleNegativesRankingLoss(TinyEncoder(), mini_batch_size=mini_batch_size, directions=directions)
    loss.calculate_loss(_chunks(embeddings), with_backward=with_backward)

    assert sum(normalized_rows) <= sum(len(embedding) for embedding in embeddings)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"directions": ALL_DIRECTIONS},
        {"directions": ("query_to_doc", "doc_to_query"), "partition_mode": "per_direction"},
        {"hardness_mode": "in_batch_negatives", "hardness_strength": 5.0},
        {"hardness_mode": "hard_negatives", "hardness_strength": 5.0},
        {"hardness_mode": "all_negatives", "hardness_strength": 5.0},
    ],
)
@pytest.mark.parametrize("mini_batch_size", [2, 20])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cached_cosine_training_matches_uncached(config, mini_batch_size, dtype):
    torch.manual_seed(92)
    model = TinyEncoder().to(dtype)
    reference = deepcopy(model)
    features = [{"values": torch.randn(9, 5, dtype=dtype)} for _ in range(3)]
    labels = torch.zeros(9, dtype=torch.long)

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size, **config)(features, labels) / 4
    expected = MultipleNegativesRankingLoss(reference, **config)(features, labels) / 4
    loss.backward()
    expected.backward()

    tolerance = 1e-10 if dtype == torch.float64 else 1e-5
    torch.testing.assert_close(loss, expected, rtol=tolerance, atol=tolerance)
    for actual_parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
        assert actual_parameter.grad is not None
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_backward", [False, True])
@pytest.mark.parametrize("frozen_column", [None, 0, 1])
def test_cached_cosine_embedding_gradients(dtype, with_backward, frozen_column):
    embeddings = _embeddings(dtype)
    if dtype in (torch.float32, torch.float64):
        with torch.no_grad():
            for embedding in embeddings:
                embedding[0] = 0
                embedding[1] *= 1e-13
    if frozen_column is not None:
        embeddings[frozen_column].requires_grad_(False)
    chunks = [
        [chunk.detach().requires_grad_(embedding.requires_grad) for chunk in embedding.split(4)]
        for embedding in embeddings
    ]
    reference = [x.detach().clone().requires_grad_(x.requires_grad) for x in embeddings]
    model = TinyEncoder()
    cached = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2, directions=ALL_DIRECTIONS)
    plain = MultipleNegativesRankingLoss(model, directions=ALL_DIRECTIONS)

    actual = cached.calculate_loss(chunks, with_backward=with_backward)
    expected = plain.compute_loss_from_embeddings(reference, None)
    if not with_backward:
        actual.backward()
    expected.backward()

    tolerance = 1e-10 if dtype == torch.float64 else 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for column, reference_column in zip(chunks, reference):
        if not reference_column.requires_grad:
            assert all(chunk.grad is None for chunk in column)
            continue
        gradients = torch.cat([chunk.grad for chunk in column])
        assert torch.isfinite(gradients).all()
        torch.testing.assert_close(gradients, reference_column.grad, rtol=tolerance, atol=tolerance)


def test_cached_cosine_no_grad_and_custom_similarity():
    def custom_similarity(queries, candidates):
        return cos_sim(queries, candidates) * 0.7 + 0.2

    for similarity_fct in (cos_sim, custom_similarity):
        embeddings = _embeddings()
        model = TinyEncoder()
        config = {"similarity_fct": similarity_fct, "directions": ALL_DIRECTIONS}
        with torch.no_grad():
            actual = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2, **config).calculate_loss(
                _chunks(embeddings)
            )
            expected = MultipleNegativesRankingLoss(model, **config).compute_loss_from_embeddings(embeddings, None)
        assert not actual.requires_grad
        torch.testing.assert_close(actual, expected)


@skip_bfloat16_cpu_crash
@pytest.mark.parametrize("output_dtype", [None, torch.float32])
def test_cached_cosine_autocast_matches_unnormalized_cache(output_dtype):
    torch.manual_seed(94)
    model = TinyEncoder(output_dtype)
    reference = deepcopy(model)
    features = [{"values": torch.randn(9, 5)} for _ in range(3)]
    cached = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2, directions=ALL_DIRECTIONS)

    # A wrapper preserves the original per-chunk cos_sim path, including its
    # promotion and autocast behavior, for an independent gradient comparison.
    def reference_similarity(queries, candidates):
        return cos_sim(queries, candidates)

    reference_loss = CachedMultipleNegativesRankingLoss(
        reference, mini_batch_size=2, directions=ALL_DIRECTIONS, similarity_fct=reference_similarity
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = cached(features, None)
        expected = reference_loss(features, None)
        assert actual.dtype == expected.dtype == torch.bfloat16
    actual.backward()
    expected.backward()
    tolerance = 1e-5 if output_dtype == torch.float32 else 0
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=tolerance, atol=tolerance)


def _distributed_cosine_worker(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=init_method, timeout=timedelta(seconds=45))
    try:
        for config in (
            {},
            {"directions": ALL_DIRECTIONS, "hardness_mode": "all_negatives", "hardness_strength": 3.0},
            {"directions": ("query_to_doc", "doc_to_query"), "partition_mode": "per_direction"},
        ):
            embeddings = _embeddings(torch.float64, rank)
            chunks = _chunks(embeddings, 4)
            reference = [x.detach().clone().requires_grad_() for x in embeddings]
            model = TinyEncoder()
            cached = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2, gather_across_devices=True, **config)
            plain = MultipleNegativesRankingLoss(model, gather_across_devices=True, **config)
            actual = cached.calculate_loss(chunks, with_backward=True)
            expected = plain.compute_loss_from_embeddings(reference, None)
            expected.backward()
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
            for column, reference_column in zip(chunks, reference):
                torch.testing.assert_close(
                    torch.cat([chunk.grad for chunk in column]), reference_column.grad, rtol=1e-10, atol=1e-10
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_cached_cosine_gathered_gradients(tmp_path):
    mp.spawn(_distributed_cosine_worker, args=(f"file://{tmp_path / 'gloo-init'}",), nprocs=2, join=True)
