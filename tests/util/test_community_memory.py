from __future__ import annotations

import tracemalloc

import pytest
import torch

from sentence_transformers.util import community_detection, normalize_embeddings


def _dense_communities(embeddings, threshold, minimum, batch_size):
    embeddings = normalize_embeddings(embeddings)
    threshold = torch.tensor(threshold, device=embeddings.device)
    candidates = []
    for start in range(0, len(embeddings), batch_size):
        scores = embeddings[start : start + batch_size] @ embeddings.T
        for offset, row in enumerate(scores):
            members = (row >= threshold).nonzero(as_tuple=True)[0].tolist()
            if len(members) >= minimum:
                candidates.append((start + offset, members))
    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    used = set()
    result = []
    for center, members in candidates:
        if center in used:
            continue
        remaining = [member for member in members if member not in used]
        if len(remaining) >= minimum:
            result.append([center] + [member for member in remaining if member != center])
            used.update(result[-1])
    return sorted(result, key=len, reverse=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [2, 7, 31])
@pytest.mark.parametrize("threshold_kind", ["fixed", "boundary"])
def test_dense_candidate_graph_matches_exhaustive_membership(dtype, batch_size, threshold_kind):
    generator = torch.Generator().manual_seed(17)
    centers = torch.randn(7, 8, generator=generator)
    embeddings = centers[torch.arange(157) % 7] + 0.2 * torch.randn(157, 8, generator=generator)
    embeddings = embeddings.to(dtype)
    threshold = 0.9
    if threshold_kind == "boundary":
        normalized = normalize_embeddings(embeddings)
        threshold = float((normalized[:batch_size] @ normalized.T)[0, 7])
    original = embeddings.clone()
    actual = community_detection(embeddings, threshold=threshold, min_community_size=3, batch_size=batch_size)
    expected = _dense_communities(embeddings, threshold, 3, batch_size)

    # topk has no fixed ordering for equal scores, but centers, membership, and
    # the stable ordering of equally sized communities must agree.
    assert [group[0] for group in actual] == [group[0] for group in expected]
    assert [sorted(group) for group in actual] == [sorted(group) for group in expected]
    torch.testing.assert_close(embeddings, original, rtol=0, atol=0)


def test_late_largest_community_is_reconstructed_after_member_cache_fills():
    embeddings = torch.eye(3).repeat_interleave(torch.tensor([51, 123, 200]), dim=0)
    actual = community_detection(embeddings, threshold=1.0, min_community_size=10, batch_size=2)
    assert [group[0] for group in actual] == [174, 51, 0]
    assert [sorted(group) for group in actual] == [list(range(174, 374)), list(range(51, 174)), list(range(51))]


def test_dense_candidate_graph_does_not_retain_quadratic_host_members():
    num_embeddings = 1536
    embeddings = torch.ones(num_embeddings, 16)
    tracemalloc.start()
    try:
        communities = community_detection(embeddings, threshold=1.0, min_community_size=10, batch_size=16)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(communities) == 1 and len(communities[0]) == num_embeddings
    # One uint32 array per center alone needs 4*N*N bytes on the original path.
    # Allow ample Python bookkeeping while requiring the dense graph to stay below N*N.
    assert peak < num_embeddings**2
