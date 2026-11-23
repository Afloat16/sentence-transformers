from __future__ import annotations

import pytest
import torch

from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling


def storage_bytes(tensors):
    return sum({tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in tensors}.values())


def logical_bytes(tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def make_documents(lengths, dtype=torch.float32, dim=8):
    generator = torch.Generator().manual_seed(29)
    return [
        torch.nn.functional.normalize(torch.randn(length, dim, generator=generator, dtype=dtype), dim=-1)
        for length in lengths
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "options", [{"pool_factor": 2}, {"max_tokens": 5}, {"distance_threshold": 0.2}, {"pool_factor": 1}]
)
def test_ragged_pooled_outputs_do_not_retain_padding(dtype, options):
    documents = make_documents([33, 0, 3, 1, 15], dtype)
    original = [document.clone() for document in documents]
    pooler = HierarchicalTokenPooling(use_kernels=False, **options)
    with torch.no_grad():
        actual = pooler.pool(documents)
        expected = [pooler.pool_one(document) for document in documents]
    for pooled, reference, document, before in zip(actual, expected, documents, original):
        torch.testing.assert_close(pooled, reference, rtol=0, atol=0)
        torch.testing.assert_close(document, before, rtol=0, atol=0)
        assert pooled.dtype == dtype
    assert storage_bytes(actual) == logical_bytes(actual)


@pytest.mark.parametrize("normalize", [False, True])
def test_uniform_outputs_keep_compact_storage(normalize):
    documents = make_documents([13] * 4)
    pooler = HierarchicalTokenPooling(pool_factor=2, normalize_embeddings=normalize, use_kernels=False)
    with torch.no_grad():
        actual = pooler.pool(documents)
    assert storage_bytes(actual) == logical_bytes(actual)
    for pooled, document in zip(actual, documents):
        torch.testing.assert_close(pooled, pooler.pool_one(document), rtol=0, atol=0)


def test_empty_outputs_have_no_retained_storage():
    documents = make_documents([0, 0, 0])
    actual = HierarchicalTokenPooling(pool_factor=2, use_kernels=False).pool(documents)
    assert [tuple(row.shape) for row in actual] == [(0, 8)] * 3
    assert storage_bytes(actual) == 0


def test_heterogeneous_groups_stay_compact_and_in_order():
    documents = make_documents([33, 3]) + make_documents([17, 1], torch.float64, dim=4) + make_documents([21, 2])
    pooler = HierarchicalTokenPooling(pool_factor=2, use_kernels=False)
    with torch.no_grad():
        actual = pooler.pool(documents)
    assert storage_bytes(actual) == logical_bytes(actual)
    for pooled, document in zip(actual, documents):
        torch.testing.assert_close(pooled, pooler.pool_one(document), rtol=0, atol=0)
        assert pooled.dtype == document.dtype


def test_pretrained_encoded_pooled_documents_do_not_retain_padding(mve_bert_tiny_model):
    texts = ["A document " + "with many words about science " * 20, "A sentence.", "Text.", "A short phrase."]
    pooler = HierarchicalTokenPooling(pool_factor=2, use_kernels=False)
    model = mve_bert_tiny_model
    actual = model.encode_document(texts, batch_size=4, token_pooling=pooler, convert_to_numpy=False)
    assert storage_bytes(actual) == logical_bytes(actual)
    unpooled = model.encode_document(texts, batch_size=4, convert_to_numpy=False)
    with torch.no_grad():
        expected = [pooler.pool_one(document) for document in unpooled]
    for pooled, reference in zip(actual, expected):
        torch.testing.assert_close(pooled, reference, rtol=0, atol=0)
