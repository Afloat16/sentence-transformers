from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sentence_transformers.util.tensor import _convert_to_tensor


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["rows", "columns", "rotated"])
@pytest.mark.parametrize("read_only", [False, True])
def test_negative_stride_conversion_preserves_values_and_does_not_alias(dtype, layout, read_only):
    original = np.arange(24, dtype=dtype).reshape(4, 6)
    array = original[::-1] if layout == "rows" else original[:, ::-1] if layout == "columns" else np.rot90(original)
    if read_only:
        array.setflags(write=False)
    before = array.copy()
    actual = _convert_to_tensor(array)
    np.testing.assert_array_equal(actual.numpy(), before)
    assert actual.dtype == torch.from_numpy(before).dtype
    assert not np.shares_memory(actual.numpy(), original)
    actual.fill_(9)
    np.testing.assert_array_equal(array, before)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("score_function", [util.cos_sim, util.dot_score])
@pytest.mark.parametrize("axis", [0, 1])
def test_semantic_search_accepts_reversed_numpy_views(dtype, score_function, axis):
    generator = np.random.default_rng(27)
    queries = np.flip(generator.normal(size=(3, 5)).astype(dtype), axis=axis)
    corpus = np.flip(generator.normal(size=(8, 5)).astype(dtype), axis=axis)
    queries.setflags(write=False)
    expected_queries, expected_corpus = queries.astype(np.float64), corpus.astype(np.float64)
    if score_function is util.cos_sim:
        expected_queries /= np.linalg.norm(expected_queries, axis=1, keepdims=True)
        expected_corpus /= np.linalg.norm(expected_corpus, axis=1, keepdims=True)
    scores = expected_queries @ expected_corpus.T
    actual = util.semantic_search(
        queries, corpus, top_k=3, query_chunk_size=1, corpus_chunk_size=2, score_function=score_function
    )
    for hits, row in zip(actual, scores):
        indices = row.argsort()[::-1][:3]
        assert [hit["corpus_id"] for hit in hits] == indices.tolist()
        np.testing.assert_allclose([hit["score"] for hit in hits], row[indices], rtol=3e-6, atol=2e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_maxsim_accepts_lists_of_reversed_numpy_token_embeddings(dtype):
    generator = np.random.default_rng(31)
    queries = [generator.normal(size=(length, 4)).astype(dtype)[::-1] for length in [2, 5]]
    documents = [generator.normal(size=(length, 4)).astype(dtype)[:, ::-1] for length in [3, 4]]
    expected = [
        [
            math.fsum(max(math.fsum(float(q[k]) * float(d[k]) for k in range(4)) for d in document) for q in query)
            for document in documents
        ]
        for query in queries
    ]
    actual = util.maxsim(queries, documents, chunk_elements=20)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=3e-6, atol=2e-7)


def test_reversed_offline_encoder_output_can_be_searched():
    tokenizer = WhitespaceTokenizer(vocab=["[PAD]", "alpha", "beta", "gamma"], stop_words=[])
    words = WordEmbeddings(tokenizer, np.eye(4, dtype=np.float32))
    model = SentenceTransformer(modules=[words, Pooling(4, "mean")])
    corpus = model.encode(["alpha", "beta", "gamma"], convert_to_numpy=True)[::-1]
    query = model.encode(["alpha"], convert_to_numpy=True)
    assert util.semantic_search(query, corpus, top_k=1)[0][0]["corpus_id"] == 2


def test_positive_stride_writable_arrays_keep_zero_copy_conversion():
    array = np.arange(30, dtype=np.float32).reshape(5, 6)[:, ::2]
    actual = _convert_to_tensor(array)
    assert actual.data_ptr() == array.__array_interface__["data"][0]
    np.testing.assert_array_equal(actual.numpy(), array)
