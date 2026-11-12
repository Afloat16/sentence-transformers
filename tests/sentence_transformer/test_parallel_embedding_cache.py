from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.datasets import ParallelSentencesDataset
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer


@pytest.fixture
def teacher() -> SentenceTransformer:
    tokenizer = WhitespaceTokenizer(["[PAD]", "alpha", "beta", "gamma"], stop_words=[])
    return SentenceTransformer(
        modules=[WordEmbeddings(tokenizer, torch.eye(4)), Pooling(4, pooling_mode="mean")], device="cpu"
    )


def test_cache_encodes_each_new_sentence_once(teacher: SentenceTransformer) -> None:
    dataset = ParallelSentencesDataset(teacher, teacher)
    dataset.get_embeddings(["gamma"])
    sentences = ["beta", "alpha", "beta", "gamma", "alpha"]

    with patch.object(teacher, "encode", wraps=teacher.encode) as encode:
        embeddings = dataset.get_embeddings(sentences)
        assert encode.call_args.args[0] == ["beta", "alpha"]
        np.testing.assert_array_equal(embeddings, teacher.encode(sentences))

    with patch.object(teacher, "encode", wraps=teacher.encode) as encode:
        np.testing.assert_array_equal(dataset.get_embeddings(sentences), embeddings)
        encode.assert_not_called()


def test_weighted_dataset_generates_labels_without_reencoding_duplicates(teacher: SentenceTransformer) -> None:
    dataset = ParallelSentencesDataset(teacher, teacher)
    dataset.add_dataset([["alpha", "beta"]], weight=100)

    with patch.object(teacher, "encode", wraps=teacher.encode) as encode:
        dataset.generate_data()
        assert encode.call_args.args[0] == ["alpha"]

    assert len(dataset.cache) == 200
    for example in dataset.cache:
        np.testing.assert_array_equal(example.label, [0.0, 1.0, 0.0, 0.0])


def test_disabled_cache_preserves_input_batch(teacher: SentenceTransformer) -> None:
    dataset = ParallelSentencesDataset(teacher, teacher, use_embedding_cache=False)
    sentences = ["alpha", "beta", "alpha"]
    with patch.object(teacher, "encode", wraps=teacher.encode) as encode:
        embeddings = dataset.get_embeddings(sentences)
        assert encode.call_args.args[0] == sentences
    assert not dataset.embedding_cache
    np.testing.assert_array_equal(embeddings, teacher.encode(sentences))
