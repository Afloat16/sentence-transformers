from __future__ import annotations

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings, WordWeights
from sentence_transformers.sentence_transformer.modules.tokenizer import PhraseTokenizer, WhitespaceTokenizer


@pytest.mark.parametrize("tokenizer_type", [WhitespaceTokenizer, PhraseTokenizer])
@pytest.mark.parametrize(
    "vocab_type", [list, iter, lambda words: (word for word in words)], ids=["list", "iterator", "generator"]
)
@pytest.mark.parametrize("replace_vocab", [False, True])
def test_iterable_vocab_keeps_token_ids_phrases_and_weights(tmp_path, tokenizer_type, vocab_type, replace_vocab):
    if tokenizer_type is PhraseTokenizer:
        pytest.importorskip("nltk")

    vocab = ["PADDING_TOKEN", "New_York", "glows"]
    if replace_vocab:
        tokenizer = tokenizer_type(["obsolete"], stop_words=[])
        tokenizer.set_vocab(vocab_type(vocab))
    else:
        tokenizer = tokenizer_type(vocab_type(vocab), stop_words=[])

    model = SentenceTransformer(
        modules=[
            WordEmbeddings(tokenizer, torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 6.0]])),
            WordWeights(tokenizer.get_vocab(), {"New_York": 2.0, "glows": 1.0}),
            Pooling(2, "mean"),
        ],
        device="cpu",
    )
    text = "New York glows." if tokenizer_type is PhraseTokenizer else "New_York glows."
    assert tokenizer.tokenize(text) == [1, 2]
    assert tokenizer.tokenize("obsolete") == []
    assert list(tokenizer.get_vocab()) == vocab
    assert list(tokenizer.get_vocab()) == vocab

    expected = torch.tensor([[2.0, 2.0], [0.0, 0.0]])
    torch.testing.assert_close(model.encode([text, "unknown"], convert_to_tensor=True), expected)
    model.save(str(tmp_path))
    loaded = SentenceTransformer(str(tmp_path), device="cpu")
    assert list(loaded.tokenizer.get_vocab()) == vocab
    torch.testing.assert_close(loaded.encode([text, "unknown"], convert_to_tensor=True), expected)
