from __future__ import annotations

import gzip

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("separator", [" ", "::"])
@pytest.mark.parametrize("header", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("limit", [None, 1])
def test_word_embedding_text_loading_preserves_weights_and_vocabulary(
    tmp_path, compressed: bool, separator: str, header: bool, dtype: torch.dtype, limit: int | None
) -> None:
    rows = [
        ["apple", "0.125", "-0.25", "1.000000059604644775390626", "7.25"],
        ["skip_bad_width", "1", "2", "3"],
        ["βeta", "-0.0", "3.5", "2.25", "1e-40"],
    ]
    text = (separator.join(["2", "4"]) + "\n" if header else "") + "\n".join(separator.join(row) for row in rows)
    path = tmp_path / ("vectors.txt.gz" if compressed else "vectors.txt")
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    tokenizer = WhitespaceTokenizer(stop_words=[], do_lower_case=False)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        words = WordEmbeddings.from_text_file(
            str(path), tokenizer=tokenizer, item_separator=separator, max_vocab_size=limit, update_embeddings=True
        )
        valid_rows = [rows[0]] if limit else [rows[0], rows[2]]
        expected = torch.tensor([[0.0] * 4, *[[float(value) for value in row[1:]] for row in valid_rows]])
        torch.testing.assert_close(words.emb_layer.weight, expected, rtol=0, atol=0)
        assert words.tokenizer is tokenizer
        assert list(tokenizer.get_vocab()) == ["PADDING_TOKEN", *[row[0] for row in valid_rows]]
        assert words.emb_layer.weight.requires_grad
        model = SentenceTransformer(modules=[words, Pooling(4, "mean")])
        embedded = model.encode(" ".join(row[0] for row in valid_rows), convert_to_tensor=True)
        torch.testing.assert_close(embedded, expected[1:].mean(dim=0), rtol=0, atol=0)
    finally:
        torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_word_embedding_text_loading_preserves_numeric_formats(tmp_path, dtype: torch.dtype) -> None:
    numbers = ["1_000", "1.000000059604644775390626", "1e-200", "-0.0", "NaN", "INF", "-Infinity"]
    path = tmp_path / "numbers.txt"
    path.write_text("numeric " + " ".join(numbers), encoding="utf-8")
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        words = WordEmbeddings.from_text_file(str(path))
        expected = torch.tensor([float(value) for value in numbers], dtype=dtype)
        torch.testing.assert_close(words.emb_layer.weight[1], expected, rtol=0, atol=0, equal_nan=True)
        assert torch.signbit(words.emb_layer.weight[1, 3])
    finally:
        torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("text", ["word 0 invalid", "word 1 2\nsecond 3 4invalid"])
def test_word_embedding_text_loading_rejects_invalid_numbers(tmp_path, text: str) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        WordEmbeddings.from_text_file(str(path))


@pytest.mark.parametrize("text", ["", "100 300\n"])
def test_word_embedding_text_loading_empty_inputs_keep_error(tmp_path, text: str) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        WordEmbeddings.from_text_file(str(path))
