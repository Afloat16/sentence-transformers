from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding


def local_tokenizer(has_unknown_token):
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "apple": 2, "pear": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]" if has_unknown_token else None,
        pad_token="[PAD]",
        model_max_length=32,
    )


@pytest.mark.parametrize("has_unknown_token", [False, True])
@pytest.mark.parametrize("explicit_unknown_weight", [False, True])
@pytest.mark.parametrize("reverse_entries", [False, True])
def test_idf_import_does_not_alias_out_of_vocabulary_tokens(
    tmp_path, has_unknown_token, explicit_unknown_weight, reverse_entries
):
    tokenizer = local_tokenizer(has_unknown_token)
    entries = [("apple", 2.0), ("pear", 4.0)]
    if explicit_unknown_weight:
        entries.append(("[UNK]", 0.5))
    entries.extend([("outside-vocabulary", 100.0), ("another-missing-token", 200.0)])
    if reverse_entries:
        entries.reverse()
    path = tmp_path / "idf.json"
    path.write_text(json.dumps(dict(entries)), encoding="utf-8")
    module = SparseStaticEmbedding.from_json(str(path), tokenizer, frozen=True)
    model = SparseEncoder(modules=[module], device="cpu")
    unknown_weight = 0.5 if explicit_unknown_weight else 0.0
    expected = torch.tensor([[0, 0, 2, 0], [0, 0, 0, 4], [0, unknown_weight, 0, 0]], dtype=torch.float32)
    inputs = ["apple", "pear", "mango"]
    embeddings = model.encode_query(inputs, convert_to_sparse_tensor=False)
    torch.testing.assert_close(embeddings, expected)
    # Actual sparse retrieval scores must reflect only the configured in-vocabulary IDF values.
    documents = model.encode_document(["apple pear", "pear"], convert_to_sparse_tensor=False)
    torch.testing.assert_close(
        model.similarity(embeddings, documents), torch.tensor([[4, 0], [16, 16], [0, 0]]).float()
    )

    saved = tmp_path / "saved"
    model.save_pretrained(saved)
    restored = SparseEncoder(str(saved), device="cpu")
    torch.testing.assert_close(restored.encode_query(inputs, convert_to_sparse_tensor=False), expected)
    assert not restored[0].weight.requires_grad


@pytest.mark.parametrize("mapping", [{}, {"outside-vocabulary": 100.0}])
def test_idf_without_matching_tokens_produces_zero_embeddings(tmp_path, mapping):
    path = tmp_path / "idf.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    module = SparseStaticEmbedding.from_json(str(path), local_tokenizer(False), frozen=True)
    model = SparseEncoder(modules=[module], device="cpu")
    embeddings = model.encode_query(["apple pear", "mango"], convert_to_sparse_tensor=False)
    torch.testing.assert_close(embeddings, torch.zeros(2, 4))
