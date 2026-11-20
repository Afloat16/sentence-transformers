from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules import Dense
from sentence_transformers.sentence_transformer.modules import Pooling, StaticEmbedding, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize("input_module", ["word", "static"])
@pytest.mark.parametrize(
    ("first_safe", "second_safe"),
    [(True, False), (False, True), (True, True), (False, False)],
    ids=["safe-to-bin", "bin-to-safe", "safe-to-safe", "bin-to-bin"],
)
def test_resaving_trained_model_uses_latest_weights(tmp_path: Path, input_module, first_safe, second_safe):
    vocab = ["[PAD]", "cat", "dog", "bird"]
    weights = torch.tensor([[0.0, 0.0], [1.0, 0.5], [-1.0, 1.0], [0.5, -0.5]])
    if input_module == "word":
        words = WordEmbeddings(WhitespaceTokenizer(vocab, stop_words=[]), weights, update_embeddings=True)
        modules = [words, Pooling(2)]
    else:
        tokenizer = Tokenizer(WordLevel({word: index for index, word in enumerate(vocab)}))
        tokenizer.pre_tokenizer = Whitespace()
        modules = [StaticEmbedding(tokenizer, embedding_weights=weights)]
    modules.append(Dense(2, 2, init_weight=torch.tensor([[0.5, -0.5], [0.25, 0.75]]), init_bias=torch.zeros(2)))
    model = SentenceTransformer(modules=modules, device="cpu")
    texts = ["cat bird", "dog", "cat dog"]
    original = model.encode(texts, convert_to_tensor=True)
    model.save_pretrained(str(tmp_path), safe_serialization=first_safe, create_model_card=False)

    # Additional assets beside the two standard weight files must survive a format change.
    weight_dirs = {path.parent for path in tmp_path.rglob("model.safetensors")} | {
        path.parent for path in tmp_path.rglob("pytorch_model.bin")
    }
    for directory in weight_dirs:
        (directory / "custom.safetensors").write_bytes(b"custom model asset")
        (directory / "training.bin").write_bytes(b"training state")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    model.train()
    loss = model(model.preprocess(texts))["sentence_embedding"].square().mean()
    loss.backward()
    optimizer.step()
    expected = model.encode(texts, convert_to_tensor=True)
    assert not torch.allclose(expected, original)

    model.save_pretrained(str(tmp_path), safe_serialization=second_safe, create_model_card=False)
    restored = SentenceTransformer(str(tmp_path), device="cpu", local_files_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(restored.encode(texts, convert_to_tensor=True), expected, rtol=0, atol=0)

    expected_filename = "model.safetensors" if second_safe else "pytorch_model.bin"
    stale_filename = "pytorch_model.bin" if second_safe else "model.safetensors"
    for directory in weight_dirs:
        assert (directory / expected_filename).is_file()
        assert not (directory / stale_filename).exists()
        assert (directory / "custom.safetensors").read_bytes() == b"custom model asset"
        assert (directory / "training.bin").read_bytes() == b"training state"


@pytest.mark.parametrize("safe_serialization", [True, False])
def test_failed_format_change_preserves_previous_weights(tmp_path: Path, monkeypatch, safe_serialization):
    model = Dense(2, 2)
    model.save(str(tmp_path), safe_serialization=not safe_serialization)
    old_filename = "pytorch_model.bin" if safe_serialization else "model.safetensors"
    old_bytes = (tmp_path / old_filename).read_bytes()
    old_state = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        model.linear.weight.add_(1)

    def fail_save(*args, **kwargs):
        raise OSError("cannot write new checkpoint")

    if safe_serialization:
        monkeypatch.setattr("sentence_transformers.base.modules.module.save_safetensors_model", fail_save)
    else:
        monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="cannot write new checkpoint"):
        model.save(str(tmp_path), safe_serialization=safe_serialization)

    assert (tmp_path / old_filename).read_bytes() == old_bytes
    restored = Dense.load(str(tmp_path), local_files_only=True)
    for key, value in old_state.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
