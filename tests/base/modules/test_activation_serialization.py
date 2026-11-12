from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.base.modules import Dense
from sentence_transformers.sentence_transformer.modules import StaticEmbedding


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "first": 2, "second": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


@pytest.mark.parametrize(
    "activation",
    [nn.LeakyReLU(0.2), nn.Softplus(beta=2), nn.Threshold(0.0, -0.4), nn.PReLU(3), nn.GELU("tanh"), nn.Tanh()],
)
def test_dense_activation_options_round_trip(tmp_path: Path, activation: nn.Module) -> None:
    embedding = StaticEmbedding(
        _tokenizer(),
        embedding_weights=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-2.0, -0.5, 1.5], [0.5, -1.5, -0.8]]),
    )
    dense = Dense(3, 3, bias=False, init_weight=torch.eye(3), activation_function=activation)
    if isinstance(activation, nn.PReLU):
        with torch.no_grad():
            activation.weight.copy_(torch.tensor([0.1, 0.2, 0.3]))
    model = SentenceTransformer(modules=[embedding, dense], device="cpu")
    texts = ["first", "second"]
    expected = model.encode(texts, convert_to_tensor=True)
    model.save_pretrained(str(tmp_path), create_model_card=False)

    restored = SentenceTransformer(str(tmp_path), local_files_only=True, device="cpu")
    torch.testing.assert_close(restored.encode(texts, convert_to_tensor=True), expected, rtol=0, atol=0)
    if isinstance(activation, nn.Tanh):
        config = json.loads((tmp_path / "1_Dense" / "config.json").read_text())
        assert "activation_function_kwargs" not in config


@pytest.fixture
def local_cross_encoder(tmp_path: Path) -> Path:
    torch.manual_seed(3)
    checkpoint = tmp_path / "backbone"
    PreTrainedTokenizerFast(
        tokenizer_object=_tokenizer(), pad_token="[PAD]", unk_token="[UNK]", model_max_length=32
    ).save_pretrained(checkpoint)
    backbone = BertForSequenceClassification(
        BertConfig(
            vocab_size=4,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=1,
        )
    )
    with torch.no_grad():
        backbone.classifier.bias.fill_(-0.6)
    backbone.save_pretrained(checkpoint)
    return checkpoint


@pytest.mark.parametrize(
    "activation", [nn.LeakyReLU(0.2), nn.Softplus(beta=2), nn.Threshold(0.0, -0.4), nn.GELU("tanh"), nn.Sigmoid()]
)
def test_cross_encoder_activation_options_round_trip(
    tmp_path: Path, local_cross_encoder: Path, activation: nn.Module
) -> None:
    model = CrossEncoder(str(local_cross_encoder), activation_fn=activation, device="cpu")
    pairs = [("first", "second"), ("second", "first")]
    expected = model.predict(pairs, convert_to_tensor=True)
    raw_scores = model.predict(pairs, activation_fn=nn.Identity(), convert_to_tensor=True)
    destination = tmp_path / "saved"
    model.save_pretrained(str(destination), create_model_card=False)

    restored = CrossEncoder(str(destination), local_files_only=True, device="cpu")
    torch.testing.assert_close(restored.predict(pairs, convert_to_tensor=True), expected, rtol=0, atol=0)
    overridden = CrossEncoder(str(destination), activation_fn=nn.Identity(), local_files_only=True, device="cpu")
    torch.testing.assert_close(overridden.predict(pairs, convert_to_tensor=True), raw_scores, rtol=0, atol=0)
    if isinstance(activation, nn.Sigmoid):
        config = json.loads((destination / "config_sentence_transformers.json").read_text())
        assert "activation_fn_kwargs" not in config


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_cross_encoder_activation_factory_requires_trust(
    tmp_path: Path, local_cross_encoder: Path, monkeypatch: pytest.MonkeyPatch, trust_remote_code: bool
) -> None:
    model = CrossEncoder(str(local_cross_encoder), device="cpu")
    destination = tmp_path / "saved"
    model.save_pretrained(str(destination), create_model_card=False)
    config_path = destination / "config_sentence_transformers.json"
    config = json.loads(config_path.read_text())
    config["activation_fn"] = "torch.serialization.clear_safe_globals"
    config["activation_fn_kwargs"] = {"sentinel": 7}
    config_path.write_text(json.dumps(config))
    factory = MagicMock(return_value=nn.Identity())
    monkeypatch.setattr(torch.serialization, "clear_safe_globals", factory, raising=False)

    if trust_remote_code:
        restored = CrossEncoder(str(destination), local_files_only=True, trust_remote_code=True, device="cpu")
        factory.assert_called_once_with(sentinel=7)
        assert isinstance(restored.activation_fn, nn.Identity)
    else:
        with pytest.raises(ValueError, match="must be a torch.nn.Module class"):
            CrossEncoder(str(destination), local_files_only=True, device="cpu")
        factory.assert_not_called()
