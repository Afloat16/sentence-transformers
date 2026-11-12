from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from sentence_transformers import CrossEncoder
from sentence_transformers.base.modules import Transformer
from sentence_transformers.cross_encoder.modules import LogitScore


def save_causal_checkpoint(path, missing_tokens=()):
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
            pad_token_id=0,
            bos_token_id=0,
            eos_token_id=0,
        )
    ).save_pretrained(path)
    vocab = {"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3, "yes": 4, "no": 5, "y": 6, "n": 7}
    for token in missing_tokens:
        del vocab[token]
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    ).save_pretrained(path)


@pytest.mark.parametrize("missing_tokens", [("yes",), ("no",), ("yes", "no")])
def test_causal_checkpoint_rejects_unknown_default_scoring_tokens(tmp_path, missing_tokens):
    save_causal_checkpoint(tmp_path, missing_tokens)
    with pytest.raises(ValueError, match="tokenizer does not have 'yes' and/or 'no' tokens"):
        CrossEncoder(str(tmp_path), local_files_only=True, device="cpu")


def test_causal_checkpoint_with_known_scoring_tokens(tmp_path):
    save_causal_checkpoint(tmp_path)
    model = CrossEncoder(str(tmp_path), local_files_only=True, device="cpu")
    assert model[1].true_token_id == 4
    assert model[1].false_token_id == 5
    assert torch.isfinite(model.predict([("a", "b"), ("b", "a")], convert_to_tensor=True)).all()


def test_causal_checkpoint_accepts_explicit_scoring_tokens(tmp_path):
    save_causal_checkpoint(tmp_path, ("yes", "no"))
    transformer = Transformer(str(tmp_path), transformer_task="text-generation")
    model = CrossEncoder(modules=[transformer, LogitScore(6, 7)], device="cpu")
    assert torch.isfinite(model.predict([("a", "b"), ("b", "a")], convert_to_tensor=True)).all()
