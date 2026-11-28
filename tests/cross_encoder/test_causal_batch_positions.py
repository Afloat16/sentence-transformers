from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    PreTrainedTokenizerFast,
)

from sentence_transformers import CrossEncoder
from sentence_transformers.base.modules import Transformer
from sentence_transformers.cross_encoder.modules import LogitScore


@pytest.fixture(params=["gpt2", "opt", "llama"])
def local_causal_model(request, tmp_path):
    torch.manual_seed(42)
    common = {"vocab_size": 16, "pad_token_id": 0, "eos_token_id": 0, "bos_token_id": 0}
    if request.param == "gpt2":
        model = GPT2LMHeadModel(
            GPT2Config(
                **common,
                n_embd=16,
                n_layer=1,
                n_head=2,
                n_positions=64,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
            )
        )
    elif request.param == "opt":
        model = OPTForCausalLM(
            OPTConfig(
                **common,
                hidden_size=16,
                ffn_dim=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=64,
                dropout=0,
                attention_dropout=0,
            )
        )
    else:
        model = LlamaForCausalLM(
            LlamaConfig(
                **common,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=64,
            )
        )
    model.save_pretrained(tmp_path)
    tokenizer = Tokenizer(
        WordLevel({"[PAD]": 0, "[UNK]": 1, "yes": 2, "no": 3, "a": 4, "b": 5, "c": 6}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=64
    ).save_pretrained(tmp_path)
    return str(tmp_path)


def causal_encoder(path, use_chat_template=False):
    processor_kwargs = (
        {"chat_template": "{% for message in messages %}{{ message['content'] + ' ' }}{% endfor %}yes"}
        if use_chat_template
        else None
    )
    transformer = Transformer(path, transformer_task="text-generation", processor_kwargs=processor_kwargs)
    return CrossEncoder(
        modules=[transformer, LogitScore(2, 3)], activation_fn=torch.nn.Identity(), device="cpu"
    ).eval()


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("use_chat_template", [False, True])
def test_causal_scores_match_individual_inputs(local_causal_model, batch_size, use_chat_template):
    model = causal_encoder(local_causal_model, use_chat_template)
    pairs = [("a", "b"), ("a b c", "b c a b"), ("c", "a b"), ("a b", "c a b c a b")]
    inputs = (
        [[{"role": "user", "content": left}, {"role": "assistant", "content": right}] for left, right in pairs]
        if use_chat_template
        else pairs
    )
    individual = model.predict(inputs, batch_size=1, convert_to_tensor=True, show_progress_bar=False)
    batched = model.predict(inputs, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    torch.testing.assert_close(batched, individual, rtol=1e-5, atol=1e-6)


def test_causal_positions_survive_save_and_explicit_forward_override(local_causal_model, tmp_path):
    model = causal_encoder(local_causal_model)
    inputs = [("a", "b"), ("a b c", "b c a b")]
    individual = model.predict(inputs, batch_size=1, convert_to_tensor=True, show_progress_bar=False)
    destination = tmp_path / "cross-encoder"
    model.save(str(destination))
    restored = CrossEncoder(str(destination), device="cpu").eval()
    batched = restored.predict(inputs, batch_size=2, convert_to_tensor=True, show_progress_bar=False)
    torch.testing.assert_close(batched, individual, rtol=1e-5, atol=1e-6)

    transformer = restored[0]
    features = restored.preprocess(inputs)
    explicit_positions = torch.ones_like(features["input_ids"])
    with torch.inference_mode():
        expected = transformer.model(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            token_type_ids=features.get("token_type_ids"),
            position_ids=explicit_positions,
            logits_to_keep=1,
        ).logits
        actual = transformer(features, position_ids=explicit_positions)["causal_logits"]
    torch.testing.assert_close(actual, expected)


def test_feature_extraction_does_not_add_causal_positions(local_causal_model):
    transformer = Transformer(local_causal_model, transformer_task="feature-extraction")
    features = transformer.preprocess(["a", "a b c"])
    assert "position_ids" not in features
