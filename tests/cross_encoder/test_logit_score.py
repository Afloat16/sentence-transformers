from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from sentence_transformers import CrossEncoder
from sentence_transformers.base.modules import Transformer
from sentence_transformers.cross_encoder.modules import LogitScore


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_log_odds_preserve_ranking_and_gradients(dtype):
    epsilon = 2**-10 if dtype == torch.bfloat16 else 2**-13
    logits = torch.tensor([[[2.0, -epsilon, 9.0]], [[2.0, epsilon, 9.0]]], dtype=dtype, requires_grad=True)
    scores = LogitScore(0, 1)({"causal_logits": logits})["scores"]
    expected_dtype = torch.float32 if torch.finfo(dtype).bits < 32 else dtype
    expected = logits[:, -1, :1].to(expected_dtype) - logits[:, -1, 1:2].to(expected_dtype)
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)
    assert scores[0] > scores[1]

    scores.sum().backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[1.0, -1.0, 0.0]]] * 2, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_single_logit_preserves_dtype(dtype):
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=dtype)
    scores = LogitScore(1)({"causal_logits": logits})["scores"]
    torch.testing.assert_close(scores, logits[:, -1, 1:2])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cross_encoder_causal_scores_match_float32_log_odds(tmp_path, dtype):
    torch.manual_seed(42)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=16,
            pad_token_id=0,
            eos_token_id=0,
            bos_token_id=0,
        )
    ).save_pretrained(tmp_path)
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3, "yes": 4, "no": 5}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    ).save_pretrained(tmp_path)
    transformer = Transformer(str(tmp_path), transformer_task="text-generation", model_kwargs={"dtype": dtype})
    model = CrossEncoder(
        modules=[transformer, LogitScore(4, 5)], activation_fn=torch.nn.Identity(), device="cpu"
    ).eval()
    pairs = [("a", "b"), ("b", "a"), ("a", "a"), ("b", "b")]
    with torch.inference_mode():
        logits = transformer(model.preprocess(pairs))["causal_logits"][:, -1]
        expected = logits[:, 4].float() - logits[:, 5].float()
        scores = model.predict(pairs, convert_to_tensor=True, show_progress_bar=False)
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)
