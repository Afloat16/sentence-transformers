from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.multiprocessing.reductions import StorageWeakRef
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from sentence_transformers import CrossEncoder
from sentence_transformers.base.modules import Transformer
from sentence_transformers.cross_encoder.modules import LogitScore


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_single_logit_owns_compact_storage_and_preserves_gradients(dtype, batch_size):
    logits = torch.randn(batch_size, 7, 128, dtype=dtype, requires_grad=True)
    scores = LogitScore(17)({"causal_logits": logits})["scores"]
    torch.testing.assert_close(scores, logits[:, -1, 17:18], rtol=0, atol=0)
    scores.sum().backward()
    expected_gradient = torch.zeros_like(logits)
    expected_gradient[:, -1, 17] = 1
    torch.testing.assert_close(logits.grad, expected_gradient, rtol=0, atol=0)
    assert scores.untyped_storage().nbytes() == scores.numel() * scores.element_size()


@pytest.fixture
def local_causal_transformer(tmp_path):
    torch.manual_seed(42)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8192,
            n_embd=8,
            n_layer=1,
            n_head=2,
            n_positions=32,
            pad_token_id=0,
            eos_token_id=0,
            bos_token_id=0,
        )
    ).save_pretrained(tmp_path)
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "yes": 2, "no": 3, "a": 4, "b": 5}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", model_max_length=32
    ).save_pretrained(tmp_path)
    return Transformer(str(tmp_path), transformer_task="text-generation")


@pytest.mark.parametrize("false_token_id", [None, 3])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_raw_cross_encoder_predictions_do_not_retain_prior_vocabulary_logits(
    local_causal_transformer, false_token_id, batch_size
):
    scoring = LogitScore(2, false_token_id)
    model = CrossEncoder(
        modules=[local_causal_transformer, scoring], activation_fn=torch.nn.Identity(), device="cpu"
    ).eval()
    pairs = [("a", "b"), ("b", "a"), ("a", "a"), ("b", "b")] * 8
    with torch.inference_mode():
        logits = local_causal_transformer(model.preprocess(pairs))["causal_logits"][:, -1]
        expected = logits[:, 2].clone()
        if false_token_id is not None:
            expected -= logits[:, false_token_id]
        del logits

    roots = []
    retained_bytes = []

    def observe_storage(module, args, output):
        # Observe the real model allocations without keeping them alive ourselves.
        storage = output["causal_logits"].untyped_storage()
        roots.append((StorageWeakRef(storage), storage.nbytes()))
        retained_bytes.append(sum(size for ref, size in roots if not ref.expired()))

    handle = scoring.register_forward_hook(observe_storage)
    try:
        actual = model.predict(pairs, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    # The current forward and the previous output dict can coexist. Older logits
    # must not be kept by the per-pair scores collected across the entire corpus.
    assert max(retained_bytes) <= 2 * batch_size * 8192 * 4
    assert all(ref.expired() for ref, _ in roots)
