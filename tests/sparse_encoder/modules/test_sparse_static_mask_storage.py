from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding


def _tokenizer():
    vocab = {word: index for index, word in enumerate(["[PAD]", "[UNK]", "a", "b", "c", "d", "e"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", model_max_length=32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("mask_dtype", [torch.int64, torch.bool, torch.uint8])
@pytest.mark.parametrize("with_embedding", [False, True])
def test_sparse_static_presence_matches_dense_reference(dtype, mask_dtype, with_embedding):
    module = SparseStaticEmbedding(_tokenizer(), weight=torch.arange(7, dtype=dtype) / 4)
    input_ids = torch.tensor([[2, 3, 2, 0], [4, 5, 0, 0], [0, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=mask_dtype)
    features = dict(input_ids=input_ids, attention_mask=attention_mask)
    reference_weight = module.weight.detach().clone().requires_grad_()
    presence = torch.tensor([[0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=dtype)
    expected = presence * reference_weight
    if with_embedding:
        features["sentence_embedding"] = torch.arange(21, dtype=dtype).reshape(3, 7).requires_grad_()
        reference_embedding = features["sentence_embedding"].detach().clone().requires_grad_()
        expected = expected * reference_embedding
    actual = module(dict(features))["sentence_embedding"]
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(module.weight.grad, reference_weight.grad, rtol=0, atol=0)
    if with_embedding:
        torch.testing.assert_close(features["sentence_embedding"].grad, reference_embedding.grad, rtol=0, atol=0)


def test_trainable_presence_mask_uses_one_byte_per_entry():
    module = SparseStaticEmbedding(_tokenizer())
    features = module.preprocess(["a b", "c"])
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor
    ):
        embedding = module(features)["sentence_embedding"]
    masks = [tensor for tensor in saved if tensor.shape == (2, 7)]
    assert len(masks) == 1
    assert masks[0].untyped_storage().nbytes() == 14
    embedding.sum().backward()
    torch.testing.assert_close(module.weight.grad, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]))


def test_sparse_static_offline_encoder_training():
    module = SparseStaticEmbedding(_tokenizer())
    model = SparseEncoder(modules=[module])
    output = model(model.preprocess(["a b", "c"]))["sentence_embedding"]
    expected = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    torch.testing.assert_close(module.weight.grad, expected.sum(0))
