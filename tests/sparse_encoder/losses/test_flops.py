from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder, util
from sentence_transformers.sparse_encoder.losses import (
    CachedSpladeLoss,
    FlopsLoss,
    SparseMultipleNegativesRankingLoss,
    SpladeLoss,
)
from sentence_transformers.sparse_encoder.modules import SpladePooling, Transformer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("threshold", [None, 0, 30522])
def test_flops_loss_reduction_precision(dtype: torch.dtype, threshold: int | None) -> None:
    embeddings = torch.full((2, 30522), 2.0, dtype=dtype)
    embeddings[0, ::3] = 1.0
    embeddings.requires_grad_()
    reference_embeddings = embeddings.detach().to(torch.float64 if dtype == torch.float64 else torch.float32)
    reference_embeddings.requires_grad_()
    selected = reference_embeddings
    if threshold is not None:
        selected = selected * ((selected != 0).sum(1) > threshold).unsqueeze(1)
    reference = selected.mean(0).square().sum()

    loss = FlopsLoss(None, threshold=threshold).compute_loss_from_embeddings(embeddings)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, reference, check_dtype=False, rtol=1e-6, atol=0)
    assert loss.dtype == reference.dtype
    actual_gradient = torch.autograd.grad(loss, embeddings)[0]
    reference_gradient = torch.autograd.grad(reference, reference_embeddings)[0].to(dtype)
    torch.testing.assert_close(actual_gradient, reference_gradient)


@pytest.mark.parametrize("cached", [False, True])
def test_splade_sum_pooling_regularization_under_cpu_autocast(tmp_path, cached: bool) -> None:
    torch.manual_seed(1)
    vocab = {"[PAD]": 0, "[UNK]": 1, **{f"t{i}": i + 2 for i in range(1022)}}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=512
    ).save_pretrained(tmp_path)
    BertForMaskedLM(
        BertConfig(
            vocab_size=1024,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
    ).save_pretrained(tmp_path)
    model = SparseEncoder(
        modules=[Transformer(str(tmp_path), transformer_task="fill-mask"), SpladePooling(pooling_strategy="sum")],
        device="cpu",
    )
    model.train()
    texts = [" ".join(["t1", "t2", "t3", "t4"] * 64), " ".join(["t5", "t6", "t7", "t8"] * 64)]
    loss_cls = CachedSpladeLoss if cached else SpladeLoss
    loss = loss_cls(
        model,
        SparseMultipleNegativesRankingLoss(model, similarity_fct=util.cos_sim, scale=20),
        document_regularizer_weight=3e-5,
        query_regularizer_weight=5e-5,
        **({"mini_batch_size": 1} if cached else {}),
    )
    with torch.autocast("cpu", dtype=torch.float16):
        with torch.no_grad():
            embeddings = model(model.preprocess(texts))["sentence_embedding"]
            reference = embeddings.float().mean(0).square().sum()
        components = loss([model.preprocess(texts), model.preprocess(texts)], None)

    assert embeddings.dtype == torch.float16
    assert reference > torch.finfo(torch.float16).max
    assert all(torch.isfinite(value) for value in components.values())
    torch.testing.assert_close(components["document_regularizer_loss"], reference * 3e-5, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(components["query_regularizer_loss"], reference * 5e-5, rtol=1e-3, atol=1e-5)
    total = sum(components.values())
    assert torch.isfinite(total)
    total.backward()
    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0
