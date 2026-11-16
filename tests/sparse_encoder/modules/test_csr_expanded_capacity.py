from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sentence_transformers import SparseEncoder
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sentence_transformers.sparse_encoder.losses import CSRLoss
from sentence_transformers.sparse_encoder.losses.csr import CSRReconstructionLoss
from sentence_transformers.sparse_encoder.modules import SparseAutoEncoder


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("k_aux", [0, 2])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("override", [False, True])
def test_csr_expanded_reconstruction_with_valid_k(
    k: int, k_aux: int, normalize: bool, dtype: torch.dtype, override: bool
) -> None:
    torch.manual_seed(17)
    module = SparseAutoEncoder(3, hidden_dim=8, k=1 if override else k, k_aux=k_aux, normalize=normalize).to(dtype)
    with torch.no_grad():
        module.pre_bias.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=dtype))
        module.latent_bias.copy_(torch.linspace(-0.2, 0.4, 8, dtype=dtype))
    inputs = torch.randn(4, 3, dtype=dtype, requires_grad=True)
    outputs = module({"sentence_embedding": inputs}, max_active_dims=k if override else None)

    # Construct the expanded reconstruction directly. An expanded budget that
    # exceeds the latent width retains every positive latent, without padding.
    if normalize:
        mean = inputs.mean(dim=-1, keepdim=True)
        std = inputs.std(dim=-1, keepdim=True)
        prepared = (inputs - mean) / (std + 1e-5)
    else:
        prepared = inputs
    encoded = F.linear(prepared - module.pre_bias, module.encoder.weight, module.latent_bias)
    if 4 * k < module.hidden_dim:
        order = encoded.argsort(dim=-1, descending=True)
        mask = torch.zeros_like(encoded, dtype=torch.bool).scatter_(-1, order[:, : 4 * k], True)
        expected_latents = encoded.masked_fill(~mask, 0).relu()
    else:
        expected_latents = encoded.relu()
    expected_reconstruction = F.linear(expected_latents, module.encoder.weight.T) + module.pre_bias
    if normalize:
        expected_reconstruction = expected_reconstruction * std + mean

    torch.testing.assert_close(outputs["sentence_embedding_encoded_4k"], expected_latents)
    torch.testing.assert_close(outputs["decoded_embedding_4k"], expected_reconstruction)
    assert torch.all(torch.count_nonzero(outputs["sentence_embedding"], dim=-1) <= k)
    actual_loss = CSRReconstructionLoss(None).compute_loss_from_embeddings([outputs])["reconstruction_loss_4k"]
    expected_loss = F.mse_loss(expected_reconstruction, inputs) / 8
    torch.testing.assert_close(actual_loss, expected_loss)
    targets = [inputs, *module.parameters()]
    actual_gradients = torch.autograd.grad(actual_loss, targets, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_loss, targets)
    for actual, expected in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("override", [False, True])
def test_csr_still_rejects_main_budget_exceeding_latent_width(override: bool) -> None:
    module = SparseAutoEncoder(3, hidden_dim=8, k=1 if override else 9, k_aux=0)
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        module({"sentence_embedding": torch.randn(2, 3)}, max_active_dims=9 if override else None)


@pytest.mark.parametrize("k_aux", [0, 2])
def test_csr_loss_trains_with_expanded_budget_larger_than_hidden_dim(k_aux: int) -> None:
    torch.manual_seed(23)
    tokenizer = WhitespaceTokenizer(vocab=["[PAD]", "red", "green", "blue"], stop_words=[])
    embeddings = WordEmbeddings(tokenizer, torch.randn(4, 3), update_embeddings=True)
    autoencoder = SparseAutoEncoder(3, hidden_dim=8, k=3, k_aux=k_aux)
    model = SparseEncoder(modules=[embeddings, Pooling(3, "mean"), autoencoder])
    loss = CSRLoss(model)
    features = [model.preprocess(texts) for texts in (["red green", "blue"], ["green red", "blue green"])]
    losses = loss(features)
    total = torch.stack(list(losses.values())).sum()
    assert torch.isfinite(total)
    total.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert embeddings.emb_layer.weight.grad.abs().sum() > 0
    assert autoencoder.encoder.weight.grad.abs().sum() > 0
