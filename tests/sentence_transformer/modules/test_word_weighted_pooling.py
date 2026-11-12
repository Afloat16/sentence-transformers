from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.modules import Pooling, WordWeights


@pytest.mark.parametrize("mode", ["mean", "mean_sqrt_len_tokens", "weightedmean"])
@pytest.mark.parametrize("include_prompt", [False, True])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_word_weighted_pooling_normalizes_only_used_tokens(mode: str, include_prompt: bool, padding_side: str) -> None:
    vocab = ["[PAD]", "prompt", "alpha", "beta"]
    weights = {"[PAD]": 37.0, "prompt": 10.0, "alpha": 2.0, "beta": 3.0}
    word_weights = WordWeights(vocab, weights).double()
    ids = [1, 2, 3, 0] if padding_side == "right" else [0, 1, 2, 3]
    mask = torch.tensor([[index != 0 for index in ids]])
    embeddings = torch.tensor([[[100.0, 100.0], [7.0, 9.0], [2.0, 0.0], [0.0, 3.0]][i] for i in ids])
    embeddings = embeddings.double().unsqueeze(0).requires_grad_()
    features = {
        "input_ids": torch.tensor([ids]),
        "attention_mask": mask,
        "token_embeddings": embeddings,
        "prompt_length": 1,
    }
    actual = Pooling(2, pooling_mode=mode, include_prompt=include_prompt)(word_weights(features))["sentence_embedding"]

    reference_embeddings = embeddings.detach().clone().requires_grad_()
    reference_weights = word_weights.emb_layer.weight.detach().clone().requires_grad_()
    terms, normalizers = [], []
    for position, word_id in enumerate(ids):
        if word_id == 0 or (word_id == 1 and not include_prompt):
            continue
        token_position = position + 1 - (padding_side == "left")
        weight = reference_weights[word_id, 0]
        if mode == "weightedmean":
            weight = weight * token_position
        terms.append(reference_embeddings[0, position] * weight)
        normalizers.append(weight)
    denominator = torch.stack(normalizers).sum()
    if mode == "mean_sqrt_len_tokens":
        denominator = denominator.sqrt()
    expected = torch.stack(terms).sum(0).unsqueeze(0) / denominator

    actual.backward(torch.ones_like(actual))
    expected.backward(torch.ones_like(expected))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(embeddings.grad, reference_embeddings.grad)
    torch.testing.assert_close(word_weights.emb_layer.weight.grad, reference_weights.grad)
    torch.testing.assert_close(features["attention_mask"], mask)
