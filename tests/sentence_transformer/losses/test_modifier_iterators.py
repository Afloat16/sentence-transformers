from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules import Transformer
from sentence_transformers.sentence_transformer.losses import (
    AdaptiveLayerLoss,
    CachedMultipleNegativesRankingLoss,
    Matryoshka2dLoss,
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.modules import Pooling


@pytest.fixture
def local_model(tmp_path: Path) -> SentenceTransformer:
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alpha", "beta", "gamma"]
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        BertModel(config).save_pretrained(tmp_path)
    BertTokenizer(vocab={token: index for index, token in enumerate(vocab)}).save_pretrained(tmp_path)
    return SentenceTransformer(modules=[Transformer(str(tmp_path)), Pooling(8)], device="cpu")


@pytest.mark.parametrize("modifier", ["dimensions", "layers", "both", "cached"])
def test_loss_modifiers_replay_iterator_columns(local_model: SentenceTransformer, modifier: str) -> None:
    model = local_model
    inner = MultipleNegativesRankingLoss(model)
    if modifier == "cached":
        inner = CachedMultipleNegativesRankingLoss(model, mini_batch_size=1)
    if modifier in ("dimensions", "cached"):
        loss = MatryoshkaLoss(model, inner, [8, 4])
    elif modifier == "layers":
        loss = AdaptiveLayerLoss(model, inner, n_layers_per_step=-1)
    else:
        loss = Matryoshka2dLoss(model, inner, [8, 4], n_layers_per_step=-1, n_dims_per_step=-1)

    def features():
        return [model.preprocess(["alpha beta", "gamma"]), model.preprocess(["alpha", "gamma beta"])]

    expected = loss(features(), None)
    expected.backward()
    expected_gradients = {
        name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)

    actual = loss(iter(features()), None)
    actual.backward()

    torch.testing.assert_close(actual, expected)
    actual_gradients = {
        name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    assert actual_gradients.keys() == expected_gradients.keys()
    for name, expected_gradient in expected_gradients.items():
        torch.testing.assert_close(actual_gradients[name], expected_gradient)
