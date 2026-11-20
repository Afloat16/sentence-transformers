from __future__ import annotations

import copy

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import (
    DistillKLDivLoss,
    EmbedDistillLoss,
    MarginMSELoss,
    MatryoshkaLoss,
    MSELoss,
)
from sentence_transformers.sentence_transformer.modules import StaticEmbedding


def _model(dtype=torch.float32):
    vocabulary = {"[UNK]": 0, **{f"t{i}": i + 1 for i in range(16)}}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    weights = torch.randn(len(vocabulary), 8, dtype=dtype, generator=torch.Generator().manual_seed(12))
    return SentenceTransformer(modules=[StaticEmbedding(tokenizer, embedding_weights=weights)], device="cpu")


@pytest.mark.parametrize("target_kind", ["kl_scores", "mse_scores", "mse_margins"])
@pytest.mark.parametrize("target_width", [7, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matryoshka_preserves_score_targets(target_kind, target_width, dtype) -> None:
    model = _model(dtype)
    reference_model = copy.deepcopy(model)
    loss_class = DistillKLDivLoss if target_kind == "kl_scores" else MarginMSELoss
    num_candidates = target_width + (target_kind == "mse_margins")
    columns = [["t0 t1", "t2 t3"]] + [[f"t{i} t4", f"t{i} t5"] for i in range(num_candidates)]
    labels = torch.randn(2, target_width, dtype=dtype, generator=torch.Generator().manual_seed(15))
    original_labels = labels.clone()
    dims, weights = [8, 4], [1.0, 0.5]

    actual = MatryoshkaLoss(model, loss_class(model), dims, weights)(
        [model.preprocess(column) for column in columns], labels
    )
    full_embeddings = [reference_model(reference_model.preprocess(column))["sentence_embedding"] for column in columns]
    reference_loss = loss_class(reference_model)
    expected = sum(
        weight
        * reference_loss.compute_loss_from_embeddings(
            [torch.nn.functional.normalize(embedding[:, :dim], dim=-1) for embedding in full_embeddings], labels
        )
        for dim, weight in zip(dims, weights)
    )

    torch.testing.assert_close(labels, original_labels, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, model[0].embedding.weight)[0]
    expected_grad = torch.autograd.grad(expected, reference_model[0].embedding.weight)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("loss_kind", ["mse", "embed_mse", "embed_l2", "embed_cosine", "custom"])
@pytest.mark.parametrize("per_column", [False, True])
def test_matryoshka_keeps_embedding_target_truncation(loss_kind, per_column) -> None:
    class CustomEmbeddingLoss(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.target_widths = []

        def forward(self, features, labels):
            self.target_widths.append(labels.shape[-1])
            embeddings = [self.model(feature)["sentence_embedding"] for feature in features]
            targets = (
                [labels[:, index] for index in range(len(embeddings))]
                if labels.ndim == 3
                else [labels] * len(embeddings)
            )
            return torch.stack(
                [(embedding - target).square().mean() for embedding, target in zip(embeddings, targets)]
            ).mean()

    def make_loss(model):
        if loss_kind == "mse":
            return MSELoss(model)
        if loss_kind == "custom":
            return CustomEmbeddingLoss(model)
        return EmbedDistillLoss(model, distance_metric=loss_kind.removeprefix("embed_"))

    model = _model()
    reference_model = copy.deepcopy(model)
    loss = make_loss(model)
    reference_loss = make_loss(reference_model)
    columns = [["t0 t1", "t2 t3"], ["t4 t5", "t6 t7"]]
    label_shape = (2, 2, 8) if per_column else (2, 8)
    labels = torch.randn(label_shape, generator=torch.Generator().manual_seed(32))
    original_labels = labels.clone()
    actual = MatryoshkaLoss(model, loss, [8, 4])([model.preprocess(column) for column in columns], labels)

    expected = sum(
        MatryoshkaLoss(reference_model, reference_loss, [dim])(
            [reference_model.preprocess(column) for column in columns], labels[..., :dim]
        )
        for dim in [8, 4]
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(labels, original_labels, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, model[0].embedding.weight)[0],
        torch.autograd.grad(expected, reference_model[0].embedding.weight)[0],
    )
    if loss_kind == "custom":
        assert loss.target_widths == [8, 4]


def test_matryoshka_respects_custom_score_target_declaration() -> None:
    class CustomScoreLoss(torch.nn.Module):
        labels_are_embeddings = False

        def __init__(self, model):
            super().__init__()
            self.model = model
            self.seen_labels = []

        def forward(self, features, labels):
            self.seen_labels.append(labels)
            embeddings = self.model(features[0])["sentence_embedding"]
            return (embeddings[:, 0] - labels.mean(dim=-1)).square().mean()

    model = _model()
    loss = CustomScoreLoss(model)
    labels = torch.randn(2, 8)
    value = MatryoshkaLoss(model, loss, [8, 4])([model.preprocess(["t0 t1", "t2 t3"])], labels)
    value.backward()
    assert len(loss.seen_labels) == 2
    assert all(seen is labels for seen in loss.seen_labels)
    assert torch.isfinite(model[0].embedding.weight.grad).all()
