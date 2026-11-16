from __future__ import annotations

import pytest
import torch
from torch import nn

from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.cross_encoder.losses import MarginMSELoss as CrossMarginMSELoss
from sentence_transformers.multi_vector_encoder.losses import MultiVectorMarginMSELoss
from sentence_transformers.sentence_transformer.losses import MarginMSELoss
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sentence_transformers.sparse_encoder.losses import SparseMarginMSELoss
from sentence_transformers.util import pairwise_dot_score


class _TokenModel(nn.Module):
    def forward(self, features, **kwargs):
        return features

    @staticmethod
    def similarity_pairwise(first, second, **kwargs):
        return (first[:, 0] * second[:, 0]).sum(dim=-1)


class _PairModel(CrossEncoder):
    num_labels = 1
    device = torch.device("cpu")

    def __init__(self, values):
        nn.Module.__init__(self)
        self.values = nn.Parameter(values)

    def preprocess(self, pairs, **kwargs):
        return {"indices": torch.tensor([int(document) for _, document in pairs])}

    def forward(self, features):
        return {"scores": self.values[features["indices"]]}


@pytest.mark.parametrize("kind", ["dense", "sparse", "multi_vector", "cross_encoder"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("raw_scores", [False, True])
@pytest.mark.parametrize("scale", [1.0, 40000.0])
def test_margin_targets_are_converted_before_subtraction(kind, dtype, raw_scores, scale):
    student_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    teacher = torch.tensor([[scale, -scale, 0.0], [0.5 * scale, scale, -scale]], dtype=dtype)
    labels = teacher if raw_scores else teacher[:, 1:]
    labels = labels.clone().requires_grad_()
    original = labels.detach().clone()
    generator = torch.Generator().manual_seed(5)
    embeddings = [torch.randn(2, 5, generator=generator, dtype=student_dtype).requires_grad_() for _ in range(4)]
    reference_embeddings = [embedding.detach().double().requires_grad_() for embedding in embeddings]
    if kind == "cross_encoder":
        model = _PairModel(torch.tensor([0.3, 0.7, -0.1, 1.2, 0.6, -0.5], dtype=student_dtype))
        actual = CrossMarginMSELoss(model)([["q", "q"], ["0", "3"], ["1", "4"], ["2", "5"]], labels)
        reference_values = model.values.detach().double().requires_grad_()
        predictions = reference_values.view(2, 3)
        actual_parameters, reference_parameters = [model.values], [reference_values]
    else:
        if kind == "multi_vector":
            loss = MultiVectorMarginMSELoss(_TokenModel())
            features = [
                {"token_embeddings": embedding[:, None], "attention_mask": torch.ones(2, 1)}
                for embedding in embeddings
            ]
            actual = loss(features, labels)
        else:
            loss_type = MarginMSELoss if kind == "dense" else SparseMarginMSELoss
            actual = loss_type(None, similarity_fct=pairwise_dot_score).compute_loss_from_embeddings(
                embeddings, labels
            )
        predictions = torch.stack(
            [(reference_embeddings[0] * document).sum(dim=-1) for document in reference_embeddings[1:]], dim=1
        )
        actual_parameters, reference_parameters = embeddings, reference_embeddings
    targets = labels.detach().double()
    if raw_scores:
        targets = targets[:, :1] - targets[:, 1:]
    expected = ((predictions[:, :1] - predictions[:, 1:] - targets) ** 2).mean()
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-7)
    actual.backward()
    expected.backward()
    for parameter, reference in zip(actual_parameters, reference_parameters):
        assert torch.isfinite(parameter.grad).all()
        if scale > 1 and student_dtype == torch.float32:
            # Opposite large margins can cancel in a component of the input gradient.
            # Compare the full vector with float64, without dividing by that tiny component.
            error = torch.linalg.vector_norm(parameter.grad.double() - reference.grad)
            assert error <= 2e-6 * torch.linalg.vector_norm(reference.grad)
        else:
            torch.testing.assert_close(parameter.grad.double(), reference.grad, rtol=2e-6, atol=1e-6)
    assert labels.grad is None
    torch.testing.assert_close(labels.detach(), original, rtol=0, atol=0)


def test_fp16_teacher_margins_train_an_offline_encoder():
    tokenizer = WhitespaceTokenizer(vocab=["[PAD]", "q", "p", "n"], stop_words=[])
    weights = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.3, 0.4]])
    words = WordEmbeddings(tokenizer, weights, update_embeddings=True)
    model = SentenceTransformer(modules=[words, Pooling(2, "mean")])
    loss = MarginMSELoss(model)
    value = loss([model.preprocess([text]) for text in ("q", "p", "n")], torch.tensor([0.5], dtype=torch.float16))
    torch.testing.assert_close(value, torch.tensor(0.81))
    value.backward()
    assert torch.isfinite(words.emb_layer.weight.grad).all()
    assert words.emb_layer.weight.grad.abs().sum() > 0
