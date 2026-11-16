from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.multi_vector_encoder.losses import MultiVectorDistillKLDivLoss
from sentence_transformers.sentence_transformer.losses import DistillKLDivLoss
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sentence_transformers.sparse_encoder.losses import SparseDistillKLDivLoss
from sentence_transformers.util import pairwise_dot_score


class _TokenModel(nn.Module):
    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        return features


def _one_token_scores(queries, documents, **kwargs):
    return (queries[:, None, 0] * documents[:, :, 0]).sum(dim=-1)


@pytest.mark.parametrize(
    "kind,autocast",
    [("dense", False), ("dense", True), ("sparse", False), ("multi_vector", False), ("multi_vector", True)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("teacher_temperature", [0.001, 0.7, 2.0])
def test_distill_teacher_scores_preserve_probability_precision(kind, dtype, teacher_temperature, autocast):
    generator = torch.Generator().manual_seed(29)
    embeddings = [torch.randn(3, 7, generator=generator).requires_grad_() for _ in range(3)]
    reference_embeddings = [embedding.detach().double().requires_grad_() for embedding in embeddings]
    labels = torch.tensor([[80.0, 79.0], [0.3, 0.1], [-0.2, 0.4]], dtype=dtype, requires_grad=True)
    original_labels = labels.detach().clone()
    student_temperature = 0.8
    kwargs = dict(student_temperature=student_temperature, teacher_temperature=teacher_temperature)
    with torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext():
        if kind == "multi_vector":
            loss = MultiVectorDistillKLDivLoss(_TokenModel(), similarity_fct=_one_token_scores, **kwargs)
            features = [
                {"token_embeddings": embedding[:, None], "attention_mask": torch.ones(3, 1)}
                for embedding in embeddings
            ]
            actual = loss(features, labels)
        else:
            loss_type = DistillKLDivLoss if kind == "dense" else SparseDistillKLDivLoss
            loss = loss_type(None, similarity_fct=pairwise_dot_score, **kwargs)
            # SparseEncoder modules also return dense activations during training;
            # converting the activations to COO is an inference/storage choice.
            actual = loss.compute_loss_from_embeddings(embeddings, labels)

    scores = torch.stack(
        [(reference_embeddings[0] * document).sum(dim=-1) for document in reference_embeddings[1:]], dim=1
    )
    teacher = torch.softmax(labels.detach().double() / teacher_temperature, dim=-1)
    expected = (
        F.kl_div(F.log_softmax(scores / student_temperature, dim=-1), teacher, reduction="batchmean")
        * student_temperature**2
    )
    assert torch.isfinite(actual)
    assert actual.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    torch.testing.assert_close(actual.double(), expected, rtol=3e-6, atol=1e-7)
    actual.backward()
    expected.backward()
    for embedding, reference in zip(embeddings, reference_embeddings):
        assert torch.isfinite(embedding.grad).all()
        torch.testing.assert_close(embedding.grad.double(), reference.grad, rtol=3e-6, atol=1e-7)
    assert labels.grad is None
    torch.testing.assert_close(labels.detach(), original_labels, rtol=0, atol=0)


def test_fp16_teacher_labels_train_an_offline_sentence_transformer():
    tokenizer = WhitespaceTokenizer(vocab=["[PAD]", "query", "first", "second"], stop_words=[])
    weights = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.3, 0.4]])
    words = WordEmbeddings(tokenizer, weights, update_embeddings=True)
    model = SentenceTransformer(modules=[words, Pooling(2, "mean")])
    loss = DistillKLDivLoss(model, teacher_temperature=0.001)
    features = [model.preprocess([text]) for text in ("query", "first", "second")]
    actual = loss(features, torch.tensor([[80.0, 79.0]], dtype=torch.float16))
    expected = -F.log_softmax(torch.tensor([[0.3, 0.7]]), dim=-1)[0, 0]
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(words.emb_layer.weight.grad).all()
    assert words.emb_layer.weight.grad.abs().sum() > 0
