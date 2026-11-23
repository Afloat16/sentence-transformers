from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from sentence_transformers.multi_vector_encoder.losses import MultiVectorDistillKLDivLoss
from sentence_transformers.sentence_transformer.losses import DistillKLDivLoss
from sentence_transformers.sparse_encoder.losses import SparseDistillKLDivLoss, SpladeLoss


class TokenModel(nn.Module):
    def forward(self, features, task=None):
        return features


def dense_scores(query, document):
    return (query * document).sum(dim=-1)


def token_scores(query, documents, **kwargs):
    return (query[:, None, 0] * documents[:, :, 0]).sum(dim=-1)


@pytest.mark.parametrize("loss_class", [DistillKLDivLoss, SparseDistillKLDivLoss, MultiVectorDistillKLDivLoss])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["teacher_scaling", "student_scaling", "ordinary"])
def test_distillation_temperatures_preserve_finite_precision(loss_class, dtype, case):
    student_temperature = 0.001 if case == "student_scaling" else 0.7
    teacher_temperature = 0.001 if case == "teacher_scaling" else 1.3
    scores = [[100.0, 99.0], [100.0, 100.0]] if case == "student_scaling" else [[0.25, 0.5], [1.0, -1.0]]
    teacher = [[100.0, 100.0], [100.0, 99.0]] if case == "teacher_scaling" else [[0.5, 1.0], [0.5, 0.0]]
    embeddings = [torch.ones(2, 1, dtype=dtype, requires_grad=True)]
    embeddings.extend(torch.tensor(scores, dtype=dtype)[:, i : i + 1].clone().requires_grad_() for i in range(2))
    teacher = torch.tensor(teacher, dtype=dtype, requires_grad=True)
    reference_embeddings = [value.detach().clone().requires_grad_() for value in embeddings]
    kwargs = dict(student_temperature=student_temperature, teacher_temperature=teacher_temperature)
    if loss_class is MultiVectorDistillKLDivLoss:
        loss = loss_class(TokenModel(), similarity_fct=token_scores, **kwargs)
        features = [
            {"token_embeddings": value[:, None, :], "attention_mask": torch.ones(2, 1, dtype=torch.bool)}
            for value in embeddings
        ]
        actual = loss(features, teacher)
    else:
        loss = loss_class(None, similarity_fct=dense_scores, **kwargs)
        actual = loss.compute_loss_from_embeddings(embeddings, teacher)

    # Preserve the original scorer's quantization, then evaluate the probability
    # distributions and relative entropy independently in double precision.
    reference_scores = torch.stack(
        [dense_scores(reference_embeddings[0], value) for value in reference_embeddings[1:]], dim=1
    ).double()
    log_student = reference_scores / student_temperature
    log_student = log_student - torch.logsumexp(log_student, dim=1, keepdim=True)
    log_teacher = teacher.detach().double() / teacher_temperature
    log_teacher = log_teacher - torch.logsumexp(log_teacher, dim=1, keepdim=True)
    probabilities = log_teacher.exp()
    expected = (probabilities * (log_teacher - log_student)).sum() / 2 * student_temperature**2

    assert actual.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual.double(), expected, atol=2e-7, rtol=2e-5)
    actual_grads = torch.autograd.grad(actual, [*embeddings, teacher], allow_unused=True)
    assert actual_grads[-1] is None
    expected_grads = torch.autograd.grad(expected, reference_embeddings)
    for actual_grad, expected_grad in zip(actual_grads[:-1], expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize("kind", ["dense", "sparse", "multi_vector"])
def test_distillation_half_teacher_matches_pretrained_float_teacher(kind, request):
    fixture = {
        "dense": "stsb_bert_tiny_model",
        "sparse": "splade_bert_tiny_model",
        "multi_vector": "mve_bert_tiny_model",
    }[kind]
    model = request.getfixturevalue(fixture)
    model.train()
    texts = [["What is Python?", "What is PyTorch?"], ["A language.", "A framework."], ["A snake.", "A lunch."]]
    features = [model.preprocess(column, task="query" if i == 0 else "document") for i, column in enumerate(texts)]
    features = [
        {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in feature.items()}
        for feature in features
    ]
    labels = torch.tensor([[100.0, 100.0], [100.0, 99.0]], dtype=torch.float16, device=model.device)
    if kind == "multi_vector":
        loss = MultiVectorDistillKLDivLoss(model, teacher_temperature=0.001)
    elif kind == "sparse":
        base = SparseDistillKLDivLoss(model, teacher_temperature=0.001)
        loss = SpladeLoss(model, base, document_regularizer_weight=3e-5)
    else:
        loss = DistillKLDivLoss(model, teacher_temperature=0.001)

    def scalar(labels):
        torch.manual_seed(77)
        result = loss(copy.deepcopy(features), labels)
        return sum(result.values()) if isinstance(result, dict) else result

    actual = scalar(labels)
    assert torch.isfinite(actual)
    actual.backward()
    actual_grads = {
        name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    assert actual_grads
    assert all(torch.isfinite(gradient).all() for gradient in actual_grads.values())
    model.zero_grad()
    expected = scalar(labels.float())
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(actual_grads[name], parameter.grad, rtol=0, atol=0)
