from __future__ import annotations

import pytest
import torch
from torch import nn

from sentence_transformers.cross_encoder.losses import ListMLELoss, PListMLELoss


class ScoreModel(nn.Module):
    num_labels = 1
    device = torch.device("cpu")

    def __init__(self, scores):
        super().__init__()
        self.scores = nn.Parameter(torch.tensor(scores))

    def preprocess(self, pairs, **kwargs):
        return {"indices": torch.tensor([int(document) for _, document in pairs])}

    def forward(self, inputs):
        return {"scores": self.scores[inputs["indices"]]}


@pytest.mark.parametrize("loss_cls", [ListMLELoss, PListMLELoss])
@pytest.mark.parametrize("respect_input_order", [False, True])
def test_single_document_batches_keep_zero_model_gradients(loss_cls, respect_input_order):
    model = ScoreModel([1.0, 2.0])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    loss_function = loss_cls(model, respect_input_order=respect_input_order, mini_batch_size=1)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        value = loss_function((["q1", "q2"], [["0"], ["1"]]), [torch.tensor([1.0]), torch.tensor([1.0])])
        assert value.item() == 0.0
        scaler.scale(value).backward()
        torch.testing.assert_close(model.scores.grad, torch.zeros_like(model.scores))
        scaler.step(optimizer)
        scaler.update()
    torch.testing.assert_close(model.scores.detach(), torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("respect_input_order", [False, True])
def test_loss_rounded_to_zero_keeps_nonzero_ranking_signal(respect_input_order):
    model = ScoreModel([0.0, 40.0])
    docs = ["1", "0"] if respect_input_order else ["0", "1"]
    labels = torch.tensor([1.0, 0.0] if respect_input_order else [0.0, 1.0])
    value = PListMLELoss(model, respect_input_order=respect_input_order)((["query"], [docs]), [labels])
    # A two-document ListMLE is ordinary two-class cross entropy. The default
    # positional weights are [1, 0], so PListMLE has the same objective here.
    reference_scores = model.scores.detach().clone().requires_grad_()
    expected = nn.functional.cross_entropy(reference_scores.unsqueeze(0), torch.tensor([1]))
    assert value.item() == expected.item() == 0.0
    value.backward()
    expected.backward()
    assert model.scores.grad is not None
    torch.testing.assert_close(model.scores.grad, reference_scores.grad, rtol=1e-6, atol=0)
    assert model.scores.grad[0] > 0
    assert torch.isfinite(model.scores.grad).all()


@pytest.mark.parametrize("loss_cls", [ListMLELoss, PListMLELoss])
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_zero_evaluation_loss_respects_gradient_context(loss_cls, context):
    model = ScoreModel([1.0])
    with context():
        value = loss_cls(model)((["query"], [["0"]]), [torch.tensor([1.0])])
    assert value.item() == 0.0
    assert not value.requires_grad


@pytest.mark.parametrize("loss_cls", [ListMLELoss, PListMLELoss])
def test_pretrained_single_document_step_with_gradient_scaling(reranker_bert_tiny_model_v54, loss_cls):
    model = reranker_bert_tiny_model_v54.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    loss_function = loss_cls(model, mini_batch_size=1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        value = loss_function(
            (["Capital of France?", "A programming language?"], [["Paris."], ["Python."]]),
            [torch.tensor([1.0]), torch.tensor([1.0])],
        )
    assert value.item() == 0.0
    scaler.scale(value).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() and (gradient == 0).all() for gradient in gradients)
    scaler.step(optimizer)
    scaler.update()
