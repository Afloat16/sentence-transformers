from __future__ import annotations

import copy

import pytest
import torch

from sentence_transformers import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import MSELoss
from sentence_transformers.util import is_training_available
from tests.utils import skip_bfloat16_cpu_crash, skip_fp16_cpu_layer_norm


class ScoreModel(CrossEncoder):
    def __init__(self, scores):
        torch.nn.Module.__init__(self)
        self.scores = torch.nn.Parameter(scores)

    @property
    def num_labels(self):
        return 1

    @property
    def device(self):
        return self.scores.device

    def preprocess(self, pairs, **kwargs):
        return {"indices": torch.tensor([int(document) for _, document in pairs])}

    def forward(self, features):
        return {"scores": self.scores[features["indices"]].unsqueeze(-1)}


INPUTS = (["query"] * 3, ["0", "1", "2"])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("label_dtype", [torch.bool, torch.int64, torch.float64])
def test_mse_loss_score_and_target_dtypes(dtype, label_dtype):
    model = ScoreModel(torch.tensor([0.2, -0.3, 1.5], dtype=dtype))
    reference_scores = model.scores.detach().clone().requires_grad_()
    labels = torch.tensor([0, 1, 0], dtype=label_dtype)

    actual = MSELoss(model)(INPUTS, labels)
    scores = reference_scores.float() if dtype in (torch.float16, torch.bfloat16) else reference_scores
    expected = (scores - labels.to(scores.dtype)).square().mean()

    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    assert model.scores.grad.dtype == dtype
    assert torch.isfinite(model.scores.grad).all()
    torch.testing.assert_close(model.scores.grad, reference_scores.grad)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_mse_loss_preserves_column_labels_and_reduction(reduction):
    model = ScoreModel(torch.tensor([0.2, -0.3, 1.5], dtype=torch.float64))
    reference_scores = model.scores.detach().clone().requires_grad_()
    labels = torch.tensor([[0.1], [0.7], [0.9]], dtype=torch.float64, requires_grad=True)

    actual = MSELoss(model, reduction=reduction)(INPUTS, labels)
    errors = (reference_scores - labels.detach().flatten()).square()
    expected = errors.mean() if reduction == "mean" else errors.sum() if reduction == "sum" else errors

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(model.scores.grad, reference_scores.grad)
    assert labels.grad is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_mse_loss_preserves_activation_dtype_and_gradients(dtype):
    class ScoreActivation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.75, dtype=dtype))
            self.input_dtype = None

        def forward(self, scores):
            self.input_dtype = scores.dtype
            return (scores * self.scale).sigmoid()

    model = ScoreModel(torch.tensor([0.2, -0.3, 1.5], dtype=dtype))
    activation = ScoreActivation()
    reference_activation = copy.deepcopy(activation)
    reference_scores = model.scores.detach().clone().requires_grad_()
    labels = torch.tensor([0.1, 0.7, 0.9], dtype=torch.float64)

    actual = MSELoss(model, activation_fn=activation)(INPUTS, labels)
    scores = reference_activation(reference_scores)
    if dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    expected = (scores - labels.to(scores.dtype)).square().mean()

    assert activation.input_dtype == dtype
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(model.scores.grad, reference_scores.grad)
    torch.testing.assert_close(activation.scale.grad, reference_activation.scale.grad)
    assert torch.isfinite(activation.scale.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mse_loss_preserves_autocast_behavior(dtype):
    model = ScoreModel(torch.tensor([0.2, -0.3, 1.5], dtype=dtype))
    reference_scores = model.scores.detach().clone().requires_grad_()
    labels = torch.tensor([0.1, 0.7, 0.9])

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = MSELoss(model)(INPUTS, labels)
        expected = torch.nn.functional.mse_loss(reference_scores, labels)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(model.scores.grad, reference_scores.grad, rtol=0, atol=0)


@pytest.mark.skipif(not is_training_available(), reason="Requires the train extra.")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, marks=skip_fp16_cpu_layer_norm),
        pytest.param(torch.bfloat16, marks=skip_bfloat16_cpu_crash),
        torch.float32,
        torch.float64,
    ],
)
def test_mse_loss_trains_native_cross_encoder(reranker_bert_tiny_model, tmp_path, dtype):
    from datasets import Dataset

    model = reranker_bert_tiny_model.to(device="cpu", dtype=dtype)
    original = [parameter.detach().clone() for parameter in model.parameters()]
    dataset = Dataset.from_dict(
        {
            "query": ["What is Python?", "What is PyTorch?"],
            "document": ["A programming language.", "A deep learning framework."],
            "label": [0.25, 0.75],
        }
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=CrossEncoderTrainingArguments(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=2,
            learning_rate=0.01,
            optim="sgd",
            use_cpu=True,
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
        ),
        train_dataset=dataset,
        loss=MSELoss(model),
    )

    result = trainer.train()

    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert any(not torch.equal(before, after) for before, after in zip(original, model.parameters()))
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
