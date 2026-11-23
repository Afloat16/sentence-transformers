from __future__ import annotations

import copy

import pytest
import torch

from sentence_transformers.sparse_encoder.modules import SpladePooling


def input_features(layout, dtype):
    generator = torch.Generator().manual_seed(37)
    segments = [torch.randn(length, 13, generator=generator, dtype=dtype) for length in [4, 7]]
    if layout == "flat":
        logits = torch.cat(segments).unsqueeze(0).requires_grad_()
        return {"token_embeddings": logits, "cu_seq_lens_q": torch.tensor([0, 4, 11])}, segments
    logits = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True).requires_grad_()
    mask = torch.arange(7)[None, :] < torch.tensor([[4], [7]])
    return {"token_embeddings": logits, "attention_mask": mask}, segments


@pytest.mark.parametrize("layout,chunk_size", [("padded", None), ("padded", 2), ("flat", None)])
@pytest.mark.parametrize("pooling_strategy", ["max", "sum"])
@pytest.mark.parametrize("activation_function", ["relu", "log1p_relu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_splade_eval_supports_gradients(layout, chunk_size, pooling_strategy, activation_function, dtype):
    features, segments = input_features(layout, dtype)
    logits = features["token_embeddings"]
    original = logits.detach().clone()
    reference_segments = [segment.clone().requires_grad_() for segment in segments]
    pooling = SpladePooling(
        pooling_strategy=pooling_strategy, activation_function=activation_function, chunk_size=chunk_size
    ).eval()
    actual = pooling(dict(features))["sentence_embedding"]
    rows = []
    for segment in reference_segments:
        transformed = torch.log1p(torch.relu(segment))
        if activation_function == "log1p_relu":
            transformed = torch.log1p(transformed)
        rows.append(transformed.max(dim=0).values if pooling_strategy == "max" else transformed.sum(dim=0))
    expected = torch.stack(rows)
    weights = torch.linspace(0.1, 1.0, actual.numel(), dtype=dtype).reshape_as(actual)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad((actual * weights).sum(), logits)[0]
    expected_grads = torch.autograd.grad((expected * weights).sum(), reference_segments)
    if layout == "flat":
        reference_grad = torch.cat(expected_grads).unsqueeze(0)
    else:
        reference_grad = torch.nn.utils.rnn.pad_sequence(expected_grads, batch_first=True)
    torch.testing.assert_close(actual_grad, reference_grad)
    torch.testing.assert_close(logits.detach(), original, rtol=0, atol=0)
    assert not pooling.training


@pytest.mark.parametrize("layout", ["padded", "flat"])
@pytest.mark.parametrize("activation_function", ["relu", "log1p_relu"])
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_splade_eval_keeps_inference_values(layout, activation_function, context):
    features, _ = input_features(layout, torch.float32)
    reference = SpladePooling(activation_function=activation_function).train()
    expected = reference(dict(features))["sentence_embedding"].detach()
    pooling = SpladePooling(activation_function=activation_function).eval()
    with context():
        actual = pooling(dict(features))["sentence_embedding"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad
    assert not pooling.training


@pytest.mark.parametrize("layout", ["padded", "flat"])
@pytest.mark.parametrize("activation_function", ["relu", "log1p_relu"])
def test_splade_eval_supports_second_derivatives(layout, activation_function):
    logits = torch.linspace(0.1, 0.9, 12, dtype=torch.float64).reshape(1, 4, 3).requires_grad_()
    pooling = SpladePooling(pooling_strategy="sum", activation_function=activation_function).eval()
    metadata = {"attention_mask": torch.ones(1, 4)} if layout == "padded" else {"cu_seq_lens_q": torch.tensor([0, 4])}

    def forward(logits):
        return pooling({"token_embeddings": logits, **metadata})["sentence_embedding"]

    assert torch.autograd.gradcheck(forward, (logits,), fast_mode=True)
    assert torch.autograd.gradgradcheck(forward, (logits,), fast_mode=True)


def test_pretrained_splade_eval_gradients_match_safe_pooling(splade_bert_tiny_model):
    model = splade_bert_tiny_model.eval()
    reference = copy.deepcopy(model)
    # Pooling has no dropout or running statistics; its training path is the
    # original non-inplace reference while the encoder stays in evaluation mode.
    for module in reference.modules():
        if isinstance(module, SpladePooling):
            module.train()
    features = model.preprocess(["A programming language.", "A snake."])
    actual = model(copy.deepcopy(features))["sentence_embedding"]
    expected = reference(copy.deepcopy(features))["sentence_embedding"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    count = 0
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)
            count += 1
    assert count > 0
    assert all(not module.training for module in model.modules())
