from __future__ import annotations

import copy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sentence_transformers.sparse_encoder.modules import SpladePooling


class LogWork(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.elements = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.log1p.default, torch.ops.aten.log1p_.default):
            self.elements += args[0].numel()
        return func(*args, **(kwargs or {}))


def features_for(layout, dtype=torch.float32):
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(3, 17, 41, generator=generator, dtype=dtype)
    # Include negative-only dimensions, exact ties and values whose transforms
    # tie after rounding. The last row is fully masked in the padded layout.
    logits[:, :, 0] = -1
    logits[:, :, 1] = 2
    logits[:, 0, 2] = 1024
    logits[:, 1, 2] = 1025
    if layout == "flat":
        return {"token_embeddings": logits.reshape(1, 51, 41), "cu_seq_lens_q": torch.tensor([0, 17, 34, 51])}
    mask = torch.arange(17)[None, :] < torch.tensor([[9], [17], [0]])
    return {"token_embeddings": logits, "attention_mask": mask}


def original_max(features, activation, chunk_size):
    logits = features["token_embeddings"]

    def transform(values):
        result = values.relu().log1p()
        return result.log1p() if activation == "log1p_relu" else result

    if "cu_seq_lens_q" in features:
        bounds = features["cu_seq_lens_q"].tolist()
        return torch.stack([transform(logits[0, a:b]).amax(dim=0) for a, b in zip(bounds, bounds[1:])])
    mask = features["attention_mask"].unsqueeze(-1).to(logits.dtype)
    result = logits.new_full((logits.shape[0], logits.shape[-1]), -torch.inf)
    step = chunk_size or logits.shape[1]
    for begin in range(0, logits.shape[1], step):
        values = transform(logits[:, begin : begin + step] * mask[:, begin : begin + step])
        result = torch.maximum(result, values.max(dim=1).values)
    return result


@pytest.mark.parametrize("layout,chunk_size", [("padded", None), ("padded", 4), ("flat", None)])
@pytest.mark.parametrize("activation", ["relu", "log1p_relu"])
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_inference_max_transforms_only_pooled_values(layout, chunk_size, activation, context):
    features = features_for(layout)
    pooling = SpladePooling(activation_function=activation, chunk_size=chunk_size).eval()
    with context(), LogWork() as work:
        result = pooling(dict(features))["sentence_embedding"]
    chunks = 5 if chunk_size else 1
    expected_elements = 3 * 41 * chunks * (2 if activation == "log1p_relu" else 1)
    assert work.elements == expected_elements
    torch.testing.assert_close(result, original_max(features, activation, chunk_size), rtol=0, atol=0)


@pytest.mark.parametrize("layout,chunk_size", [("padded", None), ("padded", 4), ("flat", None)])
@pytest.mark.parametrize("activation", ["relu", "log1p_relu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("training", [False, True])
def test_max_inference_keeps_values_dtype_and_input(layout, chunk_size, activation, dtype, training):
    features = features_for(layout, dtype)
    original = features["token_embeddings"].clone()
    expected = original_max(features, activation, chunk_size)
    pooling = SpladePooling(activation_function=activation, chunk_size=chunk_size).train(training)
    with torch.no_grad():
        actual = pooling(dict(features))["sentence_embedding"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(features["token_embeddings"], original, rtol=0, atol=0)
    assert actual.dtype == dtype
    assert pooling.training is training


@pytest.mark.parametrize("layout,chunk_size", [("padded", None), ("padded", 4), ("flat", None)])
@pytest.mark.parametrize("activation", ["relu", "log1p_relu"])
def test_gradient_enabled_max_preserves_rounded_tie_gradients(layout, chunk_size, activation):
    features = features_for(layout, torch.float16)
    features["token_embeddings"].requires_grad_()
    reference_features = copy.deepcopy(features)
    expected = original_max(reference_features, activation, chunk_size)
    pooling = SpladePooling(activation_function=activation, chunk_size=chunk_size).train()
    actual = pooling(dict(features))["sentence_embedding"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        features["token_embeddings"].grad, reference_features["token_embeddings"].grad, rtol=0, atol=0
    )


@pytest.mark.parametrize("layout", ["padded", "flat"])
@pytest.mark.parametrize("activation", ["relu", "log1p_relu"])
@pytest.mark.parametrize("training", [False, True])
def test_no_grad_preserves_forward_mode_tangents(layout, activation, training):
    # no_grad disables reverse-mode recording, but forward-mode AD is still active.
    # Distinct half-precision inputs tie after log1p; their selected tangents differ
    # if the maximum is moved ahead of that rounding step.
    logits = torch.tensor([[[1024.0], [1025.0]]], dtype=torch.float16)
    tangent = torch.tensor([[[1.0], [0.0]]], dtype=torch.float16)
    metadata = {"attention_mask": torch.ones(1, 2)} if layout == "padded" else {"cu_seq_lens_q": torch.tensor([0, 2])}
    pooling = SpladePooling(activation_function=activation).train(training)

    def forward(value):
        return pooling({"token_embeddings": value, **metadata})["sentence_embedding"]

    def reference(value):
        return original_max({"token_embeddings": value, **metadata}, activation, None)

    with torch.no_grad():
        actual_value, actual_tangent = torch.func.jvp(forward, (logits,), (tangent,))
        expected_value, expected_tangent = torch.func.jvp(reference, (logits,), (tangent,))
    assert (expected_tangent > 0).all()
    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    torch.testing.assert_close(actual_tangent, expected_tangent, rtol=0, atol=0)


def test_pretrained_inference_matches_transform_before_max(splade_bert_tiny_model):
    model = splade_bert_tiny_model.eval()
    texts = ["A programming language.", "Paris is the capital of France.", "A snake."]
    pooling = next(module for module in model.modules() if isinstance(module, SpladePooling))
    features = model.preprocess(texts)
    with torch.no_grad():
        # Capture actual MLM outputs at the public pooling boundary.
        captured = []
        handle = pooling.register_forward_pre_hook(lambda module, args: captured.append(copy.deepcopy(args[0])))
        try:
            actual = model(copy.deepcopy(features))["sentence_embedding"]
        finally:
            handle.remove()
        expected = original_max(captured[0], pooling.activation_function, pooling.chunk_size)
        encoded = model.encode(texts, convert_to_sparse_tensor=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(encoded, expected, rtol=0, atol=0)
