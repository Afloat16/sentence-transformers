from __future__ import annotations

import copy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, WeightedLayerPooling


def original_average(layers, weights, start):
    stacked = torch.stack(layers)[start:]
    expanded = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(stacked.size())
    return (expanded * stacked).sum(0) / weights.sum()


class LargestWorkingTensor(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.elements = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.stack.default, torch.ops.aten.mul.Tensor, torch.ops.aten.add.Tensor):
            self.elements = max(self.elements, result.numel())
        return result


@pytest.mark.parametrize("requires_grad", [False, True])
def test_weighted_pooling_does_not_copy_discarded_layers(requires_grad):
    layers = [torch.randn(2, 31, 17, requires_grad=requires_grad) for _ in range(13)]
    pooling = WeightedLayerPooling(17, num_hidden_layers=12, layer_start=4)
    with LargestWorkingTensor() as work:
        output = pooling({"all_layer_embeddings": layers})["token_embeddings"]
    assert work.elements <= 9 * layers[0].numel()
    if requires_grad:
        output.square().sum().backward()
        assert all(layer.grad is not None for layer in layers)


@pytest.mark.parametrize("layer_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("start", [0, 2, -2])
def test_weighted_pooling_keeps_values_and_all_gradients(layer_dtype, weight_dtype, start):
    torch.manual_seed(93)
    layers = [torch.randn(2, 3, 5, dtype=layer_dtype).requires_grad_() for _ in range(5)]
    reference_layers = [layer.detach().clone().requires_grad_() for layer in layers]
    weights = torch.linspace(0.4, 1.2, len(layers[start:]), dtype=weight_dtype).requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    pooling = WeightedLayerPooling(5, num_hidden_layers=4, layer_start=start, layer_weights=weights)
    output = pooling({"all_layer_embeddings": tuple(layers)})["token_embeddings"]
    expected = original_average(reference_layers, reference_weights, start)
    assert output.dtype == expected.dtype
    torch.testing.assert_close(output, expected)
    derivative = torch.randn(output.shape, dtype=output.dtype)
    actual_grads = torch.autograd.grad(output, (*layers, weights), derivative)
    expected_grads = torch.autograd.grad(expected, (*reference_layers, reference_weights), derivative)
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_sum_keeps_wide_accumulation(dtype):
    values = [60000, 60000, -60000] if dtype == torch.float16 else [256, 1, -256]
    layers = [torch.full((1, 2, 3), value, dtype=dtype) for value in values]
    weights = torch.ones(3, dtype=dtype, requires_grad=True)
    model = WeightedLayerPooling(3, num_hidden_layers=2, layer_start=0, layer_weights=weights)
    output = model({"all_layer_embeddings": layers})["token_embeddings"]
    expected = original_average(layers, weights, 0)
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 5)])
def test_weighted_pooling_preserves_cancellation_reduction_order(shape):
    values = [7, 7, 1e20, 1, -1e20, 1, 1, 1, 1, 1, 1]
    layers = [torch.full(shape, value, dtype=torch.float32, requires_grad=True) for value in values]
    weights = torch.ones(9, requires_grad=True)
    model = WeightedLayerPooling(shape[-1], num_hidden_layers=10, layer_start=2, layer_weights=weights)
    actual = model({"all_layer_embeddings": layers})["token_embeddings"]
    expected = original_average(layers, weights, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual.sum(), (*layers, weights))
    expected_grads = torch.autograd.grad(expected.sum(), (*layers, weights))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_unused_layers_keep_zero_gradients_even_for_nonfinite_values():
    layers = [torch.full((2, 3, 5), float("nan"), requires_grad=True), torch.randn(2, 3, 5, requires_grad=True)]
    weights = torch.ones(1, requires_grad=True)
    model = WeightedLayerPooling(5, num_hidden_layers=1, layer_start=1, layer_weights=weights)
    output = model({"all_layer_embeddings": layers})["token_embeddings"]
    gradients = torch.autograd.grad(output.sum(), (*layers, weights))
    assert torch.isfinite(output).all()
    assert torch.equal(gradients[0], torch.zeros_like(layers[0]))
    assert torch.equal(gradients[1], torch.ones_like(layers[1]))


@pytest.mark.parametrize("variant", ["noncontiguous", "mixed_layer_dtype", "scalar_weight", "empty_selection"])
def test_weighted_pooling_preserves_existing_input_variants(variant):
    layers = [torch.randn(2, 3, 5) for _ in range(4)]
    weights = torch.tensor([1.0, 2.0, 3.0])
    start = 1
    if variant == "noncontiguous":
        layers = [layer.transpose(1, 2) for layer in layers]
    elif variant == "mixed_layer_dtype":
        layers[0] = layers[0].double()
    elif variant == "scalar_weight":
        weights = torch.tensor(2.0)
    else:
        start = 4
        weights = torch.empty(0)
    model = WeightedLayerPooling(5, num_hidden_layers=3, layer_start=start, layer_weights=weights)
    output = model({"all_layer_embeddings": layers})["token_embeddings"]
    expected = original_average(layers, weights, start)
    assert output.dtype == expected.dtype
    assert output.is_contiguous()
    torch.testing.assert_close(output, expected, equal_nan=True)


def test_weighted_pooling_keeps_features_and_inputs():
    model = WeightedLayerPooling(5, num_hidden_layers=2, layer_start=1)
    features = {"token_embeddings": torch.ones(2, 3, 5)}
    assert model(features) is features
    layers = [torch.randn(2, 3, 5) for _ in range(3)]
    copies = [layer.clone() for layer in layers]
    features["all_layer_embeddings"] = layers
    assert model(features) is features
    assert features["all_layer_embeddings"] is layers
    for layer, expected in zip(layers, copies):
        torch.testing.assert_close(layer, expected, rtol=0, atol=0)


def test_weighted_pooling_second_derivatives():
    layers = tuple(torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True) for _ in range(3))
    weights = torch.tensor([0.7, 1.1], dtype=torch.float64, requires_grad=True)

    def run(*inputs):
        model = WeightedLayerPooling(3, num_hidden_layers=2, layer_start=1, layer_weights=inputs[-1])
        return model({"all_layer_embeddings": inputs[:-1]})["token_embeddings"]

    assert torch.autograd.gradgradcheck(run, (*layers, weights))


def test_weighted_pooling_forward_mode_derivatives():
    layers = tuple(torch.randn(1, 2, 3, dtype=torch.float64) for _ in range(3))
    weights = torch.tensor([0.7, 1.1], dtype=torch.float64)
    inputs = (*layers, weights)
    tangents = tuple(torch.randn_like(value) for value in inputs)

    def run(*values):
        model = WeightedLayerPooling(3, num_hidden_layers=2, layer_start=1, layer_weights=values[-1])
        return model({"all_layer_embeddings": values[:-1]})["token_embeddings"]

    actual = torch.func.jvp(run, inputs, tangents)
    expected = torch.func.jvp(lambda *values: original_average(values[:-1], values[-1], 1), inputs, tangents)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


def test_pretrained_weighted_pooling_encoding_and_optimization(stsb_bert_tiny_model):
    transformer = stsb_bert_tiny_model[0]
    transformer.auto_model.config.output_hidden_states = True
    dim = transformer.get_embedding_dimension()
    weighted = WeightedLayerPooling(dim, num_hidden_layers=2, layer_start=1)
    model = SentenceTransformer(modules=[transformer, weighted, Pooling(dim)], device="cpu").eval()
    texts = ["Paris.", "Python is a programming language."]
    features = model.preprocess(texts)
    with torch.no_grad():
        hidden = transformer(copy.deepcopy(features))
        expected_tokens = original_average(hidden["all_layer_embeddings"], weighted.layer_weights, 1)
        expected = model[-1]({**hidden, "token_embeddings": expected_tokens})["sentence_embedding"]
        actual = model.encode(texts, convert_to_tensor=True)
    torch.testing.assert_close(actual, expected)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    original_weights = weighted.layer_weights.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        result = model(copy.deepcopy(features))["sentence_embedding"]
        result.square().mean().backward()
        assert weighted.layer_weights.grad is not None
        assert torch.isfinite(weighted.layer_weights.grad).all()
        optimizer.step()
    assert not torch.equal(original_weights, weighted.layer_weights)
