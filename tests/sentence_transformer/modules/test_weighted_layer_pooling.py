from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from sentence_transformers.sentence_transformer.modules import WeightedLayerPooling


@pytest.mark.parametrize("layer_start", [0, 4, 12, -2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("container", [list, tuple])
def test_weighted_layer_values_and_gradients(layer_start: int, dtype: torch.dtype, container: type) -> None:
    torch.manual_seed(7)
    layers = [torch.randn(2, 3, 5, dtype=dtype, requires_grad=True) for _ in range(13)]
    selected = layers[layer_start:]
    weights = nn.Parameter(torch.linspace(0.5, 1.5, len(selected), dtype=dtype))
    pooling = WeightedLayerPooling(5, num_hidden_layers=12, layer_start=layer_start, layer_weights=weights)
    actual = pooling({"all_layer_embeddings": container(layers)})["token_embeddings"]

    # Independent weighted mean: the unused hidden states have zero derivative.
    expected = sum(weight * layer for weight, layer in zip(weights, selected)) / weights.sum()
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual.square().sum(), [*layers, weights], allow_unused=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), [*layers, weights], allow_unused=True)
    for actual_gradient, expected_gradient, value in zip(actual_gradients, expected_gradients, [*layers, weights]):
        if actual_gradient is None:
            actual_gradient = torch.zeros_like(value)
        if expected_gradient is None:
            expected_gradient = torch.zeros_like(value)
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("layer_start", [0, 4, 12, -2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_weighted_layer_low_precision_reduction_is_preserved(layer_start: int, dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    layers = [torch.randn(2, 3, 5, dtype=dtype, requires_grad=True) for _ in range(13)]
    weights = nn.Parameter(torch.linspace(0.5, 1.5, len(layers[layer_start:]), dtype=dtype))
    pooling = WeightedLayerPooling(5, num_hidden_layers=12, layer_start=layer_start, layer_weights=weights)
    actual = pooling({"all_layer_embeddings": layers})["token_embeddings"]
    # Preserve the original reduced-precision reduction order, including its rounding.
    selected = torch.stack(layers)[layer_start:]
    expected = (weights[:, None, None, None] * selected).sum(dim=0) / weights.sum()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(actual.square().sum(), [*layers, weights], allow_unused=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), [*layers, weights])
    for actual_gradient, expected_gradient, value in zip(actual_gradients, expected_gradients, [*layers, weights]):
        if actual_gradient is None:
            actual_gradient = torch.zeros_like(value)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("layer_start", [1, 3])
def test_weighted_layer_gradients_reach_earlier_encoder_layers(layer_start: int) -> None:
    torch.manual_seed(31)
    encoder = nn.ModuleList(nn.Linear(5, 5).double() for _ in range(4))
    reference_encoder = deepcopy(encoder)
    pooling = WeightedLayerPooling(5, num_hidden_layers=4, layer_start=layer_start).double()
    reference_weights = pooling.layer_weights.detach().clone().requires_grad_()
    inputs = torch.randn(2, 3, 5, dtype=torch.float64)

    def hidden_states(model):
        layers = [inputs]
        for layer in model:
            layers.append(layer(layers[-1]).tanh())
        return layers

    actual = pooling({"all_layer_embeddings": hidden_states(encoder)})["token_embeddings"]
    selected = hidden_states(reference_encoder)[layer_start:]
    expected = sum(weight * layer for weight, layer in zip(reference_weights, selected)) / reference_weights.sum()
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for parameter, reference_parameter in zip(encoder.parameters(), reference_encoder.parameters()):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)
    torch.testing.assert_close(pooling.layer_weights.grad, reference_weights.grad)


def test_weighted_layer_pooling_does_not_save_unused_layer_copies() -> None:
    layers = [torch.randn(2, 16, 32, requires_grad=True) for _ in range(13)]
    pooling = WeightedLayerPooling(32, num_hidden_layers=12, layer_start=9)
    saved_storage = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        saved_storage[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = pooling({"all_layer_embeddings": layers})["token_embeddings"]

    # The selected hidden states, weighted result and small weight/reduction tensors fit
    # below six hidden-state buffers; retaining a thirteen-layer stack does not.
    assert sum(saved_storage.values()) < 6 * layers[0].numel() * layers[0].element_size()
    output.sum().backward()


def test_weighted_layer_pooling_without_hidden_states_is_unchanged() -> None:
    features = {"token_embeddings": torch.randn(2, 3, 5)}
    original = features["token_embeddings"]
    assert WeightedLayerPooling(5)(features) is features
    assert features["token_embeddings"] is original
