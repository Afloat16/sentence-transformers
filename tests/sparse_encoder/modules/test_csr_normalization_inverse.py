from __future__ import annotations

import pytest
import torch

from sentence_transformers.sparse_encoder.modules import SparseAutoEncoder


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("eps", [1e-5, 1e-2])
@pytest.mark.parametrize("scale", [1e-8, 1e-4, 1.0])
@pytest.mark.parametrize("offset", [0.0, 0.25])
def test_csr_normalization_metadata_inverts_the_actual_denominator(dtype, eps, scale, offset):
    module = SparseAutoEncoder(input_dim=4, hidden_dim=4, k=1, k_aux=0, normalize=True).to(dtype)
    with torch.no_grad():
        module.encoder.weight.copy_(torch.eye(4, dtype=dtype))
    inputs = torch.tensor([[-3, -1, 2, 4], [4, 1, -2, -5]], dtype=dtype) * scale + offset
    normalized, mean, denominator = module.LN(inputs, eps=eps)
    latents = module.encode_pre_act(normalized)
    actual = module.decode(latents, {"mu": mean, "std": denominator})
    torch.testing.assert_close(
        actual, inputs, rtol=2e-6 if dtype == torch.float32 else 1e-12, atol=1e-10 if dtype == torch.float32 else 1e-15
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1e-8, 1e-5, 1.0])
def test_csr_exact_signed_dictionary_reconstructs_small_variance_and_gradient(dtype, scale):
    module = SparseAutoEncoder(input_dim=4, hidden_dim=16, k=4, k_aux=0, normalize=True).to(dtype)
    with torch.no_grad():
        module.encoder.weight.zero_()
        module.encoder.weight[:4].copy_(torch.eye(4, dtype=dtype))
        module.encoder.weight[4:8].copy_(-torch.eye(4, dtype=dtype))
    inputs = (torch.tensor([[-3, -1, 2, 4], [4, 1, -2, -5]], dtype=dtype) * scale).requires_grad_()
    features = module({"sentence_embedding": inputs})
    for key in ["decoded_embedding_k", "decoded_embedding_4k"]:
        torch.testing.assert_close(features[key], inputs, rtol=2e-6, atol=scale * 1e-7)
    probe = torch.tensor([[0.2, -0.4, 0.7, -0.1], [-0.5, 0.8, 0.1, -0.9]], dtype=dtype)
    actual_gradient = torch.autograd.grad((features["decoded_embedding_k"] * probe).sum(), inputs)[0]
    torch.testing.assert_close(actual_gradient, probe, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("value", [0.0, 0.5, 8.0])
def test_csr_constant_inputs_have_finite_reconstruction_and_gradient(dtype, value):
    module = SparseAutoEncoder(input_dim=4, hidden_dim=16, k=4, k_aux=0, normalize=True).to(dtype)
    with torch.no_grad():
        module.encoder.weight.zero_()
    inputs = torch.full((2, 4), value, dtype=dtype, requires_grad=True)
    output = module({"sentence_embedding": inputs})["decoded_embedding_k"]
    torch.testing.assert_close(output, inputs, rtol=0, atol=0)
    gradient = torch.autograd.grad(output.sum(), inputs)[0]
    torch.testing.assert_close(gradient, torch.ones_like(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_csr_auxiliary_bias_uses_the_same_stabilized_scale(dtype):
    module = SparseAutoEncoder(input_dim=4, hidden_dim=16, k=4, k_aux=4, normalize=True).to(dtype)
    with torch.no_grad():
        module.encoder.weight.zero_()
        module.pre_bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=dtype))
    inputs = torch.tensor([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=dtype) * 1e-7
    features = module({"sentence_embedding": inputs})
    # The sample standard deviation is the L2 norm of the centered row divided by sqrt(n-1).
    centered = inputs - inputs.mean(dim=-1, keepdim=True)
    denominator = centered.square().sum(dim=-1, keepdim=True).sqrt() / 3**0.5 + 1e-5
    expected = module.pre_bias * denominator
    torch.testing.assert_close(features["decoded_embedding_aux"], expected, rtol=2e-6, atol=1e-12)
