from __future__ import annotations

import pytest
import torch

from sentence_transformers.sparse_encoder.modules import SpladePooling


def _transform(logits, activation):
    scores = logits.relu().log1p()
    return scores.log1p() if activation == "log1p_relu" else scores


@pytest.mark.parametrize("activation", ["relu", "log1p_relu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("chunk_size", [None, 1, 3, 20])
@pytest.mark.parametrize("training", [False, True])
def test_max_inference_matches_transforming_all_padded_tokens(activation, dtype, chunk_size, training):
    generator = torch.Generator().manual_seed(41)
    logits = torch.randn(3, 7, 13, generator=generator).to(dtype)
    logits[0, :, 0] = -3
    logits[0, :, 1] = 0
    logits[0, 1:4, 2] = 4
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]])
    original = logits.clone()
    pooling = SpladePooling("max", activation, chunk_size=chunk_size).train(training)
    with torch.no_grad():
        result = pooling({"token_embeddings": logits, "attention_mask": mask})["sentence_embedding"]
        expected = _transform(logits * mask.unsqueeze(-1).to(dtype), activation).max(dim=1).values
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
    assert result.dtype == dtype
    assert pooling.embedding_dimension == logits.size(-1)


@pytest.mark.parametrize("activation", ["relu", "log1p_relu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("training", [False, True])
def test_max_inference_matches_transforming_all_flattened_tokens(activation, dtype, training):
    generator = torch.Generator().manual_seed(43)
    logits = torch.randn(1, 8, 13, generator=generator).to(dtype)
    logits[:, :, 0] = -2
    lengths = [0, 1, 5, 8]
    pooling = SpladePooling("max", activation).train(training)
    with torch.inference_mode():
        result = pooling({"token_embeddings": logits, "cu_seq_lens_q": torch.tensor(lengths)})["sentence_embedding"]
        expected = torch.stack(
            [_transform(logits[0, start:end], activation).amax(dim=0) for start, end in zip(lengths[:-1], lengths[1:])]
        )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype == dtype


@pytest.mark.parametrize("activation", ["relu", "log1p_relu"])
@pytest.mark.parametrize("flattened", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_training_preserves_ties_created_by_rounding(activation, flattened, dtype):
    # Distinct logits can become equal after log1p rounds. Training must still select
    # or distribute gradients at those transformed ties, as the existing pooling does.
    smaller = torch.tensor(3.0, dtype=dtype)
    larger = torch.nextafter(smaller, torch.tensor(float("inf"), dtype=dtype))
    values = torch.stack([smaller, larger, -smaller]).reshape(1, 3, 1).requires_grad_()
    pooling = SpladePooling("max", activation).train()
    features = {"token_embeddings": values}
    if flattened:
        features["cu_seq_lens_q"] = torch.tensor([0, 3])
        expected = _transform(values[0], activation).amax(dim=0, keepdim=True)
    else:
        features["attention_mask"] = torch.ones(1, 3, dtype=torch.int64)
        expected = _transform(values, activation).max(dim=1).values
    actual = pooling(features)["sentence_embedding"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.sum(), values)[0]
    expected_grad = torch.autograd.grad(expected.sum(), values)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("activation", ["relu", "log1p_relu"])
@pytest.mark.parametrize("pooling_strategy", ["sum", "max"])
def test_pooling_inference_preserves_cpu_autocast(activation, pooling_strategy):
    logits = torch.tensor([[[1.0, -1.0], [3.0, 5.0], [-2.0, 2.0]]])
    pooling = SpladePooling(pooling_strategy, activation).eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        result = pooling({"token_embeddings": logits, "attention_mask": torch.ones(1, 3)})["sentence_embedding"]
        transformed = _transform(logits, activation)
        expected = transformed.max(dim=1).values if pooling_strategy == "max" else transformed.sum(dim=1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype == expected.dtype
